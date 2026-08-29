"""G1 fixed-line climbing task for MuJoCo Playground (MJX).

A Unitree G1 climbs a slope while its right hand is attached to a fixed
line (a long thin cylinder) through an idealized ascender:

* Slope: the floor plane is tilted by `climb_config.slope_deg` about the
  world +y axis so the surface rises toward +x; the in-plane uphill
  direction is `(cos s, 0, sin s)`. The feet get tangential friction
  (`condim=3`) so the robot can stand on the incline.
* Fixed line: a static collision-free cylinder parallel to the slope at
  the height of the robot's right palm in its rest pose (roughly waist
  height), since the hand must grip it. The cylinder is visual only —
  the physical line is the axis of the carrier's slide joint.
* Ascender: a carrier body with a single slide joint along the line,
  point-attached to the `right_palm` site by a stiff `connect` equality
  (the hand can never leave the line). Unidirectionality is enforced
  every physics substep: slide qpos is clamped non-decreasing and slide
  qvel non-negative, so the hand slides up freely but never down,
  regardless of load. MJX has no `set_mjcb_control` callback, so the
  ratchet is JAX array ops on `data.qpos`/`data.qvel` inside a
  `jax.lax.scan` over substeps (jit/vmap-safe for training).

Model layout: the appended `ascender_slide` joint is the LAST qpos/qvel
coordinate. Everything upstream that slices `qpos[7:]` / `qvel[6:]`
would pick it up as a phantom 30th joint, so `_get_obs` is rebuilt with
trimmed slices and the small reward helpers that receive full joint
arrays are overridden to drop the last element. Observations otherwise
match upstream G1Joystick exactly (103-dim `state`), so the mels demo
policy still loads.

Reset: upstream randomizes base xy/yaw and joint poses, which would
start the palm far off the line and yank it back through the stiff
equality. This env resets deterministically in the `knees_bent` pose at
the line (palm coincident with the carrier), keeping only the upstream
base-velocity randomization. Slope/line randomization is future work.
"""

import functools
import math
import os

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import locomotion
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.g1 import base as g1_base
from mujoco_playground._src.locomotion.g1 import g1_constants as consts
from mujoco_playground._src.locomotion.g1 import joystick as g1_joystick

_ROBOT_NQ = 36  # freejoint (7) + 29 actuated joints, upstream G1.
_ROBOT_NV = 35

# Terrain surface materials, ported from the teammate's Himalaya pad
# (assets/terrain on feat/add-himalaya-terrain-3m; USD materials:
# packed snow 0.50/0.45, ice 0.10/0.08, rock 0.80/0.75).
_MU_SNOW = 0.50
_MU_ICE = 0.10
_MU_ROCK = 0.80


def default_config() -> config_dict.ConfigDict:
  """Upstream G1 joystick config plus `climb_config`."""
  cfg = g1_joystick.default_config()
  cfg.climb_config = config_dict.create(
      # Terrain mode: "slope" (single tilted plane) or "himalaya" (the
      # 3 m multi-material test pad from the terrain PR, rope up the
      # 40 deg snow wedge).
      terrain="slope",
      slope_deg=30.0,  # slope angle (slope mode), degrees.
      # Fixed-line geometry (m).
      rope_radius=0.02,  # visual cylinder radius.
      rope_length=15.0,  # cylinder length upslope from the grip start.
      rope_tail=0.5,  # cylinder length downslope of the grip start.
      line_offset_y=0.0,  # lateral tweak of the line off the rest palm.
      # Ascender carrier dynamics (slide-up stays nearly free).
      carrier_mass=0.1,
      slide_damping=1.0,
      slide_frictionloss=0.2,
      # Equality (grip) solver parameters; verified stable on MJX.
      grip_solref=(0.004, 1.0),
      grip_solimp=(0.95, 0.99, 0.001, 0.5, 2.0),
      # Foot traction (effective friction of the foot-floor contact,
      # set on the explicit pair so it actually applies).
      foot_friction=0.8,
  )
  # Extra constraint rows: connect equality + slide frictionloss +
  # terrain contacts (himalaya mode: feet can touch floor, wedges,
  # ice, and wall in one solve) + margin.
  cfg.njmax = 29 * 2 + 8 * 4 + 40
  return cfg


def _rewrite_mesh_paths(spec: mujoco.MjSpec) -> None:
  """Point spec mesh files at the vendored menagerie assets.

  `MjSpec.from_file` resolves the upstream G1 mesh paths to
  `site-packages/mujoco_menagerie/...`, which does not exist; the real
  copy lives under `mujoco_playground/external_deps/mujoco_menagerie`.
  """
  real_dir = mjx_env.MENAGERIE_PATH / "unitree_g1" / "assets"
  for mesh in spec.meshes:
    if mesh.file:
      cand = real_dir / os.path.basename(mesh.file)
      if cand.exists():
        mesh.file = cand.as_posix()


def _load_scene_spec(task: str) -> mujoco.MjSpec:
  """Load the upstream G1 scene spec with vendored mesh paths."""
  spec = mujoco.MjSpec.from_file(consts.task_to_xml(task).as_posix())
  _rewrite_mesh_paths(spec)
  return spec


def _set_foot_traction(spec: mujoco.MjSpec, mu: float) -> None:
  """Set the effective foot-floor friction.

  The upstream scene declares explicit `<pair>` contacts for
  foot<->floor, which override geom-level friction, so the pair (not
  the foot geoms) must be edited.
  """
  for name in ("left_foot_floor", "right_foot_floor"):
    spec.pair(name).friction = np.array([mu, mu, 0.005, 1e-4, 1e-4])


class G1ClimbAscender(g1_joystick.Joystick):
  """G1 climbing a fixed line with an idealized right-hand ascender."""

  def __init__(
      self,
      task: str = "flat_terrain",
      config: config_dict.ConfigDict | None = None,
      config_overrides: dict | None = None,
  ):
    if task != "flat_terrain":
      raise ValueError(
          "G1ClimbAscender requires the flat_terrain plane floor; the"
          f" rough_terrain hfield cannot be tilted. Got task={task!r}."
      )
    config = config or default_config()
    # Velocity-impulse pushes are meaningless while gripping a fixed
    # line; the ascender itself is the perturbation-resistant element.
    config.push_config.enable = False
    self._task = task
    super().__init__(
        task=task, config=config, config_overrides=config_overrides
    )
    # After super().__init__ the model is the upstream scene; rebuild it
    # with the slope + line + ascender, then refresh the model-derived
    # fields that _post_init computed from the old model.
    self._build_model()
    self._n_substeps = int(round(self.dt / self.sim_dt))

  # ------------------------------------------------------------------
  # Model construction.
  # ------------------------------------------------------------------

  def _build_model(self) -> None:
    cc = self._config.climb_config
    terrain = str(cc.terrain)
    if terrain == "slope":
      spec, axis = self._build_slope_spec()
    elif terrain == "himalaya":
      spec, axis = self._build_himalaya_spec()
    else:
      raise ValueError(
          f"climb_config.terrain must be 'slope' or 'himalaya', got {terrain!r}"
      )

    self._finish_model(spec, axis)

  def _build_slope_spec(self) -> tuple[mujoco.MjSpec, np.ndarray]:
    """Single tilted plane rising toward +x; rope parallel to it."""
    cc = self._config.climb_config
    slope = math.radians(float(cc.slope_deg))
    # Uphill direction in the world xz-plane (floor rises toward +x).
    axis = np.array([math.cos(slope), 0.0, math.sin(slope)])

    spec = _load_scene_spec(self._task)
    floor = spec.geom("floor")
    if floor.type != mujoco.mjtGeom.mjGEOM_PLANE:
      raise ValueError(f"expected a plane floor, got {floor.type}")
    # Tilt about +y so the surface rises toward +x (uphill +x).
    floor.quat = (
        math.cos(slope / 2),
        0.0,
        -math.sin(slope / 2),
        0.0,
    )
    _set_foot_traction(spec, float(cc.foot_friction))
    return spec, axis

  def _build_himalaya_spec(self) -> tuple[mujoco.MjSpec, np.ndarray]:
    """Port of the teammate's 3 m Himalaya test pad (terrain PR).

    Mirrors assets/terrain/build_terrain.py geometry and USD materials:
    packed-snow floor (mu 0.5), 1 m ice slab (mu 0.1) in the NE
    quadrant, 10 deg / 40 deg snow wedges rising toward -X in the W
    quadrants, and a rock wall (mu 0.8) at x=1.0 in the SE quadrant.
    The robot spawns at the centre facing -X (up the wedges) and the
    fixed line runs up the 40 deg wedge surface.

    Contact sensors for the wedges are spliced into the sensor.xml
    include content (the MjSpec Python binding cannot express
    <contact geom1 geom2> directly); terrain geoms, pairs, and the
    ascender are then added via the spec API.
    """
    cc = self._config.climb_config
    wedge_deg = 40.0
    th = math.radians(wedge_deg)
    # Uphill direction on the 40 deg wedge: rises toward -X.
    axis = np.array([-math.cos(th), 0.0, math.sin(th)])

    # --- scene spec with extra foot<->wedge contact sensors -----------
    from etils import epath  # noqa: PLC0415

    scene = epath.Path(consts.task_to_xml(self._task)).read_text()
    assets = g1_base.get_assets()
    extra_sensors = "".join(
        f'<contact name="{foot}_wedge_found" geom1="{foot}"'
        f' geom2="slope_{wedge_deg:.0f}deg" reduce="mindist" num="1"'
        f' data="found"/>\n'
        for foot in ("left_foot", "right_foot")
    )
    include = {}
    for name in ("g1_mjx_feetonly.xml", "sensor.xml"):
      content = assets[name]
      if name == "sensor.xml":
        text = content.decode() if isinstance(content, bytes) else content
        content = text.replace("</sensor>", extra_sensors + "</sensor>").encode()
      include[name] = content
    mesh_assets = {k: v for k, v in assets.items() if not k.endswith(".xml")}
    spec = mujoco.MjSpec.from_string(scene, include=include, assets=mesh_assets)

    # --- snow floor ----------------------------------------------------
    # Reuse the scene floor geom as packed snow: drop the checker
    # material, snow-white rgba, snow friction on the foot pairs.
    floor = spec.geom("floor")
    floor.material = ""
    floor.rgba = (0.92, 0.94, 0.97, 1.0)
    _set_foot_traction(spec, _MU_SNOW)

    wb = spec.worldbody

    def add_box(name, center, size, quat, rgba):
      return wb.add_geom(
          name=name,
          type=mujoco.mjtGeom.mjGEOM_BOX,
          size=size,
          pos=center,
          quat=quat,
          rgba=rgba,
          contype=0,
          conaffinity=0,
      )

    def wedge(deg, cx, cy, mu):
      """Box whose TOP face is the ramp: 1 m run rising toward -X."""
      th_ = math.radians(deg)
      rise = math.tan(th_)  # run = 1.0 m
      length = 1.0 / math.cos(th_)
      thick = 0.3
      # Top-face centre at (cx, cy, rise/2); box centre sunk thick/2
      # along the face normal so the box reaches below the floor.
      n = np.array([math.sin(th_), 0.0, math.cos(th_)])  # tilts toward +X
      center = np.array([cx, cy, rise / 2]) - n * thick / 2
      # Rotate about +y so the top-face normal tilts toward +X and the
      # surface rises toward -X (in-plane uphill = (-cos, 0, sin)).
      quat = (math.cos(th_ / 2), 0.0, math.sin(th_ / 2), 0.0)
      name = f"slope_{deg:.0f}deg"
      add_box(
          name, tuple(center), (length / 2, 0.5, thick / 2), quat,
          (0.92, 0.94, 0.97, 1.0),
      )
      for foot in ("left_foot", "right_foot"):
        spec.add_pair(
            name=f"{foot}_{name}",
            geomname1=foot,
            geomname2=name,
            condim=3,
            friction=np.array([mu, mu, 0.005, 1e-4, 1e-4]),
        )
      return name

    # 10 deg (NW) and 40 deg (SW) snow wedges; 1x1 m ice slab (NE);
    # rock wall (SE): 1 m long, face at x=1.0.
    wedge(10.0, -0.75, 0.75, _MU_SNOW)
    wedge_name = wedge(40.0, -0.75, -0.75, _MU_SNOW)
    add_box(
        "ice", (0.75, 0.75, 0.005), (0.5, 0.5, 0.005),
        (1.0, 0.0, 0.0, 0.0), (0.75, 0.88, 1.0, 1.0),
    )
    add_box(
        "wall", (1.075, -0.75, 0.5), (0.075, 0.5, 0.5),
        (1.0, 0.0, 0.0, 0.0), (0.35, 0.33, 0.31, 1.0),
    )
    # Foot contacts on ice and wall (explicit pairs set their friction).
    for foot in ("left_foot", "right_foot"):
      spec.add_pair(
          name=f"{foot}_ice", geomname1=foot, geomname2="ice", condim=3,
          friction=np.array([_MU_ICE, _MU_ICE, 0.005, 1e-4, 1e-4]),
      )
      spec.add_pair(
          name=f"{foot}_wall", geomname1=foot, geomname2="wall", condim=3,
          friction=np.array([_MU_ROCK, _MU_ROCK, 0.005, 1e-4, 1e-4]),
      )

    # Robot spawns upright on the flat snow at the 40 deg wedge base,
    # facing uphill (-X). The fixed line passes through the rest palm
    # and runs up parallel to the ramp surface, ~waist height above it;
    # as the robot climbs the wedge the hand rides up the line.
    # Spawn pose is applied to the keyframes in _finish_model.
    self._spawn_yaw = math.pi  # face -X (uphill on the wedges)
    self._spawn_pos = np.array([-0.05, -0.75, 0.755])
    self._wedge_name = wedge_name
    return spec, axis

  def _finish_model(self, spec: mujoco.MjSpec, axis: np.ndarray) -> None:
    """Add the fixed line + ascender, compile, refresh derived fields."""
    cc = self._config.climb_config

    # Right-palm world position in the rest pose on this terrain. In
    # himalaya mode the rest pose is translated to the wedge base and
    # rotated to face uphill (-X); the keyframe stays upright.
    base_model = spec.compile()
    base_data = mujoco.MjData(base_model)
    kf = next(k for k in spec.keys if k.name == "knees_bent")
    base_data.qpos[:] = kf.qpos
    yaw = getattr(self, "_spawn_yaw", 0.0)
    spawn_pos = getattr(self, "_spawn_pos", None)
    if yaw:
      q = base_data.qpos[3:7].copy()
      rot = np.array([
          math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)
      ])
      w1, x1, y1, z1 = rot
      w2, x2, y2, z2 = q
      base_data.qpos[3:7] = (
          w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
          w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
          w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
          w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
      )
    if spawn_pos is not None:
      # Keep the keyframe's z: place the root at spawn_pos with the
      # keyframe pelvis height (feet just above the ramp surface).
      base_data.qpos[0:3] = spawn_pos
    mujoco.mj_forward(base_model, base_data)
    palm0 = base_data.site_xpos[base_model.site("right_palm").id].copy()

    # Fixed line through the rest palm (optionally nudged sideways).
    # Parallel to the climb surface by construction (axis in-plane).
    line_pt = palm0 + np.array([0.0, float(cc.line_offset_y), 0.0])

    wb = spec.worldbody
    rope = wb.add_body(name="rope", pos=line_pt)
    rope_geom = rope.add_geom(
        name="rope_geom",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=(
            float(cc.rope_radius),
            (float(cc.rope_length) + float(cc.rope_tail)) / 2,
            0.0,
        ),
        rgba=(0.35, 0.25, 0.15, 1.0),
        mass=0.0,
        contype=0,
        conaffinity=0,
    )
    # fromto is body-local; body origin sits at line_pt.
    rope_geom.fromto = np.concatenate(
        (-float(cc.rope_tail) * axis, float(cc.rope_length) * axis)
    )

    # Ascender carrier: a point constrained to the line by its slide
    # joint (the physics line; the cylinder above is visual).
    carrier = wb.add_body(name="ascender_carrier", pos=line_pt)
    carrier.add_joint(
        name="ascender_slide",
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=axis,
        damping=float(cc.slide_damping),
        frictionloss=float(cc.slide_frictionloss),
        range=(-float(cc.rope_tail), float(cc.rope_length)),
    )
    carrier.add_site(name="carrier_site", pos=(0.0, 0.0, 0.0))
    carrier.add_geom(
        name="carrier_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=(0.03,),
        mass=float(cc.carrier_mass),
        rgba=(0.9, 0.1, 0.1, 0.3),
        contype=0,
        conaffinity=0,
    )

    # The grip: right palm point-attached to the carrier. The hand can
    # never leave the line; it can only move along it.
    eq = spec.add_equality(
        name="ascender_grip",
        type=mujoco.mjtEq.mjEQ_CONNECT,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        name1="right_palm",
        name2="carrier_site",
    )
    eq.solref = tuple(float(v) for v in cc.grip_solref)
    eq.solimp = tuple(float(v) for v in cc.grip_solimp)

    # Keyframes gain the slide coordinate (0 = carrier at line_pt) and
    # the spawn orientation/position (robot stands at the wedge base
    # facing uphill in himalaya mode).
    yawq = np.array([
        math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)
    ])
    for k in spec.keys:
      qpos = np.asarray(k.qpos, dtype=float).copy()
      if yaw:
        q = qpos[3:7]
        w1, x1, y1, z1 = yawq
        w2, x2, y2, z2 = q
        qpos[3:7] = (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )
      if spawn_pos is not None:
        qpos[0:3] = spawn_pos
      k.qpos = np.concatenate([qpos, [0.0]])

    self._mj_model = spec.compile()
    self._mj_model.opt.timestep = self.sim_dt
    self._mj_model.vis.global_.offwidth = 3840
    self._mj_model.vis.global_.offheight = 2160
    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
    self._xml_path = consts.task_to_xml(self._task).as_posix()

    # Model-derived fields that _post_init computed from the pre-surgery
    # model and that change with the appended slide joint. Everything
    # else (name-based ids, qposadr-based index lists) is unchanged: the
    # slide joint is appended after all robot joints.
    self._init_q = jp.array(self._mj_model.keyframe("knees_bent").qpos)
    self._slide_qposadr = int(
        self._mj_model.jnt_qposadr[self._mj_model.joint("ascender_slide").id]
    )
    self._slide_dofadr = int(
        self._mj_model.jnt_dofadr[self._mj_model.joint("ascender_slide").id]
    )
    assert self._slide_qposadr == self._mj_model.nq - 1
    assert self._slide_dofadr == self._mj_model.nv - 1
    # Robot joints only (drop the trailing slide coordinate).
    self._default_pose = jp.array(
        self._mj_model.keyframe("knees_bent").qpos[7 : self._slide_qposadr]
    )
    self._lowers, self._uppers = (
        jp.array(self._mj_model.jnt_range[1 : self._mj_model.njnt - 1].T)
    )
    c = (self._lowers + self._uppers) / 2
    r = self._uppers - self._lowers
    f = self._config.soft_joint_pos_limit_factor
    self._soft_lowers = c - 0.5 * r * f
    self._soft_uppers = c + 0.5 * r * f

    self._palm_site_id = self._mj_model.site("right_palm").id
    self._carrier_site_id = self._mj_model.site("carrier_site").id
    self._line_pt = jp.array(line_pt)
    self._slope_axis = jp.array(axis)
    # Support contact sensors: floor (+ climb wedge in himalaya mode).
    self._feet_support_sensors = (
        list(self._feet_floor_found_sensor)
        + getattr(self, "_feet_extra_found_sensor", [])
    )

  # ------------------------------------------------------------------
  # MJX stepping with the ascender ratchet.
  # ------------------------------------------------------------------

  def _step_physics(self, data: mjx.Data, ctrl: jax.Array) -> mjx.Data:
    """Substep loop with the ascender ratchet (replaces mjx_env.step)."""
    slide_q, slide_dof = self._slide_qposadr, self._slide_dofadr

    def single_step(data, _):
      data = data.replace(ctrl=ctrl)
      prev_slide = data.qpos[slide_q]
      data = mjx.step(self.mjx_model, data)
      # Ascender ratchet: the hand may move up the line, never down.
      qvel = data.qvel.at[slide_dof].set(
          jp.maximum(data.qvel[slide_dof], 0.0)
      )
      qpos = data.qpos.at[slide_q].set(
          jp.maximum(data.qpos[slide_q], prev_slide)
      )
      return data.replace(qpos=qpos, qvel=qvel), None

    return jax.lax.scan(single_step, data, (), self._n_substeps)[0]


  def _feet_contact(self, data: mjx.Data) -> jax.Array:
    """Per-foot support contact (floor OR climb wedge), shape (2,)."""
    raw = jp.array([
        data.sensordata[self._mj_model.sensor_adr[sensorid]] > 0
        for sensorid in self._feet_support_sensors
    ])
    if raw.shape[0] == 2:
      return raw
    # Sensors alternate [lf_floor, rf_floor, lf_wedge, rf_wedge, ...].
    n_surfaces = raw.shape[0] // 2
    return jp.any(raw.reshape(n_surfaces, 2), axis=0)

  # ------------------------------------------------------------------
  # Env API.
  # ------------------------------------------------------------------

  def reset(self, rng: jax.Array):
    # Deterministic pose at the line (palm coincident with the carrier):
    # upstream xy/yaw/joint-pose randomization would start the palm far
    # off the line and the stiff equality would yank it back. Base
    # velocity randomization is kept.
    qpos = self._init_q
    qvel = jp.zeros(self.mjx_model.nv)

    rng, key = jax.random.split(rng)
    qvel = qvel.at[0:6].set(
        jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5)
    )

    data = mjx_env.make_data(
        self.mj_model,
        qpos=qpos,
        qvel=qvel,
        ctrl=qpos[7 : self._slide_qposadr],
        impl=self.mjx_model.impl.value,
        naconmax=self._config.naconmax,
        njmax=self._config.njmax,
    )
    data = mjx.forward(self.mjx_model, data)

    # Phase, freq=U(1.25, 1.5)
    rng, key = jax.random.split(rng)
    gait_freq = jax.random.uniform(key, (1,), minval=1.25, maxval=1.5)
    phase_dt = 2 * jp.pi * self.dt * gait_freq
    phase = jp.array([0, jp.pi])

    rng, cmd_rng = jax.random.split(rng)
    cmd = self.sample_command(cmd_rng)

    info = {
        "rng": rng,
        "step": 0,
        "command": cmd,
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
        "motor_targets": jp.zeros(self.mjx_model.nu),
        "feet_air_time": jp.zeros(2),
        "last_contact": jp.zeros(2, dtype=bool),
        "swing_peak": jp.zeros(2),
        "phase_dt": phase_dt,
        "phase": phase,
        # Pushes are disabled; keys kept for a stable info signature.
        "push": jp.array([0.0, 0.0]),
        "push_step": 0,
        "push_interval_steps": jp.zeros((), dtype=jp.int32),
    }

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())
    metrics["swing_peak"] = jp.zeros(())

    contact = self._feet_contact(data)
    obs = self._get_obs(data, info, contact)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state, action):
    # Same control-step structure as upstream Joystick.step, with
    # `_step_physics` (ratcheted substeps) in place of mjx_env.step and
    # no push impulses.
    motor_targets = self._default_pose + action * self._config.action_scale
    data = self._step_physics(state.data, motor_targets)
    state_info = {**state.info, "motor_targets": motor_targets}

    contact = self._feet_contact(data)
    contact_filt = contact | state_info["last_contact"]
    first_contact = (state_info["feet_air_time"] > 0.0) * contact_filt
    state_info["feet_air_time"] += self.dt
    p_f = data.site_xpos[self._feet_site_id]
    p_fz = p_f[..., -1]
    state_info["swing_peak"] = jp.maximum(state_info["swing_peak"], p_fz)

    obs = self._get_obs(data, state_info, contact)
    done = self._get_termination(data)

    rewards = self._get_reward(
        data, action, state_info, state.metrics, done, first_contact, contact
    )
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    reward = sum(rewards.values()) * self.dt

    state_info["push"] = jp.array([0.0, 0.0])
    state_info["step"] += 1
    state_info["push_step"] += 1
    phase_tp1 = state_info["phase"] + state_info["phase_dt"]
    state_info["phase"] = jp.fmod(phase_tp1 + jp.pi, 2 * jp.pi) - jp.pi
    state_info["last_last_act"] = state_info["last_act"]
    state_info["last_act"] = action
    state_info["rng"], cmd_rng = jax.random.split(state_info["rng"])
    state_info["command"] = jp.where(
        state_info["step"] > 500,
        self.sample_command(cmd_rng),
        state_info["command"],
    )
    state_info["step"] = jp.where(
        done | (state_info["step"] > 500),
        0,
        state_info["step"],
    )
    state_info["feet_air_time"] *= ~contact
    state_info["last_contact"] = contact
    state_info["swing_peak"] *= ~contact
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v
    state.metrics["swing_peak"] = jp.mean(state_info["swing_peak"])

    done = done.astype(reward.dtype)
    return state.replace(
        data=data, info=state_info, obs=obs, reward=reward, done=done
    )

  # ------------------------------------------------------------------
  # Observations and rewards with the slide coordinate trimmed out.
  # ------------------------------------------------------------------

  def _get_obs(self, data, info, contact):
    # Copy of upstream Joystick._get_obs with the joint slices trimmed
    # to the 29 robot joints (the appended slide coordinate is dropped).
    gyro = self.get_gyro(data, "pelvis")
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gyro = (
        gyro
        + (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gyro
    )

    gravity = data.site_xmat[self._pelvis_imu_site_id].T @ jp.array([0, 0, -1])
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gravity = (
        gravity
        + (2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gravity
    )

    joint_angles = data.qpos[7 : self._slide_qposadr]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_angles = (
        joint_angles
        + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_pos
    )

    joint_vel = data.qvel[6 : self._slide_dofadr]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_vel = (
        joint_vel
        + (2 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_vel
    )

    cos = jp.cos(info["phase"])
    sin = jp.sin(info["phase"])
    phase = jp.concatenate([cos, sin])

    linvel = self.get_local_linvel(data, "pelvis")
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_linvel = (
        linvel
        + (2 * jax.random.uniform(noise_rng, shape=linvel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.linvel
    )

    state = jp.hstack([
        noisy_linvel,  # 3
        noisy_gyro,  # 3
        noisy_gravity,  # 3
        info["command"],  # 3
        noisy_joint_angles - self._default_pose,  # 29
        noisy_joint_vel,  # 29
        info["last_act"],  # 29
        phase,  # 4
    ])

    accelerometer = self.get_accelerometer(data, "pelvis")
    global_angvel = self.get_global_angvel(data, "pelvis")
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel()
    root_height = data.qpos[2]

    privileged_state = jp.hstack([
        state,
        gyro,  # 3
        accelerometer,  # 3
        gravity,  # 3
        linvel,  # 3
        global_angvel,  # 3
        joint_angles - self._default_pose,  # 29
        joint_vel,  # 29
        root_height,  # 1
        data.actuator_force,  # 29
        contact,  # 2
        feet_vel,  # 4*3
        info["feet_air_time"],  # 2
    ])

    return {
        "state": state,
        "privileged_state": privileged_state,
    }

  # Reward helpers that receive full `qpos[7:]`/`qvel[6:]`-style arrays
  # from upstream `_get_reward`; each drops the trailing slide value.

  def _cost_stand_still(self, commands, qpos):
    return super()._cost_stand_still(commands, qpos[:-1])

  def _cost_joint_deviation_hip(self, qpos, cmd):
    return super()._cost_joint_deviation_hip(qpos[:-1], cmd)

  def _cost_joint_deviation_knee(self, qpos):
    return super()._cost_joint_deviation_knee(qpos[:-1])

  def _cost_joint_pos_limits(self, qpos):
    return super()._cost_joint_pos_limits(qpos[:-1])

  def _cost_pose(self, qpos):
    return super()._cost_pose(qpos[:-1])

  def _cost_energy(self, qvel, qfrc_actuator):
    return super()._cost_energy(qvel[:-1], qfrc_actuator)

  def _cost_dof_acc(self, qacc):
    return super()._cost_dof_acc(qacc[:-1])

  # ------------------------------------------------------------------
  # Climbing-specific accessors.
  # ------------------------------------------------------------------

  def hand_line_error(self, data: mjx.Data) -> jax.Array:
    """Right-palm distance to the fixed line (0 = on the line)."""
    palm = data.site_xpos[self._palm_site_id]
    rel = palm - self._line_pt
    along = rel @ self._slope_axis
    perp = rel - along * self._slope_axis
    return jp.linalg.norm(perp)

  def hand_height_on_line(self, data: mjx.Data) -> jax.Array:
    """Right-palm arc length up the line from the reset grip point."""
    palm = data.site_xpos[self._palm_site_id]
    return (palm - self._line_pt) @ self._slope_axis


# Registered on import; see wind_g1/__init__.py.
locomotion.register_environment(
    "G1ClimbAscender",
    functools.partial(G1ClimbAscender, task="flat_terrain"),
    default_config,
)


def _himalaya_config() -> config_dict.ConfigDict:
  """Slope-mode defaults with the himalaya terrain selected."""
  cfg = default_config()
  cfg.climb_config.terrain = "himalaya"
  return cfg


# Himalaya-terrain variant: the 3 m multi-material test pad (snow/ice/
# rock, 10/40 deg wedges) from the terrain PR, rope up the 40 deg
# wedge. Same class; only climb_config.terrain differs.
locomotion.register_environment(
    "G1ClimbAscenderHimalaya",
    functools.partial(
        G1ClimbAscender, task="flat_terrain",
    ),
    _himalaya_config,
)

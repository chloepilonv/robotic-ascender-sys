"""G1 fixed-line climbing base env for MuJoCo Playground (MJX). NOT registered.

This is the machinery parent of `climb_terrain_env.G1ClimbTerrain` (the
registered `G1ClimbTerrain` env: the same task on the merged Lhotse terrain
heightfield). It is intentionally NOT registered in the playground registry —
importing `rl.environment` does not expose it; use `G1ClimbTerrain`.

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
from mujoco_playground._src.locomotion.g1 import g1_constants as consts
from mujoco_playground._src.locomotion.g1 import joystick as g1_joystick

_ROBOT_NQ = 36  # freejoint (7) + 29 actuated joints, upstream G1.
_ROBOT_NV = 35


def default_config() -> config_dict.ConfigDict:
  """Upstream G1 joystick config plus `climb_config`."""
  cfg = g1_joystick.default_config()
  cfg.climb_config = config_dict.create(
      slope_deg=30.0,  # slope angle, degrees above horizontal.
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
      # Foot friction on the slope surface.
      foot_friction=0.8,
  )
  # Extra constraint rows: 3 (connect equality) + 1 (slide frictionloss)
  # + margin.
  cfg.njmax = 29 * 2 + 8 * 4 + 8
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
    slope = math.radians(float(cc.slope_deg))
    # Uphill direction in the world xz-plane (floor rises toward +x).
    axis = np.array([math.cos(slope), 0.0, math.sin(slope)])

    spec = mujoco.MjSpec.from_file(consts.task_to_xml(self._task).as_posix())
    _rewrite_mesh_paths(spec)

    # Tilt the floor about +y so the surface rises toward +x (uphill +x).
    floor = spec.geom("floor")
    if floor.type != mujoco.mjtGeom.mjGEOM_PLANE:
      raise ValueError(f"expected a plane floor, got {floor.type}")
    floor.quat = (
        math.cos(slope / 2),
        0.0,
        -math.sin(slope / 2),
        0.0,
    )
    # Tangential foot friction on the incline (upstream feet are
    # condim=1, frictionless normals).
    for g in spec.geoms:
      if g.name in ("left_foot", "right_foot"):
        g.condim = 3
        g.friction = (float(cc.foot_friction), 0.005, 0.0001)

    # Right-palm world position in the rest pose, on the tilted slope.
    base_model = spec.compile()
    base_data = mujoco.MjData(base_model)
    kf = next(k for k in spec.keys if k.name == "knees_bent")
    base_data.qpos[:] = kf.qpos
    mujoco.mj_forward(base_model, base_data)
    palm0 = base_data.site_xpos[base_model.site("right_palm").id].copy()

    # Fixed line through the rest palm (optionally nudged sideways).
    # Parallel to the slope surface by construction (axis lies in-plane).
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

    # Keyframes gain the slide coordinate (0 = carrier at line_pt).
    for k in spec.keys:
      k.qpos = np.concatenate([np.asarray(k.qpos), [0.0]])

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

    contact = jp.array([
        data.sensordata[self._mj_model.sensor_adr[sensorid]] > 0
        for sensorid in self._feet_floor_found_sensor
    ])
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

    contact = jp.array([
        data.sensordata[self._mj_model.sensor_adr[sensorid]] > 0
        for sensorid in self._feet_floor_found_sensor
    ])
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



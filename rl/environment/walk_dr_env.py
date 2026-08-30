"""Domain-randomized G1 walking task for MuJoCo Playground (MJX).

Fine-tuning environment for the pretrained G1 joystick ("mels") walking
policy, with launch-time terrain/dynamics randomization and realistic
per-episode wind:

* Slope (launch time): the domain randomizer tilts the floor plane of
  each parallel environment by a fixed angle drawn from
  U(slope_min_deg, slope_max_deg) (default 0-40 deg, rising toward +x).
  Slopes are baked into per-env vectorized models by the playground
  `BraxDomainRandomizationVmapWrapper` and stay fixed for the whole run.
  The floor-foot contact pairs already have condim=3 upstream, so
  tangential friction acts on the incline; the randomizer also samples
  floor-foot friction U(0.4, 1.0) and the upstream G1 dynamics recipe
  (frictionloss / armature / link masses / torso mass / qpos0 jitter).

* Slope-aware env internals: reset places the robot standing on the
  slope (keyframe height along the terrain normal, stance aligned with
  the terrain), termination and the orientation reward are computed
  against the terrain normal, and the gravity block of the `state`
  observation is expressed relative to the terrain ("down" = slope
  normal), so a policy standing upright on any slope reads the same
  gravity as on flat ground. On flat terrain all of these reduce exactly
  to upstream Joystick behavior.

* Wind (per episode + in-episode dynamics): at reset a baseline wind is
  drawn - speed U(0, wind_max_speed_kmph / 3.6) m/s (default up to
  150 kmph ~ 41.7 m/s), heading U(0, 2pi). During the episode the wind
  evolves as a smooth Ornstein-Uhlenbeck process around that baseline:
  the heading wanders (stationary std `direction_wander_deg`, time scale
  `direction_persist_s`) and the speed gusts/dips (clipped to
  +-`gust_fraction` of the baseline, time scale `gust_persist_s`). The
  wind pushes the torso with the same quadratic drag as `wind_env`
  (F = 0.5 rho Cd A |v_rel| v_rel), written to `xfrc_applied` each
  control step.

Episode boundary note: with the default brax `AutoResetWrapper`
(full_reset=False) the wind OU state (`wind_base` / offsets) persists
across auto-resets, so each parallel env keeps its launch wind climate;
only the robot state is reset. This is intentional (a "weather system"
per env).

Observations match upstream G1Joystick exactly (103-dim `state`, 216-dim
`privileged_state`), so the mels policy fine-tunes with its original
network (512-256-128, 58 = 2x29 distribution params).
"""

import functools
import math

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
import numpy as np
from mujoco import mjx
from mujoco.mjx._src import math as mjx_math

from mujoco_playground._src import locomotion
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.g1 import joystick as g1_joystick


def default_config() -> config_dict.ConfigDict:
  """Upstream G1 joystick config plus `dr_config`."""
  cfg = g1_joystick.default_config()
  cfg.dr_config = config_dict.create(
      # Launch-time terrain randomization (per parallel env, fixed per run).
      slope_min_deg=0.0,
      slope_max_deg=40.0,
      # Launch-time dynamics randomization (upstream G1 recipe).
      friction_range=[0.4, 1.0],
      frictionloss_scale_range=[0.5, 2.0],
      armature_scale_range=[1.0, 1.05],
      mass_scale_range=[0.9, 1.1],
      torso_mass_delta_range=[-1.0, 1.0],
      qpos0_jitter=0.05,
      # Wind: per-episode baseline + smooth OU gusts/heading wander.
      wind_max_speed_kmph=150.0,
      gust_fraction=0.25,  # max +-25% of baseline speed.
      gust_persist_s=3.0,  # gust/dip time scale.
      direction_wander_deg=12.0,  # stationary std of heading wander.
      direction_persist_s=8.0,  # heading-wander time scale.
      # Quadratic-drag coefficients (same model as wind_env).
      rho=1.225,  # air density kg/m^3.
      cd_torso=1.2,  # torso drag coefficient.
      area_torso=0.5,  # torso frontal area m^2.
  )
  return cfg


def domain_randomize(
    model: mjx.Model,
    rng: jax.Array,
    dr_cfg: config_dict.ConfigDict | None = None,
):
  """Launch-time randomization: per-env slope + dynamics (G1 recipe).

  Called once by `BraxDomainRandomizationVmapWrapper` with a batch of
  rng keys (one per parallel env); produces a vectorized model whose
  floor tilt, floor-foot friction, and dynamics differ per env.

  Args:
    model: the base mjx model.
    rng: (num_envs,) rng keys.
    dr_cfg: randomization ranges; defaults to `default_config().dr_config`
      (pass the loaded env's effective `dr_config` to honor overrides).

  Returns:
    (vectorized model, in_axes) as expected by the wrapper.
  """
  cfg = dr_cfg or default_config().dr_config
  floor_id = int(
      np.argmax(np.asarray(model.geom_type) == int(mujoco.mjtGeom.mjGEOM_PLANE))
  )
  if np.asarray(model.geom_type)[floor_id] != int(
      mujoco.mjtGeom.mjGEOM_PLANE
  ):
    raise ValueError("no plane geom found: cannot randomize the slope.")

  slope_min = math.radians(float(cfg.slope_min_deg))
  slope_max = math.radians(float(cfg.slope_max_deg))
  fmin, fmax = (float(v) for v in cfg.friction_range)
  flmin, flmax = (float(v) for v in cfg.frictionloss_scale_range)
  amin, amax = (float(v) for v in cfg.armature_scale_range)
  mmin, mmax = (float(v) for v in cfg.mass_scale_range)
  tmin, tmax = (float(v) for v in cfg.torso_mass_delta_range)
  qjitter = float(cfg.qpos0_jitter)

  @jax.vmap
  def rand(rng):
    # Slope: floor plane tilted about +y, rising toward +x. quat from
    # axis-angle(-y, slope) = [cos(s/2), 0, -sin(s/2), 0].
    rng, key = jax.random.split(rng)
    slope = jax.random.uniform(key, minval=slope_min, maxval=slope_max)
    quat = jp.array([
        jp.cos(slope / 2), 0.0, -jp.sin(slope / 2), 0.0
    ])
    geom_quat = model.geom_quat.at[floor_id].set(quat)

    # Floor / foot friction: =U(fmin, fmax) on the sliding coefficient of
    # the floor-foot pairs (first two pairs, condim 3).
    rng, key = jax.random.split(rng)
    friction = jax.random.uniform(key, minval=fmin, maxval=fmax)
    pair_friction = model.pair_friction.at[0:2, 0:2].set(friction)

    # Scale static friction: *U(flmin, flmax).
    rng, key = jax.random.split(rng)
    frictionloss = model.dof_frictionloss[6:] * jax.random.uniform(
        key, shape=(29,), minval=flmin, maxval=flmax
    )
    dof_frictionloss = model.dof_frictionloss.at[6:].set(frictionloss)

    # Scale armature: *U(amin, amax).
    rng, key = jax.random.split(rng)
    armature = model.dof_armature[6:] * jax.random.uniform(
        key, shape=(29,), minval=amin, maxval=amax
    )
    dof_armature = model.dof_armature.at[6:].set(armature)

    # Scale all link masses: *U(mmin, mmax); torso += U(tmin, tmax).
    rng, key = jax.random.split(rng)
    dmass = jax.random.uniform(
        key, shape=(model.nbody,), minval=mmin, maxval=mmax
    )
    body_mass = model.body_mass.at[:].set(model.body_mass * dmass)
    rng, key = jax.random.split(rng)
    dmass = jax.random.uniform(key, minval=tmin, maxval=tmax)
    body_mass = body_mass.at[g1_torso_body_id].set(
        body_mass[g1_torso_body_id] + dmass
    )

    # Jitter qpos0: +U(-qjitter, qjitter).
    rng, key = jax.random.split(rng)
    qpos0 = model.qpos0
    qpos0 = qpos0.at[7:].set(
        qpos0[7:]
        + jax.random.uniform(key, shape=(29,), minval=-qjitter, maxval=qjitter)
    )

    return (
        geom_quat,
        pair_friction,
        dof_frictionloss,
        dof_armature,
        body_mass,
        qpos0,
    )

  (geom_quat, pair_friction, frictionloss, armature, body_mass, qpos0) = (
      rand(rng)
  )

  in_axes = jax.tree_util.tree_map(lambda x: None, model)
  in_axes = in_axes.tree_replace({
      "geom_quat": 0,
      "pair_friction": 0,
      "dof_frictionloss": 0,
      "dof_armature": 0,
      "body_mass": 0,
      "qpos0": 0,
  })

  model = model.tree_replace({
      "geom_quat": geom_quat,
      "pair_friction": pair_friction,
      "dof_frictionloss": frictionloss,
      "dof_armature": armature,
      "body_mass": body_mass,
      "qpos0": qpos0,
  })

  return model, in_axes


# Upstream G1 torso body id (see mujoco_playground g1/randomize.py).
g1_torso_body_id = 16


class G1JoystickWalkDR(g1_joystick.Joystick):
  """G1 joystick walk with launch-time slopes and OU wind."""

  def __init__(
      self,
      task: str = "flat_terrain",
      config: config_dict.ConfigDict | None = None,
      config_overrides: dict | None = None,
  ):
    if task != "flat_terrain":
      raise ValueError(
          "G1JoystickWalkDR requires the flat_terrain plane floor (the"
          " per-env slope tilts the plane geom; the rough_terrain hfield"
          f" cannot be tilted). Got task={task!r}."
      )
    config = config or default_config()
    # Wind replaces the upstream random velocity-impulse pushes.
    config.push_config.enable = False
    super().__init__(
        task=task, config=config, config_overrides=config_overrides
    )
    wc = self._config.dr_config
    # Drag.
    self._drag_coeff_torso = (
        0.5 * float(wc.rho) * float(wc.cd_torso) * float(wc.area_torso)
    )
    # Baseline wind sampling range.
    self._wind_speed_max = float(wc.wind_max_speed_kmph) / 3.6
    # OU parameters (host constants; exact exponential decay per step).
    self._gust_clip = float(wc.gust_fraction)
    self._gust_decay = math.exp(-self.dt / float(wc.gust_persist_s))
    self._gust_sigma = (
        0.5 * self._gust_clip * math.sqrt(2.0 / float(wc.gust_persist_s))
    )
    self._dir_decay = math.exp(-self.dt / float(wc.direction_persist_s))
    self._dir_sigma = math.radians(
        float(wc.direction_wander_deg)
    ) * math.sqrt(2.0 / float(wc.direction_persist_s))

  # ------------------------------------------------------------------
  # Terrain frame (per-env slope from the (possibly vectorized) model).
  # ------------------------------------------------------------------

  def slope_frame(self) -> jax.Array:
    """(3,3) floor rotation: columns [uphill, lateral, normal].

    Identity on flat ground; under the domain-randomization wrapper this
    reads the per-env tilted floor, so all slope-aware code below is
    automatically per-env under vmap.
    """
    q = self.mjx_model.geom_quat[self._floor_geom_id]
    return mjx_math.quat_to_mat(q).reshape(3, 3)

  def _slope_gravity(self, data: mjx.Data, frame: str) -> jax.Array:
    """Up-axis of `frame` expressed in slope coordinates (flat: world)."""
    return self.slope_frame().T @ self.get_gravity(data, frame)

  # ------------------------------------------------------------------
  # Wind.
  # ------------------------------------------------------------------

  def _advance_wind(self, info: dict) -> dict:
    """One OU step of the wind around the episode baseline."""
    rng, e_dir, e_gust = jax.random.split(info["rng"], 3)
    sqrt_dt = math.sqrt(self.dt)
    dtheta = (
        info["wind_dtheta"] * self._dir_decay
        + self._dir_sigma * sqrt_dt * jax.random.normal(e_dir)
    )
    gust = (
        info["wind_gust"] * self._gust_decay
        + self._gust_sigma * sqrt_dt * jax.random.normal(e_gust)
    )
    gust = jp.clip(gust, -self._gust_clip, self._gust_clip)
    base = info["wind_base"]
    speed = jp.linalg.norm(base)
    heading = jp.arctan2(base[1], base[0])
    theta = heading + dtheta
    wind = speed * (1.0 + gust) * jp.array([jp.cos(theta), jp.sin(theta)])
    return {
        **info,
        "rng": rng,
        "wind_dtheta": dtheta,
        "wind_gust": gust,
        "wind": wind,
    }

  # ------------------------------------------------------------------
  # Env API.
  # ------------------------------------------------------------------

  def reset(self, rng: jax.Array):
    # Upstream Joystick reset with slope-aware base placement and the
    # wind baseline sampled per episode. On flat ground the placement is
    # exactly upstream (n = +z, floor quat = identity).
    qpos = self._init_q
    qvel = jp.zeros(self.mjx_model.nv)

    # x=+U(-0.5, 0.5), y=+U(-0.5, 0.5), yaw=U(-3.14, 3.14).
    rng, key = jax.random.split(rng)
    dxy = jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)
    qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
    rng, key = jax.random.split(rng)
    yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
    quat = mjx_math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
    new_quat = mjx_math.quat_mul(qpos[3:7], quat)

    # Slope-aware placement: keep the keyframe base height measured
    # along the terrain normal, and align the stance with the terrain.
    # Flat: n=[0,0,1] -> z unchanged; floor quat identity -> no-op.
    floor_quat = self.mjx_model.geom_quat[self._floor_geom_id]
    n = self.slope_frame()[:, 2]
    h0 = self._init_q[2]
    z = (h0 - qpos[0] * n[0] - qpos[1] * n[1]) / n[2]
    qpos = qpos.at[2].set(z)
    new_quat = mjx_math.quat_mul(floor_quat, new_quat)
    qpos = qpos.at[3:7].set(new_quat)

    # qpos[7:]=*U(0.5, 1.5)
    rng, key = jax.random.split(rng)
    qpos = qpos.at[7:].set(
        qpos[7:] * jax.random.uniform(key, (29,), minval=0.5, maxval=1.5)
    )

    # d(xyzrpy)=U(-0.5, 0.5)
    rng, key = jax.random.split(rng)
    qvel = qvel.at[0:6].set(
        jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5)
    )

    data = mjx_env.make_data(
        self.mjx_model,
        qpos=qpos,
        qvel=qvel,
        ctrl=qpos[7:],
        impl=self.mjx_model.impl.value,
        naconmax=self._config.naconmax,
        njmax=self._config.njmax,
    )
    data = mjx.forward(self.mjx_model, data)

    # Phase, freq=U(1.0, 1.5)
    rng, key = jax.random.split(rng)
    gait_freq = jax.random.uniform(key, (1,), minval=1.25, maxval=1.5)
    phase_dt = 2 * jp.pi * self.dt * gait_freq
    phase = jp.array([0, jp.pi])

    rng, cmd_rng = jax.random.split(rng)
    cmd = self.sample_command(cmd_rng)

    # Sample push interval (pushes disabled; key kept for info parity).
    rng, push_rng = jax.random.split(rng)
    push_interval = jax.random.uniform(
        push_rng,
        minval=self._config.push_config.interval_range[0],
        maxval=self._config.push_config.interval_range[1],
    )
    push_interval_steps = jp.round(push_interval / self.dt).astype(jp.int32)

    # Wind baseline: speed=U(0, max), heading=U(0, 2pi).
    rng, wind_rng = jax.random.split(rng)
    wind_speed = jax.random.uniform(
        wind_rng, minval=0.0, maxval=self._wind_speed_max
    )
    rng, wind_rng = jax.random.split(rng)
    wind_heading = jax.random.uniform(wind_rng, maxval=2 * jp.pi)
    wind_base = wind_speed * jp.array(
        [jp.cos(wind_heading), jp.sin(wind_heading)]
    )

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
        # Phase related.
        "phase_dt": phase_dt,
        "phase": phase,
        # Push related.
        "push": jp.array([0.0, 0.0]),
        "push_step": 0,
        "push_interval_steps": push_interval_steps,
        # Wind state (OU offsets evolve in step; baseline fixed).
        "wind": wind_base,
        "wind_base": wind_base,
        "wind_dtheta": jp.zeros(()),
        "wind_gust": jp.zeros(()),
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

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    # Evolve the wind (smooth OU around the episode baseline), then push
    # the torso with quadratic drag; the force acts across all physics
    # substeps taken by super().step.
    info = self._advance_wind(state.info)
    state = state.replace(info=info)
    wind = info["wind"]
    torso_linvel = self.get_global_linvel(state.data, "torso")
    rel = wind - torso_linvel[:2]  # v_wind - v_torso.
    speed = jp.linalg.norm(rel)
    force_xy = self._drag_coeff_torso * speed * rel
    xfrc = state.data.xfrc_applied
    xfrc = xfrc.at[self._torso_body_id, :2].set(force_xy)
    xfrc = xfrc.at[self._torso_body_id, 2:].set(jp.zeros(4))
    state = state.replace(data=state.data.replace(xfrc_applied=xfrc))
    return super().step(state, action)

  # ------------------------------------------------------------------
  # Slope-aware termination / orientation (flat: identical to upstream).
  # ------------------------------------------------------------------

  def _get_termination(self, data: mjx.Data) -> jax.Array:
    # Fall = torso tips past horizontal relative to the terrain.
    fall_termination = self._slope_gravity(data, "torso")[-1] < 0.0
    contact_termination = data.sensordata[
        self._mj_model.sensor_adr[self._right_foot_left_foot_found_sensor]
    ] > 0
    contact_termination |= data.sensordata[
        self._mj_model.sensor_adr[self._left_foot_right_shin_found_sensor]
    ] > 0
    contact_termination |= data.sensordata[
        self._mj_model.sensor_adr[self._right_foot_left_shin_found_sensor]
    ] > 0
    return (
        fall_termination
        | contact_termination
        | jp.isnan(data.qpos).any()
        | jp.isnan(data.qvel).any()
    )

  def _cost_orientation(self, torso_zaxis: jax.Array) -> jax.Array:
    # Upright = torso up-axis along the terrain normal.
    return super()._cost_orientation(
        self.slope_frame().T @ torso_zaxis
    )

  # ------------------------------------------------------------------
  # Observations: gravity block relative to the terrain.
  # ------------------------------------------------------------------

  def _get_obs(
      self, data: mjx.Data, info: dict, contact: jax.Array
  ) -> mjx_env.Observation:
    # Copy of upstream Joystick._get_obs with the `state` gravity block
    # expressed against the terrain ("down" = slope normal), so an
    # upright-on-slope stance reads the same gravity as on flat ground.
    # The privileged gravity stays in the body frame (true tilt vs
    # world) so the value function sees the actual slope.
    gyro = self.get_gyro(data, "pelvis")
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gyro = (
        gyro
        + (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gyro
    )

    gravity_world = (
        data.site_xmat[self._pelvis_imu_site_id].T @ jp.array([0, 0, -1])
    )
    down_terrain = -self.slope_frame()[:, 2]  # flat: [0, 0, -1].
    gravity = data.site_xmat[self._pelvis_imu_site_id].T @ down_terrain
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gravity = (
        gravity
        + (2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gravity
    )

    joint_angles = data.qpos[7:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_angles = (
        joint_angles
        + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_pos
    )

    joint_vel = data.qvel[6:]
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
        noisy_gravity,  # 3 (terrain-relative)
        info["command"],  # 3
        noisy_joint_angles - self._default_pose,  # 29
        noisy_joint_vel,  # 29
        info["last_act"],  # 29
        phase,
    ])

    accelerometer = self.get_accelerometer(data, "pelvis")
    global_angvel = self.get_global_angvel(data, "pelvis")
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel()
    root_height = data.qpos[2]

    privileged_state = jp.hstack([
        state,
        gyro,  # 3
        accelerometer,  # 3
        gravity_world,  # 3 (body-frame vs world: true slope signal)
        linvel,  # 3
        global_angvel,  # 3
        joint_angles - self._default_pose,
        joint_vel,
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


# Registered on import; see rl/environment/__init__.py.
locomotion.register_environment(
    "G1JoystickWalkDR",
    functools.partial(G1JoystickWalkDR, task="flat_terrain"),
    default_config,
)

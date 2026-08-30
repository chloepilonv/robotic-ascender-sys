"""Environment + PPO config for the G1 fixed-rope ascender climb (mjlab)."""

from __future__ import annotations

import math

import torch

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs import mdp as base_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import NoiseModelWithAdditiveBiasCfg as BiasedNoise
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from mjlab.asset_zoo.robots.unitree_g1 import g1_constants as _g1
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_ACTION_SCALE

from . import climb_mode as CM
from . import mdp
from . import robot as R

# The 29 real G1 joints; excludes the rope slide joint (not on the real robot).
G1_JOINTS = SceneEntityCfg("robot", joint_names=(".*_joint",))
# Joints that get reset noise: everything except the right arm, which is on the
# rope at reset (noise there would start the weld violated and yank the tool).
RIGHT_ARM = SceneEntityCfg("robot", joint_names=("right_(shoulder|elbow|wrist).*_joint",))
NOISY_JOINTS = SceneEntityCfg("robot", joint_names=(r"(?!right_(shoulder|elbow|wrist)).*_joint",))
FEET = SceneEntityCfg("robot", geom_names=(R.FOOT_GEOM_REGEX,))
SLIDE_JOINT_CFG = SceneEntityCfg("robot", joint_names=(R.SLIDE_JOINT,))

def _actuator_groups(*targets: str) -> tuple[list[int], list[int]]:
  """Split the robot's actuator GROUPS into (matching, rest), by index.

  dr.pd_gains indexes `entity.actuators`, which holds one entry per actuator
  *group* (6 on the G1). SceneEntityCfg(actuator_names=...) instead resolves
  against `entity.actuator_names`, one entry per joint (29). Handing pd_gains a
  name subset therefore indexes the 6-long list with joint indices and raises
  IndexError. mjlab 1.6.0 only survives the all-actuators case, where resolve()
  collapses the ids to slice(None). So select groups by index instead.
  """
  hit = [
    i
    for i, a in enumerate(_g1.G1_ARTICULATION.actuators)
    if any(t in a.target_names_expr for t in targets)
  ]
  rest = [i for i in range(len(_g1.G1_ARTICULATION.actuators)) if i not in hit]
  return hit, rest


# The 4010 group (wrist pitch/yaw, 5 N.m effort limit) is the weak one: mjlab gives
# it kp=16.8 from rotor inertia at 10 Hz, while the gantry robot was commanded kp=2.
# +-20 % spans neither regime, so the policy has never seen a wrist that cannot hold
# its target. NOTE: wrist_roll -- the joint with the measured 0.445 rad droop -- sits
# in the 5020 group with the shoulders and elbow, so it cannot be widened without
# splitting that group, which would reorder the actuators and invalidate the resume
# checkpoint. Left at +-20 % for now; see ROADMAP.
_LOW_GAIN_GROUPS, _STIFF_GROUPS = _actuator_groups(
  ".*_wrist_pitch_joint", ".*_wrist_yaw_joint"
)
LOW_GAIN_ACTS = SceneEntityCfg("robot", actuator_ids=_LOW_GAIN_GROUPS)
STIFF_ACTS = SceneEntityCfg("robot", actuator_ids=_STIFF_GROUPS)


def _biased(white: float, bias: float) -> BiasedNoise:
  """White noise per step + one constant offset per episode.

  Real sensor error (encoder offset, gyro bias, FK error from a drooping wrist)
  is persistent within an episode. Measured on the gantry G1 at equal magnitude,
  a held bias left 4.3x the residual of white noise after averaging over a gait
  cycle -- the gait averages white noise away and cannot average away a bias.
  """
  return BiasedNoise(
    noise_cfg=Unoise(n_min=-white, n_max=white),
    bias_noise_cfg=Unoise(n_min=-bias, n_max=bias),
  )


class RatchetEnv(ManagerBasedRlEnv):
  """ManagerBasedRlEnv + the ascender cam: the rope slide joint never moves down.

  Before every physics substep the slide joint's lower limit (per env) is set
  to the highest position reached, so the constraint solver enforces "up only"
  together with the weld. (Overwriting qpos instead fights the solver.)
  """

  MAX_ACTION_DELAY = 2  # policy steps (0-2 @ 50 Hz = 0-40 ms, the real G1 pipeline)

  def reset(self, *args, **kwargs):
    out = super().reset(*args, **kwargs)
    self._ratchet_release(torch.arange(self.num_envs, device=self.device))
    self._mode_reset(torch.arange(self.num_envs, device=self.device))
    return out

  # --- climb rhythm (mode command) ---------------------------------------
  def _mode_reset(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    self.climb_mode[env_ids] = torch.randint(0, 2, (n,), device=self.device).float()
    self._slide_at_switch[env_ids] = self.sim.data.qpos[env_ids, self._slide_qadr]

  def _mode_update(self) -> None:
    robot = self.scene["robot"]
    carrier_x = robot.data.body_link_pos_w[:, self._carrier_body, 0]
    rel_x = carrier_x - robot.data.root_link_pos_w[:, 0]
    self.climb_mode, self._slide_at_switch = CM.update_mode(
      self.climb_mode, self.sim.data.qpos[:, self._slide_qadr], self._slide_at_switch, rel_x
    )

  def __init__(self, cfg, device: str, **kwargs):
    self.climb_mode = torch.zeros(cfg.scene.num_envs, device=device)  # obs term reads it during init
    super().__init__(cfg, device=device, **kwargs)
    n, a = self.num_envs, self.action_manager.total_action_dim
    self._act_hist = torch.zeros(self.MAX_ACTION_DELAY + 1, n, a, device=self.device)
    self._act_delay = torch.randint(0, self.MAX_ACTION_DELAY + 1, (n,), device=self.device)
    model = self.sim.mj_model
    jid = model.joint(f"robot/{R.SLIDE_JOINT}").id
    self._slide_jid = jid
    self.sim.expand_model_fields(("jnt_range",))  # per-env lower limit (the cam)
    self._carrier_body = self.scene["robot"].find_bodies(R.CARRIER_BODY)[0][0]
    self.climb_mode = torch.zeros(self.num_envs, device=self.device)
    self._slide_at_switch = torch.zeros(self.num_envs, device=self.device)
    self._slide_qadr = int(model.jnt_qposadr[jid])
    self._slide_dadr = int(model.jnt_dofadr[jid])
    self._sim_step = self.sim.step
    self.sim.step = self._ratcheted_step  # type: ignore[method-assign]

  def step(self, action: torch.Tensor):
    """Apply the action from `delay` steps ago (delay random per env, 0..2)."""
    self._act_hist = torch.roll(self._act_hist, 1, dims=0)
    self._act_hist[0] = action
    idx = torch.arange(self.num_envs, device=self.device)
    delayed = self._act_hist[self._act_delay, idx]
    self._mode_update()
    out = super().step(delayed)
    done = self.reset_buf.nonzero(as_tuple=False).flatten()
    if len(done):
      self._ratchet_release(done)
      self._mode_reset(done)
      self._act_hist[:, done] = 0.0
      self._act_delay[done] = torch.randint(
        0, self.MAX_ACTION_DELAY + 1, (len(done),), device=self.device
      )
    return out

  def _ratcheted_step(self) -> None:
    """The cam: the slide's lower limit (per env) = highest point reached, then step."""
    jr = self.sim.model.jnt_range
    jr[:, self._slide_jid, 0] = torch.maximum(
      jr[:, self._slide_jid, 0], self.sim.data.qpos[:, self._slide_qadr]
    )
    self._sim_step()

  def _ratchet_release(self, env_ids: torch.Tensor) -> None:
    """After a reset the cam re-engages where the carriage now is."""
    self.sim.model.jnt_range[env_ids, self._slide_jid, 0] = self.sim.data.qpos[env_ids, self._slide_qadr]


def make_env_cfg(slope_deg: float = 20.0, play: bool = False) -> ManagerBasedRlEnvCfg:
  """Climb task on a `slope_deg` incline (gravity-tilted; world +x is uphill)."""

  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=base_mdp.base_ang_vel, noise=_biased(0.2, 0.05)  # 0.05 rad/s gyro bias
    ),
    "projected_gravity": ObservationTermCfg(
      func=base_mdp.projected_gravity, noise=_biased(0.05, 0.02)
    ),
    "joint_pos": ObservationTermCfg(
      func=base_mdp.joint_pos_rel,
      # biased=True routes data.encoder_bias into the observation; without it the
      # `encoder_bias` event below is silently a no-op on the actor obs.
      params={"asset_cfg": G1_JOINTS, "biased": True},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=base_mdp.joint_vel_rel,
      params={"asset_cfg": G1_JOINTS},
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "actions": ObservationTermCfg(func=base_mdp.last_action),
    # On hardware this comes from wrist forward kinematics, so it inherits the
    # position error of every joint in the arm chain (0.445 rad measured at
    # left_wrist_roll => centimetres at the tool). In sim it is exact, so until now
    # the policy was free to trust a signal that is substantially wrong on the robot.
    "ascender_pos_b": ObservationTermCfg(
      func=mdp.ascender_pos_b,
      params={"asset_cfg": mdp.CARRIER},
      noise=_biased(0.01, 0.03),
    ),
    # "climb_mode": ObservationTermCfg(func=mdp.climb_mode),  # disabled: no mode FSM
  }
  critic_terms = {
    **actor_terms,
    # Privileged: true base velocity (no IMU integration drift on the real robot).
    "base_lin_vel": ObservationTermCfg(func=base_mdp.base_lin_vel),
  }
  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms, concatenate_terms=True, enable_corruption=not play
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms, concatenate_terms=True, enable_corruption=False
    ),
  }

  actions = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=G1_ACTION_SCALE,
      use_default_offset=True,
    )
  }

  events = {
    "reset_base": EventTermCfg(
      func=base_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        # No pose jitter: the carriage resets to the rope start = the ascender
        # channel of the nominal reset pose, so the weld starts satisfied.
        "pose_range": {},
        "velocity_range": {},
      },
    ),
    "reset_joints": EventTermCfg(
      func=base_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.05, 0.05),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": NOISY_JOINTS,
      },
    ),
    "reset_right_arm": EventTermCfg(
      func=base_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),  # exact solved IK pose — no noise on the rope arm
        "velocity_range": (0.0, 0.0),
        "asset_cfg": RIGHT_ARM,
      },
    ),
    "reset_slide": EventTermCfg(
      func=base_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SLIDE_JOINT_CFG,
      },
    ),
    # Ground friction, fixed per env for the run.
    "ice_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": FEET,
        "operation": "abs",
        "ranges": (0.4, 0.9),  # v0: packed snow..crampons; widen to 0.2 (ice) once it climbs
        "shared_random": True,
      },
    ),
    # Joint friction. The MJCF default class sets frictionloss = 0.3 N.m on every
    # joint; left_wrist_roll held 0.69 N.m at rest on the real robot against 0.003
    # N.m of gravity about that axis, so the sim under-models it by ~2x and never
    # varies it. Wide absolute range: 0.05 (free) .. 1.0 (worse than measured).
    "joint_friction": EventTermCfg(
      mode="startup",
      func=dr.joint_friction,  # alias of dr.dof_frictionloss
      params={
        "asset_cfg": G1_JOINTS,  # R.SLIDE_JOINT is "rope_slide", so the cam is untouched
        "operation": "abs",
        "ranges": (0.05, 1.0),
      },
    ),
    # Encoder calibration offset, held for the episode. The action pipeline
    # subtracts this from the PD target too, so it models a real miscalibration
    # rather than a cosmetic observation perturbation.
    "encoder_bias": EventTermCfg(
      mode="reset",
      func=dr.encoder_bias,
      params={"asset_cfg": NOISY_JOINTS, "bias_range": (-0.03, 0.03)},
    ),
    # Smaller on the rope arm: a biased target there drives the arm into the rigid
    # weld (solref 0.004) instead of just moving it. Widen once a run is stable.
    "encoder_bias_right_arm": EventTermCfg(
      mode="reset",
      func=dr.encoder_bias,
      params={"asset_cfg": RIGHT_ARM, "bias_range": (-0.01, 0.01)},
    ),
    # Motor strength: PD gains +-20 % per env (sim-to-real: gains never match).
    "motor_strength": EventTermCfg(
      mode="startup",
      func=dr.pd_gains,
      params={
        "asset_cfg": STIFF_ACTS,
        "operation": "scale",
        "kp_range": (0.8, 1.2),
        "kd_range": (0.8, 1.2),
      },
    ),
    # Wrists and ankle_roll separately, and much wider. mjlab derives kp = 14-17
    # for the wrists from rotor inertia at 10 Hz; the gantry robot was commanded
    # kp = 2. +-20 % spans neither regime, so the policy has never seen a wrist
    # that cannot hold its target. 0.15-1.5x spans ~2 .. ~25.
    "low_gain_strength": EventTermCfg(
      mode="startup",
      func=dr.pd_gains,
      params={
        "asset_cfg": LOW_GAIN_ACTS,
        "operation": "scale",
        "kp_range": (0.15, 1.5),
        "kd_range": (0.5, 1.5),
      },
    ),
    # Payload: torso mass +-10 % and CoM +-3 cm (battery, jacket, rope drag).
    "torso_mass": EventTermCfg(
      mode="startup",
      func=dr.body_mass,
      params={"asset_cfg": mdp.TORSO, "operation": "scale", "ranges": (0.9, 1.1)},
    ),
    "torso_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": mdp.TORSO,
        "operation": "add",
        "ranges": {0: (-0.03, 0.03), 1: (-0.03, 0.03), 2: (-0.03, 0.03)},
      },
    ),
    # Wind: steady, random horizontal direction, resampled each episode.
    "wind": EventTermCfg(
      func=mdp.wind_on_torso,
      mode="reset",
      params={"speed_range": (0.0, 10.0), "asset_cfg": mdp.TORSO},
    ),
  }

  rewards = {
    "uphill_velocity": RewardTermCfg(
      func=mdp.uphill_velocity, weight=4.0, params={"target": 0.4, "std": 0.3}
    ),
    "ascender_progress": RewardTermCfg(  # reward longer ascender slides
      func=mdp.ascender_progress, weight=1.5, params={"asset_cfg": mdp.SLIDE, "max_vel": 2.0}
    ),
    "rope_tension": RewardTermCfg(
      func=mdp.rope_tension_band, weight=0.5, params={"lo": 20.0, "hi": 150.0}
    ),
    "rope_jerk": RewardTermCfg(func=mdp.rope_tension_rate, weight=-0.002),
    "face_uphill": RewardTermCfg(func=mdp.face_uphill, weight=0.5),
    "facing_forward": RewardTermCfg(func=mdp.facing_forward, weight=0.3, params={"asset_cfg": mdp.TORSO}),
    "hiking_posture": RewardTermCfg(
      func=mdp.hiking_posture,
      weight=0.2,
      params={"targets": mdp.HIKE_POSE, "std": 0.4},
    ),
    # "stillness": RewardTermCfg(  # mode-dependent, disabled with climb_mode
    #   func=mdp.stillness, weight=-0.01, params={"asset_cfg": G1_JOINTS}
    # ),
    "foot_clearance": RewardTermCfg(
      func=mdp.foot_clearance, weight=0.5, params={"target": 0.12, "std": 0.05, "asset_cfg": mdp.FEET}
    ),
    "feet_air_time": RewardTermCfg(
      func=mdp.feet_air_time, weight=0.3, params={"threshold": 0.05, "asset_cfg": mdp.FEET}
    ),
    "upright": RewardTermCfg(
      func=vel_mdp.upright,
      weight=1.0,
      params={"std": math.sqrt(0.2), "asset_cfg": mdp.TORSO},
    ),
    "rope_side": RewardTermCfg(
      func=mdp.rope_side, weight=-5.0, params={"margin": 0.1, "asset_cfg": mdp.CARRIER}
    ),
    "hand_behind": RewardTermCfg(
      func=mdp.hand_behind_pelvis, weight=-2.0, params={"asset_cfg": mdp.CARRIER}
    ),
    "dof_pos_limits": RewardTermCfg(
      func=base_mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": G1_JOINTS}
    ),
    "action_rate_l2": RewardTermCfg(func=base_mdp.action_rate_l2, weight=-0.2),
    "joint_torques_l2": RewardTermCfg(
      func=base_mdp.joint_torques_l2, weight=-1e-5, params={"asset_cfg": G1_JOINTS}
    ),
    "alive": RewardTermCfg(func=base_mdp.is_alive, weight=0.5),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=base_mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=base_mdp.bad_orientation, params={"limit_angle": math.radians(60.0)}
    ),
    "facing_downhill": TerminationTermCfg(func=mdp.facing_downhill),
    "base_low": TerminationTermCfg(
      func=base_mdp.root_height_below_minimum, params={"minimum_height": 0.35}
    ),
  }

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": R.get_robot_cfg(slope_deg)},
      num_envs=16 if play else 4096,
      env_spacing=0.0,  # envs share the same rope line; MuJoCo worlds don't collide.
    ),
    observations=observations,
    actions=actions,
    commands={},
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum={},
    metrics={"rope_tension_N": MetricsTermCfg(func=mdp.rope_tension)},
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name=R.TORSO_BODY,
      distance=3.0,
      elevation=-10.0,
      azimuth=140.0,
    ),
    sim=SimulationCfg(
      nconmax=100,
      njmax=1000,  # equality + limits + contacts; 200 silently dropped the rope constraint.
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
        gravity=R.gravity_for_slope(slope_deg),
      ),
    ),
    decimation=4,  # 50 Hz policy, the G1 deployment rate.
    episode_length_s=15.0,
  )
  return cfg


def make_ppo_cfg(slope_deg: float) -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name=f"g1_ascender_slope{int(slope_deg)}",
    logger="tensorboard",  # switch to "wandb" (--agent.logger wandb) once wandb is set up.
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=5000,
  )

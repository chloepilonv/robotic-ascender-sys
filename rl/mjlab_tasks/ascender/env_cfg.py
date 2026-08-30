"""Environment + PPO config for the G1 fixed-rope ascender climb (mjlab)."""

from __future__ import annotations

import math

import torch

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs import mdp as base_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_ACTION_SCALE

from . import mdp
from . import robot as R

# The 29 real G1 joints; excludes the rope slide joint (not on the real robot).
G1_JOINTS = SceneEntityCfg("robot", joint_names=(".*_joint",))
FEET = SceneEntityCfg("robot", geom_names=(R.FOOT_GEOM_REGEX,))
SLIDE_JOINT_CFG = SceneEntityCfg("robot", joint_names=(R.SLIDE_JOINT,))


class RatchetEnv(ManagerBasedRlEnv):
  """ManagerBasedRlEnv + the ascender cam: the rope slide joint never moves down.

  After every physics substep the slide velocity is clamped to >= 0 and its
  position to >= the value before the substep. Cheap, jit-free, and exactly
  what a real ascender does under load.
  """

  def __init__(self, cfg, device: str, **kwargs):
    super().__init__(cfg, device=device, **kwargs)
    model = self.sim.mj_model
    jid = model.joint(f"robot/{R.SLIDE_JOINT}").id
    self._slide_qadr = int(model.jnt_qposadr[jid])
    self._slide_dadr = int(model.jnt_dofadr[jid])
    self._sim_step = self.sim.step
    self.sim.step = self._ratcheted_step  # type: ignore[method-assign]

  def _ratcheted_step(self) -> None:
    qpos = self.sim.data.qpos
    qvel = self.sim.data.qvel
    prev = qpos[:, self._slide_qadr].clone()
    self._sim_step()
    qvel[:, self._slide_dadr].clamp_(min=0.0)
    qpos[:, self._slide_qadr] = torch.maximum(qpos[:, self._slide_qadr], prev)


def make_env_cfg(slope_deg: float = 20.0, play: bool = False) -> ManagerBasedRlEnvCfg:
  """Climb task on a `slope_deg` incline (gravity-tilted; world +x is uphill)."""

  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=base_mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)
    ),
    "projected_gravity": ObservationTermCfg(
      func=base_mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)
    ),
    "joint_pos": ObservationTermCfg(
      func=base_mdp.joint_pos_rel,
      params={"asset_cfg": G1_JOINTS},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=base_mdp.joint_vel_rel,
      params={"asset_cfg": G1_JOINTS},
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "actions": ObservationTermCfg(func=base_mdp.last_action),
    "ascender_pos_b": ObservationTermCfg(
      func=mdp.ascender_pos_b, params={"asset_cfg": mdp.CARRIER}
    ),
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
        # No x/y jitter: the rope carrier resets to the rope start, which is the
        # wrist position of the nominal reset pose.
        "pose_range": {"z": (0.0, 0.02)},
        "velocity_range": {},
      },
    ),
    "reset_joints": EventTermCfg(
      func=base_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.05, 0.05),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": G1_JOINTS,
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
    # Ground: bare ice (~0.05) to crampon-on-neve (~0.9). Fixed per env for the run.
    "ice_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": FEET,
        "operation": "abs",
        "ranges": (0.05, 0.9),
        "shared_random": True,
      },
    ),
    # Wind: 0-30 m/s steady, random horizontal direction, resampled each episode.
    "wind": EventTermCfg(
      func=mdp.wind_on_torso,
      mode="reset",
      params={"speed_range": (0.0, 30.0), "asset_cfg": mdp.TORSO},
    ),
  }

  rewards = {
    "uphill_velocity": RewardTermCfg(
      func=mdp.uphill_velocity, weight=2.0, params={"target": 0.3, "std": 0.3}
    ),
    "ascender_progress": RewardTermCfg(
      func=mdp.ascender_progress, weight=1.0, params={"asset_cfg": mdp.SLIDE}
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
    "action_rate_l2": RewardTermCfg(func=base_mdp.action_rate_l2, weight=-0.1),
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

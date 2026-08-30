"""Task-specific MDP terms (observations, rewards, events) for the ascender climb."""

from __future__ import annotations

import math

import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

from . import robot as R

ROBOT = SceneEntityCfg("robot")
SLIDE = SceneEntityCfg("robot", joint_names=(R.SLIDE_JOINT,))
CARRIER = SceneEntityCfg("robot", body_names=(R.CARRIER_BODY,))
TORSO = SceneEntityCfg("robot", body_names=(R.TORSO_BODY,))

# Air density at ~7000 m is roughly half of sea level.
AIR_DENSITY = 0.55  # kg/m^3
DRAG_COEF = 1.0
TORSO_AREA = 0.45  # m^2 frontal area of a G1 torso + arms.


def _carrier_pos_w(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.body_link_pos_w[:, asset_cfg.body_ids, :].reshape(env.num_envs, 3)


# ----------------------------------------------------------------------------
# Observations
# ----------------------------------------------------------------------------


def ascender_pos_b(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = CARRIER) -> torch.Tensor:
  """Rope-carrier position in the pelvis frame.

  Deployable: the carrier is on the wrist, so on the real robot this is the
  wrist position from forward kinematics.
  """
  asset: Entity = env.scene[asset_cfg.name]
  rel_w = _carrier_pos_w(env, asset_cfg) - asset.data.root_link_pos_w
  return quat_apply_inverse(asset.data.root_link_quat_w, rel_w)


# ----------------------------------------------------------------------------
# Rewards
# ----------------------------------------------------------------------------


def uphill_velocity(env: ManagerBasedRlEnv, target: float, std: float) -> torch.Tensor:
  """Track a target base velocity along world +x (uphill)."""
  asset: Entity = env.scene[ROBOT.name]
  vx = asset.data.root_link_lin_vel_w[:, 0]
  return torch.exp(-torch.square(vx - target) / std**2)


def ascender_progress(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SLIDE, max_vel: float = 1.0
) -> torch.Tensor:
  """Reward sliding the ascender up the rope (velocity is already >= 0 by the ratchet)."""
  asset: Entity = env.scene[asset_cfg.name]
  vel = asset.data.joint_vel[:, asset_cfg.joint_ids].reshape(env.num_envs)
  return torch.clamp(vel, 0.0, max_vel)


def rope_side(
  env: ManagerBasedRlEnv, margin: float, asset_cfg: SceneEntityCfg = CARRIER
) -> torch.Tensor:
  """Penalty when the pelvis crosses to the rope's side (rope is on the right, -y)."""
  asset: Entity = env.scene[asset_cfg.name]
  rope_y = _carrier_pos_w(env, asset_cfg)[:, 1]
  pelvis_y = asset.data.root_link_pos_w[:, 1]
  return torch.relu(rope_y + margin - pelvis_y)


def hand_behind_pelvis(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = CARRIER
) -> torch.Tensor:
  """Penalty when the ascender is downhill of the pelvis (arm dragged behind)."""
  asset: Entity = env.scene[asset_cfg.name]
  dx = _carrier_pos_w(env, asset_cfg)[:, 0] - asset.data.root_link_pos_w[:, 0]
  return torch.relu(-dx)


# ----------------------------------------------------------------------------
# Events
# ----------------------------------------------------------------------------


def wind_on_torso(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  speed_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = TORSO,
) -> None:
  """Per-episode steady wind: quadratic drag on the torso, random horizontal direction.

  F = 1/2 rho Cd A v^2.  Himalayan reference: 10 m/s ~ 12 N (breeze), 20 m/s ~ 50 N
  (typical summit-day wind), 30 m/s ~ 110 N (storm; climbers turn back).
  """
  asset: Entity = env.scene[asset_cfg.name]
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  n = len(env_ids)
  speed = torch.empty(n, device=env.device).uniform_(*speed_range)
  heading = torch.empty(n, device=env.device).uniform_(-math.pi, math.pi)
  mag = 0.5 * AIR_DENSITY * DRAG_COEF * TORSO_AREA * speed**2
  force = torch.stack(
    (mag * torch.cos(heading), mag * torch.sin(heading), torch.zeros_like(mag)), dim=-1
  )
  forces = force.unsqueeze(1).expand(n, 1, 3)
  torques = torch.zeros_like(forces)
  asset.write_external_wrench_to_sim(
    forces, torques, env_ids=env_ids, body_ids=asset_cfg.body_ids
  )


# ----------------------------------------------------------------------------
# Metrics (logged, not rewarded)
# ----------------------------------------------------------------------------


def rope_tension(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SLIDE) -> torch.Tensor:
  """|constraint force| on the rope slide dof, N — the sim twin of the load cell."""
  return env.sim.data.qfrc_constraint[:, env._slide_dadr].abs()  # type: ignore[attr-defined]


# ----------------------------------------------------------------------------
# Heading (face uphill) and the climb rhythm (mode command)
# ----------------------------------------------------------------------------


def _forward_x(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Cosine between the pelvis forward axis and world +x (uphill)."""
  asset: Entity = env.scene[ROBOT.name]
  fwd = torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(env.num_envs, 3)
  return quat_apply(asset.data.root_link_quat_w, fwd)[:, 0]


def face_uphill(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Reward: 1 when the torso faces uphill, -1 when it faces downhill."""
  return _forward_x(env)


def facing_downhill(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Termination: turned more than 90 deg away from uphill."""
  return _forward_x(env) < 0.0


def climb_mode(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Observation: the mode command (0 walk, 1 slide), kept by RatchetEnv."""
  return env.climb_mode.unsqueeze(-1)  # type: ignore[attr-defined]


def mode_uphill_velocity(env: ManagerBasedRlEnv, target: float, std: float) -> torch.Tensor:
  """WALK: track `target` uphill speed. SLIDE: no reward (standing must not pay; v7 loophole)."""
  asset: Entity = env.scene[ROBOT.name]
  vx = asset.data.root_link_lin_vel_w[:, 0]
  r = torch.exp(-torch.square(vx - target) / std**2)
  return torch.where(env.climb_mode > 0.5, torch.zeros_like(r), r)  # type: ignore[attr-defined]


def in_slide(env: ManagerBasedRlEnv) -> torch.Tensor:
  """1 while in SLIDE mode (used as a per-step time-pressure penalty)."""
  return (env.climb_mode > 0.5).float()  # type: ignore[attr-defined]


def mode_ascender_progress(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SLIDE, max_vel: float = 1.0
) -> torch.Tensor:
  """SLIDE: reward pushing the ascender up. WALK: penalise moving it."""
  asset: Entity = env.scene[asset_cfg.name]
  vel = asset.data.joint_vel[:, asset_cfg.joint_ids].reshape(env.num_envs)
  slide = env.climb_mode > 0.5  # type: ignore[attr-defined]
  return torch.where(slide, torch.clamp(vel, 0.0, max_vel), -torch.abs(vel))


# ----------------------------------------------------------------------------
# Hiking posture + stillness
# ----------------------------------------------------------------------------

HIKE_POSE = {  # legs flexed like a hiker going uphill (rad); ankle set per slope in env_cfg
  ".*_hip_pitch_joint": -0.45,
  ".*_knee_joint": 0.85,
  "waist_pitch_joint": 0.15,
}


def hiking_posture(env: ManagerBasedRlEnv, targets: dict, std: float, asset_cfg: SceneEntityCfg = ROBOT) -> torch.Tensor:
  """Reward: legs/waist near a hiking pose (hips and knees flexed, slight forward lean)."""
  asset: Entity = env.scene[asset_cfg.name]
  ids, vals = [], []
  for i, name in enumerate(asset.joint_names):
    for pat, v in targets.items():
      if __import__("re").fullmatch(pat, name):
        ids.append(i)
        vals.append(v)
  q = asset.data.joint_pos[:, ids]
  err = q - torch.tensor(vals, device=env.device)
  return torch.exp(-torch.sum(torch.square(err), dim=1) / (std**2 * len(ids)))


def stillness(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = ROBOT) -> torch.Tensor:
  """Penalty on joint speed while in SLIDE mode (the body should be posed, not jiggling)."""
  asset: Entity = env.scene[asset_cfg.name]
  v2 = torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
  return torch.where(env.climb_mode > 0.5, v2, torch.zeros_like(v2))  # type: ignore[attr-defined]


# ----------------------------------------------------------------------------
# Rope as support: moderate, continuous tension, no jerking
# ----------------------------------------------------------------------------


def _tension(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.sim.data.qfrc_constraint[:, env._slide_dadr].abs()  # type: ignore[attr-defined]


def rope_tension_band(env: ManagerBasedRlEnv, lo: float, hi: float) -> torch.Tensor:
  """Reward ~1 when rope tension is inside [lo, hi] N, decaying outside (soft band)."""
  t = _tension(env)
  mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
  return torch.exp(-torch.square((t - mid) / half))


def rope_tension_rate(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Penalty on |change of tension| per step (N) — no sudden pulling or jerking."""
  t = _tension(env)
  prev = getattr(env, "_prev_tension", None)
  if prev is None or prev.shape != t.shape:
    prev = t.clone()
  env._prev_tension = t.clone()  # type: ignore[attr-defined]
  return torch.abs(t - prev)

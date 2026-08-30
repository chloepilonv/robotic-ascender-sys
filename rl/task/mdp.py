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


def walk_velocity(env: ManagerBasedRlEnv, target: float, std: float) -> torch.Tensor:
  """Track a target base velocity along +x, scaled by env.walk_command (0 = stand, 1 = walk)."""
  asset: Entity = env.scene[ROBOT.name]
  vx = asset.data.root_link_lin_vel_w[:, 0]
  want = target * env.walk_command  # type: ignore[attr-defined]
  return torch.exp(-torch.square(vx - want) / std**2)


def walk_command(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Observation: the walk/stop command (1 = walk at target speed, 0 = stand still)."""
  return env.walk_command.unsqueeze(-1)  # type: ignore[attr-defined]


def ascender_progress(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SLIDE, max_vel: float = 1.0
) -> torch.Tensor:
  """Reward sliding the ascender up the rope (velocity is already >= 0 by the ratchet)."""
  asset: Entity = env.scene[asset_cfg.name]
  vel = asset.data.joint_vel[:, asset_cfg.joint_ids].reshape(env.num_envs)
  return torch.clamp(vel, 0.0, max_vel)

FEET = SceneEntityCfg("robot", body_names=(".*_ankle_roll_link",))


def foot_clearance(env: ManagerBasedRlEnv, target: float = 0.12, std: float = 0.05,
                   asset_cfg: SceneEntityCfg = FEET) -> torch.Tensor:
  """Reward feet reaching `target` height (m) — encourages lifting, not shuffling."""
  asset: Entity = env.scene[asset_cfg.name]
  foot_z = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2]  # (num_envs, num_feet)
  return torch.exp(-torch.sum(torch.square(foot_z - target) / std**2, dim=1) / foot_z.shape[1])


def feet_air_time(env: ManagerBasedRlEnv, threshold: float = 0.05,
                  asset_cfg: SceneEntityCfg = FEET) -> torch.Tensor:
  """Reward feet that are airborne (z velocity above threshold) — encourages committed steps, not shuffling."""
  asset: Entity = env.scene[asset_cfg.name]
  foot_vel_z = asset.data.body_link_vel_w[:, asset_cfg.body_ids, 5]  # vz component
  airborne = (foot_vel_z.abs() > threshold).float()
  return airborne.mean(dim=1)


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

def facing_forward(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = TORSO) -> torch.Tensor:
  """Reward: mean of pelvis and torso forward-axis alignment with uphill (+x).

  Both the hips (pelvis/root) and the torso should face uphill. Returns the
  average cosine of each body's forward axis with world +x: +1 = both facing
  uphill, -1 = both facing downhill.
  """
  pelvis_fwd = _forward_x(env)
  asset: Entity = env.scene[asset_cfg.name]
  torso_quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].reshape(env.num_envs, 4)
  fwd = torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(env.num_envs, 3)
  torso_fwd = quat_apply(torso_quat, fwd)[:, 0]
  return 0.5 * (pelvis_fwd + torso_fwd)


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


# ----------------------------------------------------------------------------
# Per-env slope (0-40 deg) inside one run
# ----------------------------------------------------------------------------
# Global gravity is tilted for NOMINAL_SLOPE_DEG. Each env gets a constant extra
# force on every body, m_i * (g_slope - g_nominal), which is exactly its own
# tilted gravity. The reset pose comes from a table solved per slope, and the
# "IMU" observation / upright reward use the effective gravity.

_G = 9.81


def _g_dir(slope_deg: torch.Tensor) -> torch.Tensor:
  s = torch.deg2rad(slope_deg)
  return torch.stack((-torch.sin(s), torch.zeros_like(s), -torch.cos(s)), dim=-1)


def reset_slope_wind(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  slope_range: tuple[float, float],
  speed_range: tuple[float, float],
) -> None:
  """Sample slope + wind per env; write spawn pose and the constant per-body forces."""
  import numpy as np

  asset: Entity = env.scene["robot"]
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  n = len(env_ids)
  table = R.spawn_table()
  keys = np.array(sorted(table))
  slopes = torch.empty(n, device=env.device).uniform_(*slope_range)
  # snap the spawn pose to the nearest table entry (physics uses the exact slope)
  snapped = keys[np.abs(keys[None, :] - slopes.cpu().numpy()[:, None]).argmin(1)]
  env.slope_deg[env_ids] = slopes  # type: ignore[attr-defined]

  # --- spawn pose per env ---
  root = torch.zeros(n, 13, device=env.device)
  jpos = asset.data.default_joint_pos[env_ids].clone()
  names = asset.joint_names
  slide_i = names.index(R.SLIDE_JOINT)
  cx20 = table[R.NOMINAL_SLOPE_DEG][3]
  import re
  for k, sd in enumerate(snapped):
    pos, rot, jp, cx = table[float(sd)]
    root[k, :3] = torch.tensor(pos, dtype=torch.float32)
    root[k, 3:7] = torch.tensor(rot, dtype=torch.float32)
    for j, nm in enumerate(names):
      for pat, val in jp.items():
        if re.fullmatch(pat, nm):
          jpos[k, j] = val
    jpos[k, slide_i] = cx - cx20  # carriage sits where this pose puts the channel
  asset.write_root_state_to_sim(root, env_ids=env_ids)
  asset.write_joint_state_to_sim(jpos, torch.zeros_like(jpos), env_ids=env_ids)

  # --- constant forces: gravity correction on every body (+ wind on the torso) ---
  m = env.sim.mj_model
  body_ids = list(range(len(asset.body_names)))
  masses = torch.tensor([float(m.body(f"robot/{b}").mass[0]) for b in asset.body_names], device=env.device, dtype=torch.float32)
  dg = _g_dir(slopes) - _g_dir(torch.full_like(slopes, R.NOMINAL_SLOPE_DEG))  # [n,3]
  forces = (_G * masses[None, :, None] * dg[:, None, :]).float()  # [n, nbody, 3]
  speed = torch.empty(n, device=env.device).uniform_(*speed_range)
  heading = torch.empty(n, device=env.device).uniform_(-math.pi, math.pi)
  mag = 0.5 * AIR_DENSITY * DRAG_COEF * TORSO_AREA * speed**2
  ti = asset.body_names.index(R.TORSO_BODY)
  forces[:, ti, 0] += mag * torch.cos(heading)
  forces[:, ti, 1] += mag * torch.sin(heading)
  asset.write_external_wrench_to_sim(forces, torch.zeros_like(forces), env_ids=env_ids, body_ids=body_ids)


def projected_gravity_eff(env: ManagerBasedRlEnv) -> torch.Tensor:
  """What the IMU would read: the env's own gravity direction in the pelvis frame."""
  asset: Entity = env.scene[ROBOT.name]
  return quat_apply_inverse(asset.data.root_link_quat_w, _g_dir(env.slope_deg))  # type: ignore[attr-defined]


def upright_eff(env: ManagerBasedRlEnv, std: float, asset_cfg: SceneEntityCfg = TORSO) -> torch.Tensor:
  """Torso upright w.r.t. the env's own gravity."""
  asset: Entity = env.scene[asset_cfg.name]
  q = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].reshape(env.num_envs, 4)
  g_b = quat_apply_inverse(q, _g_dir(env.slope_deg))  # type: ignore[attr-defined]
  return torch.exp(-torch.sum(torch.square(g_b[:, :2]), dim=1) / std**2)


def bad_orientation_eff(env: ManagerBasedRlEnv, limit_angle: float) -> torch.Tensor:
  asset: Entity = env.scene[ROBOT.name]
  g_b = quat_apply_inverse(asset.data.root_link_quat_w, _g_dir(env.slope_deg))  # type: ignore[attr-defined]
  return torch.acos(torch.clamp(-g_b[:, 2], -1.0, 1.0)) > limit_angle

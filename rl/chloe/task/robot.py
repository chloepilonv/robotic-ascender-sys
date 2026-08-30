"""G1 + ascender as an mjlab Entity, on the rope rail from assets/robots/mujoco/rope_rail.py.

Slope: we do NOT tilt the floor. We tilt gravity (see env_cfg.py), which is
physically identical and keeps the rope a plain +x line at fixed height.
"""

from __future__ import annotations

import functools
import math
import sys
from pathlib import Path

import mujoco

from mjlab.asset_zoo.robots.unitree_g1 import g1_constants as g1
from mjlab.entity import EntityCfg

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "assets/robots/mujoco"))
import rope_rail as rail  # noqa: E402  (shared, mujoco-only)

G1_ASCENDER_XML = REPO_ROOT / "assets/robots/mujoco/g1_unitree_ascender.xml"

ROPE_BODY, CARRIER_BODY = rail.ROPE_BODY, rail.CARRIER_BODY
SLIDE_JOINT, WRIST_BODY = rail.SLIDE_JOINT, rail.WRIST_BODY
TORSO_BODY = "torso_link"
FOOT_GEOM_REGEX = ".*_foot_[0-3]"

BASE_JOINT_POS = dict(g1.KNEES_BENT_KEYFRAME.joint_pos)
FOOT_CLEARANCE = 0.005  # m between the lowest foot sphere and the floor at spawn


def slope_quat(slope_deg: float) -> tuple[float, float, float, float]:
  """Root orientation (w,x,y,z) whose body-up is anti-parallel to tilted gravity."""
  s = math.radians(slope_deg)
  return (math.cos(s / 2), 0.0, math.sin(s / 2), 0.0)


def gravity_for_slope(slope_deg: float) -> tuple[float, float, float]:
  """Gravity vector so that world +x is 'uphill' on a slope of `slope_deg`."""
  s = math.radians(slope_deg)
  return (-9.81 * math.sin(s), 0.0, -9.81 * math.cos(s))


def _base_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(G1_ASCENDER_XML))
  # mjlab supplies its own actuators (Unitree motor specs) and init keyframe.
  for act in list(spec.actuators):
    spec.delete(act)
  for key in list(spec.keys):
    spec.delete(key)
  # Name the foot contact spheres so domain randomisation can target them.
  for side in ("left", "right"):
    k = 0
    for geom in spec.body(f"{side}_ankle_roll_link").geoms:
      if geom.classname is not None and geom.classname.name == "foot":
        geom.name = f"{side}_foot_{k}"
        k += 1
  return spec


def _body_joint_pos(slope_deg: float) -> dict:
  joint_pos = dict(BASE_JOINT_POS)
  # Facing uphill the ankles dorsiflex by the slope angle (soft limit -0.87 rad).
  joint_pos[".*_ankle_pitch_joint"] = max(-0.85, -0.363 - math.radians(slope_deg))
  return joint_pos


_Z_CACHE: dict[float, float] = {}
_POSE_CACHE: dict[float, dict] = {}


def init_pos(slope_deg: float) -> tuple[float, float, float]:
  """Spawn position: feet just touching the floor in the tilted reset pose (cached)."""
  if slope_deg not in _Z_CACHE:
    model = _base_spec().compile()
    data = mujoco.MjData(model)
    rail.set_pose(model, data, (0, 0, 0.8), slope_quat(slope_deg), _body_joint_pos(slope_deg))
    feet = [i for i in range(model.ngeom) if model.geom(i).name.endswith(tuple(f"_foot_{k}" for k in range(4)))]
    lowest = min(data.geom_xpos[i][2] - model.geom_size[i][0] for i in feet)
    _Z_CACHE[slope_deg] = 0.8 - lowest + FOOT_CLEARANCE
  return (0.0, 0.0, _Z_CACHE[slope_deg])


def reset_joint_pos(slope_deg: float) -> dict:
  """Reset joint angles incl. the solved arm and the rail joint (cached per slope)."""
  if slope_deg not in _POSE_CACHE:
    _POSE_CACHE[slope_deg] = rail.add_rope_rail(
      _base_spec(), init_pos(slope_deg), slope_quat(slope_deg), _body_joint_pos(slope_deg)
    )
  return _POSE_CACHE[slope_deg]


def get_spec(slope_deg: float = 0.0) -> mujoco.MjSpec:
  spec = _base_spec()
  rail.add_rope_rail(spec, init_pos(slope_deg), slope_quat(slope_deg), _body_joint_pos(slope_deg))
  return spec


def get_robot_cfg(slope_deg: float) -> EntityCfg:
  init = EntityCfg.InitialStateCfg(
    pos=init_pos(slope_deg),
    rot=slope_quat(slope_deg),
    joint_pos=reset_joint_pos(slope_deg),
    joint_vel={".*": 0.0},
  )
  return EntityCfg(
    init_state=init,
    spec_fn=functools.partial(get_spec, slope_deg),
    articulation=g1.G1_ARTICULATION,
  )


# ----------------------------------------------------------------------------
# Per-env slope: spawn table (used by the reset event in mdp.py)
# ----------------------------------------------------------------------------

NOMINAL_SLOPE_DEG = 20.0  # the global gravity / rope geometry is built for this slope
SLOPE_TABLE_DEG = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0)
_SPAWN_CACHE: dict = {}


def spawn_table() -> dict:
  """slope_deg -> (root_pos, root_quat, joint_pos dict, channel_x_world). Cached."""
  if _SPAWN_CACHE:
    return _SPAWN_CACHE
  for sd in SLOPE_TABLE_DEG:
    pos, rot, jp = init_pos(sd), slope_quat(sd), reset_joint_pos(sd)
    model = _base_spec().compile()
    data = mujoco.MjData(model)
    rail.set_pose(model, data, pos, rot, jp)
    wb = model.body(WRIST_BODY).id
    grip_link, _ = rail.ascender_channel(model)
    cx = float((data.xpos[wb] + data.xmat[wb].reshape(3, 3) @ grip_link)[0])
    _SPAWN_CACHE[sd] = (pos, rot, jp, cx)
  return _SPAWN_CACHE

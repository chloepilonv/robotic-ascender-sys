"""G1 + ascender as an mjlab Entity, with the fixed rope and the one-way carrier.

Model layout (one MjSpec, one mjlab entity called "robot"):

  world
  ├── pelvis (freejoint) ... right_wrist_yaw_link (+ ascender meshes)   <- the G1
  ├── rope            static visual cylinder along world +x (no collision)
  └── rope_carriage   body with ONE slide joint along +x (the physics rope)
        `connect` equality: carrier origin == right_wrist_yaw_link origin

The wrist origin is the rope axis by design (see assets/ascender/MOUNT.md: the
cam head is centred on the wrist joint so the rope load bypasses the ±5 N·m
wrist motor). The carrier can move along the rope only; `RatchetEnv` in
env_cfg.py forbids moving *down* (qvel >= 0), which is the real ascender cam.

Slope: we do NOT tilt the floor. We tilt gravity (see env_cfg.py), which is
physically identical and keeps the rope a plain +x line at fixed height.
"""

from __future__ import annotations

import functools
import math
import re
from pathlib import Path

import mujoco
import numpy as np

from mjlab.asset_zoo.robots.unitree_g1 import g1_constants as g1
from mjlab.entity import EntityCfg

REPO_ROOT = Path(__file__).resolve().parents[3]
G1_ASCENDER_XML = REPO_ROOT / "assets/robots/mujoco/g1_unitree_ascender.xml"

ROPE_BODY = "rope"
CARRIER_BODY = "rope_carriage"
SLIDE_JOINT = "rope_slide"  # no "_joint" suffix: ".*_joint" regexes skip it.
WRIST_BODY = "right_wrist_yaw_link"
TORSO_BODY = "torso_link"
FOOT_GEOM_REGEX = ".*_foot_[0-3]"

ROPE_LENGTH = 30.0  # m of rope uphill of the start.
ROPE_TAIL = 1.0  # m of rope downhill of the start.
ROPE_RADIUS = 0.0055  # 11 mm static rope (Petzl Basic takes 8-13 mm)
CARRIER_MASS = 0.05

# Joint angles at reset (mjlab knees-bent pose; ankles adapted per slope in env_cfg).
BASE_JOINT_POS = dict(g1.KNEES_BENT_KEYFRAME.joint_pos)


def _wrist_pos_in_reset_pose(spec: mujoco.MjSpec, slope_deg: float) -> np.ndarray:
  """World position of the wrist origin in the reset pose used by get_robot_cfg."""
  model = spec.compile()
  data = mujoco.MjData(model)
  init = get_robot_cfg(slope_deg).init_state
  data.qpos[:3] = init.pos
  data.qpos[3:7] = init.rot
  for j in range(model.njnt):
    if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
      continue
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
    for pat, val in init.joint_pos.items():
      if re.fullmatch(pat, name):
        data.qpos[model.jnt_qposadr[j]] = val
  mujoco.mj_forward(model, data)
  return data.xpos[model.body(WRIST_BODY).id].copy()


def get_spec(slope_deg: float = 0.0) -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(G1_ASCENDER_XML))

  # mjlab supplies its own actuator model (stiffness/damping/armature from the
  # Unitree motor specs) and its own init keyframe; drop the XML's.
  for act in list(spec.actuators):
    spec.delete(act)
  for key in list(spec.keys):
    spec.delete(key)
  # Name the foot contact spheres so domain randomisation can target them.
  for side in ("left", "right"):
    body = spec.body(f"{side}_ankle_roll_link")
    k = 0
    for geom in body.geoms:
      if geom.classname is not None and geom.classname.name == "foot":
        geom.name = f"{side}_foot_{k}"
        k += 1

  spec.body(WRIST_BODY).add_site(name="ascender_anchor", pos=[0, 0, 0], group=5)
  wrist0 = _wrist_pos_in_reset_pose(spec, slope_deg)
  wb = spec.worldbody

  # Visual rope: a static cylinder along +x through the rest wrist position.
  rope = wb.add_body(name=ROPE_BODY, pos=wrist0)
  rope_geom = rope.add_geom(
    name="rope_geom",
    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
    size=[ROPE_RADIUS, 1.0, 0.0],
    rgba=[0.9, 0.3, 0.1, 1.0],
    contype=0,
    conaffinity=0,
    group=2,
  )
  rope_geom.fromto = [-ROPE_TAIL, 0, 0, ROPE_LENGTH, 0, 0]

  # Physics rope: a light carrier on a slide joint along +x, welded (point
  # constraint) to the wrist origin.
  carrier = wb.add_body(name=CARRIER_BODY, pos=wrist0)
  carrier.add_joint(
    name=SLIDE_JOINT,
    type=mujoco.mjtJoint.mjJNT_SLIDE,
    axis=[1, 0, 0],
    range=[-ROPE_LENGTH, ROPE_LENGTH],  # symmetric: mjlab soft limits (0.9) must include 0
    damping=2.0,
    frictionloss=0.1,
  )
  carrier.add_geom(
    name="carrier_geom",
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    size=[0.02, 0, 0],
    mass=CARRIER_MASS,
    rgba=[1.0, 0.9, 0.0, 1.0],
    contype=0,
    conaffinity=0,
    group=2,
  )
  carrier.add_site(name="carrier_anchor", pos=[0, 0, 0], group=5)
  # Site-to-site connect: no compiler-computed anchor offsets, the two site
  # origins are simply forced to coincide.
  eq = spec.add_equality(
    type=mujoco.mjtEq.mjEQ_CONNECT,
    name="ascender_grip",
    objtype=mujoco.mjtObj.mjOBJ_SITE,
    name1="carrier_anchor",
    name2="ascender_anchor",
  )
  eq.solref = [0.004, 1.0]
  eq.solimp = [0.99, 0.999, 0.001, 0.5, 2.0]  # near-hard constraint (rope does not stretch)
  return spec


def slope_quat(slope_deg: float) -> tuple[float, float, float, float]:
  """Root orientation (w,x,y,z) whose body-up is anti-parallel to tilted gravity."""
  s = math.radians(slope_deg)
  return (math.cos(s / 2), 0.0, math.sin(s / 2), 0.0)


def gravity_for_slope(slope_deg: float) -> tuple[float, float, float]:
  """Gravity vector so that world +x is 'uphill' on a slope of `slope_deg`."""
  s = math.radians(slope_deg)
  return (-9.81 * math.sin(s), 0.0, -9.81 * math.cos(s))


def get_robot_cfg(slope_deg: float) -> EntityCfg:
  joint_pos = dict(BASE_JOINT_POS)
  # Facing uphill the ankles dorsiflex by the slope angle (soft limit -0.87 rad).
  joint_pos[".*_ankle_pitch_joint"] = max(-0.85, -0.363 - math.radians(slope_deg))
  joint_pos[SLIDE_JOINT] = 0.0
  init = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.80),
    rot=slope_quat(slope_deg),
    joint_pos=joint_pos,
    joint_vel={".*": 0.0},
  )
  return EntityCfg(
    init_state=init,
    spec_fn=functools.partial(get_spec, slope_deg),
    articulation=g1.G1_ARTICULATION,
  )


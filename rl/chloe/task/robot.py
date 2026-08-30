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
SPIN_JOINT = "rope_spin"  # hinge about the rope axis: the tool can rotate around the rope only.
WRIST_BODY = "right_wrist_yaw_link"
TORSO_BODY = "torso_link"
FOOT_GEOM_REGEX = ".*_foot_[0-3]"

ROPE_LENGTH = 30.0  # m of rope uphill of the start.
ROPE_TAIL = 1.0  # m of rope downhill of the start.
ROPE_RADIUS = 0.0055  # 11 mm static rope (Petzl Basic takes 8-13 mm)
CARRIER_MASS = 0.05

# Joint angles at reset (mjlab knees-bent pose; ankles adapted per slope in env_cfg).
BASE_JOINT_POS = dict(g1.KNEES_BENT_KEYFRAME.joint_pos)


ASCENDER_MESH = REPO_ROOT / "assets/robots/mujoco/meshes/ascender_collision.obj"
RIGHT_WRIST = ("right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint")


def ascender_channel(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
  """(channel centre, channel axis) of the ascender in the wrist-link frame.

  The rope runs through the middle of the tool: bounding-box centre of the
  collision mesh, mapped through the mount pose on `right_wrist_yaw_link`
  (same definition as app/harness/robot_variants.py). The channel axis is the
  tool-frame Z (cam head up).
  """
  wb = model.body(WRIST_BODY).id
  gid = next(
    i for i in range(model.ngeom)
    if model.geom_bodyid[i] == wb
    and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
    and "ascender_collision" in model.mesh(model.geom_dataid[i]).name
  )
  verts = np.array(
    [list(map(float, l.split()[1:4])) for l in open(ASCENDER_MESH) if l.startswith("v ")]
  )
  centre_mesh = (verts.min(0) + verts.max(0)) / 2
  rot = np.zeros(9)
  mujoco.mju_quat2Mat(rot, model.geom_quat[gid])
  rot = rot.reshape(3, 3)
  return model.geom_pos[gid] + rot @ centre_mesh, rot @ np.array([0.0, 0.0, 1.0])


def _set_pose(model, data, init) -> None:
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


def _solve_wrist(model, data, base_joint_pos: dict) -> dict:
  """Wrist roll/pitch/yaw that make the channel axis parallel to the rope (+x)."""
  from scipy.optimize import minimize

  wb = model.body(WRIST_BODY).id
  _, axis_link = ascender_channel(model)
  adr = [model.jnt_qposadr[model.joint(n).id] for n in RIGHT_WRIST]
  lo = [model.jnt_range[model.joint(n).id][0] * 0.9 for n in RIGHT_WRIST]
  hi = [model.jnt_range[model.joint(n).id][1] * 0.9 for n in RIGHT_WRIST]

  def cost(q):
    data.qpos[adr] = q
    mujoco.mj_forward(model, data)
    axis_w = data.xmat[wb].reshape(3, 3) @ axis_link
    return (1.0 - axis_w[0]) + 0.02 * float(np.sum(q**2))

  res = minimize(cost, np.zeros(3), bounds=list(zip(lo, hi)), method="L-BFGS-B")
  data.qpos[adr] = res.x
  mujoco.mj_forward(model, data)
  return {n: float(v) for n, v in zip(RIGHT_WRIST, res.x)}


def _channel_pose_in_reset(spec: mujoco.MjSpec, slope_deg: float):
  """World position of the ascender channel centre in the reset pose (wrist solved)."""
  model = spec.compile()
  data = mujoco.MjData(model)
  init = get_robot_cfg(slope_deg).init_state
  _set_pose(model, data, init)
  wb = model.body(WRIST_BODY).id
  grip_link, axis_link = ascender_channel(model)
  rot = data.xmat[wb].reshape(3, 3)
  grip_w = data.xpos[wb] + rot @ grip_link
  axis_w = rot @ axis_link
  return grip_link, grip_w, axis_w, data.xpos[wb].copy(), data.xquat[wb].copy()


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

  grip_link, wrist0, axis_w, wrist_pos_w, wrist_quat_w = _channel_pose_in_reset(spec, slope_deg)
  assert axis_w[0] > 0.99, f"ascender channel not aligned with the rope: {axis_w}"
  _, axis_link = ascender_channel(spec.compile())
  # Anchor = the ascender's rope channel (not the wrist origin) + a cylinder
  # inside the channel so the alignment is visible in the viewer.
  wrist = spec.body(WRIST_BODY)
  wrist.add_site(name="ascender_anchor", pos=grip_link.tolist(), group=5)
  chan = wrist.add_geom(
    name="ascender_channel",
    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
    size=[0.006, 0.055, 0],
    rgba=[0.1, 1.0, 0.2, 0.6],
    contype=0,
    conaffinity=0,
    mass=0.0,
    group=2,
  )
  chan.fromto = np.concatenate([grip_link - 0.055 * axis_link, grip_link + 0.055 * axis_link]).tolist()
  wb = spec.worldbody

  # Visual rope: a static cylinder along +x through the ascender channel at reset.
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
    frictionloss=3.0,  # cam drag on the rope, Petzl Basic ~2-5 N
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
  carrier.add_joint(name=SPIN_JOINT, type=mujoco.mjtJoint.mjJNT_HINGE, axis=[1, 0, 0], damping=0.05)
  # Weld the tool to the carriage: the ascender can only slide along the rope
  # and spin around it (a real ascender's channel), never tilt off it.
  # relpose = wrist pose in the carriage frame at reset (carriage frame = world
  # frame translated to the channel centre, since the rail is world +x).
  eq = spec.add_equality(
    type=mujoco.mjtEq.mjEQ_WELD,
    name="ascender_grip",
    objtype=mujoco.mjtObj.mjOBJ_BODY,
    name1=CARRIER_BODY,
    name2=WRIST_BODY,
  )
  rel = wrist_pos_w - wrist0
  eq.data[:3] = [0.0, 0.0, 0.0]
  eq.data[3:6] = rel.tolist()
  eq.data[6:10] = wrist_quat_w.tolist()
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


_WRIST_CACHE: dict[float, dict] = {}


def _wrist_pose(slope_deg: float, joint_pos: dict, rot) -> dict:
  if slope_deg not in _WRIST_CACHE:
    spec = mujoco.MjSpec.from_file(str(G1_ASCENDER_XML))
    for act in list(spec.actuators):
      spec.delete(act)
    for key in list(spec.keys):
      spec.delete(key)
    model = spec.compile()
    data = mujoco.MjData(model)
    _set_pose(model, data, EntityCfg.InitialStateCfg(pos=(0, 0, 0.8), rot=rot, joint_pos=joint_pos))
    _WRIST_CACHE[slope_deg] = _solve_wrist(model, data, joint_pos)
  return _WRIST_CACHE[slope_deg]


def get_robot_cfg(slope_deg: float) -> EntityCfg:
  joint_pos = dict(BASE_JOINT_POS)
  # Facing uphill the ankles dorsiflex by the slope angle (soft limit -0.87 rad).
  joint_pos[".*_ankle_pitch_joint"] = max(-0.85, -0.363 - math.radians(slope_deg))
  joint_pos[SLIDE_JOINT] = 0.0
  joint_pos[SPIN_JOINT] = 0.0
  # Wrist angles that put the ascender channel parallel to the rope.
  joint_pos.update(_wrist_pose(slope_deg, dict(joint_pos), slope_quat(slope_deg)))
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


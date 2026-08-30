"""Rope + ascender rail for MuJoCo — plain `mujoco`, no RL dependency.

Adds to a robot MjSpec (g1_unitree_ascender.xml):

  world
  ├── rope            visual cylinder along +x through the ascender channel (no contact)
  └── rope_carriage   ONE slide joint `rope_slide` along +x (a prismatic joint)
        weld  rope_carriage <-> right_wrist_yaw_link, so the rope always runs
        through the ascender's channel: the tool slides along the rope and
        nothing else. `ratchet()` is the cam: the slide joint's LOWER LIMIT is
        the highest point reached, so it can never go back down (solved with
        the weld by the constraint solver — never overwrite qpos, that fights
        the solver and the weld drifts by centimetres).

The channel = the tool-frame Z axis through the collision-mesh centre, mapped
through the mount pose (assets/ascender/MOUNT.md: the cam head is centred on
the wrist joint). `solve_wrist()` finds wrist angles that put the channel
parallel to the rope for the reset pose.

Used by rl/task/robot.py (mjlab training) and usable by app/harness.
"""

from __future__ import annotations

import re
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
ASCENDER_MESH = HERE / "meshes/ascender_collision.obj"

ROPE_BODY = "rope"
CARRIER_BODY = "rope_carriage"
SLIDE_JOINT = "rope_slide"  # no "_joint" suffix so ".*_joint" regexes skip it
WRIST_BODY = "right_wrist_yaw_link"
RIGHT_WRIST = ("right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint")

ROPE_LENGTH = 30.0  # m uphill of the start
ROPE_TAIL = 1.0  # m downhill of the start
ROPE_RADIUS = 0.0055  # 11 mm static rope (Petzl Basic: 8-13 mm)
CARRIER_MASS = 0.05
CAM_FRICTION_N = 3.0  # push needed to slide the cam up the rope
# The rope channel (Petzl: rope groove) in the tool frame, set from renders
# (front view: rope inside the U of the plate; side view: tool parallel to the
# rope; 3/4 view: rope clear of the forearm), 2026-08-30. The Ø16 mm bore at
# x=-21 mm found on the mesh is the carabiner hole, not the channel.
CHANNEL_CENTRE_TOOL = (0.015, -0.0012, 0.055)  # m, tool frame (z = mid-tool)
CHANNEL_PITCH_DEG = 5.0  # set by eye from side renders with the channel at x=+15 mm: tool parallel to the rope (2026-08-30)
ROPE_HEIGHT = 0.60  # m above the ground at reset (arm pose is solved for it)
RIGHT_ARM_IK = ("right_shoulder_pitch_joint", "right_elbow_joint") + RIGHT_WRIST


def ascender_channel(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
  """(channel centre, channel axis) in the wrist-link frame."""
  wb = model.body(WRIST_BODY).id
  gid = next(
    i for i in range(model.ngeom)
    if model.geom_bodyid[i] == wb
    and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
    and "ascender_collision" in model.mesh(model.geom_dataid[i]).name
  )
  rot = np.zeros(9)
  mujoco.mju_quat2Mat(rot, model.geom_quat[gid])
  rot = rot.reshape(3, 3)
  a = np.radians(CHANNEL_PITCH_DEG)
  axis_tool = np.array([np.sin(a), 0.0, np.cos(a)])  # tool Z pitched about tool y
  return model.geom_pos[gid] + rot @ np.array(CHANNEL_CENTRE_TOOL), rot @ axis_tool


def set_pose(model, data, root_pos, root_quat, joint_pos: dict) -> None:
  """Write a pose given as regex->angle dict (mjlab style) and run forward kinematics."""
  data.qpos[:3] = root_pos
  data.qpos[3:7] = root_quat
  for j in range(model.njnt):
    if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
      continue
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
    for pat, val in joint_pos.items():
      if re.fullmatch(pat, name):
        data.qpos[model.jnt_qposadr[j]] = val
  mujoco.mj_forward(model, data)


def solve_wrist(model, data, rope_height: float = ROPE_HEIGHT) -> dict:
  """Right shoulder-pitch/elbow/wrist angles that put the channel parallel to +x
  at `rope_height` above the ground (starting from the current pose)."""
  from scipy.optimize import minimize

  wb = model.body(WRIST_BODY).id
  grip_link, axis_link = ascender_channel(model)
  adr = [model.jnt_qposadr[model.joint(n).id] for n in RIGHT_ARM_IK]
  bounds = [tuple(model.jnt_range[model.joint(n).id] * 0.9) for n in RIGHT_ARM_IK]
  q0 = data.qpos[adr].copy()

  def cost(q):
    data.qpos[adr] = q
    mujoco.mj_forward(model, data)
    rot = data.xmat[wb].reshape(3, 3)
    axis_w = rot @ axis_link
    z = (data.xpos[wb] + rot @ grip_link)[2]
    return 5.0 * (1.0 - axis_w[0]) + 20.0 * (z - rope_height) ** 2 + 0.02 * float(np.sum((q - q0) ** 2))

  res = minimize(cost, q0, bounds=bounds, method="L-BFGS-B")
  data.qpos[adr] = res.x
  mujoco.mj_forward(model, data)
  return {n: float(v) for n, v in zip(RIGHT_ARM_IK, res.x)}


def add_rope_rail(
  spec: mujoco.MjSpec, root_pos, root_quat, joint_pos: dict, rope_height: float = ROPE_HEIGHT
) -> dict:
  """Add rope + carriage + weld to `spec`, sized for the given reset pose.

  `joint_pos` may omit the wrist: it is solved here so the channel is parallel
  to the rope. Returns the full joint_pos dict used (wrist + rail joints at 0).
  """
  model = spec.compile()
  data = mujoco.MjData(model)
  set_pose(model, data, root_pos, root_quat, joint_pos)
  wrist_q = solve_wrist(model, data, rope_height)
  wb = model.body(WRIST_BODY).id
  grip_link, axis_link = ascender_channel(model)
  rot = data.xmat[wb].reshape(3, 3)
  grip_w = data.xpos[wb] + rot @ grip_link
  axis_w = rot @ axis_link
  assert axis_w[0] > 0.98, f"ascender channel not aligned with the rope: {axis_w}"
  print(f"[rope_rail] channel at z={grip_w[2]:.3f} m (target {rope_height}), axis·x={axis_w[0]:.3f}")

  wrist = spec.body(WRIST_BODY)
  wrist.add_site(name="ascender_anchor", pos=grip_link.tolist(), group=5)
  chan = wrist.add_geom(  # the "cylinder inside the hole": shows the channel in the viewer
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

  wb_spec = spec.worldbody
  rope = wb_spec.add_body(name=ROPE_BODY, pos=grip_w)
  rg = rope.add_geom(
    name="rope_geom",
    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
    size=[ROPE_RADIUS, 1.0, 0.0],
    rgba=[0.9, 0.3, 0.1, 1.0],
    # Collides with the robot's body (legs, torso...) so it cannot walk through
    # the rope — EXCEPT the wrist/tool (excluded below): rope-in-channel contact
    # would fight the weld and jitter. Note: a rigid cylinder, not a sagging rope.
    contype=1,
    conaffinity=1,
    condim=3,
    friction=[0.2, 0.005, 0.0001],
    group=2,
  )
  rg.fromto = [-ROPE_TAIL, 0, 0, ROPE_LENGTH, 0, 0]
  spec.add_exclude(name="rope_vs_tool", bodyname1=ROPE_BODY, bodyname2=WRIST_BODY)

  carrier = wb_spec.add_body(name=CARRIER_BODY, pos=grip_w)
  sj = carrier.add_joint(
    name=SLIDE_JOINT,
    type=mujoco.mjtJoint.mjJNT_SLIDE,
    axis=[1, 0, 0],
    range=[-ROPE_LENGTH, ROPE_LENGTH],  # lower bound is moved up by ratchet(); symmetric so soft limits include 0
    damping=2.0,
    frictionloss=CAM_FRICTION_N,
  )
  sj.solref_limit = [0.01, 1.0]  # stiff limit: the cam does not sag under load
  sj.solimp_limit = [0.99, 0.999, 0.001, 0.5, 2.0]
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
  eq = spec.add_equality(
    type=mujoco.mjtEq.mjEQ_WELD,
    name="ascender_grip",
    objtype=mujoco.mjtObj.mjOBJ_BODY,
    name1=CARRIER_BODY,
    name2=WRIST_BODY,
  )
  # relpose = wrist pose in the carriage frame (carriage frame = world, at the channel).
  eq.data[:3] = [0.0, 0.0, 0.0]
  eq.data[3:6] = (data.xpos[wb] - grip_w).tolist()
  eq.data[6:10] = data.xquat[wb].tolist()
  eq.solref = [0.004, 1.0]
  eq.solimp = [0.99, 0.999, 0.001, 0.5, 2.0]

  # Specific right-arm keys must precede wildcard keys (e.g. .*_elbow_joint)
  # so resolve_expr's first-match picks the solved IK value, not the base 0.6.
  out = {**wrist_q, **joint_pos, SLIDE_JOINT: 0.0}
  return out


def ratchet(model: mujoco.MjModel, data: mujoco.MjData, prefix: str = "") -> None:
  """The cam: call BEFORE every mj_step. Lower limit of the slide = highest point reached."""
  jid = model.joint(prefix + SLIDE_JOINT).id
  model.jnt_range[jid, 0] = max(model.jnt_range[jid, 0], data.qpos[model.jnt_qposadr[jid]])


def ratchet_reset(model: mujoco.MjModel, data: mujoco.MjData, prefix: str = "") -> None:
  """Call after you move the robot (reset): the cam releases and re-engages where it is now."""
  jid = model.joint(prefix + SLIDE_JOINT).id
  model.jnt_range[jid, 0] = data.qpos[model.jnt_qposadr[jid]]


def rope_state(model, data, prefix: str = "", slide_joint: str = SLIDE_JOINT) -> dict:
  """Telemetry with the real end-effector's keys (assets/ascender/ELECTRONICS.md)."""
  jid = model.joint(prefix + slide_joint).id
  dof = int(model.jnt_dofadr[jid])
  tension = float(abs(data.qfrc_constraint[dof]))
  return {
    "rope_progress_m": float(data.qpos[model.jnt_qposadr[jid]]),
    "tension_N": tension,
    "engaged": bool(tension > 5.0),
  }

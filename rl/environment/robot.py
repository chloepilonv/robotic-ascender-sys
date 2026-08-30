"""Robot profiles for the climb scene, and the adapter that reconciles them.

TWO ROBOTS
  playground  `.reference/` -- mujoco_playground's G1, the model the mels
              walking policy was trained in. Physics reference.
  himalaya    `assets/robots/mujoco/g1_unitree_ascender.xml` -- this project's
              G1: jacket, boots, logos, D435i/Mid-360 sites, boot friction 0.8,
              and the ascender tool on the right wrist. Built by
              `assets/robots/mujoco/build.py` from the same menagerie source.

They share the 29-joint actuator order exactly, and every body mass matches
except `right_wrist_yaw_link` (+0.1 kg, the ascender). Everything else the
climb scene needs is missing from the himalaya model, because it ships as a
bare robot with no scene:

  * no `floor` geom          -- deliberate; terrain lives in the scene
  * no `right_palm` site     -- the grip anchor for the ascender equality
  * no `knees_bent` keyframe -- it has `stand`, a different pose
  * only 5 raw IMU sensors   -- the policy needs pelvis linvel and gyro
  * unnamed foot geoms       -- the icyness axis needs to find them

`adapt()` adds each of those if absent, so the same scene builder works on
either robot.

THE PART THAT IS NOT COSMETIC
The himalaya model uses stock menagerie dynamics; playground retuned them for
RL. The mels policy learned against playground's closed-loop response, so
running it on stock gains is a different plant:

    actuator kp     75 / 20 / 2 per joint   vs  500 uniform
    actuator kv     0                       vs  -17 to -43 (dampratio 1)
    dof damping     0.2 / 1.0 / 2.0         vs  0
    dof armature    0.0036 - 0.0251         vs  0.01 uniform
    dof frictionloss 0.1                    vs  0.3

`policy_compat=True` (the default) writes playground's values back over the
himalaya model so the warm start remains valid. Set it False to simulate the
robot exactly as this project specifies it -- and expect the walking policy to
behave differently, because it is then driving a plant it never saw.

Also differing, and NOT patched: `right_hip_roll_joint` range is
[-2.9671, 0.5236] here (stock menagerie, correctly mirrored from the left leg)
against playground's [-0.5236, 2.9671] (unmirrored, matching the left). Both
contain the knees_bent value of 0 and the policy's working range, so this is
recorded rather than "fixed" -- silently rewriting a joint limit to match a
quirk of the training model would be the wrong repair.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import mujoco
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

PLAYGROUND_SCENE = os.path.join(
    REPO_ROOT, ".reference", "scene_mjx_feetonly_flat_terrain.xml"
)
PLAYGROUND_VISUAL_SCENE = os.path.join(
    REPO_ROOT, ".reference", "full", "scene_mjx_feetonly_flat_terrain.xml"
)
HIMALAYA_ROBOT = os.path.join(
    REPO_ROOT, "assets", "robots", "mujoco", "g1_unitree_ascender.xml"
)
HIMALAYA_ROBOT_BARE = os.path.join(
    REPO_ROOT, "assets", "robots", "mujoco", "g1_unitree.xml"
)

N_JOINTS = 29

# --- playground training-time dynamics, transcribed from the pinned v0.2.0
# --- model. These are the plant the mels policy was optimised against.
TRAIN_KP = np.array([
    75, 75, 75, 75, 20, 2,   75, 75, 75, 75, 20, 2,
    75, 75, 75,
    75, 75, 75, 75, 2, 2, 2,   75, 75, 75, 75, 2, 2, 2,
], dtype=float)
TRAIN_DAMPING = np.array([
    2, 2, 2, 2, 1, 0.2,   2, 2, 2, 2, 1, 0.2,
    2, 2, 2,
    2, 2, 2, 2, 0.2, 0.2, 0.2,   2, 2, 2, 2, 0.2, 0.2, 0.2,
], dtype=float)
TRAIN_ARMATURE = np.array([
    0.0101775, 0.0251019, 0.0101775, 0.0251019, 0.00721945, 0.00721945,
    0.0101775, 0.0251019, 0.0101775, 0.0251019, 0.00721945, 0.00721945,
    0.0101775, 0.00721945, 0.00721945,
    0.00360972, 0.00360972, 0.00360972, 0.00360972, 0.00360972, 0.00425, 0.00425,
    0.00360972, 0.00360972, 0.00360972, 0.00360972, 0.00360972, 0.00425, 0.00425,
], dtype=float)
TRAIN_FRICTIONLOSS = np.full(N_JOINTS, 0.1)

# `knees_bent` from the playground scene: the pose the policy's action deltas
# are defined about. The himalaya model ships `stand`, which is a different pose.
KNEES_BENT_QPOS = np.array([
    0, 0, 0.755, 1, 0, 0, 0,
    -0.312, 0, 0, 0.669, -0.363, 0,
    -0.312, 0, 0, 0.669, -0.363, 0,
    0, 0, 0.073,
    0.2, 0.2, 0, 0.6, 0, 0, 0,
    0.2, -0.2, 0, 0.6, 0, 0, 0,
], dtype=float)
KNEES_BENT_CTRL = KNEES_BENT_QPOS[7:]

# Playground puts `right_palm` here on the wrist link. On the himalaya robot the
# ascender's rope channel sits at local [0.074, 0, 0.010], so the same anchor
# lands on the actual tool rather than on a bare wrist.
PALM_BODY = "right_wrist_yaw_link"
PALM_POS = (0.08, 0.0, 0.0)

# Playground's foot: one box per ankle_roll link. The himalaya robot instead
# uses four 5 mm spheres at the corners -- arguably the better foot model, but a
# different contact plant, and measurably the one thing that still stops the
# mels policy from standing once everything else is matched (it falls at ~3 s on
# flat ground with spheres, stands indefinitely with the box).
FOOT_BOX_POS = (0.04, 0.0, -0.029)
FOOT_BOX_SIZE = (0.09, 0.03, 0.008)


@dataclass(frozen=True)
class RobotProfile:
    name: str
    xml: str
    has_gear: bool


def resolve(name: str, visual: bool = False) -> RobotProfile:
    if name == "himalaya":
        return RobotProfile("himalaya", HIMALAYA_ROBOT, True)
    if name == "himalaya-bare":
        return RobotProfile("himalaya-bare", HIMALAYA_ROBOT_BARE, True)
    if name == "playground":
        xml = PLAYGROUND_VISUAL_SCENE if visual else PLAYGROUND_SCENE
        if visual and not os.path.exists(xml):
            raise SystemExit(
                "--visual needs the renderable playground G1:\n"
                "    python -m rl.tools.fetch_visual_assets"
            )
        return RobotProfile("playground", xml, False)
    raise ValueError(f"unknown robot {name!r}; try playground / himalaya")


def _site_names(spec):
    return {s.name for s in spec.sites}


def _sensor_names(spec):
    return {s.name for s in spec.sensors}


def adapt(spec: mujoco.MjSpec, policy_compat: bool = True) -> dict:
    """Give `spec` everything the climb scene and the walking policy need.

    Idempotent: each piece is added only if absent, so this is a no-op on the
    playground model and does the real work on the himalaya one. Returns a
    report of what it changed.
    """
    report = {"added": [], "retuned": []}

    # -- grip anchor ----------------------------------------------------
    if "right_palm" not in _site_names(spec):
        spec.body(PALM_BODY).add_site(name="right_palm", pos=list(PALM_POS),
                                      size=[0.01, 0, 0], group=5)
        report["added"].append("site right_palm")

    # -- sensors the 103-dim observation reads --------------------------
    have = _sensor_names(spec)
    if "local_linvel_pelvis" not in have:
        # Playground declares this as <velocimeter site="imu_in_pelvis"/>, i.e.
        # base linear velocity in the SITE frame. A framelinvel sensor with the
        # site as its own reference measures velocity relative to itself and
        # reads exactly zero -- which compiles, runs, and quietly feeds the
        # policy a dead three dims until it falls over.
        spec.add_sensor(
            name="local_linvel_pelvis",
            type=mujoco.mjtSensor.mjSENS_VELOCIMETER,
            objtype=mujoco.mjtObj.mjOBJ_SITE,
            objname="imu_in_pelvis",
        )
        report["added"].append("sensor local_linvel_pelvis")
    if "gyro_pelvis" not in have:
        spec.add_sensor(
            name="gyro_pelvis",
            type=mujoco.mjtSensor.mjSENS_GYRO,
            objtype=mujoco.mjtObj.mjOBJ_SITE,
            objname="imu_in_pelvis",
        )
        report["added"].append("sensor gyro_pelvis")

    # -- the pose the policy's actions are deltas about -----------------
    if "knees_bent" not in {k.name for k in spec.keys}:
        k = spec.add_key(name="knees_bent")
        k.qpos = KNEES_BENT_QPOS.copy()
        k.ctrl = KNEES_BENT_CTRL.copy()
        report["added"].append("keyframe knees_bent")

    # -- foot contact model --------------------------------------------
    if policy_compat and _swap_feet_for_boxes(spec):
        report["retuned"].append("foot contact: 4 spheres -> playground box")

    # -- training-time dynamics ----------------------------------------
    if policy_compat:
        for i, act in enumerate(spec.actuators[:N_JOINTS]):
            act.gainprm = [TRAIN_KP[i]] + [0.0] * 9
            act.biasprm = [0.0, -TRAIN_KP[i], 0.0] + [0.0] * 7
        joints = [j for j in spec.joints if j.type == mujoco.mjtJoint.mjJNT_HINGE]
        for i, j in enumerate(joints[:N_JOINTS]):
            # damping is a 3-vector on MjsJoint (ball/free joints use all
            # three); a hinge reads only the first.
            j.damping = [TRAIN_DAMPING[i], 0.0, 0.0]
            j.armature = float(TRAIN_ARMATURE[i])
            j.frictionloss = float(TRAIN_FRICTIONLOSS[i])
        report["retuned"].append("actuator kp/kv, dof damping/armature/frictionloss")

    return report


def _swap_feet_for_boxes(spec) -> bool:
    """Replace sphere-corner feet with playground's box. Returns True if it did.

    The spheres are retired by clearing contype/conaffinity rather than deleted,
    so the visual model is untouched and the change is easy to see in the
    exported XML.
    """
    changed = False
    for body in spec.bodies:
        if body.name not in ("left_ankle_roll_link", "right_ankle_roll_link"):
            continue
        spheres = [
            g for g in body.geoms
            if g.type == mujoco.mjtGeom.mjGEOM_SPHERE and g.contype != 0
        ]
        if not spheres:
            continue  # already box-footed (the playground model)
        friction = list(spheres[0].friction)
        for g in spheres:
            g.contype = 0
            g.conaffinity = 0
        side = body.name.split("_")[0]
        box = body.add_geom(
            name=f"{side}_foot",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=list(FOOT_BOX_POS),
            size=list(FOOT_BOX_SIZE),
            condim=3,
            priority=1,
            friction=friction,
            rgba=[0.2, 0.2, 0.2, 1.0],
            group=3,
        )
        changed = True
    return changed


def foot_contact_geoms(model: mujoco.MjModel) -> list[int]:
    """Geom ids that actually carry foot-ground contact, on either robot.

    Playground names two box geoms `left_foot`/`right_foot`. The himalaya model
    uses four unnamed priority-1 spheres per ankle_roll link -- and priority
    matters: a priority-1 geom's friction wins outright, so setting terrain
    friction alone leaves the boots at whatever they were built with.
    """
    out = []
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[i]) or ""
        if name in ("left_foot", "right_foot"):
            out.append(i)
        elif "ankle_roll" in body and model.geom_contype[i] != 0:
            out.append(i)
    return out

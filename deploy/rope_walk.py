"""G1 "hiker on a fixed rope" mimic — unitree_sdk2py.

    python deploy/rope_walk.py --iface eth0            # real robot
    python deploy/rope_walk.py --dry-run               # print the plan, no SDK

Sequence (strictly sequential, one phase at a time):
    0. arm_sdk takes over the RIGHT arm (weight ramp 0->1), left arm untouched.
    1. GRIP     : right hand placed on the rope (rope runs along +x, on the robot's
                  right side, ROPE_Z above ground). Hand starts slightly ahead of the hip.
    2. WALK     : LocoClient walks forward WALK_DIST (~3-4 steps). Every tick the arm is
                  re-solved from the body's odometry so the palm stays PINNED to the same
                  WORLD POINT on the rope; once the body is too far ahead (HAND_X_BACK) the
                  hand slides along the rope. It never leaves the rope.
    3. RE-GRIP  : hand lifts LIFT_Z off the rope, swings forward, lowers back on the rope.
    4. WALK     : same as 2.
    (steps 3-4 repeat CYCLES times)
    5. stop, keep hand on the rope, hand arm control back (weight 1->0) only on --release.

Balance/upright is owned by Unitree's onboard loco controller (LocoClient). We never
command legs/waist. If IMU roll/pitch exceeds TILT_ABORT_RAD we StopMove + drop the arm
weight immediately.

Arm geometry is a planar 2-link IK in the sagittal plane (shoulder pitch + elbow) with a
fixed shoulder roll to reach out to the rope. Sign conventions (SIGN_*) are the ones we
expect from the G1 29-DoF URDF; verify on the robot with --dry-run first, then at
--speed 0.5 with someone holding the e-stop.
"""
from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------- geometry (metres)
ROPE_Z = 0.60          # rope height above ground (user spec); override with --rope-z
ROPE_Y = -0.25         # rope lateral offset from body centre (right side = -y); beside the hip
# measured in MuJoCo from assets/robots/mujoco/g1_unitree.xml (pelvis z=0.79, standing):
SHOULDER_Z = 1.05      # right_shoulder_pitch_link z (1.085 in 'stand', ~1.05 in the walking 'knees_bent' pose)
SHOULDER_Y = -0.100    # right_shoulder_pitch_link y
UPPER_ARM = 0.19       # shoulder -> elbow
FOREARM = 0.24         # elbow -> palm (wrist_yaw link is 0.20, +0.04 to the palm)
HAND_OFFSET = 0.04     # palm ahead of right_wrist_yaw_link along its x (sim metric only)
PELVIS_Z0 = 0.755      # pelvis height in the walking stance (knees_bent); shoulder = pelvis + SHOULDER_OFF
SHOULDER_OFF = (0.0, SHOULDER_Y, SHOULDER_Z - PELVIS_Z0)

HAND_X_FRONT = 0.15    # hand ahead of shoulder right after a grip
HAND_X_BACK = -0.15    # furthest the hand can trail behind the shoulder while on the rope;
                       # beyond that the hand SLIDES along the rope (still on it)
SLIDE_Z = 0.0          # extra push into the rope while sliding (0: just rest on it)
REACH_MAX = (UPPER_ARM + FOREARM) * 0.95   # targets beyond this are pulled back onto the reach sphere
MAX_DQ_TICK = 0.03     # rad per control tick (1.5 rad/s): no arm thrashing whatever the IK says
LIFT_Z = 0.08          # how high the hand lifts off the rope when re-gripping
WALK_DIST = 0.75       # per walk phase, ~3 steps of 0.25 m
CYCLES = 2             # number of (re-grip + walk) after the first grip+walk

# ---------------------------------------------------------------- joint map (G1 29-DoF)
R_SHOULDER_PITCH, R_SHOULDER_ROLL, R_SHOULDER_YAW = 22, 23, 24
R_ELBOW, R_WRIST_ROLL, R_WRIST_PITCH, R_WRIST_YAW = 25, 26, 27, 28
RIGHT_ARM = [R_SHOULDER_PITCH, R_SHOULDER_ROLL, R_SHOULDER_YAW,
             R_ELBOW, R_WRIST_ROLL, R_WRIST_PITCH, R_WRIST_YAW]
ARM_SDK_WEIGHT_IDX = 29            # "kNotUsedJoint": q of motor 29 = arm_sdk weight [0,1]

# Signs verified in MuJoCo on the G1 MJCF (same URDF as the robot):
#   shoulder pitch <0 raises the arm forward; right shoulder roll <0 abducts;
#   elbow q=0 is ALREADY bent 90 deg (forearm horizontal), q=+pi/2 ~ straight arm.
SIGN_SHOULDER_PITCH_FWD = -1.0
SIGN_R_SHOULDER_ROLL_OUT = -1.0
ELBOW_Q_STRAIGHT = math.pi / 2           # q = ELBOW_Q_STRAIGHT - flex

KP_ARM, KD_ARM = 60.0, 1.5          # gentle: the rope reacts on the hand
CTRL_HZ = 50
TILT_ABORT_RAD = math.radians(25)


@dataclass
class ArmPose:
    q: dict  # joint index -> target rad


def ik_right_arm(hand_dx: float, hand_dz: float) -> ArmPose:
    """Planar 2-link IK. hand_dx: forward of shoulder (+x), hand_dz: below shoulder (negative)."""
    r = math.hypot(hand_dx, hand_dz)
    r = min(r, (UPPER_ARM + FOREARM) * 0.98)          # never fully straight (clamps if too far)
    # elbow flexion from law of cosines (0 = straight)
    c = (UPPER_ARM**2 + FOREARM**2 - r**2) / (2 * UPPER_ARM * FOREARM)
    flex = math.pi - math.acos(max(-1.0, min(1.0, c)))
    # shoulder: angle of the target below/forward + inner angle of the triangle
    phi = math.atan2(hand_dx, -hand_dz)                # 0 = straight down, + = forward
    c2 = (UPPER_ARM**2 + r**2 - FOREARM**2) / (2 * UPPER_ARM * r)
    inner = math.acos(max(-1.0, min(1.0, c2)))
    shoulder = phi + inner                             # elbow bends so the hand is under it
    # lateral reach to the rope
    roll = math.atan2(abs(ROPE_Y - SHOULDER_Y), max(abs(hand_dz), 0.05))
    return ArmPose({
        R_SHOULDER_PITCH: SIGN_SHOULDER_PITCH_FWD * shoulder,
        R_SHOULDER_ROLL: SIGN_R_SHOULDER_ROLL_OUT * roll,
        R_SHOULDER_YAW: 0.0,
        R_ELBOW: ELBOW_Q_STRAIGHT - flex,
        R_WRIST_ROLL: 0.0,
        R_WRIST_PITCH: 0.0,
        R_WRIST_YAW: 0.0,
    })


class MjIK:
    """Numerical IK on the G1 MJCF (same URDF as the robot): palm site -> 4 arm joints.
    Damped least squares on shoulder pitch/roll + elbow (3 joints for a 3D point -> a single
    branch; adding shoulder yaw let the solver flip to a wild branch mid-walk). Wrist and
    shoulder yaw fixed at 0. Used by every backend when mujoco is importable; analytic
    ik_right_arm() is the fallback."""
    JOINTS = [R_SHOULDER_PITCH, R_SHOULDER_ROLL, R_ELBOW]

    def __init__(self):
        import mujoco, numpy as np
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from rl.environment import robot as robot_mod
        self.mj, self.np = mujoco, np
        spec = mujoco.MjSpec.from_file(robot_mod.HIMALAYA_ROBOT_BARE); robot_mod.adapt(spec)
        self.m = spec.compile(); self.d = mujoco.MjData(self.m)
        mujoco.mj_resetDataKeyframe(self.m, self.d, self.m.key("knees_bent").id)
        mujoco.mj_forward(self.m, self.d)
        self.palm = self.m.site("right_palm").id
        self.shoulder = self.d.xpos[self.m.body("right_shoulder_pitch_link").id].copy()
        self.dofs = [self.m.jnt_dofadr[1 + j] for j in self.JOINTS]      # joint j is MJCF joint 1+j
        self.qadr = [self.m.jnt_qposadr[1 + j] for j in self.JOINTS]
        self.q0 = self.d.qpos.copy()
        self.cache = {}

    def solve(self, hand_dx: float, hand_dy: float, hand_dz: float) -> ArmPose:
        key = (round(hand_dx * 200), round(hand_dy * 200), round(hand_dz * 200))     # 5 mm grid
        if key in self.cache: return self.cache[key]
        mj, np = self.mj, self.np
        target = self.shoulder + np.array([hand_dx, hand_dy, hand_dz])
        jacp = np.zeros((3, self.m.nv))                                    # warm start: previous solution
        for _ in range(100):
            mj.mj_forward(self.m, self.d)
            err = target - self.d.site_xpos[self.palm]
            if np.linalg.norm(err) < 1e-3: break
            mj.mj_jacSite(self.m, self.d, jacp, None, self.palm)
            J = jacp[:, self.dofs]
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(3), err)
            for a, delta in zip(self.qadr, dq):
                self.d.qpos[a] += 0.5 * float(delta)
            for a, j in zip(self.qadr, self.JOINTS):                       # respect joint limits
                lo, hi = self.m.jnt_range[1 + j]
                self.d.qpos[a] = min(max(self.d.qpos[a], lo), hi)
        pose = ArmPose({j: float(self.d.qpos[a]) for j, a in zip(self.JOINTS, self.qadr)})
        for j in (R_SHOULDER_YAW, R_WRIST_ROLL, R_WRIST_PITCH, R_WRIST_YAW): pose.q[j] = 0.0
        self.cache[key] = pose
        return pose


_IK = None

def hand_at_world(pt, base) -> tuple[ArmPose, float]:
    """Arm pose that puts the palm at world point `pt`, given the base pose (x, y, z, yaw).
    Returns (pose, dx) where dx is the hand's forward offset from the shoulder, clamped to the
    reach window [HAND_X_BACK, HAND_X_FRONT] -> outside it the hand slides along the rope."""
    bx, by, bz, yaw = base
    c, s_ = math.cos(yaw), math.sin(yaw)
    ox, oy, oz = SHOULDER_OFF
    shoulder = (bx + c * ox - s_ * oy, by + s_ * ox + c * oy, bz + oz)
    wx, wy, wz = pt[0] - shoulder[0], pt[1] - shoulder[1], pt[2] - shoulder[2]
    dx, dy = c * wx + s_ * wy, -s_ * wx + c * wy                 # into the body frame
    dx_c = min(max(dx, HAND_X_BACK), HAND_X_FRONT)
    dz = wz + (SLIDE_Z if dx_c != dx else 0.0)
    r = math.sqrt(dx_c * dx_c + dy * dy + dz * dz)
    if r > REACH_MAX:                                              # unreachable: closest reachable point
        k = REACH_MAX / r; dx_c, dy, dz = dx_c * k, dy * k, dz * k
    return _ik(dx_c, dy, dz), dx


def _ik(dx, dy, dz) -> ArmPose:
    global _IK
    if _IK is None:
        try: _IK = MjIK()
        except Exception as e:                                          # no mujoco on the robot
            print(f"note: analytic IK fallback ({e.__class__.__name__})"); _IK = False
    return _IK.solve(dx, dy, dz) if _IK else ik_right_arm(dx, dz)


def hand_on_rope(dx: float, lift: float = 0.0) -> ArmPose:
    return _ik(dx, ROPE_Y - SHOULDER_Y, (ROPE_Z + lift) - SHOULDER_Z)


def rate_limit(prev: ArmPose, new: ArmPose, max_dq: float = MAX_DQ_TICK) -> ArmPose:
    return ArmPose({j: prev.q[j] + max(-max_dq, min(max_dq, new.q[j] - prev.q[j])) for j in new.q})


def lerp_pose(a: ArmPose, b: ArmPose, s: float) -> ArmPose:
    s = max(0.0, min(1.0, s))
    return ArmPose({j: a.q[j] + (b.q[j] - a.q[j]) * s for j in a.q})


# ---------------------------------------------------------------- robot backends
class DryRun:
    """Prints what would be sent. Lets you check the sequence and IK without a robot."""
    def __init__(self):
        self.t0, self.x, self.vx = time.time(), 0.0, 0.0

    def _log(self, msg): print(f"[{time.time() - self.t0:6.2f}s] {msg}")
    def start(self): self._log("loco: Start (balance stand)")
    def set_arm_weight(self, w): self._log(f"arm_sdk weight={w:.2f}")
    def send_arm(self, pose: ArmPose):
        sp, el, ro = pose.q[R_SHOULDER_PITCH], pose.q[R_ELBOW], pose.q[R_SHOULDER_ROLL]
        self._log(f"arm  sh_pitch={sp:+.2f} sh_roll={ro:+.2f} elbow={el:+.2f}")
    def move(self, vx): self.vx = vx; self._log(f"loco: Move vx={vx:.2f}")
    def stop_move(self): self.vx = 0.0; self._log("loco: StopMove")
    def tilt_ok(self): return True
    def current_arm(self): return ArmPose({j: 0.0 for j in RIGHT_ARM})
    def base_pose(self): return (self.x, 0.0, PELVIS_Z0, 0.0)
    def sleep(self, s): self.x += self.vx * s; time.sleep(min(s, 0.02))   # fast-forward


class G1:
    def __init__(self, iface: str):
        from unitree_sdk2py.core.channel import (ChannelFactoryInitialize, ChannelPublisher,
                                                 ChannelSubscriber)
        from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC

        ChannelFactoryInitialize(0, iface)
        self.loco = LocoClient(); self.loco.SetTimeout(10.0); self.loco.Init()
        self.pub = ChannelPublisher("rt/arm_sdk", LowCmd_); self.pub.Init()
        self.cmd = unitree_hg_msg_dds__LowCmd_()
        self.crc = CRC()
        self.weight = 0.0
        self.lock = threading.Lock(); self.low = None
        ChannelSubscriber("rt/lowstate", LowState_).Init(self._on_low, 1)
        self.odom, self.vx, self.dr_x, self.dr_t = None, 0.0, 0.0, time.time()
        try:                                              # body odometry (position + yaw) if published
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
            ChannelSubscriber("rt/odommodestate", SportModeState_).Init(self._on_odom, 1)
        except Exception as e:
            print(f"note: no odometry topic ({e.__class__.__name__}); dead-reckoning from Move()")
        t0 = time.time()
        while self.low is None and time.time() - t0 < 3.0:
            time.sleep(0.05)
        if self.low is None:
            sys.exit("no rt/lowstate — check --iface and that the robot is on")

    def _on_low(self, m):
        with self.lock: self.low = m
    def _on_odom(self, m):
        with self.lock: self.odom = m

    def base_pose(self):
        with self.lock: od, low = self.odom, self.low
        if od is not None:
            return (od.position[0], od.position[1], PELVIS_Z0, od.imu_state.rpy[2])
        now = time.time(); self.dr_x += self.vx * (now - self.dr_t); self.dr_t = now
        return (self.dr_x, 0.0, PELVIS_Z0, low.imu_state.rpy[2] if low else 0.0)

    def start(self):
        # robot must already be standing (remote: L2+A -> R2+B ...). Start() = walk-ready mode.
        # Start() is SetFsmId(200) on the installed SDK and RETURNS A CODE. Discarding it
        # makes a refused transition look identical to a successful one: on the gantry robot
        # this printed every phase cleanly while nothing moved.
        code = self.loco.SetFsmId(200)
        if code != 0:
            sys.exit(f"loco SetFsmId(200) refused (code {code}); the robot is not in a state "
                     f"that accepts motion. Put it in a standing stance first "
                     f"(physical remote: L2+A then R2+B).")
        time.sleep(3.0)

    def set_arm_weight(self, w): self.weight = max(0.0, min(1.0, w))

    def send_arm(self, pose: ArmPose):
        self.cmd.motor_cmd[ARM_SDK_WEIGHT_IDX].q = self.weight
        for j, q in pose.q.items():
            m = self.cmd.motor_cmd[j]
            m.mode, m.q, m.dq, m.tau, m.kp, m.kd = 1, float(q), 0.0, 0.0, KP_ARM, KD_ARM
        self.cmd.crc = self.crc.Crc(self.cmd)
        self.pub.Write(self.cmd)

    def move(self, vx):
        # LocoClient.Move() is SetVelocity(..., duration=1) and returns None, so a single
        # call expires after 1 s while the caller tracks for several seconds. Re-issue the
        # SHORT command instead of using continous_move=True (864000 s): a 1 s duration
        # means the robot stops by itself within a second if this process dies.
        self.base_pose(); self.vx = vx
        code = self.loco.SetVelocity(vx, 0.0, 0.0, 1.0)
        if code != 0:
            raise RuntimeError(f"SetVelocity refused, code {code}")
    def stop_move(self): self.base_pose(); self.vx = 0.0; self.loco.StopMove()

    def tilt_ok(self):
        with self.lock: low = self.low
        r, p, _ = low.imu_state.rpy
        return abs(r) < TILT_ABORT_RAD and abs(p) < TILT_ABORT_RAD

    def current_arm(self):
        with self.lock: low = self.low
        return ArmPose({j: low.motor_state[j].q for j in RIGHT_ARM})

    def sleep(self, s): time.sleep(s)



class Sim:
    """MuJoCo stand-in with a REAL gait. Same interface as G1.

    Legs/waist/left arm are driven by the mels G1 joystick walking policy
    (rl/environment/walk_policy.py, pure NumPy) standing in for Unitree's onboard loco
    controller. The right arm gets weight*target + (1-weight)*policy_target, mirroring what
    arm_sdk's weight does on the robot. The G1 MJCF is patched by rl.environment.robot.adapt()
    (sensors, knees_bent keyframe, RL-tuned gains) so the policy sees the plant it trained on."""
    DT = 0.002
    HEADING_KP = 2.0                      # yaw-rate command = -KP * yaw, keeps the walk straight
    POS_KP = 0.6                          # the policy never stands still; hold a goal x,y like StopMove does
    POS_KP_Y = 2.0                        # lateral: stay on the rope line (the gait veers left otherwise)
    LEVEL_RAD = math.radians(6)           # freeze the stand only when roughly level
    # SIM HACK, documented: the mels policy veers ~-30 deg per 3 s and barely answers yaw
    # commands (even on its own training model). Unitree's controller walks straight. A
    # virtual yaw spring + lateral spring on the pelvis stands in for that heading control.
    # It only applies a torque about z and a force along y: the legs still do all the walking.
    GUIDE_K_YAW, GUIDE_D_YAW = 150.0, 15.0        # N m / rad, N m s / rad
    GUIDE_K_Y, GUIDE_D_Y = 300.0, 60.0            # N / m, N s / m
    # While STANDING the real controller also balances; the frozen pose + ankle strategy is
    # marginal, so add weak pelvis springs on x / roll / pitch (never on z: legs carry the weight).
    STAND_K_X, STAND_D_X = 200.0, 40.0
    STAND_K_ROT, STAND_D_ROT = 200.0, 20.0

    def __init__(self, viewer: bool, video: str | None):
        import mujoco, mujoco.viewer, numpy as np
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from rl.environment import robot as robot_mod
        from rl.environment.walk_policy import WalkController
        self.mj, self.np = mujoco, np

        spec = mujoco.MjSpec.from_file(robot_mod.HIMALAYA_ROBOT_BARE)
        robot_mod.adapt(spec)
        spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05],
                                rgba=[0.35, 0.4, 0.45, 1])
        spec.worldbody.add_geom(name="rope", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.008, 0, 0],
                                fromto=[-1, ROPE_Y, ROPE_Z, 6, ROPE_Y, ROPE_Z], rgba=[0.9, 0.2, 0.1, 1],
                                condim=3, friction=[0.5, 0.005, 0.0001])       # solid; some drag when the ascender slides
        palm = spec.body("right_wrist_yaw_link")
        palm.add_geom(name="right_palm_pad", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.02, 0, 0],
                      pos=[HAND_OFFSET + 0.04, 0, 0], rgba=[0.1, 0.8, 0.2, 0.6], group=3)
        spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 720
        self.m = spec.compile(); self.m.opt.timestep = self.DT
        self.d = mujoco.MjData(self.m)
        mujoco.mj_resetDataKeyframe(self.m, self.d, self.m.key("knees_bent").id)
        self.walk = WalkController(self.m, command=(0.0, 0.0, 0.0))
        mujoco.mj_forward(self.m, self.d)

        self.weight, self.target = 0.0, None
        self.vx, self.goal = 0.0, self.d.qpos[:2].copy()
        self.key_ctrl = self.m.key("knees_bent").ctrl.copy()
        self.gain_rl, self.bias_rl = self.m.actuator_gainprm.copy(), self.m.actuator_biasprm.copy()
        self.mode, self.mode_t, self.walk_t = "stand", 0.0, 0.0
        self.stand_ctrl = self.key_ctrl.copy()
        self.feet = [self.m.site("left_foot").id, self.m.site("right_foot").id]
        self.com_err_prev = None
        self.pelvis_id = self.m.body("pelvis").id
        self.palm = self.m.site("right_palm").id
        self.t, self.phase, self.log = 0.0, "init", []
        self.viewer = mujoco.viewer.launch_passive(self.m, self.d) if viewer else None
        self.frames, self.renderer, self.video = [], None, video
        if video:
            self.renderer = mujoco.Renderer(self.m, 720, 1280)
            self.cam = mujoco.MjvCamera(); self.cam.azimuth, self.cam.elevation, self.cam.distance = 150, -15, 3.0

    def hand_xyz(self): return self.d.site_xpos[self.palm]

    def _yaw(self):
        w, x, y, z = self.d.qpos[3:7]
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def start(self): print("sim: standing (stiff knees_bent hold); walking policy used only while moving")
    def set_arm_weight(self, w): self.weight = max(0.0, min(1.0, w))
    def send_arm(self, pose: ArmPose): self.target = pose
    # -- stand / walk modes -------------------------------------------------------------
    # The walking policy cannot stand still (it marches in place and drifts), unlike the
    # real loco controller. So STAND = stiff hold of the knees_bent keyframe (kp 500, like
    # the stock MJCF), WALK = policy with the RL-tuned gains. Transitions blend over BLEND_S.
    STAND_KP, BLEND_S, SETTLE_S, SETTLE_V, FOOT_DOWN_Z, PREWALK_S, FADE_S = 500.0, 0.3, 0.5, 0.2, 0.042, 1.0, 1.5
    ANKLE_KP, ANKLE_KD, COM_X_REF = 3.0, 0.3, 0.03   # ankle strategy; hold the CoM 3 cm ahead of the feet centre
    L_ANKLE_PITCH, L_ANKLE_ROLL, R_ANKLE_PITCH, R_ANKLE_ROLL = 4, 5, 10, 11

    def _set_gains(self, a):
        """a=0: RL gains (policy), a=1: stiff stand gains."""
        kp = self.STAND_KP
        self.m.actuator_gainprm[:, 0] = (1 - a) * self.gain_rl[:, 0] + a * kp
        self.m.actuator_biasprm[:, 1] = (1 - a) * self.bias_rl[:, 1] + a * (-kp)
        self.m.actuator_biasprm[:, 2] = (1 - a) * self.bias_rl[:, 2] + a * (-math.sqrt(kp))

    def move(self, vx):
        self.vx = vx
        if self.mode not in ("walk", "unblend", "prewalk"):
            self.mode, self.mode_t = "prewalk", 0.0      # legs back to the keyframe first
    def stop_move(self):
        self.vx = 0.0
        if self.mode in ("walk", "unblend", "prewalk"):
            self.mode, self.mode_t = "settle", 0.0
            self.goal[:] = [self.d.qpos[0] + 0.1, 0.0]      # one small step ahead; y = the rope line

    def _leg_ctrl(self):
        """Fill d.ctrl for the whole body according to the mode; returns blend a (0 policy, 1 stand)."""
        if self.mode == "prewalk":                      # stiff, springs on: pose -> knees_bent keyframe
            a = min(1.0, self.mode_t / self.PREWALK_S)
            self.d.ctrl[:] = (1 - a) * self.stand_ctrl + a * self.key_ctrl
            if a >= 1.0:
                self.stand_ctrl = self.key_ctrl.copy(); self.walk.reset()
                self.mode, self.mode_t, self.walk_t = "unblend", 0.0, 0.0
            return 1.0
        if self.mode in ("walk", "unblend"):           # unblend: gains stiff -> RL while policy runs
            a = 0.0 if self.mode == "walk" else max(0.0, 1.0 - self.mode_t / self.BLEND_S)
            if self.mode == "unblend" and a <= 0.0: self.mode = "walk"
            self.goal[0] += self.vx * self.DT
            err = self.goal - self.d.qpos[:2]
            self.walk.command[0] = max(-1.0, min(1.0, self.vx + self.POS_KP * err[0]))
            self.walk.command[1] = max(-0.5, min(0.5, self.POS_KP * err[1]))
            self.walk.command[2] = max(-0.5, min(0.5, -self.HEADING_KP * self._yaw()))
            self.walk.substep(self.d)
            if a > 0: self.d.ctrl[:] = (1 - a) * self.d.ctrl + a * self.stand_ctrl
            return a
        if self.mode == "settle":                       # zero command: policy marches in place, slows
            self.walk.command[:] = [0.0, 0.0, max(-0.5, min(0.5, -self.HEADING_KP * self._yaw()))]
            self.walk.substep(self.d)
            slow = float(self.np.linalg.norm(self.d.qvel[:2])) < self.SETTLE_V
            feet = self.d.site_xpos[self.feet][:, 2]
            both_down = bool((feet < self.FOOT_DOWN_Z).all())      # freeze only in double support
            level = self._tilt() < self.LEVEL_RAD
            if self.mode_t >= self.SETTLE_S and ((slow and both_down and level) or self.mode_t > 4.0):
                self.mode, self.mode_t = "blend", 0.0
                self.stand_ctrl = self.d.qpos[7:7 + 29].copy()   # freeze where the feet ARE (not the keyframe)
            return 0.0
        if self.mode == "blend":                        # hold the actual pose, ramp gains -> stiff
            a = min(1.0, self.mode_t / self.BLEND_S)
            self.d.ctrl[:] = self.stand_ctrl; self._ankle_balance()
            if a >= 1.0: self.mode = "stand"; self.goal[:] = [self.d.qpos[0], 0.0]
            return a
        self.d.ctrl[:] = self.stand_ctrl; self._ankle_balance(); return 1.0   # stand

    def _ankle_balance(self):
        """A stiff statue topples slowly; nudge the ankles to keep the CoM over the feet."""
        e = (self.d.subtree_com[0][:2] - self.d.site_xpos[self.feet].mean(0)[:2])
        v = (e - self.com_err_prev) / self.DT if self.com_err_prev is not None else self.np.zeros(2)
        self.com_err_prev = e.copy()
        yaw = self._yaw(); c, s_ = math.cos(yaw), math.sin(yaw)
        ex, ey = c * e[0] + s_ * e[1], -s_ * e[0] + c * e[1]          # body frame
        vx, vy = c * v[0] + s_ * v[1], -s_ * v[0] + c * v[1]
        for j in (self.L_ANKLE_PITCH, self.R_ANKLE_PITCH): self.d.ctrl[j] += self.ANKLE_KP * (ex - self.COM_X_REF) + self.ANKLE_KD * vx
        for j in (self.L_ANKLE_ROLL, self.R_ANKLE_ROLL):   self.d.ctrl[j] += self.ANKLE_KP * ey + self.ANKLE_KD * vy

    def _tilt(self):
        w, x, y, z = self.d.qpos[3:7]
        roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
        return max(abs(roll), abs(pitch))
    def tilt_ok(self): return self._tilt() < TILT_ABORT_RAD
    def current_arm(self): return ArmPose({j: float(self.d.qpos[7 + j]) for j in RIGHT_ARM})
    def base_pose(self): return (float(self.d.qpos[0]), float(self.d.qpos[1]), float(self.d.qpos[2]), self._yaw())

    def _guide(self):
        """Virtual heading/lateral springs on the pelvis (see GUIDE_* comment)."""
        yaw = self._yaw(); wz = float(self.d.qvel[5])
        ey = float(self.goal[1] - self.d.qpos[1]); vy = float(self.qvel_world_y())
        f = self.d.xfrc_applied[self.pelvis_id]
        f[:] = 0.0
        f[1] = self.GUIDE_K_Y * ey - self.GUIDE_D_Y * vy
        f[5] = -self.GUIDE_K_YAW * yaw - self.GUIDE_D_YAW * wz
        k = 1.0
        if self.mode in ("walk", "unblend"):                   # fade the stand springs out after the handover
            k = max(0.0, 1.0 - self.walk_t / self.FADE_S)
        if k > 0.0:
            w, x, y, z = self.d.qpos[3:7]
            roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
            pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
            wx, wy = float(self.d.qvel[3]), float(self.d.qvel[4])
            f[0] = k * (self.STAND_K_X * float(self.goal[0] - self.d.qpos[0]) - self.STAND_D_X * float(self.d.qvel[0]))
            f[3] = k * (-self.STAND_K_ROT * roll - self.STAND_D_ROT * wx)
            f[4] = k * (-self.STAND_K_ROT * pitch - self.STAND_D_ROT * wy)

    def qvel_world_y(self):
        return float(self.d.qvel[1])                      # free-joint linear vel is in world frame

    def sleep(self, s):
        t_wall = time.time()
        for _ in range(max(1, int(round(s / self.DT)))):
            self._guide()
            a = self._leg_ctrl(); self._set_gains(a)
            if self.target is not None:                     # arm_sdk-style override, right arm only
                for j, q in self.target.q.items():
                    self.d.ctrl[j] = self.weight * q + (1 - self.weight) * self.d.ctrl[j]
            self.mj.mj_step(self.m, self.d); self.t += self.DT; self.mode_t += self.DT
            if self.mode in ("walk", "unblend"): self.walk_t += self.DT
        self.log.append((self.t, self.phase, self.hand_xyz().copy(), self.d.qpos[:3].copy()))
        if self.viewer is not None:
            self.viewer.sync(); time.sleep(max(0.0, s - (time.time() - t_wall)))
        if self.renderer is not None and int(self.t * 30) > len(self.frames):
            self.cam.lookat[:] = self.d.qpos[:3] + [0.2, -0.2, 0.0]
            self.renderer.update_scene(self.d, camera=self.cam)
            self.frames.append(self.renderer.render().copy())

    def finish(self):
        np = self.np
        if self.renderer is not None:
            import imageio
            imageio.mimwrite(self.video, self.frames, fps=30)
            print(f"sim: wrote {self.video} ({len(self.frames)} frames)")
        print("\nsim report (hand = right_palm site, world frame):")
        print(f"  {'phase':<8}{'pelvis x [m]':>13}{'hand z-rope z':>15}{'hand x drift':>14}{'hand y-rope y':>15}")
        phases = []
        for t, ph, xyz, pel in self.log:
            if not phases or phases[-1][0] != ph: phases.append((ph, [], []))
            phases[-1][1].append(xyz); phases[-1][2].append(pel)
        for ph, pts, pels in phases:
            a, b = np.array(pts), np.array(pels)
            print(f"  {ph:<8}{b[-1,0]:>13.2f}{a[-1,2]-ROPE_Z:>+15.3f}{a[:,0].max()-a[:,0].min():>14.3f}{a[-1,1]-ROPE_Y:>+15.3f}")
        print("  (WALK: pelvis x should advance; hand x drift ~0 while tracking, then grows as it slides)")
        if self.viewer is not None:
            print("sim: close the viewer window to exit")
            while self.viewer.is_running(): time.sleep(0.1)


# ---------------------------------------------------------------- sequencer
class RopeWalker:
    def __init__(self, bot, speed: float):
        self.bot, self.speed = bot, speed
        self.dt = 1.0 / CTRL_HZ
        self.cur = bot.current_arm()
        self.grip_pt = None                             # world point the palm is pinned to
        self.slid = 0.0                                 # total ascender slide along the rope [m]

    def _phase(self, name):
        print(f"== {name}")
        if hasattr(self.bot, "phase"): self.bot.phase = name.split()[0].strip(".:")

    def _ramp(self, target: ArmPose, dur: float):
        """Interpolate the arm to `target` over `dur` s at CTRL_HZ, checking tilt every tick."""
        start, n = self.cur, max(1, int(dur * CTRL_HZ))
        for i in range(1, n + 1):
            if not self.bot.tilt_ok():
                self.abort("tilt")
            self.cur = lerp_pose(start, target, i / n)
            self.bot.send_arm(self.cur)
            self.bot.sleep(self.dt)

    def abort(self, why):
        print(f"!! ABORT ({why}) at t={getattr(self.bot, 't', 0):.2f}s: StopMove + arm weight -> 0")
        self.bot.stop_move(); self.bot.set_arm_weight(0.0); self.bot.send_arm(self.cur)
        if hasattr(self.bot, "finish"): self.bot.finish()
        sys.exit(1)

    def take_arm(self):
        self._phase("TAKE: arm_sdk takes the right arm")
        for i in range(1, 51):                       # weight 0 -> 1 in 1 s, holding current q
            self.bot.set_arm_weight(i / 50); self.bot.send_arm(self.cur); self.bot.sleep(0.02)

    def _shoulder_world(self):
        bx, by, bz, yaw = self.bot.base_pose()
        ox, oy, oz = SHOULDER_OFF
        return (bx + math.cos(yaw) * ox - math.sin(yaw) * oy, by + math.sin(yaw) * ox + math.cos(yaw) * oy, bz + oz)

    def _new_grip_point(self):
        """Where the hand goes next: HAND_X_FRONT ahead of the shoulder, on the rope."""
        sx, _, _ = self._shoulder_world()
        return (sx + HAND_X_FRONT, ROPE_Y, ROPE_Z)

    def _track(self, dur: float, lift: float = 0.0):
        """Hold the palm at self.grip_pt (world) for `dur` s, recomputing the arm from the base
        pose every tick. ASCENDER RATCHET: the cam slides freely UP the rope (forward, the walking
        direction) and bites when loaded the other way, so when the arm runs out of reach the
        grip point advances along the rope; it never moves back."""
        n = max(1, int(dur * CTRL_HZ)); sliding = False
        for i in range(n):
            if not self.bot.tilt_ok(): self.abort("tilt")
            # The velocity command carries a 1 s duration (see G1.move), so re-issue it
            # twice a second while walking or the robot stops partway through the track.
            if getattr(self.bot, "vx", 0.0) and i and i % max(1, CTRL_HZ // 2) == 0:
                self.bot.move(self.bot.vx)
            base = self.bot.base_pose()
            pt = (self.grip_pt[0], self.grip_pt[1], self.grip_pt[2] + lift)
            target, dx = hand_at_world(pt, base)
            if dx < HAND_X_BACK:
                adv = HAND_X_BACK - dx
                self.grip_pt = (self.grip_pt[0] + adv, self.grip_pt[1], self.grip_pt[2]); self.slid += adv
                target, dx = hand_at_world((self.grip_pt[0], self.grip_pt[1], self.grip_pt[2] + lift), base)
                if not sliding: sliding = True; print("   arm at reach limit -> ascender slides forward along the rope")
            self.cur = rate_limit(self.cur, target)
            self.bot.send_arm(self.cur); self.bot.sleep(self.dt)

    def grip(self, first: bool):
        if first:
            self._phase("GRIP: hand on the rope, ahead of the hip")
            self.grip_pt = self._new_grip_point()
            self._ramp(hand_at_world(self.grip_pt, self.bot.base_pose())[0], dur=2.5)
            return
        self._phase("REGRIP: lift, move the hand forward along the rope, put it back down")
        base = self.bot.base_pose()
        lifted = (self.grip_pt[0], self.grip_pt[1], self.grip_pt[2] + LIFT_Z)
        self._ramp(hand_at_world(lifted, base)[0], dur=0.8)             # lift straight up
        self.grip_pt = self._new_grip_point()
        lifted = (self.grip_pt[0], self.grip_pt[1], self.grip_pt[2] + LIFT_Z)
        self._ramp(hand_at_world(lifted, base)[0], dur=2.5)             # swing forward, above the rope (slow: it rocks the torso)
        self._ramp(hand_at_world(self.grip_pt, base)[0], dur=0.8)          # lower onto the rope
        self._track(0.5)                                                # settle, hand pinned

    def walk(self):
        dur = WALK_DIST / self.speed
        self._phase(f"WALK {WALK_DIST:.2f} m at {self.speed:.2f} m/s ({dur:.1f} s); hand pinned to the rope")
        self.bot.move(self.speed)
        self._track(dur + 1.0)                                          # hand static in the world (+1 s: sim prewalk)
        self.bot.stop_move()
        self._track(3.0)                                                # gait settles, hand still pinned

    def run(self, cycles: int, release: bool):
        self.bot.start()
        self.take_arm()
        self.grip(first=True)
        self.walk()
        for _ in range(cycles):
            self.grip(first=False)
            self.walk()
        self._phase(f"DONE: standing, hand on rope (ascender slid {self.slid:.2f} m in total)")
        if release:
            for i in range(50, -1, -1):
                self.bot.set_arm_weight(i / 50); self.bot.send_arm(self.cur); self.bot.sleep(0.02)
        else:
            self._track(3.0)                         # hold 3 s so the pose is visible
        if hasattr(self.bot, "finish"): self.bot.finish()


def main():
    global ROPE_Z
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iface", default="eth0", help="network interface to the G1")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, no robot")
    ap.add_argument("--sim", action="store_true", help="MuJoCo stand-in (opens a viewer unless --video)")
    ap.add_argument("--video", default=None, help="with --sim: render headless to this .mp4 instead of a viewer")
    ap.add_argument("--rope-z", type=float, default=ROPE_Z, help="rope height [m]")
    ap.add_argument("--speed", type=float, default=0.3, help="walk speed m/s (start at 0.2)")
    ap.add_argument("--cycles", type=int, default=CYCLES, help="re-grip + walk repetitions")
    ap.add_argument("--release", action="store_true", help="give the arm back to the loco controller at the end")
    a = ap.parse_args()
    ROPE_Z = a.rope_z

    # sanity: is the rope reachable? (IK clamps instead of failing, but warn loudly)
    for dx in (HAND_X_FRONT, HAND_X_BACK):
        need = math.sqrt(dx ** 2 + (ROPE_Y - SHOULDER_Y) ** 2 + (ROPE_Z - SHOULDER_Z) ** 2)
        if need > REACH_MAX:
            print(f"WARNING hand dx={dx:+.2f} at rope (y={ROPE_Y:.2f}, z={ROPE_Z:.2f}) needs {need:.2f} m reach > "
                  f"{REACH_MAX:.2f}; hand will hover ~{need - REACH_MAX:.2f} m above the rope (raise the rope / bring it closer)")

    if a.dry_run: bot = DryRun()
    elif a.sim:   bot = Sim(viewer=a.video is None, video=a.video)
    else:         bot = G1(a.iface)
    if not (a.dry_run or a.sim):
        input("G1 standing, area clear, e-stop in hand. ENTER to start / Ctrl-C to quit ")
    RopeWalker(bot, a.speed).run(a.cycles, a.release)


if __name__ == "__main__":
    main()

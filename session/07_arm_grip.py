"""STEP 7 -- rope GRIP using our own low-level control. No arm_sdk, no LocoClient.

Why this exists: in NORMAL mode every service-mediated path is inert on this
robot. rt/lowcmd is blocked, arm_sdk has no active service to blend into, and
LocoClient will not locomote a robot that is not actively standing. All three
need the 'ai' service to be doing something, and it is idle.

But DEVELOPER mode + rt/lowcmd demonstrably works -- session/tracktest measured
torque = kp * error on every joint and the robot tracked commanded positions.
So we drive the right arm ourselves, with the same planar IK rope_walk.py uses,
and hold everything else where it already is.

  robot dev-mode          # REQUIRED: releases the motion service
  python3 session/07_arm_grip.py --iface enP8p1s0            # dry
  python3 session/07_arm_grip.py --iface enP8p1s0 --arm      # live

Legs/waist are held at their CURRENT measured pose, so the robot keeps standing
exactly as it is; only the right arm moves. Ctrl-C damps all 29 joints.
"""
import argparse
import math
import os
import signal
import sys
import time

import numpy as np
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deploy import constants as C                       # noqa: E402

# --- geometry and IK, lifted verbatim from deploy/rope_walk.py ---------------

ROPE_Z = 0.60          # rope height above ground (user spec)
SHOULDER_Z = 0.95      # right shoulder pitch axis height when standing (G1, ~1.32 m tall)
ROPE_Y = -0.30         # rope lateral offset from body centre (right side = -y)
SHOULDER_Y = -0.15     # right shoulder lateral offset from body centre
UPPER_ARM = 0.22       # shoulder -> elbow
FOREARM = 0.24         # elbow -> hand (incl. ascender/palm)

HAND_X_FRONT = 0.20    # hand ahead of shoulder right after a grip
HAND_X_BACK = -0.20    # furthest the hand can trail behind the shoulder while on the rope;
                       # beyond that the hand SLIDES along the rope (still on it)
LIFT_Z = 0.08          # how high the hand lifts off the rope when re-gripping
WALK_DIST = 0.75       # per walk phase, ~3 steps of 0.25 m
CYCLES = 2             # number of (re-grip + walk) after the first grip+walk

# ---------------------------------------------------------------- joint map (G1 29-DoF)
R_SHOULDER_PITCH, R_SHOULDER_ROLL, R_SHOULDER_YAW = 22, 23, 24
R_ELBOW, R_WRIST_ROLL, R_WRIST_PITCH, R_WRIST_YAW = 25, 26, 27, 28
RIGHT_ARM = [R_SHOULDER_PITCH, R_SHOULDER_ROLL, R_SHOULDER_YAW,
             R_ELBOW, R_WRIST_ROLL, R_WRIST_PITCH, R_WRIST_YAW]
ARM_SDK_WEIGHT_IDX = 29            # "kNotUsedJoint": q of motor 29 = arm_sdk weight [0,1]

# Expected G1 signs: shoulder pitch <0 raises the arm forward; elbow >0 flexes;
# right shoulder roll <0 abducts (moves the arm away from the body).
SIGN_SHOULDER_PITCH_FWD = -1.0
SIGN_ELBOW_FLEX = +1.0
SIGN_R_SHOULDER_ROLL_OUT = -1.0

KP_ARM, KD_ARM = 60.0, 1.5          # gentle: the rope reacts on the hand
CTRL_HZ = 50
TILT_ABORT_RAD = math.radians(25)


@dataclass
class ArmPose:
    q: dict  # joint index -> target rad


def ik_right_arm(hand_dx: float, hand_dz: float) -> ArmPose:
    """Planar 2-link IK. hand_dx: forward of shoulder (+x), hand_dz: below shoulder (negative)."""
    r = math.hypot(hand_dx, hand_dz)
    r = min(r, (UPPER_ARM + FOREARM) * 0.98)          # never fully straight
    # elbow angle from law of cosines (0 = straight)
    c = (UPPER_ARM**2 + FOREARM**2 - r**2) / (2 * UPPER_ARM * FOREARM)
    elbow = math.pi - math.acos(max(-1.0, min(1.0, c)))
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
        R_ELBOW: SIGN_ELBOW_FLEX * elbow,
        R_WRIST_ROLL: 0.0,
        R_WRIST_PITCH: 0.0,
        R_WRIST_YAW: 0.0,
    })


def hand_on_rope(dx: float, lift: float = 0.0) -> ArmPose:
    return ik_right_arm(dx, (ROPE_Z + lift) - SHOULDER_Z)


def lerp_pose(a: ArmPose, b: ArmPose, s: float) -> ArmPose:
    s = max(0.0, min(1.0, s))
    return ArmPose({j: a.q[j] + (b.q[j] - a.q[j]) * s for j in a.q})




# Full fixed-rope gesture, per cycle:
#   PULL   hand travels front -> back (the haul: body would move forward past a
#          hand fixed on the rope, so in the body frame the hand trails backward)
#   LIFT   hand comes off the rope
#   SWING  hand travels back -> front while clear of the rope
#   LOWER  hand settles back onto the rope, ready to haul again
RAMP_S   = 3.0     # initial reach to the first grip
PULL_S   = 3.0
LIFT_S   = 0.8
SWING_S  = 2.0
LOWER_S  = 0.8
SETTLE_S = 0.4
HOLD_S   = 2.0     # final hold so the last pose is visible
DAMP_KD  = 8.0
ARM_KP, ARM_KD = 40.0, 1.0          # gentle on the arm
MAX_STEP = 0.03                     # rad/step slew limit


class Robot:
    def __init__(self, iface, armed):
        self.armed = armed
        from unitree_sdk2py.core.channel import (ChannelFactoryInitialize,
                                                 ChannelPublisher, ChannelSubscriber)
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.utils.crc import CRC
        ChannelFactoryInitialize(0, iface)
        self._crc = CRC()
        self._pub = ChannelPublisher("rt/lowcmd", LowCmd_); self._pub.Init()
        self._cmd = unitree_hg_msg_dds__LowCmd_()
        self.state = None; self.mm = None
        ChannelSubscriber("rt/lowstate", LowState_).Init(self._on, 1)

    def _on(self, m):
        self.state = m
        if self.mm is None:
            self.mm = int(m.mode_machine)

    def wait(self):
        t0 = time.time()
        while self.state is None or self.mm is None:
            if time.time() - t0 > 10:
                raise RuntimeError("no lowstate")
            time.sleep(0.05)
        print(f"lowstate up, mode_machine={self.mm}")

    def q(self):
        return np.array([self.state.motor_state[i].q for i in range(C.N_JOINTS)], np.float32)

    def tau_arm(self):
        return [float(self.state.motor_state[j].tau_est) for j in RIGHT_ARM]

    def send(self, q, kp, kd):
        self._cmd.mode_pr = C.MODE_PR; self._cmd.mode_machine = self.mm
        for i in range(C.N_JOINTS):
            c = self._cmd.motor_cmd[i]
            c.mode = C.MOTOR_MODE_ENABLE
            c.q = float(q[i]); c.dq = 0.0; c.tau = 0.0
            c.kp = float(kp[i]); c.kd = float(kd[i])
        if self.armed:
            self._cmd.crc = self._crc.Crc(self._cmd); self._pub.Write(self._cmd)

    def damp(self):
        self.send(np.zeros(C.N_JOINTS), np.zeros(C.N_JOINTS),
                  np.full(C.N_JOINTS, DAMP_KD))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default="enP8p1s0")
    ap.add_argument("--arm", action="store_true")
    ap.add_argument("--cycles", type=int, default=2,
                    help="pull + re-grip repetitions (default 2)")
    a = ap.parse_args()
    if not a.arm:
        print("DRY RUN -- nothing published.\n")

    r = Robot(a.iface, a.arm); r.wait()
    q0 = r.q()

    def arm_target(pose):
        """Full 29-joint target: everything held at q0, right arm from `pose`."""
        t = q0.copy()
        for j, v in pose.q.items():
            t[j] = v
        return t

    front   = arm_target(hand_on_rope(HAND_X_FRONT))
    back    = arm_target(hand_on_rope(HAND_X_BACK))
    back_up = arm_target(hand_on_rope(HAND_X_BACK, LIFT_Z))
    front_up = arm_target(hand_on_rope(HAND_X_FRONT, LIFT_Z))

    phases = [("reach to rope", front, RAMP_S)]
    for i in range(a.cycles):
        phases += [
            (f"PULL {i+1}/{a.cycles}",  back,     PULL_S),
            (f"lift {i+1}",             back_up,  LIFT_S),
            (f"swing fwd {i+1}",        front_up, SWING_S),
            (f"lower {i+1}",            front,    LOWER_S),
            (f"settle {i+1}",           front,    SETTLE_S),
        ]
    phases += [("hold", front, HOLD_S)]
    total = sum(d for _, _, d in phases)
    print(f"holding all joints at their current pose; moving ONLY the right arm")
    print(f"{a.cycles} pull/re-grip cycles, {total:.1f}s total")
    print(f"  hand travels x = {HAND_X_FRONT:+.2f} -> {HAND_X_BACK:+.2f} m "
          f"(pull), lifting {LIFT_Z:.2f} m to swing back\n")

    kp = C.TRAIN_KP.copy(); kd = C.TRAIN_KD.copy()
    for j in RIGHT_ARM:
        kp[j], kd[j] = ARM_KP, ARM_KD

    def stop(sig, _f):
        print(f"\nsignal {sig} -> damping")
        for _ in range(15):
            r.damp(); time.sleep(0.01)
        sys.exit(0)
    signal.signal(signal.SIGINT, stop); signal.signal(signal.SIGTERM, stop)

    prev = q0.copy()
    try:
        for name, tgt_end, dur in phases:
            tgt_start = prev.copy()
            n = max(1, int(dur / C.CTRL_DT))
            t_phase = time.time()
            for i in range(1, n + 1):
                tick = time.perf_counter()
                s_ = i / n
                tgt = (1 - s_) * tgt_start + s_ * tgt_end
                d = tgt - prev
                step = float(np.abs(d).max())
                if step > MAX_STEP:
                    tgt = prev + d * (MAX_STEP / step)
                r.send(tgt, kp, kd); prev = tgt.copy()
                time.sleep(max(0.0, C.CTRL_DT - (time.perf_counter() - tick)))
            q = r.q()
            err = max(abs(q[j] - prev[j]) for j in RIGHT_ARM)
            tau = max(abs(v) for v in r.tau_arm())
            print(f"  {name:16s} {time.time()-t_phase:4.1f}s  "
                  f"arm |q-tgt|max={err:.3f}  max|tau_arm|={tau:5.2f} N.m")
    finally:
        for _ in range(15):
            r.damp(); time.sleep(0.01)
        print("\ndamped, exiting. CONFIRM ON CAMERA.")


if __name__ == "__main__":
    main()

"""G1 "hiker on a fixed rope" mimic — unitree_sdk2py.

    python deploy/rope_walk.py --iface eth0            # real robot
    python deploy/rope_walk.py --dry-run               # print the plan, no SDK

Sequence (strictly sequential, one phase at a time):
    0. arm_sdk takes over the RIGHT arm (weight ramp 0->1), left arm untouched.
    1. GRIP     : right hand placed on the rope (rope runs along +x, on the robot's
                  right side, ROPE_Z above ground). Hand starts slightly ahead of the hip.
    2. WALK     : LocoClient walks forward WALK_DIST (~3-4 steps). While the body moves,
                  the arm is interpolated backward at the same rate so the hand stays at
                  the SAME WORLD POINT on the rope until the arm reaches HAND_X_BACK; from
                  there the hand slides along the rope. It never leaves the rope.
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

# ---------------------------------------------------------------- geometry (metres)
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


# ---------------------------------------------------------------- robot backends
class DryRun:
    """Prints what would be sent. Lets you check the sequence and IK without a robot."""
    def __init__(self):
        self.t0 = time.time()

    def _log(self, msg): print(f"[{time.time() - self.t0:6.2f}s] {msg}")
    def start(self): self._log("loco: Start (balance stand)")
    def set_arm_weight(self, w): self._log(f"arm_sdk weight={w:.2f}")
    def send_arm(self, pose: ArmPose):
        sp, el, ro = pose.q[R_SHOULDER_PITCH], pose.q[R_ELBOW], pose.q[R_SHOULDER_ROLL]
        self._log(f"arm  sh_pitch={sp:+.2f} sh_roll={ro:+.2f} elbow={el:+.2f}")
    def move(self, vx): self._log(f"loco: SetVelocity vx={vx:.2f} (1 s, re-issued)"); return 0
    def stop_move(self): self._log("loco: StopMove")
    def tilt_ok(self): return True
    def current_arm(self): return ArmPose({j: 0.0 for j in RIGHT_ARM})
    def sleep(self, s): time.sleep(min(s, 0.02))   # fast-forward


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
        t0 = time.time()
        while self.low is None and time.time() - t0 < 3.0:
            time.sleep(0.05)
        if self.low is None:
            sys.exit("no rt/lowstate — check --iface and that the robot is on")

    def _on_low(self, m):
        with self.lock: self.low = m

    def start(self):
        # robot must already be standing (remote: L2+A -> R2+B ...). Start() = walk-ready mode.
        # Start() is SetFsmId(200) on the installed SDK and RETURNS A CODE. The original
        # discarded it, so a refused transition looked identical to a successful one --
        # the script printed every phase while the robot never moved.
        code = self.loco.SetFsmId(200)
        if code != 0:
            sys.exit(f"loco SetFsmId(200) refused with code {code}; the robot is not in a "
                     f"state that accepts motion. Put it in a standing/walk-ready stance "
                     f"first (physical remote: L2+A then R2+B) and retry.")
        print(f"   loco FSM 200 (walk-ready) accepted, code={code}")
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
        # LocoClient.Move() is SetVelocity(..., duration=1) and returns None. The original
        # issued it ONCE and then spent WALK_DIST/speed seconds ramping the arm, so the
        # command expired after 1 s and the robot walked at most 0.2 m of the intended
        # 0.75 m. We re-issue the SHORT command instead of using continous_move=True
        # (which sets a 864000 s duration): if this process dies mid-walk, a 1 s duration
        # means the robot stops by itself within a second. That is a safety property.
        code = self.loco.SetVelocity(vx, 0.0, 0.0, 1.0)
        if code != 0:
            raise RuntimeError(f"SetVelocity refused, code {code}")
        return code

    def stop_move(self): self.loco.StopMove()

    def tilt_ok(self):
        with self.lock: low = self.low
        r, p, _ = low.imu_state.rpy
        return abs(r) < TILT_ABORT_RAD and abs(p) < TILT_ABORT_RAD

    def current_arm(self):
        with self.lock: low = self.low
        return ArmPose({j: low.motor_state[j].q for j in RIGHT_ARM})

    def sleep(self, s): time.sleep(s)


# ---------------------------------------------------------------- sequencer
class RopeWalker:
    def __init__(self, bot, speed: float):
        self.bot, self.speed = bot, speed
        self.dt = 1.0 / CTRL_HZ
        self.cur = bot.current_arm()
        self._walk_vx = 0.0        # non-zero while a walk phase is running

    def _ramp(self, target: ArmPose, dur: float):
        """Interpolate the arm to `target` over `dur` s at CTRL_HZ, checking tilt every tick.

        If a walk is in progress, re-issue the 1 s velocity command twice a second so it
        does not expire mid-ramp. Stops within 1 s on its own if this process dies.
        """
        start, n = self.cur, max(1, int(dur * CTRL_HZ))
        for i in range(1, n + 1):
            if not self.bot.tilt_ok():
                self.abort("tilt")
            if self._walk_vx and i % max(1, CTRL_HZ // 2) == 0:
                self.bot.move(self._walk_vx)
            self.cur = lerp_pose(start, target, i / n)
            self.bot.send_arm(self.cur)
            self.bot.sleep(self.dt)

    def abort(self, why):
        print(f"!! ABORT ({why}): StopMove + arm weight -> 0")
        self._walk_vx = 0.0
        self.bot.stop_move(); self.bot.set_arm_weight(0.0); self.bot.send_arm(self.cur)
        sys.exit(1)

    def take_arm(self):
        print("== 0. arm_sdk takes the right arm")
        for i in range(1, 51):                       # weight 0 -> 1 in 1 s, holding current q
            self.bot.set_arm_weight(i / 50); self.bot.send_arm(self.cur); self.bot.sleep(0.02)

    def grip(self, first: bool):
        if first:
            print("== 1. GRIP: hand on the rope, ahead of the hip")
            self._ramp(hand_on_rope(HAND_X_FRONT), dur=2.5)
            return
        print("== 3. RE-GRIP: lift, swing forward, put the hand back on the rope")
        dx_now = HAND_X_BACK
        self._ramp(hand_on_rope(dx_now, LIFT_Z), dur=0.8)             # lift straight up
        self._ramp(hand_on_rope(HAND_X_FRONT, LIFT_Z), dur=1.5)       # swing forward, above rope
        self._ramp(hand_on_rope(HAND_X_FRONT), dur=0.8)               # lower onto rope
        self.bot.sleep(0.5)                                           # settle

    def walk(self):
        dur = WALK_DIST / self.speed
        print(f"== WALK {WALK_DIST:.2f} m at {self.speed:.2f} m/s ({dur:.1f} s); hand stays on rope")
        self._walk_vx = self.speed
        self.bot.move(self.speed)
        # body moves at `speed`, so the hand (fixed in the world) moves backward in the body
        # frame at the same rate, until the arm hits HAND_X_BACK; then it slides on the rope.
        track = (HAND_X_FRONT - HAND_X_BACK) / self.speed
        self._ramp(hand_on_rope(HAND_X_BACK), dur=min(track, dur))
        if dur > track:
            self._ramp(self.cur, dur=dur - track)                 # hand slides, arm holds
        self._walk_vx = 0.0
        self.bot.stop_move()
        self.bot.sleep(1.5)                                           # let the gait settle
        # hold pose while settling (keeps arm_sdk alive)
        self._ramp(self.cur, dur=0.5)

    def run(self, cycles: int, release: bool):
        self.bot.start()
        self.take_arm()
        self.grip(first=True)
        self.walk()
        for _ in range(cycles):
            self.grip(first=False)
            self.walk()
        print("== 5. done: standing, hand on rope")
        if release:
            for i in range(50, -1, -1):
                self.bot.set_arm_weight(i / 50); self.bot.send_arm(self.cur); self.bot.sleep(0.02)
        else:
            for _ in range(int(3 * CTRL_HZ)):        # hold 3 s so the pose is visible
                self.bot.send_arm(self.cur); self.bot.sleep(self.dt)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iface", default="eth0", help="network interface to the G1")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, no robot")
    ap.add_argument("--speed", type=float, default=0.3, help="walk speed m/s (start at 0.2)")
    ap.add_argument("--cycles", type=int, default=CYCLES, help="re-grip + walk repetitions")
    ap.add_argument("--release", action="store_true", help="give the arm back to the loco controller at the end")
    a = ap.parse_args()

    # sanity: is the rope reachable?
    reach = UPPER_ARM + FOREARM
    for dx in (HAND_X_FRONT, HAND_X_BACK):
        need = math.hypot(dx, ROPE_Z + LIFT_Z - SHOULDER_Z)
        if need > reach * 0.98:
            sys.exit(f"hand dx={dx:+.2f} needs {need:.2f} m reach > {reach:.2f}; reduce HAND_X_FRONT/BACK")

    bot = DryRun() if a.dry_run else G1(a.iface)
    if not a.dry_run:
        input("G1 standing, area clear, e-stop in hand. ENTER to start / Ctrl-C to quit ")
    RopeWalker(bot, a.speed).run(a.cycles, a.release)


if __name__ == "__main__":
    main()

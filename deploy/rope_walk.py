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
from pathlib import Path

# ---------------------------------------------------------------- geometry (metres)
ROPE_Z = 0.60          # rope height above ground (user spec); override with --rope-z
ROPE_Y = -0.30         # rope lateral offset from body centre (right side = -y)
# measured in MuJoCo from assets/robots/mujoco/g1_unitree.xml (pelvis z=0.79, standing):
SHOULDER_Z = 1.085     # right_shoulder_pitch_link z
SHOULDER_Y = -0.100    # right_shoulder_pitch_link y
UPPER_ARM = 0.19       # shoulder -> elbow
FOREARM = 0.24         # elbow -> palm (wrist_yaw link is 0.20, +0.04 to the palm)
HAND_OFFSET = 0.04     # palm ahead of right_wrist_yaw_link along its x (sim metric only)

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
    def move(self, vx): self._log(f"loco: Move vx={vx:.2f}")
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
        self.loco.Start(); time.sleep(3.0)

    def set_arm_weight(self, w): self.weight = max(0.0, min(1.0, w))

    def send_arm(self, pose: ArmPose):
        self.cmd.motor_cmd[ARM_SDK_WEIGHT_IDX].q = self.weight
        for j, q in pose.q.items():
            m = self.cmd.motor_cmd[j]
            m.mode, m.q, m.dq, m.tau, m.kp, m.kd = 1, float(q), 0.0, 0.0, KP_ARM, KD_ARM
        self.cmd.crc = self.crc.Crc(self.cmd)
        self.pub.Write(self.cmd)

    def move(self, vx): self.loco.Move(vx, 0.0, 0.0)
    def stop_move(self): self.loco.StopMove()

    def tilt_ok(self):
        with self.lock: low = self.low
        r, p, _ = low.imu_state.rpy
        return abs(r) < TILT_ABORT_RAD and abs(p) < TILT_ABORT_RAD

    def current_arm(self):
        with self.lock: low = self.low
        return ArmPose({j: low.motor_state[j].q for j in RIGHT_ARM})

    def sleep(self, s): time.sleep(s)



class Sim:
    """MuJoCo stand-in. Same interface as G1. Legs/waist/left arm are held at the 'stand'
    keyframe by the MJCF position actuators; the pelvis is welded to a mocap anchor that we
    drag forward to fake walking (the real loco controller is not simulable). Right arm gets
    weight*target + (1-weight)*keyframe, mirroring what arm_sdk's weight does on the robot."""
    DT = 0.002

    def __init__(self, viewer: bool, video: str | None):
        import mujoco
        self.mj = mujoco
        here = Path(__file__).parent / "sim" / "scene.xml"
        self.m = mujoco.MjModel.from_xml_path(str(here))
        self.m.geom("rope").pos[:] = [2.0, ROPE_Y, ROPE_Z]         # move rope to --rope-z
        self.d = mujoco.MjData(self.m)
        mujoco.mj_resetDataKeyframe(self.m, self.d, self.m.key("stand").id)
        self.key_ctrl = self.m.key("stand").ctrl.copy()
        self.ctrl = self.key_ctrl.copy()
        self.d.mocap_pos[0] = self.d.qpos[:3]
        self.weight, self.vx = 0.0, 0.0
        self.wrist = self.m.body("right_wrist_yaw_link").id
        self.t = 0.0
        self.log = []                                              # (t, phase, hand xyz)
        self.phase = "init"
        self.viewer = mujoco.viewer.launch_passive(self.m, self.d) if viewer else None
        self.frames, self.renderer = [], None
        if video:
            self.renderer = mujoco.Renderer(self.m, 720, 1280)
            self.video = video
        mujoco.mj_forward(self.m, self.d)

    def hand_xyz(self):
        import numpy as np
        R = self.d.xmat[self.wrist].reshape(3, 3)
        return self.d.xpos[self.wrist] + R @ np.array([HAND_OFFSET, 0, 0])

    def start(self): print("sim: standing (welded pelvis)")
    def set_arm_weight(self, w): self.weight = max(0.0, min(1.0, w))
    def send_arm(self, pose: ArmPose):
        for j, q in pose.q.items():
            self.ctrl[j] = self.weight * q + (1 - self.weight) * self.key_ctrl[j]
    def move(self, vx): self.vx = vx
    def stop_move(self): self.vx = 0.0
    def tilt_ok(self): return True
    def current_arm(self): return ArmPose({j: float(self.d.ctrl[j]) for j in RIGHT_ARM})

    def sleep(self, s):
        n = max(1, int(round(s / self.DT)))
        for _ in range(n):
            self.d.mocap_pos[0][0] += self.vx * self.DT
            self.d.ctrl[:] = self.ctrl
            self.mj.mj_step(self.m, self.d)
            self.t += self.DT
        self.log.append((self.t, self.phase, self.hand_xyz().copy()))
        if self.viewer is not None:
            self.viewer.sync(); time.sleep(s)
        if self.renderer is not None and int(self.t * 30) > len(self.frames):
            self.renderer.update_scene(self.d, camera=-1)
            self.frames.append(self.renderer.render().copy())

    def finish(self):
        import numpy as np
        if self.renderer is not None:
            import imageio
            imageio.mimwrite(self.video, self.frames, fps=30)
            print(f"sim: wrote {self.video} ({len(self.frames)} frames)")
        print("\nsim report (hand = palm, world frame):")
        print(f"  {'phase':<8}{'hand z-rope z [m]':>20}{'hand x drift [m]':>18}{'hand y-rope y [m]':>20}")
        phases = []
        for t, ph, xyz in self.log:
            if not phases or phases[-1][0] != ph: phases.append((ph, []))
            phases[-1][1].append(xyz)
        for ph, pts in phases:
            a = np.array(pts)
            print(f"  {ph:<8}{a[-1,2]-ROPE_Z:>+20.3f}{a[:,0].max()-a[:,0].min():>18.3f}{a[-1,1]-ROPE_Y:>+20.3f}")
        print("  (walk phases: x drift should be ~0 while the arm tracks, then grow when the hand slides)")
        if self.viewer is not None:
            print("sim: close the viewer window to exit"); 
            while self.viewer.is_running(): time.sleep(0.1)


# ---------------------------------------------------------------- sequencer
class RopeWalker:
    def __init__(self, bot, speed: float):
        self.bot, self.speed = bot, speed
        self.dt = 1.0 / CTRL_HZ
        self.cur = bot.current_arm()

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
        print(f"!! ABORT ({why}): StopMove + arm weight -> 0")
        self.bot.stop_move(); self.bot.set_arm_weight(0.0); self.bot.send_arm(self.cur)
        sys.exit(1)

    def take_arm(self):
        self._phase("TAKE: arm_sdk takes the right arm")
        for i in range(1, 51):                       # weight 0 -> 1 in 1 s, holding current q
            self.bot.set_arm_weight(i / 50); self.bot.send_arm(self.cur); self.bot.sleep(0.02)

    def grip(self, first: bool):
        if first:
            self._phase("GRIP: hand on the rope, ahead of the hip")
            self._ramp(hand_on_rope(HAND_X_FRONT), dur=2.5)
            return
        self._phase("REGRIP: lift, swing forward, put the hand back on the rope")
        dx_now = HAND_X_BACK
        self._ramp(hand_on_rope(dx_now, LIFT_Z), dur=0.8)             # lift straight up
        self._ramp(hand_on_rope(HAND_X_FRONT, LIFT_Z), dur=1.5)       # swing forward, above rope
        self._ramp(hand_on_rope(HAND_X_FRONT), dur=0.8)               # lower onto rope
        self.bot.sleep(0.5)                                           # settle

    def walk(self):
        dur = WALK_DIST / self.speed
        self._phase(f"WALK {WALK_DIST:.2f} m at {self.speed:.2f} m/s ({dur:.1f} s); hand stays on rope")
        self.bot.move(self.speed)
        # body moves at `speed`, so the hand (fixed in the world) moves backward in the body
        # frame at the same rate, until the arm hits HAND_X_BACK; then it slides on the rope.
        track = (HAND_X_FRONT - HAND_X_BACK) / self.speed
        self._ramp(hand_on_rope(HAND_X_BACK), dur=min(track, dur))
        if dur > track:
            self._ramp(self.cur, dur=dur - track)                 # hand slides, arm holds
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
        self._phase("DONE: standing, hand on rope")
        if release:
            for i in range(50, -1, -1):
                self.bot.set_arm_weight(i / 50); self.bot.send_arm(self.cur); self.bot.sleep(0.02)
        else:
            for _ in range(int(3 * CTRL_HZ)):        # hold 3 s so the pose is visible
                self.bot.send_arm(self.cur); self.bot.sleep(self.dt)
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
    reach = (UPPER_ARM + FOREARM) * 0.98
    for dx in (HAND_X_FRONT, HAND_X_BACK):
        need = math.hypot(dx, ROPE_Z - SHOULDER_Z)
        if need > reach:
            print(f"WARNING hand dx={dx:+.2f} at rope z={ROPE_Z:.2f} needs {need:.2f} m reach > {reach:.2f}; "
                  f"hand will hover ~{need - reach:.2f} m above the rope (lower the stance or raise the rope)")

    if a.dry_run: bot = DryRun()
    elif a.sim:   bot = Sim(viewer=a.video is None, video=a.video)
    else:         bot = G1(a.iface)
    if not (a.dry_run or a.sim):
        input("G1 standing, area clear, e-stop in hand. ENTER to start / Ctrl-C to quit ")
    RopeWalker(bot, a.speed).run(a.cycles, a.release)


if __name__ == "__main__":
    main()

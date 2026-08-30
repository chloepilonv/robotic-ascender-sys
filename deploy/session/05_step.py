"""STEP 5 -- a few steps at LOW command. Deliberately short and slow.

Run ONLY after 04 has stood cleanly. Developer mode, camera open, hand on Ctrl-C.

Permitted by the access doc §5: "stepping in place and short walks -- a metre".
NOT permitted: walking any real distance. This script is bounded so it cannot
drift into that -- see the budget below.

THE KNOWN DEFECT, stated plainly: obs[0:3] is base linear velocity and there is
no estimator, so it is fed as ZEROS. Standing, that is correct (true velocity is
~0). Moving, it is WRONG, and the error grows with speed -- roughly 0.05 rad of
joint-target error at 0.2 m/s and 0.13 rad at 0.75 m/s. That is exactly why the
command here is small and brief: we are probing a known-degraded regime on
purpose, close to the edge where it is still honest.

Budget: CMD_VX=0.3 -> ~0.22 m/s (the policy does ~0.75 m/s at cmd 1.0), held
WALK_S=1.5 s plus ramps, so ~0.47 m of travel -- half the one-metre rule, so
even a large overshoot stays legal.

Sequence, all interruptible with Ctrl-C -> damping:
  Stage A  hold current pose, gains ramped from 0 -> TRAIN gains  (2 s)
  Stage B  interpolate current pose -> DEFAULT_POSE               (3 s)
  Stage C  policy closed loop at zero command, action blended in  (2 s ramp)

    python session/04_stand.py --iface eth0                 # dry, sends nothing
    python session/04_stand.py --iface eth0 --arm           # live

Gains are the TRAINING gains from the playground model (constants.TRAIN_KP/KD),
not the SDK example's -- the policy learned against that closed-loop response,
and every joint differs, ankle_roll and the wrists by 20x.
"""
import argparse
import os
import signal
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from deploy import constants as C                       # noqa: E402
from deploy.observation import ObservationBuilder       # noqa: E402
from deploy.policy import Policy                        # noqa: E402

RAMP_GAINS_S, RAMP_POSE_S, RAMP_ACTION_S = 3.0, 8.0, 3.0
SETTLE_S     = 4.0     # stand at zero command before asking for any motion
CMD_RAMP_S   = 1.0     # ease the command in and out; never step-change it
WALK_S       = 1.5     # ~2 gait cycles at 1.375 Hz -> 2-4 footfalls
CMD_VX       = 0.25    # ~0.19 m/s -> ~0.47 m total, half the 1 m limit
MAX_TILT_DEG = 30.0    # abort if the robot leans past this (rest pose is ~13 deg)
CONVERGE_TOL = 0.15    # rad: how close to DEFAULT_POSE before the policy engages
CONVERGE_MAX_S = 12.0  # give up waiting after this and report, rather than engage
DAMP_KD = 8.0          # per the access doc: kp=0, kd~8 on all 29 joints.
MAX_TARGET_STEP   = 0.05   # rad/step: SLEW LIMIT -- clamp changes above this
ABORT_TARGET_STEP = 0.30   # rad/step: genuinely wrong -> damp and stop


class Robot:
    def __init__(self, iface, armed):
        self.armed = armed
        from unitree_sdk2py.core.channel import (ChannelFactoryInitialize,
                                                 ChannelPublisher,
                                                 ChannelSubscriber)
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.utils.crc import CRC
        ChannelFactoryInitialize(0, iface)
        self._crc = CRC()
        self._pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self._pub.Init()
        self._cmd = unitree_hg_msg_dds__LowCmd_()
        self.state = None
        self.mode_machine = None
        ChannelSubscriber("rt/lowstate", LowState_).Init(self._on_state, 1)

    def _on_state(self, m):
        self.state = m
        if self.mode_machine is None:
            self.mode_machine = int(m.mode_machine)

    def wait_ready(self, timeout=10.0):
        t0 = time.time()
        while self.state is None or self.mode_machine is None:
            if time.time() - t0 > timeout:
                raise RuntimeError("no lowstate; wrong --iface or robot off")
            time.sleep(0.1)
        print(f"lowstate up, mode_machine={self.mode_machine}")

    def read(self):
        m = self.state
        q = np.array([m.motor_state[i].q for i in range(C.N_JOINTS)], np.float32)
        dq = np.array([m.motor_state[i].dq for i in range(C.N_JOINTS)], np.float32)
        gyro = np.array(m.imu_state.gyroscope, np.float32)
        quat = np.array(m.imu_state.quaternion, np.float32)
        return q, dq, gyro, quat

    def _write(self, q, kp, kd):
        self._cmd.mode_pr = C.MODE_PR
        self._cmd.mode_machine = self.mode_machine
        for i in range(C.N_JOINTS):
            mc = self._cmd.motor_cmd[i]
            mc.mode = C.MOTOR_MODE_ENABLE
            mc.q = float(q[i]); mc.dq = 0.0; mc.tau = 0.0
            mc.kp = float(kp[i]); mc.kd = float(kd[i])
        if not self.armed:
            return
        self._cmd.crc = self._crc.Crc(self._cmd)
        self._pub.Write(self._cmd)

    def send(self, q, kp, kd):
        self._write(np.asarray(q, np.float32), kp, kd)

    def damp(self):
        self._write(np.zeros(C.N_JOINTS, np.float32),
                    np.zeros(C.N_JOINTS, np.float32),
                    np.full(C.N_JOINTS, DAMP_KD, np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default="eth0")
    ap.add_argument("--policy", default="rl/policies/mels_g1_joystick.npz")
    ap.add_argument("--arm", action="store_true")
    ap.add_argument("--seconds", type=float, default=30.0)
    a = ap.parse_args()

    if not a.arm:
        print("DRY RUN: nothing is published. Add --arm to go live.\n")

    policy, builder = Policy(a.policy), ObservationBuilder()
    robot = Robot(a.iface, a.arm)
    robot.wait_ready()

    def stop(signum, _f):
        print(f"\nsignal {signum} -> damping")
        for _ in range(10):
            robot.damp(); time.sleep(0.01)
        print("damped. CONFIRM ON CAMERA.")
        sys.exit(0)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    def rss_mb():
        with open("/proc/self/status") as f:
            for ln in f:
                if ln.startswith("VmRSS:"):
                    return int(ln.split()[1]) / 1024.0
        return float("nan")

    prof = {k: [] for k in ("read", "obs", "policy", "send", "total")}
    n_slew = 0
    n_overlimit = 0
    converged = False
    t_conv = 0.0
    rss0 = rss_mb()
    print(f"RSS at start: {rss0:.1f} MB")

    q0, _, _, _ = robot.read()
    print(f"start pose max |q-default| = {np.abs(q0-C.DEFAULT_POSE).max():.3f} rad")
    prev = q0.copy()
    t0 = time.time()
    try:
        while True:
            tick = time.perf_counter()
            t = time.time() - t0
            _t = time.perf_counter()
            q, dq, gyro, quat = robot.read()
            prof["read"].append(time.perf_counter() - _t)

            if t < RAMP_GAINS_S:                       # A: wake gains on q0
                r = t / RAMP_GAINS_S
                target, kp, kd = q0, C.TRAIN_KP * r, C.TRAIN_KD * r
                stage = "A gains"
            elif t < RAMP_GAINS_S + RAMP_POSE_S:       # B: glide to default
                r = (t - RAMP_GAINS_S) / RAMP_POSE_S
                target, kp, kd = (1 - r) * q0 + r * C.DEFAULT_POSE, C.TRAIN_KP, C.TRAIN_KD
                stage = "B pose"
            elif not converged:                        # B2: HOLD until arrived
                # The armed run showed the robot stalls ~0.5 rad short of the
                # commanded pose. Engaging the policy there puts it far out of
                # distribution and its output swings ~1.8 rad between steps.
                # Wait for actual arrival instead of trusting the timer.
                err = float(np.abs(q - C.DEFAULT_POSE).max())
                target, kp, kd = C.DEFAULT_POSE, C.TRAIN_KP, C.TRAIN_KD
                stage = "B2 settle"
                if err < CONVERGE_TOL:
                    converged = True
                    t_conv = t
                    print(f"  CONVERGED at t={t:.1f}s, |q-default|max={err:.3f}")
                elif t > RAMP_GAINS_S + RAMP_POSE_S + CONVERGE_MAX_S:
                    print(f"\nSTOP: pose did not converge in {CONVERGE_MAX_S:.0f}s "
                          f"(|q-default|max={err:.3f}, need <{CONVERGE_TOL}). "
                          f"NOT engaging the policy.")
                    break
            else:                                      # C: policy
                tC = t - t_conv
                # Command schedule: settle at zero -> ease in -> hold -> ease out.
                if tC < SETTLE_S:
                    vx, stage_c = 0.0, "C stand"
                elif tC < SETTLE_S + CMD_RAMP_S:
                    vx = CMD_VX * (tC - SETTLE_S) / CMD_RAMP_S; stage_c = "D ease-in"
                elif tC < SETTLE_S + CMD_RAMP_S + WALK_S:
                    vx, stage_c = CMD_VX, "E WALK"
                elif tC < SETTLE_S + 2 * CMD_RAMP_S + WALK_S:
                    r2 = (tC - SETTLE_S - CMD_RAMP_S - WALK_S) / CMD_RAMP_S
                    vx = CMD_VX * (1.0 - r2); stage_c = "F ease-out"
                else:
                    vx, stage_c = 0.0, "G stand"
                cmd = np.array([vx, 0.0, 0.0], np.float32)
                _t = time.perf_counter()
                obs = builder.build(q, dq, gyro, quat, cmd)
                prof["obs"].append(time.perf_counter() - _t)
                # Falling detector: gravity z collapses as the robot tips over.
                tilt = np.degrees(np.arccos(np.clip(-obs[C.SLICE_GRAVITY][2], -1, 1)))
                if tilt > MAX_TILT_DEG:
                    print(f"\nABORT: tilt {tilt:.1f} deg > {MAX_TILT_DEG} deg")
                    break
                _t = time.perf_counter()
                action = policy(obs).copy()
                prof["policy"].append(time.perf_counter() - _t)
                builder.set_last_action(action); builder.advance_phase()
                r = min(1.0, tC / RAMP_ACTION_S)
                target = C.DEFAULT_POSE + r * C.ACTION_SCALE * action
                kp, kd, stage = C.TRAIN_KP, C.TRAIN_KD, stage_c

            # Slew-rate limit rather than abort. A policy output can legitimately
            # change fast; what must never reach the joints is a large STEP.
            delta = target - prev
            step = float(np.abs(delta).max())
            if step > ABORT_TARGET_STEP:
                if robot.armed:
                    print(f"\nABORT: target jumped {step:.3f} rad in one step "
                          f"(limit {ABORT_TARGET_STEP})")
                    break
                # DRY: nothing is published, so there is nothing to protect.
                # Clamp and continue so the run profiles the whole timeline.
                n_overlimit += 1
                delta *= ABORT_TARGET_STEP / step
                target = prev + delta
                step = ABORT_TARGET_STEP
            if step > MAX_TARGET_STEP:
                delta *= MAX_TARGET_STEP / step
                target = prev + delta
                n_slew += 1
            _t = time.perf_counter()
            robot.send(target, kp, kd)
            prof["send"].append(time.perf_counter() - _t)
            prev = target.copy()
            prof["total"].append(time.perf_counter() - tick)

            if int(t * 50) % 50 == 0:
                print(f"  {stage:10s} t={t:5.1f}s  |tgt-default|max="
                      f"{np.abs(target-C.DEFAULT_POSE).max():.3f}  "
                      f"|q-tgt|max={np.abs(q-target).max():.3f}")
            if t > a.seconds:
                print("\nduration reached")
                break
            time.sleep(max(0.0, C.CTRL_DT - (time.perf_counter() - tick)))
    finally:
        for _ in range(10):
            robot.damp(); time.sleep(0.01)
        print("exited via damping path. CONFIRM ON CAMERA.")
        print(f"\n=== PROFILE ({len(prof['total'])} control steps, budget "
              f"{C.CTRL_DT*1e3:.0f} ms) ===")
        print(f"{'stage':10s} {'n':>6s} {'p50 ms':>9s} {'p99 ms':>9s} {'max ms':>9s}")
        for k in ("read", "obs", "policy", "send", "total"):
            v = np.array(prof[k]) * 1e3
            if v.size:
                print(f"{k:10s} {v.size:6d} {np.percentile(v,50):9.3f} "
                      f"{np.percentile(v,99):9.3f} {v.max():9.3f}")
        tot = np.array(prof["total"]) * 1e3
        if tot.size:
            print(f"\nbudget used at p99: {np.percentile(tot,99)/(C.CTRL_DT*1e3)*100:.1f}%")
            print(f"overruns (>20 ms)  : {(tot > C.CTRL_DT*1e3).sum()} / {tot.size}")
        print(f"RSS: {rss0:.1f} -> {rss_mb():.1f} MB  (growth {rss_mb()-rss0:+.1f} MB)")
        n_pol = len(prof["policy"])
        if n_pol:
            print(f"slew-limited steps : {n_slew} / {n_pol} policy steps "
                  f"({100.0*n_slew/n_pol:.0f}%)")
            print(f"over ABORT limit   : {n_overlimit} (clamped; dry runs only)")
            print("  NOTE: in a DRY run the robot never moves, so the observation stays "
                  "far\n  out of distribution and the policy output is erratic. High slew "
                  "counts here\n  are expected and are NOT predictive of the armed run --"
                  "\n  step 04 ran armed and stood without tripping a 0.05 rad guard.")


if __name__ == "__main__":
    main()

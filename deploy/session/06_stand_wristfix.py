"""STEP 6 -- closed-loop STANDING with the wrist gain correction. No walking.

What changed from 05, and why:

  * WRIST KP 2 -> 40. With the trained kp=2 the six wrist joints droop ~0.45 rad
    under gravity, which is 5 sigma outside the policy's training distribution
    for that input; a dense MLP propagates that everywhere. Measured in shadow
    mode: max |z| 5.06 -> 3.16 and step-to-step action jitter p99 0.441 -> 0.160.
    Wrists carry no load and do not affect balance, so this is an OBSERVATION
    fix, not a control change.

  * ABORT 0.30 -> 0.60 rad/step, with the 0.05 rad/step SLEW LIMIT unchanged.
    Shadow mode showed occasional outliers implying a 0.315 rad target step --
    right on the old threshold. The slew limiter is what protects the joints;
    the abort is a backstop for something genuinely wrong, and it was set tight
    enough to trip on normal behaviour.

  * No convergence gate. The 05 gate demanded |q-default| < 0.15 on all 29
    joints, which is physically unsatisfiable: kp=2 wrists cannot beat gravity
    and kp=20 ankle_pitch sits ~0.24 rad off under load. It blocked forever.

  * ZERO command throughout. This stands; it does not walk.

    python3 session/06_stand_wristfix.py --iface enP8p1s0            # dry
    python3 session/06_stand_wristfix.py --iface enP8p1s0 --arm      # live
"""
import argparse
import json
import os
import signal
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from deploy import constants as C                       # noqa: E402
from deploy.observation import ObservationBuilder       # noqa: E402
from deploy.policy import Policy                        # noqa: E402

RAMP_GAINS_S, RAMP_POSE_S, SETTLE_S, RAMP_ACTION_S = 3.0, 8.0, 3.0, 3.0
HOLD_S = 12.0
WRIST_KP, WRIST_KD = 40.0, 1.0
SLEW_STEP, ABORT_STEP = 0.05, 0.60
MAX_TILT_DEG, DAMP_KD = 30.0, 8.0


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
        self._pub = ChannelPublisher("rt/lowcmd", LowCmd_); self._pub.Init()
        self._cmd = unitree_hg_msg_dds__LowCmd_()
        self.state = None; self.mode_machine = None
        ChannelSubscriber("rt/lowstate", LowState_).Init(self._on, 1)

    def _on(self, m):
        self.state = m
        if self.mode_machine is None:
            self.mode_machine = int(m.mode_machine)

    def wait(self, timeout=10.0):
        t0 = time.time()
        while self.state is None or self.mode_machine is None:
            if time.time() - t0 > timeout:
                raise RuntimeError("no lowstate -- wrong --iface or robot off")
            time.sleep(0.05)
        print(f"lowstate up, mode_machine={self.mode_machine}")

    def read(self):
        m = self.state
        return (np.array([m.motor_state[i].q for i in range(C.N_JOINTS)], np.float32),
                np.array([m.motor_state[i].dq for i in range(C.N_JOINTS)], np.float32),
                np.array(m.imu_state.gyroscope, np.float32),
                np.array(m.imu_state.quaternion, np.float32))

    def send(self, q, kp, kd):
        self._cmd.mode_pr = C.MODE_PR
        self._cmd.mode_machine = self.mode_machine
        for i in range(C.N_JOINTS):
            c = self._cmd.motor_cmd[i]
            c.mode = C.MOTOR_MODE_ENABLE
            c.q = float(q[i]); c.dq = 0.0; c.tau = 0.0
            c.kp = float(kp[i]); c.kd = float(kd[i])
        if self.armed:
            self._cmd.crc = self._crc.Crc(self._cmd)
            self._pub.Write(self._cmd)

    def damp(self):
        self.send(np.zeros(C.N_JOINTS), np.zeros(C.N_JOINTS),
                  np.full(C.N_JOINTS, DAMP_KD))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default="enP8p1s0")
    ap.add_argument("--policy", default="rl/policies/mels_g1_joystick.npz")
    ap.add_argument("--arm", action="store_true")
    ap.add_argument("--out", default="stand06.jsonl")
    a = ap.parse_args()
    if not a.arm:
        print("DRY RUN -- nothing is published.\n")

    kp = C.TRAIN_KP.copy(); kd = C.TRAIN_KD.copy()
    wrists = [i for i, n in enumerate(C.JOINT_NAMES) if "wrist" in n]
    kp[wrists] = WRIST_KP; kd[wrists] = WRIST_KD
    print(f"wrist kp 2 -> {WRIST_KP:.0f} on {wrists}")
    print(f"slew {SLEW_STEP} rad/step, abort {ABORT_STEP} rad/step, "
          f"tilt abort {MAX_TILT_DEG} deg, ZERO command\n")

    policy, builder = Policy(a.policy), ObservationBuilder()
    robot = Robot(a.iface, a.arm); robot.wait()

    def stop(sig, _f):
        print(f"\nsignal {sig} -> damping")
        for _ in range(15):
            robot.damp(); time.sleep(0.01)
        print("damped. CONFIRM ON CAMERA.")
        sys.exit(0)
    signal.signal(signal.SIGINT, stop); signal.signal(signal.SIGTERM, stop)

    q0, _, _, _ = robot.read()
    print(f"start |q-default|max = {np.abs(q0-C.DEFAULT_POSE).max():.3f} rad")
    prev, n_slew, log = q0.copy(), 0, []
    T_POLICY = RAMP_GAINS_S + RAMP_POSE_S + SETTLE_S
    t0 = time.time()
    try:
        while True:
            tick = time.perf_counter()
            t = time.time() - t0
            if t > T_POLICY + RAMP_ACTION_S + HOLD_S:
                print("\ncompleted full standing hold")
                break
            q, dq, gyro, quat = robot.read()

            if t < RAMP_GAINS_S:
                r = t / RAMP_GAINS_S
                tgt, k, d, stage = q0, kp * r, kd * r, "A gains"
            elif t < RAMP_GAINS_S + RAMP_POSE_S:
                r = (t - RAMP_GAINS_S) / RAMP_POSE_S
                tgt, k, d, stage = (1-r)*q0 + r*C.DEFAULT_POSE, kp, kd, "B pose"
            elif t < T_POLICY:
                tgt, k, d, stage = C.DEFAULT_POSE, kp, kd, "C settle"
            else:
                obs = builder.build(q, dq, gyro, quat, np.zeros(3, np.float32))
                tilt = np.degrees(np.arccos(np.clip(-obs[C.SLICE_GRAVITY][2], -1, 1)))
                if tilt > MAX_TILT_DEG:
                    print(f"\nABORT: tilt {tilt:.1f} deg > {MAX_TILT_DEG}")
                    break
                act = policy(obs).copy()
                builder.set_last_action(act); builder.advance_phase()
                r = min(1.0, (t - T_POLICY) / RAMP_ACTION_S)
                tgt = C.DEFAULT_POSE + r * C.ACTION_SCALE * act
                k, d, stage = kp, kd, "D policy"
                log.append({"t": t, "q": q.tolist(), "action": act.tolist(),
                            "tilt": float(tilt)})

            delta = tgt - prev
            step = float(np.abs(delta).max())
            if step > ABORT_STEP:
                print(f"\nABORT: target jumped {step:.3f} rad (limit {ABORT_STEP})")
                break
            if step > SLEW_STEP:
                delta *= SLEW_STEP / step; tgt = prev + delta; n_slew += 1
            robot.send(tgt, k, d); prev = tgt.copy()

            if int(t * 50) % 50 == 0:
                g = np.degrees(np.arccos(np.clip(
                    -ObservationBuilder().build(q, dq, gyro, quat,
                        np.zeros(3, np.float32))[C.SLICE_GRAVITY][2], -1, 1)))
                print(f"  {stage:9s} t={t:5.1f}s  |q-tgt|max={np.abs(q-tgt).max():.3f}  "
                      f"tilt={g:4.1f}deg  slew={n_slew}")
            time.sleep(max(0.0, C.CTRL_DT - (time.perf_counter() - tick)))
    finally:
        for _ in range(15):
            robot.damp(); time.sleep(0.01)
        print("exited via damping path. CONFIRM ON CAMERA.")
        if log:
            with open(a.out, "w") as f:
                for r_ in log:
                    f.write(json.dumps(r_) + "\n")
            A = np.array([r_["action"] for r_ in log])
            D = np.abs(np.diff(A, axis=0)).max(1)
            print(f"\npolicy ran {len(log)} steps, slew-limited {n_slew}")
            print(f"  |action| mean {np.abs(A).mean():.3f}  max {np.abs(A).max():.3f}")
            if D.size:
                print(f"  step-to-step  p99 {np.percentile(D,99):.3f}  max {D.max():.3f}")
            print(f"  log -> {a.out}")


if __name__ == "__main__":
    main()

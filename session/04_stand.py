"""STEP 4 -- closed loop, ZERO command. The robot stands; it does not walk.

Run ONLY after 01, 02 and 03 are all clean. Requires developer mode (face
light GREEN) and the camera open.

Why standing and not walking:
  * obs[0:3] is base linear velocity and there is no estimator, so it is fed
    as zeros. Suspended and standing, true velocity is ~0, so zeros is the
    CORRECT input, not an approximation. The moment the robot translates,
    that input is wrong by an amount that matters (~0.13 rad mean joint-target
    error at 0.75 m/s).
  * the gantry rules forbid walking any real distance.

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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deploy import constants as C                       # noqa: E402
from deploy.observation import ObservationBuilder       # noqa: E402
from deploy.policy import Policy                        # noqa: E402

RAMP_GAINS_S, RAMP_POSE_S, RAMP_ACTION_S = 2.0, 3.0, 2.0
DAMP_KD = 2.0
MAX_TARGET_STEP = 0.05     # rad per control step; refuse bigger jumps


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
    ap.add_argument("--policy", default="policies/mels_g1_joystick.npz")
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

    q0, _, _, _ = robot.read()
    print(f"start pose max |q-default| = {np.abs(q0-C.DEFAULT_POSE).max():.3f} rad")
    prev = q0.copy()
    t0 = time.time()
    try:
        while True:
            tick = time.perf_counter()
            t = time.time() - t0
            q, dq, gyro, quat = robot.read()

            if t < RAMP_GAINS_S:                       # A: wake gains on q0
                r = t / RAMP_GAINS_S
                target, kp, kd = q0, C.TRAIN_KP * r, C.TRAIN_KD * r
                stage = "A gains"
            elif t < RAMP_GAINS_S + RAMP_POSE_S:       # B: glide to default
                r = (t - RAMP_GAINS_S) / RAMP_POSE_S
                target, kp, kd = (1 - r) * q0 + r * C.DEFAULT_POSE, C.TRAIN_KP, C.TRAIN_KD
                stage = "B pose"
            else:                                      # C: policy, zero cmd
                obs = builder.build(q, dq, gyro, quat, np.zeros(3, np.float32))
                action = policy(obs).copy()
                builder.set_last_action(action); builder.advance_phase()
                r = min(1.0, (t - RAMP_GAINS_S - RAMP_POSE_S) / RAMP_ACTION_S)
                target = C.DEFAULT_POSE + r * C.ACTION_SCALE * action
                kp, kd, stage = C.TRAIN_KP, C.TRAIN_KD, "C policy"

            step = np.abs(target - prev).max()
            if step > MAX_TARGET_STEP:                 # refuse violent jumps
                print(f"\nABORT: target jumped {step:.3f} rad in one step")
                break
            robot.send(target, kp, kd); prev = target.copy()

            if int(t * 50) % 50 == 0:
                print(f"  {stage:8s} t={t:5.1f}s  |tgt-default|max="
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


if __name__ == "__main__":
    main()

"""STEP 1 -- read-only. Log rt/lowstate and introspect the SDK. NO commands sent.

Run this FIRST, before developer mode, before any controller. It publishes
nothing and cannot move the robot; it only subscribes, exactly like the
existing app/bms/real/monitor_battery.py.

What it settles, all of which is currently assumed rather than known:
  * the real motor ordering and each joint's sign and zero offset
  * WHICH IMU `imu_state` reports -- pelvis or torso. The policy needs pelvis.
    The G1 has both, and if imu_state is the torso IMU then gravity AND gyro
    are wrong whenever the waist moves (the default pose already carries
    waist_pitch = 0.073 rad, so there would be a standing offset too).
  * actual message rates
  * the true field names on LowState_ / LowCmd_, printed rather than guessed

    python session/01_log_lowstate.py --iface eth0 --seconds 60 --out limp.jsonl

Then, with the robot hanging limp, move ONE joint at a time by hand and watch
which index changes and in which direction. That is the joint-order, sign and
offset check, and it needs no policy and no developer mode.
"""
import argparse
import json
import time

import numpy as np

N_MOTORS = 29

# Playground actuator order. What we ASSUME the SDK order is; this script is
# how you find out whether that is true.
EXPECTED_ORDER = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee",
    "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee",
    "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]
DEFAULT_POSE = [-0.312, 0, 0, 0.669, -0.363, 0,
                -0.312, 0, 0, 0.669, -0.363, 0,
                0, 0, 0.073,
                0.2, 0.2, 0, 0.6, 0, 0, 0,
                0.2, -0.2, 0, 0.6, 0, 0, 0]


def introspect():
    """Print the ACTUAL SDK message fields. Never guess these."""
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_, LowCmd_
    for name, cls in (("LowState_", LowState_), ("LowCmd_", LowCmd_)):
        fields = getattr(cls, "__annotations__", {})
        print(f"\n--- {name} fields ---")
        for f, t in fields.items():
            print(f"    {f:24s} {t}")
    try:
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import MotorCmd_
        print("\n--- MotorCmd_ fields (what a damping command must fill) ---")
        for f, t in getattr(MotorCmd_, "__annotations__", {}).items():
            print(f"    {f:24s} {t}")
    except ImportError:
        print("\n(MotorCmd_ not importable under that name -- check the IDL "
              "module listing above)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default="eth0")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--out", default="limp.jsonl")
    ap.add_argument("--hz", type=float, default=20.0, help="log rate")
    a = ap.parse_args()

    introspect()

    from unitree_sdk2py.core.channel import (ChannelFactoryInitialize,
                                             ChannelSubscriber)
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

    ChannelFactoryInitialize(0, a.iface)
    box = {"msg": None, "n": 0}

    def on_low(m):
        box["msg"] = m
        box["n"] += 1

    ChannelSubscriber("rt/lowstate", LowState_).Init(on_low, 1)
    print(f"\nsubscribed rt/lowstate on {a.iface}; logging {a.seconds}s "
          f"to {a.out}\nNOTHING IS PUBLISHED -- this cannot move the robot.\n")

    t0 = time.time()
    n_last, t_last = 0, t0
    with open(a.out, "w") as f:
        while time.time() - t0 < a.seconds:
            time.sleep(1.0 / a.hz)
            m = box["msg"]
            if m is None:
                print("  ... no lowstate yet; wrong --iface, or robot off?")
                continue
            q = [float(m.motor_state[i].q) for i in range(N_MOTORS)]
            dq = [float(m.motor_state[i].dq) for i in range(N_MOTORS)]
            tau = [float(m.motor_state[i].tau_est) for i in range(N_MOTORS)]
            imu = m.imu_state
            rec = {
                "t": time.time() - t0,
                "q": q, "dq": dq, "tau_est": tau,
                "quat": [float(v) for v in imu.quaternion],
                "gyro": [float(v) for v in imu.gyroscope],
                "accel": [float(v) for v in imu.accelerometer],
                "rpy": [float(v) for v in imu.rpy],
                "mode_machine": int(getattr(m, "mode_machine", -1)),
                "tick": int(getattr(m, "tick", -1)),
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()

            now = time.time()
            if now - t_last >= 2.0:
                hz = (box["n"] - n_last) / (now - t_last)
                n_last, t_last = box["n"], now
                dev = np.abs(np.array(q) - np.array(DEFAULT_POSE))
                loud = int(np.argmax(np.abs(dq)))
                print(f"  lowstate {hz:6.1f} Hz | quat {np.round(rec['quat'],3)} "
                      f"| largest |dq| joint {loud} ({EXPECTED_ORDER[loud]}) "
                      f"{rec['dq'][loud]:+.3f} rad/s "
                      f"| max |q-default| {dev.max():.3f} rad")
    print(f"\ndone: {box['n']} messages, log -> {a.out}")


if __name__ == "__main__":
    main()

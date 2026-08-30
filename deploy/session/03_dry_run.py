"""STEP 3 -- policy runs on real telemetry, sends NOTHING. No publisher at all.

This is the last check before closing the loop, and it is safe enough to run
with the robot in NORMAL mode with the built-in controller still holding it
upright -- there is no publisher in this file, so it cannot move anything.
Running it that way is actually the most useful version: the robot is in a
real standing pose, so the observation should look like the sim's standing
observation, and the policy's action should be small.

    python session/03_dry_run.py --iface eth0 --seconds 60 --out dryrun.jsonl

What to look for, in order of how badly it bites:
  * gravity ~ [0, 0, -1] while the robot is upright. If it is not, imu_state
    is not what we think it is, or the quaternion convention differs.
  * |q - default_pose| small in a normal standing pose. Large means a joint
    ordering, sign or zero-offset mismatch -- stop and use 01's hand sweep.
  * action magnitude comparable to the sim's standing value (~0.37 rad of
    target deviation). Wildly larger means the policy is seeing nonsense.
  * loop timing p99 well under 20 ms.

Nothing here is authoritative about whether the robot will WALK. It only
proves the robot's telemetry maps correctly onto the policy's input.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from deploy import constants as C                       # noqa: E402
from deploy.observation import ObservationBuilder       # noqa: E402
from deploy.policy import Policy                        # noqa: E402

SIM_STANDING_DEV = 0.367   # rad, from tests/test_deploy.py golden value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default="eth0")
    ap.add_argument("--policy", default="rl/policies/mels_g1_joystick.npz")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--out", default="dryrun.jsonl")
    ap.add_argument("--command", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                    help="vx vy wyaw -- LEAVE AT ZERO for the first run")
    a = ap.parse_args()

    policy = Policy(a.policy)
    builder = ObservationBuilder()
    print(f"policy {a.policy}: obs_dim={policy.obs_dim} "
          f"uses_linvel={policy.uses_linvel}")
    if policy.uses_linvel:
        print("  NOTE: linvel is stubbed to zeros. Correct while standing "
              "(true velocity ~ 0), wrong once moving.")

    from unitree_sdk2py.core.channel import (ChannelFactoryInitialize,
                                             ChannelSubscriber)
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    ChannelFactoryInitialize(0, a.iface)
    box = {"msg": None}
    ChannelSubscriber("rt/lowstate", LowState_).Init(
        lambda m: box.__setitem__("msg", m), 1)

    print("\nNO PUBLISHER IN THIS PROCESS -- it cannot move the robot.\n")
    while box["msg"] is None:
        print("  waiting for lowstate...")
        time.sleep(0.5)

    cmd = np.array(a.command, dtype=np.float32)
    t0, dts, n = time.time(), [], 0
    with open(a.out, "w") as f:
        while time.time() - t0 < a.seconds:
            tick = time.perf_counter()
            m = box["msg"]
            q = np.array([m.motor_state[i].q for i in range(C.N_JOINTS)], np.float32)
            dq = np.array([m.motor_state[i].dq for i in range(C.N_JOINTS)], np.float32)
            obs = builder.build(q, dq, np.array(m.imu_state.gyroscope, np.float32),
                                np.array(m.imu_state.quaternion, np.float32), cmd)
            action = policy(obs).copy()
            targets = policy.motor_targets(action)
            builder.set_last_action(action)
            builder.advance_phase()
            dts.append(time.perf_counter() - tick)
            n += 1

            f.write(json.dumps({
                "t": time.time() - t0,
                "q": q.tolist(), "dq": dq.tolist(),
                "quat": [float(v) for v in m.imu_state.quaternion],
                "gravity": obs[C.SLICE_GRAVITY].tolist(),
                "action": action.tolist(),
                "would_send_targets": targets.tolist(),
            }) + "\n")

            if n % 50 == 0:
                g = obs[C.SLICE_GRAVITY]
                dev = float(np.abs(targets - C.DEFAULT_POSE).max())
                jdev = float(np.abs(q - C.DEFAULT_POSE).max())
                print(f"  t={time.time()-t0:5.1f}s  gravity={np.round(g,3)} "
                      f"(want ~[0,0,-1])  |q-default|max={jdev:.3f}  "
                      f"action dev={dev:.3f} (sim standing {SIM_STANDING_DEV:.3f})  "
                      f"p99={np.percentile(dts,99)*1e3:.2f}ms")
            time.sleep(max(0.0, C.CTRL_DT - (time.perf_counter() - tick)))

    d = np.array(dts) * 1e3
    print(f"\n{n} steps. loop p50={np.percentile(d,50):.2f}ms "
          f"p99={np.percentile(d,99):.2f}ms max={d.max():.2f}ms "
          f"(budget {C.CTRL_DT*1e3:.0f}ms)\nlog -> {a.out}")


if __name__ == "__main__":
    main()

"""Climb mime on the real Unitree G1 (29 DoF): built-in locomotion + scripted right arm.

    python -m rl.deterministic.mime <network_iface>            # e.g. eth0
    python -m rl.deterministic.mime <iface> --cycles 5 --no-walk

Robot must be standing in the built-in balance controller (R1+X / app) first.
Arm goes through `rt/arm_sdk` (Unitree's arm overlay: the loco controller keeps
running), legs through `LocoClient.Move`. Fake ascender telemetry (engaged,
tension_N) is printed as JSON per phase for the app/BMS bridge.

SAFETY: first run on the gantry. Ctrl-C ramps the arm back to the controller in 2 s.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from rl.deterministic import choreo

# SDK joint indices for the right arm (unitree_sdk2 G1JointIndex).
RIGHT_ARM_IDX = [22, 23, 24, 25, 26, 27, 28]
WAIST_PITCH_IDX = 14  # 29-DoF G1 only (locked-waist models: set to None)
NOT_USED_JOINT = 29  # motor_cmd[29].q = arm_sdk weight (1 = SDK owns the arms)
KP, KD = 60.0, 1.5
DT = 0.02


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("iface")
  p.add_argument("--cycles", type=int, default=4)
  p.add_argument("--no-walk", action="store_true", help="arm only, robot stands still")
  p.add_argument("--speed", type=float, default=choreo.WALK_SPEED)
  args = p.parse_args()

  from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
  from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
  from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
  from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
  from unitree_sdk2py.utils.crc import CRC

  ChannelFactoryInitialize(0, args.iface)
  state = {"msg": None}
  sub = ChannelSubscriber("rt/lowstate", LowState_)
  sub.Init(lambda msg: state.__setitem__("msg", msg), 10)
  pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
  pub.Init()
  loco = LocoClient()
  loco.SetTimeout(10.0)
  loco.Init()
  crc = CRC()
  cmd = unitree_hg_msg_dds__LowCmd_()
  while state["msg"] is None:
    time.sleep(0.1)
  print("lowstate OK", file=sys.stderr)

  def send(arm, weight):
    cmd.motor_cmd[NOT_USED_JOINT].q = weight
    targets = list(zip(RIGHT_ARM_IDX, arm))
    if WAIST_PITCH_IDX is not None:
      targets.append((WAIST_PITCH_IDX, choreo.WAIST_PITCH_LEAN * weight))
    for j, q in targets:
      mc = cmd.motor_cmd[j]
      mc.q, mc.dq, mc.tau, mc.kp, mc.kd = float(q), 0.0, 0.0, KP, KD
    cmd.crc = crc.Crc(cmd)
    pub.Write(cmd)

  q_now = lambda: np.array([state["msg"].motor_state[j].q for j in RIGHT_ARM_IDX])
  walking = False
  yaw0 = state["msg"].imu_state.rpy[2]  # heading at start = "uphill"
  try:
    # Ramp 2 s from the current arm pose to POSE_HIP while taking over the arm.
    q0 = q_now()
    for i in range(int(2.0 / DT)):
      s = choreo._smooth(i * DT / 2.0)
      send(choreo._lerp(q0, choreo.POSE_HIP, s), s)
      time.sleep(DT)
    t0 = time.time()
    last = None
    while time.time() - t0 < args.cycles * choreo.PERIOD_S:
      c = choreo.step(time.time() - t0)
      send(c["arm"], 1.0)
      yaw_rate = choreo.yaw_rate_cmd(state["msg"].imu_state.rpy[2], yaw0)
      if not args.no_walk:  # continuous shuffle, faster during the pull; always face uphill
        loco.Move(c["speed"] * args.speed / choreo.WALK_SPEED, 0.0, yaw_rate)
        walking = True
      want = c["walk"]
      if c["phase"] != last:
        last = c["phase"]
        print(json.dumps({"t_ms": int((time.time() - t0) * 1000), "phase": c["phase"],
                          "engaged": c["engaged"], "tension_N": c["tension_N"], "walk": want}))
      time.sleep(DT)
  finally:
    if walking:
      loco.Move(0.0, 0.0, 0.0)
    # Release the arm back to the controller over 2 s.
    q0 = q_now()
    for i in range(int(2.0 / DT)):
      s = i * DT / 2.0
      send(q0, 1.0 - s)
      time.sleep(DT)
    print("arm released", file=sys.stderr)


if __name__ == "__main__":
  main()

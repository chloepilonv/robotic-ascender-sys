"""Does the loco FSM accept a walk-ready transition?

Start() on the installed SDK is SetFsmId(200) -- balance-stand / walk-ready.
rope_walk.py called it and threw the return code away, which is why it printed
a clean run while the robot never moved. This calls it and SHOWS the code.

  code 0     -> accepted. rope_walk.py should now be able to walk.
  code != 0  -> refused. That number is the answer; the robot likely needs to be
                put in a standing stance from the physical remote (L2+A, R2+B).

THIS IS A STATE CHANGE: FSM 200 is balance-stand, so the robot may take its own
weight and shift its feet. Camera up before running.
"""
import math
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

ChannelFactoryInitialize(0, "enP8p1s0")
box = {"m": None}
ChannelSubscriber("rt/lowstate", LowState_).Init(lambda m: box.__setitem__("m", m), 1)
for _ in range(50):
    time.sleep(0.1)
    if box["m"]:
        break

def snapshot(tag):
    m = box["m"]
    r, p, y = m.imu_state.rpy
    knees = (m.motor_state[3].q, m.motor_state[9].q)
    tau = max(abs(m.motor_state[i].tau_est) for i in range(12))
    print(f"  {tag:8s} rpy=({math.degrees(r):+6.1f},{math.degrees(p):+6.1f},"
          f"{math.degrees(y):+7.1f})  knees=({knees[0]:+.3f},{knees[1]:+.3f})  "
          f"max|tau_leg|={tau:5.2f} N.m")

ms = MotionSwitcherClient(); ms.SetTimeout(5.0); ms.Init()
print("active motion service:", ms.CheckMode())
print()
snapshot("before")

c = LocoClient(); c.SetTimeout(5.0); c.Init()
print("\ncalling SetFsmId(200)  [balance-stand / walk-ready] ...")
code = c.SetFsmId(200)
print(f"  -> return code: {code}   {'ACCEPTED' if code == 0 else 'REFUSED'}")

for i in range(6):
    time.sleep(0.5)
snapshot("after")
print("\nIf the code is 0 and the legs now carry torque, the FSM was the blocker.")
print("If the code is 0 but nothing changed, the service accepted the message")
print("without acting -- the robot probably needs a standing stance set on the")
print("physical remote (L2+A then R2+B) before it will locomote.")

"""Put the robot into an ACTIVE standing stance via the loco API.

The robot is on the ground but passive -- left knee locked straight, right bent,
near-zero leg torque. SetFsmId(200) is accepted and ignored because the service
will not locomote a robot that is not actively standing. These calls exist to
fix exactly that.

ONE action per run, chosen explicitly. Each prints leg torque and knee angles
before and after, so "did anything happen" is a measurement, not a guess.

  python3 stand_test.py balance     # BalanceStand(0) -- set balance mode, gentlest
  python3 stand_test.py standup     # Squat2StandUp() -- stand up, REAL MOTION
  python3 stand_test.py lowstand    # LowStand()      -- low standing posture
  python3 stand_test.py highstand   # HighStand()     -- taller standing posture

CAMERA UP. 'standup' in particular commands a whole-body motion.
Recovery if anything looks wrong: `robot normal`, or the admin killswitch.
"""
import math
import sys
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

ACTIONS = {
    "balance":   ("BalanceStand(0)", lambda c: c.BalanceStand(0)),
    "standup":   ("Squat2StandUp()", lambda c: c.Squat2StandUp()),
    "lowstand":  ("LowStand()",      lambda c: c.LowStand()),
    "highstand": ("HighStand()",     lambda c: c.HighStand()),
}
if len(sys.argv) < 2 or sys.argv[1] not in ACTIONS:
    sys.exit("usage: stand_test.py {%s}" % "|".join(ACTIONS))
label, fn = ACTIONS[sys.argv[1]]

ChannelFactoryInitialize(0, "enP8p1s0")
box = {"m": None}
ChannelSubscriber("rt/lowstate", LowState_).Init(lambda m: box.__setitem__("m", m), 1)
for _ in range(50):
    time.sleep(0.1)
    if box["m"]:
        break

def snap(tag):
    m = box["m"]
    r, p, y = m.imu_state.rpy
    tau = [abs(float(m.motor_state[i].tau_est)) for i in range(12)]
    print(f"  {tag:7s} knees=({m.motor_state[3].q:+.3f},{m.motor_state[9].q:+.3f})  "
          f"max|tau_leg|={max(tau):6.2f}  sum|tau_leg|={sum(tau):7.2f} N.m  "
          f"rpy=({math.degrees(r):+5.1f},{math.degrees(p):+5.1f},{math.degrees(y):+6.1f})")

c = LocoClient(); c.SetTimeout(10.0); c.Init()
snap("before")
print(f"\ncalling {label} ...")
code = fn(c)
print(f"  -> code {code}   {'ACCEPTED' if code == 0 else 'REFUSED'}")
for i in range(1, 7):
    time.sleep(1.0)
    snap(f"t+{i}s")
print("\nsum|tau_leg| rising from ~1 N.m means the robot is now actively holding itself.")

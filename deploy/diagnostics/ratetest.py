"""Does the command RATE decide whether motors enable?

Publishes a HOLD-CURRENT-POSE command (target = measured q, so the robot has
nowhere to go) first at 50 Hz, then at 500 Hz, watching motor_state[].mode.
mode 1 = enabled, 0 = disabled. Damps on exit and on any signal.

Safe by construction: the target is wherever the robot already is.
"""
import signal, sys, time
import numpy as np
from unitree_sdk2py.core.channel import (ChannelFactoryInitialize,
                                         ChannelPublisher, ChannelSubscriber)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.utils.crc import CRC

N = 29
KP = np.array([75.,75.,75.,75.,20.,2., 75.,75.,75.,75.,20.,2., 75.,75.,75.,
               75.,75.,75.,75.,2.,2.,2., 75.,75.,75.,75.,2.,2.,2.])
KD = np.array([2.,2.,2.,2.,1.,.2, 2.,2.,2.,2.,1.,.2, 2.,2.,2.,
               2.,2.,2.,2.,.2,.2,.2, 2.,2.,2.,2.,.2,.2,.2])
ARMED = "--arm" in sys.argv

ChannelFactoryInitialize(0, "enP8p1s0")
box = {"m": None}
ChannelSubscriber("rt/lowstate", LowState_).Init(lambda m: box.__setitem__("m", m), 1)
pub = ChannelPublisher("rt/lowcmd", LowCmd_); pub.Init()
crc = CRC(); cmd = unitree_hg_msg_dds__LowCmd_()
while box["m"] is None:
    time.sleep(0.05)
mm = int(box["m"].mode_machine)
print("mode_machine=%d  armed=%s" % (mm, ARMED))

def send(q, kp, kd):
    cmd.mode_pr = 0; cmd.mode_machine = mm
    for i in range(N):
        c = cmd.motor_cmd[i]
        c.mode = 1; c.q = float(q[i]); c.dq = 0.0; c.tau = 0.0
        c.kp = float(kp[i]); c.kd = float(kd[i])
    if ARMED:
        cmd.crc = crc.Crc(cmd); pub.Write(cmd)

def damp():
    for _ in range(20):
        send(np.zeros(N), np.zeros(N), np.full(N, 8.0)); time.sleep(0.005)

def bail(sig, _f):
    print("\nsignal %d -> damping" % sig); damp(); sys.exit(0)
signal.signal(signal.SIGINT, bail); signal.signal(signal.SIGTERM, bail)

def phase(hz, secs):
    q_hold = np.array([box["m"].motor_state[i].q for i in range(N)])
    dt, t0, n = 1.0 / hz, time.time(), 0
    modes, taus = [], []
    while time.time() - t0 < secs:
        send(q_hold, KP, KD); n += 1
        m = box["m"]
        modes.append(max(int(m.motor_state[i].mode) for i in range(N)))
        taus.append(max(abs(float(m.motor_state[i].tau_est)) for i in range(N)))
        time.sleep(dt)
    print("  %4d Hz | %5d frames | max motor.mode seen = %d | max |tau| = %.2f N.m"
          % (hz, n, max(modes), max(taus)))
    return max(modes)

try:
    print("holding CURRENT pose (target = measured q, nowhere to move)")
    a = phase(50, 3.0)
    b = phase(500, 3.0)
    print("\n50 Hz -> motors %s | 500 Hz -> motors %s"
          % ("ENABLED" if a == 1 else "disabled", "ENABLED" if b == 1 else "disabled"))
finally:
    if ARMED:
        damp()
    print("damped, exiting")

"""Which joints can actually reach their commanded target?

Ramps from the current pose to DEFAULT_POSE over 8 s with the TRAINING gains,
holds 5 s, then reports PER-JOINT tracking error and torque. Same motion the
robot has already done in stage B -- the difference is the reporting.

Damps on exit and on any signal. --arm required to publish.
"""
import signal, sys, time
import numpy as np
from unitree_sdk2py.core.channel import (ChannelFactoryInitialize,
                                         ChannelPublisher, ChannelSubscriber)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.utils.crc import CRC

N = 29
NAMES = ["L_hip_p","L_hip_r","L_hip_y","L_knee","L_ank_p","L_ank_r",
         "R_hip_p","R_hip_r","R_hip_y","R_knee","R_ank_p","R_ank_r",
         "waist_y","waist_r","waist_p",
         "L_sh_p","L_sh_r","L_sh_y","L_elb","L_wr_r","L_wr_p","L_wr_y",
         "R_sh_p","R_sh_r","R_sh_y","R_elb","R_wr_r","R_wr_p","R_wr_y"]
DEFAULT = np.array([-0.312,0,0,0.669,-0.363,0, -0.312,0,0,0.669,-0.363,0,
                    0,0,0.073, 0.2,0.2,0,0.6,0,0,0, 0.2,-0.2,0,0.6,0,0,0])
KP = np.array([75.,75.,75.,75.,20.,2., 75.,75.,75.,75.,20.,2., 75.,75.,75.,
               75.,75.,75.,75.,2.,2.,2., 75.,75.,75.,75.,2.,2.,2.])
KD = np.array([2.,2.,2.,2.,1.,.2, 2.,2.,2.,2.,1.,.2, 2.,2.,2.,
               2.,2.,2.,2.,.2,.2,.2, 2.,2.,2.,2.,.2,.2,.2])
ARMED = "--arm" in sys.argv
RAMP, HOLD, DT = 8.0, 5.0, 0.02

ChannelFactoryInitialize(0, "enP8p1s0")
box = {"m": None}
ChannelSubscriber("rt/lowstate", LowState_).Init(lambda m: box.__setitem__("m", m), 1)
pub = ChannelPublisher("rt/lowcmd", LowCmd_); pub.Init()
crc = CRC(); cmd = unitree_hg_msg_dds__LowCmd_()
while box["m"] is None:
    time.sleep(0.05)
mm = int(box["m"].mode_machine)
print("mode_machine=%d armed=%s" % (mm, ARMED))

def read():
    m = box["m"]
    return (np.array([m.motor_state[i].q for i in range(N)]),
            np.array([m.motor_state[i].tau_est for i in range(N)]))

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

def bail(s, _f):
    print("\nsignal %d -> damping" % s); damp(); sys.exit(0)
signal.signal(signal.SIGINT, bail); signal.signal(signal.SIGTERM, bail)

q0, _ = read()
print("start |q-default|max = %.3f rad" % np.abs(q0 - DEFAULT).max())
try:
    t0 = time.time()
    while True:
        t = time.time() - t0
        if t > RAMP + HOLD: break
        r = min(1.0, t / RAMP)
        send((1 - r) * q0 + r * DEFAULT, KP, KD)
        time.sleep(DT)
    q, tau = read()
    err = q - DEFAULT
    print("\nAFTER %ds hold at DEFAULT_POSE, with the motors driven:" % HOLD)
    print("%3s %-9s %5s %8s %8s %8s" % ("idx","joint","kp","q","err","tau"))
    for i in np.argsort(-np.abs(err))[:12]:
        print("%3d %-9s %5.0f %8.3f %+8.3f %+8.2f" % (i, NAMES[i], KP[i], q[i], err[i], tau[i]))
    legs = np.abs(err[:12]); low = KP < 10
    print("\n  LEGS  (idx 0-11)      max err %.3f  mean %.3f" % (legs.max(), legs.mean()))
    print("  kp>=20 joints         max err %.3f" % np.abs(err[~low]).max())
    print("  kp==2  joints         max err %.3f  <- expect large, gravity beats kp=2" % np.abs(err[low]).max())
    print("  within 0.15 rad       %d/29  (kp>=20 only: %d/%d)"
          % ((np.abs(err) < .15).sum(), (np.abs(err[~low]) < .15).sum(), (~low).sum()))
finally:
    if ARMED: damp()
    print("damped, exiting")

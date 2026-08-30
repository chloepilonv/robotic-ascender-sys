"""SHADOW MODE: hold the robot at DEFAULT_POSE, run the policy, apply NOTHING.

The robot is driven open-loop to a fixed pose (the same hold tracktest just did
safely). The policy runs on the real observation each step, and its output is
LOGGED but never sent. So we can see whether the policy is well-behaved on real
hardware without letting it touch the joints.

This answers the question the aborts could not: is the policy's output smooth
and small in a realistic standing observation, or is it inherently oscillating?
"""
import signal, sys, time
import numpy as np
sys.path.insert(0, "/home/unitree")
from deploy import constants as C
from deploy.observation import ObservationBuilder
from deploy.policy import Policy
from unitree_sdk2py.core.channel import (ChannelFactoryInitialize,
                                         ChannelPublisher, ChannelSubscriber)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.utils.crc import CRC

N, RAMP, WATCH, DT = C.N_JOINTS, 8.0, 12.0, C.CTRL_DT
ARMED = "--arm" in sys.argv

# Sim-to-real gain correction. TRAIN_KP has kp=2 on the six wrist joints, which
# holds fine in simulation (negligible wrist inertia) but loses to gravity on
# hardware -- the wrist droops ~0.45 rad, which is 5 sigma outside the policy's
# training distribution for that input. Raising it is not a control change that
# matters (wrists carry no load and do not affect balance); it is an OBSERVATION
# fix, putting joint_pos back where the policy expects it.
WRIST_KP = float(next((a.split("=")[1] for a in sys.argv if a.startswith("--wrist-kp=")), 2.0))
KP = C.TRAIN_KP.copy(); KD = C.TRAIN_KD.copy()
WRISTS = [i for i, n in enumerate(C.JOINT_NAMES) if "wrist" in n]
if WRIST_KP != 2.0:
    KP[WRISTS] = WRIST_KP
    KD[WRISTS] = 1.0
    print("wrist kp 2 -> %.0f on joints %s" % (WRIST_KP, WRISTS))
ChannelFactoryInitialize(0, "enP8p1s0")
box = {"m": None}
ChannelSubscriber("rt/lowstate", LowState_).Init(lambda m: box.__setitem__("m", m), 1)
pub = ChannelPublisher("rt/lowcmd", LowCmd_); pub.Init()
crc = CRC(); cmd = unitree_hg_msg_dds__LowCmd_()
while box["m"] is None: time.sleep(0.05)
mm = int(box["m"].mode_machine)
print("mode_machine=%d armed=%s  (policy output is NEVER applied)" % (mm, ARMED))

def read():
    m = box["m"]
    return (np.array([m.motor_state[i].q for i in range(N)], np.float32),
            np.array([m.motor_state[i].dq for i in range(N)], np.float32),
            np.array(m.imu_state.gyroscope, np.float32),
            np.array(m.imu_state.quaternion, np.float32))

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

policy, builder = Policy("rl/policies/mels_g1_joystick.npz"), ObservationBuilder()
q0, _, _, _ = read()
acts, dacts, zs, prev = [], [], [], None
try:
    t0 = time.time()
    while True:
        t = time.time() - t0
        if t > RAMP + WATCH: break
        r = min(1.0, t / RAMP)
        send((1 - r) * q0 + r * C.DEFAULT_POSE, KP, KD)
        if t > RAMP:                                  # shadow the policy
            q, dq, gyro, quat = read()
            obs = builder.build(q, dq, gyro, quat, np.zeros(3, np.float32))
            a = policy(obs).copy()
            builder.set_last_action(a); builder.advance_phase()
            acts.append(a)
            zs.append(np.abs((obs - policy._mu) / policy._sd))
            if prev is not None: dacts.append(np.abs(a - prev).max())
            prev = a
        time.sleep(DT)
    A = np.array(acts); D = np.array(dacts)
    print("\nPOLICY OUTPUT over %d steps at a real standing observation:" % len(A))
    print("  |action|      mean %.3f  p99 %.3f  max %.3f" % (np.abs(A).mean(), np.percentile(np.abs(A),99), np.abs(A).max()))
    print("  step-to-step  mean %.3f  p99 %.3f  max %.3f  <- 0.30 aborts an armed run" % (D.mean(), np.percentile(D,99), D.max()))
    print("  implied joint-target step (x0.5): max %.3f  <- vs 0.05 slew / 0.30 abort" % (0.5*D.max()))
    j = int(np.abs(A).mean(0).argmax())
    print("  most active joint: %s (mean |a| %.3f)" % (C.JOINT_NAMES[j], np.abs(A).mean(0)[j]))
    Z = np.array(zs)                       # (steps, 103)
    zm = Z.mean(0)
    print("  obs |z| vs training dist: mean %.2f  max %.2f" % (Z.max(1).mean(), Z.max()))
    groups = [("linvel",C.SLICE_LINVEL),("gyro",C.SLICE_GYRO),("gravity",C.SLICE_GRAVITY),
              ("command",C.SLICE_COMMAND),("joint_pos",C.SLICE_JOINT_POS),
              ("joint_vel",C.SLICE_JOINT_VEL),("last_act",C.SLICE_LAST_ACT),
              ("phase",C.SLICE_PHASE)]
    print("\n  worst observation dimensions still out of distribution:")
    for i in np.argsort(-zm)[:8]:
        g = next(n for n, sl in groups if sl.start <= i < sl.stop)
        sl = next(sl for n, sl in groups if sl.start <= i < sl.stop)
        lbl = C.JOINT_NAMES[i - sl.start] if g in ("joint_pos","joint_vel","last_act") else "[%d]" % (i - sl.start)
        kp = C.TRAIN_KP[i - sl.start] if g == "joint_pos" else float("nan")
        print("    dim %3d %-10s %-22s |z|=%5.2f  train_kp=%s"
              % (i, g, lbl, zm[i], ("%.0f" % kp) if kp == kp else "-"))
finally:
    if ARMED: damp()
    print("damped, exiting")

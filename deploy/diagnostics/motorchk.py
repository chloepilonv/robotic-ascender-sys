import time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
ChannelFactoryInitialize(0, "enP8p1s0")
box = {"m": None}
ChannelSubscriber("rt/lowstate", LowState_).Init(lambda m: box.__setitem__("m", m), 1)
for _ in range(50):
    time.sleep(0.1)
    if box["m"]:
        break
m = box["m"]
print("mode_machine=%d  mode_pr=%d  tick=%d" % (m.mode_machine, m.mode_pr, m.tick))
names = ["L_hip_p","L_hip_r","L_hip_y","L_knee","L_ank_p","L_ank_r",
         "R_hip_p","R_hip_r","R_hip_y","R_knee","R_ank_p","R_ank_r",
         "waist_y","waist_r","waist_p",
         "L_sh_p","L_sh_r","L_sh_y","L_elb","L_wr_r","L_wr_p","L_wr_y",
         "R_sh_p","R_sh_r","R_sh_y","R_elb","R_wr_r","R_wr_p","R_wr_y"]
hdr = ("idx", "joint", "mode", "motorstate", "tempC", "volt", "tau")
print("%3s %9s %5s %11s %6s %6s %7s" % hdr)
bad = 0
for i in range(29):
    ms = m.motor_state[i]
    st, mo = int(ms.motorstate), int(ms.mode)
    if st != 0:
        bad += 1
    print("%3d %9s %5d %11d %6d %6.1f %+7.2f%s" % (
        i, names[i], mo, st, int(ms.temperature[0]), float(ms.vol),
        float(ms.tau_est), "  <-- FAULT" if st else ""))
print("\nmotors with nonzero motorstate: %d/29" % bad)
print("distinct motor .mode values:", sorted({int(m.motor_state[i].mode) for i in range(29)}))

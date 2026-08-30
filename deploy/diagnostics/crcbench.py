import time
import numpy as np
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.utils.crc import CRC
crc = CRC(); cmd = unitree_hg_msg_dds__LowCmd_()
cmd.mode_pr = 0; cmd.mode_machine = 5
for i in range(29):
    c = cmd.motor_cmd[i]
    c.mode, c.q, c.dq, c.tau, c.kp, c.kd = 1, 0.1, 0.0, 0.0, 75.0, 2.0
def fill():
    for i in range(29):
        c = cmd.motor_cmd[i]
        c.mode = 1; c.q = 0.1; c.dq = 0.0; c.tau = 0.0; c.kp = 75.0; c.kd = 2.0
for _ in range(50): crc.Crc(cmd); fill()
def bench(fn, n):
    t = time.perf_counter()
    for _ in range(n): fn()
    return (time.perf_counter() - t) / n * 1e3
n = 300
t_fill = bench(fill, n)
t_crc  = bench(lambda: crc.Crc(cmd), n)
print("fill 29 motor_cmd fields : %.3f ms" % t_fill)
print("CRC over LowCmd_         : %.3f ms" % t_crc)
print("fill + CRC               : %.3f ms" % (t_fill + t_crc))
print()
for hz in (50, 200, 333, 500):
    budget = 1000.0 / hz
    used = (t_fill + t_crc) / budget * 100
    print("  %3d Hz -> %5.2f ms budget, publish path uses %5.1f%%  %s"
          % (hz, budget, used, "FEASIBLE" if used < 60 else "TOO SLOW"))

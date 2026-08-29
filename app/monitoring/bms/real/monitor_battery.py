"""REAL G1 monitor: battery + motors + IMU + mainboard over DDS (unitree_sdk2py).

    python app/monitoring/real/monitor_battery.py --iface eth0 --log app/monitoring/real/log.jsonl

Subscribes:
    rt/lowstate           unitree_hg LowState_      ~500 Hz  motors, IMU, tick
    rt/lf/bmsstate        unitree_hg BmsState_      low freq battery (measured, printed as hz)
    rt/lf/mainboardstate  unitree_hg MainBoardState_ low freq fans/temps
Prints a 1 Hz table + measured Hz per topic, logs one JSON line per second.
NOTE: hg LowState_ has NO bms_state field (that's the Go2 IDL) -> bms/monitor.py is wrong.
"""
import argparse, json, sys, threading, time
from pathlib import Path

import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_, BmsState_, MainBoardState_

sys.path.insert(0, str(Path(__file__).parent))
from derived import EnergyEstimator, bms_stats, imu_stats, motor_stats  # noqa: E402

N_MOTORS = 29


class Rate:
    """Measure how often a callback fires (Hz)."""
    def __init__(self): self.n, self.t0 = 0, time.time()
    def tick(self): self.n += 1
    def hz(self):
        now = time.time(); hz = self.n / max(now - self.t0, 1e-6); self.n, self.t0 = 0, now; return hz


class Monitor:
    def __init__(self, bms_topic, mb_topic):
        self.lock = threading.Lock()
        self.low, self.bms, self.mb = None, None, None
        self.r_low, self.r_bms, self.r_mb = Rate(), Rate(), Rate()
        self.energy = EnergyEstimator()
        ChannelSubscriber("rt/lowstate", LowState_).Init(self._on_low, 1)
        ChannelSubscriber(bms_topic, BmsState_).Init(self._on_bms, 1)
        ChannelSubscriber(mb_topic, MainBoardState_).Init(self._on_mb, 1)

    def _on_low(self, m): self.r_low.tick(); self.low = m
    def _on_bms(self, m): self.r_bms.tick(); self.bms = m
    def _on_mb(self, m):  self.r_mb.tick();  self.mb = m

    def snapshot(self) -> dict:
        t = time.time()
        out = {"t": t, "hz_lowstate": self.r_low.hz(), "hz_bms": self.r_bms.hz(), "hz_mainboard": self.r_mb.hz()}
        if self.low:
            ms = self.low.motor_state[:N_MOTORS]
            out.update(motor_stats(
                q=[m.q for m in ms], dq=[m.dq for m in ms], tau=[m.tau_est for m in ms],
                temp_winding=[m.temperature[0] for m in ms], temp_board=[m.temperature[1] for m in ms],
                vol=[m.vol for m in ms]))
            out["motor_error_mask"] = int(sum(1 for m in ms if m.motorstate != 0))
            out["tick_ms"] = int(self.low.tick)
            i = self.low.imu_state
            out.update(imu_stats(i.gyroscope, i.accelerometer, i.rpy, i.temperature))
        if self.bms:
            b = self.bms
            out.update(bms_stats(b.cell_vol, b.bmsvoltage, b.current, b.soc, b.soh, b.temperature, b.cycle))
            out.update(self.energy.update(t, out["pack_V"], out["current_A"]))
            out["time_to_empty_min"] = EnergyEstimator.time_to_empty_min(
                out["soc_pct"], out["power_avg_W"], out["bms_temp_min_C"])
        if self.mb:
            out["mainboard_temp_max_C"] = int(max(self.mb.temperature))
            out["fan_rpm"] = [int(f) for f in self.mb.fan_state]
        return out


def fmt(s: dict) -> str:
    g = lambda k, f="{:.1f}": f.format(s[k]) if k in s else "--"
    return (f"low {g('hz_lowstate','{:.0f}')}Hz bms {g('hz_bms')}Hz | "
            f"SOC {g('soc_pct','{}')}% {g('pack_V','{:.2f}')}V {g('current_A','{:+.2f}')}A "
            f"{g('power_avg_W','{:.0f}')}W  TTE {g('time_to_empty_min','{:.0f}')}min "
            f"Tbat {g('bms_temp_min_C','{}')}C f={g('capacity_factor','{:.2f}')} | "
            f"cell {g('cell_min_V','{:.3f}')}-{g('cell_max_V','{:.3f}')}V | "
            f"mech {g('mech_power_W','{:.0f}')}W  maxT {g('max_temp_winding_C','{}')}C j{g('hot_joint','{}')} "
            f"maxTau {g('max_abs_tau_Nm')}Nm | pitch {g('pitch_deg')}°")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--iface", default="eth0")
    p.add_argument("--bms_topic", default="rt/lf/bmsstate")
    p.add_argument("--mb_topic", default="rt/lf/mainboardstate")
    p.add_argument("--log", default="app/monitoring/real/log.jsonl")
    p.add_argument("--period", type=float, default=1.0, help="print/log period in s")
    a = p.parse_args()
    ChannelFactoryInitialize(0, a.iface)
    mon = Monitor(a.bms_topic, a.mb_topic)
    with open(a.log, "a") as f:
        while True:
            time.sleep(a.period)
            s = mon.snapshot()
            print(fmt(s)); f.write(json.dumps(s) + "\n"); f.flush()

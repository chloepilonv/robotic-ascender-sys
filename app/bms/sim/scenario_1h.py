"""Accelerated 1 h Himalaya duty cycle through battery_model.py — no MuJoCo.

Synthetic tau/dq at realistic levels (climb ~463 W electrical, idle ~63 W),
5300 m / outside -24 C / 30 km/h wind, pack starts at +15 C (warm tent).
Prints a 5-min table for the jacketed pack (JACKET_PCT as shipped) and a
no-jacket control. Rerun after calibrating the constants from real logs.

    python app/bms/sim/scenario_1h.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "real"))
import battery_model                                          # noqa: E402
from battery_model import BatteryThermalModel, Environment    # noqa: E402
from derived import EnergyEstimator                           # noqa: E402

N_MOTORS, DT = 29, 0.05
# (name, minutes, tau N.m per joint, dq rad/s per joint)
# climb: P_mech = 29*12*0.6 = 209 W -> ~463 W electrical (hard uphill walk)
# idle:  pose hold, dq 0 -> ~63 W (the Jetson + sensors keep drawing)
PHASES = [("setup idle", 10, 2.0, 0.0), ("climb", 25, 12.0, 0.6),
          ("rest", 10, 2.0, 0.0), ("climb", 15, 12.0, 0.6)]


def run(label):
    env = Environment(altitude_m=5300, wind_kmh=30, t_sea_c=10)
    m = BatteryThermalModel(N_MOTORS, env, soc0=100.0, t_bat0=15.0, t_motor0=15.0)
    print(f"\n=== {label} | outside {env.t_amb:.1f} C (wind chill {env.t_wind_chill:.1f} C) ===")
    print(f"{'t_min':>5} {'phase':<10} {'SOC%':>6} {'Tbat_C':>7} {'V':>5} {'I_A':>5} "
          f"{'P_W':>5} {'f':>5} {'TTE_min':>7} {'Tmot_C':>6}")
    t = 0.0
    for name, minutes, tau_v, dq_v in PHASES:
        tau, dq = np.full(N_MOTORS, tau_v), np.full(N_MOTORS, dq_v)
        steps = int(minutes * 60 / DT)
        for k in range(steps):
            s = m.step(tau, dq, DT); t += DT
            if k % int(300 / DT) == 0 or k == steps - 1:      # every 5 min + phase end
                tte = EnergyEstimator.time_to_empty_min(s["soc_pct"], s["power_W"], m.t_bat)
                print(f"{t/60:5.0f} {name:<10} {s['soc_pct']:6.1f} {s['bms_temp_min_C']:7.1f} "
                      f"{s['pack_V']:5.1f} {s['current_A']:5.1f} {s['power_W']:5.0f} "
                      f"{s['capacity_factor']:5.2f} "
                      f"{tte if np.isfinite(tte) else float('inf'):7.0f} "
                      f"{s['max_temp_winding_C']:6.1f}" + ("  CUTOFF" if s["bms_cutoff"] else ""))
        if m.cutoff:
            print(f"  !! BMS cut-off at t={t/60:.0f} min ({m.soc:.0f}% stranded in the cold cells)")
            break
    print(f"end: SOC {m.soc:.1f}%  T_bat {m.t_bat:.1f} C  energy {m.wh_used:.0f} Wh")


if __name__ == "__main__":
    run(f"JACKET {battery_model.JACKET_PCT:.0f} % (as shipped)")
    battery_model.JACKET_PCT = 0.0
    run("NO JACKET (control)")

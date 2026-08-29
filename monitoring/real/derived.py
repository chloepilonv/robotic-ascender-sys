"""Pure functions: raw SDK fields -> monitored values. Shared by REAL and SIM.

No DDS import here on purpose: the sim reuses the same maths on simulated inputs.
"""
from collections import deque
import numpy as np

# G1 pack: 9000 mAh, 13S Li-ion, 48 V nominal (Unitree spec). Derating below is derived from generic Li-ion cold curves.
PACK_CAPACITY_AH = 9.0
PACK_NOMINAL_V = 48.0

# Temperature derating of usable Li-ion capacity (fraction of the 25 °C capacity).
# Cold raises internal resistance -> voltage sags -> BMS cut-off with charge still inside.
# Equation:  f(T) = clip(1 - K_COLD * (T_REF - T), F_MIN, 1)   for T < T_REF, else 1
# K_COLD = 0.0085 /°C gives ~0.79 at 0 °C and ~0.62 at -20 °C (typical 18650/21700 data).
T_REF_C, K_COLD, F_MIN = 25.0, 0.0085, 0.4
T_CUTOFF_C = -20.0            # below this most BMS refuse discharge -> report 0 min


def capacity_factor(temp_c: float) -> float:
    """Usable-capacity fraction at battery temperature `temp_c`."""
    if temp_c <= T_CUTOFF_C:
        return 0.0
    return float(np.clip(1.0 - K_COLD * (T_REF_C - temp_c), F_MIN, 1.0)) if temp_c < T_REF_C else 1.0


class EnergyEstimator:
    """Integrate electrical power over time and estimate time-to-empty."""

    def __init__(self, window_s: float = 60.0):
        self.wh_used = 0.0
        self.t_prev = None
        self.p_hist = deque()          # (t, P_W) for a moving average
        self.window_s = window_s

    def update(self, t: float, pack_v: float, pack_a: float) -> dict:
        p_w = pack_v * pack_a                      # W (sign: + = discharging, per SDK convention)
        if self.t_prev is not None:
            self.wh_used += p_w * (t - self.t_prev) / 3600.0
        self.t_prev = t
        self.p_hist.append((t, p_w))
        while self.p_hist and t - self.p_hist[0][0] > self.window_s:
            self.p_hist.popleft()
        p_avg = float(np.mean([p for _, p in self.p_hist]))
        return {"power_W": p_w, "power_avg_W": p_avg, "energy_used_Wh": self.wh_used}

    @staticmethod
    def time_to_empty_min(soc_pct: float, p_avg_w: float, temp_c: float = T_REF_C,
                          capacity_ah=PACK_CAPACITY_AH, nominal_v=PACK_NOMINAL_V) -> float:
        """Minutes left at the current average draw and battery temperature (inf if not discharging).

        TTE = SOC * C_25 * V_nom * f(T) / P_avg   with f(T) = capacity_factor(T)
        """
        wh_left = soc_pct / 100.0 * capacity_ah * nominal_v * capacity_factor(temp_c)
        return float("inf") if p_avg_w <= 1.0 else 60.0 * wh_left / p_avg_w


def motor_stats(q, dq, tau, temp_winding, temp_board, vol) -> dict:
    """Vectors (n_motors,) in SDK order -> aggregate joint metrics."""
    tau, dq = np.asarray(tau), np.asarray(dq)
    p_mech = tau * dq                              # W per joint (+ = motor drives, - = braking)
    return {
        "mech_power_W": float(np.abs(p_mech).sum()),
        "max_abs_tau_Nm": float(np.abs(tau).max()),
        "max_abs_tau_joint": int(np.abs(tau).argmax()),
        "max_temp_winding_C": int(np.max(temp_winding)),
        "max_temp_board_C": int(np.max(temp_board)),
        "hot_joint": int(np.argmax(temp_winding)),
        "bus_v_min": float(np.min(vol)),
        "bus_v_max": float(np.max(vol)),
    }


def bms_stats(cell_mv, pack_mv_arr, current_ma, soc, soh, temps_c, cycle) -> dict:
    cells = np.asarray([c for c in cell_mv if c > 0], dtype=float)   # unused slots are 0
    return {
        "soc_pct": int(soc), "soh_pct": int(soh), "cycles": int(cycle),
        "pack_V": pack_mv_arr[0] / 1000.0,         # bmsvoltage[0] = total pack, mV
        "current_A": current_ma / 1000.0,
        "n_cells": int(len(cells)),
        "cell_min_V": float(cells.min() / 1000) if len(cells) else float("nan"),
        "cell_max_V": float(cells.max() / 1000) if len(cells) else float("nan"),
        "cell_spread_mV": float(cells.max() - cells.min()) if len(cells) else float("nan"),
        "bms_temp_max_C": int(max(temps_c)),
        "bms_temp_min_C": int(min(t for t in temps_c if t != 0) or 0),  # coldest cell, 0 = unused slot
        "capacity_factor": capacity_factor(min(t for t in temps_c if t != 0) or T_REF_C),
    }


def imu_stats(gyro, acc, rpy, temp_c) -> dict:
    return {
        "roll_deg": float(np.degrees(rpy[0])), "pitch_deg": float(np.degrees(rpy[1])),
        "yaw_deg": float(np.degrees(rpy[2])),
        "acc_norm_mps2": float(np.linalg.norm(acc)),
        "gyro_norm_radps": float(np.linalg.norm(gyro)),
        "imu_temp_C": int(temp_c),
    }

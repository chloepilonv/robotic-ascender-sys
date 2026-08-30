"""Battery Management System *simulation* for the harness.

Nothing electrical is re-implemented here: this wraps app/bms/sim/battery_model.py
(MATH.md sections 3-5) and app/bms/real/derived.py, feeding them the joint
torques and velocities the harness already has from MuJoCo.

Per control tick:
    tau  = data.actuator_force      (N.m, actuator order)
    dq   = data.actuator_velocity   (rad/s, actuator order)
    P_mech = sum |tau * dq|, P_elec = P_mech / eta + copper + idle
    -> pack current, SOC, pack V (sag through R_int(T)), T_bat, T_motor

Knobs from the page: t_amb (ambient temperature, C) and soc0 (start SOC %).
"""
import sys
from pathlib import Path

import numpy as np

_BMS = Path(__file__).resolve().parents[1] / "bms"
for _sub in ("sim", "real"):
    _p = str(_BMS / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from battery_model import BatteryThermalModel, Environment, R_INT_25  # noqa: E402
from derived import EnergyEstimator, capacity_factor                  # noqa: E402

DEFAULT_T_AMB_C = 15.0
DEFAULT_SOC0 = 100.0


def r_int_curve(t_min=-30.0, t_max=45.0, n=16):
    """R_int(T) samples for the page's curve: R25 * 2^((25-T)/15)."""
    temps = np.linspace(t_min, t_max, n)
    return [[float(t), float(R_INT_25 * 2 ** ((25.0 - t) / 15.0)), capacity_factor(float(t))]
            for t in temps]


class BmsSim:
    def __init__(self, n_actuators: int, t_amb_c=DEFAULT_T_AMB_C, soc0=DEFAULT_SOC0):
        self.n = n_actuators
        self.t_amb_c = float(t_amb_c)
        self.soc0 = float(soc0)
        self.reset()

    def reset(self) -> None:
        self.env = Environment(altitude_m=0.0, wind_kmh=0.0, t_sea_c=self.t_amb_c)
        self.model = BatteryThermalModel(self.n, self.env, soc0=self.soc0)
        self.energy = EnergyEstimator(window_s=60.0)
        self.last = None

    def set_ambient(self, t_amb_c: float) -> None:
        """Live knob. The pack has a ~50 min thermal lag (C_bat*R_th = 3000 s),
        so a slider would show nothing for a long time; instead the pack and
        motors are assumed cold-soaked at the new ambient (they jump to it),
        and self-heating then plays out with the real lag."""
        self.t_amb_c = float(t_amb_c)
        self.env.t_sea_c = self.t_amb_c
        self.model.t_bat = self.t_amb_c
        self.model.t_motor[:] = self.t_amb_c

    def set_soc0(self, soc0: float) -> None:
        self.soc0 = float(np.clip(soc0, 0.0, 100.0))
        self.reset()

    def step(self, data, dt: float, time_seconds: float) -> dict:
        tau = np.asarray(data.actuator_force[:self.n], dtype=np.float64)
        dq = np.asarray(data.actuator_velocity[:self.n], dtype=np.float64)
        b = self.model.step(tau, dq, dt)
        e = self.energy.update(time_seconds, b["pack_V"], b["current_A"])
        tte = EnergyEstimator.time_to_empty_min(b["soc_pct"], e["power_avg_W"], self.model.t_bat)
        per_joint_power = tau * dq
        out = {
            "soc_pct": b["soc_pct"], "pack_V": b["pack_V"], "current_A": b["current_A"],
            "v_ocv_V": float(self.model.v_ocv(self.model.soc)),
            "power_W": b["power_W"], "power_avg_W": e["power_avg_W"],
            "mech_power_W": b["mech_power_W"], "copper_loss_W": b["copper_loss_W"],
            "efficiency": b["efficiency"], "energy_used_Wh": e["energy_used_Wh"],
            "t_bat_C": b["bms_temp_min_C"], "t_amb_C": self.t_amb_c,
            "capacity_factor": b["capacity_factor"], "r_int_ohm": b["r_int_ohm"],
            "bms_cutoff": b["bms_cutoff"],
            "time_to_empty_min": None if not np.isfinite(tte) else float(tte),
            "max_temp_winding_C": b["max_temp_winding_C"], "hot_joint": b["hot_joint"],
            "max_abs_tau_Nm": float(np.abs(tau).max()) if tau.size else 0.0,
            "max_abs_tau_joint": int(np.abs(tau).argmax()) if tau.size else 0,
            "tau_Nm": tau.tolist(), "dq_radps": dq.tolist(),
            "joint_power_W": per_joint_power.tolist(),
            "motor_temps_C": b["motor_temps_C"],
        }
        self.last = out
        return out

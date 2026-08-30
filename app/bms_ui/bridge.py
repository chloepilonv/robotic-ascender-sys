"""app/bms_ui — Battery Management System *simulation* plugged into app/harness.

Owner: BMS work. app/harness is the walker; this file is the only glue.

What it does per control tick: read two MuJoCo arrays,
    tau = data.actuator_force      (N.m,  actuator order, 29 joints)
    dq  = data.actuator_velocity   (rad/s, actuator order)
and feed app/bms/sim/battery_model.py (MATH.md sections 3-6, jacket hardcoded):
    P_mech = sum |tau*dq|, P_elec = P_mech/eta + I^2R + idle
    -> pack current, SOC, pack V (sag through R_int(T)), pack/motor temperature.
Readout only: nothing is written back to the physics or the policy.

Touch points in app/harness/runtime.py (each one line, grep "bms"):
    self.bms = BmsPlugin(model, self.substeps)      Episode.__init__
    self.bms.reset()                                Episode.reset
    **self.bms.on_tick(self.data, time_seconds)     Episode.step return dict
    episode.bms.apply_knobs(server.knobs)           live loop
    **episode.bms.state()                           websocket state message
Page: one <script src="/app/bms_ui/bms.js"> in app/web/index.html.

Standalone check (no browser):  python -m app.bms_ui.selftest
"""
import sys
from pathlib import Path

import numpy as np

_BMS = Path(__file__).resolve().parents[1] / "bms"
for _sub in ("sim", "real"):
    _p = str(_BMS / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from battery_model import BatteryThermalModel, Environment  # noqa: E402
from derived import EnergyEstimator, capacity_factor        # noqa: E402

KNOB_DEFAULTS = {"t_amb": 15.0, "soc0": 100.0}      # page knobs: ambient C, start SOC %

# Scalars recorded per tick (land in hud.json -> Replay tab). Registered on the
# harness recorder by BmsPlugin so recorder.py stays untouched.
HUD_FIELDS = ["bms_soc_pct", "bms_pack_V", "bms_current_A", "bms_power_W",
              "bms_mech_power_W", "bms_t_bat_C", "bms_r_int_ohm",
              "bms_energy_used_Wh", "bms_max_abs_tau_Nm"]


class BmsPlugin:
    def __init__(self, model, substeps: int, t_amb_c=KNOB_DEFAULTS["t_amb"],
                 soc0=KNOB_DEFAULTS["soc0"]):
        self.n = int(model.nu)
        self.dt = float(model.opt.timestep) * int(substeps)
        self.t_amb_c, self.soc0 = float(t_amb_c), float(soc0)
        self.actuator_names = [_actuator_name(model, i) for i in range(self.n)]
        self.last = None
        self.reset()
        _register_hud_fields()

    # -- lifecycle ---------------------------------------------------------
    def reset(self) -> None:
        self.env = Environment(altitude_m=0.0, wind_kmh=0.0, t_sea_c=self.t_amb_c)
        self.model = BatteryThermalModel(self.n, self.env, soc0=self.soc0)
        self.energy = EnergyEstimator(window_s=60.0)

    def apply_knobs(self, knobs: dict) -> None:
        t_amb = float(knobs.get("t_amb", self.t_amb_c))
        if abs(t_amb - self.t_amb_c) > 1e-9:
            self.set_ambient(t_amb)
        soc0 = float(knobs.get("soc0", self.soc0))
        if abs(soc0 - self.soc0) > 1e-9:
            self.set_soc0(soc0)

    def set_ambient(self, t_amb_c: float) -> None:
        """The pack has a ~50 min thermal lag (C_bat*R_th = 3000 s), so a slider
        would show nothing for a long time; instead pack and motors are assumed
        cold-soaked at the new ambient (they jump to it), and self-heating then
        plays out with the real lag."""
        self.t_amb_c = float(t_amb_c)
        self.env.t_sea_c = self.t_amb_c
        self.model.t_bat = self.t_amb_c
        self.model.t_motor[:] = self.t_amb_c
        if self.model.soc > 0 and capacity_factor(self.t_amb_c) > 0:
            self.model.cutoff = False        # a warmed-up pack recovers from a cold cut-off
            self.model.undervoltage_s = 0.0

    def set_soc0(self, soc0: float) -> None:
        self.soc0 = float(np.clip(soc0, 0.0, 100.0))
        self.reset()

    # -- per tick ----------------------------------------------------------
    def on_tick(self, data, time_seconds: float) -> dict:
        """Run the battery model on this tick's torques. Returns ONLY flat
        scalars (bms_*), safe to record; the full readout is in self.last."""
        tau = np.asarray(data.actuator_force[:self.n], dtype=np.float64)
        dq = np.asarray(data.actuator_velocity[:self.n], dtype=np.float64)
        b = self.model.step(tau, dq, self.dt)
        e = self.energy.update(time_seconds, b["pack_V"], b["current_A"])
        tte = EnergyEstimator.time_to_empty_min(b["soc_pct"], e["power_avg_W"], self.model.t_bat)
        self.last = {
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
            "joint_power_W": (tau * dq).tolist(),
            "motor_temps_C": b["motor_temps_C"],
        }
        L = self.last
        return {"bms_soc_pct": L["soc_pct"], "bms_pack_V": L["pack_V"],
                "bms_current_A": L["current_A"], "bms_power_W": L["power_W"],
                "bms_mech_power_W": L["mech_power_W"], "bms_t_bat_C": L["t_bat_C"],
                "bms_r_int_ohm": L["r_int_ohm"], "bms_energy_used_Wh": L["energy_used_Wh"],
                "bms_max_abs_tau_Nm": L["max_abs_tau_Nm"]}

    def state(self) -> dict:
        """Extra keys for the websocket state message (read by bms.js)."""
        return {"bms": self.last, "actuator_names": self.actuator_names}


def _actuator_name(model, i):
    import mujoco
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"actuator_{i}"


def _register_hud_fields():
    try:
        from app.harness import recorder
    except ImportError:
        return
    for name in HUD_FIELDS:
        if name not in recorder.HUD_FIELD_NAMES:
            recorder.HUD_FIELD_NAMES.append(name)

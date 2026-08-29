"""SIM models for what MuJoCo cannot give us: battery, motor/battery temperature, environment.
Equations: app/monitoring/VALUES.md sections 3-5. Pure numpy, steps at sim dt.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "real"))
from derived import PACK_CAPACITY_AH, capacity_factor  # noqa: E402  (same maths as REAL)

# --- motor electrical constants: derived from typical BLDC+planetary actuators of this class; calibrate from real logs ---
ETA = 0.70            # gearbox+driver efficiency
R_WIND = 0.10         # winding resistance, ohm
KT_JOINT = 2.0        # joint-level torque constant N.m/A (= motor kt x gear ratio): 40 N.m -> 20 A -> 40 W copper loss
P_IDLE_W = 60.0       # Jetson + boards + LiDAR + camera
# --- pack (13S) ---
N_CELLS, V_CELL_MIN, V_CELL_MAX, R_INT_25 = 13, 3.0, 4.2, 0.08
# --- thermal ---
C_TH_MOTOR, R_TH_MOTOR = 50.0, 2.0       # J/K, K/W
C_TH_BAT, R_TH_BAT = 2000.0, 1.5


class Environment:
    """altitude (m), wind (km/h), sea-level temp (C) -> ambient temp, air density, cooling factor."""
    def __init__(self, altitude_m=0.0, wind_kmh=0.0, t_sea_c=15.0):
        self.altitude_m, self.wind_kmh, self.t_sea_c = altitude_m, wind_kmh, t_sea_c

    @property
    def t_amb(self):  return self.t_sea_c - 6.5e-3 * self.altitude_m                    # ISA lapse
    @property
    def rho(self):    return 1.225 * np.exp(-self.altitude_m / 8400.0)
    @property
    def cooling_scale(self): return (1.225 / self.rho) ** 0.5                           # R_th multiplier
    @property
    def t_wind_chill(self):
        T, v = self.t_amb, self.wind_kmh
        return T if v < 5 else 13.12 + 0.6215 * T - 11.37 * v**0.16 + 0.3965 * T * v**0.16

    def as_dict(self):
        return {"altitude_m": self.altitude_m, "wind_kmh": self.wind_kmh, "T_amb_C": self.t_amb,
                "T_wind_chill_C": self.t_wind_chill, "air_density": float(self.rho)}


class BatteryThermalModel:
    def __init__(self, n_motors, env: Environment, soc0=100.0, t_bat0=None, t_motor0=None):
        self.env = env
        self.soc = soc0
        self.t_bat = env.t_amb if t_bat0 is None else t_bat0
        self.t_motor = np.full(n_motors, env.t_amb if t_motor0 is None else t_motor0)
        self.wh_used = 0.0
        self.cutoff = False

    @staticmethod
    def v_ocv(soc): return N_CELLS * (V_CELL_MIN + (V_CELL_MAX - V_CELL_MIN) * soc / 100.0)
    def r_int(self):  return R_INT_25 * 2 ** ((25.0 - self.t_bat) / 15.0)

    def step(self, tau, dq, dt) -> dict:
        tau, dq = np.asarray(tau), np.asarray(dq)
        i_motor = tau / KT_JOINT                              # A per motor
        p_cu = i_motor**2 * R_WIND                              # copper loss per motor, W
        p_mech = np.abs(tau * dq)
        p_elec = 0.0 if self.cutoff else p_mech.sum() / ETA + p_cu.sum() + P_IDLE_W
        # pack: solve V = Vocv - I*R with I = P/V  ->  V^2 - Vocv V + P R = 0
        vocv, r = self.v_ocv(self.soc), self.r_int()
        disc = vocv**2 - 4 * p_elec * r
        v_pack = (vocv + np.sqrt(disc)) / 2 if disc > 0 else N_CELLS * V_CELL_MIN
        i_pack = p_elec / v_pack
        f_t = capacity_factor(self.t_bat)
        if f_t > 0 and not self.cutoff:
            self.soc -= 100.0 * i_pack * dt / (3600.0 * PACK_CAPACITY_AH * f_t)
        self.soc = max(self.soc, 0.0)
        self.wh_used += p_elec * dt / 3600.0
        if v_pack < N_CELLS * V_CELL_MIN or self.soc <= 0 or f_t == 0:
            self.cutoff = True
        # thermal (first order), thinner air -> worse cooling
        rth_m, rth_b = R_TH_MOTOR * self.env.cooling_scale, R_TH_BAT * self.env.cooling_scale
        self.t_motor += dt / C_TH_MOTOR * (p_cu - (self.t_motor - self.env.t_amb) / rth_m)
        self.t_bat += dt / C_TH_BAT * (i_pack**2 * r - (self.t_bat - self.env.t_amb) / rth_b)
        return {
            "soc_pct": float(self.soc), "soh_pct": 100, "pack_V": float(v_pack), "current_A": float(i_pack),
            "power_W": float(p_elec), "mech_power_W": float(p_mech.sum()), "copper_loss_W": float(p_cu.sum()),
            "efficiency": float(p_mech.sum() / p_elec) if p_elec > 0 else 0.0,
            "energy_used_Wh": float(self.wh_used), "bms_temp_min_C": float(self.t_bat), "bms_temp_max_C": float(self.t_bat),
            "capacity_factor": float(f_t), "r_int_ohm": float(r), "bms_cutoff": bool(self.cutoff),
            "cell_min_V": float(v_pack / N_CELLS), "cell_max_V": float(v_pack / N_CELLS), "cell_spread_mV": 0.0,
            "max_temp_winding_C": float(self.t_motor.max()), "hot_joint": int(self.t_motor.argmax()),
            "motor_temps_C": [float(t) for t in self.t_motor],
        }

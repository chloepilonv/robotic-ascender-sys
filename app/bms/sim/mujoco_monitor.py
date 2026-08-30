"""SIM monitor: read MuJoCo mjData every step, run battery/thermal models, write the SAME log.jsonl as REAL.

Library use (inside your own policy loop):
    mon = SimMonitor(model, env=Environment(altitude_m=5300, wind_kmh=30), log="app/bms/sim/log.jsonl")
    while running:
        mujoco.mj_step(model, data)
        mon.step(data)                 # call after every mj_step

Standalone demo (no policy, PD hold of the standing keyframe + arm swing), 60 s of sim in a few s real time:
    python app/bms/sim/mujoco_monitor.py --xml path/to/mujoco_menagerie/unitree_g1/g1.xml --seconds 60 --altitude 5300
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import mujoco

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "real"))
from battery_model import BatteryThermalModel, Environment   # noqa: E402
from derived import EnergyEstimator, imu_stats               # noqa: E402

G1_MASS_KG = 33.3


class SimMonitor:
    def __init__(self, model, env=None, soc0=100.0, log=None, period=1.0, root_body="pelvis"):
        self.m, self.env = model, env or Environment()
        self.dt = model.opt.timestep
        self.bat = BatteryThermalModel(model.nu, self.env, soc0)
        self.energy = EnergyEstimator()
        self.period, self.t_next = period, 0.0
        self.root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_body)
        self.gyro = self._sensor("imu-pelvis-angular-velocity"); self.acc = self._sensor("imu-pelvis-linear-acceleration")
        self.log = open(log, "a") if log else None
        self.dist, self.last = 0.0, None

    def _sensor(self, name):
        i = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SENSOR, name)
        return None if i < 0 else slice(self.m.sensor_adr[i], self.m.sensor_adr[i] + self.m.sensor_dim[i])

    def step(self, data):
        tau, dq = data.actuator_force, data.qvel[6:6 + self.m.nu]       # joint torques (N.m), joint vel (rad/s)
        b = self.bat.step(tau, dq, self.dt)
        if data.time < self.t_next:
            return None
        self.t_next = data.time + self.period
        # base (pelvis) pose / velocity: sim ground truth (REAL has no equivalent)
        v_world = data.cvel[self.root][3:6]; speed = float(np.linalg.norm(v_world[:2]))
        q = data.xquat[self.root]                                          # w x y z
        w, x, y, z = q
        rpy = (np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y)), np.arcsin(np.clip(2*(w*y-z*x), -1, 1)), np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z)))
        gyro = data.sensordata[self.gyro] if self.gyro else data.cvel[self.root][:3]
        acc = data.sensordata[self.acc] if self.acc else np.zeros(3)
        e = self.energy.update(data.time, b["pack_V"], b["current_A"])
        out = {"t": float(data.time), "source": "sim", "hz_lowstate": 1.0 / self.dt, "hz_bms": 1.0 / self.period,
               **b, **e, **imu_stats(gyro, acc, rpy, self.bat.t_bat),
               "base_pos_m": [float(v) for v in data.xpos[self.root]], "speed_mps": speed,
               "cost_of_transport": float(b["power_W"] / (G1_MASS_KG * 9.81 * speed)) if speed > 0.05 else None,
               "fallen": bool(abs(rpy[0]) > 1.05 or abs(rpy[1]) > 1.05),
               "max_abs_tau_Nm": float(np.abs(tau).max()), "max_abs_tau_joint": int(np.abs(tau).argmax()),
               "contact_force_N": float(sum(np.linalg.norm(self._cf(data, i)[:3]) for i in range(data.ncon))),
               "time_to_empty_min": EnergyEstimator.time_to_empty_min(b["soc_pct"], e["power_avg_W"], self.bat.t_bat),
               **self.env.as_dict()}
        if self.log:
            self.log.write(json.dumps(out) + "\n"); self.log.flush()
        return out

    def _cf(self, data, i):
        f = np.zeros(6); mujoco.mj_contactForce(self.m, data, i, f); return f


def fmt(s):
    return (f"t={s['t']:6.1f}s SOC {s['soc_pct']:5.1f}% {s['pack_V']:5.1f}V {s['current_A']:5.2f}A {s['power_avg_W']:4.0f}W "
            f"TTE {s['time_to_empty_min']:5.0f}min | Tbat {s['bms_temp_min_C']:5.1f}C f={s['capacity_factor']:.2f} "
            f"Tmot {s['max_temp_winding_C']:5.1f}C j{s['hot_joint']} | mech {s['mech_power_W']:4.0f}W eff {s['efficiency']:.2f} "
            f"| v {s['speed_mps']:.2f} m/s Tamb {s['T_amb_C']:.0f}C")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--xml", required=True, help="robot MJCF, e.g. assets/robots/mujoco/g1_unitree_ascender.xml (a floor is added)")
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--altitude", type=float, default=0.0); p.add_argument("--wind", type=float, default=0.0)
    p.add_argument("--t_sea", type=float, default=15.0); p.add_argument("--soc0", type=float, default=100.0)
    p.add_argument("--log", default=str(HERE / "log.jsonl")); p.add_argument("--viewer", action="store_true")
    a = p.parse_args()
    scene = f'<mujoco><include file="{os.path.abspath(a.xml)}"/><worldbody><light pos="0 0 3"/><geom type="plane" size="0 0 0.05"/></worldbody></mujoco>'
    m = mujoco.MjModel.from_xml_string(scene); d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0); q0 = d.qpos[7:7 + m.nu].copy()
    # menagerie G1 actuators are position servos (kp=500 in the MJCF): ctrl = target joint angle (rad)
    mon = SimMonitor(m, Environment(a.altitude, a.wind, a.t_sea), a.soc0, a.log)
    viewer = mujoco.viewer.launch_passive(m, d) if a.viewer else None
    while d.time < a.seconds:
        target = q0.copy(); target[15:29] += 0.4 * np.sin(2 * np.pi * 0.5 * d.time)   # swing arms (SDK order: arms = 15..28)
        d.ctrl[:] = target
        mujoco.mj_step(m, d)
        s = mon.step(d)
        if s: print(fmt(s))
        if viewer: viewer.sync()

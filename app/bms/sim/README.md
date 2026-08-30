# app/bms/sim — test the monitoring stack in MuJoCo (no robot needed)

Same output schema as `app/bms/real/log.jsonl`, plus sim-only keys (`base_pos_m`, `speed_mps`, `cost_of_transport`,
`contact_force_N`, `fallen`, `altitude_m`, `T_amb_C`, …). Key names and math: `MATH.md`.

## 1. Setup
    pip install mujoco numpy
    python assets/robots/mujoco/build.py --fetch   # stock Unitree meshes for our G1 MJCF (see assets/robots/mujoco/README.md)

## 2. Run the demo (robot stands, swings arms; 30 s sim at 5,300 m in −19 °C, 30 km/h wind)
    python app/bms/sim/mujoco_monitor.py --xml assets/robots/mujoco/g1_unitree_ascender.xml --seconds 30 --altitude 5300 --wind 30
    python app/bms/sim/mujoco_monitor.py --xml ... --soc0 30          # start with a low pack
    python app/bms/sim/mujoco_monitor.py --xml ... --viewer           # open the MuJoCo window
Writes `app/bms/sim/log.jsonl`, one line per sim second → point the app at that file.

## 3. Hook into your own policy loop
    from app.bms.sim.mujoco_monitor import SimMonitor, Environment
    mon = SimMonitor(model, Environment(altitude_m=5300, wind_kmh=30), soc0=100, log="app/bms/sim/log.jsonl")
    while ...:
        mujoco.mj_step(model, data)
        row = mon.step(data)          # None except once per `period` (1 s); row is the dict that was logged

## What MuJoCo gives directly vs what is modelled
| Read from `mjData` | Modelled (`battery_model.py`) |
|---|---|
| joint q / q̇ (`qpos`, `qvel`), torque (`actuator_force`) | SOC, pack V, current, energy, TTE |
| base pose/velocity (`xpos`, `xquat`, `cvel`), IMU sensors | motor + battery temperature (1st-order thermal) |
| contact forces (`mj_contactForce`) | ambient temp from altitude, wind chill, air density |

## Derived constants (top of `battery_model.py`) — from generic Li-ion / BLDC data, calibrate with real logs
`ETA` 0.70, `R_WIND` 0.10 Ω, `KT_JOINT` 2.0 N·m/A, `P_IDLE_W` 60, `R_INT_25` 0.08 Ω, thermal C/R,
`JACKET_PCT` 60 (hardcoded battery insulation: KAILAS jacket vents closed; 30 ≈ vents/fans open, 0 = no jacket —
the robot shell is already in `R_TH_BAT`, don't count it twice). Calibrate against `real/log.jsonl`
(R_int from voltage sag ÷ current step; R_th·C from one power-off cooldown curve; η from V·I ÷ Σ|τ·q̇|).
Sanity: standing + arm swing ≈ 70 W; walking should land ~150–300 W.

## Known limits
- Menagerie G1 actuators are position servos (kp=500); `actuator_force` is still the joint torque, so the maths holds.
- No per-cell model (all cells = pack/13). Insulation is one hardcoded scalar (`JACKET_PCT`); no sun load, no vent/fan dynamics.

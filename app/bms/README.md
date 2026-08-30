# monitoring — what we can read on the G1, units, and rates

Order of work: **2. REAL first** (`app/bms/real/`) → **1. SIM** (`app/bms/sim/`, transposes the same values).

## A) Scripts (REAL)
| Script | Does |
|---|---|
| `real/discover_topics.py` | Lists live DDS topics + types. Run first to confirm BMS/mainboard topic names. |
| `real/monitor_battery.py` | Subscribes lowstate + bmsstate + mainboardstate, prints 1 Hz table, logs `log.jsonl`, measures real Hz. |
| `real/derived.py` | Pure maths (power, Wh, time-to-empty, hot joint…). Reused as-is by SIM. |

`pip install unitree_sdk2py cyclonedds` on the Jetson / a PC on the robot LAN (`--iface eth0`).

## B) Values, units, source field
### Raw — `rt/lowstate` (`unitree_hg::LowState_`), **500 Hz**
| Value | Field | Unit |
|---|---|---|
| joint position / velocity / accel | `motor_state[i].q / dq / ddq` | rad, rad/s, rad/s² |
| joint torque (estimated) | `motor_state[i].tau_est` | N·m |
| motor temp winding / driver board | `motor_state[i].temperature[0] / [1]` | °C |
| motor bus voltage | `motor_state[i].vol` | V |
| motor fault | `motor_state[i].motorstate` | bitmask, 0 = ok |
| IMU quat / gyro / accel / rpy | `imu_state.*` | –, rad/s, m/s², rad |
| IMU temp | `imu_state.temperature` | °C |
| board uptime | `tick` | ms |
| FSM mode | `mode_machine` | enum |

### Raw — `rt/lf/bmsstate` (`unitree_hg::BmsState_`), **low freq (~1–10 Hz, script prints measured)**
| Value | Field | Unit |
|---|---|---|
| state of charge | `soc` | % |
| state of health | `soh` | % |
| pack voltage | `bmsvoltage[0]` | mV |
| pack current | `current` | mA (+ discharge) |
| per-cell voltage | `cell_vol[0..n]` | mV (13S expected, unused = 0) |
| BMS temps | `temperature[0..11]` | °C |
| charge cycles | `cycle` | count |
| BMS status flags | `bmsstate[0..4]` | bitmask |

### Raw — `rt/lf/mainboardstate` (`unitree_hg::MainBoardState_`), low freq
fan speed `fan_state[6]` (rpm), board temps `temperature[6]` (°C).

### Computed (`derived.py`)
| Value | Formula | Unit |
|---|---|---|
| electrical power | `pack_V × current_A` | W |
| 60 s avg power | moving mean | W |
| energy used | `∫ P dt` | Wh |
| **capacity factor f(T)** | `clip(1 − 0.0085·(25 − T_bat), 0.4, 1)`, 0 below −20 °C | – (0.79 @ 0 °C, 0.62 @ −20 °C) |
| **time-to-empty** | `SOC% × 9 Ah × 48 V × f(T_bat) / P_avg` | min |
| mechanical power | `Σ |τ·q̇|` | W |
| drivetrain efficiency | `P_mech / P_elec` | – |
| hottest joint / max torque joint | argmax | index (SDK order) |
| cell spread | `max − min cell_vol` | mV (>50 mV = unbalanced pack) |

## C) Frequency
- `rt/lowstate` **500 Hz** (control rate). Also `rt/lf/lowstate` reduced copy.
- `rt/lf/bmsstate`, `rt/lf/mainboardstate`: low freq, not documented → **measure with the script** (`hz_bms` column).
- We print/log at **1 Hz** (`--period`); torque/temp aggregates come from the last 500 Hz sample.

## Caveats
- `bms/monitor.py` and `bms/battery_alarm.py` read `msg.bms_state` / `power_v` on the hg `LowState_` — those fields only exist in the Go2 IDL. Will crash on G1. Replace with `rt/lf/bmsstate`.
- Topic names `rt/lf/bmsstate` / `rt/lf/mainboardstate` are the best guess for firmware ≥1.0; confirm with `discover_topics.py` and pass `--bms_topic`.
- Pack capacity 9 Ah / 48 V nominal is the G1 EDU spec; edit `derived.py` if the BMS reports another cell count.

## SIM (`app/bms/sim/`) — done, see `sim/README.md`
MuJoCo gives τ, q̇, pose, contacts; `sim/battery_model.py` models battery/thermal/environment + hardcoded jacket insulation (MATH.md §3–6);
`sim/mujoco_monitor.py` writes the same `log.jsonl` schema. Test the app against it before any robot time.

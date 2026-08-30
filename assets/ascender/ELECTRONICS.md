# Ascender end-effector — sensing + transceiver

The ascender is passive (spring cam ratchets on the rope, the wrist orients it). The end-effector only **senses and reports**.
Mechanical mount: `MOUNT.md`.

## Signals
| Signal | Sensor | Rate | Why |
|---|---|---|---|
| rope tension (N) | 100 kg S-type load cell between adapter and ascender, HX711 (24-bit) | 80 Hz | "the ascender is holding my weight" → climb policy feedback, fall detection |
| cam engaged (bool) | Hall sensor + magnet on the cam lever | 50 Hz | "clipped in / released" → FSM gate before climb |
| tool temperature (°C) | on-board (ESP32) | 1 Hz | cold-start sanity, dashboard |
| battery (V) | ADC divider | 1 Hz | dashboard |

## BOM (~€60, 15 g electronics)
| Part | Qty | Note |
|---|---|---|
| Petzl Basic ascender (B18) | 1 | the scanned model; rope 8–13 mm |
| ESP32-S3 mini (Wi-Fi + BLE transceiver) | 1 | e.g. Seeed XIAO ESP32-S3, 5 g |
| HX711 load-cell amplifier | 1 | |
| 100 kg S-type load cell (M6 studs) | 1 | in-line: wrist adapter → cell → ascender eye |
| Hall sensor A3144 + 3 mm magnet | 1 | on the cam lever |
| 1S LiPo 400 mAh + TP4056 charger | 1 | ~8 h; or tap the wrist power connector (voltage TBD, see MOUNT.md) |
| Adapter stem (PA12-CF / 6061 Al) | 1 | from `MOUNT.md` calipers |

## Wiring (ESP32-S3)
    HX711  DOUT→GPIO4  SCK→GPIO5  VCC→3V3
    Hall   OUT →GPIO6 (pull-up)
    VBAT   divider 100k/100k →GPIO1 (ADC)
Firmware: read at 80 Hz, send UDP JSON to the Jetson every 20 ms (50 Hz):
    {"t_ms":123456,"tension_N":312.5,"engaged":true,"temp_C":-12.0,"vbat_V":3.9}
Wi-Fi: join the G1's AP (the Jetson is 192.168.123.164; end-effector gets a static lease). BLE fallback if the AP is off.

## Robot side
`real/ascender_bridge.py` (to write): UDP listener → publishes `rt/ascender_state` (custom IDL: same fields) at 50 Hz next to
`rt/lowstate` and `rt/lf/bmsstate`, and appends the fields to `app/bms/real/log.jsonl` → dashboard.
Sim twin: `app/bms/sim/mujoco_monitor.py` emits the same keys from the rail joint force (`tension_N`) and the FSM (`engaged`).

## Failure modes
- Wi-Fi drop → bridge marks `engaged=None` after 200 ms → FSM holds (fail-closed, same rule as the human-gate).
- Load cell over-range (>1 kN, a fall arrest) → latch a `overload` flag; inspect the ascender before reuse.
- Cold: LiPo below −10 °C loses ~40 % capacity (same `capacity_factor` as the main pack); keep the puck inside the adapter shell.

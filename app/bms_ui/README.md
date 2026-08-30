# app/bms_ui — Battery Management System panel (simulated)

Sits on top of `app/harness` (the walker) without changing how it works. Everything
BMS lives in this folder; the harness has 6 one-line hooks (`grep -n bms app/harness/runtime.py`)
and `index.html` has one `<script>` tag.

| File | Role |
|---|---|
| `bridge.py` | MuJoCo `actuator_force` / `actuator_velocity` → `app/bms/sim/battery_model.py`; knobs; state for the page |
| `bms.js` | The panel + sidebar knobs; injects itself into the page |
| `selftest.py` | `python -m app.bms_ui.selftest` — 3 s headless walk, prints the numbers |

## Try it
    python -m app.harness.runtime --live --world free_0     # then http://localhost:8766/
Click the view, hold **W**. The **BMS** tab (top bar) shows/hides the panel; **details ▸** opens
joint torques, power breakdown and the SOC/V/P history. Sidebar: **ambient temperature**, **start SOC**.

## What is shown
SOC + time-to-empty · pack V + I·R sag · current + Wh used · P_elec (with P_mech = Σ|τ·q̇|) · pack °C
(+ ambient, capacity factor) · R_int in mΩ · R_int(T) curve with the live point. Nothing feeds back
into the physics — a cut-off battery does not stop the legs (yet).

## Remove it
Delete this folder, the 6 lines in `runtime.py`, the `<script>` tag. `hud.json` simply loses its `bms_*` arrays.

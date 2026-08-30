# app/bms_ui — Battery Management System panel (simulated)

Sits on top of `app/harness` (the walker) without changing how it works.
`bridge.py` runs `app/bms/sim/battery_model.py` (MATH.md §3–6, jacket insulation
hardcoded at 60 %) on each control tick's `actuator_force` / `actuator_velocity`;
the harness attaches it in `runtime.py` `make_battery_plugin` (always on,
best-effort) and broadcasts the readout as `state["bms"]`.

| File | Role |
|---|---|
| `bridge.py` | MuJoCo τ, q̇ → battery/thermal model; knobs; `state["bms"]` for the page |
| `selftest.py` | `python -m app.bms_ui.selftest` — 3 s headless walk, prints the numbers |

The panel itself is the **BMS card in `app/web/render3d.html`** (`#bmsCard`), showing:
electrical power · battery pack temperature (pulled down by the outside cold through
the jacket) · battery life (SOC + time-to-empty) · a foldable "torque and velocity
of joints". Each name carries an ⓘ with the math or the real-G1 source
(`rt/lowstate`, `rt/lf/bmsstate`). Sidebar knobs: ambient temperature, start SOC.

Nothing feeds back into the physics — a cut-off battery does not stop the legs (yet).
`hud.json` gains per-tick `bms_*` arrays for replay.

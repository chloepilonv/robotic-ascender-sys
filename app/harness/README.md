# app/harness — the interactive walker on the team climb env

Runs `rl.environment.climb_env.G1ClimbAscender` in plain MuJoCo at 50 Hz on a
laptop and streams it to the browser. Nothing physical is re-implemented: the
harness loads the env's compiled model (see `PARITY.md`, `fingerprint.json`).

## Run it on localhost

From the repo root, in the `everest` env (the one that runs the trainer/viewer —
it already has `playground==0.2.0`, `mujoco==3.12.0`, `brax==0.14.2`):

    pip install websockets pillow          # the only extras the harness needs
    # optional, for episode.mp4: brew install ffmpeg / apt install ffmpeg

    python -m app.harness.runtime --live --world free_0

then open **http://localhost:8766/app/web/index.html**. Click the view to take
the pointer, hold **W** to walk/climb, move the mouse to look around, **R**
resets, **Esc** pauses. Map selector: `free_0`, `climb_0`, `free_30`,
`climb_30` (slope × rope on/off, all built from the team env). Wind dial is in
m/s using `wind_env`'s drag law (the climb env is trained without wind).

Every live session is recorded under `app/harness/episodes/<stamp>_<world>/`
(frames.npz, hud.json, header.json, episode.mp4) and playable from the page's
Replay tab.

## BMS panel (battery simulation)

The strip under the view (toggle with the **BMS** tab) runs
`app/bms/sim/battery_model.py` on the actuators every control tick — nothing
physical changes, it is a readout: SOC / pack V / current / time-to-empty,
electrical vs mechanical power (P_mech = Σ|τ·q̇|, P_elec = P_mech/η + I²R + 60 W),
pack and motor temperature, R_int(T) curve with the live point, per-joint |τ|
bars, and a SOC / V / P lifecycle strip. Sidebar knobs: **ambient temperature**
(cold-soaks the pack, so R_int and the capacity factor move at once) and
**start SOC**. The same numbers land in `hud.json` (`bms_*`) and show in Replay.
Sanity: standing ≈ 70 W, walking on `free_0` ≈ 200–250 W. Glue lives in
`app/harness/bms_bridge.py`.

Other entry points:

    python -m app.harness.runtime --world climb_30 --duration 10 --hold-w --no-render   # headless case
    python -m app.harness.runtime --live --policy path/to/policy.npz                    # a trained policy (mels npz layout)
    python -m app.harness.test_parity                                                   # obs + rollout parity vs the JAX env

`--port` moves the websocket (HTTP is port+1); `--command-speed` sets the W
forward command (default 0.5 m/s).

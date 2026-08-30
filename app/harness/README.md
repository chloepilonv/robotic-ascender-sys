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

Other entry points:

    python -m app.harness.runtime --world climb_30 --duration 10 --hold-w --no-render   # headless case
    python -m app.harness.runtime --live --policy path/to/policy.npz                    # a trained policy (mels npz layout)
    python -m app.harness.test_parity                                                   # obs + rollout parity vs the JAX env

`--port` moves the websocket (HTTP is port+1); `--command-speed` sets the W
forward command (default 0.5 m/s).

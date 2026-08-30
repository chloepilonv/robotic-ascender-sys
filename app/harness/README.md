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
`climb_5/8/10/12/30` (slope × rope on/off) plus four **Pemba G1** worlds
(`climb_30_pemba`, `climb_12_pemba`, `climb_8_pemba`, `free_0_pemba`) that fly
the real demo robot — jacket, snow boots, ascender instead of the right hand.
All twelve are built from the team env. Wind dial is in m/s using `wind_env`'s
drag law (the climb env is trained without wind).

Every live session is recorded under `app/harness/episodes/<stamp>_<world>/`
(frames.npz, hud.json, header.json, episode.mp4) and playable from the page's
Replay tab.

Other entry points:

    python -m app.harness.runtime --world climb_30 --duration 10 --hold-w --no-render   # headless case
    python -m app.harness.runtime --live --policy path/to/policy.npz                    # a trained policy (mels npz layout)
    python -m app.harness.test_parity                                                   # obs + rollout parity vs the JAX env

`--port` moves the websocket (HTTP is port+1); `--command-speed` sets the W
forward command (default 0.5 m/s).


## The Pemba G1 (the real demo robot)

`app/harness/robot_variants.py` generates a Playground-compatible scene that
wraps `assets/robots/mujoco/g1_unitree_ascender.xml`, then points the ONE line
their `_build_model` uses to choose a starting scene (`consts.task_to_xml`) at
it for the duration of one env construction. Their builder does everything else
— tilts the floor, adds the rope and carrier, connects the palm, sets foot
friction — unchanged and unaware. Nothing under `assets/robots/mujoco/` is
edited, and the generated files (absolute mesh paths, so machine-specific) live
gitignored in `app/harness/generated/`.

    python -m app.harness.robot_variants     # regenerate + print what it did

Joint parity is checked before anything else and raises if it ever fails: the
demo robot has the same 29 actuated joints in Playground's exact order, which
is the only reason its `knees_bent` keyframe and the policy's 29 actions are
transferable. See `PARITY.md` for the full diff and the open ASK about
actuator gains.

## Chloe: BMS plugs in here

`Episode.physics_step_hooks` is a list of `callable(model, data) -> dict | None`
called after **every** `mj_step` — that is `model.opt.timestep` (2 ms / 500 Hz),
the rate a battery or thermal model integrates at, not the 50 Hz control tick.
The last non-None dict any hook returns during a control tick becomes
`episode.latest_bms`, which is broadcast in the live state message as
`state["bms"]` and written to `hud.json` as a per-tick `bms` list.

    episode.physics_step_hooks.append(my_hook)   # that is the whole seam

`--bms` wires `app/bms/sim/mujoco_monitor.SimMonitor` in for you:

    python -m app.harness.runtime --live --world free_0 --bms

It builds `Environment(altitude_m=<world's altitude_meters, default 6907>,
wind_kmh=3.6 × wind dial)` and keeps `wind_kmh` live as the dial moves. Your
`step(data)` signature is adapted at the call site, so nothing in `app/bms`
needs to change. The whole attach is best-effort: if the import fails, the
harness prints why and runs on without a battery readout.

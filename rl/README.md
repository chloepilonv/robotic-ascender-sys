# rl — the climbing environment

A single MuJoCo model containing the Himalaya G1, real Lhotse Face terrain, and
a fixed rope the robot's ascender rides. Four things are parameterised for
domain randomisation — **wind, friction, slope, surface variation** — and the
mels walking policy runs in it as a baseline.

Everything here runs on a laptop CPU. Training needs a GPU and is not set up
yet; see [Training, from here](#training-from-here).

```bash
python -m rl.tools.fetch_visual_assets     # one-time: playground G1 (optional)
python assets/robots/mujoco/build.py --fetch   # one-time: stock G1 link meshes

python -m rl.scripts.climb_scene --verify-policy      # is the baseline sane?
python -m rl.scripts.climb_scene --check              # is the scene sound?
python -m rl.scripts.climb_scene --patch B_slope0 --view    # look at it
```

Deps: `mujoco`, `numpy`, `scipy`, `pillow`. No jax needed for any of the above.

## Start here

```bash
python -m rl.scripts.climb_scene --list                      # what terrain exists
python -m rl.scripts.climb_scene --patch B_slope0 --view     # flat: stands indefinitely
python -m rl.scripts.climb_scene --patch B_slope0 --cmd-x 1.0 --view   # walks, 0.66 m/s
python -m rl.scripts.climb_scene --patch B --view            # 38.6 deg: falls, hangs on the rope
python -m rl.scripts.climb_scene --patch B --export build/scene.xml    # standalone MJCF
```

macOS: use plain `python3`. `--view` drives the ratchet from a
`set_mjcb_control` hook because `launch_passive` demands `mjpython`.

## Three traps that look like bugs

Read this before concluding something is broken. Each produces "the robot falls"
and none of them is a policy fault.

**1. Friction below 0.2 and it cannot stand — on level ground.** Measured on the
flat patch over 20 s: 0.20+ stands indefinitely, 0.15 falls at 4.4 s, 0.10 at
1.1 s, 0.01 at 0.6 s. Use `--friction 0.3` or higher when you are testing
anything else.

> The flag used to be `--ice`, which reads backwards: `--ice 0` looks like "no
> ice" but means *zero friction*, the slipperiest possible setting. It is now
> `--friction` (`--ice` still works and is **not** inverted).

**2. Any slope from 25 deg up and the walking policy falls, even dry and
windless.** It was trained on flat ground and has never seen an incline. This is
the gap the RL is meant to close, and it is the expected baseline result. The
hand stays on the line and the descent saturates near −0.7 m: the robot is
hanging off the ascender, not tumbling down the face.

**3. `--visual` does nothing** unless you also pass `--robot playground`. The
project's MJCF ships its own meshes.

Isolating the axes on the roped baseline, 8 s, command 0: **flat ground survives
everything** — friction 0.1 *and* 20 m/s wind. Slope alone is what defeats the
policy.

## The four randomisation axes

| axis | flag | MuJoCo field | per-env cost at 4096 envs |
|---|---|---|---|
| wind | `--wind`, `--wind-heading` | `xfrc_applied` on the torso | free (data, not model) |
| friction | `--friction` | `geom_friction`, `pair_friction`, priority geoms | negligible |
| slope | `--patch`, `--slope` | `geom_quat` of the terrain | 0.066 MB |
| surface variation | `--rough`, `--seed` | `hfield_size[2]` scales a shared grid | 0.066 MB |

All four are per-geom or per-body fields rather than model topology, so one
compiled model covers the whole distribution and MJX can `vmap` them without
recompiling. Roughness *amplitude* is one float per env because the grid is
normalised to [0,1] and `hfield_size[2]` is its elevation scale; roughness
*realisation* (the seed) is a whole new grid, so vary it between training
iterations — batching `hfield_data` would cost **2.46 GB** at 4096 envs.

Friction is the axis most likely to be silently dead, because two separate
mechanisms override geom friction: explicit `<pair>` elements on the playground
robot, and priority-1 foot geoms on the himalaya one. `--check` asserts the
value the solver actually uses, not a trajectory statistic.

## Layout

| file | what |
|---|---|
| `environment/terrain.py` | Lhotse patches as heightfields; splits slope from roughness |
| `environment/ascender.py` | the rope polyline, arc-length projection, one-way ratchet |
| `environment/robot.py` | robot profiles and the adapter that reconciles them |
| `environment/climb_scene.py` | assembles robot + terrain + rope into one model |
| `environment/walk_policy.py` | the mels baseline, pure NumPy, no jax |
| `scripts/climb_scene.py` | CLI: build, check, verify, export, render, view |
| `tools/` | fetch the playground G1, generate the flat patch, extract USD gear |
| `tests/test_climb_scene.py` | 48 tests, all CPU, ~14 s |

`environment/climb_env.py` and `wind_env.py` are the **older** MJX envs, still on
the slide-joint mechanism and a tilted plane. They are not what the scene above
builds; porting them onto `climb_scene` is the next step.

`CLIMB_SCENE.md` is the design document — why the ascender is a mocap body, why
slope and roughness separate cleanly, what the robot adapter changes and why,
and the list of silent bugs this surfaced.

## The ascender

A real ascender slides up a fixed rope and jams under load. Modelled as a
**bead on a wire**: the carrier is a body with three slide joints, and after each
substep its perpendicular offset from the rope polyline is removed and its
perpendicular velocity cancelled, while the along-rope component is left to the
dynamics. A `connect` equality ties `right_palm` to the carrier, so the hand
drags the carrier up the line. The ratchet clamps arc length non-decreasing:
slides up, jams under load.

The obvious construction — a zero-DOF mocap carrier written to the palm's
projection each substep — does **not** work, and fails silently. `connect` is an
isotropic 3-DOF point constraint, so the palm cannot move relative to the carrier
at all; its projection therefore never changes, so the carrier never moves, and
the hand ends up welded to a fixed point. Measured: a 1.2 rad shoulder sweep
advanced it 0.000 m.

The carrier's three joints are appended after the robot's, so `qpos[7:7+29]` and
`qvel[6:6+29]` still address the robot alone and the observation stays 103-dim —
but only with **bounded** slices. An open-ended `qpos[7:]` picks the carrier up
as three phantom joints.

`sc.ascender.progress` is arc length climbed since reset: the natural progress
reward.

**If the hand looks stuck, it probably is not.** The walking policy cannot climb:
on a slope it sags onto the rope and hangs, so the hand legitimately stays put.
`--haul N` applies a force up the fall line so you can watch the ascender slide.
The robot weighs 338 N, so on patch B (38.6 deg) about 211 N just holds station;
230-300 N climbs smoothly with the hand staying within 4 mm of the rope, and
400 N tears the grip open.

```bash
python -m rl.scripts.climb_scene --patch B --haul 260 --view   # ascender rides up
python -m rl.scripts.climb_scene --patch B_slope0 --view       # walking drags it ~1.5 m / 10 s
```

### The rope is solid

Limbs cannot pass through it: shoved sideways into the line the robot travels
0.13 m instead of 0.66 m. `--rope-scenery` restores the old non-colliding rope.

The **gripping arm is exempt** (`GRIP_BODIES`, shoulder through hand). The palm
is pinned to the line, so that arm necessarily lies alongside it and snags on the
rope it is holding — the elbow alone was 1215 contacts over a 10 s roped run, the
only body touching the rope, and it reads as the rope being "too solid".
Excluding a held object from the limb holding it is the normal treatment. Torso,
pelvis, legs, head and the left arm still collide.

Contact is firm normally and slippery tangentially (`solref (0.02, 1)`,
friction 0.15). Softening the normal direction does not help and breaks the
blocking: at `solref 0.03` the robot pushes straight through.

Verified: hauling the robot up the fall line advances the ascender 15 m and it
clamps at the rope's end; roped it sags 0.8 m and holds; unroped it falls off the
face. Ratchet backsliding over 2000 steps is exactly 0, and the hand stays within
9 mm of the carrier (rope radius 25 mm) while the policy runs.

## Training, from here

Not started. `--verify-policy` and `--check` are green, so the scene and the
baseline are sound; what remains is the RL loop, which needs jax + brax +
mujoco_playground on a GPU box.

Suggested order:

1. **Port the env.** Rewrite `climb_env.py` on top of `climb_scene`. The ratchet
   is deliberately branch-free (an `argmin` and a clipped `searchsorted`) so it
   drops into an MJX `lax.scan` where the old `qpos` clamp lived.
2. **Warm-start from the walking policy.** `policies/mels_g1_joystick.npz` holds
   `hidden_{0..3}_kernel/bias` for an MLP 103→512→256→128→58 plus
   `obs_mean`/`obs_std`. Build the brax network with
   `policy_hidden_layer_sizes=(512, 256, 128)` so the pytree matches, copy the
   kernels in, and seed the normaliser from mean/std (brax stores summed
   statistics with a count — set the count high or the first batch washes the
   prior out). The value network starts fresh. **This mapping has not been
   executed**; verify the shapes before trusting it.
3. **Keep the observation at 103 dims.** Widen it and the checkpoint no longer
   loads and the warm start is gone. Feed rope progress through the *reward*.
4. **Randomise** with the table above. Start slope at 25 deg and curriculum up;
   the baseline already fails there, so there is signal immediately.

## Provenance — read before claiming realism

Patch *slope angle and location* are measured from Copernicus GLO-30. Everything
finer than ~30 m is synthetic: a 25 x 15 m patch covers **0.447 of one DEM cell**.
Randomising roughness explores a synthetic noise family, not observed Everest
micro-terrain. The `curriculum/` patches additionally have their slope
**overridden** — `B_slope45` is not the Lhotse Face at 45 degrees, and
`B_slope0` is a flat reference for checking a controller, nothing more.

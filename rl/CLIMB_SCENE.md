# The merged climb scene

One MuJoCo model containing the G1, the Lhotse Face terrain and the fixed rope,
plus the four randomisation axes the climbing policy is meant to be robust to.

```bash
python -m rl.scripts.climb_scene --list                       # available patches
python -m rl.scripts.climb_scene --check                      # acceptance suite
python -m rl.scripts.climb_scene --patch B --export build/climb_B.xml
python -m rl.tools.fetch_visual_assets                        # G1 meshes
python -m rl.tools.usd_gear                                   # jacket/boots/ascender from the USD
python -m rl.scripts.climb_scene --visual --gear --view       # the project's dressed G1
python -m rl.scripts.climb_scene --visual --render build/vis  # preview stills
python -m rl.scripts.climb_scene --visual --view              # interactive viewer
```

### Viewer on macOS

Use plain `python3`. `--view` calls `mujoco.viewer.launch` and drives the ratchet from a
`set_mjcb_control` hook, because `launch_passive` refuses to run outside `mjpython` on macOS
and `mjpython` crashes on this machine (Cocoa `NSException` in the `_Simulate` constructor).
`launch` owns the stepping loop and takes no per-step argument, so without the control
callback the carrier would never move and the hand would be welded to a fixed point.
`--passive` selects the `launch_passive` path if you are running under a working `mjpython`.

## What was merged

| piece | was | now |
|---|---|---|
| robot | playground G1, plus an Isaac-only `g1_unitree.usd` | `assets/robots/mujoco/g1_unitree_ascender.xml` |
| terrain | `terrain/patches/*.npz`, only ever viewed standalone | the model's `floor` geom |
| rope | a straight cylinder in `climb_env.py`; a separate polyline in `terrain/mujoco_scene.py` | one polyline draped on the terrain |
| ascender | slide joint + qpos clamp | mocap carrier + connect equality + arc-length ratchet |

`rl/environment/terrain.py` (terrain), `ascender.py` (rope + ratchet),
`climb_scene.py` (assembly). The first two are plain numpy and import without
jax, so the scene builds and runs on a laptop.

## The ascender

The old mechanism was a slide joint along a straight cylinder. A slide joint has
one axis, so it cannot follow a rope that drapes over terrain — which is why
`climb_env.py` refuses anything but a flat plane it can tilt.

The replacement is a **mocap carrier**: each substep the palm is projected onto
the polyline to get its arc length `s`, `s` is clamped non-decreasing, and the
carrier is written to `polyline(s)`. A `connect` equality between `right_palm`
and the carrier does the physics — perpendicular to the rope it holds the hand
on the line, along the rope the carrier tracks the hand, and after a slip the
high-water mark hauls the hand back up. That is an ascender.

Because a mocap body has no degrees of freedom, **`nq` stays 36**. The slide
joint made it 37, which is why `climb_env.py` carries trimmed `qpos[7:...]`
slices and eight `_cost_*` overrides to hide the phantom coordinate. None of
that is needed here, and the observation stays natively 103-dimensional, so the
`mels` baseline policy still loads.

## The robot

Default is **`assets/robots/mujoco/g1_unitree_ascender.xml`** — this project's G1
with jacket, boots, logos, D435i/Mid-360 sites and the ascender on the right
wrist. `--robot playground` selects mujoco_playground's G1 instead, the model
the mels walking policy was trained in, for comparison.

The earlier USD route is superseded. `assets/robots/g1_unitree.usd` is an Isaac
asset and MuJoCo cannot load USD as a model; `rl/tools/usd_gear.py` extracted its
gear meshes as a workaround, and `--gear` still does that for the playground
robot. The MJCF does it properly and needs none of it.

### What the scene adds to the robot file

It ships as a bare robot with no scene, so `rl/environment/robot.py::adapt()`
supplies what the rope and the policy need, each only if absent:

| added | why |
|---|---|
| `floor` geom | no ground in the robot file, by design |
| `right_palm` site | grip anchor; at `[0.08, 0, 0]` on the wrist it lands on the ascender's rope channel |
| `local_linvel_pelvis`, `gyro_pelvis` | the 103-dim observation reads them; the file has only 5 raw IMU sensors |
| `knees_bent` keyframe | the pose the policy's action deltas are about; the file ships `stand` |

### `policy_compat` — on by default, and not cosmetic

The himalaya model uses stock menagerie dynamics; playground retuned them for
RL, and the mels policy learned against *that* plant:

| | playground | himalaya (stock) |
|---|---|---|
| actuator kp | 75 / 20 / 2 per joint | 500 uniform |
| actuator kv | 0 | −17 to −43 (dampratio 1) |
| dof damping | 0.2 / 1.0 / 2.0 | 0 |
| dof armature | 0.0036–0.0251 | 0.01 uniform |
| dof frictionloss | 0.1 | 0.3 |
| foot contact | one 0.18 × 0.06 m box | four 5 mm spheres |

`policy_compat=True` writes playground's values over all of these. `--stock-plant`
keeps the robot exactly as this project specifies it — and the walking policy
then falls at ~1.8 s on flat ground, because it is driving a plant it never saw.

Measured on flat ground, command 0, 8 s:

| robot | result |
|---|---|
| playground | stands indefinitely, upright +0.99 |
| himalaya, `policy_compat` | **stands indefinitely, upright +0.99** |
| himalaya, `--stock-plant` | falls at 1.8 s |
| himalaya, cmd 0.5 m/s | walks 3.64 m in 8 s (playground: 3.59 m) |

The foot contact model was the last and largest gap: with everything else
matched, sphere feet still fell at ~3 s and the box stands. Four point contacts
is arguably the better foot model — this is a compatibility shim for the warm
start, not a claim that the box is more correct.

### Verified equivalences

- 103-dim observation **bit-identical** between the two robots at the same state
- actuator order identical, all 29
- every body mass identical except `right_wrist_yaw_link`, +0.1 kg (the ascender)
- `right_palm` at `[0.1436, -0.2268, 0.6635]` in `knees_bent` on both

Recorded and deliberately **not** patched: `right_hip_roll_joint` range is
[−2.9671, 0.5236] here (stock menagerie, correctly mirrored from the left leg)
against playground's [−0.5236, 2.9671] (unmirrored). Both contain the working
range; rewriting a joint limit to match a quirk of the training model would be
the wrong repair.

## Slope and roughness are separable

`terrain/build_patch_set.py` builds every patch as `Z = U*tan(slope) + noise(seed)`.
A least-squares de-plane recovers the noise exactly (residual RMS 0.11 m; ~0
correlation between patches, since each uses its own seed). So the heightfield
carries **roughness only** and slope is the terrain geom's quaternion.

Contrary to the comment in `climb_env.py`, MuJoCo heightfield geoms *do* respect
orientation — a rotated hfield collides correctly. `--check` verifies the
compiled surface against `terrain.surface_z()` by raycast, to ~2 mm.

## The four axes

| axis | field | per-env cost at 4096 envs |
|---|---|---|
| slope | `geom_quat` of `floor` | 0.066 MB |
| surface variation | `hfield_size[2]` scales the shared grid linearly | 0.066 MB |
| friction | `pair_friction` + `geom_friction` + priority geoms | negligible |
| wind | `xfrc_applied` on the torso (data, not model) | — |

Roughness *amplitude* is one float per env because the grid is normalised to
[0,1] and `hfield_size[2]` is its elevation scale. Roughness *realisation*
(the seed) is a whole new grid, so vary it between training iterations rather
than across envs: batching `hfield_data` would cost **2.46 GB** at 4096 envs.

## Is the policy actually loaded?

```bash
python -m rl.scripts.climb_scene --verify-policy
```

Checkpoint shape and hash, forward-pass determinism, all six observation-layout
invariants, a 30 s stand on flat ground against a no-policy control, and
body-frame velocity tracking. Every line is a number with a verdict, because
"the robot fell" has at least four unrelated causes here -- no policy running, a
broken observation, an incompatible plant, or terrain the policy cannot handle
-- and they look identical from the viewer.

Measured: stands 30 s at upright +1.00 (no-policy control topples to −0.16);
body-frame vx 0.52 m/s at cmd 0.6 and 0.66 at cmd 1.0, against the documented
0.75 at cmd 1.0. Commands below ~0.4 do not initiate a gait, the policy just
stands.

Note the command is **body-frame** forward velocity. Yaw drifts freely, so
world-frame x displacement is not a measure of tracking — measure
`local_linvel_pelvis`.

## Testing it by hand

Start flat and add one axis at a time. `--slope 0` synthesises flat ground;
there is no `B_slope0` patch (`--list` shows what exists).

```bash
python -m rl.scripts.climb_scene --patch B_slope0 --view               # stands indefinitely
python -m rl.scripts.climb_scene --patch B_slope0 --cmd-x 1.0 --view   # walks at 0.66 m/s
python -m rl.scripts.climb_scene --patch B_slope25 --view              # falls at ~7 s
python -m rl.scripts.climb_scene --patch B --view                      # falls at ~2 s
```

`B_slope0` is the flat reference, generated by `python -m rl.tools.make_flat_patch`
with the same octave recipe and seed convention as the rest of the family
(`--slope 0` synthesises equivalent terrain without a patch file).

**Do not use `--ice 0` for a policy sanity check.** It floors to 0.01, which is
an ice rink: the robot cannot stand on it even on level ground, because there is
nothing to push against. Measured on `B_slope0` over 20 s: mu 0.9 and 0.2 stand
at upright +0.99; mu 0.05 and 0 collapse to +0.25. That run looks exactly like a
broken policy and is not one.

Isolating the axes on the roped baseline, 8 s, command 0: **flat ground survives
everything** — ice 0.1 *and* 20 m/s wind, never falls. **Every slope from 25 deg
up falls, dry and windless.** Slope alone is what defeats the flat-ground
policy; ice and wind only change how quickly. In each case the hand stays on the
line and the descent saturates around −0.7 m: it is hanging off the ascender,
not tumbling down the face.

### The flag is `--friction`, not `--ice`

It was `--ice`, which reads backwards: `--ice 0` looks like "no ice" but sets
*zero friction*, the slipperiest setting available. Renamed to `--friction`;
`--ice` still parses and is **not** inverted.

Below `STAND_FRICTION = 0.20` the robot cannot stand on level ground at all —
measured on the flat patch over 20 s: 0.20+ never falls, 0.15 falls at 4.4 s,
0.10 at 1.1 s, 0.01 at 0.6 s. The CLI warns when a run is below it, because such
a run says nothing about the policy.

### Friction has a floor

`--friction 0` is raised to 0.01. A condim=3 contact with friction exactly zero has a
zero-width friction cone and the solver diverges within ~30 ms. The NaN appears
at whichever DOF is lightest — here a wrist — so it reads as a rope-attachment
fault. It is not: it reproduces with the grip equality disabled, on flat ground,
with no policy running. Every non-zero value tested, down to 0.001, is stable,
and zero is not a physical surface anyway (rubber or steel on wet ice is
0.02–0.10). The clamp is reported whenever it fires.

## The reset pose leans on a slope

Standing gravity-upright on a slope makes the ankle absorb the entire slope
angle. `knees_bent` already sits at −20.8 deg and the G1's ankle pitch stops at
−50 deg:

| slope | ankle needed to stand upright | foot contacts at reset |
|---|---|---|
| 0 deg | −20.8 | 7 (soles flat) |
| 25 deg | −45.8 | 1 |
| 38.6 deg | −59.4 (past the stop) | 1 |
| 45 deg | −65.8 (past the stop) | 1 |

Past ~29 deg the robot balances on a foot *edge*. That is kinematic, not
something a policy can learn around, and it presented as "falls instantly on
every steep patch". The reset pose now splits the slope between a base lean and
the ankle pitch (45 deg becomes 18.8 deg lean, −47 deg ankle), which puts the
soles back down; `--lean-frac` overrides the split.

The policy's `default_pose` stays pinned to the training `knees_bent` and is
deliberately NOT read from the scene keyframe — otherwise the operating point
its actions are defined about would drift with the terrain.

## Bugs this surfaced

Each of these compiled, ran, and looked plausible while being wrong.

- **Foot friction was inert.** The G1 XML declares
  `<pair name="left_foot_floor" friction="0.6 0.6"/>`, and an explicit contact
  pair overrides both geoms' friction. Setting `geom_friction` alone — which is
  what `climb_env.py`'s `foot_friction` config knob does — leaves the real
  coefficient pinned at 0.6. **`climb_env.py` still has this bug.**
- **Stale kinematics at reset.** `mj_resetDataKeyframe` writes `qpos` but leaves
  `site_xpos` stale, so placing the carrier from the palm read the *previous*
  pose — a 10 m equality error that detonated the solver on step one.
- **Heightfield asset cache.** MuJoCo caches assets by path; two terrains
  sharing a filename both got whichever grid compiled first, off by the whole
  macro slope. Files are now named by content hash.
- **Baked geom offset sign**, and a dropped `h*sin(slope)` term in the separable
  inverse mapping (~0.2 m at 38°, enough to hang the rope underground).
- **Rope invisible.** Geom group 3 is hidden by MuJoCo's default view mask.

## Why the robot falls without `--no-policy`

It doesn't, any more. `--view` now runs the mels walking policy by default
(`rl/environment/walk_policy.py`, pure NumPy, ~50 us/step against a 20 ms
budget). `--no-policy` restores the old behaviour: `d.ctrl` pinned to the
keyframe, i.e. 29 position servos holding fixed angles with no balance
feedback. That is not a controller and topples on flat ground in ~1.3 s, so a
fall in that mode says nothing about the terrain or the rope.

Measured over 6-8 s, command 0:

| terrain | control | result |
|---|---|---|
| flat | keyframe hold | topples at 1.3 s, upright −1.00 |
| flat | mels policy | **dz −0.01 m, upright +0.98, never falls** |
| flat | mels policy, cmd 0.5 m/s | walks 3.5 m in 8 s, stays upright |
| 38.6°, unroped | mels policy | falls at 0.9 s, slides 181 m off the face |
| 38.6°, roped | mels policy | slips, then **hangs from the rope**, dz −0.83 m |

Standing on flat ground is the check that the observation layout is right — the
policy was trained against exactly this 103-dim vector, and a single misplaced
slice makes it fall. The 38.6° result is the honest baseline: a flat-ground
walking policy cannot stand on the Lhotse Face, the rope catches it, and closing
that gap is what there is to learn.

## Initialising RL from the walking policy

`rl/policies/mels_g1_joystick.npz` holds `hidden_{0..3}_kernel/bias` for an
MLP 103→512→256→128→58 plus `obs_mean`/`obs_std`. To warm-start PPO, build the
brax network with `policy_hidden_layer_sizes=(512, 256, 128)` so the parameter
tree matches, then copy the kernels/biases in and seed the running normaliser
from `obs_mean`/`obs_std` (brax stores mean/std as summed statistics with a
count, so the count has to be set to something large or the first batch washes
the prior out). The value network has no counterpart in the npz and starts
fresh. This mapping has NOT been executed -- brax is not installed here -- so
verify the parameter pytree shapes before trusting it.

Keep the observation at 103 dims when fine-tuning. The merged scene's carrier is
a mocap body with no DOFs precisely so `qpos[7:]` stays the 29 robot joints; add
rope state to the observation and the checkpoint no longer loads.

## Not yet done

Training. The env classes in `rl/environment/{climb_env,wind_env}.py` still use
the old slide-joint mechanism and the flat tilted plane; porting them onto
`climb_scene` is the next step, and needs jax + `mujoco_playground` on a GPU box
(none installed here). The ratchet is written branch-free (`argmin` and a
clipped `searchsorted`) specifically so it drops into an MJX `lax.scan`.

## Provenance — read before claiming realism

Patch *slope angle and location* are measured from Copernicus GLO-30. Everything
finer than ~30 m is synthetic: a 25 × 15 m patch covers **0.447 of one DEM cell**.
Randomising roughness explores a synthetic noise family, not observed Everest
micro-terrain. The `curriculum/` patches additionally have their slope
**overridden** — `B_slope45` is not the Lhotse Face at 45°.

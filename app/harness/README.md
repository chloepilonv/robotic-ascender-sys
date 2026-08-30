# app/harness — the interactive climber, in the browser

Runs the team's **merged climb scene** (`rl/environment/climb_scene.py`) in
plain MuJoCo at 50 Hz on a laptop and streams it to the browser: the Lhotse
Face heightfield, a rope draped over it, the jacketed G1 with the ascender on
the line. Nothing physical is re-implemented — `build_scene` builds it,
`ClimbScene.step(wind)` is the physics step, and `walk_policy.WalkController`
is the policy. See `PARITY.md` and `fingerprint_lhotse_B.json`.

The older flat-plane env (`rl.environment.climb_env.G1ClimbAscender`) is still
reachable as the four `legacy_*` worlds, because the trainer still uses it.

## Run it on localhost

From the repo root, in the `everest` env (the one that runs the trainer/viewer —
it already has `playground==0.2.0`, `mujoco==3.12.0`, `brax==0.14.2`):

    pip install websockets pillow          # the only extras the harness needs

    python -m app.harness.runtime --live --world lhotse_B

then open **http://localhost:8766/** (the same page is also at
`/app/web/render3d.html`). Click the view to take
the pointer, hold **W** to walk/climb, move the mouse to look around, **R**
resets, **Esc** pauses. Map selector, twelve **ClimbScene** worlds then four **legacy** ones:

| world | what |
|---|---|
| `lhotse_B` | the default — measured Lhotse patch B, 38.6°, jacketed robot, roped |
| `lhotse_A` / `C` / `D` | the other measured patches, 33.7–36.2° |
| `flat_0` | B's roughness with the macro slope removed — the walking reference |
| `slope_25/30/35/45/50` | patch B with a synthetic slope override (curriculum) |
| `lhotse_B_free` | patch B with the grip equality off |
| `lhotse_B_playground` | patch B with the bare Playground G1, for comparison |
| the four `legacy_*` | the old flat tilted plane + slide joint, which the trainer still uses |

Also `terrain_free_5/10/15/20/25/30` (patch B's measured roughness re-tilted,
rope off — the walker gives up between 10° and 15°) and `sandbox_free` /
`sandbox_rope`, a 120 × 120 m free-roam map.

And two worlds that are **not the walking policy at all**:

| world | what |
|---|---|
| `chloe_v1_20` | Chloe's mjlab rope-ascender network on her own plant, 20° |
| `chloe_v1_25` | the same, 25° |

See "Chloe's ascender policy" below — different plant, different brain,
different controls.

Keys: **W** walk, **A/D** turn in place (they suspend the camera-follow while
held) — *unless the `guide` knob is on, in which case W walks the HUMAN up the
rope, S walks her back down it, and the robot drives itself; see "Follow the
guide" below.* Wind dial is in m/s and goes through **his** `WindParams` into
`ClimbScene.step`; the `wind_natural` knob turns the dial into a target that
gusts and drifts. Neither the wind nor this terrain is in training yet — the
trainer is still on the old `climb_env` — so the state message keeps saying
`wind_in_training: false` and `terrain_in_training: false`.

Every live session is recorded under `app/harness/episodes/<stamp>_<world>/`
(frames.npz, hud.json, header.json) — the numbers, not a video.

Other entry points:

    python -m app.harness.runtime --world lhotse_B --duration 10 --hold-w          # headless case
    python -m app.harness.climb_worlds --world lhotse_B                                  # parity gates + fingerprint
    python -m app.harness.runtime --live --policy path/to/policy.npz                    # a trained policy (mels npz layout)
    python -m app.harness.test_parity                                                   # obs + rollout parity vs the JAX env

`--port` moves the websocket (HTTP is port+1); `--command-speed` sets the W
forward command (default 0.5 m/s).

### The page draws the scene; the harness sends numbers

`--pose-stream` (ON by default, `--no-pose-stream` to turn it off) broadcasts
every body's world pose once per control tick — ~1.1 kB, ~56 kB/s, a measured
20 us of a 20 ms tick. **That is what the page draws the world from**, so with
it off you get telemetry and an empty stage. It needs the world exported once:

    python -m app.harness.export_scene --world lhotse_B      # or --all

writes `app/harness/scene_assets/<world>.glb` plus a JSON sidecar (gitignored;
lhotse_B is 18.5 MB). The map selector only offers worlds that have been
exported.

**RETIRED 2026-08-30 (user's ruling: "we're only gonna do 3D now").** There used
to be a second page, `app/web/index.html`, fed by an offscreen chase-camera
render the harness did every control tick and pushed down the socket as a raw
JPEG — plus an `episode.mp4` muxed from those frames. All of it is deleted: the
render, the JPEG frame, the browser-viewport negotiation, the video, and the
footprint stamping into the ground texture (the 3-D page draws its own decals
from the `foot_steps` events). That render was 10–20 ms of a 20 ms budget. The
one image still on the wire is `EYE0`, the robot's own left eye — a sensor
readout, not a view of the world.

### Two wind pennants, and who the camera is on

**A pennant on each head** (`app/web/three/flag.js`, user's ruling 2026-08-30 —
"the flag needs to be a lot bigger", and "put the same flag on the human too").
A 45 cm pole carrying a 36 × 24 cm tapered red cloth sits on the robot's
`torso_link` and on the hiker's `guide` mocap root. It points downwind, lifts
from hanging limp at 0 m/s to flying level by 12, and ripples faster the harder
it blows — a wind indicator you can read at a glance from anywhere the camera
happens to be.

Three times the old size, so the clearance was recomputed rather than carried
over. From `model.cam_pos` / `cam_quat` / `cam_fovy` on the compiled `flat_0`:
the eyes' half diagonal is **42.7°**, the pole's base sits 131.4° off the
optical axis, its top 98.0°, and the **worst point of the swept cloth** — over
every lift angle and every wind heading — is **56.2°**. 13.5° of margin, so even
an in-model version of this flag could not get into the robot's eyes. It is
browser-only decoration either way: no MuJoCo body, no geom, no mass.

The hiker's flag needs no visibility rule — with the guide knob off, `guide.py`
parks her whole mocap body at z = −50 m and the flag, being a child of that
node, goes with her.

**The camera follows the subject** (user's ruling 2026-08-30: "when the guide is
turned on the camera should be on the guide, not the bot"). Guide on, the chase
boom frames HER and **V** gives first person from HER head — with no yaw clamp
at all, because a person can look over her shoulder and the robot's ±150° limit
is the waist joint's real travel. Guide off, both cameras are the robot's again,
exactly as before. The switch is instantaneous: toggling re-seeds the spring arm
onto the new subject rather than flying it across the map. W/S/A/D are
unchanged, and the two numbers that go up the socket are still the CHASE
camera's, so nothing about steering moved.


## Follow the guide (`app/harness/guide.py`)

Turn the **`guide`** knob on and a human guide appears 2.5 m up the rope —
**Chloe's hiker** from `assets/humans/human.xml`, loaded whole and re-parented
into six moving limbs. Hold **W** and the *human* walks **up** the rope at
1.0 m/s; hold **S** and she walks **back down** it toward the robot, still
facing uphill, animation running in reverse; hold both and she stands. She is
0.6 m to the rope's left, snapped to the terrain, and she never slips or falls,
because she is a mocap body with no degrees of freedom and no collision. The
**robot decides for itself** what to do about that. A/D do nothing while the
guide is on: the follower owns the yaw command, and the camera-follow
controller stands down.

**Off the rope, all four keys are HERS** (user's ruling, 2026-08-30). On any
world with `rope=False` — `flat_free`, `terrain_free_*`, `sandbox_free`,
`lhotse_B_free` — W and S walk her forward and back along HER OWN heading and
**A and D turn it**, at 70°/s. Those two keys were already doing nothing while
the guide was on (the follower owns the robot's command), so this hands two idle
keys to the one body that can use them, and the legend says
`W S A D move the guide`. On a roped world nothing changes: she is on the line,
forward is up it, and A/D do nothing to her. She is initialised from the rope
route in both cases, so the spawn is identical and only the driving differs.

**The walk is distance-locked**, which is why the feet do not skate: the gait
phase is `2π × travel / 1.05 m`, a function of how far she has WALKED and never
of the clock, so one stride of ground is exactly one stride of animation
whatever the speed, and S runs the same cycle backwards. Within a stride the
planted foot is placed by the same arithmetic in reverse — its offset from the
hip ramps from +stride/4 to −stride/4 while the root advances stride/2, so the
two cancel. MEASURED on `flat_0`: the planted boot drifts **0.7 mm** per stance
phase, swing clearance is 8–13 cm, and the lowest boot corner sits between
−2.6 cm and +2.2 cm of the snow surface (on terrain whose roughness is 10.9 cm
rms). Stride is set from the CADENCE, not chosen: 1.05 m at 1.0 m/s is
114 steps/min, a brisk walk; the 0.70 m it was first drawn with gave
171 steps/min, which is a jog, and it looked like one.

    python -m app.harness.guide_walk_sheet     # the contact sheet + the skate audit

writes `render3d_shots/guide_walk_sheet.png` — eight frames across one gait
cycle with the hip/knee/shoulder angles printed on each — and the audit table.

**Her limbs have no joints, on purpose.** Six child bodies hang off the mocap
root (thigh and shin either side, left upper arm and forearm; the right arm is
posed gripping the rope and does not swing), and each is turned by writing
`model.body_quat` every control tick rather than by a hinge. The first version
used real hinges, which grew `nq` 39→45 — and six degrees of freedom the solver
carries is enough to move the walking robot **23 cm in six seconds** through
nothing but floating-point. `PARITY.md` has that number and the fix.

The robot has two RGB cameras on its head, 6 cm apart, at the `d435i` mount
already in the jacketed MJCF. Ten times a second they are rendered at 320×240
and OpenCV's semi-global block matcher turns the pair into a disparity map;
`depth = focal_pixels × baseline / disparity` gives metres. **The distance is
real passive stereo — nothing in the decision path reads the simulator.** Which
pixels are the human is a **stand-in** (a colour threshold on the guide's
orange BACKPACK, where a person detector would go — see the next subsection);
`PARITY.md` has the full ledger of what is vision, what is stand-in and what is
a labelled cheat.

Three states, with hysteresis so it cannot chatter at the threshold it is trying
to hold:

| mode | when | command |
|---|---|---|
| `FOLLOW` | range > 1.3 m, or > 1.0 m while already following | `lin_vel_x` 0.5, `ang_vel_yaw` = clamp(2·bearing, ±1) |
| `WAIT` | range ≤ 1.0 m, or ≤ 1.3 m while already waiting | zero |
| `LOST` | nothing detected for a whole second | zero |

The page gets a `guide` block in every state message and, at the vision rate, a
second binary message — the four bytes `EYE0` then a JPEG of the left eye with
the detection box drawn on it. Episodes record `guide_mode` (0 WAIT, 1 FOLLOW,
2 LOST), `guide_distance_meters`, `guide_true_distance_meters` and
`guide_human_progress_meters`.

    python -m app.harness.test_guide     # colour window, stereo, two rollouts, physics parity
    python -m app.harness.runtime --world flat_0 --duration 20 --hold-w --guide

**Measured, and worth knowing before you demo it.** The colour window keeps
**97.7%** of her jacket's pixels and **0.0%** of every other material in the
scene and of the scene itself (`flat_0`, pooled over 1/2/4/8 m, attributed with
a segmentation render — `test_guide` section A0). The stereo reads −5.8% at 2 m
and +0.0% at 4 m against the simulator's own answer. The *walker*, not the
follower, is the limit: its real ground speed is about **0.15 m/s** whatever
`lin_vel_x` says, and the guide now walks at 1.0 m/s, so holding W indefinitely
opens the gap at ~0.85 m/s and she is out of the ±29° field of view within
three seconds. **The demo is: tap W for two or three seconds, then let go**, or
hold **S** and let her walk back to the robot. Yaw authority is also nearly nil
while the palm is gripping the rope (commanding +1 rad/s and −1 rad/s for 3 s
ends up 10° apart), so the follower steers much better on rope-off worlds.

The guide is available on the **ClimbScene** worlds only. The four `legacy_*`
worlds hand back a compiled model with no `MjSpec`, so there is nothing to add
the body and the cameras to, and the feature turns itself off there.

### The guide's outfit, and why the detector keys on her backpack

She wears a **cobalt jacket and navy trousers**, slate boots and a teal beanie,
and carries a **bright safety-orange backpack** (user's ruling, 2026-08-30). The
colours are applied at attach time by `guide.GUIDE_OUTFIT_RGBA`, where her
materials are copied onto the scene spec — `assets/humans/human.xml` is shared
with `human-safety/` and is never edited for a wardrobe choice. Only `rgba` is
overridden, so the compiled model, the GLB the browser draws and the eye images
the robot detects with all read one source, and `test_guide` section D still
comes back **bit-identical**.

The detector therefore targets the **pack**, not the jacket. Choosing the
clothes and choosing the hue window is one decision: the pack is the only
saturated orange on the person, and everything else is pushed away from it in
hue, saturation or value with margin. The boots left brown for this reason —
brown renders at hue 12–14, inside the pack's own window.

**The window is `hue 9–14, saturation ≥ 180, value ≥ 40`** (OpenCV HSV), and it
keeps **100.0%** of the pack's pixels and **0.0%** of every other material and of
the scene itself (`flat_0`, pooled over 1/2/4/8 m, attributed with a
segmentation render — `test_guide` section A0; `terrain_free_10` is the same
verdict):

| material | pixels | hue 1-99% | saturation 1-99% | value 1-99% | inside the window |
|---|---|---|---|---|---|
| pack ← TARGET | 6380 | 11-12 | 246-250 | 51-239 | 100.0% |
| jacket | 21546 | 111-113 | 204-219 | 60-198 | 0.0% |
| skin | 563 | 0-178 | 39-74 | 52-223 | 0.0% |
| beanie | 6395 | 86-91 | 224-237 | 47-170 | 0.0% |
| glove | 25 | 110-116 | 73-118 | 11-31 | 0.0% |
| pants | 4301 | 114-115 | 181-208 | 26-92 | 0.0% |
| boots | 519 | 113-113 | 109-124 | 33-70 | 0.0% |
| rope carrier (orange, on the palm) | 0 | | | | not visible |
| everything else | 267471 | 1-111 | 33-238 | 74-230 | 0.0% |

Margins, so it is clear nothing sits on an edge: hue is 2 units either side of
the pack's own 11–12; saturation is 66 clear of its 246 and is the barrier that
keeps **skin** out, whose hue wraps straight through the band; value is 11 clear
of the pack's darkest face. **The other reds** (OpenCV hue = degrees ÷ 2): the
**rope** `0.85 0.08 0.05` computes to 2.25° → **hue 1**, eight units below the
window, and hue is its only barrier since it clears both floors. The browser's
**wind pennants** `0xc41414` (196,20,20) compute to 0° → **hue 0**, nine units
below. The pennants are **Three.js-only decoration** (`app/web/three/flag.js`);
they exist in no MuJoCo model and can never appear in an eye image at all. The
**ascender carrier**, translucent orange on the robot's own palm, computes to
**hue ≈ 17** — three units above the window's high end, which is why that end
stops at 14 rather than opening up. It renders 0 px in the test poses, so it is
out of frame rather than absent.

**What the smaller marker cost and bought.** Measured `flat_0`, jacket → pack:

| | jacket | pack |
|---|---|---|
| maximum detection range, `flat_0` | 10.00 m | **18.25 m** |
| maximum detection range, `terrain_free_10` | 10.00 m | **15.87 m** |
| stereo error at 2 m / 5 m | −5.8% / −6.0% | −7.0% / −4.4% |
| standing at 2 m, back turned | 100% detected, WAIT at 0.89 m | 87.0% detected, WAIT at 0.78 m |
| standing at 5 m, back turned | 79.5% detected, ends LOST | **99.0% detected, ends in FOLLOW** |
| off-axis re-acquire (the retired `test_search` J) | 0.20 s / 0.40 s | 0.20 s / 0.60 s |

The range went **up**: the pack is small but it is a solid, uniformly-lit box,
where the jacket's thin limbs anti-aliased into the snow. Detection is not the
same as a usable reading, though — past about 10 m the disparity is 1–1.5 px and
the range wanders (12 m true reads 13.07 m; 16 m true reads 13.05 m).

**Two limits the backpack has and the jacket did not**, both measured rather
than left for a demo to find (`test_guide` sections A1, A2, A2b):

* **She is invisible facing the robot.** 0 mask pixels at 2 m and at 5 m, and
  the follower goes **LOST** rather than inventing a range — 0.0% of ticks
  for a whole 20 s run at 5 m on both worlds, and at 2 m on `terrain_free_10`.
  The one exception is honest and is printed as a first-detection range: at 2 m
  on `flat_0` the robot walks *blind* from 2.00 m down to 1.01 m with **0 mask
  pixels the whole way**, and only sees the pack's edge at 0.98 m once it has
  got round her shoulder.
* **A close-range hole on the approach.** She walks 0.6 m left of the rope, so
  as the robot closes inside about 1.5 m the bearing to her crosses the ±29°
  frame edge, and a narrow marker on her back goes with it where a whole jacket
  still filled the picture. The retired camera sweep used to recover from this
  (WAIT reached at 13.7 s from a 2 m start); with it gone the follower simply
  goes LOST at close range and the ear layer's `LISTENING` waits for the next
  shout, which is the whole point of having ears.

Physics is untouched by any of it. The whole executable change is one statement
— the material's `rgba` — so no geometry, mass, joint or contact property moved,
and `test_guide` D and `test_hearing` 5 both report **0.000e+00 across every
array**.

## Hearing: she calls, the robot comes (`app/harness/hearing.py`)

**The camera sweep is gone** (user's ruling, 2026-08-30). The follower used to
answer "I have lost her" by swinging its waist through a 20/60/90° ladder
hunting for orange, then running an acquire/realign hand-over back to FOLLOW. It
worked and it was measured — `flat_0` roped: 0.20 s to acquire, hand-over at
0.24 s, 10.2° of camera-bearing error; `terrain_free_10` rope off: 0.40 s and
1.00 s; the 60° waist clamp is what survived that experiment, and it is still
`guide.WAIST_LIMIT_RADIANS` — and it is deleted, along with `test_search.py`,
because the robot now has **EARS**. A machine that has lost the person it is
following should not wave its torso about hoping. It should hold still and wait
to be called, and the next shout hands it a direction a camera sweep never
could. LOST is now `LISTENING`.

### The pipeline, and where the honesty line is

Turn the **`hearing`** knob on (it needs `guide` on too — the voice comes out of
HER mouth and the hand-over is to HER eyes) and:

1. **The page supplies the voice, in one of two modes**, and both go down the
   socket as `MIC0` + 16 kHz mono int16 PCM, block by block, in real time — so
   the runtime cannot tell them apart and neither can skip the ear model.
   **MANUAL** (the default; needs no permission) plays a recorded shout from
   `app/web/sounds/voice/`: **PEMBA** picks one of `pemba_1..4.mp3` at random and
   **STOP** plays `stop_1.mp3`. The card names the clip while it plays and both
   buttons are disabled for its duration.
   **MIC** streams the Mac microphone. The browser does **no** processing:
   `getUserMedia` is asked for `autoGainControl: false`, `noiseSuppression:
   false`, `echoCancellation: false`, and the runtime never normalises what
   arrives. That is deliberate and it is testable — **a shout has to carry
   further than a mumble** (table 1d), and automatic gain would flatten exactly
   that difference before the runtime ever saw it. The incoming rms and peak are
   printed once a second and the card draws a meter from them.
   `--inject-voice PATH[@SECONDS]` is the same thing from the command line: it
   pushes into the very same ring the browser pushes into.
2. **The ears are synthesised from truth positions**, exactly as `StereoEyes`
   renders two pictures from truth geometry. The voice is emitted at the
   **hiker's mouth** (a point on her head, read off her own `human_head` geom)
   and received by **four virtual microphones on the robot's head** —
   front/back/left/right, 7 cm out, on `torso_link`, the same body the eyes ride.
   Per microphone: `1/r` gain, a **fractional propagation delay** (c = 330 m/s),
   a gentle air-absorption low-pass, and **wind noise**. Every decision after
   that reads the four ear signals and nothing else.
3. **Three detectors, all at 10 Hz on a rolling buffer** —
   **voice activity** is `webrtcvad` (Google's WebRTC VAD, the one telephony
   ships) on 30 ms frames. *Not Silero*: Silero is a torch model and this venv is
   JAX/brax with no torch, and pulling 250 MB of torch into a 50 Hz control loop
   to answer a yes/no question a 158 kB C library answers in 40 µs is the wrong
   trade. Its verdict is binary per frame, so the "probability" reported is the
   **voiced-frame share** of a 480 ms window — an honest fraction, not a
   confidence dressed up as one.
   **the word `stop`** is `vosk` small English (`vosk-model-small-en-us-0.15`,
   40 MB) with a **grammar of exactly `["stop", "[unk]"]`**. The grammar is what
   makes a small model usable: the decoder can only emit `stop` or garbage, so
   "come here" cannot be scored as a partial match and the confidence attached to
   `stop` is a real posterior over a two-way choice. Run **once per utterance**,
   on the whole segment.
   **bearing** is GCC-PHAT between the front/back and the left/right pairs. Four
   microphones, not two, so there is **no front/back ambiguity**: one opposed
   pair gives `u_x`, the other gives `u_y`, and the azimuth is `atan2(u_y, u_x)`
   in the robot's body frame (+ = to its LEFT, the same sign the eyes and
   `ang_vel_yaw` use). Confidence is the correlation peak's sharpness.

### The behaviour, layered on the follower

| mode | when | what it commands |
|---|---|---|
| `IDLE` | nobody has called yet | zero |
| `COMING_BY_EYES` | called, and the eyes have her | the follower's own command, byte for byte |
| `COMING_BY_EARS` | called, the eyes do not have her, an ear cue does | walk, and turn toward the remembered heading |
| `WAIT` | the follower reached its 1 m band | zero; the next shout re-calls |
| `LISTENING` | called, no eyes, no cue worth steering by | zero — stand still and wait to be called |
| `STOPPED` | a confident `stop` | zero, latched |

**Voice is a trigger, not a leash.** One shout starts the walk and silence does
not stop it; the walk ends at the person, at a `stop`, or when vision takes
over. A cue is *spent* once the eyes have had her, so losing her again lands in
`LISTENING` rather than sending the robot off along a bearing that was true a
minute ago.

**THE CUE IS REMEMBERED AS A WORLD HEADING, NOT A BODY BEARING.** The ears
measure a direction relative to the robot; the robot then turns, which makes
that number stale immediately. Steering by the stored bearing is a feedback loop
dressed as a controller: measured on `terrain_free_5`, the robot spiralled for
eleven seconds, fell over, and got no closer than 2.7 m to a person 6 m away.
The bearing is converted once, at the moment it is measured, using the **torso's**
yaw (the array's own frame — 21° from the base's at the spawn of
`terrain_free_0`, and converting with the base's yaw put every cue about 20°
wide), and the heading error is re-derived from the **base's** yaw every tick.

### Measured (`python -m app.harness.test_hearing`)

All of it on the 365-clip corpus (360 `say` + the five demo recordings), the
whole 4 x 3 grid of wind 0/6/12/20 m/s against 2/5/10 m.

**The threshold is CHOSEN, not guessed.** The rule is declared before the table
is read — maximise `detection − 3 × false-stop`, the weight being the brief
("`top`, `shop`, `drop` … must NOT trigger", so a false stop costs three
misses) — and the sweep picks **0.90**, which is what
`hearing.STOP_CONFIDENCE_THRESHOLD` is set to. The test prints AGREES/DISAGREES
against the constant every run.

| `stop` DETECTED | 2 m | 5 m | 10 m |
|---|---|---|---|
| **0 m/s** | 98.9% | 94.5% | 85.7% |
| **6 m/s** | 98.9% | 93.4% | 56.0% |
| **12 m/s** | 81.3% | 74.7% | 3.3% |
| **20 m/s** | 60.4% | 15.4% | 0.0% |

| FALSE STOP (near-misses + ordinary calls) | 2 m | 5 m | 10 m |
|---|---|---|---|
| **0 m/s** | 17.9% | 14.2% | 12.8% |
| **6 m/s** | 16.1% | 12.0% | 5.8% |
| **12 m/s** | 12.4% | 7.3% | 0.0% |
| **20 m/s** | 6.9% | 0.4% | 0.0% |

**Read that second table with its parts separated, because pooling hides the
finding.** The false-stop rate on ORDINARY CALLS — "come here", "over here",
"help", and all five demo clips — is **0.0%** at 0.90 and never above 0.4%
anywhere in the sweep. Every point of it is the near-miss set, which fires
26.7% of the time at threshold **1.00**: "top", "shop" and "drop" differ from
the keyword by one phoneme and a 40 MB model asked a two-way question gets it
wrong about a quarter of the time whatever the confidence. **No threshold fixes
that**, which is exactly why the sweep is printed rather than a number asserted.
The top of the ladder is also nearly flat — 0.85, 0.90 and 0.93 sit within
0.2 points of each other — so do not read the chosen value as precise; read it
as "somewhere above 0.85, and the exact number does not matter".

A rule that DID look decisive and was measured and dropped: requiring the
utterance to BEGIN with the keyword (`"top"` decodes as `[[unk] 0.62, stop
0.93]`, so it would be rejected). Over the whole corpus and the whole grid it
moved the near-miss rate 27.9% → 27.7% and cost 1.4 points of detection. Most
near-misses come back as a BARE `stop`, so the rule was paying for a class of
failure that barely exists.

**The five demo clips**, each on the whole grid (`S` decoded as `stop`, `v`
heard as an ordinary voice, `·` not heard at all; cells are 2 m / 5 m / 10 m):

| clip | label | 0 m/s | 6 m/s | 12 m/s | 20 m/s |
|---|---|---|---|---|---|
| `stop_1.mp3` | stop | SSS | SSS | SSS | SSv |
| `pemba_1..4.mp3` | other | vvv | vvv | vvv | vvv |

**Speaking up helps, and that is the test that AGC is really off** — same clips,
5 m, 12 m/s of wind:

| source level | SNR vs the wind | voice heard | `stop` decoded | bearing mean error |
|---|---|---|---|---|
| −20 dB | −15.4 dB | 87.0% | 0.0% | 77.1° |
| 0 dB | +4.6 dB | 84.8% | 71.4% | 36.9° |
| +10 dB | +14.6 dB | 91.3% | 100.0% | 26.9° |

**Voice activity** (did the utterance become a segment at all — the ceiling on
everything above): 100% everywhere at 0 and 6 m/s, 92–96% at 12 m/s, and
**100% at 20 m/s with a voiced-frame share of 1.00** — which is not a triumph,
it is `webrtcvad` saturated ON by the gale. In a 20 m/s wind this robot believes
a voice is present at all times; what stops it walking off toward the noise is
`EAR_BEARING_MINIMUM_CONFIDENCE`, and table 3b is why that gate exists.

**Bearing**, mean / 95th percentile in degrees, over the clips the VAD heard:

| wind | 2 m | 5 m | 10 m | peak sharpness → confidence |
|---|---|---|---|---|
| **0 m/s** | 0.0–0.5 / 0.0–0.8 | 0.0–0.4 / 0.0–0.7 | 0.0–0.5 / 0.0–0.6 | 0.59–0.60 |
| **6 m/s** | 0.0–0.5 / 0.0–0.8 | 0.0–0.4 / 0.0–0.7 | 0.0–0.5 / 0.0–0.9 | 0.52–0.59 |
| **12 m/s** | 26–43 / 135–178 | 24–41 / 124–167 | 29–51 / 142–180 | 0.14–0.34 |
| **20 m/s** | 0.2–7 / 1.1–34 | 14–32 / 90–146 | 41–56 / 114–167 | 0.02–0.39 |

**Sub-degree up to 6 m/s at every range, and then it falls off a cliff** — and
the cliff is not monotone, which is the interesting part. At 12 m/s the VAD is
marginal, so the segments that do open are often mostly wind and the correlator
is cross-correlating a gale with itself. At 20 m/s the VAD saturates ON, opens a
segment around everything, and at 2 m the voice still dominates the loudest
window (0.2–7°) while at 10 m it does not (41–56°). The **confidence column is
what makes this safe**: it tracks the failure exactly, 0.59 → 0.02, and the
behaviour refuses any cue below 0.25.

**End to end**, `flat_free` (perfectly flat, no rope), the hiker 6 m away at
45° off the nose, one shout, five speakers, then a visibility ladder and a close
arm:

| visibility | start | ear bearing vs truth, in the array's own frame | eyes acquire her | arrived (≤ 1.3 m) |
|---|---|---|---|---|
| 3 m | 6.0 m | +3.3° … +6.8° (four of five speakers) | never (fog) | 1 of 5, at 43.0 s |
| 3 m | 6.0 m | — (fifth speaker: cue below the confidence gate) | never | no — correctly went `LISTENING` instead of walking |
| 10 m | 6.0 m | +5.1° | 37.4 s | **43.3 s** |
| 100 m | 6.0 m | +5.1° | 1.3 s | **44.9 s** |
| 3 m | 2.5 m | +6.1° | never (fog) | **34.4 s** |
| 100 m | 2.5 m | +6.1° | 1.3 s | **18.3 s** |

**5 of 9 arrived**, mean 36.8 s, min 18.3 s, max 44.9 s — the robot hears her,
turns, walks over, and the follower stops it in the 1 m band. One fall in nine
(at 68 s, on a run that was still walking). The 3 m arm is the one that matters:
**the eyes never see her at all and the ears are the only sensor in play.**

The fifth speaker (Moira) is worth reading as a pass, not a miss: her cue's
peak sharpness came in under `EAR_BEARING_MINIMUM_CONFIDENCE`, so the behaviour
took no cue and stood in `LISTENING` — which is precisely what it should do
with a direction it does not believe.

| what | result |
|---|---|
| `stop` mid-walk → command zero | **0.90 s, 45 ticks** — the clip's own length plus 0.20 s of silence before the segment closes plus one detector tick. The word is decided on the WHOLE utterance, so the robot cannot stop before the speaker has finished saying it |
| physics parity, hearing off vs on with an utterance decoded | **0.000e+00** on `qpos`, `qvel`, `ctrl`, `sensordata`, `qfrc_constraint`, `cfrc_ext` |

**Per-tick cost**, measured in the live loop: ear synthesis **0.12–0.31 ms every
tick**, the detectors **0.06–0.19 ms** on a detector tick (VAD 0.04 ms), and per
utterance a one-off **3.5–9.7 ms of vosk + 1.6–2.0 ms of GCC-PHAT**. About 1% of
a 20 ms tick, against an eye render that already costs 13 ms per stereo pair at
10 Hz.

### The ground the walker can and cannot hold a heading on

`flat_free` is flown because it is the one world where the plant is not the
story. Everywhere else it is. A pure yaw command for 4 s from the spawn, with
the drift at zero command as the noise floor:

| world | +1.0 rad/s | 0.0 | −1.0 rad/s | separation | drift at 0 |
|---|---|---|---|---|---|
| `flat_free` ← flown | +13° | +174° | −39° | +52° | 1.65 m |
| `terrain_free_0` | +2° | −43° | −36° | +38° | 0.52 m |
| `flat_0` (roped) | −13° | −15° | −50° | +37° | 1.04 m |
| `terrain_free_5` | −76° | +45° | −35° | −41° | 2.39 m |
| `sandbox_free` | +171° | +28° | −161° | +332° | 2.19 m |

**On `terrain_free_0` — the same flat ground with 11 cm of Lhotse roughness on
it — the identical run arrives 0 times in 9 and gets no closer than 5.5 m of 6.**
Same ears, same bearings (+0.3° to +3.0° there), same behaviour; the difference
is entirely the walker. Three further measurements, each now a constant in
`hearing.py`:

* **It cannot turn on the spot.** Yaw comes out of the stepping gait, so
  `[0, 0, +0.5]` is a robot standing still. With a pivot-first rule the heading
  error sat at +80° for **85 seconds** with the waist pinned at its limit. The
  ear layer therefore always walks while it turns.
* **Walking and turning together is what tips it over.** `sandbox_free` has the
  largest yaw response in the catalogue and falls inside four seconds under a
  walk-and-turn command.
* **The ear-driven waist aim is capped at 20°, not the sweep's 60°.**
  `WAIST_LIMIT_RADIANS` was measured on a robot standing still; walking and
  turning adds the loads. 60° → fell at 4.5 s, 25° → 89.8 s, 15° → survived.

**ASK to Mrinal:** randomise `ang_vel_yaw` in the training commands. On smooth
ground this layer works as designed; on anything with texture the walker cannot
hold the heading the ears hand it.

### The models this needs installed

    pip install vosk soundfile webrtcvad-wheels
    # then the 40 MB acoustic model, ONCE (python's urllib has no CA bundle here,
    # so vosk's own auto-download fails; curl has one):
    curl -L -o /tmp/m.zip \
      https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
    unzip /tmp/m.zip -d ~/.cache/vosk/

Installed into `.venv_everest` on 2026-08-30: `vosk 0.3.44`, `soundfile 0.14.0`,
`webrtcvad-wheels 2.0.14` (the maintained wheel build of `webrtcvad`, which has
no Python 3.12 wheels of its own). Both detectors degrade to "unavailable" with a
printed line rather than crashing, so a machine without them still runs the rest
of the harness — it simply never hears anything.

### Honesty note

The **microphone signal is real**: it is a human being's actual voice, captured
by the Mac and not processed by anything before the runtime sees it. The **ear
signals are synthesised** from truth positions, the same way the eyes are
rendered from truth geometry — the simulator supplies *where the mouth is* and
*where the head is*, and the model turns that into what four capsules would have
received. Everything after `EarArray.feed` reads only those four signals. What
this model does **not** have, and would flatter the bearing if you forgot:
no reverberation, no diffraction or shadowing around the torso, no elevation
estimate, and wind noise that is independent per capsule with no coherent
low-frequency component. The bearing numbers below are therefore an
**optimistic** bound on a real array. `PARITY.md` carries the full ledger.

## Visibility (`app/harness/storm.py`)

Slide **VISIBILITY** from `100 m` down to `3 m` and the weather closes in.
**A white-out here is FOG**, not a snow shower: what it does is take DISTANCE
away, so the far slope dissolves into white first, then the middle distance, and
at the far end of the dial you cannot see the hiker three metres in front of
you. Nothing sits on the lens.

**It is its own dial and it owes the wind nothing** (user's ruling, 2026-08-30).
Until then this was a `storm` on/off switch whose thickness came out of the wind
speed — `100 m × exp(−wind / 6)` — which made two independent things one
control: you could not have a still white-out or a clear gale, and every wind
experiment silently changed what the robot could see. The knob is now a
**distance in metres**:

    {"type":"knob", "name":"visibility", "value": 15.0}      # metres

`100 m` is CLEAR and the eye images are handed back **untouched** — not fogged
with a faint 100 m ramp, and the sensor-noise generator is not even advanced, so
"clear" is a real control arm in every table below. `3 m` is the floor rather
than zero, because a visibility of nothing is a blank screen and not a
white-out. An older page's `storm` knob is stored and ignored rather than
rejected, so a stale client cannot crash the runtime.

**One curve, three places.** The slider's scale, the page's fog and flake
density, and the fog on the robot's eyes all read the same log share —
`ln(100 / v) / ln(100 / 3)`, 0 at clear and 1 at white-out, which puts the
slider's own centre at `sqrt(100 × 3)` = 17.3 m. It lives in `whiteoutShare`
(`app/web/three/world.js`) and in `storm.whiteout_share`; the two endpoints are
duplicated in both. If one moves, move the other, or the picture and the robot
stop being in the same weather.

**One law, one colour** (user's ruling, 2026-08-30). The two sides used to fog
by two different laws in two different colours — the eyes ramped LINEARLY from
`0.15 × v` toward a flat white, the page ran `FogExp2` toward a colour that
starts as blue-grey haze and only becomes snow-white further down the dial, and
the page painted a 2-D white veil on top that the eye image never received. Same
knob, two weathers, and the eye PiP visibly disagreed with the viewport. The
eyes now adopt the page's law, `f = 1 − exp(−(d × 1.73 / v)²)`, and the page's
colour ramp; the veil is gone. The five constants live in
`app/web/three/world.js` and are mirrored in `storm.py`, and `test_storm`
section K reads them back out of the JS at run time and prints both laws at
`d = 0.25v / 0.5v / v / 2v` so a drift is a failing row. Measured on a 1920×1080
headless shot with the viewport put in first person on the ROBOT (same mount,
same 58° fovy as the stereo eye): the sky region — where the fog colour is the
whole answer — reads within **2–4 of 255** between the viewport and the PiP at
30 m, 10 m and 3 m.

**On the robot's eyes** the fog is composited per pixel from the eye renderer's
own DEPTH buffer, `out = colour*(1 − f) + fog_colour*f`, plus a couple of grey levels
of Gaussian sensor noise drawn INDEPENDENTLY per eye — the one thing fog does
not reproduce, and what leaves the block matcher nothing to match. The grain now
scales with the white-out share too (1.0 grey level at 100 m, 7.0 at 3 m); it
used to scale with the wind, which was the last place the two dials were tied
together. All of it lands BEFORE the matcher and the detector, which is the only
placement that makes the degradation honest, and the eye-view PiP shows the
degraded image because it IS the image the robot used.

It legitimately breaks the follower, which is the point (`flat_0`, the guide's
orange backpack as the target, `test_storm` sections E and F):

| visibility | detected at 2 m | detected at 5 m | max detection range |
|---|---|---|---|
| 100 m (clear) | 100% | 100% | 16 m |
| 30 m | 100% | 100% | 8 m |
| 10 m | 100% | **0%** | **3 m** |
| 3 m | **0%** | **0%** | **never seen** |

(The max ranges were 16 / 10 / 4 / never under the retired linear ramp. Adopting
the page's exp² law made the middle of the dial slightly thicker, which is where
the two ranges moved.)

and with the human parked at 9 m the follower goes to LOST on its own — 100% of
detections at 100 m and 30 m, **0%** at 10 m and 3 m, and the mode follows.

The fog eats the far half of the frame first, which is what says it is fog and
not a wash (section E0, sensor noise off, the guide at 5 m):

| visibility | mean pixel change NEAR (≤ 6 m) | mean pixel change FAR (> 6 m) |
|---|---|---|
| 100 m (clear) | 0.0 | 0.0 |
| 30 m | 5.3 | 49.0 |
| 10 m | 42.0 | 65.0 |
| 3 m | 135.4 | 68.8 |


    python -m app.harness.test_storm      # the tables above + the eye contact sheet
    python -m app.harness.runtime --world flat_0 --duration 10 --hold-w --guide \
        --visibility 3

**None of it is physics.** The fog is arithmetic on two rendered arrays and
nothing writes to the model or to `MjData`. `test_storm` section H is a
same-seed diff, 100 m clear against a 3 m white-out with the guide on:
**0.000e+00** across `qpos`, `qvel`, `ctrl`, `sensordata`, `qfrc_constraint`
and `cfrc_ext`. Section J is the row that says the split from the wind is real:
the same measurement at 10 m visibility with the wind dial at 0 m/s and at
12 m/s — two settings that used to mean 100 m and 13.5 m of visibility.

`render3d_shots/storm_eyes.png` is the left eye at each visibility;
`render3d_shots/visibility_{100,15,3}.png` is the 3D page at 8 m/s of wind with
the guide on. (`storm_before/` and `storm_after/` are the wind-indexed shots
from the 2026-08-29 fog ruling, kept as history — their file names are wind
speeds, from when visibility was derived from wind.)

`render3d_shots/fogmatch_{100,30,10,3}.png` is the one-law proof: the viewport
in first person on the robot beside the EYE0 PiP of the same camera, one shot
per visibility.

The camera-subject shots are `render3d_shots/camera_{guide,robot}_{3p,1p}.png`,
one per state of the guide switch and the **V** toggle.

## Snow, and footprints in it (`app/harness/snow.py`)

The terrain wears a procedurally generated snow texture -- metre-scale drifts,
centimetre-scale wind grain, the odd sparkling crystal -- instead of a flat grey
sheet. Every time a foot lands, an elliptical print (26 x 12 cm, turned to the
foot's own yaw, soft-edged) is painted into that texture's pixels and pushed to
the GPU with `mjr_uploadTexture`, so the robot leaves a trail behind it that
fades away over about half a minute.

**None of it touches physics.** The heightfield is never edited; what changes is
a texture, a material and the terrain geom's `matid`, none of which the solver
reads. Proven, not asserted: two 6 s same-seed runs with and without
`--no-snow` come back **bit-identical across all 39 recorded arrays** (PARITY.md).

The same landing detection does two jobs, which is why they cannot disagree: it
increments `step_count`, and it puts a `foot_steps` event on the websocket --
`[{"foot": "left"|"right", "impact_speed_mps": f}]`, empty on almost every tick
-- so the page can play one snow crunch per step at a volume set by how hard the
foot came down, and drop its footprint decal in the same instant. A landing is a
foot gaining terrain contact after at least two ticks in the air; without that
debounce a scuffing foot machine-guns. `hud.json` records `step_count`, so a
recorded episode counts the same steps the live session heard.

Cost: reading the solver's contact list, microseconds a tick. **Until
2026-08-30 this also STAMPED the print into the ground texture and pushed the
whole 4.6 MB texture back to every GL context** (0.04 ms to paint, 0.002 ms to
fade, 0.75 ms per upload at 6 Hz to the main and eye contexts) -- because the
server-rendered JPEG was the only picture. The 3-D page draws its own decals, so
all of that is deleted. `--no-snow` still turns the texture off; the step events
stay, because the sound does not depend on the picture.

## The ClimbScene worlds

`app/harness/climb_worlds.py` calls `climb_scene.build_scene(...)` and drives
what comes back. The physics step is his (`ClimbScene.step(wind)` = one
`mj_step` + the carrier projection + the arc-length ratchet); the 103-dim
observation and the walking policy are his (`walk_policy.WalkController`).
Ours is the command, the wind vector, the friction knob, the telemetry and the
bookkeeping.

Two gates run before a world is trusted, both printed:

    python -m app.harness.climb_worlds --world lhotse_B

*joint parity* — the scene's 29 actuated joints against Playground's, in order
(a mismatch means the policy's actions drive the wrong joints) — and
*observation parity* — our builder against his `observe()` at the same state.
Measured **0.000e+00**: bit-identical.

`app/harness/provision_assets.py` puts the gitignored trees the scene needs on
disk (`.reference/`, the jacketed robot's stock link STLs, `~/mujoco_menagerie`)
by copying from the installed `mujoco_playground`, because all of the repo's own
fetch paths fail here. It runs automatically; PARITY.md says what is broken.

## Chloe's ascender policy (`chloe_v1_20`, `chloe_v1_25`)

Everything above this line is the WALKING policy on the Lhotse terrain. These
two worlds are neither: they run **Chloe's mjlab-trained rope-ascender
network** on **her own plant**, and they are the only worlds in the harness
where the robot is not steered by a person.

    pip install onnxruntime                      # the one extra these need
    python -m app.harness.export_scene --worlds chloe_v1_20 chloe_v1_25
    python -m app.harness.runtime --live --world chloe_v1_20

then open **http://localhost:8766/** and hold **W**.

### How to run it, and what each entry point is for

    python -m app.harness.runtime --live --world chloe_v1_20        # the page
    python -m app.harness.runtime --world chloe_v1_20 --duration 15 --hold-w
    python -m app.harness.chloe_worlds --slope 20 --seconds 15   # no server at all
    python -m app.harness.chloe_worlds --equivalence --slope 20  # both frames
    python -m app.harness.chloe_worlds --matrix                  # slope x wind, and stop/go

`--chloe-policy PATH` swaps the ONNX (the default is
`rl/policies/g1_ascender_slope20_v3_2026-08-30_04-35-59.onnx`);
`--chloe-hold-blend SECONDS` is the one knob on the stop behaviour, below.
`--policy` is the WALKING policy's npz and does nothing here.

### The controls, and what they honestly do

| key | on every other world | on `chloe_v1_20` / `chloe_v1_25` |
|---|---|---|
| **W** | commands `lin_vel_x` 0.5 m/s | **gates the network**: held = it runs, released = held pose |
| **A** / **D** | turn in place | **nothing** (the keycaps are hidden, the legend says "she steers herself") |
| mouse look | steers by camera heading | **nothing** |
| **R** | reset | reset |
| the guide switch | the follower drives the robot | the follower still SEES; it may drive W's gate; it may not steer |

Her network **has no command input**. There is no `lin_vel_x`, no
`ang_vel_yaw`, no gait clock and no stop: 96 numbers in, 29 joint targets out,
50 times a second, and it climbs. So the harness does not pretend to steer it.
The one control that exists on a rope is go / don't go, and that is what W is
wired to (`chloe_policy.AscenderController.go`, also drivable by any other
lane — the hearing lane included — because it is a plain boolean).

**STOP IS A HELD POSE, NOT A PAUSE.** With the gate released the last PD
targets the policy wrote are frozen and the physics keeps running: the legs
hold where they are and the ascender's cam holds the body on the line. What
that measures out at is in PARITY.md; the short version is that **she stops,
and she does not start again**. See "the one-way stop" there before you build
anything on top of it.

That is also why you want the **guide switch OFF** on these two worlds if what
you want is a climb: the follower owns the command, so it owns the gate, and v3
walks uphill *backwards* — the human is behind the cameras within about three
seconds, the follower commands zero, and the gate shuts for good. Everything
still runs and every readout stays live; the climbing stops. Measured in
PARITY.md.

### The plant is hers, and every part of it is load-bearing

* actuators: mjlab's `G1_ARTICULATION` — per-motor-group `kp`/`kd`/`armature`
  (`chloe_policy.G1_ARTICULATION`), NOT the harness's walking gains
* action decode: `ctrl = default + per_joint_scale × action`, her per-joint
  scale table, NOT a flat 0.5
* feet: the XML's four spheres per foot, unchanged — no box swap, no
  `policy_compat`
* rope: `assets/robots/mujoco/rope_rail.py` — ONE straight non-colliding line
  0.60 m above the ground, the ascender WELDED to a 1-DoF `rope_carriage`
  slide, ratchet = the slide's moving lower limit, raised before every
  `mj_step`
* ground: one plane. No heightfield, no roughness.
* rates: physics 5 ms, decimation 4, control 50 Hz

Measured, both directions: give her policy the walking gains (kp 75 / kd 2) and
it falls in under two seconds; give it a flat 0.5 action scale and it falls in
under two seconds. The usable slope band is 10–30°, which is why there are two
worlds and not six.

### The frame — why the slope is a slope here and a gravity vector in `rl`

She trained on a FLAT plane with GRAVITY TILTED by the slope. That reproduces
exactly (`--frame tilted_gravity`) and it looks wrong: level ground, a
horizontal rope, a robot leaning 20° backwards, and "height gained" reading
0.00 m forever.

The shipped worlds rotate the WHOLE WORLD by −slope about +y and put gravity
back to (0, 0, −9.81) (`--frame tilted_plane`, the default). It is a change of
coordinates and nothing else: her observation is invariant under it term by
term (pelvis angular velocity is already body-frame; projected gravity, the
joint block and the carriage-minus-pelvis vector all cancel the rotation), and
the weld and the slide survive it because the carriage body rotates with
everything else, so their numbers never change.

That claim is measured, not asserted:

    python -m app.harness.chloe_worlds --equivalence --slope 20
      tilted_gravity   uphill  +5.301 m   rope  +5.504 m   STANDING
      tilted_plane     uphill  +5.300 m   rope  +5.502 m   STANDING
      |difference| 0.0018 m (0.03 %)

### Honesty notes

* **The policy is hers and the plant is hers.** `rl/` is the source of
  truth for both; `app/harness/chloe_policy.py` is a re-implementation of the
  sim2sim loop against a plain `MjData`, and `chloe_worlds.py` rebuilds the
  plant around `rope_rail.py`. Nothing here re-derives a number that lives
  there — the default pose in particular is whatever `add_rope_rail` solves,
  never a transcribed constant.
* **No command input, so no steering, so no honest way to drive her.** Stated
  in the UI as well as here.
* **The rope model is `rope_rail.py`'s**, which is a straight line and a
  prismatic joint — not the draped polyline and mocap bead the Lhotse worlds
  ride. The two are different mechanisms and their telemetry means different
  things; PARITY.md has the ledger.
* **The rope rail moved under the policy on 2026-08-30.** v3 was trained with
  the channel at the mesh hull's centre; the shared `rope_rail.py` now puts it
  at x = +15 mm, pitch +5°, set from renders. The same policy, same seed,
  15 s at 20°: **+5.61 m before the change, +5.30 m after** (rope +5.82 →
  +5.50). It still climbs and it never falls, but it has lost about 5 % — the
  network and the rail have drifted apart, and the fix is a retrain (v7 is
  training against the new rail), not a harness change.
* **Wind, visibility, snow and the storm are demo-only here too.** The state
  message says `wind_in_training: false`. `terrain_in_training` is `true` on
  these two worlds and only these two, because the slope IS what she trained
  on.

## The Pemba G1 (the real demo robot, legacy env only)

**Superseded by the ClimbScene worlds**, which build the jacketed robot by
default through the team's own `robot.resolve`/`adapt`. Kept because it is the
only way to put that robot in the *legacy* env.

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

    python -m app.harness.runtime --live --world lhotse_B --bms

It builds `Environment(altitude_m=<world's altitude_meters, default 6907>,
wind_kmh=3.6 × wind dial)` and keeps `wind_kmh` live as the dial moves. Your
`step(data)` signature is adapted at the call site, so nothing in `app/bms`
needs to change. The whole attach is best-effort: if the import fails, the
harness prints why and runs on without a battery readout.

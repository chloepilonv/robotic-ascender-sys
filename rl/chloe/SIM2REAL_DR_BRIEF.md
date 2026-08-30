# Domain-randomisation brief for the ascender climb policy

**For:** the agent extending domain randomisation in `rl/chloe/task/env_cfg.py`, starting from **v3**
(`g1_ascender_slope20_v3_2026-08-30_04-35-59`).

**Source of these findings:** a hardware session on the gantry G1 on 2026-08-30, deploying a
*different* policy — the mels G1 joystick walker (JAX/playground, 103-dim obs). Read that caveat
carefully: the failures below are properties of **the robot and the sim-to-real gap**, not of the
mels network, so they transfer. But nothing here was measured on the ascender policy itself.

Everything quoted is measured on hardware unless flagged as inference.

---

## What already works — do not "fix" these

v3 is ahead of the mels policy on three counts that mattered badly in deployment. Leave them alone:

- **No base linear velocity in the actor obs.** The mels policy has it, the G1 cannot measure it, and
  feeding zeros made the closed loop diverge (jitter p99 0.160 open-loop → **0.658** closed-loop; the
  robot nearly fell). v3 already keeps `base_lin_vel` critic-only. This is the single biggest thing
  v3 gets right.
- **`action_rate_l2` penalty.** The mels policy had `action_rate = 0.0` — nothing discouraged jitter,
  and it showed. v3 penalises it (−0.1, raised to −0.2 in v7). Keep it; consider −0.2.
- **Action delay 0–2 steps** (`env_cfg.py:80,96`). Measured real loop latency was p50 3.8 ms /
  p99 6.5 ms on top of the 20 ms period, so 0–2 steps (0–40 ms) already covers it. Do not extend.

---

## Change 1 — `ascender_pos_b` has NO observation noise. Highest priority.

**Where:** `env_cfg.py`, `actor_terms["ascender_pos_b"]` — the only actor term with no `noise=`.

**Why it matters.** On hardware this comes from **wrist forward kinematics**, so it inherits the
position error of every joint in the arm chain. Those errors are not small:

| joint | commanded kp | measured steady-state error |
|---|---|---|
| left_wrist_roll | 2 | **0.445 rad (25°)** |
| right_wrist_pitch | 2 | 0.272 rad (16°) |
| left_ankle_pitch | 20 | 0.239 rad |
| right_knee | 75 | 0.166 rad |

A 0.45 rad error at the wrist puts the FK-derived ascender position off by **centimetres**. In sim
that term is exact to machine precision. The policy is therefore free to depend on a signal that is
substantially wrong on the real robot — the same failure that broke the mels deployment, in a term
nobody has noised.

**Do:** add noise, and make it **persistent per episode**, not white per step (see Change 2 for why):

```python
"ascender_pos_b": ObservationTermCfg(
    func=mdp.ascender_pos_b, params={"asset_cfg": mdp.CARRIER},
    noise=Unoise(n_min=-0.03, n_max=0.03),      # 3 cm, start here
),
```

3 cm is a defensible first estimate from a 0.45 rad wrist error over a ~0.24 m forearm
(0.45 × 0.24 ≈ 0.11 m worst case; 3 cm is deliberately conservative for a first run). **This number
is an inference, not a measurement** — the honest way to pin it is to compute FK ascender position
from real logged joint angles versus commanded angles and measure the spread directly.

---

## Change 2 — all observation noise is white. Real sensor error is biased.

**Where:** every `Unoise(...)` in `actor_terms`. `mjlab`'s `UniformNoiseCfg` resamples each step.

**Why it matters.** We measured this directly on the mels policy. Same perturbation magnitude
(±0.1 m/s), averaged over a full gait cycle:

| error character | residual after averaging over the gait |
|---|---|
| white, resampled each step | 0.0049 |
| **constant bias, held for the episode** | **0.0209** |

**A persistent bias is 4.3× more damaging than white noise of identical magnitude**, because the gait
averages white noise away and cannot average away a bias. Real error sources — encoder offsets,
IMU bias, a joint sitting in its friction dead-zone, FK error from a drooping wrist — are all
persistent within an episode, not white.

**Do:** add a per-episode constant offset on top of the existing white noise, sampled at reset and
held. Add an `EventTermCfg(mode="reset", ...)` that stores per-env offsets and have the obs terms
add them. Suggested ranges, all conservative:

| term | current white noise | add per-episode bias |
|---|---|---|
| `joint_pos` | ±0.01 | **±0.05 rad** (see Change 3) |
| `projected_gravity` | ±0.05 | ±0.02 |
| `base_ang_vel` | ±0.2 | ±0.05 rad/s (gyro bias) |
| `ascender_pos_b` | none | ±0.03 m |

Leave `joint_vel` white — real joint velocities were *cleaner* than sim noise (measured ~0.03 rad/s
at rest against ±1.5 rad/s of injected noise), so it is already over-noised, not under.

---

## Change 3 — joint friction is ~7× underestimated, and is not randomised at all

**Where:** no `frictionloss` randomisation exists in `events`. The MJCF sets
`frictionloss = 0.1 N·m` on every joint.

**Evidence.** On the real robot, `left_wrist_roll` held **0.69 N·m** at rest. Gravity about that
axis — computed in MuJoCo at the same pose — is **0.003 N·m**, because wrist roll turns about the
forearm's own axis and the hand's mass sits on it. So ~0.69 N·m is resisting something that is not
gravity, and friction is the leading candidate: **7× the modelled 0.1 N·m**.

With `kp = 2`, a 0.69 N·m stiction band is a **0.345 rad dead zone**, against the 0.445 rad we
measured. Same order — the numbers are consistent.

**Do:** randomise `frictionloss` per joint at startup, wide:

```python
"joint_friction": EventTermCfg(
    mode="startup",
    func=dr.joint_friction,               # confirm the exact mjlab term name
    params={"asset_cfg": G1_JOINTS, "operation": "abs", "ranges": (0.05, 1.0)},
),
```

**Caveat, stated plainly:** the friction attribution is a *hypothesis*. What is certain is that
0.69 N·m of torque is being spent on something the sim does not model. Friction fits the magnitude
and the dead-zone arithmetic; an external contact or a `tau_est` bias would also fit. Confirming it
means a breakaway-torque sweep — ramp a commanded position slowly and record the torque at which the
joint starts to move. Worth doing, but randomising the range is robust either way.

---

## Change 4 — PD gain randomisation is multiplicative, which cannot save a low gain

**Where:** `motor_strength`, `kp_range=(0.8, 1.2)`.

±20% around `kp = 2` gives 1.6–2.4. **Every value in that range is still too weak to break 0.69 N·m
of stiction.** The policy therefore never experiences a wrist that tracks properly, nor learns that
it might. Multiplicative randomisation preserves the pathology it needs to randomise away.

**Do one of:**

- **(a)** Widen for low-gain joints specifically — e.g. 0.5×–5× on wrists and ankle_roll, keeping
  0.8–1.2 elsewhere. Teaches the policy not to depend on precise wrist placement.
- **(b)** Raise the nominal wrist gains in the MJCF (2 → 20–40) and keep ±20%. Simpler, and matches
  what we did on hardware — raising wrists to `kp=40` moved the observation's max |z| from **5.06 to
  3.16** and cut open-loop action jitter p99 from **0.441 to 0.160**.

(b) is better supported by evidence; (a) is more robust if the real gains are ever changed. They
compose — doing both is reasonable.

---

## How to validate — the free OOD detector

The exported policy ships an observation normaliser (`obs_normalization=True` in `RslRlModelCfg`;
mjlab stores the running mean/std). **Those statistics are a record of what the policy saw in
training**, so they give you an out-of-distribution check for free, computable offline with no robot:

```
z = (live_obs - obs_mean) / obs_std        # anything past ~3 sigma is extrapolation
```

This is how the wrist problem was found: `obs_std` for that joint was 0.088 rad (the wrist barely
moved in training), the real droop was 0.445 rad, hence **5.1σ**.

**Acceptance criterion for this work:** replay real robot telemetry (or telemetry-shaped synthetic
data with the biases above) through the trained policy's normaliser and require **no actor
observation dimension beyond ~3σ**. If a dimension still exceeds it, that dimension's randomisation
is too narrow — which is exactly the diagnosis you want, per-dimension.

---

## Method note, and it is the real lesson

Report **per joint and per observation dimension, never a max across all 29**. During the hardware
session `|q - target|max` — a single scalar — produced two confident and completely wrong diagnoses
("the pose ramp has stalled", "there is a command-rate watchdog") before per-joint reporting made
the answer obvious in one run. A max discards precisely the information that distinguishes
hypotheses. Any diagnostic added here should preserve the per-dimension breakdown.

---

## Retraining is required, and the rope model is the reason

`rl/chloe/README.md` states the rule: *a policy is only valid with the rope model it was trained on.*
v1 climbed in its own world and falls on the fixed rope. `ascender_pos_b` is in the observation, so
any change to `rope_rail.py` changes what those numbers mean and silently invalidates the network —
the same class of failure as the wrist droop, at a different scale.

If this work changes only randomisation and not geometry, v3's rope model still applies. **Confirm
that before starting**, and check `assets/robots/mujoco/ROPE_ASCENDER_ALIGNMENT.md` (on
`origin/feat/rl-ascender`, not on `main`) for the current contract.

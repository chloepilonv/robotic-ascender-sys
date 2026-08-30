"""Proof that the harness's plain-MuJoCo loop IS the team's MJX environment.

Two experiments, both printing their numbers (a loss curve, or a green test,
proves nothing on its own):

(a) OBSERVATION PARITY. For two states -- their deterministic `knees_bent`
    reset, and a random perturbation of it -- build the 103-d `state` vector
    twice: once with `app.harness.playground_policy.PlaygroundObservation` on a
    plain `mujoco.MjData`, once with the team's own
    `G1ClimbAscender._get_obs` on an `mjx.Data` holding the identical qpos /
    qvel / command / last_action / phase. Noise is forced OFF on both sides
    (`noise_config.level = 0.0`) or the comparison would be measuring two
    different RNGs. PASS = max absolute difference < 1e-4.

(b) ROLLOUT DIVERGENCE. From one shared reset state, roll 100 control steps
    (2.0 s) with the same mels policy and a fixed command (0.5, 0, 0):
    their MJX env via `env.step`, ours via
    `ctrl = default_pose + 0.5*action` then 10 x (mj_step + ratchet). MJX and
    the MuJoCo C engine are different numerical implementations of the same
    model, so they WILL separate; the question is whether they separate like
    two runs of the same system (centimetres, smoothly) or like two different
    systems. Reported: pelvis position error and ascender slide error per step,
    plus both trajectories' own travel so a reader can see they do the same
    thing.

Run:  ../.venv_everest/bin/python -m app.harness.test_parity
"""

import os
import sys

import numpy as np

_REPOSITORY_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, _REPOSITORY_ROOT)

import mujoco  # noqa: E402

from app.harness import team_env  # noqa: E402
from app.harness.playground_policy import (  # noqa: E402
    GaitPhase, MelsPolicy, PlaygroundObservation, default_policy_path,
)
from app.harness.ratchet import AscenderRatchet, step_with_ratchet  # noqa: E402

OBSERVATION_TOLERANCE = 1e-4
ROLLOUT_CONTROL_STEPS = 100
ROLLOUT_COMMAND = np.array([0.5, 0.0, 0.0])
SLOPE_DEGREES = 30.0
GAIT_FREQUENCY_HZ = 1.375  # midpoint of their U(1.25, 1.5); fixed for parity.

BLOCK_NAMES = [
    ("linear_velocity_pelvis", 0, 3), ("angular_velocity_pelvis", 3, 6),
    ("projected_gravity", 6, 9), ("command", 9, 12),
    ("joint_angle_minus_default", 12, 41), ("joint_velocity", 41, 70),
    ("last_action", 70, 99), ("gait_phase", 99, 103),
]


def main():
    import jax
    import jax.numpy as jp
    from mujoco import mjx
    from mujoco_playground._src import mjx_env

    print("=" * 72)
    print("PARITY TEST: app/harness plain-MuJoCo loop vs rl/environment MJX env")
    print("=" * 72)

    # Noise off on THEIR side too: we are comparing determinism, not RNGs.
    environment = team_env.load_team_environment(
        config_overrides={"noise_config.level": 0.0},
        slope_degrees=SLOPE_DEGREES,
    )
    model, meta = team_env.describe_team_environment(environment)
    print(f"[setup] their env loaded: nq {model.nq} nv {model.nv} nu {model.nu},"
          f" noise level {meta['noise_level']}, slope {meta['slope_degrees']} deg")

    observation_builder = PlaygroundObservation(model, meta, noise_level=0.0)
    policy = MelsPolicy(default_policy_path(_REPOSITORY_ROOT))
    print(policy.describe())

    slide_qpos_address = meta["slide_qpos_address"]
    substeps = meta["substeps_per_control_step"]
    default_pose = meta["default_pose_radians"]
    action_scale = meta["action_scale"]

    def their_mjx_data(qpos, qvel):
        data = mjx_env.make_data(
            model, qpos=jp.array(qpos), qvel=jp.array(qvel),
            ctrl=jp.array(qpos[7:slide_qpos_address]),
            impl=environment.mjx_model.impl.value,
            naconmax=environment._config.naconmax,
            njmax=environment._config.njmax,
        )
        return mjx.forward(environment.mjx_model, data)

    def their_observation(qpos, qvel, command, last_action, phase_radians):
        data = their_mjx_data(qpos, qvel)
        info = {
            "rng": jax.random.PRNGKey(0),
            "command": jp.array(command),
            "last_act": jp.array(last_action),
            "phase": jp.array(phase_radians),
            # Only the privileged half reads this; the `state` half never does.
            "feet_air_time": jp.zeros(2),
        }
        contact = jp.zeros(2, dtype=bool)
        return np.asarray(environment._get_obs(data, info, contact)["state"], np.float64)

    def our_observation(qpos, qvel, command, last_action, phase_radians):
        data = mujoco.MjData(model)
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        data.ctrl[:] = qpos[7:slide_qpos_address]
        mujoco.mj_forward(model, data)
        phase = GaitPhase(meta["control_dt_seconds"], GAIT_FREQUENCY_HZ)
        phase.phase_radians = np.asarray(phase_radians, dtype=np.float64)
        return observation_builder.build(data, command, last_action, phase)

    # ------------------------------------------------------------------ (a)
    print("\n" + "-" * 72)
    print("(a) OBSERVATION PARITY  (max |ours - theirs|, tolerance"
          f" {OBSERVATION_TOLERANCE})")
    print("-" * 72)

    random = np.random.default_rng(7)
    reset_qpos = np.asarray(meta["keyframe_qpos"], dtype=np.float64)
    reset_qvel = np.zeros(model.nv)

    perturbed_qpos = reset_qpos.copy()
    perturbed_qpos[7:slide_qpos_address] += random.uniform(-0.25, 0.25, 29)
    perturbed_qpos[0:3] += random.uniform(-0.1, 0.1, 3)
    quaternion = reset_qpos[3:7] + random.uniform(-0.08, 0.08, 4)
    perturbed_qpos[3:7] = quaternion / np.linalg.norm(quaternion)
    perturbed_qpos[slide_qpos_address] = 0.37
    perturbed_qvel = random.uniform(-0.6, 0.6, model.nv)

    cases = [
        ("reset knees_bent state", reset_qpos, reset_qvel,
         np.array([0.0, 0.0, 0.0]), np.zeros(29), np.array([0.0, np.pi])),
        ("random perturbed state", perturbed_qpos, perturbed_qvel,
         np.array([0.5, -0.2, 0.3]), random.uniform(-1.0, 1.0, 29),
         np.array([0.9, -2.1])),
    ]
    worst_difference = 0.0
    for label, qpos, qvel, command, last_action, phase_radians in cases:
        theirs = their_observation(qpos, qvel, command, last_action, phase_radians)
        ours = our_observation(qpos, qvel, command, last_action, phase_radians)
        difference = np.abs(ours - theirs)
        worst_difference = max(worst_difference, float(difference.max()))
        print(f"\n  {label}: overall max abs diff {difference.max():.3e}"
              f"  (their obs norm {np.linalg.norm(theirs):.4f})")
        for name, start, stop in BLOCK_NAMES:
            block = difference[start:stop]
            print(f"    {name:<28} width {stop - start:>3}"
                  f"  max {block.max():.3e}  mean {block.mean():.3e}")

    observation_passed = worst_difference < OBSERVATION_TOLERANCE
    print(f"\n  ==> worst max abs diff across cases: {worst_difference:.3e}"
          f"   {'PASS' if observation_passed else 'FAIL'}")

    # ------------------------------------------------------------------ (b)
    print("\n" + "-" * 72)
    print(f"(b) ROLLOUT DIVERGENCE  ({ROLLOUT_CONTROL_STEPS} control steps ="
          f" {ROLLOUT_CONTROL_STEPS * meta['control_dt_seconds']:.1f} s,"
          f" command {ROLLOUT_COMMAND.tolist()})")
    print("-" * 72)

    jit_reset = jax.jit(environment.reset)
    jit_step = jax.jit(environment.step)
    state = jit_reset(jax.random.PRNGKey(0))
    # One shared starting state: take THEIRS, copy it into ours. This sidesteps
    # the fact that their reset draws base velocity from a JAX RNG that numpy
    # cannot reproduce -- the point of (b) is the dynamics, not the RNG.
    start_qpos = np.asarray(state.data.qpos, dtype=np.float64)
    start_qvel = np.asarray(state.data.qvel, dtype=np.float64)
    # Pin the gait clock on both sides so the phase input is identical.
    phase_step = 2.0 * np.pi * meta["control_dt_seconds"] * GAIT_FREQUENCY_HZ
    state = state.replace(info={**state.info,
                                "phase": jp.array([0.0, np.pi]),
                                "phase_dt": jp.array([phase_step])})
    print(f"[setup] shared start: pelvis {start_qpos[0:3].round(5).tolist()},"
          f" |base qvel| {np.linalg.norm(start_qvel[0:6]):.4f} m/s,"
          f" slide {start_qpos[slide_qpos_address]:.4f} m")

    data = mujoco.MjData(model)
    data.qpos[:] = start_qpos
    data.qvel[:] = start_qvel
    data.ctrl[:] = start_qpos[7:slide_qpos_address]
    mujoco.mj_forward(model, data)
    ratchet = AscenderRatchet(meta["slide_qpos_address"], meta["slide_dof_address"])
    ratchet.reset(data)
    phase = GaitPhase(meta["control_dt_seconds"], GAIT_FREQUENCY_HZ)
    last_action = np.zeros(29)

    rows = []
    for step_index in range(ROLLOUT_CONTROL_STEPS):
        observation = observation_builder.build(data, ROLLOUT_COMMAND, last_action, phase)
        action = policy.act(observation)
        data.ctrl[:] = default_pose + action_scale * action
        step_with_ratchet(mujoco, model, data, ratchet, substeps)
        phase.advance()
        last_action = action

        state = state.replace(info={**state.info, "command": jp.array(ROLLOUT_COMMAND)})
        their_observation_vector = np.asarray(state.obs["state"], dtype=np.float64)
        their_action = policy.act(their_observation_vector)
        state = jit_step(state, jp.array(their_action))

        their_qpos = np.asarray(state.data.qpos, dtype=np.float64)
        rows.append({
            "step": step_index + 1,
            "time_seconds": (step_index + 1) * meta["control_dt_seconds"],
            "our_pelvis": data.qpos[0:3].copy(),
            "their_pelvis": their_qpos[0:3].copy(),
            "our_slide": float(data.qpos[slide_qpos_address]),
            "their_slide": float(their_qpos[slide_qpos_address]),
            "action_difference": float(np.abs(action - their_action).max()),
        })

    print(f"\n  {'step':>5} {'t(s)':>6} {'pelvis err(m)':>14} {'slide err(m)':>13}"
          f" {'our slide':>10} {'their slide':>12} {'our pelvis x':>13}"
          f" {'their pelvis x':>15}")
    for row in rows:
        if row["step"] % 10 and row["step"] not in (1, 5, ROLLOUT_CONTROL_STEPS):
            continue
        pelvis_error = float(np.linalg.norm(row["our_pelvis"] - row["their_pelvis"]))
        print(f"  {row['step']:>5} {row['time_seconds']:>6.2f}"
              f" {pelvis_error:>14.5f} {abs(row['our_slide'] - row['their_slide']):>13.5f}"
              f" {row['our_slide']:>10.4f} {row['their_slide']:>12.4f}"
              f" {row['our_pelvis'][0]:>13.4f} {row['their_pelvis'][0]:>15.4f}")

    pelvis_errors = np.array(
        [np.linalg.norm(r["our_pelvis"] - r["their_pelvis"]) for r in rows]
    )
    slide_errors = np.array([abs(r["our_slide"] - r["their_slide"]) for r in rows])
    print(f"\n  pelvis divergence: final {pelvis_errors[-1]:.5f} m,"
          f" max {pelvis_errors.max():.5f} m, mean {pelvis_errors.mean():.5f} m")
    print(f"  slide  divergence: final {slide_errors[-1]:.5f} m,"
          f" max {slide_errors.max():.5f} m")
    print(f"  travel over {ROLLOUT_CONTROL_STEPS} steps: ours"
          f" {rows[-1]['our_slide'] - start_qpos[slide_qpos_address]:+.4f} m,"
          f" theirs {rows[-1]['their_slide'] - start_qpos[slide_qpos_address]:+.4f} m")
    print(f"  pelvis displacement: ours"
          f" {np.linalg.norm(rows[-1]['our_pelvis'] - start_qpos[0:3]):.4f} m,"
          f" theirs {np.linalg.norm(rows[-1]['their_pelvis'] - start_qpos[0:3]):.4f} m")
    print(f"  max |our action - their action| over the roll:"
          f" {max(r['action_difference'] for r in rows):.5f}")

    print("\n" + "=" * 72)
    print(f"(a) observation parity: {'PASS' if observation_passed else 'FAIL'}"
          f" (worst {worst_difference:.3e} vs tol {OBSERVATION_TOLERANCE})")
    print("(b) rollout divergence: reported above -- MJX and MuJoCo-C are"
          " different numerics of\n    the same model, so read the CURVE, not a"
          " threshold.")
    print("=" * 72)
    return 0 if observation_passed else 1


if __name__ == "__main__":
    sys.exit(main())

"""Run the mels G1 joystick walking policy inside the merged climb scene.

Pure NumPy -- no jax, no brax, no mujoco_playground. The checkpoint
(`rl/policies/mels_g1_joystick.npz`) is an MLP 103->512->256->128->58 with
swish activations and an observation normaliser, so CPU inference is ~50 us
against a 20 ms control budget.

WHY THIS EXISTS
Without a controller the scene holds `d.ctrl` at the keyframe pose: 29 position
servos told to hold fixed angles, with no balance feedback at all. That topples
on flat ground and topples immediately on a 39 degree face. "The robot falls"
is the absence of a policy, not a fault in the model.

OBSERVATION FIDELITY
The 103-dim vector must match `mujoco_playground` g1/joystick.py `_get_obs()`
exactly or the policy sees an input it was never trained on:

    linvel(3) gyro(3) gravity(3) command(3)
    joint_pos - default(29) joint_vel(29) last_act(29) phase(4)

`last_act` and the gait clock are hidden training-loop state that nothing in the
simulation reports; they are reproduced here. Unlike the onboard deployment
path, `linvel` is read from the `local_linvel_pelvis` sensor rather than
estimated -- in simulation the ground truth is available, and zeroing it costs
about 0.13 rad of mean joint-target error while walking.

The merged scene's carrier is a mocap body with no degrees of freedom, so
`qpos[7:]` is still exactly the 29 robot joints and this observation needs no
special-casing for the rope.
"""
from __future__ import annotations

import os

import mujoco
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_POLICY = os.path.join(REPO_ROOT, "rl", "policies", "mels_g1_joystick.npz")

N_JOINTS = 29
OBS_DIM = 103
ACTION_SCALE = 0.5        # motor_target = default_pose + ACTION_SCALE * action
CTRL_DT = 0.02            # 50 Hz policy
GAIT_FREQ_HZ = 1.375      # trained over U(1.25, 1.5); hold mid-range
PHASE_INIT = np.array([0.0, np.pi], dtype=np.float64)
# Command ranges the policy was trained over; clamp before feeding.
CMD_LIMITS = np.array([[-1.0, 1.0], [-0.5, 0.5], [-1.0, 1.0]])


def _swish(x: np.ndarray) -> np.ndarray:
    """x * sigmoid(x), the brax PPO default activation.

    Computed as exp(-|x|) so the exponential never sees a large positive
    argument. The naive x / (1 + exp(-x)) overflows for strongly negative
    inputs; the result is still correct once the inf propagates, but it warns
    on every step and the warning would mask a real numerical problem later.
    """
    e = np.exp(-np.abs(x))
    sig = np.where(x >= 0, 1.0 / (1.0 + e), e / (1.0 + e))
    return x * sig


class WalkPolicy:
    """The mels joystick checkpoint, as a plain NumPy forward pass."""

    def __init__(self, npz_path: str = DEFAULT_POLICY):
        if not os.path.exists(npz_path):
            raise FileNotFoundError(npz_path)
        z = np.load(npz_path)
        self.w = [z[f"hidden_{i}_kernel"].astype(np.float64) for i in range(4)]
        self.b = [z[f"hidden_{i}_bias"].astype(np.float64) for i in range(4)]
        self.mu = z["obs_mean"].astype(np.float64)
        self.sd = z["obs_std"].astype(np.float64)
        self.obs_dim = int(self.w[0].shape[0])
        if self.obs_dim != OBS_DIM:
            raise ValueError(f"expected a {OBS_DIM}-dim policy, got {self.obs_dim}")

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Canonical observation -> raw action (29,). Deterministic.

        The head emits 58 = 2*29 values, mean followed by log-std; the log-std
        half is dropped rather than sampled.
        """
        h = (np.asarray(obs, dtype=np.float64) - self.mu) / self.sd
        for i in range(3):
            h = _swish(h @ self.w[i] + self.b[i])
        return (h @ self.w[3] + self.b[3])[:N_JOINTS]


class WalkController:
    """Closed-loop driver: builds the observation, ticks the gait clock, acts.

    Call `substep(model, data)` every physics step; the policy itself is
    evaluated once per `CTRL_DT` and its target held in between, exactly as the
    training env's `n_substeps` decimation does.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        policy: WalkPolicy | None = None,
        command=(0.0, 0.0, 0.0),
        key: str = "knees_bent",
        gait_freq_hz: float = GAIT_FREQ_HZ,
    ):
        self.policy = policy or WalkPolicy()
        self.model = model
        # The pose the policy's action deltas are defined about is fixed by the
        # checkpoint. It is NOT the scene's reset keyframe: on a slope that pose
        # leans and pitches its ankles, and reading the default from it would
        # silently move the policy's operating point.
        from rl.environment import robot as robot_mod

        self.default_pose = robot_mod.KNEES_BENT_QPOS[7 : 7 + N_JOINTS].copy()
        self.command = np.clip(
            np.asarray(command, dtype=np.float64), CMD_LIMITS[:, 0], CMD_LIMITS[:, 1]
        )
        self.decimation = max(1, int(round(CTRL_DT / model.opt.timestep)))
        self._phase_dt = 2.0 * np.pi * CTRL_DT * gait_freq_hz

        self._linvel_adr = model.sensor_adr[model.sensor("local_linvel_pelvis").id]
        self._gyro_adr = model.sensor_adr[model.sensor("gyro_pelvis").id]
        self._imu_site = model.site("imu_in_pelvis").id
        self.reset()

    def reset(self) -> None:
        self.phase = PHASE_INIT.copy()
        self.last_action = np.zeros(N_JOINTS)
        self.motor_targets = self.default_pose.copy()
        self._k = 0

    def observe(self, data: mujoco.MjData) -> np.ndarray:
        gravity = data.site_xmat[self._imu_site].reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
        return np.concatenate([
            data.sensordata[self._linvel_adr : self._linvel_adr + 3],   # 3
            data.sensordata[self._gyro_adr : self._gyro_adr + 3],       # 3
            gravity,                                                    # 3
            self.command,                                               # 3
            data.qpos[7 : 7 + N_JOINTS] - self.default_pose,            # 29
            data.qvel[6 : 6 + N_JOINTS],                                # 29
            self.last_action,                                           # 29
            np.concatenate([np.cos(self.phase), np.sin(self.phase)]),    # 4
        ])

    def substep(self, data: mujoco.MjData) -> np.ndarray:
        """Advance one physics substep's worth of control; returns motor targets."""
        if self._k % self.decimation == 0:
            action = self.policy(self.observe(data))
            self.last_action = action.copy()
            self.motor_targets = self.default_pose + ACTION_SCALE * action
            self.phase = (
                np.fmod(self.phase + self._phase_dt + np.pi, 2.0 * np.pi) - np.pi
            )
        self._k += 1
        data.ctrl[:] = self.motor_targets
        return self.motor_targets

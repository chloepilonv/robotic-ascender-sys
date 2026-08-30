"""The 103-d observation and the mels demo policy, in numpy, matching theirs.

Two halves, both ports of team code that stays the source of truth:

1. `PlaygroundObservation` reproduces `G1ClimbAscender._get_obs`'s `"state"`
   vector (`rl/environment/climb_env.py:419-480`) from a plain `mujoco.MjData`.
   Their version runs on `mjx.Data` under JAX; ours reads the same sensors and
   the same slices off the C data struct. The layout, in order:

       index    width  meaning                       source (climb_env.py)
       0:3      3      pelvis linear velocity,       get_local_linvel(data,
                       PELVIS frame, m/s               "pelvis")  -> sensor
                                                       `local_linvel_pelvis`
       3:6      3      pelvis angular velocity,      get_gyro(data, "pelvis")
                       PELVIS frame, rad/s             -> sensor `gyro_pelvis`
       6:9      3      projected gravity, unit,      site_xmat[imu_in_pelvis].T
                       pelvis-IMU frame                @ [0, 0, -1]      :431
       9:12     3      command [lin_vel_x m/s,       info["command"]
                       lin_vel_y m/s,
                       ang_vel_yaw rad/s]
       12:41    29     joint angle - default_pose,   qpos[7:36] - default   :440
                       radians
       41:70    29     joint velocity, rad/s         qvel[6:35]             :449
       70:99    29     last action (raw policy       info["last_act"]
                       output, unscaled)
       99:103   4      gait phase                    [cos(p0), cos(p1),
                                                      sin(p0), sin(p1)]     :458

   NOTE the phase packing: they concatenate cos(phase) THEN sin(phase), where
   phase is the 2-vector [left, right] -- so it is cos,cos,sin,sin, not
   cos,sin,cos,sin.

   NOTE the joint slices stop at `_slide_qposadr` (36) / `_slide_dofadr` (35):
   the appended ascender coordinate is deliberately NOT in the observation.

   NOISE: theirs adds uniform noise scaled by `noise_config` (level 1.0 at
   training time; gravity 0.05, gyro 0.2, joint_pos 0.03, joint_vel 1.5,
   linvel 0.1). The harness runs with noise OFF -- a demo wants the policy's
   best behaviour, and a deterministic obs is what makes the parity test
   meaningful. Declared as a gap in PARITY.md.

   SENSOR FIREWALL: `local_linvel_pelvis` is a simulator body-velocity sensor.
   On a real G1 it is not directly measurable and would come from a state
   estimator fusing IMU + kinematics. It is THEIR observation contract, not
   ours, so we reproduce it -- but it is a cheat and it is flagged here.

2. `MelsPolicy` is `rl/scripts/viewer.py:80-100`'s `load_mels_policy` with the
   numpy body lifted out and the JAX wrapper dropped. MLP
   103 -> 512 -> 256 -> 128 -> 58 with per-channel obs normalisation and swish
   activations; the first 29 outputs are the action means (the second 29 are
   log-stds, ignored -- deterministic evaluation).

   Output: (29,) raw action. The caller turns it into motor targets with
   `default_pose + action_scale * action` (climb_env.py:359).

`GaitPhase` advances the clock exactly as their `step` does
(climb_env.py:388-389): phase starts at [0, pi], gait frequency is drawn
U(1.25, 1.5) Hz at reset (climb_env.py:317), phase_dt = 2*pi*dt*freq, and the
increment is wrapped to (-pi, pi] AFTER the observation is built.
"""

import numpy as np

DEFAULT_POLICY_PATH_PARTS = ("rl", "policies", "mels_g1_joystick.npz")
OBSERVATION_SIZE = 103
ACTION_SIZE = 29
PHASE_INITIAL = np.array([0.0, np.pi])
GAIT_FREQUENCY_RANGE_HZ = (1.25, 1.5)


class GaitPhase:
    """The two-legged gait clock their observation carries."""

    def __init__(self, control_dt_seconds: float, gait_frequency_hz: float = 1.375):
        self.control_dt_seconds = float(control_dt_seconds)
        self.set_frequency(gait_frequency_hz)
        self.phase_radians = PHASE_INITIAL.copy()

    def set_frequency(self, gait_frequency_hz: float) -> None:
        """phase_dt = 2*pi*dt*freq -- climb_env.py:318."""
        self.gait_frequency_hz = float(gait_frequency_hz)
        self.phase_step_radians = (
            2.0 * np.pi * self.control_dt_seconds * self.gait_frequency_hz
        )

    def reset(self) -> None:
        self.phase_radians = PHASE_INITIAL.copy()

    def advance(self) -> None:
        """fmod(phase + phase_dt + pi, 2pi) - pi -- climb_env.py:388-389."""
        wrapped = self.phase_radians + self.phase_step_radians + np.pi
        self.phase_radians = np.fmod(wrapped, 2.0 * np.pi) - np.pi

    def as_observation(self) -> np.ndarray:
        """(4,) [cos(p0), cos(p1), sin(p0), sin(p1)] -- climb_env.py:458-460."""
        return np.concatenate([np.cos(self.phase_radians), np.sin(self.phase_radians)])


class PlaygroundObservation:
    """Builds their 103-d `state` vector from a plain `mujoco.MjData`.

    Inputs : model metadata from `team_env.describe_team_environment`, then per
             call a forwarded `MjData`, the 3-vector command, the previous raw
             action (29,), and a `GaitPhase`.
    Output : (103,) float64 observation, layout as documented at module top.
    """

    def __init__(self, model, meta, noise_level: float = 0.0, random_seed: int = 0):
        self.model = model
        self.default_pose_radians = np.asarray(meta["default_pose_radians"])
        self.joint_qpos_slice = slice(7, meta["slide_qpos_address"])
        self.joint_qvel_slice = slice(6, meta["slide_dof_address"])
        self.pelvis_imu_site_id = meta["pelvis_imu_site_id"]
        addresses = meta["sensor_addresses"]  # keyed by ROLE, not sensor name
        self.local_linvel_slice = slice(*addresses["pelvis_local_linvel"])
        self.gyro_slice = slice(*addresses["pelvis_gyro"])
        # Noise is OFF by default; kept switchable so a future run can measure
        # the policy's robustness against the level it was trained at (1.0).
        self.noise_level = float(noise_level)
        self.noise_scales = dict(meta["noise_scales"])
        self.random = np.random.default_rng(random_seed)

    def _noisy(self, value, scale_name):
        if self.noise_level == 0.0:
            return value
        scale = self.noise_scales[scale_name]
        span = 2.0 * self.random.random(np.shape(value)) - 1.0
        return value + span * self.noise_level * scale

    def projected_gravity(self, data) -> np.ndarray:
        """site_xmat[imu_in_pelvis].T @ [0,0,-1] -- climb_env.py:431.

        MuJoCo's C `site_xmat` is a flat row-major 9-vector; reshaped to 3x3 it
        is the site's rotation matrix R (columns = site axes in world). The
        transpose maps world -> site, so this is the world down-direction
        expressed in the pelvis IMU frame: [0, 0, -1] when upright.
        """
        rotation = np.asarray(data.site_xmat[self.pelvis_imu_site_id]).reshape(3, 3)
        return rotation.T @ np.array([0.0, 0.0, -1.0])

    def build(self, data, command, last_action, gait_phase) -> np.ndarray:
        linear_velocity_pelvis = np.asarray(
            data.sensordata[self.local_linvel_slice], dtype=np.float64
        )
        angular_velocity_pelvis = np.asarray(
            data.sensordata[self.gyro_slice], dtype=np.float64
        )
        gravity_pelvis = self.projected_gravity(data)
        joint_angles = np.asarray(data.qpos[self.joint_qpos_slice], dtype=np.float64)
        joint_velocities = np.asarray(data.qvel[self.joint_qvel_slice], dtype=np.float64)

        observation = np.concatenate([
            self._noisy(linear_velocity_pelvis, "linvel"),        # 3
            self._noisy(angular_velocity_pelvis, "gyro"),         # 3
            self._noisy(gravity_pelvis, "gravity"),               # 3
            np.asarray(command, dtype=np.float64),                # 3
            self._noisy(joint_angles, "joint_pos") - self.default_pose_radians,  # 29
            self._noisy(joint_velocities, "joint_vel"),           # 29
            np.asarray(last_action, dtype=np.float64),            # 29
            gait_phase.as_observation(),                          # 4
        ])
        assert observation.shape == (OBSERVATION_SIZE,), observation.shape
        return observation


class MelsPolicy:
    """The mels.ai demo G1 joystick MLP, numpy only. viewer.py:80-100."""

    def __init__(self, npz_path: str):
        self.npz_path = npz_path
        weights = np.load(npz_path)
        self.observation_mean = weights["obs_mean"].astype(np.float64)
        self.observation_std = weights["obs_std"].astype(np.float64)
        self.kernels = [weights[f"hidden_{i}_kernel"].astype(np.float64) for i in range(4)]
        self.biases = [weights[f"hidden_{i}_bias"].astype(np.float64) for i in range(4)]

    def act(self, observation: np.ndarray) -> np.ndarray:
        """(103,) observation -> (29,) raw action (deterministic mean)."""
        x = (np.asarray(observation, dtype=np.float64) - self.observation_mean) / self.observation_std
        for kernel, bias in zip(self.kernels[:3], self.biases[:3]):
            x = _swish(x @ kernel + bias)
        return (x @ self.kernels[3] + self.biases[3])[:ACTION_SIZE]

    def describe(self) -> str:
        return (f"[policy] mels MLP {self.kernels[0].shape[0]}"
                + "".join(f"->{k.shape[1]}" for k in self.kernels)
                + f", first {ACTION_SIZE} outputs used, from {self.npz_path}")


def _swish(value):
    """brax PPO's default activation -- viewer.py:90."""
    return value / (1.0 + np.exp(-value))


def default_policy_path(repository_root: str) -> str:
    import os
    return os.path.join(repository_root, *DEFAULT_POLICY_PATH_PARTS)


class TerminationCheck:
    """Their `_get_termination`, in numpy. joystick.py:426-442.

    Fallen when the torso up-vector's world z goes negative (tipped past
    horizontal), OR the feet/shins have collided with each other, OR the state
    has gone NaN. The climb env does not override this.
    """

    def __init__(self, meta):
        # Both the up-vector sensor NAME and the self-collision sensor ids come
        # from their constants / their env object via team_env, never from a
        # name typed here -- the robot is expected to change (boots, an
        # ascender end-effector, a jacket) and the harness has to follow.
        self.upvector_slice = slice(*meta["sensor_addresses"]["torso_upvector"])
        self.contact_slices = [
            slice(*address) for address in meta["self_collision_sensor_addresses"]
        ]

    def reasons(self, data) -> dict:
        upvector_z = float(data.sensordata[self.upvector_slice][-1])
        contacts = [float(data.sensordata[s][0]) > 0 for s in self.contact_slices]
        return {
            "tipped_over": upvector_z < 0.0,
            "self_collision": any(contacts),
            "not_finite": bool(
                not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all()
            ),
            "torso_upvector_z": upvector_z,
        }

    def fallen(self, data) -> bool:
        reasons = self.reasons(data)
        return bool(
            reasons["tipped_over"] or reasons["self_collision"] or reasons["not_finite"]
        )

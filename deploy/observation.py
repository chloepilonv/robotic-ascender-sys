"""Build the 103-dim policy observation from real-robot telemetry.

Mirrors `mujoco_playground` g1/joystick.py `_get_obs()["state"]` exactly. The
training-time observation is assembled in this order:

    linvel(3) gyro(3) gravity(3) command(3)
    joint_pos - default(29) joint_vel(29) last_act(29) phase(4)

Of those, 102 dims come straight off `rt/lowstate` (unitree_hg LowState_) or
are tracked by us. Exactly one -- `linvel`, the base linear velocity in the
pelvis frame -- has no sensor on the robot. In simulation playground reads it
from a `framelinvel` ground-truth sensor; on hardware it must be estimated.
It is injected here as a callable so an estimator can be dropped in without
touching this file. The default returns zeros, which is WRONG while walking
(~0.13 rad mean joint-target error at 0.75 m/s) but safe while standing.
"""
import numpy as np

from . import constants as C


def gravity_from_quaternion(quat_wxyz) -> np.ndarray:
    """Gravity direction expressed in the pelvis frame.

    Playground computes `R_pelvis^T @ [0, 0, -1]`, i.e. the world down-vector
    rotated into the body frame. Given the IMU quaternion (w, x, y, z) giving
    body->world, `R.T @ [0,0,-1]` is the negated third *row* of R (not the
    third column -- that transpose slip is sign-correct under yaw and unit
    norm, so it survives casual checks; it is caught by test_vs_mujoco.py).
    Upright robot -> [0, 0, -1].
    """
    w, x, y, z = (float(v) for v in quat_wxyz)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-9:
        raise ValueError("degenerate IMU quaternion")
    w, x, y, z = w / n, x / n, y / n, z / n
    # -R[2, :] where R is the body->world rotation matrix.
    return np.array([
        -(2.0 * (x * z - w * y)),
        -(2.0 * (y * z + w * x)),
        -(1.0 - 2.0 * (x * x + y * y)),
    ], dtype=np.float32)


def zero_linvel() -> np.ndarray:
    """Placeholder base-velocity 'estimator'. Valid only near standstill."""
    return np.zeros(3, dtype=np.float32)


class ObservationBuilder:
    """Stateful obs assembly. Owns the gait clock and the last-action memory.

    Both are hidden state that exist only inside the training loop -- the robot
    cannot tell you either one, so we must reproduce them here or the policy
    sees an observation it was never trained on.
    """

    def __init__(self, linvel_fn=zero_linvel, gait_freq_hz=C.GAIT_FREQ_HZ):
        self._linvel_fn = linvel_fn
        self._phase_dt = 2.0 * np.pi * C.CTRL_DT * gait_freq_hz
        self._obs = np.zeros(C.OBS_DIM, dtype=np.float32)  # reused every step
        self.reset()

    def reset(self) -> None:
        self._phase = C.PHASE_INIT.copy()
        self._last_action = np.zeros(C.ACTION_DIM, dtype=np.float32)
        self._obs[:] = 0.0

    @property
    def phase(self) -> np.ndarray:
        return self._phase.copy()

    def set_last_action(self, action) -> None:
        """Record the raw (unscaled) policy output, as playground does."""
        self._last_action[:] = np.asarray(action, dtype=np.float32)

    def advance_phase(self) -> None:
        """Tick the gait clock, wrapped to (-pi, pi] like the training env."""
        self._phase = np.fmod(self._phase + self._phase_dt + np.pi,
                              2.0 * np.pi) - np.pi

    def build(self, joint_pos, joint_vel, gyro, quat_wxyz, command) -> np.ndarray:
        """Assemble the observation. Returns an internal buffer -- copy to keep.

        joint_pos/joint_vel are in SDK order, length 29 (motor_state q / dq).
        """
        jp = np.asarray(joint_pos, dtype=np.float32)
        jv = np.asarray(joint_vel, dtype=np.float32)
        if jp.shape != (C.N_JOINTS,) or jv.shape != (C.N_JOINTS,):
            raise ValueError(f"expected {C.N_JOINTS} joints, got "
                             f"{jp.shape} / {jv.shape}")
        cmd = np.clip(np.asarray(command, dtype=np.float32),
                      C.CMD_LIMITS[:, 0], C.CMD_LIMITS[:, 1])

        o = self._obs
        o[C.SLICE_LINVEL]    = self._linvel_fn()
        o[C.SLICE_GYRO]      = np.asarray(gyro, dtype=np.float32)
        o[C.SLICE_GRAVITY]   = gravity_from_quaternion(quat_wxyz)
        o[C.SLICE_COMMAND]   = cmd
        o[C.SLICE_JOINT_POS] = jp - C.DEFAULT_POSE
        o[C.SLICE_JOINT_VEL] = jv
        o[C.SLICE_LAST_ACT]  = self._last_action
        o[C.SLICE_PHASE]     = np.concatenate(
            [np.cos(self._phase), np.sin(self._phase)])
        return o

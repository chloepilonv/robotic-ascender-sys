"""Shared constants for onboard G1 deployment.

Everything here is transcribed from the training-time definition and must not
drift from it:

  - joint order + default pose: mujoco_playground g1/xmls, `knees_bent` keyframe
  - action scale, rates:        mujoco_playground g1/joystick.py default_config()

The joint order below matches the Unitree SDK `G1JointIndex` enum one-for-one
(leg / leg / waist / arm / arm). That is asserted, not assumed -- see
check_joints.py, which verifies order, sign and zero-offset against the real
robot while it hangs limp in the harness.
"""
import numpy as np

N_JOINTS = 29

# Playground actuator order == expected SDK motor_state index order.
JOINT_NAMES = (
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw",
    "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw",
    "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
)
assert len(JOINT_NAMES) == N_JOINTS

# `knees_bent` keyframe ctrl block. Policy actions are deltas about this.
DEFAULT_POSE = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,      # left leg
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,      # right leg
    0.0, 0.0, 0.073,                            # waist
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,          # left arm
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,         # right arm
], dtype=np.float32)
assert DEFAULT_POSE.shape == (N_JOINTS,)

ACTION_SCALE = 0.5      # motor_target = DEFAULT_POSE + ACTION_SCALE * action
CTRL_DT = 0.02          # 50 Hz policy
LOWSTATE_HZ = 500.0     # rt/lowstate publish rate
DECIMATION = int(round(LOWSTATE_HZ * CTRL_DT))  # 10 lowstate samples per policy step

# Gait clock. Trained with freq ~ U(1.25, 1.5) Hz sampled per episode; we hold a
# fixed mid-range value onboard so the clock is reproducible run to run.
GAIT_FREQ_HZ = 1.375
PHASE_INIT = np.array([0.0, np.pi], dtype=np.float32)

OBS_DIM = 103
ACTION_DIM = 29

# Named obs slices. The one place these offsets are written down.
SLICE_LINVEL    = slice(0, 3)      # base linear velocity, pelvis frame -- ESTIMATED
SLICE_GYRO      = slice(3, 6)      # imu_state.gyroscope
SLICE_GRAVITY   = slice(6, 9)      # gravity in pelvis frame, from imu quaternion
SLICE_COMMAND   = slice(9, 12)     # [vx, vy, wyaw]
SLICE_JOINT_POS = slice(12, 41)    # motor_state.q - DEFAULT_POSE
SLICE_JOINT_VEL = slice(41, 70)    # motor_state.dq
SLICE_LAST_ACT  = slice(70, 99)    # previous raw policy output
SLICE_PHASE     = slice(99, 103)   # [cos p0, cos p1, sin p0, sin p1]

# ---------------------------------------------------------------------------
# Observation views.
#
# The 103-dim vector above is CANONICAL: observation.py always builds it in
# full. A policy trained without base linear velocity (see the estimator
# problem) takes a 100-dim observation that is exactly this vector with
# dims 0:3 removed -- a strict subset, not a different layout.
#
# So one telemetry pipeline feeds every policy variant; each one just declares
# its view. That is what makes running two policies side by side cheap: build
# the obs once, slice it per policy.
OBS_VIEWS = {
    103: None,                      # full canonical vector
    100: np.arange(3, OBS_DIM),     # canonical minus linvel
}


def view_for(obs_dim: int):
    """Index array mapping the canonical obs to what `obs_dim` expects."""
    if obs_dim not in OBS_VIEWS:
        raise ValueError(
            f"no known observation view for {obs_dim} dims; "
            f"known: {sorted(OBS_VIEWS)}")
    return OBS_VIEWS[obs_dim]


# Command ranges the policy was trained over. Clamp before feeding.
CMD_LIMITS = np.array([[-1.0, 1.0], [-0.5, 0.5], [-1.0, 1.0]], dtype=np.float32)

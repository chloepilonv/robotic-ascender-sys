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
LOWSTATE_HZ = 1050.0    # MEASURED on the robot; docs claim 500.
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


# ---------------------------------------------------------------------------
# Unitree SDK facts. VERIFIED against unitree_sdk2_python (unitree_hg IDL and
# example/g1/low_level/g1_low_level_example.py), not assumed.
#
# The SDK's G1JointIndex enum matches JOINT_NAMES above one-for-one, so there
# is NO permutation between playground order and SDK motor order.
#
# LowCmd_.motor_cmd has 35 slots but the G1 has 29 joints -- never size a loop
# off len(motor_cmd).
MOTOR_CMD_SLOTS = 35
MOTOR_MODE_ENABLE = 1     # MotorCmd_.mode: 1 = enable, 0 = disable
MODE_PR = 0               # series control for ankle pitch/roll -- what we want
MODE_AB = 1               # parallel control for ankle A/B

# LowCmd_.mode_machine MUST be copied from the incoming LowState_.mode_machine
# before publishing, or commands are rejected. See session/02, session/04.

# PD gains the policy was TRAINED with, read from the playground model
# (actuator_gainprm and dof_damping). Use THESE on hardware, not the SDK
# example's -- the policy learned against this closed-loop response, and every
# joint differs from the example, ankle_roll and the wrists by 20x.
TRAIN_KP = np.array([
    75.0, 75.0, 75.0, 75.0, 20.0, 2.0,       # left leg
    75.0, 75.0, 75.0, 75.0, 20.0, 2.0,       # right leg
    75.0, 75.0, 75.0,                        # waist
    75.0, 75.0, 75.0, 75.0, 2.0, 2.0, 2.0,   # left arm
    75.0, 75.0, 75.0, 75.0, 2.0, 2.0, 2.0,   # right arm
], dtype=np.float32)
TRAIN_KD = np.array([
    2.0, 2.0, 2.0, 2.0, 1.0, 0.2,
    2.0, 2.0, 2.0, 2.0, 1.0, 0.2,
    2.0, 2.0, 2.0,
    2.0, 2.0, 2.0, 2.0, 0.2, 0.2, 0.2,
    2.0, 2.0, 2.0, 2.0, 0.2, 0.2, 0.2,
], dtype=np.float32)
assert TRAIN_KP.shape == TRAIN_KD.shape == (N_JOINTS,)

# For comparison only -- the gains in Unitree's own low-level example.
SDK_EXAMPLE_KP = np.array([
    60, 60, 60, 100, 40, 40, 60, 60, 60, 100, 40, 40, 60, 40, 40,
    40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40], dtype=np.float32)

# WAIST LOCK WARNING: the SDK example annotates WaistRoll (13) and WaistPitch
# (14) as "INVALID for g1 23dof/29dof with waist locked". The policy commands
# all 29 joints and its default pose carries waist_pitch = 0.073 rad. Confirm
# on the actual robot whether the waist is unlocked before running the policy.
WAIST_ROLL_IDX, WAIST_PITCH_IDX = 13, 14

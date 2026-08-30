"""The "climb mime": right-arm choreography + walk pulses, shared by sim and robot.

Cycle (period T): reach up along the imaginary rope -> grab -> pull down while
stepping forward -> hold. Angles in rad, G1 joint conventions (shoulder pitch
negative = arm forward/up, elbow positive = flexed).

    phase      t/T        arm                      legs   engaged
    reach      0.00-0.35  hip -> overhead           stop   no
    grab       0.35-0.45  hold overhead             stop   yes (cam bites)
    pull       0.45-0.85  overhead -> hip           walk   yes (tension)
    hold       0.85-1.00  hold at hip               stop   yes
"""

from __future__ import annotations

import math

RIGHT_ARM = (
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)
# Hand at the hip, elbow flexed (the ascender pulled down to the waist).
POSE_HIP = (0.35, -0.15, 0.0, 1.35, 0.0, 0.0, 0.0)
# Arm reaching up in front of the face (the ascender pushed up the rope).
POSE_UP = (-1.35, -0.20, 0.0, 0.45, 0.0, 0.0, 0.0)

# Torso: always face the rope/uphill (+x) and lean slightly into the slope.
WAIST_PITCH_LEAN = 0.20  # rad forward
YAW_GAIN = 2.0  # yaw-rate command = -YAW_GAIN * heading error (rad/s per rad)
YAW_RATE_MAX = 1.0

PERIOD_S = 3.0
WALK_SPEED = 0.35  # m/s during the pull phase
WALK_SPEED_IDLE = 0.15  # m/s during reach/grab/hold (continuous shuffle keeps the walker stable)
FAKE_TENSION_N = 300.0


def _smooth(x: float) -> float:
  """Cosine ease 0->1."""
  x = min(max(x, 0.0), 1.0)
  return 0.5 - 0.5 * math.cos(math.pi * x)


def _lerp(a, b, s):
  return tuple(ai + (bi - ai) * s for ai, bi in zip(a, b))


def yaw_rate_cmd(yaw: float, target: float = 0.0) -> float:
  """Heading hold: turn back toward `target` (rad), like a climber squaring up to the rope."""
  err = (yaw - target + math.pi) % (2 * math.pi) - math.pi
  return float(min(max(-YAW_GAIN * err, -YAW_RATE_MAX), YAW_RATE_MAX))


def step(t: float, period: float = PERIOD_S) -> dict:
  """Choreography at time t (s) -> {arm: 7 angles, walk: bool, phase, engaged, tension_N}."""
  u = (t % period) / period
  if u < 0.35:
    phase, arm, walk, eng = "reach", _lerp(POSE_HIP, POSE_UP, _smooth(u / 0.35)), False, False
  elif u < 0.45:
    phase, arm, walk, eng = "grab", POSE_UP, False, True
  elif u < 0.85:
    phase, arm, walk, eng = "pull", _lerp(POSE_UP, POSE_HIP, _smooth((u - 0.45) / 0.40)), True, True
  else:
    phase, arm, walk, eng = "hold", POSE_HIP, False, True
  return {
    "phase": phase,
    "arm": arm,
    "walk": walk,
    "engaged": eng,
    "tension_N": FAKE_TENSION_N if phase == "pull" else (60.0 if eng else 0.0),
    "waist_pitch": WAIST_PITCH_LEAN,
    "speed": WALK_SPEED if walk else WALK_SPEED_IDLE,
  }

"""Chloe's mjlab-trained rope-ascender policy, driven from plain MuJoCo.

    rl/chloe/policies/g1_ascender_slope20_v3_2026-08-30_04-35-59.onnx

This is the harness-side counterpart of `rl/chloe/scripts/sim2sim.py`: the same
observation, the same action decode, the same 50 Hz decimation, but reading a
plain `mujoco.MjData` instead of an mjlab environment. `rl/chloe/` stays the
source of truth; nothing here re-derives a number that lives there, and the
contract below was verified by a from-scratch reproduction before a line of
this file was written.

THE CONTRACT (verified, do not re-derive)

  input   `obs`  (batch, 96) float32, RAW -- the observation normaliser is
                 baked INTO the exported graph, so anything that normalises
                 here would apply it twice.
  output  `action` (batch, 29) float32 -- the Gaussian MEAN, unbounded. There
                 is no tanh and no clip; a healthy run peaks around |a| = 5.

  obs layout, in order (96 = 3 + 3 + 29 + 29 + 29 + 3):

    index    width  meaning                              source
    0:3      3      pelvis angular velocity, BODY frame  qvel[3:6]  (MuJoCo
                                                         stores a free joint's
                                                         angular velocity in
                                                         the body frame)
    3:6      3      projected gravity, unit, pelvis      quat_inv_rotate(
                    frame                                  qpos[3:7],
                                                           normalize(gravity))
    6:35     29     joint angle - default pose, rad      qpos[7:36] - default
    35:64    29     joint velocity, rad/s                qvel[6:35]
    64:93    29     last RAW action (unscaled)           the previous output
    93:96    3      carriage position - pelvis           quat_inv_rotate(
                    position, rotated into the pelvis      qpos[3:7],
                    frame, metres                          xpos[carriage]
                                                           - qpos[0:3])

  action decode:  ctrl = default + per_joint_scale * action, at 50 Hz, held
                  between evaluations. Joint order = the model's joint
                  declaration order = the actuator order (measured identical).

  THE DEFAULT POSE IS NOT A CONSTANT. It is whatever
  `rope_rail.add_rope_rail` returns for this slope -- the right arm is
  IK-solved so the ascender channel lies on the rope, so the numbers depend on
  the rope height and the spawn pose. Call it; never transcribe it.

WHAT THE POLICY DOES NOT HAVE. No command input, no gait clock, no stop. It
was trained on one job -- climb a fixed line on a 10-30 degree slope -- and it
does that job for as long as it is stepped.

SO STOP IS NOT A NETWORK INPUT, IT IS A HELD POSE (`.go`, user's ruling
2026-08-30). On a rope the only control that matters is go / don't go, and the
honest way to get it out of a network with no command port is to stop asking
the network: `.go = False` freezes `data.ctrl` at the last targets the policy
wrote, so the PD holds the legs where they are and the ascender's ratchet
holds the body on the line. Physics keeps stepping -- this is a robot standing
still, not a paused simulation. `.go = True` hands control back, and the
network's first observation carries the FROZEN `last_action`, which is the
honest thing to feed it: from the policy's point of view no time passed.
`hold_blend_seconds` optionally eases the frozen targets toward the reset pose
while stopped; 0.0 (the default, measured sufficient) is a pure freeze.

Inputs  : a compiled `mujoco.MjModel` built by `chloe_worlds.build_plant`
          (mjlab gains, mjlab armature, the rope rail, tilted gravity or a
          tilted plane), then a forwarded `MjData` per substep.
Outputs : `substep(data)` writes `data.ctrl` (29 floats, radians, PD targets)
          and returns nothing; `last_action` is the raw 29-vector the network
          last produced; `observe(data)` is the 96-vector, exposed for tests.
"""

from __future__ import annotations

import os
import re
import sys

import numpy as np

_HARNESS_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
REPOSITORY_ROOT = os.path.dirname(os.path.dirname(_HARNESS_DIRECTORY))
# `assets/` is a directory of programs, not a package -- `rl/chloe/task/robot.py`
# reaches `rope_rail` the same way, by putting its directory on the path. One
# helper so the two lines are written once.
ROPE_RAIL_DIRECTORY = os.path.join(REPOSITORY_ROOT, "assets", "robots", "mujoco")


def rope_rail_module():
    """`assets/robots/mujoco/rope_rail.py`, imported the way rl/chloe does."""
    if ROPE_RAIL_DIRECTORY not in sys.path:
        sys.path.insert(0, ROPE_RAIL_DIRECTORY)
    import rope_rail
    return rope_rail

# The policy this harness ships. v3 is the one the contract was verified
# against; SMOKE and v1 are earlier checkpoints in the same directory and are
# NOT interchangeable (v1 falls over inside two seconds).
DEFAULT_POLICY_RELATIVE_PATH = os.path.join(
    "rl", "chloe", "policies", "g1_ascender_slope20_v3_2026-08-30_04-35-59.onnx")

OBSERVATION_SIZE = 96
ACTION_SIZE = 29
CONTROL_DT_SECONDS = 0.02          # 50 Hz, her decimation 4 x 0.005 s

# mjlab's G1_ARTICULATION, by joint-name suffix:
#     (stiffness kp, damping kd, armature, action scale)
# Recomputed from the published rotor inertias and gear ratios. THE WHOLE ROW
# IS LOAD-BEARING: swapping the gains for the harness's walking-robot table
# (kp 75 / kd 2) or the per-joint scales for a flat 0.5 each independently
# turns +5.6 m of climbing into a fall inside two seconds. Measured, both ways.
G1_ARTICULATION = {
    "hip_pitch":       (40.1792, 2.5579, 0.0101775, 0.54755),
    "hip_yaw":         (40.1792, 2.5579, 0.0101775, 0.54755),
    "waist_yaw":       (40.1792, 2.5579, 0.0101775, 0.54755),
    "hip_roll":        (99.0984, 6.3088, 0.0251019, 0.35066),
    "knee":            (99.0984, 6.3088, 0.0251019, 0.35066),
    "ankle_pitch":     (28.5012, 1.8144, 0.0072194, 0.43858),
    "ankle_roll":      (28.5012, 1.8144, 0.0072194, 0.43858),
    "waist_roll":      (28.5012, 1.8144, 0.0072194, 0.43858),
    "waist_pitch":     (28.5012, 1.8144, 0.0072194, 0.43858),
    "shoulder_pitch":  (14.2506, 0.9072, 0.0036097, 0.43858),
    "shoulder_roll":   (14.2506, 0.9072, 0.0036097, 0.43858),
    "shoulder_yaw":    (14.2506, 0.9072, 0.0036097, 0.43858),
    "elbow":           (14.2506, 0.9072, 0.0036097, 0.43858),
    "wrist_roll":      (14.2506, 0.9072, 0.0036097, 0.43858),
    "wrist_pitch":     (16.7783, 1.0681, 0.0042500, 0.07450),
    "wrist_yaw":       (16.7783, 1.0681, 0.0042500, 0.07450),
}


def articulation_for(joint_name: str):
    """(kp, kd, armature, action_scale) for one `*_joint` name.

    Longest suffix first, so `hip_pitch` never shadows `ankle_pitch` and
    `shoulder_yaw` never shadows `waist_yaw`.
    """
    for suffix in sorted(G1_ARTICULATION, key=len, reverse=True):
        if joint_name.endswith(suffix + "_joint"):
            return G1_ARTICULATION[suffix]
    raise KeyError(f"no mjlab articulation row for joint {joint_name!r}")


def default_policy_path(repository_root: str = REPOSITORY_ROOT) -> str:
    return os.path.join(repository_root, DEFAULT_POLICY_RELATIVE_PATH)


def quaternion_inverse_rotate(quaternion_wxyz, vector_world) -> np.ndarray:
    """World vector -> the frame the quaternion describes. MuJoCo w,x,y,z."""
    import mujoco
    out = np.zeros(3)
    w, x, y, z = [float(v) for v in quaternion_wxyz]
    mujoco.mju_rotVecQuat(out, np.asarray(vector_world, dtype=float),
                          np.array([w, -x, -y, -z]))
    return out


class AscenderController:
    """Her policy, wired to a plain MuJoCo model.

    The same interface `ClimbSceneEpisode` drives on the walking worlds:

        controller.command = (3,)     accepted and IGNORED except for its
                                      first element, which is read as go/stop
                                      (see `.go`). The network has no command
                                      port; pretending otherwise would be the
                                      lie this docstring exists to prevent.
        controller.substep(data)      called once per physics substep; writes
                                      `data.ctrl` and evaluates the network on
                                      every `decimation`-th call
        controller.last_action        (29,) the raw network output
        controller.reset()            forget the action history and the held
                                      targets

    Inputs  : a compiled model whose joint order is the declaration order and
              whose actuators are one-per-hinge in that same order; then an
              `MjData` per substep.
    Outputs : `data.ctrl` (29,) PD position targets in radians, in ACTUATOR
              order (the controller does the actuator<-joint permutation
              itself, so a model that reorders actuators still works).
    """

    def __init__(self, model, default_joint_positions: dict,
                 policy_path: str | None = None,
                 control_dt_seconds: float = CONTROL_DT_SECONDS,
                 hold_blend_seconds: float = 0.0, verbose: bool = True):
        import mujoco
        import onnxruntime

        self._mujoco = mujoco
        self._rope_rail = rope_rail_module()
        self.model = model
        self.policy_path = policy_path or default_policy_path()
        self.session = onnxruntime.InferenceSession(
            self.policy_path, providers=["CPUExecutionProvider"])
        input_shape = self.session.get_inputs()[0].shape
        self.input_name = self.session.get_inputs()[0].name
        self.observation_size = int(input_shape[-1])
        if self.observation_size not in (OBSERVATION_SIZE, OBSERVATION_SIZE + 1):
            raise ValueError(
                f"{self.policy_path} takes {input_shape}, not (batch,"
                f" {OBSERVATION_SIZE}) or (batch, {OBSERVATION_SIZE + 1})."
                " This is not the ascender policy.")
        # v4+ (her v7 onward in the app) carries ONE extra input: the climb
        # MODE bit, WALK 0 / SLIDE 1, flipped by the runtime from rope
        # progress and the ascender-to-pelvis gap (rl/chloe/task/climb_mode.py;
        # re-implemented below in numpy because hers is torch).
        self.has_mode_bit = self.observation_size == OBSERVATION_SIZE + 1

        # --- the 29 actuated joints, in declaration order ------------------
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
                 for j in range(model.njnt)]
        self.joint_names = [n for n in names
                            if n and n.endswith("_joint") and n != "floating_base_joint"]
        if len(self.joint_names) != ACTION_SIZE:
            raise ValueError(f"expected {ACTION_SIZE} robot joints, found"
                             f" {len(self.joint_names)}: {self.joint_names}")
        joint_ids = np.array([model.joint(n).id for n in self.joint_names])
        self.joint_qpos_addresses = model.jnt_qposadr[joint_ids]
        self.joint_dof_addresses = model.jnt_dofadr[joint_ids]

        # --- the operating point the action deltas are defined about -------
        self.default_joint_positions = dict(default_joint_positions)
        self.default_pose_radians = np.array(
            [_matching_value(self.default_joint_positions, name)
             for name in self.joint_names])
        self.action_scale_radians = np.array(
            [articulation_for(name)[3] for name in self.joint_names])

        # --- actuator <- joint permutation ---------------------------------
        joint_index_of = {int(joint): index for index, joint in enumerate(joint_ids)}
        self.control_index = np.array([
            joint_index_of[int(model.actuator_trnid[actuator, 0])]
            for actuator in range(model.nu)])

        # --- the world -----------------------------------------------------
        self.carriage_body_id = int(model.body("rope_carriage").id)
        self.slide_joint_id = int(model.joint("rope_slide").id)
        self.slide_qpos_address = int(model.jnt_qposadr[self.slide_joint_id])
        gravity = np.asarray(model.opt.gravity, dtype=float)
        self.gravity_direction = gravity / np.linalg.norm(gravity)

        self.control_dt_seconds = float(control_dt_seconds)
        self.decimation = max(1, int(round(
            self.control_dt_seconds / float(model.opt.timestep))))
        self.hold_blend_seconds = float(hold_blend_seconds)

        # --- state ----------------------------------------------------------
        self.command = np.zeros(3)
        self.go = True
        self.last_action = np.zeros(ACTION_SIZE, dtype=np.float32)
        self.control_targets_radians = self.default_pose_radians.copy()
        self.substep_counter = 0
        self.hold_seconds = 0.0
        self.evaluations = 0
        self._reset_mode()

        if verbose:
            print(f"[chloe] policy {os.path.basename(self.policy_path)}:"
                  f" obs {self.observation_size} -> action {ACTION_SIZE},"
                  f"{' mode bit (WALK/SLIDE FSM),' if self.has_mode_bit else ''}"
                  f" decimation {self.decimation}"
                  f" ({1.0 / self.control_dt_seconds:.0f} Hz over a"
                  f" {model.opt.timestep * 1000:.0f} ms step)", flush=True)
            print(f"[chloe] gravity direction {np.round(self.gravity_direction, 4).tolist()}"
                  f"  action scale min/max"
                  f" {self.action_scale_radians.min():.4f}/{self.action_scale_radians.max():.4f}"
                  f"  default pose |max| {np.abs(self.default_pose_radians).max():.4f} rad",
                  flush=True)

    # ------------------------------------------------------------------ api
    def reset(self) -> None:
        self.last_action = np.zeros(ACTION_SIZE, dtype=np.float32)
        self.control_targets_radians = self.default_pose_radians.copy()
        self.substep_counter = 0
        self.hold_seconds = 0.0
        self.evaluations = 0
        self._reset_mode()

    # ------------------------------------------------- the climb-mode FSM
    # rl/chloe/task/climb_mode.py, in numpy. SLIDE: push the ascender
    # STROKE_M up the rope with the feet still; WALK: walk until the ascender
    # is within CATCH_UP_M (along the rope) of the pelvis; a SLIDE that has
    # not moved the stroke by SLIDE_TIMEOUT_S ends anyway. Constants copied
    # from her file (it imports torch, which this venv does not have).
    MODE_WALK, MODE_SLIDE = 0.0, 1.0
    STROKE_METERS = 0.5
    CATCH_UP_METERS = 0.30
    SLIDE_TIMEOUT_SECONDS = 3.0

    def _reset_mode(self) -> None:
        # Start in WALK: if the ascender is already within reach the FSM
        # flips to SLIDE on the first tick, so this is the safe default.
        self.mode = self.MODE_WALK
        self.slide_at_switch = 0.0
        self.phase_seconds = 0.0
        self.mode_switches = 0

    def _update_mode(self, data) -> None:
        slide_q = float(data.qpos[self.slide_qpos_address])
        # The ascender-to-pelvis gap ALONG THE ROPE, which is her world +x
        # (tilted gravity, flat floor) and the slide joint's world axis in
        # either of our frames.
        axis = np.asarray(data.xaxis[self.slide_joint_id], dtype=float)
        gap = np.asarray(data.xpos[self.carriage_body_id], dtype=float) \
            - np.asarray(data.qpos[0:3], dtype=float)
        relative_along_rope = float(np.dot(gap, axis))
        self.phase_seconds += self.control_dt_seconds
        done_slide = (self.mode == self.MODE_SLIDE and (
            slide_q - self.slide_at_switch >= self.STROKE_METERS
            or self.phase_seconds >= self.SLIDE_TIMEOUT_SECONDS))
        done_walk = (self.mode == self.MODE_WALK
                     and relative_along_rope <= self.CATCH_UP_METERS)
        if done_slide or done_walk:
            self.mode = self.MODE_WALK if done_slide else self.MODE_SLIDE
            self.slide_at_switch = slide_q
            self.phase_seconds = 0.0
            self.mode_switches += 1

    @property
    def mode_name(self) -> str:
        if not self.has_mode_bit:
            return "free"
        return "slide" if self.mode == self.MODE_SLIDE else "walk"

    def observe(self, data) -> np.ndarray:
        """(96,) or (97,) float32, exactly the layout in the module docstring
        (+ the mode bit last, for policies that take one)."""
        quaternion = data.qpos[3:7]
        carriage_offset = np.asarray(data.xpos[self.carriage_body_id], dtype=float) \
            - np.asarray(data.qpos[0:3], dtype=float)
        if self.has_mode_bit:
            self._update_mode(data)
        return np.concatenate([
            np.asarray(data.qvel[3:6], dtype=float),                        # 3
            quaternion_inverse_rotate(quaternion, self.gravity_direction),  # 3
            np.asarray(data.qpos[self.joint_qpos_addresses], dtype=float)
            - self.default_pose_radians,                                    # 29
            np.asarray(data.qvel[self.joint_dof_addresses], dtype=float),   # 29
            np.asarray(self.last_action, dtype=float),                      # 29
            quaternion_inverse_rotate(quaternion, carriage_offset),         # 3
        ] + ([np.array([self.mode])] if self.has_mode_bit else [])).astype(np.float32)

    def act(self, data) -> np.ndarray:
        """One network evaluation. -> the raw (29,) action, also stored."""
        observation = self.observe(data)[None, :]
        action = self.session.run(None, {self.input_name: observation})[0][0]
        self.last_action = np.asarray(action, dtype=np.float32)
        self.evaluations += 1
        return self.last_action

    def substep(self, data) -> None:
        """One physics substep's worth of control. Writes `data.ctrl`.

        THE RATCHET GOES HERE, before the `mj_step` this call precedes: the
        ascender's cam is the slide joint's moving lower limit, and it has to
        be raised from the CURRENT position before the solver runs or the
        carriage is free to slide back down that step's worth. It is also
        applied in `ChloeScene.step` -- the operation is `max()`, so doing it
        twice is doing it once, and a scene stepped without a controller still
        ratchets.
        """
        self._rope_rail.ratchet(self.model, data)

        if self.go:
            self.hold_seconds = 0.0
            if self.substep_counter % self.decimation == 0:
                action = self.act(data)
                self.control_targets_radians = (
                    self.default_pose_radians + self.action_scale_radians * action)
            self.substep_counter += 1
        else:
            # HOLD POSE. The targets stay where the policy left them and the
            # PD holds them there; `last_action` is frozen too, so the network
            # resumes from the step it was interrupted on. The substep counter
            # is reset so RESUME evaluates immediately rather than up to three
            # substeps later.
            self.substep_counter = 0
            self.hold_seconds += float(self.model.opt.timestep)
            if self.hold_blend_seconds > 0.0:
                fraction = min(1.0, self.hold_seconds / self.hold_blend_seconds)
                self.control_targets_radians = (
                    (1.0 - fraction) * self.control_targets_radians
                    + fraction * self.default_pose_radians)

        data.ctrl[:] = self.control_targets_radians[self.control_index]

    def describe(self) -> str:
        return (f"[chloe] AscenderController {os.path.basename(self.policy_path)}"
                f" {self.observation_size}->{ACTION_SIZE}, decimation"
                f" {self.decimation}, hold-pose stop"
                f" (blend {self.hold_blend_seconds:.2f} s)")


def _matching_value(patterns: dict, joint_name: str, fallback: float = 0.0) -> float:
    """mjlab's regex joint_pos dict -> one joint's value. Last match wins.

    `add_rope_rail` returns exactly this shape: broad patterns like
    `.*_knee_joint` alongside the specific `right_wrist_yaw_joint` angles it
    solved, and the specific ones are appended after the broad ones.
    """
    value = float(fallback)
    for pattern, candidate in patterns.items():
        if re.fullmatch(pattern, joint_name):
            value = float(candidate)
    return value

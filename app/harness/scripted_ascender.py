"""A deterministic, non-learned climbing gait for the G1 on the fixed rope.

The counterpart of `chloe_policy.AscenderController`: same plant, same rope,
same interface, no network. Where her policy is 96 numbers in and 29 numbers
out of an ONNX graph, this is a clock and a table of poses.

WHY A SCRIPT CAN CLIMB HERE AT ALL. The ascender is WELDED (mjEQ_WELD) to the
`rope_carriage`, and the carriage has exactly one degree of freedom: slide
along the rope. So the right wrist is pinned to a straight line in space and
cannot rotate at all -- the robot is holding a rigid rail. With both feet down
that is a three-point support, and a gait that only ever lifts ONE foot while
the other foot and the rail carry the body is quasi-static: nothing has to be
caught, so nothing has to be learned. The rail is also why no balance
controller appears in this file. It is not that balance was solved; it is that
the weld took it away.

THE CYCLE (user's ruling 2026-08-30: "right foot, left foot, right hand slides
up"), each phase a cosine-eased interpolation between joint keyframes:

    1  settle_left    weight onto the left foot, arm braced on the rail
    2  right_step     lift the right foot, swing it uphill, set it down
    3  settle_right   weight onto the right foot
    4  left_step      the mirror
    5  hand_slide     the right arm extends uphill along the rope; because the
                      wrist is welded to it, the CARRIAGE is what moves, and
                      `rope_rail.ratchet` catches it at its new high point
    6  pull           the arm flexes back to the braced pose with both feet
                      planted, which draws the body up under the hand

Phases 1-4 are what actually advances the pelvis; 5-6 advance the rope and add
a second propulsion channel. The carriage would follow the body up on its own
even with a rigid arm (the slide is nearly free: 3 N of cam friction), so the
arm phase is a contribution, not a requirement -- worth knowing before blaming
it for a number.

THE ANKLE IS NEVER A KEYFRAME. On this plant the pelvis stands upright in
world and the GROUND is what tilts, so a sole lies flat on the slope when

    hip_pitch + knee + ankle_pitch = -slope_radians

(verified against the reset pose: -0.312 + 0.669 - 0.712 = -0.355 rad against
a 20 degree slope's -0.349). Every leg keyframe therefore names hip pitch and
knee, and the ankle is SOLVED from them. Writing ankle angles by hand is how a
gait ends up walking on its toes at 30 degrees and its heels at 10.

STOP IS A HELD POSE, AND IT WAITS FOR THE FOOT. `.go = False` freezes the
targets -- but not mid-swing, because a frozen swing leaves a foot in the air
and one support point gone. The phase clock runs on to the next PHASE BOUNDARY
where both feet are planted (the end of `right_step`, `left_step` or `pull`)
and only then freezes. `.go = True` resumes from exactly there. This is the
stop the RL policy cannot offer at all, so it is worth doing properly.

Inputs  : a compiled `mujoco.MjModel` from `chloe_worlds.build` (the rope rail
          present, one actuator per hinge in joint declaration order), the
          `default_joint_positions` regex->radians map `rope_rail.add_rope_rail`
          returned for that build, and an `MjData` per substep.
Outputs : `substep(data)` writes `data.ctrl` -- 29 PD position targets in
          radians, in ACTUATOR order (the joint->actuator permutation is done
          here, as in `AscenderController`). `last_action` is the (29,) target
          vector expressed as a delta from the default pose, in radians, so the
          recorder's `action` column keeps meaning "what the controller asked
          for" on these worlds too. `describe()` is one line for the log.
"""

from __future__ import annotations

import math

import numpy as np

from app.harness import chloe_policy

ACTION_SIZE = chloe_policy.ACTION_SIZE
CONTROL_DT_SECONDS = chloe_policy.CONTROL_DT_SECONDS


# ------------------------------------------------------------------ the dials
# Phase durations, seconds. Slow is the whole point: a quasi-static gait is one
# whose accelerations never matter, and these numbers are what buys that.
PHASE_SECONDS = {
    "settle_left": 0.30,
    "right_step": 0.80,
    "settle_right": 0.30,
    "left_step": 0.80,
    "hand_slide": 0.45,
}
PHASE_ORDER = ("settle_left", "right_step", "settle_right", "left_step",
               "hand_slide")
# Phases that END with both feet on the ground -- the only places a hold may
# begin.
DOUBLE_SUPPORT_END = ("right_step", "left_step", "hand_slide",
                      "settle_left", "settle_right")

# The gait, in radians of joint angle, as deltas from the plant's reset pose.
HIP_PITCH_SWEEP = 0.15      # +-this about the reset hip pitch: the step
KNEE_FLOOR_RADIANS = 0.30   # the knee never straightens past this
KNEE_LIFT = 0.45            # extra knee flexion that picks the swing foot up
FOOT_CLEARANCE_METERS = 0.045   # how far the swing foot rises off the slope
WEIGHT_SHIFT_HIP_ROLL = 0.10    # both hips roll together -> the pelvis leans
WAIST_PITCH_LEAN = 0.08         # lean into the slope, like a climber
ARM_PUSH_SHOULDER_PITCH = -0.12  # the one arm command: push the cam uphill

# Stiffer than her RL gains, because nothing here has to stay inside the
# distribution a network was trained on. Multipliers on the mjlab
# (stiffness, damping) rows, by joint-name suffix.
#
# THE RIGHT ARM IS ABSENT FROM THIS TABLE ON PURPOSE, and so is every arm
# keyframe below. The wrist is WELDED to a carriage with one degree of freedom,
# so the arm is a closed kinematic chain: pelvis pose plus a locked wrist pose
# determines the arm angles, and any PD target that disagrees is a fight the
# constraint solver wins. Measured, the first way round: driving the shoulder
# and elbow through a "reach" keyframe put 1.72 rad of tracking error into
# right_wrist_yaw and dropped the pelvis from 0.67 m to 0.30 m in six seconds.
# The arm now holds the pose `rope_rail.solve_wrist` solved for the rope and
# lets the carriage -- which is nearly free, 3 N of cam friction -- follow the
# body up the line.
SCRIPTED_GAIN_SCALE = {
    "hip_pitch": (6.0, 3.0),
    "hip_roll": (6.0, 3.0),
    "hip_yaw": (6.0, 3.0),
    "knee": (6.0, 3.0),
    "ankle_pitch": (6.0, 3.0),
    "ankle_roll": (6.0, 3.0),
    "waist_yaw": (4.0, 3.0),
    "waist_roll": (4.0, 3.0),
    "waist_pitch": (4.0, 3.0),
}


def cosine_ease(fraction: float) -> float:
    """0 -> 1 with zero slope at both ends. `choreo._smooth`, same curve."""
    fraction = min(max(float(fraction), 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * fraction)


class ScriptedAscenderController:
    """The gait, wired to the same plant `AscenderController` drives.

        controller.command = (3,)   accepted; only element 0 is read, as go/stop
        controller.substep(data)    once per physics substep; writes data.ctrl
        controller.go               True = advance the gait, False = hold
        controller.last_action      (29,) targets as a delta from the reset pose
        controller.reset()          back to the start of the cycle

    Inputs  : model, `default_joint_positions` (the regex->radians map from
              `rope_rail.add_rope_rail`), the slope in degrees (the ankle
              solution needs it), and the control period.
    Outputs : `data.ctrl` (29,) radians, actuator order.
    """

    def __init__(self, model, default_joint_positions: dict,
                 slope_degrees: float = 20.0,
                 control_dt_seconds: float = CONTROL_DT_SECONDS,
                 phase_seconds: dict | None = None,
                 verbose: bool = True):
        import mujoco

        self._mujoco = mujoco
        self._rope_rail = chloe_policy.rope_rail_module()
        self.model = model
        self.slope_radians = math.radians(float(slope_degrees))
        self.control_dt_seconds = float(control_dt_seconds)
        self.phase_seconds = dict(PHASE_SECONDS)
        if phase_seconds:
            self.phase_seconds.update(phase_seconds)
        self.cycle_seconds = sum(self.phase_seconds[p] for p in PHASE_ORDER)

        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
                 for j in range(model.njnt)]
        self.joint_names = [n for n in names
                            if n and n.endswith("_joint")
                            and n != "floating_base_joint"]
        if len(self.joint_names) != ACTION_SIZE:
            raise ValueError(f"expected {ACTION_SIZE} robot joints, found"
                             f" {len(self.joint_names)}")
        self.joint_index = {name: index
                            for index, name in enumerate(self.joint_names)}
        joint_ids = np.array([model.joint(n).id for n in self.joint_names])
        self.joint_qpos_addresses = np.array(
            [int(model.jnt_qposadr[j]) for j in joint_ids])
        joint_index_of = {int(j): i for i, j in enumerate(joint_ids)}
        self.control_index = np.array([
            joint_index_of[int(model.actuator_trnid[actuator, 0])]
            for actuator in range(model.nu)])
        self.foot_geom_ids = {
            side: [i for i in range(model.ngeom)
                   if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "")
                   .startswith(f"{side}_foot_")]
            for side in ("left", "right")}

        self.default_joint_positions = dict(default_joint_positions)
        self.default_pose_radians = np.array(
            [chloe_policy._matching_value(self.default_joint_positions, name)
             for name in self.joint_names])

        self._scratch = mujoco.MjData(model)
        # THE STANCE HEIGHT IS SET BY THE HARDEST POSE, NOT THE EASIEST ONE.
        # A leg with its thigh swung uphill by the full sweep reaches LESS far
        # down than one hanging at the reset angle, so a gait that targets the
        # reset pose's foot height asks the extreme keyframes for a knee that
        # does not exist -- the bisection saturates at a straight leg, the
        # target is missed by centimetres, and the pelvis loses a little height
        # every step until the robot is sitting down. Measured that way round
        # first (0.67 m -> 0.23 m in twelve seconds). So the reference is what
        # the WORST keyframe can deliver with a knee still bent to the floor
        # angle, and the whole gait crouches to it once, on purpose.
        self.reference_foot_height = {}
        for side in ("left", "right"):
            probe = self.default_pose_radians.copy()
            probe[self.joint_index[f"{side}_hip_pitch_joint"]] -= HIP_PITCH_SWEEP
            probe[self.joint_index[f"{side}_knee_joint"]] = KNEE_FLOOR_RADIANS
            self._solve_ankle(probe, side)
            self.reference_foot_height[side] = self._foot_height(probe, side)
        self.keyframes = self._build_keyframes()

        self.command = np.zeros(3)
        self.go = True
        self.phase_index = 0
        self.phase_elapsed_seconds = 0.0
        self.holding = False
        self.control_targets_radians = self.keyframes["home"].copy()
        self.last_action = np.zeros(ACTION_SIZE, dtype=np.float32)
        self.cycles_completed = 0
        self.total_seconds = 0.0

        if verbose:
            print(f"[scripted] gait {self.cycle_seconds:.2f} s/cycle over"
                  f" {len(PHASE_ORDER)} phases"
                  f" ({', '.join(f'{p} {self.phase_seconds[p]:.2f}s' for p in PHASE_ORDER)})",
                  flush=True)
            print(f"[scripted] slope {math.degrees(self.slope_radians):.1f} deg;"
                  f" hip sweep +-{HIP_PITCH_SWEEP:.2f} rad, knees solved to hold"
                  f" the foot at {self.reference_foot_height['left']:+.4f} m"
                  f" under the pelvis (reset pose reaches"
                  f" {self._foot_height(self.default_pose_radians, 'left'):+.4f} m,"
                  f" so the gait crouches"
                  f" {abs(self.reference_foot_height['left'] - self._foot_height(self.default_pose_radians, 'left')) * 100:.1f} cm),"
                  f" ankles solved for a flat sole;"
                  f" right arm never commanded away from the rope pose",
                  flush=True)

    # --------------------------------------------------------- the kinematics
    def _foot_height(self, pose, side) -> float:
        """Lowest point of one foot, metres, in a pelvis frame at the origin.

        Pure forward kinematics on a scratch `MjData`: no contacts, no weld, no
        solver. It answers the one question every leg keyframe has to ask --
        "with these hip and knee angles, where is the sole?" -- so the gait can
        keep the pelvis at one height instead of sinking a little every cycle.
        """
        mujoco = self._mujoco
        data = self._scratch
        mujoco.mj_resetData(self.model, data)
        data.qpos[0:3] = (0.0, 0.0, 0.0)
        data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        data.qpos[self.joint_qpos_addresses] = pose
        mujoco.mj_forward(self.model, data)
        return min(float(data.geom_xpos[i][2] - self.model.geom_size[i][0])
                   for i in self.foot_geom_ids[side])

    def _solve_knee(self, pose, side, target_height) -> np.ndarray:
        """Bisect this leg's knee until the sole sits at `target_height`.

        The foot rises monotonically with knee flexion, so a bisection over the
        joint's own range is enough and cannot get stuck. The ankle is re-solved
        inside the loop, because the flat-sole condition reads the knee.
        """
        knee_index = self.joint_index[f"{side}_knee_joint"]
        low, high = KNEE_FLOOR_RADIANS, 2.2
        pose = pose.copy()
        for _ in range(40):
            middle = 0.5 * (low + high)
            pose[knee_index] = middle
            self._solve_ankle(pose, side)
            if self._foot_height(pose, side) < target_height:
                low = middle
            else:
                high = middle
        pose[knee_index] = 0.5 * (low + high)
        self._solve_ankle(pose, side)
        return pose

    def _solve_ankle(self, pose, side, extra_toe_up: float = 0.0) -> None:
        """The flat-sole condition, in place: hip + knee + ankle = -slope."""
        hip = pose[self.joint_index[f"{side}_hip_pitch_joint"]]
        knee = pose[self.joint_index[f"{side}_knee_joint"]]
        pose[self.joint_index[f"{side}_ankle_pitch_joint"]] = (
            -self.slope_radians - hip - knee - extra_toe_up)

    # --------------------------------------------------------- the keyframes
    def _pose(self, left_hip=0.0, right_hip=0.0, left_lift=0.0, right_lift=0.0,
              hip_roll=0.0, shoulder_push=0.0) -> np.ndarray:
        """One keyframe.

        `left_hip` / `right_hip` displace hip pitch (negative = thigh uphill);
        `left_lift` / `right_lift` are how far that sole should ride above the
        slope, in METRES, and the knee is solved to deliver it; `hip_roll` goes
        on BOTH hips with the same sign, which leans the pelvis sideways rather
        than splaying the legs; `shoulder_push` is the only arm command in the
        file and it is small.
        """
        pose = self.default_pose_radians.copy()
        pose[self.joint_index["waist_pitch_joint"]] += WAIST_PITCH_LEAN
        pose[self.joint_index["left_hip_roll_joint"]] += hip_roll
        pose[self.joint_index["right_hip_roll_joint"]] += hip_roll
        # The ankles roll back by the same amount, so the pelvis leans but the
        # SOLES stay flat on the slope. Without this the weight shift rolls the
        # robot onto the edges of its feet and stops paying for itself long
        # before the centre of mass has gone anywhere.
        pose[self.joint_index["left_ankle_roll_joint"]] -= hip_roll
        pose[self.joint_index["right_ankle_roll_joint"]] -= hip_roll
        pose[self.joint_index["right_shoulder_pitch_joint"]] += shoulder_push
        for side, hip_delta, lift in (("left", left_hip, left_lift),
                                      ("right", right_hip, right_lift)):
            pose[self.joint_index[f"{side}_hip_pitch_joint"]] += hip_delta
            pose = self._solve_knee(
                pose, side, self.reference_foot_height[side] + lift)
            if lift > 0.0:
                self._solve_ankle(pose, side, extra_toe_up=0.15)
        return pose

    def _build_keyframes(self) -> dict:
        """The pose table. Every entry is a (29,) vector of absolute radians.

        THE INVARIANT that makes the cycle close: a stance leg's hip pitch
        sweeps from -S (foot uphill of the pelvis, just planted) to +S (foot
        trailing, about to lift), and the swing leg does the reverse while its
        sole is lifted clear. Both feet stay where they were put; it is the
        PELVIS that travels, one sweep per step, and the carriage follows it up
        the rope.
        """
        sweep = HIP_PITCH_SWEEP
        # THE SIGN IS MEASURED, NOT GUESSED. Rolling BOTH hips by +0.10 rad
        # carries both feet toward +y (left foot +0.119 -> +0.174 m, right foot
        # -0.119 -> -0.063 m by forward kinematics), and with the feet on the
        # ground it is the PELVIS that goes the other way -- so a positive roll
        # loads the RIGHT foot. Getting this backwards puts the weight on the
        # foot that is about to leave the ground, which is what made the first
        # version sit down after three steps.
        shift_left = -WEIGHT_SHIFT_HIP_ROLL
        shift_right = +WEIGHT_SHIFT_HIP_ROLL
        clearance = FOOT_CLEARANCE_METERS
        return {
            # both feet level, the spawn stance
            "home": self._pose(),
            # weight left, right foot still trailing and ready to swing
            "left_loaded": self._pose(left_hip=-sweep, right_hip=+sweep,
                                      hip_roll=shift_left),
            "right_lifted": self._pose(left_hip=-sweep, right_hip=+sweep,
                                       right_lift=clearance,
                                       hip_roll=shift_left),
            "right_forward": self._pose(left_hip=0.0, right_hip=-sweep,
                                        right_lift=clearance,
                                        hip_roll=shift_left),
            # right foot planted uphill, left now trailing
            "right_planted": self._pose(left_hip=+sweep, right_hip=-sweep,
                                        hip_roll=shift_left),
            "right_loaded": self._pose(left_hip=+sweep, right_hip=-sweep,
                                       hip_roll=shift_right),
            "left_lifted": self._pose(left_hip=+sweep, right_hip=-sweep,
                                      left_lift=clearance,
                                      hip_roll=shift_right),
            "left_forward": self._pose(left_hip=-sweep, right_hip=0.0,
                                       left_lift=clearance,
                                       hip_roll=shift_right),
            "left_planted": self._pose(left_hip=-sweep, right_hip=+sweep,
                                       hip_roll=shift_right),
            # both feet down, the shoulder gives the cam a shove uphill
            "hand_pushed": self._pose(left_hip=-sweep, right_hip=+sweep,
                                      shoulder_push=ARM_PUSH_SHOULDER_PITCH),
            "hand_settled": self._pose(left_hip=-sweep, right_hip=+sweep),
        }

    # ---------------------------------------------------------------- the api
    def reset(self) -> None:
        self.phase_index = 0
        self.phase_elapsed_seconds = 0.0
        self.holding = False
        self.control_targets_radians = self.keyframes["home"].copy()
        self.last_action = np.zeros(ACTION_SIZE, dtype=np.float32)
        self.cycles_completed = 0
        self.total_seconds = 0.0

    @property
    def phase(self) -> str:
        return PHASE_ORDER[self.phase_index]

    def _phase_legs(self, phase: str):
        """The keyframe chain one phase interpolates along, in order."""
        return {
            "settle_left": ("hand_settled" if self.cycles_completed else "home",
                            "left_loaded"),
            "right_step": ("left_loaded", "right_lifted", "right_forward",
                           "right_planted"),
            "settle_right": ("right_planted", "right_loaded"),
            "left_step": ("right_loaded", "left_lifted", "left_forward",
                          "left_planted"),
            "hand_slide": ("left_planted", "hand_pushed", "hand_settled"),
        }[phase]

    def _targets_now(self) -> np.ndarray:
        """The interpolated pose at the current phase clock.

        A step is three legs of one interpolation -- lift, swing, plant -- not
        one, so the foot goes UP before it goes uphill.
        """
        phase = self.phase
        duration = self.phase_seconds[phase]
        fraction = 0.0 if duration <= 0.0 else min(
            1.0, self.phase_elapsed_seconds / duration)
        legs = self._phase_legs(phase)
        count = len(legs) - 1
        leg_index = min(count - 1, int(fraction * count))
        leg_fraction = fraction * count - leg_index
        start = self.keyframes[legs[leg_index]]
        end = self.keyframes[legs[leg_index + 1]]
        return start + (end - start) * cosine_ease(leg_fraction)

    def substep(self, data) -> None:
        """One physics substep of control. Writes `data.ctrl`.

        THE RATCHET GOES HERE, before the `mj_step` this call precedes -- the
        same contract `AscenderController.substep` documents, for the same
        reason (the cam is the slide joint's moving lower limit and has to be
        raised from the CURRENT position before the solver runs).
        """
        self._rope_rail.ratchet(self.model, data)
        timestep = float(self.model.opt.timestep)

        if self.holding and self.go:
            self.holding = False
        if not self.holding:
            self.control_targets_radians = self._targets_now()
            self.phase_elapsed_seconds += timestep
            self.total_seconds += timestep
            if self.phase_elapsed_seconds >= self.phase_seconds[self.phase]:
                finished = self.phase
                self.phase_elapsed_seconds = 0.0
                self.phase_index = (self.phase_index + 1) % len(PHASE_ORDER)
                if self.phase_index == 0:
                    self.cycles_completed += 1
                # STOP WAITS FOR THE FOOT: the gate may have been released
                # mid-swing, and a frozen swing leaves a foot in the air. Only
                # a boundary with both feet planted may become a hold.
                if not self.go and finished in DOUBLE_SUPPORT_END:
                    self.holding = True

        self.last_action = np.asarray(
            self.control_targets_radians - self.default_pose_radians,
            dtype=np.float32)
        data.ctrl[:] = self.control_targets_radians[self.control_index]

    def describe(self) -> str:
        return (f"[scripted] ScriptedAscenderController: {len(PHASE_ORDER)}-phase"
                f" quasi-static gait, {self.cycle_seconds:.2f} s/cycle,"
                f" no network, hold-at-double-support stop")


# ------------------------------------------------------------------ the gates
def run_headless(slope_degrees=20.0, seconds=15.0, frame="tilted_plane",
                 wind_speed=0.0, wind_heading_degrees=180.0,
                 stop_at_seconds=None, resume_at_seconds=None,
                 phase_seconds=None, verbose=True, print_every_seconds=1.0):
    """The scripted gait with no server and no graphics. -> a report dict.

    Deliberately the same shape as `chloe_worlds.run_headless` (same columns,
    same keys, same wind convention: 180 degrees is straight DOWN the slope,
    the heading that fights the climb) so the two controllers can be put in one
    table without a translation layer.
    """
    from app.harness import chloe_worlds
    from rl.environment import climb_scene as climb_scene_module

    scene = chloe_worlds.ChloeScene(slope_degrees, frame=frame, verbose=verbose,
                                    gain_scale=SCRIPTED_GAIN_SCALE,
                                    warn_outside_band=False)
    definition = dict(chloe_worlds._definition("headless", slope_degrees,
                                               "headless", "headless")[1])
    meta = chloe_worlds.describe_chloe_scene(scene, definition)
    controller = ScriptedAscenderController(
        scene.model, scene.default_joint_positions,
        slope_degrees=slope_degrees,
        control_dt_seconds=meta["control_dt_seconds"],
        phase_seconds=phase_seconds, verbose=verbose)
    substeps = meta["substeps_per_control_step"]
    control_hz = 1.0 / meta["control_dt_seconds"]

    wind = None
    if wind_speed > 0.0:
        wind = climb_scene_module.WindParams(
            speed=float(wind_speed),
            heading=math.radians(float(wind_heading_degrees)))

    spawn = scene.data.qpos[0:3].copy()
    uphill = scene.uphill_direction_world
    slide_address = scene.ascender.qpos_address
    imu_site = scene.imu_torso_site_id

    marks = {}
    stop_state = None
    report = {"slope_degrees": float(slope_degrees), "frame": frame,
              "wind_speed_mps": float(wind_speed), "fell_at_seconds": None}
    if verbose:
        print(f"{'t':>6} {'uphill':>8} {'rope':>8} {'pelvis_z':>9} {'up_z':>6}"
              f" {'gap_cm':>7} {'go':>5} phase")
    for tick in range(int(round(seconds * control_hz))):
        time_seconds = tick / control_hz
        controller.go = not (stop_at_seconds is not None
                             and stop_at_seconds <= time_seconds
                             and (resume_at_seconds is None
                                  or time_seconds < resume_at_seconds))
        for _ in range(substeps):
            controller.substep(scene.data)
            scene.step(wind)
        upright = float(scene.data.site_xmat[imu_site].reshape(3, 3)[2, 2])
        if report["fell_at_seconds"] is None and (
                upright < 0.0 or not np.isfinite(scene.data.qpos).all()):
            report["fell_at_seconds"] = time_seconds

        now = time_seconds + 1.0 / control_hz
        if stop_at_seconds is not None and stop_state is None \
                and now >= stop_at_seconds:
            stop_state = {
                "uphill_meters": float(np.dot(scene.data.qpos[0:3] - spawn, uphill)),
                "rope_meters": float(scene.data.qpos[slide_address]),
                "pelvis_height_meters": float(scene.data.qpos[2]),
                "upright": upright}
        if resume_at_seconds is not None and "at_resume" not in marks \
                and now >= resume_at_seconds:
            marks["at_resume"] = {
                "uphill_meters": float(np.dot(scene.data.qpos[0:3] - spawn, uphill)),
                "rope_meters": float(scene.data.qpos[slide_address]),
                "pelvis_height_meters": float(scene.data.qpos[2]),
                "upright": upright}
        if verbose and tick % max(1, int(round(print_every_seconds * control_hz))) == 0:
            print(f"{time_seconds:6.1f}"
                  f" {np.dot(scene.data.qpos[0:3] - spawn, uphill):+8.2f}"
                  f" {scene.data.qpos[slide_address]:+8.2f}"
                  f" {scene.data.qpos[2]:9.3f} {upright:+6.2f}"
                  f" {scene.hand_rope_distance() * 100:7.2f}"
                  f" {('GO' if controller.go else 'HOLD'):>5}"
                  f" {controller.phase}")

    report.update({
        "uphill_meters": float(np.dot(scene.data.qpos[0:3] - spawn, uphill)),
        "rope_meters": float(scene.data.qpos[slide_address]),
        "pelvis_height_meters": float(scene.data.qpos[2]),
        "upright": float(scene.data.site_xmat[imu_site].reshape(3, 3)[2, 2]),
        "hand_line_error_meters": scene.hand_rope_distance(),
        "standing": bool(report["fell_at_seconds"] is None),
        "at_stop": stop_state,
        "at_resume": marks.get("at_resume"),
        "cycles": controller.cycles_completed,
    })
    if stop_state is not None and marks.get("at_resume") is not None:
        report["hold_slide_meters"] = (marks["at_resume"]["rope_meters"]
                                       - stop_state["rope_meters"])
        report["hold_uphill_drift_meters"] = (marks["at_resume"]["uphill_meters"]
                                              - stop_state["uphill_meters"])
        report["hold_sag_meters"] = (marks["at_resume"]["pelvis_height_meters"]
                                     - stop_state["pelvis_height_meters"])
        report["resume_uphill_meters"] = (report["uphill_meters"]
                                          - marks["at_resume"]["uphill_meters"])
        report["stood_through_hold"] = bool(marks["at_resume"]["upright"] > 0.0)
    if verbose:
        print(f"[scripted] end: {'STANDING' if report['standing'] else 'FELL'}"
              f"  uphill={report['uphill_meters']:+.2f} m"
              f"  rope={report['rope_meters']:+.2f} m"
              f"  cycles={report['cycles']}"
              f"  pelvis_z={report['pelvis_height_meters']:.3f} m", flush=True)
    return report


def _matrix(frame="tilted_plane", slopes=(0.0, 10.0, 20.0, 30.0)):
    """The two tables: straight climb (15 s / 30 s, calm and wind) and stop/go."""
    print(f"\n=== SCRIPTED TABLE 1  straight climb, frame={frame} ===")
    print(f"{'slope':>6} {'secs':>5} {'wind':>6} {'uphill_m':>9} {'rope_m':>8}"
          f" {'cycles':>7} {'up_z':>6}  outcome")
    for slope in slopes:
        for seconds in (15.0, 30.0):
            for wind in (0.0, 6.0):
                report = run_headless(slope, seconds, frame=frame,
                                      wind_speed=wind, verbose=False)
                print(f"{slope:6.0f} {seconds:5.0f} {wind:6.1f}"
                      f" {report['uphill_meters']:9.2f}"
                      f" {report['rope_meters']:8.2f}"
                      f" {report['cycles']:7d}"
                      f" {report['upright']:+6.2f}"
                      f"  {'STANDING' if report['standing'] else 'FELL at %.1f s' % report['fell_at_seconds']}")

    print(f"\n=== SCRIPTED TABLE 2  stop and go: 5 s climb / 5 s HOLD / 10 s climb ===")
    print(f"{'slope':>6} {'wind':>6} {'stood':>6} {'slide_m':>8} {'drift_m':>8}"
          f" {'sag_m':>7} {'resume_m':>9} {'total_m':>8}  outcome")
    for slope in slopes:
        for wind in (0.0, 6.0):
            report = run_headless(slope, 20.0, frame=frame, wind_speed=wind,
                                  stop_at_seconds=5.0, resume_at_seconds=10.0,
                                  verbose=False)
            print(f"{slope:6.0f} {wind:6.1f}"
                  f" {('yes' if report.get('stood_through_hold') else 'NO'):>6}"
                  f" {report['hold_slide_meters']:+8.3f}"
                  f" {report['hold_uphill_drift_meters']:+8.3f}"
                  f" {report['hold_sag_meters']:+7.3f}"
                  f" {report['resume_uphill_meters']:+9.2f}"
                  f" {report['uphill_meters']:8.2f}"
                  f"  {'STANDING' if report['standing'] else 'FELL at %.1f s' % report['fell_at_seconds']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slope", type=float, default=20.0)
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--frame", default="tilted_plane",
                        choices=("tilted_plane", "tilted_gravity"))
    parser.add_argument("--wind", type=float, default=0.0,
                        help="wind speed m/s, blowing straight down the slope")
    parser.add_argument("--stop-at", type=float, default=None)
    parser.add_argument("--resume-at", type=float, default=None)
    parser.add_argument("--matrix", action="store_true")
    arguments = parser.parse_args()

    if arguments.matrix:
        _matrix(frame=arguments.frame)
    else:
        run_headless(arguments.slope, arguments.seconds, frame=arguments.frame,
                     wind_speed=arguments.wind,
                     stop_at_seconds=arguments.stop_at,
                     resume_at_seconds=arguments.resume_at)

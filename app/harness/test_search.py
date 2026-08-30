"""Can the robot find the person again? Run it; read the tables.

    ../.venv_everest/bin/python -m app.harness.test_search

The G1 has no neck: the stereo pair rides `torso_link`, and the only joint that
pans it is WAIST YAW (`guide.WaistYaw`). So "turn the head" is a waist-yaw offset
injected into the walking policy's own PD target, and the question this file
answers is whether that is enough to re-find a human who has walked out of the
+/-29 degree field of view.

  J  THE OFF-AXIS ACQUISITION. The human is PLACED 60 degrees off the robot's
     axis -- well outside the FOV, so the robot genuinely cannot see her at
     t = 0 -- and the follower is left to it. Reports every phase transition,
     the time to acquire, and the CAMERA-BEARING ERROR at the end, which is the
     number that says the cameras really did end up pointing at her.
  K  ROPE ON vs ROPE OFF. The palm is clipped to a fixed line on the roped
     worlds, so the same sweep has to work against a constraint. Both are run.
  L  S WALKS HER BEHIND THE ROBOT. She retreats down the rope past the robot
     until she is behind it; the follower must lose her, go to SEARCH, and sweep.
  M  PHYSICS PARITY, of the only kind that can be true. A SEARCHING robot's
     physics is not identical to a still one's and must not be -- the waist
     offset is a real command on a real actuator. What must be identical is the
     OFF case: the machinery built and its hook registered on every substep, the
     knob off, and not one bit different from a run with no guide at all.

WHY 60 DEGREES. The camera's horizontal half-angle is about 29 degrees at
320x240 and fovy 58, so 60 is comfortably outside it: at t = 0 the detector sees
nothing, and everything that follows is the machine finding her.
"""
import argparse
import math

import numpy as np

from app.harness import guide as guide_module
from app.harness import test_guide as test_guide_module

DEFAULT_WORLDS = ("flat_0", "terrain_free_10")
OFF_AXIS_DEGREES = 60.0
ACQUIRE_SECONDS = 30.0


def horizontal_half_field_of_view_degrees(eyes) -> float:
    """The camera's own half-width, from its intrinsics. -> degrees."""
    return math.degrees(math.atan2(eyes.width / 2.0, eyes.focal_pixels))


def place_off_axis(scene, episode, system, degrees) -> dict:
    """Turn the ROBOT so the human sits `degrees` off its axis. -> a report.

    THE ROBOT IS TURNED, NOT THE HUMAN, and that is deliberate: the human walks
    a fixed route that the rope and the terrain both follow, so moving her off
    it would also move her off the ground and out of the world the follower was
    built for. Yawing the robot's free joint puts exactly the same angle between
    them and leaves every other thing in the scene where it belongs.
    """
    import mujoco

    data = episode.data
    system.guide.place_ahead_of(np.asarray(data.qpos[0:3]))
    system.guide.write(scene.model, data)
    mujoco.mj_kinematics(scene.model, data)
    mujoco.mj_camlight(scene.model, data)

    # Where she is now, in the world, relative to the robot.
    eye = 0.5 * (data.cam_xpos[system.eyes.left_camera_id]
                 + data.cam_xpos[system.eyes.right_camera_id])
    to_human = system.guide.reference_point_world() - eye
    human_yaw = math.atan2(float(to_human[1]), float(to_human[0]))
    # Yaw the robot so its facing is `degrees` to one side of that.
    target_yaw = human_yaw - math.radians(degrees)
    half = 0.5 * target_yaw
    data.qpos[3:7] = (math.cos(half), 0.0, 0.0, math.sin(half))
    mujoco.mj_forward(scene.model, data)
    system.guide.write(scene.model, data)
    mujoco.mj_kinematics(scene.model, data)
    mujoco.mj_camlight(scene.model, data)
    return {"human_world_yaw_degrees": math.degrees(human_yaw),
            "robot_yaw_degrees": math.degrees(target_yaw),
            "requested_off_axis_degrees": degrees}


def acquisition_run(world, seconds=ACQUIRE_SECONDS, off_axis=OFF_AXIS_DEGREES,
                    walk_human=False, back_human=False) -> dict:
    """Fly the follower from an off-axis start. -> transitions and the outcome."""
    scene, episode = test_guide_module.open_world(world)
    system = guide_module.GuideSystem(scene, scene.model, episode.control_hz,
                                      verbose=False)
    episode.control_hooks.append(system.waist.apply)
    episode.reset()
    system.place(episode.spawn_position_world)
    # The guide has to be ON before the first update, or `update` re-places her
    # in front of the robot and undoes the whole point of the setup.
    system.enabled = True
    system.guide.enabled = True
    placement = place_off_axis(scene, episode, system, off_axis)

    half_fov = horizontal_half_field_of_view_degrees(system.eyes)
    ticks = int(seconds * episode.control_hz)
    transitions, samples = [], []
    previous = None
    acquired_at = None
    handover = None
    vision_ticks = vision_detections = 0
    for tick in range(ticks):
        time_seconds = tick / episode.control_hz
        command = system.update(episode.data, tick, True,
                                walk_human, back_human)
        # COUNTED ON VISION TICKS ONLY. `follower.bearing_radians` persists
        # between them, so "the bearing is not None" reads 94% on a run whose
        # detector last saw her ten seconds ago. The eyes are the thing being
        # measured, so the eyes are what gets counted.
        if tick % guide_module.EYE_RENDER_EVERY_N_TICKS == 0:
            vision_ticks += 1
            if system.latest is not None and system.latest["detected"]:
                vision_detections += 1
        episode.step(command if command is not None else np.zeros(3),
                     np.zeros(2))
        state = (system.follower.mode, system.follower.search_phase)
        if state != previous:
            # THE HAND-OVER is the moment REALIGN gives the robot back to the
            # ordinary follower, and it is where "did the cameras end up on her"
            # has to be measured. The LAST tick of a long run answers a
            # different question -- by then she has usually been walked past.
            if (previous is not None and previous[1] == "realign"
                    and state[0] != "SEARCH" and handover is None):
                handover = {
                    "time_seconds": time_seconds,
                    "camera_bearing_degrees": final_camera_bearing_degrees(
                        scene, episode, system),
                    "waist_degrees": system.waist.degrees,
                    "mode": state[0],
                }
            transitions.append((time_seconds, previous, state))
            previous = state
            if state[1] == "acquire" and acquired_at is None:
                acquired_at = time_seconds
        samples.append({
            "time_seconds": time_seconds,
            "mode": system.follower.mode,
            "phase": system.follower.search_phase,
            "waist_degrees": system.waist.degrees,
            "bearing_degrees": (None if system.follower.bearing_radians is None
                                else math.degrees(system.follower.bearing_radians)),
            "true_meters": system.true_range_meters,
            "command": np.asarray(command if command is not None else np.zeros(3),
                                  dtype=float).copy(),
        })
    # THE ANSWER: where are the cameras pointing, relative to the human, at the
    # end? Measured from the simulator (a labelled cheat, for grading only) so
    # it cannot be flattered by the same detector that did the aiming.
    final = final_camera_bearing_degrees(scene, episode, system)
    system.close()
    return {
        "world": world, "placement": placement, "half_fov_degrees": half_fov,
        "transitions": transitions, "samples": samples,
        "acquired_at_seconds": acquired_at,
        "handover": handover,
        "final_camera_bearing_degrees": final,
        "final_mode": samples[-1]["mode"],
        "final_phase": samples[-1]["phase"],
        "final_waist_degrees": samples[-1]["waist_degrees"],
        "detected_fraction": vision_detections / max(vision_ticks, 1),
        "vision_ticks": vision_ticks,
        "fell_at_seconds": episode.fell_at_seconds,
        "maximum_waist_degrees": max(abs(s["waist_degrees"]) for s in samples),
    }


def final_camera_bearing_degrees(scene, episode, system) -> float:
    """LABELLED CHEAT, grading only: the true angle between the camera's
    optical axis and the human, in the horizontal plane. -> degrees."""
    data = episode.data
    camera_id = system.eyes.left_camera_id
    axis = np.asarray(data.cam_xmat[camera_id], dtype=float).reshape(3, 3)[:, 2]
    # A MuJoCo camera looks down its own -z.
    forward = -axis
    to_human = system.guide.reference_point_world() - np.asarray(
        data.cam_xpos[camera_id], dtype=float)
    forward[2] = 0.0
    to_human[2] = 0.0
    if np.linalg.norm(forward) < 1e-9 or np.linalg.norm(to_human) < 1e-9:
        return float("nan")
    cosine = float(np.dot(forward, to_human)
                   / (np.linalg.norm(forward) * np.linalg.norm(to_human)))
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def print_run(title, result, every_seconds=1.0) -> None:
    print(f"\n{title}")
    placement = result["placement"]
    off_axis = placement["requested_off_axis_degrees"]
    half = result["half_fov_degrees"]
    print(f"  human placed {off_axis:.0f} deg off the robot's axis; the"
          f" camera's horizontal half-FOV is {half:.1f} deg, so she starts"
          f" {'OUTSIDE' if off_axis > half else 'inside'} it")
    print("  transitions: " + ", ".join(
        f"{time:.1f}s {'/'.join(x for x in (old or ('-', '')) if x)}"
        f"->{'/'.join(x for x in new if x)}"
        for time, old, new in result["transitions"]))
    handover = result["handover"]
    print("  HAND-OVER (REALIGN -> the ordinary follower): "
          + ("never reached" if handover is None else
             f"at {handover['time_seconds']:.2f} s into {handover['mode']},"
             f" camera-bearing error {handover['camera_bearing_degrees']:.1f} deg,"
             f" waist {handover['waist_degrees']:+.1f} deg"))
    acquired = result["acquired_at_seconds"]
    print(f"  time to ACQUIRE: "
          f"{'never' if acquired is None else f'{acquired:.2f} s'}"
          f" | final mode {result['final_mode']}"
          f"{'/' + result['final_phase'] if result['final_phase'] else ''}"
          f" | camera-bearing error at the END of the run"
          f" {result['final_camera_bearing_degrees']:.1f} deg"
          f" | waist ended {result['final_waist_degrees']:+.1f} deg"
          f" (peak {result['maximum_waist_degrees']:.1f})"
          f" | detector saw her on {100 * result['detected_fraction']:.0f}%"
          f" of {result['vision_ticks']} vision frames"
          f" | fell {result['fell_at_seconds']}")
    print("  | t s | mode | phase | waist deg | image bearing deg | true m |"
          " cmd x | cmd yaw |")
    print("  |---|---|---|---|---|---|---|---|")
    stride = max(1, int(every_seconds * 50))
    for sample in result["samples"][::stride]:
        bearing = ("    --" if sample["bearing_degrees"] is None
                   else f"{sample['bearing_degrees']:+6.1f}")
        print(f"  | {sample['time_seconds']:5.1f} | {sample['mode']:6s} |"
              f" {sample['phase'] or '-':8s} | {sample['waist_degrees']:+7.1f} |"
              f" {bearing} | {sample['true_meters']:5.2f} |"
              f" {sample['command'][0]:+.2f} | {sample['command'][2]:+.2f} |")


def physics_parity(world, seconds=6.0) -> dict:
    """M: with the guide OFF, none of this exists as far as the robot knows.

    THE CLAIM HAS TO BE STATED CAREFULLY, because the obvious one is false. A
    SEARCHING robot's physics is NOT identical to a non-searching one's, and it
    must not be: the waist offset is a real command on a real actuator, and the
    whole feature is that it turns the torso. Claiming otherwise would be
    claiming the feature does nothing.

    What must be identical is the OFF case. The guide's machinery is built, the
    waist hook is registered on `control_hooks` and runs on every substep of
    every tick -- and while the knob is off it must add exactly nothing. So:

      arm A   no guide system at all, no control hooks.
      arm B   guide system built, waist hook registered, knob OFF throughout.

    Same reset, same scripted command, tick for tick. Anything but zero means
    the search leaks into a run that never asked for it.
    """
    scene, episode = test_guide_module.open_world(world)
    fields = ("qpos", "qvel", "ctrl", "sensordata", "qfrc_constraint")
    ticks = int(seconds * episode.control_hz)

    def fly(with_guide_machinery):
        system = None
        episode.control_hooks.clear()
        if with_guide_machinery:
            system = guide_module.GuideSystem(scene, scene.model,
                                              episode.control_hz, verbose=False)
            episode.control_hooks.append(system.waist.apply)
        episode.reset()
        if system is not None:
            system.place(episode.spawn_position_world)
        history = {name: [] for name in fields}
        for tick in range(ticks):
            command = np.array([0.5, 0.0, 0.0])
            if system is not None:
                # enabled=False: the knob is OFF for the whole run.
                system.update(episode.data, tick, False, False)
            episode.step(command, np.zeros(2))
            for name in fields:
                history[name].append(
                    np.asarray(getattr(episode.data, name), dtype=float).copy())
        if system is not None:
            system.close()
        return {name: np.array(values) for name, values in history.items()}

    bare, machinery = fly(False), fly(True)
    result = {name: float(np.max(np.abs(bare[name] - machinery[name])))
              for name in fields}
    episode.control_hooks.clear()
    return result


def main(arguments) -> None:
    for world in arguments.worlds:
        print_run(f"J. OFF-AXIS ACQUISITION -- {world}, human standing still",
                  acquisition_run(world, arguments.seconds, arguments.off_axis))
    print_run(
        f"L. S WALKS HER BEHIND THE ROBOT -- {arguments.worlds[0]}"
        " (she retreats down the rope, past and behind)",
        acquisition_run(arguments.worlds[0], arguments.seconds, off_axis=0.0,
                        back_human=True))

    print(f"\nM. PHYSICS PARITY, no guide at all vs the guide's machinery"
          f" present with the knob OFF -- {arguments.worlds[0]}"
          f" (same reset, same command, tick for tick)")
    differences = physics_parity(arguments.worlds[0])
    print("| array | max abs difference |")
    print("|---|---|")
    for name, difference in differences.items():
        print(f"| `{name}` | {difference:.3e} |")
    worst = max(differences.values())
    print(f"  worst: {worst:.3e}"
          + ("  -- BIT-IDENTICAL" if worst == 0.0 else "  -- NOT IDENTICAL"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    # K is the same test on both kinds of world: the first is roped, the second
    # is not, and the rope is what limits the body's yaw authority.
    parser.add_argument("--worlds", nargs="+", default=list(DEFAULT_WORLDS))
    parser.add_argument("--seconds", type=float, default=ACQUIRE_SECONDS)
    parser.add_argument("--off-axis", type=float, default=OFF_AXIS_DEGREES)
    main(parser.parse_args())

"""Headless evidence for the guide follower. Run it; read the tables.

    ../.venv_everest/bin/python -m app.harness.test_guide
    ../.venv_everest/bin/python -m app.harness.test_guide --worlds flat_0

Three things get measured, because three things could be wrong and the loss
curve of a demo is "it looked fine":

  A  STEREO ACCURACY. The guide is placed at 1, 2, 4 and 8 m true range and the
     stereo measurement is compared against the simulator's own answer. This is
     the only check that the DISTANCE is real; everything downstream is built on
     it.
  B  FOLLOW, with the human walking away the whole time (W held). Reports the
     gap over time and every mode transition.
  C  CATCH UP AND STOP: the human walks for 5 s and then stands still. This is
     the behaviour the feature exists for -- the robot should close the gap and
     come to rest inside the WAIT band -- and it is the one a demo actually
     shows.

The loop here is the same one `runtime.run` flies (guide.update -> gate ->
episode.step); what is missing is only the renderer, the websocket and the
recorder, none of which can change the robot's behaviour.
"""
import argparse
import math
import os
import sys

import numpy as np

from app.harness import climb_worlds as climb_worlds_module
from app.harness import graphics as graphics_module
from app.harness import guide as guide_module

sys.path.insert(0, os.path.join(  # human-safety/ is a program, not a package
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "human-safety"))
from human_gate import HumanGate  # noqa: E402

STEREO_TEST_RANGES_METERS = (1.0, 2.0, 4.0, 8.0)
DEFAULT_WORLDS = ("flat_0", "terrain_free_10")


def open_world(name):
    """The same order `runtime.open_world` uses: surgery, sky, look, episode."""
    library = climb_worlds_module.ClimbSceneLibrary(verbose=False)
    scene, meta, definition = library.load(name)
    guide_module.attach_guide(scene, verbose=False)
    graphics_module.add_skybox(scene, verbose=False)
    graphics_module.apply_alpine_look(
        scene.model, terrain_size_meters=scene.terrain.size_xy)
    episode = climb_worlds_module.ClimbSceneEpisode(
        scene, meta, definition, name, seed=0)
    return scene, episode


def stereo_table(scene, episode) -> list:
    """A: measured range vs the simulator's, at four true ranges. -> rows."""
    import mujoco

    model, data = scene.model, scene.data
    guide = guide_module.Guide(scene.route, scene.terrain, model)
    guide.enabled = True
    eyes = guide_module.StereoEyes(model, verbose=False)
    left, right = eyes.left_camera_id, eyes.right_camera_id
    start, _ = scene.route.project_arclen(data.qpos[0:3])

    def true_range():
        eye = 0.5 * (data.cam_xpos[left] + data.cam_xpos[right])
        return float(np.linalg.norm(guide.reference_point_world() - eye))

    rows = []
    for target in STEREO_TEST_RANGES_METERS:
        # Bisect the arc length that puts the human at exactly `target`.
        low, high = start, min(start + target * 1.8 + 3.0, scene.route.length)
        for _ in range(44):
            middle = 0.5 * (low + high)
            guide.arclength_meters = middle
            guide.write(model, data)
            mujoco.mj_kinematics(model, data)
            mujoco.mj_camlight(model, data)
            if true_range() < target:
                low = middle
            else:
                high = middle
        guide.write(model, data)
        mujoco.mj_kinematics(model, data)
        mujoco.mj_camlight(model, data)
        truth = true_range()
        measurement = eyes.look(data)
        rows.append({
            "target_meters": target,
            "true_meters": truth,
            # The same truth measured to the guide's FRONT SURFACE rather than
            # its axis. Both are printed because the measurement sits between
            # them and neither alone tells the whole story: a dense matcher's
            # median over a convex body reads its near face, so the surface
            # column is the like-for-like comparison, while the axis column is
            # the literal "distance to the human" the HUD reports.
            "true_surface_meters": max(truth - guide_module.TORSO_RADIUS_METERS, 0.0),
            "measured_meters": measurement["range_meters"],
            "disparity_pixels": measurement["disparity_pixels"],
            "mask_pixels": measurement["pixels"],
            "bearing_degrees": (None if measurement["bearing_radians"] is None
                                else math.degrees(measurement["bearing_radians"])),
            "render_milliseconds": eyes.render_milliseconds,
            "match_milliseconds": eyes.match_milliseconds,
        })
    eyes.close()
    return rows


def follow_rollout(scene, episode, seconds, walk_seconds) -> dict:
    """B and C: fly the follower. `walk_seconds` is how long the human walks."""
    system = guide_module.GuideSystem(scene, scene.model, episode.control_hz,
                                      verbose=False)
    gate = HumanGate(guide_module.GuideVisionDetector(system),
                     clear_after_seconds=0.0)
    episode.reset()
    system.place(episode.spawn_position_world)
    ticks = int(seconds * episode.control_hz)
    transitions, samples = [], []
    previous_mode = None
    for tick in range(ticks):
        time_seconds = tick / episode.control_hz
        walking = time_seconds < walk_seconds
        command = system.update(episode.data, tick, True, walking)
        gate.update(episode.data, time_seconds)
        command = gate.mask(command)
        episode.step(command, np.zeros(2))
        mode = system.follower.mode
        if mode != previous_mode:
            transitions.append((time_seconds, previous_mode, mode))
            previous_mode = mode
        samples.append({
            "time_seconds": time_seconds, "mode": mode,
            "true_meters": system.true_range_meters,
            "measured_meters": system.follower.range_meters,
            "human_progress_meters": system.guide.arclength_meters,
            "command": np.asarray(command, dtype=float).copy(),
        })
    system.close()
    following = [s["true_meters"] for s in samples if s["mode"] == "FOLLOW"]
    waiting = [s for s in samples if s["mode"] == "WAIT"]
    detected = [s for s in samples if s["measured_meters"] is not None]
    errors = [abs(s["measured_meters"] - s["true_meters"]) / s["true_meters"]
              for s in detected]
    return {
        "samples": samples,
        "transitions": transitions,
        "mean_gap_while_following_meters": (float(np.mean(following))
                                            if following else float("nan")),
        "wait_episodes": sum(1 for time, old, new in transitions if new == "WAIT"),
        "wait_fraction": len(waiting) / max(len(samples), 1),
        "detected_fraction": len(detected) / max(len(samples), 1),
        "median_relative_error": (float(np.median(errors)) if errors else float("nan")),
        "final_gap_meters": samples[-1]["true_meters"],
        "final_mode": samples[-1]["mode"],
        "fell_at_seconds": episode.fell_at_seconds,
    }


def print_stereo_table(world, rows) -> None:
    print(f"\nA. STEREO ACCURACY -- {world}")
    print("| true to axis m | true to surface m | measured m | err vs axis |"
          " err vs surface | disparity px | mask px | bearing deg |")
    print("|---|---|---|---|---|---|---|---|")
    for row in rows:
        if row["measured_meters"] is None:
            print(f"| {row['true_meters']:.3f} | {row['true_surface_meters']:.3f}"
                  f" | NOT DETECTED | | | | {row['mask_pixels']} | |")
            continue
        axis_error = row["measured_meters"] - row["true_meters"]
        surface_error = row["measured_meters"] - row["true_surface_meters"]
        print(f"| {row['true_meters']:.3f} | {row['true_surface_meters']:.3f} |"
              f" {row['measured_meters']:.3f} |"
              f" {100 * axis_error / row['true_meters']:+.1f}% |"
              f" {100 * surface_error / row['true_surface_meters']:+.1f}% |"
              f" {row['disparity_pixels']:.2f} | {row['mask_pixels']} |"
              f" {row['bearing_degrees']:+.1f} |")


def print_rollout(title, result, every_seconds=2.0) -> None:
    print(f"\n{title}")
    print(f"  mode transitions: " + ", ".join(
        f"{time:.1f}s {old}->{new}" for time, old, new in result["transitions"]))
    print(f"  mean gap while FOLLOW {result['mean_gap_while_following_meters']:.2f} m"
          f" | WAIT episodes {result['wait_episodes']}"
          f" | WAIT for {100 * result['wait_fraction']:.0f}% of ticks"
          f" | detected on {100 * result['detected_fraction']:.0f}% of ticks"
          f" | median |error| {100 * result['median_relative_error']:.1f}%"
          f" | final {result['final_gap_meters']:.2f} m in {result['final_mode']}"
          f" | fell {result['fell_at_seconds']}")
    print("  | t s | mode | true gap m | measured m | human s m | cmd x | cmd yaw |")
    print("  |---|---|---|---|---|---|---|")
    stride = max(1, int(every_seconds * 50))
    for sample in result["samples"][::stride]:
        measured = ("   --" if sample["measured_meters"] is None
                    else f"{sample['measured_meters']:5.2f}")
        print(f"  | {sample['time_seconds']:5.1f} | {sample['mode']:6s} |"
              f" {sample['true_meters']:5.2f} | {measured} |"
              f" {sample['human_progress_meters']:5.2f} |"
              f" {sample['command'][0]:+.2f} | {sample['command'][2]:+.2f} |")


def main(arguments) -> None:
    for world in arguments.worlds:
        scene, episode = open_world(world)
        print_stereo_table(world, stereo_table(scene, episode))
        print_rollout(
            f"B. FOLLOW, human walking the whole {arguments.seconds:.0f} s"
            f" -- {world}",
            follow_rollout(scene, episode, arguments.seconds, arguments.seconds))
        print_rollout(
            f"C. CATCH UP AND STOP, human walks {arguments.walk_seconds:.0f} s"
            f" then stands -- {world}",
            follow_rollout(scene, episode, arguments.catchup_seconds,
                           arguments.walk_seconds),
            every_seconds=4.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds", nargs="+", default=list(DEFAULT_WORLDS))
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--walk-seconds", type=float, default=5.0)
    # C runs longer than B on purpose: the walker's real ground speed is about
    # 0.15 m/s (measured), so closing a 3-4 m gap to the WAIT band is a
    # half-minute of walking, not five seconds.
    parser.add_argument("--catchup-seconds", type=float, default=45.0)
    main(parser.parse_args())

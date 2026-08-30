"""Headless evidence for the guide follower. Run it; read the tables.

    ../.venv_everest/bin/python -m app.harness.test_guide
    ../.venv_everest/bin/python -m app.harness.test_guide --worlds flat_0

Four things get measured, because four things could be wrong and the loss
curve of a demo is "it looked fine":

  A0 THE COLOUR WINDOW, on HER jacket. The detector is a colour threshold, so
     the number that decides whether it works at all is what fraction of the
     JACKET's pixels the window keeps and what fraction of everything ELSE it
     wrongly takes. Both are measured, not eyeballed: the eye camera is rendered
     twice at each range -- once in colour, once in SEGMENTATION -- so every
     pixel is attributed to the geom it actually came from before its hue is
     counted. Re-run it whenever the human's materials change.
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
  D  THE PHYSICS CLAIM. The same scripted command flown twice, guide off and
     guide on with the human walking: the robot's state must come back
     BIT-identical. It did not, once -- the first animated guide hung its limbs
     on real hinge joints, which grew `nv` by six and moved the walker 23 cm in
     six seconds through nothing but solver arithmetic. The limbs are welded
     bodies posed through `model.body_quat` because of this table.

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


def _range_placer(scene, guide, left_camera_id, right_camera_id):
    """-> (true_range(), place_at(target_meters) -> the achieved true range).

    Shared by A0 and A so the two tables are looking at the SAME poses: both
    bisect the guide's arc length until the eye-to-chest range is the one asked
    for, which is the only way to compare a measurement against a stated range.
    """
    import mujoco

    model, data = scene.model, scene.data
    start, _ = scene.route.project_arclen(data.qpos[0:3])

    def true_range():
        eye = 0.5 * (data.cam_xpos[left_camera_id]
                     + data.cam_xpos[right_camera_id])
        return float(np.linalg.norm(guide.reference_point_world() - eye))

    def settle():
        guide.write(model, data)
        mujoco.mj_kinematics(model, data)
        mujoco.mj_camlight(model, data)

    def place_at(target):
        low, high = start, min(start + target * 1.8 + 3.0, scene.route.length)
        for _ in range(44):
            middle = 0.5 * (low + high)
            guide.arclength_meters = middle
            settle()
            if true_range() < target:
                low = middle
            else:
                high = middle
        settle()
        return true_range()

    return true_range, place_at


# ------------------------------------------------- A0: the colour window
# Every geom the hiker is made of, grouped by the material it wears -- which is
# what the colour detector can possibly distinguish. The jacket group is the
# TARGET; every other group is a distractor that must stay out of the window.
HUMAN_MATERIAL_GROUPS = {
    "jacket": ("human_hips", "human_torso", "human_chest", "human_collar",
               "human_shoulder_l", "human_shoulder_r", "human_upper_arm_l",
               "human_forearm_l", "human_upper_arm_r", "human_forearm_r"),
    "skin": ("human_neck", "human_head"),
    "beanie": ("human_beanie", "human_pompom", "human_mat"),
    "pack": ("human_pack", "human_pack_lid"),
    "glove": ("human_hand_l", "human_hand_r"),
    "pants": ("human_thigh_l", "human_thigh_r", "human_shin_l", "human_shin_r"),
    "boots": ("human_boot_l", "human_boot_r"),
}
DETECTOR_TARGET_GROUP = "jacket"


def colour_window_table(scene, episode) -> dict:
    """A0: hue/saturation/value per material, and what the window keeps.

    The colour render and the SEGMENTATION render come from the same camera at
    the same pose, so `segmentation[row, column]` names the geom that painted
    `colour[row, column]`. Pixels are pooled over all four test ranges, because
    a material that is clean at 1 m and mush at 8 m is a material the detector
    will fail on in the demo, not in this table.

    Output: `groups` -- name -> {pixels, hue/sat/value 1st-99th percentile,
    inside_window_fraction}; `scene_pixels`/`scene_inside_window` -- the same
    question asked of everything that is NOT the human (snow, sky, rope, robot),
    which is the false-positive number.
    """
    import cv2
    import mujoco

    model, data = scene.model, scene.data
    guide = guide_module.Guide(scene.route, scene.terrain, model)
    guide.enabled = True
    eyes = guide_module.StereoEyes(model, verbose=False)
    _, place_at = _range_placer(scene, guide, eyes.left_camera_id,
                                eyes.right_camera_id)

    geom_id_of = {}
    for group, names in HUMAN_MATERIAL_GROUPS.items():
        for name in names:
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if geom_id >= 0:
                geom_id_of[geom_id] = group

    pooled = {group: [] for group in HUMAN_MATERIAL_GROUPS}
    pooled["everything else"] = []
    for target in STEREO_TEST_RANGES_METERS:
        place_at(target)
        eyes.renderer.update_scene(data, camera=eyes.left_camera_id)
        colour = eyes.renderer.render().copy()
        eyes.renderer.enable_segmentation_rendering()
        eyes.renderer.update_scene(data, camera=eyes.left_camera_id)
        segmentation = eyes.renderer.render().copy()
        eyes.renderer.disable_segmentation_rendering()
        hsv = cv2.cvtColor(colour, cv2.COLOR_RGB2HSV)
        # Channel 0 is the object id, channel 1 the object type; -1 is the sky.
        object_ids = segmentation[:, :, 0]
        is_geom = segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
        claimed = np.zeros(object_ids.shape, dtype=bool)
        for geom_id, group in geom_id_of.items():
            selection = is_geom & (object_ids == geom_id)
            if selection.any():
                pooled[group].append(hsv[selection])
            claimed |= selection
        pooled["everything else"].append(hsv[~claimed])
    eyes.close()

    def window_fraction(pixels):
        if len(pixels) == 0:
            return float("nan")
        inside = ((pixels[:, 0] >= guide_module.GUIDE_HUE_RANGE[0])
                  & (pixels[:, 0] <= guide_module.GUIDE_HUE_RANGE[1])
                  & (pixels[:, 1] >= guide_module.GUIDE_MINIMUM_SATURATION)
                  & (pixels[:, 2] >= guide_module.GUIDE_MINIMUM_VALUE))
        return float(inside.mean())

    groups = {}
    for group, chunks in pooled.items():
        pixels = (np.concatenate(chunks).astype(np.int32) if chunks
                  else np.zeros((0, 3), dtype=np.int32))
        groups[group] = {
            "pixels": int(len(pixels)),
            "hue": (None if len(pixels) == 0
                    else tuple(np.percentile(pixels[:, 0], (1, 99)).round(0))),
            "saturation": (None if len(pixels) == 0
                           else tuple(np.percentile(pixels[:, 1], (1, 99)).round(0))),
            "value": (None if len(pixels) == 0
                      else tuple(np.percentile(pixels[:, 2], (1, 99)).round(0))),
            "inside_window_fraction": window_fraction(pixels),
        }
    return groups


def print_colour_window_table(world, groups) -> None:
    print(f"\nA0. THE COLOUR WINDOW ON HER JACKET -- {world}"
          f"  (window: hue {guide_module.GUIDE_HUE_RANGE[0]}-"
          f"{guide_module.GUIDE_HUE_RANGE[1]}, saturation >="
          f" {guide_module.GUIDE_MINIMUM_SATURATION}, value >="
          f" {guide_module.GUIDE_MINIMUM_VALUE};"
          f" pooled over {', '.join(f'{r:.0f}' for r in STEREO_TEST_RANGES_METERS)} m)")
    print("| material | pixels | hue 1-99% | saturation 1-99% | value 1-99% |"
          " inside the window |")
    print("|---|---|---|---|---|---|")
    order = ([DETECTOR_TARGET_GROUP]
             + [name for name in groups if name not in
                (DETECTOR_TARGET_GROUP, "everything else")]
             + ["everything else"])
    for name in order:
        row = groups.get(name)
        if row is None or row["pixels"] == 0:
            print(f"| {name} | 0 | | | | not visible |")
            continue
        mark = " <- TARGET" if name == DETECTOR_TARGET_GROUP else ""
        print(f"| {name}{mark} | {row['pixels']} |"
              f" {row['hue'][0]:.0f}-{row['hue'][1]:.0f} |"
              f" {row['saturation'][0]:.0f}-{row['saturation'][1]:.0f} |"
              f" {row['value'][0]:.0f}-{row['value'][1]:.0f} |"
              f" {100 * row['inside_window_fraction']:.1f}% |")


def stereo_table(scene, episode) -> list:
    """A: measured range vs the simulator's, at four true ranges. -> rows."""
    model, data = scene.model, scene.data
    guide = guide_module.Guide(scene.route, scene.terrain, model)
    guide.enabled = True
    eyes = guide_module.StereoEyes(model, verbose=False)
    _, place_at = _range_placer(scene, guide, eyes.left_camera_id,
                                eyes.right_camera_id)

    rows = []
    for target in STEREO_TEST_RANGES_METERS:
        truth = place_at(target)
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


# --------------------------------------------------- D: the physics claim
def physics_parity(scene, episode, seconds=6.0) -> dict:
    """D: the guide cannot move the robot. -> the worst difference, per array.

    THE CLAIM. Everything the guide does to `MjData` -- parking or walking the
    mocap root, turning six welded limb bodies through `model.body_quat`, and
    the `mj_kinematics`/`mj_camlight` it runs so the cameras see the current
    pose rather than the last one -- must leave the robot's own trajectory
    EXACTLY where it was.

    THE TEST. The same scripted command, tick for tick, flown twice from the
    same reset: once with the guide off and parked under the world, once with
    it ON and the human WALKING (so the mocap moves, the limbs swing and the
    eyes render every fifth tick). The robot's `qpos`, `qvel`, `ctrl`,
    `sensordata`, `qfrc_constraint` and `cfrc_ext` are compared at every tick.
    Anything but zero is a bug, and a small number is not a pass: a walking
    robot is chaotic, so a 1e-16 perturbation at tick 1 is a metre by tick 300.

    The command is scripted rather than taken from the follower for the obvious
    reason: with the guide on the follower writes the command, so a run driven
    by it would differ for a reason that is not physics.
    """
    fields = ("qpos", "qvel", "ctrl", "sensordata", "qfrc_constraint", "cfrc_ext")
    ticks = int(seconds * episode.control_hz)

    def fly(guide_on):
        system = guide_module.GuideSystem(scene, scene.model, episode.control_hz,
                                          verbose=False)
        episode.reset()
        system.place(episode.spawn_position_world)
        history = {name: [] for name in fields}
        for tick in range(ticks):
            # A command that makes the robot work: walk, and weave, so the
            # contact set changes and the solver is not sitting in one corner.
            command = np.array([0.5, 0.0, 0.4 * math.sin(tick / 25.0)])
            system.update(episode.data, tick, guide_on, True)
            episode.step(command, np.zeros(2))
            for name in fields:
                history[name].append(
                    np.asarray(getattr(episode.data, name), dtype=float).copy())
        system.close()
        return {name: np.array(values) for name, values in history.items()}

    off, on = fly(False), fly(True)
    return {name: float(np.max(np.abs(off[name] - on[name]))) for name in fields}


def print_physics_parity(world, differences) -> None:
    print(f"\nD. PHYSICS PARITY, guide OFF vs guide ON with the human walking"
          f" -- {world}  (same reset, same scripted command, tick for tick)")
    print("| array | max abs difference |")
    print("|---|---|")
    for name, difference in differences.items():
        print(f"| `{name}` | {difference:.3e} |")
    worst = max(differences.values())
    print(f"  worst over all {len(differences)} arrays: {worst:.3e}"
          + ("  -- BIT-IDENTICAL" if worst == 0.0 else "  -- NOT IDENTICAL"))


def main(arguments) -> None:
    for world in arguments.worlds:
        scene, episode = open_world(world)
        # A0 and A read the robot's pose to decide where to put the human, so
        # they run FIRST, on the fresh reset. Anything that flies the robot
        # (B, C, D) leaves it somewhere else, and a stereo table measured from
        # a different standpoint is a different experiment.
        print_colour_window_table(world, colour_window_table(scene, episode))
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
        print_physics_parity(world, physics_parity(scene, episode))


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

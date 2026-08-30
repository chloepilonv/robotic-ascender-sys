"""What the blizzard does to the robot's eyes. Run it; read the tables.

    ../.venv_everest/bin/python -m app.harness.test_storm
    ../.venv_everest/bin/python -m app.harness.test_storm --world flat_0

A storm that only looks like a storm is a screensaver. These are the numbers
that say the robot is really being blinded, and they are measured the same way
the follower measures: the guide is placed at a known true range, the eyes are
rendered, degraded, matched and detected, and the answer is compared against the
simulator's.

  E  DETECTION AND STEREO vs WIND SPEED, at two ranges. For each speed the same
     pose is looked at `--repeats` times, because the sensor grain is re-drawn
     every frame and one frame is an anecdote. Reports how often the human was seen at
     all, and the stereo error over the frames where she was.
  F  MAXIMUM DETECTION RANGE per speed: the furthest range at which she is still
     seen on more than half the frames, found by walking outward.
  G  A CONTACT SHEET of the left eye at each speed, written to
     `render3d_shots/storm_eyes.png`, because a table cannot show you that the
     white-out looks like a white-out.

STORM OFF IS A ROW IN EVERY TABLE. If the "off" row is not identical to a clean
run, the degradation is leaking.

THE STORM IS FOG, not a snow shower: everything it does is distance-dependent,
so the numbers to read are the RANGES, not the picture's texture.
"""
import argparse
import math
import os

import numpy as np

from app.harness import climb_worlds as climb_worlds_module
from app.harness import graphics as graphics_module
from app.harness import guide as guide_module
from app.harness import storm as storm_module
from app.harness import test_guide as test_guide_module

WIND_SPEEDS_MPS = (0.0, 6.0, 12.0, 20.0)
TEST_RANGES_METERS = (2.0, 5.0)
RANGE_LADDER_METERS = (1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 13.0,
                       16.0, 20.0)
DETECTION_MAJORITY = 0.5


def fog_reaches_the_eyes(scene, episode) -> list:
    """E0: the white-out really is on the eye images, and really is by DISTANCE.

    Not a formality. MuJoCo's own GL fog cannot be moved at runtime from Python
    (storm.py's docstring has the four measurements), and the first version of
    this feature drove the one field that IS live -- the fog COLOUR -- which
    washes every pixel equally whatever its depth. This table separates the two:
    the NEAR half of the frame (the guide, 5 m away) and the FAR half (the slope
    behind her) are reported apart, and a real white-out has to eat the far half
    first.

    Sensor noise is off here, so any change in the picture is the fog.
    """
    model = scene.model
    guide = guide_module.Guide(scene.route, scene.terrain, model)
    guide.enabled = True
    storm_vision = storm_module.StormVision(seed=0)
    eyes = guide_module.StereoEyes(model, verbose=False)   # no degradation hook
    _, place_at = test_guide_module._range_placer(
        scene, guide, eyes.left_camera_id, eyes.right_camera_id)
    place_at(5.0)

    eyes.renderer.update_scene(scene.data, camera=eyes.left_camera_id)
    clear = eyes.renderer.render().copy()
    depth = storm_module.render_depth(eyes.renderer)
    near = depth <= 6.0        # the guide and the ground under her
    far = depth > 6.0          # the slope behind, and the sky

    rows = [{"speed": None, "visibility": float("inf"),
             "near_change": 0.0, "far_change": 0.0,
             "brightness": float(clear.mean())}]
    for speed in WIND_SPEEDS_MPS:
        storm_vision.update(True, speed)
        picture = storm_module.fog_image(
            clear, depth, storm_vision.visibility_meters).astype(np.float32)
        difference = np.abs(picture - clear.astype(np.float32)).mean(axis=2)
        rows.append({
            "speed": speed,
            "visibility": storm_vision.visibility_meters,
            "near_change": float(difference[near].mean()),
            "far_change": float(difference[far].mean()),
            "brightness": float(picture.mean()),
        })
    eyes.close()
    return rows


def print_fog_table(world, rows) -> None:
    print(f"\nE0. THE WHITE-OUT IS BY DISTANCE -- {world}, left eye, guide at"
          " 5 m, sensor noise off so only the fog moves")
    print("| wind m/s | visibility m | mean change NEAR (<=6 m) |"
          " mean change FAR (>6 m) | mean brightness |")
    print("|---|---|---|---|---|")
    for row in rows:
        name = "off (clear)" if row["speed"] is None else f"{row['speed']:.0f}"
        visibility = ("--" if row["speed"] is None
                      else f"{row['visibility']:.1f}")
        print(f"| {name} | {visibility} | {row['near_change']:.1f} |"
              f" {row['far_change']:.1f} | {row['brightness']:.1f} |")
    print("  A flat colour wash would move NEAR and FAR by the same amount."
          " Fog eats the far half first, and only reaches the near half once"
          " the visibility falls below the subject's own range.")


def measure_cell(scene, eyes, guide, place_at, storm_vision, speed, target,
                 repeats) -> dict:
    """One (speed, range) cell. -> detection rate and the stereo error."""
    truth = place_at(target)
    storm_vision.update(speed is not None, 0.0 if speed is None else speed)
    seen, errors, mask_pixels = 0, [], []
    for _ in range(repeats):
        measurement = eyes.look(scene.data)
        if measurement["detected"] and measurement["range_meters"] is not None:
            seen += 1
            errors.append((measurement["range_meters"] - truth) / truth)
            mask_pixels.append(measurement["pixels"])
    return {
        "true_meters": truth,
        "detection_rate": seen / max(repeats, 1),
        "median_relative_error": (float(np.median(errors)) if errors
                                  else float("nan")),
        "spread_relative_error": (float(np.percentile(np.abs(errors), 90))
                                  if errors else float("nan")),
        "median_mask_pixels": (float(np.median(mask_pixels)) if mask_pixels
                               else 0.0),
        "visibility_meters": (float("inf") if speed is None
                              else storm_module.visibility_meters(speed)),
    }


def storm_tables(scene, episode, repeats, world) -> dict:
    """E and F, plus the images G draws."""
    model = scene.model
    guide = guide_module.Guide(scene.route, scene.terrain, model)
    guide.enabled = True
    storm_vision = storm_module.StormVision(seed=0)
    eyes = guide_module.StereoEyes(model, verbose=False,
                                   degradation=storm_vision.degrade)
    _, place_at = test_guide_module._range_placer(
        scene, guide, eyes.left_camera_id, eyes.right_camera_id)

    # `None` is the storm OFF control: no fog change and no noise.
    arms = [None] + list(WIND_SPEEDS_MPS)
    detection = {}
    for speed in arms:
        for target in TEST_RANGES_METERS:
            detection[(speed, target)] = measure_cell(
                scene, eyes, guide, place_at, storm_vision, speed, target,
                repeats)

    maximum_range = {}
    for speed in arms:
        furthest = None
        for target in RANGE_LADDER_METERS:
            if target > scene.route.length - 2.0:
                break
            cell = measure_cell(scene, eyes, guide, place_at, storm_vision,
                                speed, target, max(4, repeats // 2))
            if cell["detection_rate"] > DETECTION_MAJORITY:
                furthest = cell["true_meters"]
            else:
                break
        maximum_range[speed] = furthest

    # G: one degraded left eye per speed, at 5 m, for the contact sheet.
    pictures = []
    for speed in arms:
        place_at(5.0)
        storm_vision.update(speed is not None, 0.0 if speed is None else speed)
        measurement = eyes.look(scene.data)
        pictures.append({
            "label": ("storm OFF" if speed is None
                      else f"{speed:.0f} m/s  vis"
                           f" {storm_module.visibility_meters(speed):.1f} m"),
            "image": measurement["left_image"],
            "box": measurement["box"],
            "detected": bool(measurement["detected"]),
            "range_meters": measurement["range_meters"],
        })
    storm_vision.update(False, 0.0)
    eyes.close()
    return {"detection": detection, "maximum_range": maximum_range,
            "pictures": pictures, "world": world}


def print_tables(result, repeats) -> None:
    world = result["world"]
    print(f"\nE. DETECTION AND STEREO vs WIND SPEED -- {world},"
          f" {repeats} frames per cell (the sensor grain is re-drawn every frame)")
    print("| wind m/s | visibility m | range | detected | median err |"
          " 90th |err| | median mask px |")
    print("|---|---|---|---|---|---|---|")
    for speed in [None] + list(WIND_SPEEDS_MPS):
        for target in TEST_RANGES_METERS:
            cell = result["detection"][(speed, target)]
            name = "off" if speed is None else f"{speed:.0f}"
            visibility = ("--" if speed is None
                          else f"{cell['visibility_meters']:.1f}")
            if cell["detection_rate"] == 0.0:
                print(f"| {name} | {visibility} | {cell['true_meters']:.2f} m |"
                      f" **0%** | -- | -- | 0 |")
                continue
            print(f"| {name} | {visibility} | {cell['true_meters']:.2f} m |"
                  f" {100 * cell['detection_rate']:.0f}% |"
                  f" {100 * cell['median_relative_error']:+.1f}% |"
                  f" {100 * cell['spread_relative_error']:.1f}% |"
                  f" {cell['median_mask_pixels']:.0f} |")

    print(f"\nF. MAXIMUM DETECTION RANGE -- {world}, the furthest rung of"
          f" {RANGE_LADDER_METERS} still seen on more than half the frames")
    print("| wind m/s | visibility m | max detection range m |")
    print("|---|---|---|")
    for speed in [None] + list(WIND_SPEEDS_MPS):
        furthest = result["maximum_range"][speed]
        name = "off" if speed is None else f"{speed:.0f}"
        visibility = ("--" if speed is None
                      else f"{storm_module.visibility_meters(speed):.1f}")
        print(f"| {name} | {visibility} |"
              f" {'never seen' if furthest is None else f'{furthest:.1f}'} |")


def write_contact_sheet(pictures, output_path) -> str:
    from PIL import Image, ImageDraw, ImageFont

    tiles = [np.asarray(entry["image"]) for entry in pictures]
    height, width = tiles[0].shape[:2]
    scale = 2
    sheet = Image.new("RGB", (len(tiles) * width * scale, height * scale),
                      (12, 14, 18))
    try:
        font = ImageFont.load_default(size=15)
    except TypeError:
        font = ImageFont.load_default()
    for index, entry in enumerate(pictures):
        tile = Image.fromarray(np.asarray(entry["image"])).resize(
            (width * scale, height * scale), Image.NEAREST)
        draw = ImageDraw.Draw(tile)
        if entry["box"] is not None:
            draw.rectangle([value * scale for value in entry["box"]],
                           outline=(60, 255, 90), width=3)
        text = entry["label"] + ("\n" + (
            f"seen at {entry['range_meters']:.2f} m" if entry["detected"]
            else "NOT SEEN"))
        draw.multiline_text((8, 6), text, fill=(255, 255, 255), font=font,
                            stroke_width=3, stroke_fill=(0, 0, 0))
        sheet.paste(tile, (index * width * scale, 0))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sheet.save(output_path)
    return output_path


FOLLOWER_TEST_RANGE_METERS = 9.0


def follower_response(scene, episode, seconds=6.0,
                      range_meters=FOLLOWER_TEST_RANGE_METERS) -> list:
    """I: what the FOLLOWER does about it, which is the point of the exercise.

    The human is parked at a fixed range and the follower is fed the real vision
    stream at the real 10 Hz for `seconds`, per wind speed. Nothing is scripted:
    if the white-out hides her, the detector misses, the 1 s timeout expires and the
    machine says LOST on its own. The robot is not stepped, so the only thing
    that varies between rows is the weather.

    THE RANGE IS CHOSEN TO STRADDLE, and it has to be: 9 m is inside the clear
    weather's 10 m reach and outside the 12 and 20 m/s reaches (8 m and 6 m,
    table F). Parked at 5 m she is seen 81% of frames even in a whiteout, and
    one detection a second resets the 1 s timeout -- so a 5 m row says 100%
    FOLLOW everywhere and proves nothing at all.
    """
    model = scene.model
    guide = guide_module.Guide(scene.route, scene.terrain, model)
    guide.enabled = True
    storm_vision = storm_module.StormVision(seed=0)
    eyes = guide_module.StereoEyes(model, verbose=False,
                                   degradation=storm_vision.degrade)
    _, place_at = test_guide_module._range_placer(
        scene, guide, eyes.left_camera_id, eyes.right_camera_id)
    truth = place_at(range_meters)

    dt_seconds = 1.0 / episode.control_hz
    ticks = int(seconds * episode.control_hz)
    rows = []
    for speed in [None] + list(WIND_SPEEDS_MPS):
        storm_vision.update(speed is not None, 0.0 if speed is None else speed)
        follower = guide_module.GuideFollower()
        modes = {"FOLLOW": 0, "WAIT": 0, "LOST": 0}
        looks = seen = 0
        for tick in range(ticks):
            measurement = None
            if tick % guide_module.EYE_RENDER_EVERY_N_TICKS == 0:
                measurement = eyes.look(scene.data)
                looks += 1
                seen += 1 if measurement["detected"] else 0
            follower.update(measurement, dt_seconds)
            modes[follower.mode] += 1
        rows.append({"speed": speed, "true_meters": truth,
                     "detection_rate": seen / max(looks, 1),
                     "modes": {name: count / ticks
                               for name, count in modes.items()}})
    storm_vision.update(False, 0.0)
    eyes.close()
    return rows


def print_follower_response(world, rows, seconds) -> None:
    print(f"\nI. THE FOLLOWER\'S OWN VERDICT -- {world}, human parked at"
          f" {rows[0]['true_meters']:.2f} m, {seconds:.0f} s of real vision per"
          " row, robot not stepped")
    print("| wind m/s | visibility m | vision frames with a detection |"
          " FOLLOW | WAIT | LOST |")
    print("|---|---|---|---|---|---|")
    for row in rows:
        name = "off" if row["speed"] is None else f"{row['speed']:.0f}"
        visibility = ("--" if row["speed"] is None
                      else f"{storm_module.visibility_meters(row['speed']):.1f}")
        modes = row["modes"]
        print(f"| {name} | {visibility} |"
              f" {100 * row['detection_rate']:.0f}% |"
              f" {100 * modes['FOLLOW']:.0f}% |"
              f" {100 * modes['WAIT']:.0f}% | {100 * modes['LOST']:.0f}% |")
    print("  LOST needs a WHOLE SECOND with no detection, and the eyes run at"
          " 10 Hz -- so ten consecutive misses. The mode therefore only flips"
          " once detection falls well below half; a row that still says FOLLOW"
          " at 40% detection is the hysteresis doing its job, not the storm"
          " failing to bite.")


def physics_parity(scene, episode, seconds=6.0) -> dict:
    """H: the storm cannot move the robot. -> the worst difference, per array.

    Same scripted command, flown twice from the same reset: once with no storm,
    once with a 20 m/s white-out and the guide on, so the fog is rewritten every
    tick and the eyes render through it every fifth. The fog lives in
    `model.vis` and in the render contexts, and the noise is added to a numpy
    array after the render, so nothing the solver reads is touched -- and this
    is the table that says so rather than the sentence that claims it.
    """
    fields = ("qpos", "qvel", "ctrl", "sensordata", "qfrc_constraint", "cfrc_ext")
    ticks = int(seconds * episode.control_hz)

    def fly(storm_on):
        storm_vision = storm_module.StormVision(seed=0)
        system = guide_module.GuideSystem(scene, scene.model, episode.control_hz,
                                          verbose=False,
                                          degradation=storm_vision.degrade)
        episode.reset()
        system.place(episode.spawn_position_world)
        history = {name: [] for name in fields}
        for tick in range(ticks):
            command = np.array([0.5, 0.0, 0.4 * math.sin(tick / 25.0)])
            storm_vision.update(storm_on, 20.0 if storm_on else 0.0)
            system.update(episode.data, tick, True, True)
            episode.step(command, np.zeros(2))
            for name in fields:
                history[name].append(
                    np.asarray(getattr(episode.data, name), dtype=float).copy())
        storm_vision.update(False, 0.0)
        system.close()
        return {name: np.array(values) for name, values in history.items()}

    off, on = fly(False), fly(True)
    return {name: float(np.max(np.abs(off[name] - on[name]))) for name in fields}


def print_physics_parity(world, differences) -> None:
    print(f"\nH. PHYSICS PARITY, storm OFF vs a 20 m/s blizzard -- {world}"
          "  (same reset, same scripted command, tick for tick)")
    print("| array | max abs difference |")
    print("|---|---|")
    for name, difference in differences.items():
        print(f"| `{name}` | {difference:.3e} |")
    worst = max(differences.values())
    print(f"  worst over all {len(differences)} arrays: {worst:.3e}"
          + ("  -- BIT-IDENTICAL" if worst == 0.0 else "  -- NOT IDENTICAL"))


def main(arguments) -> None:
    scene, episode = test_guide_module.open_world(arguments.world)
    print_fog_table(arguments.world, fog_reaches_the_eyes(scene, episode))
    result = storm_tables(scene, episode, arguments.repeats, arguments.world)
    print_tables(result, arguments.repeats)
    path = write_contact_sheet(result["pictures"], arguments.output)
    print(f"\nG. the left eye at each speed, the guide 5.00 m away -> {path}")
    print_follower_response(arguments.world,
                            follower_response(scene, episode),
                            6.0)
    print_physics_parity(arguments.world, physics_parity(scene, episode))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", default="flat_0")
    parser.add_argument("--repeats", type=int, default=16)
    parser.add_argument("--output", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "render3d_shots", "storm_eyes.png"))
    main(parser.parse_args())

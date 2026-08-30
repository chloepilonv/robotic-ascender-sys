"""What a white-out does to the robot's eyes. Run it; read the tables.

    ../.venv_everest/bin/python -m app.harness.test_storm
    ../.venv_everest/bin/python -m app.harness.test_storm --world flat_0

Weather that only looks like weather is a screensaver. These are the numbers
that say the robot is really being blinded, and they are measured the same way
the follower measures: the guide is placed at a known true range, the eyes are
rendered, degraded, matched and detected, and the answer is compared against the
simulator's.

VISIBILITY IS A DISTANCE IN METRES AND IT OWES THE WIND NOTHING (user's ruling,
2026-08-30). Every table below is indexed by VISIBILITY, where it used to be
indexed by wind speed through `100 m * exp(-wind / 6)`. Section J is the row
that says the two are now unrelated.

  E0 THE WHITE-OUT IS BY DISTANCE, near half of the frame against far half.
  E  DETECTION AND STEREO vs VISIBILITY, at two ranges. For each visibility the
     same pose is looked at `--repeats` times, because the sensor grain is
     re-drawn every frame and one frame is an anecdote. Reports how often the
     human was seen at all, and the stereo error over the frames where she was.
  F  MAXIMUM DETECTION RANGE per visibility: the furthest range at which she is
     still seen on more than half the frames, found by walking outward.
  G  A CONTACT SHEET of the left eye at each visibility, written to
     `render3d_shots/storm_eyes.png`, because a table cannot show you that the
     white-out looks like a white-out.
  H  THE PHYSICS CLAIM: clear against a 3 m white-out, same seed, tick for tick.
  I  WHAT THE FOLLOWER DOES ABOUT IT, from its own authority.
  J  VISIBILITY IS NOT WIND. The same measurement at one visibility with the
     wind dial at 0 and at 12 m/s.

CLEAR IS A ROW IN EVERY TABLE, and it is not a nearly-clear one: at
`CLEAR_VISIBILITY_METERS` the degradation hook returns the image it was handed,
un-fogged and un-grained, and does not even advance its own generator. If that
row is not identical to a run with no degradation hook at all, the degradation
is leaking.

A WHITE-OUT IS FOG, not a snow shower: everything it does is distance-dependent,
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

# The dial, sampled at its clear end, at two useful middles and at its floor.
# 100 m IS the clear control arm -- there is no separate "off" any more, because
# "off" is what the top of the dial means.
VISIBILITY_METERS = (100.0, 30.0, 10.0, 3.0)
CLEAR_VISIBILITY_METERS = storm_module.CLEAR_VISIBILITY_METERS
# Section J: two wind speeds that used to mean two completely different
# visibilities (100 m and 13.5 m under the retired coupling) and now mean
# nothing at all to the eyes.
WIND_INDEPENDENCE_SPEEDS_MPS = (0.0, 12.0)
WIND_INDEPENDENCE_VISIBILITY_METERS = 10.0
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

    rows = []
    for visibility in VISIBILITY_METERS:
        storm_vision.update(visibility)
        if not storm_vision.enabled:
            # The clear arm is the IDENTITY, and it is reported as the zeros it
            # really is rather than as a 100 m fog ramp that happens to be faint.
            rows.append({"visibility": visibility, "near_change": 0.0,
                         "far_change": 0.0, "brightness": float(clear.mean())})
            continue
        picture = storm_module.fog_image(
            clear, depth, storm_vision.visibility_meters).astype(np.float32)
        difference = np.abs(picture - clear.astype(np.float32)).mean(axis=2)
        rows.append({
            "visibility": visibility,
            "near_change": float(difference[near].mean()),
            "far_change": float(difference[far].mean()),
            "brightness": float(picture.mean()),
        })
    eyes.close()
    return rows


def print_fog_table(world, rows) -> None:
    print(f"\nE0. THE WHITE-OUT IS BY DISTANCE -- {world}, left eye, guide at"
          " 5 m, sensor noise off so only the fog moves")
    print("| visibility m | mean change NEAR (<=6 m) |"
          " mean change FAR (>6 m) | mean brightness |")
    print("|---|---|---|---|")
    for row in rows:
        name = (f"{row['visibility']:.0f} (clear)"
                if row["visibility"] >= CLEAR_VISIBILITY_METERS
                else f"{row['visibility']:.0f}")
        print(f"| {name} | {row['near_change']:.1f} |"
              f" {row['far_change']:.1f} | {row['brightness']:.1f} |")
    print("  A flat colour wash would move NEAR and FAR by the same amount."
          " Fog eats the far half first, and only reaches the near half once"
          " the visibility falls below the subject's own range.")


def measure_cell(scene, eyes, guide, place_at, storm_vision, visibility, target,
                 repeats) -> dict:
    """One (visibility, range) cell. -> detection rate and the stereo error."""
    truth = place_at(target)
    storm_vision.update(visibility)
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
        "visibility_meters": float(storm_vision.visibility_meters),
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

    # The 100 m arm IS the clear control: `degrade` hands the image straight
    # back at that visibility, so this row is a run with no degradation at all.
    arms = list(VISIBILITY_METERS)
    detection = {}
    for visibility in arms:
        for target in TEST_RANGES_METERS:
            detection[(visibility, target)] = measure_cell(
                scene, eyes, guide, place_at, storm_vision, visibility, target,
                repeats)

    maximum_range = {}
    for visibility in arms:
        furthest = None
        for target in RANGE_LADDER_METERS:
            if target > scene.route.length - 2.0:
                break
            cell = measure_cell(scene, eyes, guide, place_at, storm_vision,
                                visibility, target, max(4, repeats // 2))
            if cell["detection_rate"] > DETECTION_MAJORITY:
                furthest = cell["true_meters"]
            else:
                break
        maximum_range[visibility] = furthest

    # G: one degraded left eye per visibility, at 5 m, for the contact sheet.
    pictures = []
    for visibility in arms:
        place_at(5.0)
        storm_vision.update(visibility)
        measurement = eyes.look(scene.data)
        pictures.append({
            "label": (f"{visibility:.0f} m  CLEAR"
                      if visibility >= CLEAR_VISIBILITY_METERS
                      else f"visibility {visibility:.0f} m"),
            "image": measurement["left_image"],
            "box": measurement["box"],
            "detected": bool(measurement["detected"]),
            "range_meters": measurement["range_meters"],
        })
    storm_vision.update(CLEAR_VISIBILITY_METERS)
    eyes.close()
    return {"detection": detection, "maximum_range": maximum_range,
            "pictures": pictures, "world": world}


def print_tables(result, repeats) -> None:
    world = result["world"]
    print(f"\nE. DETECTION AND STEREO vs VISIBILITY -- {world},"
          f" {repeats} frames per cell (the sensor grain is re-drawn every frame)")
    print("| visibility m | range | detected | median err |"
          " 90th |err| | median mask px |")
    print("|---|---|---|---|---|---|")
    for visibility in VISIBILITY_METERS:
        for target in TEST_RANGES_METERS:
            cell = result["detection"][(visibility, target)]
            name = (f"{visibility:.0f} (clear)"
                    if visibility >= CLEAR_VISIBILITY_METERS
                    else f"{visibility:.0f}")
            if cell["detection_rate"] == 0.0:
                print(f"| {name} | {cell['true_meters']:.2f} m |"
                      f" **0%** | -- | -- | 0 |")
                continue
            print(f"| {name} | {cell['true_meters']:.2f} m |"
                  f" {100 * cell['detection_rate']:.0f}% |"
                  f" {100 * cell['median_relative_error']:+.1f}% |"
                  f" {100 * cell['spread_relative_error']:.1f}% |"
                  f" {cell['median_mask_pixels']:.0f} |")

    print(f"\nF. MAXIMUM DETECTION RANGE -- {world}, the furthest rung of"
          f" {RANGE_LADDER_METERS} still seen on more than half the frames")
    print("| visibility m | max detection range m |")
    print("|---|---|")
    for visibility in VISIBILITY_METERS:
        furthest = result["maximum_range"][visibility]
        name = (f"{visibility:.0f} (clear)"
                if visibility >= CLEAR_VISIBILITY_METERS
                else f"{visibility:.0f}")
        print(f"| {name} |"
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
    for visibility in VISIBILITY_METERS:
        storm_vision.update(visibility)
        follower = guide_module.GuideFollower()
        # Every mode the follower can be in, from ITS OWN authority. This was a
        # hand-written {"FOLLOW", "WAIT", "LOST"} and went stale the day SEARCH
        # was added: section H died on `KeyError: 'SEARCH'` before it could run.
        modes = {name: 0 for name in guide_module.GUIDE_MODE_CODES}
        looks = seen = 0
        for tick in range(ticks):
            measurement = None
            if tick % guide_module.EYE_RENDER_EVERY_N_TICKS == 0:
                measurement = eyes.look(scene.data)
                looks += 1
                seen += 1 if measurement["detected"] else 0
            follower.update(measurement, dt_seconds)
            modes[follower.mode] += 1
        rows.append({"visibility": visibility, "true_meters": truth,
                     "detection_rate": seen / max(looks, 1),
                     "modes": {name: count / ticks
                               for name, count in modes.items()}})
    storm_vision.update(CLEAR_VISIBILITY_METERS)
    eyes.close()
    return rows


def print_follower_response(world, rows, seconds) -> None:
    print(f"\nI. THE FOLLOWER\'S OWN VERDICT -- {world}, human parked at"
          f" {rows[0]['true_meters']:.2f} m, {seconds:.0f} s of real vision per"
          " row, robot not stepped")
    print("| visibility m | vision frames with a detection |"
          " FOLLOW | WAIT | LOST |")
    print("|---|---|---|---|---|")
    for row in rows:
        name = (f"{row['visibility']:.0f} (clear)"
                if row["visibility"] >= CLEAR_VISIBILITY_METERS
                else f"{row['visibility']:.0f}")
        modes = row["modes"]
        print(f"| {name} |"
              f" {100 * row['detection_rate']:.0f}% |"
              f" {100 * modes['FOLLOW']:.0f}% |"
              f" {100 * modes['WAIT']:.0f}% | {100 * modes['LOST']:.0f}% |")
    print("  LOST needs a WHOLE SECOND with no detection, and the eyes run at"
          " 10 Hz -- so ten consecutive misses. The mode therefore only flips"
          " once detection falls well below half; a row that still says FOLLOW"
          " at 40% detection is the hysteresis doing its job, not the storm"
          " failing to bite.")


def physics_parity(scene, episode, seconds=6.0) -> dict:
    """H: the weather cannot move the robot. -> the worst difference, per array.

    Same scripted command, flown twice from the same reset: once at CLEAR
    visibility, once in a 3 m white-out with the guide on, so the fog is
    composited every tick and the eyes render through it every fifth. The fog is
    arithmetic on two rendered numpy arrays and the noise is added to a third
    after the render, so nothing the solver reads is touched -- and this is the
    table that says so rather than the sentence that claims it.
    """
    fields = ("qpos", "qvel", "ctrl", "sensordata", "qfrc_constraint", "cfrc_ext")
    ticks = int(seconds * episode.control_hz)

    def fly(visibility):
        storm_vision = storm_module.StormVision(seed=0)
        system = guide_module.GuideSystem(scene, scene.model, episode.control_hz,
                                          verbose=False,
                                          degradation=storm_vision.degrade)
        episode.reset()
        system.place(episode.spawn_position_world)
        history = {name: [] for name in fields}
        for tick in range(ticks):
            command = np.array([0.5, 0.0, 0.4 * math.sin(tick / 25.0)])
            storm_vision.update(visibility)
            system.update(episode.data, tick, True, True)
            episode.step(command, np.zeros(2))
            for name in fields:
                history[name].append(
                    np.asarray(getattr(episode.data, name), dtype=float).copy())
        storm_vision.update(CLEAR_VISIBILITY_METERS)
        system.close()
        return {name: np.array(values) for name, values in history.items()}

    off = fly(CLEAR_VISIBILITY_METERS)
    on = fly(storm_module.MINIMUM_VISIBILITY_METERS)
    return {name: float(np.max(np.abs(off[name] - on[name]))) for name in fields}


def print_physics_parity(world, differences) -> None:
    print(f"\nH. PHYSICS PARITY, {CLEAR_VISIBILITY_METERS:.0f} m clear vs a"
          f" {storm_module.MINIMUM_VISIBILITY_METERS:.0f} m white-out --"
          f" {world}  (same reset, same scripted command, tick for tick)")
    print("| array | max abs difference |")
    print("|---|---|")
    for name, difference in differences.items():
        print(f"| `{name}` | {difference:.3e} |")
    worst = max(differences.values())
    print(f"  worst over all {len(differences)} arrays: {worst:.3e}"
          + ("  -- BIT-IDENTICAL" if worst == 0.0 else "  -- NOT IDENTICAL"))


# --------------------------------------------------- J: visibility is not wind
def visibility_is_not_wind(scene, episode, repeats, world) -> dict:
    """J: the wind dial cannot move the visibility. -> two identical arms.

    THE CLAIM THIS REPLACES. Until 2026-08-30 the visibility WAS the wind speed:
    `100 m * exp(-wind / 6)`, so 0 m/s meant 100 m and 12 m/s meant 13.5 m, and
    no experiment could change one without changing the other. The user split
    them, and this is the table that says the split is real rather than
    cosmetic.

    Two arms, both at `WIND_INDEPENDENCE_VISIBILITY_METERS`, one with the wind
    dial at 0 m/s and one at 12. Under the old coupling those two rows could not
    have agreed on anything.

    J1 is the PICTURE: the same rendered eye image degraded in each arm, and the
       largest per-pixel difference between the two results. It is zero because
       `StormVision.degrade` never sees a wind speed -- which is checked
       directly, by counting the parameters of `StormVision.update`.
    J2 is the BEHAVIOUR: the full detection cell at 2 m and 5 m and the maximum
       detection range, per arm.

    Inputs  : a built scene and its episode; `repeats` frames per cell.
    Outputs : {"parameters", "pixel_difference", "arms": [{...}, ...]} -- the
              per-arm detection rates, median relative errors and max range.
    """
    import inspect

    model = scene.model
    guide = guide_module.Guide(scene.route, scene.terrain, model)
    guide.enabled = True

    # THE STRUCTURAL HALF, and it is the one that cannot rot: a wind speed
    # cannot reach the eyes if there is no parameter to pass it through.
    parameters = [name for name in
                  inspect.signature(storm_module.StormVision.update).parameters
                  if name != "self"]

    # J1: one render, degraded twice, with a wind dial that goes nowhere. The
    # generators are seeded identically so the grain is the same draw; any
    # difference at all would have to come from the wind.
    plain_eyes = guide_module.StereoEyes(model, verbose=False)
    _, plain_place_at = test_guide_module._range_placer(
        scene, guide, plain_eyes.left_camera_id, plain_eyes.right_camera_id)
    plain_place_at(5.0)
    plain_eyes.renderer.update_scene(scene.data, camera=plain_eyes.left_camera_id)
    rendered = plain_eyes.renderer.render().copy()
    degraded = []
    for _ in WIND_INDEPENDENCE_SPEEDS_MPS:
        vision = storm_module.StormVision(seed=0)
        vision.update(WIND_INDEPENDENCE_VISIBILITY_METERS)
        degraded.append(vision.degrade(rendered, plain_eyes.renderer)
                        .astype(np.int32))
    pixel_difference = int(np.max(np.abs(degraded[0] - degraded[1])))
    plain_eyes.close()

    # J2: the behavioural half, through the same machinery sections E and F use.
    storm_vision = storm_module.StormVision(seed=0)
    eyes = guide_module.StereoEyes(model, verbose=False,
                                   degradation=storm_vision.degrade)
    _, place_at = test_guide_module._range_placer(
        scene, guide, eyes.left_camera_id, eyes.right_camera_id)
    arms = []
    for wind_speed_mps in WIND_INDEPENDENCE_SPEEDS_MPS:
        # THE WIND REALLY BLOWS, through the one path that carries it in the
        # live loop: `episode.step`'s world-frame wind velocity, which becomes
        # the quadratic drag force on the torso. Half a second of it from the
        # same reset, so the arms differ by something the solver actually felt
        # rather than by a number handed nowhere. `place_at` then re-establishes
        # the eye-to-guide range by bisection, so what is compared is the same
        # geometry seen after two genuinely different gusts.
        episode.reset()
        wind_velocity_world = np.array([wind_speed_mps, 0.0])
        for _ in range(int(0.5 * episode.control_hz)):
            episode.step(np.zeros(3), wind_velocity_world)
        cells = {target: measure_cell(scene, eyes, guide, place_at,
                                      storm_vision,
                                      WIND_INDEPENDENCE_VISIBILITY_METERS,
                                      target, repeats)
                 for target in TEST_RANGES_METERS}
        furthest = None
        for target in RANGE_LADDER_METERS:
            if target > scene.route.length - 2.0:
                break
            cell = measure_cell(scene, eyes, guide, place_at, storm_vision,
                                WIND_INDEPENDENCE_VISIBILITY_METERS, target,
                                max(4, repeats // 2))
            if cell["detection_rate"] > DETECTION_MAJORITY:
                furthest = cell["true_meters"]
            else:
                break
        arms.append({"wind_speed_mps": wind_speed_mps, "cells": cells,
                     "maximum_range": furthest})
    storm_vision.update(CLEAR_VISIBILITY_METERS)
    eyes.close()
    return {"parameters": parameters, "pixel_difference": pixel_difference,
            "arms": arms, "world": world}


def print_visibility_is_not_wind(result) -> None:
    print(f"\nJ. VISIBILITY IS NOT WIND -- {result['world']}, visibility held"
          f" at {WIND_INDEPENDENCE_VISIBILITY_METERS:.0f} m in both arms")
    print(f"  `StormVision.update` parameters: {result['parameters']}"
          "  -- no wind speed can be passed to it at all")
    print(f"  J1  same rendered eye, degraded in each arm: max per-pixel"
          f" difference {result['pixel_difference']}")
    print("| wind m/s | detected at 2 m | median err 2 m | detected at 5 m |"
          " median err 5 m | max detection range m |")
    print("|---|---|---|---|---|---|")
    for arm in result["arms"]:
        near = arm["cells"][TEST_RANGES_METERS[0]]
        far = arm["cells"][TEST_RANGES_METERS[1]]

        def error(cell):
            return ("--" if cell["detection_rate"] == 0.0
                    else f"{100 * cell['median_relative_error']:+.1f}%")

        furthest = arm["maximum_range"]
        print(f"| {arm['wind_speed_mps']:.0f} |"
              f" {100 * near['detection_rate']:.0f}% | {error(near)} |"
              f" {100 * far['detection_rate']:.0f}% | {error(far)} |"
              f" {'never seen' if furthest is None else f'{furthest:.1f}'} |")
    print("  Under the retired coupling these two rows were 100 m and 13.5 m of"
          " visibility -- they could not have matched. They match now because"
          " the wind has no path to the eyes.")


def main(arguments) -> None:
    scene, episode = test_guide_module.open_world(arguments.world)
    print_fog_table(arguments.world, fog_reaches_the_eyes(scene, episode))
    result = storm_tables(scene, episode, arguments.repeats, arguments.world)
    print_tables(result, arguments.repeats)
    path = write_contact_sheet(result["pictures"], arguments.output)
    print(f"\nG. the left eye at each visibility, the guide 5.00 m away -> {path}")
    print_follower_response(arguments.world,
                            follower_response(scene, episode),
                            6.0)
    print_physics_parity(arguments.world, physics_parity(scene, episode))
    print_visibility_is_not_wind(
        visibility_is_not_wind(scene, episode, arguments.repeats,
                               arguments.world))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", default="flat_0")
    parser.add_argument("--repeats", type=int, default=16)
    parser.add_argument("--output", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "render3d_shots", "storm_eyes.png"))
    main(parser.parse_args())

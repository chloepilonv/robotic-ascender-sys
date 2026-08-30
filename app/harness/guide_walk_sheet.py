"""LOOK AT THE WALK: a contact sheet of one gait cycle, plus the skate audit.

    ../.venv_everest/bin/python -m app.harness.guide_walk_sheet

Two outputs, and the numbers matter as much as the picture:

  THE SHEET   `render3d_shots/guide_walk_sheet.png` -- eight frames evenly
              spaced across ONE stride (`GUIDE_STRIDE_METERS` of ground), from
              a camera off her left shoulder, so a human can check the four
              things a gait can get wrong: legs alternating, the left arm
              opposite its leg, boots on the snow rather than through it or
              above it, and the jacket visible (the detector's whole input).
  THE AUDIT   The gait is DISTANCE-LOCKED, which is a falsifiable claim: while a
              boot is planted its WORLD position must not move, however far the
              root travels. Sampled at 200 points around the cycle, the printed
              numbers are the planted boot's world drift per stance phase (skate,
              metres), its clearance above the surface while swinging, and how
              far the lowest corner of either boot sits from the snow.

Inputs  : a world name (default `flat_0`, the walking reference).
Outputs : the PNG, and the audit table on stdout.
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np

from app.harness import climb_worlds as climb_worlds_module
from app.harness import graphics as graphics_module
from app.harness import guide as guide_module

SHEET_COLUMNS, SHEET_ROWS = 4, 2
FRAME_WIDTH, FRAME_HEIGHT = 480, 640
CAMERA_DISTANCE_METERS = 4.2
CAMERA_ELEVATION_DEGREES = -8.0
# Off her left shoulder and slightly behind: a pure side view hides the arm
# swing behind the torso, and a pure rear view hides the stride.
CAMERA_AZIMUTH_OFFSET_DEGREES = 118.0
CAMERA_LOOKAT_HEIGHT_METERS = 0.95
AUDIT_SAMPLES = 200


def open_world(name):
    library = climb_worlds_module.ClimbSceneLibrary(verbose=False)
    scene, meta, definition = library.load(name)
    if not guide_module.attach_guide(scene, verbose=True):
        raise SystemExit(f"the guide could not be attached to {name!r}")
    graphics_module.add_skybox(scene, verbose=False)
    graphics_module.apply_alpine_look(
        scene.model, terrain_size_meters=scene.terrain.size_xy)
    return scene


def _pose(scene, guide, arclength_meters):
    """Put her at that arc length and run the kinematics. -> nothing."""
    import mujoco
    guide.arclength_meters = float(arclength_meters)
    guide.write(scene.model, scene.data)
    mujoco.mj_kinematics(scene.model, scene.data)
    mujoco.mj_camlight(scene.model, scene.data)


def audit(scene, guide, start_meters) -> dict:
    """The skate/clearance/ground numbers, sampled around one cycle."""
    import mujoco

    stride = guide_module.GUIDE_STRIDE_METERS
    boot_ids = {side: mujoco.mj_name2id(
        scene.model, mujoco.mjtObj.mjOBJ_GEOM, f"human_boot_{side}")
        for side in "lr"}
    samples = []
    for index in range(AUDIT_SAMPLES):
        travel = start_meters + stride * index / AUDIT_SAMPLES
        _pose(scene, guide, travel)
        row = {"travel_meters": travel,
               "phase_radians": guide.phase_radians() % (2.0 * math.pi)}
        for side, geom_id in boot_ids.items():
            centre = np.asarray(scene.data.geom_xpos[geom_id], dtype=float).copy()
            matrix = np.asarray(
                scene.data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
            half = np.asarray(scene.model.geom_size[geom_id][:3], dtype=float)
            corners = [centre + matrix @ (np.array([sx, sy, sz]) * half)
                       for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)
                       for sz in (-1.0, 1.0)]
            lowest = min(corners, key=lambda point: point[2])
            row[f"boot_{side}"] = centre
            row[f"sole_{side}"] = float(lowest[2])
            row[f"surface_{side}"] = float(
                scene.terrain.surface_z(float(lowest[0]), float(lowest[1])))
        samples.append(row)

    result = {}
    for side in "lr":
        # The planted half of THIS leg's cycle: the hip ramps the foot from
        # +stride/4 to -stride/4, which is `leg_phase < pi`. Left leg leads the
        # cycle; the right is half a cycle behind.
        #
        # THE STANCE PHASE WRAPS, and measuring across the wrap is how this
        # audit first "found" a 0.70 m skate that was not there: the sweep
        # starts at an arbitrary phase, so one leg's stance is split between the
        # start and the end of the sample list and its first and last planted
        # samples are a whole stride apart -- two different footfalls, not one
        # foot sliding. The skate is therefore the worst drift WITHIN a
        # contiguous run of planted samples.
        offset = 0.0 if side == "l" else math.pi
        planted_flags = [
            ((row["phase_radians"] + offset) % (2.0 * math.pi)) < math.pi
            for row in samples]
        runs, current = [], []
        for index, flag in enumerate(planted_flags):
            if flag:
                current.append(index)
            elif current:
                runs.append(current)
                current = []
        if current:
            runs.append(current)
        skate = 0.0
        spread = 0.0
        for run in runs:
            positions = np.array([samples[index][f"boot_{side}"] for index in run])
            skate = max(skate, float(np.linalg.norm(
                positions[-1][:2] - positions[0][:2])))
            spread = max(spread, float(np.ptp(positions[:, 0])))
        swinging = [samples[index] for index, flag in enumerate(planted_flags)
                    if not flag]
        result[side] = {
            "stance_skate_meters": skate,
            "stance_spread_meters": spread,
            "stance_runs": len(runs),
            "swing_clearance_meters": max(
                row[f"sole_{side}"] - row[f"surface_{side}"] for row in swinging),
            "sole_above_surface_min_meters": min(
                row[f"sole_{side}"] - row[f"surface_{side}"] for row in samples),
        }
    result["lowest_sole_gap_meters"] = min(
        min(row[f"sole_{side}"] - row[f"surface_{side}"] for side in "lr")
        for row in samples)
    result["highest_sole_gap_meters"] = max(
        min(row[f"sole_{side}"] - row[f"surface_{side}"] for side in "lr")
        for row in samples)
    return result


def render_sheet(scene, guide, start_meters, output_path) -> str:
    import mujoco
    from PIL import Image, ImageDraw, ImageFont

    frames = SHEET_COLUMNS * SHEET_ROWS
    stride = guide_module.GUIDE_STRIDE_METERS
    saved = (int(scene.model.vis.global_.offwidth),
             int(scene.model.vis.global_.offheight))
    scene.model.vis.global_.offwidth = max(saved[0], FRAME_WIDTH)
    scene.model.vis.global_.offheight = max(saved[1], FRAME_HEIGHT)
    try:
        renderer = mujoco.Renderer(scene.model, FRAME_HEIGHT, FRAME_WIDTH)
    finally:
        (scene.model.vis.global_.offwidth,
         scene.model.vis.global_.offheight) = saved
    graphics_module.apply_render_flags(renderer, shadows=True)

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = CAMERA_DISTANCE_METERS
    camera.elevation = CAMERA_ELEVATION_DEGREES

    tiles = []
    for index in range(frames):
        travel = start_meters + stride * index / frames
        _pose(scene, guide, travel)
        root = guide.root_world()
        camera.lookat[:] = (root[0], root[1], root[2] + CAMERA_LOOKAT_HEIGHT_METERS)
        camera.azimuth = (math.degrees(guide.yaw_radians())
                          + CAMERA_AZIMUTH_OFFSET_DEGREES)
        renderer.update_scene(scene.data, camera=camera)
        tiles.append(Image.fromarray(renderer.render().copy()))
    renderer.close()

    sheet = Image.new("RGB", (SHEET_COLUMNS * FRAME_WIDTH,
                             SHEET_ROWS * FRAME_HEIGHT), (12, 14, 18))
    try:
        font = ImageFont.load_default(size=22)
    except TypeError:
        font = ImageFont.load_default()
    for index, tile in enumerate(tiles):
        column, row = index % SHEET_COLUMNS, index // SHEET_COLUMNS
        sheet.paste(tile, (column * FRAME_WIDTH, row * FRAME_HEIGHT))
        draw = ImageDraw.Draw(sheet)
        angles = {}
        _pose(scene, guide, start_meters + stride * index / frames)
        angles = guide.limb_angles()
        text = (f"{index + 1}/{frames}  phase {360 * index / frames:.0f}deg  "
                f"travel {stride * index / frames:.2f} m\n"
                f"hip L{math.degrees(angles['hip_l']):+.0f} "
                f"R{math.degrees(angles['hip_r']):+.0f}  "
                f"knee L{math.degrees(angles['knee_l']):+.0f} "
                f"R{math.degrees(angles['knee_r']):+.0f}  "
                f"sh L{math.degrees(angles['shoulder_l']):+.0f}")
        draw.multiline_text((column * FRAME_WIDTH + 8, row * FRAME_HEIGHT + 6),
                            text, fill=(255, 255, 255), font=font,
                            stroke_width=3, stroke_fill=(0, 0, 0))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sheet.save(output_path)
    return output_path


def main(arguments) -> None:
    scene = open_world(arguments.world)
    guide = guide_module.Guide(scene.route, scene.terrain, scene.model)
    guide.enabled = True
    guide.motion_blend = 1.0            # the walk cycle, not the idle sway
    start, _ = scene.route.project_arclen(scene.data.qpos[0:3])
    start = float(start + arguments.lead_meters)

    numbers = audit(scene, guide, start)
    print(f"\nGAIT AUDIT -- {arguments.world}, stride"
          f" {guide_module.GUIDE_STRIDE_METERS:.2f} m,"
          f" speed {guide_module.GUIDE_SPEED_METERS_PER_SECOND:.2f} m/s"
          f" -> cadence"
          f" {120.0 * guide_module.GUIDE_SPEED_METERS_PER_SECOND / guide_module.GUIDE_STRIDE_METERS:.0f}"
          f" steps/min")
    print("| leg | stance runs | stance skate m | stance spread m |"
          " swing clearance m | min sole-above-snow m |")
    print("|---|---|---|---|---|---|")
    for side in "lr":
        row = numbers[side]
        print(f"| {'left' if side == 'l' else 'right'} |"
              f" {row['stance_runs']} |"
              f" {row['stance_skate_meters']:.4f} |"
              f" {row['stance_spread_meters']:.4f} |"
              f" {row['swing_clearance_meters']:.3f} |"
              f" {row['sole_above_surface_min_meters']:+.4f} |")
    print(f"  lowest boot corner vs snow over the cycle:"
          f" {numbers['lowest_sole_gap_meters']:+.4f} m to"
          f" {numbers['highest_sole_gap_meters']:+.4f} m"
          "  (0 = the sole touches; negative = through the snow)")

    path = render_sheet(scene, guide, start, arguments.output)
    print(f"\ncontact sheet -> {path}"
          f" ({SHEET_COLUMNS}x{SHEET_ROWS} of {FRAME_WIDTH}x{FRAME_HEIGHT})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", default="flat_0")
    parser.add_argument("--lead-meters", type=float, default=3.0)
    parser.add_argument("--output", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "render3d_shots", "guide_walk_sheet.png"))
    main(parser.parse_args())

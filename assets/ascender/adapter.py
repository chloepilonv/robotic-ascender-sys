#!/usr/bin/env python3
"""Ascender adapter for the G1 right wrist — parametric CAD (CadQuery) -> adapter.step + adapter.stl

Measured on Unitree's right_wrist_yaw_link.STL (menagerie; it is the 1.5 mm cosmetic shell of the wrist-yaw link):
  * end wall at x = 39.5..41.5 mm with a round opening Ø 38.95 mm centred at (y = -3.0, z = 0)
  * shell outline behind the wall (x 30..38) is D-shaped: y -30..+23.9, z -30.5..+30 (flat on +Y)
  * the hand's plug goes THROUGH the opening into a metal socket inside — not in any public file.
Design (wrist frame, mm; X toward the hand, Z up):
  PLUG      Ø 38.6 x 25 through the opening                            <- load path. Internal retention feature = TODO (needs the real robot)
  FACE PLATE 6 mm, outside the end wall, D-outline + 5 mm rim
  COLLAR    D-shaped sleeve over the shell (x 30..38), 0.3 clearance, split top/bottom, 4x M3 along Z
            (same axis as Unitree's stock clamp screws) -> anti-rotation + support only, the shell is plastic
  CRADLE    U-bracket under the ascender's bottom edge: 2 cheeks + Ø 12 clevis pin through the Petzl attachment hole. No drilling.
The ascender pose is read from assets/robots/g1_unitree_ascender.usd and pushed +TOOL_SHIFT_X so it clears the face plate.

Usage: python assets/ascender/adapter.py     -> adapter.step, adapter.stl, prints the new tool pose for attach_tool.py
Deps : pip install cadquery trimesh networkx usd-core
"""
import os
import numpy as np
import cadquery as cq
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
WRIST_STL = os.path.join(HERE, "..", "robots", "g1", "_menagerie", "unitree_g1", "assets", "right_wrist_yaw_link.STL")
ROBOT_USD = os.path.join(HERE, "..", "robots", "g1_unitree_ascender.usd")

# ---- measured wrist (mm) ----
WALL_X0, WALL_X1 = 39.5, 41.5
HOLE_D, HOLE_C = 38.95, (-3.0, 0.0)          # opening in the end wall, centre (y, z)
COLLAR_X0, COLLAR_X1 = 30.0, 38.0            # straight D-shaped part of the shell
CLEARANCE = 0.3
# ---- adapter ----
PLUG_D, PLUG_L = HOLE_D - 0.35, 25.0         # TODO: add the internal retention feature once measured on a real G1
PLATE_T, RIM = 6.0, 5.0
COLLAR_T, SPLIT_GAP, M3_CLEAR = 4.0, 1.0, 3.4
# ---- ascender interface (Petzl frame at its attachment hole; tune from the real part) ----
FRAME_T, PIN_D = 8.0, 12.0
HOLE_TOOL = np.array([0.0, 0.0, 12.0])       # attachment hole centre in the tool frame (mm)
CHEEK_T, CHEEK_W, BAR_T = 6.0, 26.0, 8.0
ARM_W = 12.0
TOOL_SHIFT_X = PLATE_T + 2.0                 # push the tool out so it clears the face plate


def shell_outline(x_mm=34.0):
    w = trimesh.load(WRIST_STL)
    loops = w.section(plane_origin=[x_mm / 1000, 0, 0], plane_normal=[1, 0, 0]).discrete
    outer = max(loops, key=lambda p: np.linalg.norm(p[:, 1:], axis=1).max())[:, 1:] * 1000
    return outer[::4]                                     # ~80 points is plenty


def offset_polygon(pts, d):
    """Outward offset of a convex-ish closed polygon by d (mm) along vertex normals."""
    c = pts.mean(0); n = pts - c; n /= np.linalg.norm(n, axis=1)[:, None]
    return [(float(y), float(z)) for y, z in pts + d * n]


def quat_to_R(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def tool_pose():
    from pxr import Usd, UsdGeom
    st = Usd.Stage.Open(ROBOT_USD)
    ops = {op.GetOpName(): op.Get() for op in UsdGeom.Xformable(
        st.GetPrimAtPath("/G1/right_wrist_yaw_link/tool_ascender")).GetOrderedXformOps()}
    p = np.array(ops["xformOp:translate"]) * 1000 + [TOOL_SHIFT_X, 0, 0]; q = ops["xformOp:orient"]
    return p, quat_to_R(q.GetReal(), *q.GetImaginary()), (q.GetReal(), *q.GetImaginary())


def placed(shape, R, p):
    """Apply a rotation matrix + translation (mm) to a CadQuery Workplane (axis-angle: OCC wants an exact rotation)."""
    ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    if ang > 1e-6:
        axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        if np.linalg.norm(axis) < 1e-9:                      # 180 deg: axis from the symmetric part
            axis = np.sqrt(np.maximum(np.diag(R) + 1, 0) / 2)
        axis = axis / np.linalg.norm(axis)
        shape = shape.rotate((0, 0, 0), tuple(axis), ang)
    return shape.translate(tuple(p))


def build():
    outline = shell_outline()
    inner = offset_polygon(outline, CLEARANCE); outer = offset_polygon(outline, CLEARANCE + COLLAR_T)
    yz = lambda off: cq.Workplane("YZ").workplane(offset=off)     # sketch (y, z), extrude along +x
    collar = yz(COLLAR_X0).polyline(outer).close().extrude(WALL_X1 - COLLAR_X0).cut(
        yz(COLLAR_X0 - 1).polyline(inner).close().extrude(WALL_X1 - COLLAR_X0 + 2))
    plate = yz(WALL_X1).polyline(outer).close().extrude(PLATE_T)
    plug = yz(WALL_X1 - PLUG_L).center(*HOLE_C).circle(PLUG_D / 2).extrude(PLUG_L + 0.5)
    body = collar.union(plate).union(plug)
    # 4x M3 clamp bolts along Z through the collar walls (2 per side)
    for y in (min(y for y, _ in outer) + COLLAR_T / 2 + 0.3, max(y for y, _ in outer) - COLLAR_T / 2 - 0.3):
        for x in (COLLAR_X0 + 3.5, COLLAR_X1 - 1.5):
            body = body.cut(cq.Workplane("XY").center(x, y).circle(M3_CLEAR / 2).extrude(100, both=True))
    # split the collar into top/bottom halves (gap so the bolts clamp); plate and plug stay whole
    body = body.cut(cq.Workplane("XY").workplane(offset=-SPLIT_GAP / 2).center((COLLAR_X0 + WALL_X1) / 2, 0)
                    .rect(WALL_X1 - COLLAR_X0 + 0.01, 200).extrude(SPLIT_GAP))
    # cradle in the tool frame: bar under the frame bottom edge + 2 cheeks with the clevis bore
    bar = cq.Workplane("XY").box(CHEEK_W, FRAME_T + 2 * CHEEK_T, BAR_T, centered=(True, True, False)).translate((0, 0, -BAR_T))
    cradle = bar
    for side in (-1, 1):
        cheek = (cq.Workplane("XY").box(CHEEK_W, CHEEK_T, CHEEK_W, centered=(True, True, False))
                 .translate((0, side * (FRAME_T + CHEEK_T) / 2, 0)))
        cradle = cradle.union(cheek)
    cradle = cradle.cut(cq.Workplane("XZ").center(HOLE_TOOL[0], HOLE_TOOL[2]).circle(PIN_D / 2 + 0.1).extrude(50, both=True))
    p, R, q = tool_pose()
    cradle_w = placed(cradle, R, p + R @ (HOLE_TOOL - [0, 0, HOLE_TOOL[2]]))   # cradle origin = frame bottom under the hole
    # arm: from the bottom of the face plate to the cradle bar, along the straight line between them
    a = np.array([WALL_X1 + PLATE_T / 2, HOLE_C[0], min(z for _, z in outer) + RIM / 2])
    b = p + R @ np.array([0, 0, -BAR_T / 2]); d = b - a; L = np.linalg.norm(d)
    arm = cq.Workplane("XY").box(ARM_W, ARM_W, L + ARM_W, centered=(True, True, False)).translate((0, 0, -ARM_W / 2))
    ez = d / L; ex = np.cross([0, 1, 0], ez); ex /= np.linalg.norm(ex); ey = np.cross(ez, ex)
    arm_w = placed(arm, np.c_[ex, ey, ez], a)
    body = body.union(arm_w).union(cradle_w)
    return body, p, q


if __name__ == "__main__":
    body, p, q = build()
    cq.exporters.export(body, os.path.join(HERE, "adapter.step"))
    cq.exporters.export(body, os.path.join(HERE, "adapter.stl"), tolerance=0.05)
    vol = body.val().Volume() / 1000
    print(f"adapter.step/.stl: {vol:.1f} cm3 -> {vol*1.02:.0f} g PA12-CF / {vol*2.7:.0f} g 6061 Al")
    print(f"new tool pose in the wrist frame (for attach_tool.py / mujoco build): pos = ({p[0]/1000:.5f}, {p[1]/1000:.5f}, {p[2]/1000:.5f}) m, quat wxyz = {np.round(q, 5).tolist()}")

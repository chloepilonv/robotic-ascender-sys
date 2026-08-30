#!/usr/bin/env python3
"""Ascender EOAT v1 — "if Petzl designed it for the G1": ONE frame that is both the wrist plug and the ascender body.
Keeps the Petzl Basic mechanism (cam, axle, spring, safety catch = OEM spares); replaces the hand-shaped orange frame.

Wrist frame (mm): X toward the hand, Z up. Rope channel is parallel to X (the rope runs along the forearm, like a human
arm on a fixed rope) and sits BELOW the wrist shell — the rope cannot pass through the wrist, so an offset is unavoidable.
Offset torque on the wrist: F_rope x CHANNEL_Z (printed). Above ~5 N.m the wrist motors cannot hold it -> the frame gets a
FOREARM BRACE (fork bearing on the wrist-pitch shell) so the load bypasses the wrist joints.

Outputs: eoat.step, eoat.stl (frame), cam.stl (placeholder cam for renders), prints the sim mount pose.
Usage: python assets/ascender/eoat.py         Deps: cadquery trimesh networkx
"""
import os
import numpy as np
import cadquery as cq
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
WRIST_STL = os.path.join(HERE, "..", "robots", "g1", "_menagerie", "unitree_g1", "assets", "right_wrist_yaw_link.STL")

# ---- measured wrist (mm) ----
WALL_X1 = 41.5
HOLE_D, HOLE_C = 38.95, (-3.0, 0.0)
SHELL_Z_MIN = -30.5
# ---- plug + flange ----
PLUG_D, PLUG_L = HOLE_D - 0.35, 25.0        # internal retention feature: TODO (real G1 / Unitree drawing)
FLANGE_T = 8.0
# ---- rope channel (Petzl Basic numbers: rope 8-13 mm) ----
ROPE_D_MAX = 13.0
CHANNEL_W = 16.0                             # channel width (y)
CHANNEL_Z = SHELL_Z_MIN - 4.0 - 14.0        # channel centreline below the shell: shell bottom - clearance - half body
BODY_X0, BODY_L = WALL_X1 + FLANGE_T, 95.0   # frame body along X
BODY_Y0, BODY_W = HOLE_C[0] - 14.0, 28.0     # body width (y): channel + one side wall (open side like the Petzl: +Y)
BODY_Z_H = 34.0                              # body height (z), centred on CHANNEL_Z
WALL_T = 6.0
# ---- Petzl Basic cam (OEM): eccentric toothed cam on a Ø8 axle, spring-loaded toward the channel ----
CAM_AXLE_D, CAM_R_MIN, CAM_R_MAX = 8.0, 12.0, 21.0
CAM_X = BODY_X0 + 55.0                       # axle position along the channel
CAM_Z_OFF = 22.0                             # axle above the channel centreline (cam swings down onto the rope)
CAM_T = 12.0                                 # cam thickness (y)
# ---- sensing pockets ----
HALL_POCKET = (6.0, 4.0)                     # Ø, depth at the cam rest position
PUCK_POCKET = (24.0, 14.0, 10.0)             # ESP32 + HX711 pocket in the flange (x, y, z)
FLEXURE_SLOT = 2.0                           # thin web between plug and body -> strain gauges = load cell
F_ROPE = 350.0                               # N, robot weight on the rope


def shell_outline(x_mm=34.0):
    w = trimesh.load(WRIST_STL)
    loops = w.section(plane_origin=[x_mm / 1000, 0, 0], plane_normal=[1, 0, 0]).discrete
    outer = max(loops, key=lambda p: np.linalg.norm(p[:, 1:], axis=1).max())[:, 1:] * 1000
    c = outer.mean(0); n = outer - c; n /= np.linalg.norm(n, axis=1)[:, None]
    return [(float(y), float(z)) for y, z in (outer + 5.0 * n)[::4]]


def build():
    yz = lambda off: cq.Workplane("YZ").workplane(offset=off)
    plug = yz(WALL_X1 - PLUG_L).center(*HOLE_C).circle(PLUG_D / 2).extrude(PLUG_L + 0.5)
    flange = yz(WALL_X1).polyline(shell_outline()).close().extrude(FLANGE_T)
    # neck from the flange down to the body (carries the load; the flexure slot makes it a load cell)
    neck = (cq.Workplane("XY").box(FLANGE_T + 14, BODY_W, abs(CHANNEL_Z) + BODY_Z_H / 2 + 2, centered=(False, True, False))
            .translate((WALL_X1, HOLE_C[0], CHANNEL_Z - BODY_Z_H / 2)))
    body = (cq.Workplane("XY").box(BODY_L, BODY_W, BODY_Z_H, centered=(False, False, True))
            .translate((BODY_X0, BODY_Y0, CHANNEL_Z)).edges("|X").fillet(4.0))
    frame = plug.union(flange).union(neck).union(body)
    # rope channel: open toward +Y (rope goes in sideways, the cam keeps it in), full length along X
    channel = (cq.Workplane("XY").box(BODY_L + 2, CHANNEL_W + 20, ROPE_D_MAX + 3, centered=(False, False, True))
               .translate((BODY_X0 - 1, BODY_Y0 + WALL_T, CHANNEL_Z)))
    frame = frame.cut(channel)
    # cam pocket + axle bore (axle along Y through the side wall)
    pocket = (cq.Workplane("XY").box(2 * CAM_R_MAX + 6, CAM_T + 2, CAM_R_MAX + CAM_Z_OFF + 8, centered=(True, False, False))
              .translate((CAM_X, BODY_Y0 + WALL_T, CHANNEL_Z - 2)))
    frame = frame.cut(pocket)
    frame = frame.cut(cq.Workplane("XZ").center(CAM_X, CHANNEL_Z + CAM_Z_OFF).circle(CAM_AXLE_D / 2 + 0.05).extrude(60, both=True))
    # flexure slot (load-cell web) in the neck, Hall pocket, electronics pocket in the flange
    frame = frame.cut(cq.Workplane("XY").box(FLEXURE_SLOT, BODY_W + 2, 10, centered=(True, True, True))
                      .translate((WALL_X1 + FLANGE_T + 6, HOLE_C[0], CHANNEL_Z + BODY_Z_H / 2 + 8)))
    frame = frame.cut(cq.Workplane("XZ").center(CAM_X + CAM_R_MAX + 2, CHANNEL_Z + CAM_Z_OFF + 4)
                      .circle(HALL_POCKET[0] / 2).extrude(HALL_POCKET[1]).translate((0, BODY_Y0 + WALL_T, 0)))
    frame = frame.cut(cq.Workplane("XY").box(*PUCK_POCKET, centered=(False, True, True))
                      .translate((WALL_X1 + 1, HOLE_C[0], 12)))
    # placeholder cam (eccentric disc) for renders / clearance
    cam = (cq.Workplane("XZ").center(0, 0).circle(CAM_R_MAX).extrude(CAM_T / 2, both=True)
           .cut(cq.Workplane("XZ").circle(CAM_AXLE_D / 2).extrude(20, both=True))
           .translate((CAM_X - (CAM_R_MAX - CAM_R_MIN) / 2, BODY_Y0 + WALL_T + CAM_T / 2 + 1, CHANNEL_Z + CAM_Z_OFF)))
    return frame, cam


if __name__ == "__main__":
    frame, cam = build()
    cq.exporters.export(frame, os.path.join(HERE, "eoat.step"))
    cq.exporters.export(frame, os.path.join(HERE, "eoat.stl"), tolerance=0.05)
    cq.exporters.export(cam, os.path.join(HERE, "cam.stl"), tolerance=0.05)
    vol = frame.val().Volume() / 1000
    tq = F_ROPE * abs(CHANNEL_Z) / 1000
    print(f"eoat.step/.stl: {vol:.1f} cm3 -> {vol*2.81:.0f} g in 7075-T6 ({vol*1.02:.0f} g PA12-CF for the fit-check print)")
    print(f"rope channel {abs(CHANNEL_Z):.0f} mm below the wrist axis -> {tq:.1f} N.m on the wrist at {F_ROPE:.0f} N "
          f"(wrist motors: 5 N.m) -> {'needs a forearm brace' if tq > 5 else 'ok'}")
    print("sim: the frame IS the tool; mount pose = identity on right_wrist_yaw_link (mesh already in the wrist frame)")

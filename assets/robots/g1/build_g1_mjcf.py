#!/usr/bin/env python3
"""Build the Himalaya G1 as MJCF for MuJoCo — same robot as g1_himalaya.usd, same source tables.

Output: assets/robots/g1_himalaya.xml            (jacket, boots, boot friction, D435i, Mid-360)
        assets/robots/g1_himalaya_ascender.xml   (+ ascender on the right wrist, rubber hand removed)
        assets/robots/g1/mjcf_meshes/*.obj       (generated gear + ascender meshes)
Stock link meshes stay in g1/_menagerie (git-ignored, auto-cloned by fetch_menagerie()).

Usage:  python assets/robots/g1/build_g1_mjcf.py
Deps :  pip install mujoco trimesh usd-core   (usd-core only to read the ascender mesh)
"""
import os, sys, warnings
import xml.etree.ElementTree as ET
import numpy as np
import mujoco, trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_g1_usd import (GEAR, GEAR_PRIMS, BOOT_FRICTION, JACKET_BLUE, BOOT_YELLOW, BOOT_TRIM,   # noqa: E402
                          fetch_menagerie, inflated_hull, mesh_arrays)

OUT_DIR = os.path.join(HERE, "..")
MESH_DIR = os.path.join(HERE, "mjcf_meshes")
# sensor poses: torso frame = head_sensors offset (0.0039635, 0, -0.044) + local, exactly as build_g1_usd.py
HEAD = np.array([0.0039635, 0.0, -0.044])
D435I_POS, D435I_XYAXES, D435I_FOVY = HEAD + [0.075, 0, 0.43], "0 -1 0  0 0 1", 58   # looks +X (cam -Z -> +X, cam +Y -> +Z)
MID360_POS = HEAD + [0.0, 0.0, 0.50]
# ascender: mesh from ascender.usd, mount pose read from the built USD (attach_tool.py output) = single source of truth
TOOL_LINK = "right_wrist_yaw_link"
ASCENDER_USD = os.path.join(HERE, "..", "..", "ascender", "ascender.usd")
ROBOT_TOOL_USD = os.path.join(HERE, "..", "g1_unitree_ascender.usd")


def rgba(c): return f"{c[0]} {c[1]} {c[2]} 1"


def write_obj(name, v, f):
    os.makedirs(MESH_DIR, exist_ok=True)
    trimesh.Trimesh(v, f, process=False).export(os.path.join(MESH_DIR, name + ".obj"))
    return name


def gear_meshes(model):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    """link name -> obj mesh name of its inflated jacket/boot hull (visual only)."""
    out = {}
    for b in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
        if name not in GEAR:
            continue
        color, off, gname = GEAR[name]
        vs, fs = [], []
        for g in range(model.body_geomadr[b], model.body_geomadr[b] + model.body_geomnum[b]):
            is_col = model.geom_contype[g] != 0 or model.geom_conaffinity[g] != 0
            if is_col or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            v, f = mesh_arrays(model, model.geom_dataid[g])
            R = np.zeros(9); mujoco.mju_quat2Mat(R, model.geom_quat[g])
            vs.append(v.astype(np.float64) @ R.reshape(3, 3).T + model.geom_pos[g]); fs.append(f)
        if not vs:
            continue
        v = np.concatenate(vs); f = np.concatenate([lf + sum(len(x) for x in vs[:i]) for i, lf in enumerate(fs)])
        hv, hf = inflated_hull(v, f, off)
        if name == "torso_link":
            hv[:, 2] = np.minimum(hv[:, 2], 0.30)   # collar stops below the head
        out[name] = (write_obj(f"{name}_{gname}", hv, hf), color)
    return out


def ascender_meshes():
    from pxr import Usd, UsdGeom, UsdPhysics
    st = Usd.Stage.Open(ASCENDER_USD)
    def tri(prim):
        m = UsdGeom.Mesh(prim); pts = np.array(m.GetPointsAttr().Get()); cnt = m.GetFaceVertexCountsAttr().Get(); idx = m.GetFaceVertexIndicesAttr().Get()
        faces, k = [], 0
        for n in cnt:
            poly = idx[k:k + n]; k += n
            faces += [[poly[0], poly[i], poly[i + 1]] for i in range(1, n - 1)]
        return pts, np.array(faces)
    # visual: the 25k-vert collision mesh (the textured 2M-face visual is useless in MuJoCo, no PBR);
    # collision: convex hull of a 2k-vertex subsample (MuJoCo re-hulls anyway; keeps the file small)
    cv, cf = tri(st.GetPrimAtPath("/Ascender/collision"))
    sub = cv[np.random.default_rng(0).choice(len(cv), 2000, replace=False)]
    hull = trimesh.PointCloud(sub).convex_hull
    m = UsdPhysics.MassAPI(st.GetPrimAtPath("/Ascender"))
    robot = Usd.Stage.Open(ROBOT_TOOL_USD)   # keep the stage alive while reading the prim
    ops = {op.GetOpName(): op.Get() for op in
           UsdGeom.Xformable(robot.GetPrimAtPath(f"/G1/{TOOL_LINK}/tool_ascender")).GetOrderedXformOps()}
    pos = np.array(ops["xformOp:translate"], dtype=float); q = ops["xformOp:orient"]
    quat = np.array([q.GetReal(), *q.GetImaginary()], dtype=float)          # w x y z, same convention as MuJoCo
    return (write_obj("ascender_visual", cv, cf), write_obj("ascender_collision", hull.vertices, hull.faces),
            float(m.GetMassAttr().Get()), np.array(m.GetCenterOfMassAttr().Get(), dtype=float), pos, quat)


def build(xml, with_tool):
    model = mujoco.MjModel.from_xml_path(xml)
    tree = ET.parse(xml); root = tree.getroot()
    root.set("model", "g1_himalaya" + ("_ascender" if with_tool else ""))
    # mesh paths: output xml lives in assets/robots/, stock STLs in g1/_menagerie/..., generated OBJs in g1/mjcf_meshes/
    comp = root.find("compiler"); comp.set("meshdir", "g1")
    asset = root.find("asset")
    for m in asset.findall("mesh"):
        m.set("file", "_menagerie/unitree_g1/assets/" + m.get("file"))
    for n, c in (("jacket", JACKET_BLUE), ("boot", BOOT_YELLOW), ("boot_trim", BOOT_TRIM), ("ascender", (0.85, 0.55, 0.05))):
        ET.SubElement(asset, "material", name=n, rgba=rgba(c))
    bodies = {b.get("name"): b for b in root.iter("body")}

    # gear shells (visual, no collision, no mass — class="visual" has contype=0 density=0)
    for link, (mesh, color) in gear_meshes(model).items():
        ET.SubElement(asset, "mesh", name=mesh, file="mjcf_meshes/" + mesh + ".obj")
        mat = "jacket" if color == JACKET_BLUE else "boot" if color == BOOT_YELLOW else "boot_trim"
        ET.SubElement(bodies[link], "geom", {"class": "visual", "mesh": mesh, "material": mat})
    for link, kind, pos, size, color in GEAR_PRIMS:
        mat = "boot" if color == BOOT_YELLOW else "boot_trim"
        ET.SubElement(bodies[link], "geom", {"class": "visual", "type": "cylinder", "pos": "%g %g %g" % pos,
                                             "size": "%g %g" % size, "material": mat})
    # boots: friction under the feet
    for foot in root.iter("geom"):
        if foot.get("class") == "foot":
            foot.set("friction", str(BOOT_FRICTION))
    # sensors on the torso: D435i camera, Mid-360 lidar site; base orientation sensor for the monitor
    torso = bodies["torso_link"]
    ET.SubElement(torso, "camera", name="d435i", pos="%g %g %g" % tuple(D435I_POS), xyaxes=D435I_XYAXES, fovy=str(D435I_FOVY))
    ET.SubElement(torso, "site", name="mid360", pos="%g %g %g" % tuple(MID360_POS), size="0.02", rgba="0 1 0 1")
    sensor = root.find("sensor")
    ET.SubElement(sensor, "framequat", name="imu-pelvis-quat", objtype="site", objname="imu_in_pelvis")

    if with_tool:
        wrist = bodies[TOOL_LINK]
        for g in list(wrist.findall("geom")):            # drop the rubber hand: the tool replaces it
            if "hand" in (g.get("mesh") or ""):
                wrist.remove(g)
        vis, col, mt, ct, tpos, tquat = ascender_meshes()
        TOOL_POS, TOOL_QUAT = "%.6g %.6g %.6g" % tuple(tpos), "%.6g %.6g %.6g %.6g" % tuple(tquat)
        for n in (vis, col):
            ET.SubElement(asset, "mesh", name=n, file="mjcf_meshes/" + n + ".obj")
        ET.SubElement(wrist, "geom", {"class": "visual", "mesh": vis, "material": "ascender", "pos": TOOL_POS, "quat": TOOL_QUAT})
        ET.SubElement(wrist, "geom", {"class": "collision", "mesh": col, "pos": TOOL_POS, "quat": TOOL_QUAT})
        # fold the tool mass into the explicit <inertial> (geom mass is ignored when <inertial> is present), as attach_tool.py
        inert = wrist.find("inertial"); m0 = float(inert.get("mass")); c0 = np.array(inert.get("pos").split(), dtype=float)
        ct_w = np.zeros(3); mujoco.mju_rotVecQuat(ct_w, ct, tquat); ct_w += tpos          # tool COM -> wrist frame
        c = (c0 * m0 + ct_w * mt) / (m0 + mt); r = ct_w - c
        I = np.array(inert.get("diaginertia").split(), dtype=float) + mt * (r.dot(r) - r * r)
        inert.set("mass", "%.6g" % (m0 + mt)); inert.set("pos", "%.6g %.6g %.6g" % tuple(c)); inert.set("diaginertia", "%.6g %.6g %.6g" % tuple(I))

    out = os.path.join(OUT_DIR, "g1_himalaya" + ("_ascender" if with_tool else "") + ".xml")
    ET.indent(tree, space="  "); tree.write(out, encoding="unicode")
    m = mujoco.MjModel.from_xml_path(out)   # compile check
    print(f"wrote {os.path.relpath(out)}: {m.nu} actuators, {m.nbody - 1} bodies, mass {sum(m.body_mass):.3f} kg, "
          f"{TOOL_LINK} {m.body_mass[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, TOOL_LINK)]:.3f} kg")
    return out


if __name__ == "__main__":
    xml = fetch_menagerie()
    build(xml, with_tool=False)
    build(xml, with_tool=True)

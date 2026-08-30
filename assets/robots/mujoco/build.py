#!/usr/bin/env python3
"""Build the Himalaya G1 as MJCF for MuJoCo — same robot as the USD, same source tables (see README.md).

Usage:  python assets/robots/mujoco/build.py            # rebuild g1_unitree*.xml, ascender.xml, meshes/
        python assets/robots/mujoco/build.py --fetch    # only fetch the stock Unitree STLs (needed once per clone)
Deps :  pip install mujoco trimesh usd-core pillow   (usd-core only to read the ascender USD)
"""
import os, sys, warnings
import xml.etree.ElementTree as ET
import numpy as np
import mujoco, trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "g1"))
from build_g1_usd import (GEAR, GEAR_PRIMS, BOOT_FRICTION, JACKET_BLUE, BOOT_YELLOW, BOOT_TRIM,   # noqa: E402
                          fetch_menagerie, inflated_hull, mesh_arrays)

OUT_DIR = HERE
MESH_DIR = os.path.join(OUT_DIR, "meshes")
# sensor poses: torso frame = head_sensors offset (0.0039635, 0, -0.044) + local, exactly as build_g1_usd.py
HEAD = np.array([0.0039635, 0.0, -0.044])
D435I_POS, D435I_XYAXES, D435I_FOVY = HEAD + [0.075, 0, 0.43], "0 -1 0  0 0 1", 58   # looks +X (cam -Z -> +X, cam +Y -> +Z)
MID360_POS = HEAD + [0.0, 0.0, 0.50]
# ascender: mesh from ascender.usd, mount pose read from the built USD (attach_tool.py output) = single source of truth
TOOL_LINK = "right_wrist_yaw_link"
ASCENDER_USD = os.path.join(HERE, "..", "..", "ascender", "ascender.usd")
ROBOT_TOOL_USD = os.path.join(HERE, "..", "g1_unitree_ascender.usd")
ASCENDER_TEX = os.path.join(HERE, "..", "..", "ascender", "textures", "orange_metal_pulley_3d_model_basecolor.JPEG")
LOGO_PNG = os.path.join(HERE, "..", "g1", "textures", "everest_logo.png")
LOGOS = {"logo_back": ((0.0, 0.17), 0.17, -1), "logo_chest_right": ((-0.055, 0.235), 0.06, +1)}   # (center_yz, width, side) as build_g1_usd


def rgba(c): return f"{c[0]} {c[1]} {c[2]} 1"


def write_obj(name, v, f):
    os.makedirs(MESH_DIR, exist_ok=True)
    trimesh.Trimesh(v, f, process=False).export(os.path.join(MESH_DIR, name + ".obj"))
    return name


def write_obj_uv(name, v, f, uv):
    """OBJ with per-vertex texture coords (MuJoCo reads v/vt/f)."""
    os.makedirs(MESH_DIR, exist_ok=True)
    with open(os.path.join(MESH_DIR, name + ".obj"), "w") as fh:
        fh.write("".join("v %.6f %.6f %.6f\n" % tuple(p) for p in v))
        fh.write("".join("vt %.6f %.6f\n" % tuple(t) for t in uv))
        fh.write("".join("f %d/%d %d/%d %d/%d\n" % (a+1, a+1, b+1, b+1, c+1, c+1) for a, b, c in f))
    return name


def decimate_uv(v, f, uv_wedge, voxel):
    """Vertex-clustering decimation on a voxel grid; keeps one UV per cluster (scan atlas -> minor seam artefacts)."""
    key = np.floor(v / voxel).astype(np.int64)
    _, first, cluster = np.unique(key, axis=0, return_index=True, return_inverse=True)
    cluster = cluster.ravel()
    nv = np.zeros((len(first), 3)); cnt = np.zeros(len(first))
    np.add.at(nv, cluster, v); np.add.at(cnt, cluster, 1); nv /= cnt[:, None]
    wedge_cluster = cluster[f.ravel()]                # first wedge UV seen for each cluster
    cids, widx = np.unique(wedge_cluster, return_index=True)
    uv = np.zeros((len(first), 2)); uv[cids] = uv_wedge[widx]
    nf = cluster[f]
    nf = nf[(nf[:, 0] != nf[:, 1]) & (nf[:, 1] != nf[:, 2]) & (nf[:, 0] != nf[:, 2])]
    return nv, nf, uv


def logo_patch(hull_v, hull_f, center_yz, width, side, n=(10, 7)):
    """Same geometry as build_g1_usd.logo_patch: textured quad shrink-wrapped on the jacket hull along +/-X, 4 mm proud."""
    hull = trimesh.Trimesh(hull_v, hull_f)
    ys = np.linspace(center_yz[0] - width / 2, center_yz[0] + width / 2, n[0])
    zs = np.linspace(center_yz[1] - width * 0.69 / 2, center_yz[1] + width * 0.69 / 2, n[1])
    pts, uvs = [], []
    for j, z in enumerate(zs):
        for i, y in enumerate(ys):
            loc, _, _ = hull.ray.intersects_location([[side * 1.0, y, z]], [[-side, 0, 0]])
            x = (loc[:, 0].max() if side > 0 else loc[:, 0].min()) if len(loc) else side * 0.09
            pts.append([x + side * 0.004, y, z]); uvs.append([(i / (n[0] - 1)) if side > 0 else 1 - i / (n[0] - 1), j / (n[1] - 1)])
    faces = []
    for j in range(n[1] - 1):
        for i in range(n[0] - 1):
            a = j * n[0] + i; b = a + 1; c = a + n[0]; d = c + 1
            faces += [[a, b, d], [a, d, c]] if side > 0 else [[a, d, b], [a, c, d]]
    return np.array(pts), np.array(faces), np.array(uvs)


def png_texture(src, name, max_size=2048):
    from PIL import Image
    im = Image.open(src).convert("RGB"); im.thumbnail((max_size, max_size))
    os.makedirs(MESH_DIR, exist_ok=True); im.save(os.path.join(MESH_DIR, name + ".png"))
    return "meshes/" + name + ".png"


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
        if name == "torso_link":
            out["_torso_hull"] = (hv, hf)
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
    # visual: the scanned mesh (972k verts) voxel-decimated to ~1 mm with its basecolor UVs;
    # collision: convex hull of a 2k-vertex subsample (MuJoCo re-hulls anyway; keeps the file small)
    vis_prim = st.GetPrimAtPath("/Ascender/visual/mesh")
    vv, vf = tri(vis_prim)
    st_uv = np.array(UsdGeom.PrimvarsAPI(vis_prim).GetPrimvar("st").Get(), dtype=float)
    dv, df, duv = decimate_uv(vv, vf, st_uv, voxel=0.0012)
    cv, cf = tri(st.GetPrimAtPath("/Ascender/collision"))
    sub = cv[np.random.default_rng(0).choice(len(cv), 2000, replace=False)]
    hull = trimesh.PointCloud(sub).convex_hull
    m = UsdPhysics.MassAPI(st.GetPrimAtPath("/Ascender"))
    robot = Usd.Stage.Open(ROBOT_TOOL_USD)   # keep the stage alive while reading the prim
    ops = {op.GetOpName(): op.Get() for op in
           UsdGeom.Xformable(robot.GetPrimAtPath(f"/G1/{TOOL_LINK}/tool_ascender")).GetOrderedXformOps()}
    pos = np.array(ops["xformOp:translate"], dtype=float); q = ops["xformOp:orient"]
    quat = np.array([q.GetReal(), *q.GetImaginary()], dtype=float)          # w x y z, same convention as MuJoCo
    return (write_obj_uv("ascender_visual", dv, df, duv), write_obj("ascender_collision", hull.vertices, hull.faces),
            float(m.GetMassAttr().Get()), np.array(m.GetCenterOfMassAttr().Get(), dtype=float), pos, quat)


def build(xml, with_tool):
    model = mujoco.MjModel.from_xml_path(xml)
    tree = ET.parse(xml); root = tree.getroot()
    root.set("model", "g1_unitree" + ("_ascender" if with_tool else ""))
    # paths relative to assets/robots/mujoco/: stock STLs in ../g1/_menagerie/..., generated OBJ/PNG in meshes/
    comp = root.find("compiler"); comp.set("meshdir", "."); comp.set("texturedir", ".")
    asset = root.find("asset")
    for m in asset.findall("mesh"):
        m.set("file", "../g1/_menagerie/unitree_g1/assets/" + m.get("file"))
    for n, c in (("jacket", JACKET_BLUE), ("boot", BOOT_YELLOW), ("boot_trim", BOOT_TRIM)):
        ET.SubElement(asset, "material", name=n, rgba=rgba(c))
    ET.SubElement(asset, "texture", name="everest_logo", type="2d", file=png_texture(LOGO_PNG, "everest_logo", 1024))
    ET.SubElement(asset, "material", name="logo", texture="everest_logo", specular="0.1")
    ET.SubElement(asset, "texture", name="ascender_basecolor", type="2d", file=png_texture(ASCENDER_TEX, "ascender_basecolor"))
    ET.SubElement(asset, "material", name="ascender", texture="ascender_basecolor", specular="0.6", shininess="0.5")
    bodies = {b.get("name"): b for b in root.iter("body")}

    # gear shells (visual, no collision, no mass — class="visual" has contype=0 density=0)
    gear = gear_meshes(model); hv, hf = gear.pop("_torso_hull")
    for name, (cyz, width, side) in LOGOS.items():   # sponsor patches like a real jacket
        pv, pf, puv = logo_patch(hv, hf, cyz, width, side)
        ET.SubElement(asset, "mesh", name=name, file="meshes/" + write_obj_uv(name, pv, pf, puv) + ".obj")
        ET.SubElement(bodies["torso_link"], "geom", {"class": "visual", "mesh": name, "material": "logo"})
    for link, (mesh, color) in gear.items():
        ET.SubElement(asset, "mesh", name=mesh, file="meshes/" + mesh + ".obj")
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
            ET.SubElement(asset, "mesh", name=n, file="meshes/" + n + ".obj")
        ET.SubElement(wrist, "geom", {"class": "visual", "mesh": vis, "material": "ascender", "pos": TOOL_POS, "quat": TOOL_QUAT})
        ET.SubElement(wrist, "geom", {"class": "collision", "mesh": col, "pos": TOOL_POS, "quat": TOOL_QUAT})
        # fold the tool mass into the explicit <inertial> (geom mass is ignored when <inertial> is present), as attach_tool.py
        inert = wrist.find("inertial"); m0 = float(inert.get("mass")); c0 = np.array(inert.get("pos").split(), dtype=float)
        ct_w = np.zeros(3); mujoco.mju_rotVecQuat(ct_w, ct, tquat); ct_w += tpos          # tool COM -> wrist frame
        c = (c0 * m0 + ct_w * mt) / (m0 + mt); r = ct_w - c
        I = np.array(inert.get("diaginertia").split(), dtype=float) + mt * (r.dot(r) - r * r)
        inert.set("mass", "%.6g" % (m0 + mt)); inert.set("pos", "%.6g %.6g %.6g" % tuple(c)); inert.set("diaginertia", "%.6g %.6g %.6g" % tuple(I))

    out = os.path.join(OUT_DIR, "g1_unitree" + ("_ascender" if with_tool else "") + ".xml")
    ET.indent(tree, space="  "); tree.write(out, encoding="unicode")
    m = mujoco.MjModel.from_xml_path(out)   # compile check
    print(f"wrote {os.path.relpath(out)}: {m.nu} actuators, {m.nbody - 1} bodies, mass {sum(m.body_mass):.3f} kg, "
          f"{TOOL_LINK} {m.body_mass[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, TOOL_LINK)]:.3f} kg")
    return out


ASCENDER_ALONE = """<mujoco model="ascender">
  <compiler meshdir="." texturedir="."/>
  <asset>
    <mesh name="ascender_visual" file="meshes/ascender_visual.obj"/>
    <mesh name="ascender_collision" file="meshes/ascender_collision.obj"/>
    <texture name="ascender_basecolor" type="2d" file="meshes/ascender_basecolor.png"/>
    <material name="ascender" texture="ascender_basecolor" specular="0.6" shininess="0.5"/>
  </asset>
  <worldbody>
    <light pos="0 0 1"/>
    <body name="ascender" pos="0 0 0.2">
      <freejoint/>
      <geom type="mesh" mesh="ascender_visual" material="ascender" contype="0" conaffinity="0" group="2"/>
      <geom type="mesh" mesh="ascender_collision" mass="{mass}" group="3"/>
    </body>
  </worldbody>
</mujoco>
"""


def write_ascender_alone(mass):
    with open(os.path.join(OUT_DIR, "ascender.xml"), "w") as f:
        f.write(ASCENDER_ALONE.format(mass=mass))
    mujoco.MjModel.from_xml_path(os.path.join(OUT_DIR, "ascender.xml"))   # compile check
    print("wrote ascender.xml")


if __name__ == "__main__":
    xml = fetch_menagerie()
    if "--fetch" in sys.argv:
        print("stock STLs ready:", os.path.dirname(xml)); sys.exit()
    build(xml, with_tool=False)
    build(xml, with_tool=True)
    write_ascender_alone(ascender_meshes()[2])

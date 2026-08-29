#!/usr/bin/env python3
"""Build a Unitree G1 USD for Isaac Sim / Isaac Lab from the MuJoCo Menagerie MJCF,
and dress it with a mountaineering jacket + plastic boots (visual-only shells).

Usage:  python build_g1_usd.py [--out g1_himalaya.usd] [--no-gear]
Deps :  pip install mujoco usd-core trimesh
"""
import argparse, math, os
import numpy as np
import mujoco, trimesh
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Gf, Sdf, Vt

def physx(prim, api, attrs):
    """usd-core has no PhysxSchema; apply the API by name and write its attrs (Isaac Sim reads them fine)."""
    prim.AddAppliedSchema(api)
    ns = api[0].lower() + api[1:-3]  # PhysxArticulationAPI -> physxArticulation
    for k, (t, v) in attrs.items():
        prim.CreateAttribute(f"{ns}:{k}", t).Set(v)

HERE = os.path.dirname(os.path.abspath(__file__))
MENAGERIE = os.path.join(HERE, "_menagerie")  # git-ignored cache of google-deepmind/mujoco_menagerie (unitree_g1 only)

def fetch_menagerie():
    xml = os.path.join(MENAGERIE, "unitree_g1", "g1.xml")
    if not os.path.exists(xml):
        import subprocess
        subprocess.check_call(["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--sparse",
                               "https://github.com/google-deepmind/mujoco_menagerie.git", MENAGERIE])
        subprocess.check_call(["git", "-C", MENAGERIE, "sparse-checkout", "set", "unitree_g1"])
    return xml

# ------------------------------------------------------------------ gear spec
# link -> (color, inflate_m, name)  — a shell is the link's convex hull pushed out along its normals
JACKET_BLUE = (0.05, 0.25, 0.85)
BOOT_YELLOW = (0.95, 0.80, 0.05)
BOOT_TRIM   = (0.10, 0.10, 0.10)
BOOT_FRICTION = 0.8  # stock menagerie = 0.6. Keep moderate: higher grip changes slip dynamics -> retrain the policy.
GEAR = {
    "torso_link":              (JACKET_BLUE, 0.020, "jacket_body"),
    "left_shoulder_roll_link": (JACKET_BLUE, 0.014, "jacket_sleeve"),
    "left_shoulder_yaw_link":  (JACKET_BLUE, 0.014, "jacket_sleeve"),
    "left_elbow_link":         (JACKET_BLUE, 0.012, "jacket_sleeve"),
    "right_shoulder_roll_link":(JACKET_BLUE, 0.014, "jacket_sleeve"),
    "right_shoulder_yaw_link": (JACKET_BLUE, 0.014, "jacket_sleeve"),
    "right_elbow_link":        (JACKET_BLUE, 0.012, "jacket_sleeve"),
    "left_ankle_roll_link":    (BOOT_YELLOW, 0.012, "boot"),
    "right_ankle_roll_link":   (BOOT_YELLOW, 0.012, "boot"),
    "left_ankle_pitch_link":   (BOOT_YELLOW, 0.020, "boot_cuff"),
    "right_ankle_pitch_link":  (BOOT_YELLOW, 0.020, "boot_cuff"),
}
# Extra primitive gear: (link, kind, pos, size, color)
GEAR_PRIMS = [
    # gaiter/cuff cylinders around the shin bottom (knee link, local z down to -0.30)
    ("left_knee_link",  "cyl", (0.0, 0.0, -0.245), (0.052, 0.045), BOOT_YELLOW),
    ("right_knee_link", "cyl", (0.0, 0.0, -0.245), (0.052, 0.045), BOOT_YELLOW),
    ("left_knee_link",  "cyl", (0.0, 0.0, -0.20), (0.049, 0.008), BOOT_TRIM),
    ("right_knee_link", "cyl", (0.0, 0.0, -0.20), (0.049, 0.008), BOOT_TRIM),
    # hood behind the head (torso local frame; head sits ~0.30-0.45 up)
]

def q_mj2gf(q):  # mujoco wxyz -> Gf.Quatf
    return Gf.Quatf(float(q[0]), float(q[1]), float(q[2]), float(q[3]))

def set_xform(prim, pos, quat_wxyz):
    x = UsdGeom.Xformable(prim)
    x.ClearXformOpOrder()
    x.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
    x.AddOrientOp().Set(q_mj2gf(quat_wxyz))

def make_material(stage, root, name, rgb, roughness=0.5, metallic=0.0):
    path = root.AppendChild("Looks").AppendChild(name)
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path.AppendChild("Shader"))
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat

def write_mesh(stage, path, verts, faces, material=None, collision=False):
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, v)) for v in verts]))
    m.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray([int(i) for f in faces for i in f]))
    m.CreateSubdivisionSchemeAttr("none")
    ext = np.stack([verts.min(0), verts.max(0)])
    m.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, e)) for e in ext]))
    if material:
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(material)
    if collision:
        UsdPhysics.CollisionAPI.Apply(m.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr("convexHull")
        m.CreatePurposeAttr("guide"); m.CreateVisibilityAttr("invisible")
    return m

def mesh_arrays(model, mid):
    va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
    fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
    return model.mesh_vert[va:va+vn].copy(), model.mesh_face[fa:fa+fn].copy()

def inflated_hull(verts, faces, offset):
    hull = trimesh.Trimesh(verts, faces, process=True).convex_hull
    hull = hull.subdivide()  # smoother offset
    v = hull.vertices + hull.vertex_normals * offset
    return v, hull.faces

# ------------------------------------------------------------------ build
def build(xml, out, gear=True):
    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)  # pose at qpos0 (pelvis at z=0.793)

    stage = Usd.Stage.CreateNew(out)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = Sdf.Path("/G1")
    root_prim = UsdGeom.Xform.Define(stage, root).GetPrim()
    stage.SetDefaultPrim(root_prim)
    UsdPhysics.ArticulationRootAPI.Apply(root_prim)
    physx(root_prim, "PhysxArticulationAPI", {
        "solverPositionIterationCount": (Sdf.ValueTypeNames.Int, 8),
        "solverVelocityIterationCount": (Sdf.ValueTypeNames.Int, 0),
        "enabledSelfCollisions": (Sdf.ValueTypeNames.Bool, False)})

    mats = {
        "metal": make_material(stage, root, "metal", (0.7, 0.7, 0.7), 0.4, 0.6),
        "black": make_material(stage, root, "black", (0.2, 0.2, 0.2), 0.6, 0.2),
        "jacket": make_material(stage, root, "jacket_blue", JACKET_BLUE, 0.85, 0.0),
        "boot": make_material(stage, root, "boot_yellow", BOOT_YELLOW, 0.25, 0.0),
        "trim": make_material(stage, root, "boot_trim", BOOT_TRIM, 0.7, 0.0),
    }
    color2mat = {JACKET_BLUE: mats["jacket"], BOOT_YELLOW: mats["boot"], BOOT_TRIM: mats["trim"]}

    body_paths = {}
    for b in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
        bpath = root.AppendChild(name)
        body_paths[b] = bpath
        prim = UsdGeom.Xform.Define(stage, bpath).GetPrim()
        set_xform(prim, data.xpos[b], data.xquat[b])
        UsdPhysics.RigidBodyAPI.Apply(prim)
        physx(prim, "PhysxRigidBodyAPI", {"maxDepenetrationVelocity": (Sdf.ValueTypeNames.Float, 1.0)})
        mass = UsdPhysics.MassAPI.Apply(prim)
        mass.CreateMassAttr(float(model.body_mass[b]))
        mass.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, model.body_ipos[b])))
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, model.body_inertia[b])))
        mass.CreatePrincipalAxesAttr(q_mj2gf(model.body_iquat[b]))

        vis = UsdGeom.Scope.Define(stage, bpath.AppendChild("visuals"))
        col = UsdGeom.Scope.Define(stage, bpath.AppendChild("collisions"))
        link_verts = []  # for gear hulls (visual mesh geoms in body frame)

        for g in range(model.body_geomadr[b], model.body_geomadr[b] + model.body_geomnum[b]):
            is_col = model.geom_contype[g] != 0 or model.geom_conaffinity[g] != 0
            parent = col if is_col else vis
            gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom_{g}"
            gpath = parent.GetPath().AppendChild(gname)
            gtype = model.geom_type[g]
            rgba = model.geom_rgba[g]
            mat = mats["black"] if rgba[0] < 0.5 else mats["metal"]
            if gtype == mujoco.mjtGeom.mjGEOM_MESH:
                v, f = mesh_arrays(model, model.geom_dataid[g])
                m = write_mesh(stage, gpath, v, f, None if is_col else mat, collision=is_col)
                set_xform(m.GetPrim(), model.geom_pos[g], model.geom_quat[g])
                if not is_col:
                    R = np.zeros(9); mujoco.mju_quat2Mat(R, model.geom_quat[g]); R = R.reshape(3, 3)
                    link_verts.append((v.astype(np.float64) @ R.T + model.geom_pos[g], f))
            elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
                s = UsdGeom.Sphere.Define(stage, gpath)
                s.CreateRadiusAttr(float(model.geom_size[g][0]))
                set_xform(s.GetPrim(), model.geom_pos[g], model.geom_quat[g])
                if is_col:
                    UsdPhysics.CollisionAPI.Apply(s.GetPrim()); s.CreatePurposeAttr("guide"); s.CreateVisibilityAttr("invisible")
                    pm = UsdPhysics.MaterialAPI.Apply(s.GetPrim())
                    mu = BOOT_FRICTION if gear else float(model.geom_friction[g][0])
                    pm.CreateStaticFrictionAttr(mu); pm.CreateDynamicFrictionAttr(mu)
            elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
                c = UsdGeom.Cylinder.Define(stage, gpath)
                c.CreateRadiusAttr(float(model.geom_size[g][0]))
                c.CreateHeightAttr(float(2 * model.geom_size[g][1]))
                c.CreateAxisAttr("Z")
                set_xform(c.GetPrim(), model.geom_pos[g], model.geom_quat[g])
                if is_col:
                    UsdPhysics.CollisionAPI.Apply(c.GetPrim()); c.CreatePurposeAttr("guide"); c.CreateVisibilityAttr("invisible")

        # ---- gear (visual only, no collision, no mass change)
        if gear and name in GEAR and link_verts:
            color, off, gname = GEAR[name]
            v = np.concatenate([lv for lv, _ in link_verts])
            f = np.concatenate([lf + sum(len(x[0]) for x in link_verts[:i]) for i, (_, lf) in enumerate(link_verts)])
            hv, hf = inflated_hull(v, f, off)
            if name == "torso_link":
                hv[:, 2] = np.minimum(hv[:, 2], 0.30)  # collar stops below the head
            gearscope = UsdGeom.Scope.Define(stage, bpath.AppendChild("gear"))
            write_mesh(stage, gearscope.GetPath().AppendChild(gname), hv, hf, color2mat[color])
        if gear:
            for link, kind, pos, size, color in GEAR_PRIMS:
                if link != name: continue
                gearscope = UsdGeom.Scope.Define(stage, bpath.AppendChild("gear"))
                p = gearscope.GetPath().AppendChild(f"{kind}_{len(gearscope.GetPrim().GetChildren())}")
                if kind == "cyl":
                    c = UsdGeom.Cylinder.Define(stage, p)
                    c.CreateRadiusAttr(size[0]); c.CreateHeightAttr(2 * size[1]); c.CreateAxisAttr("Z")
                    prim = c.GetPrim()
                else:
                    s = UsdGeom.Sphere.Define(stage, p); s.CreateRadiusAttr(size[0]); prim = s.GetPrim()
                set_xform(prim, pos, (1, 0, 0, 0))
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(color2mat[color])

    # ---- sensor frames
    # IMUs: from the MJCF <site>s (Isaac Lab ImuCfg / ContactSensorCfg attach to these prims)
    for sid in range(model.nsite):
        sname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, sid)
        if not sname.startswith("imu"):
            continue
        b = model.site_bodyid[sid]
        p = UsdGeom.Xform.Define(stage, body_paths[b].AppendChild(sname)).GetPrim()
        set_xform(p, model.site_pos[sid], model.site_quat[sid])
    # Head sensors (not in menagerie; positions from the G1 spec, torso frame; head spans z 0.28..0.49)
    torso = body_paths[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")]
    head = UsdGeom.Xform.Define(stage, torso.AppendChild("head_sensors")).GetPrim()
    set_xform(head, (0.0039635, 0, -0.044), (1, 0, 0, 0))  # same offset as head_link geom
    # Intel RealSense D435i: front of the face, looking +X. USD cameras look down -Z with +Y up,
    # so cam(-Z)->+X, cam(+Y)->+Z, cam(+X)->-Y.
    cam = UsdGeom.Camera.Define(stage, head.GetPath().AppendChild("d435i_camera"))
    cam.CreateFocalLengthAttr(1.93); cam.CreateHorizontalApertureAttr(3.896); cam.CreateVerticalApertureAttr(2.453)  # ~87x58 deg
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.1, 20.0))
    # Gf is row-vector convention (v * M): rows are the world-space images of cam x, y, z.
    R = Gf.Matrix3d(0, -1, 0,   0, 0, 1,   -1, 0, 0)
    assert R.GetRow(2) * -1 == Gf.Vec3d(1, 0, 0) and R.GetRow(1) == Gf.Vec3d(0, 0, 1)
    q = R.ExtractRotation().GetQuat()
    xf = UsdGeom.Xformable(cam.GetPrim()); xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(0.075, 0.0, 0.43))
    xf.AddOrientOp().Set(Gf.Quatf(q.GetReal(), *q.GetImaginary()))
    # Livox Mid-360 LiDAR: on top of the head, +Z up (Isaac Lab RayCasterCfg / RTX lidar attach here)
    lid = UsdGeom.Xform.Define(stage, head.GetPath().AppendChild("mid360_lidar")).GetPrim()
    set_xform(lid, (0.0, 0.0, 0.50), (1, 0, 0, 0))

    # ---- joints
    kp = {}; kv = {}
    for a in range(model.nu):
        j = model.actuator_trnid[a][0]
        kp[j] = float(model.actuator_gainprm[a][0]); kv[j] = float(-model.actuator_biasprm[a][2])
    for j in range(model.njnt):
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue  # free joint -> floating base, nothing to write
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        b = model.jnt_bodyid[j]; pb = model.body_parentid[b]
        axis = model.jnt_axis[j]
        ax = "XYZ"[int(np.argmax(np.abs(axis)))]
        assert abs(abs(axis[ "XYZ".index(ax)]) - 1) < 1e-6, f"non-principal axis on {name}"
        jp = body_paths[b].AppendChild(name)
        jnt = UsdPhysics.RevoluteJoint.Define(stage, jp)
        jnt.CreateBody0Rel().SetTargets([body_paths[pb]])
        jnt.CreateBody1Rel().SetTargets([body_paths[b]])
        jnt.CreateLocalPos0Attr(Gf.Vec3f(*map(float, model.body_pos[b] + model.jnt_pos[j] * 0)))
        jnt.CreateLocalRot0Attr(q_mj2gf(model.body_quat[b]))
        jnt.CreateLocalPos1Attr(Gf.Vec3f(*map(float, model.jnt_pos[j])))
        jnt.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
        jnt.CreateAxisAttr(ax)
        lo, hi = model.jnt_range[j]
        sgn = 1.0 if axis["XYZ".index(ax)] > 0 else -1.0
        lo, hi = sorted([sgn * lo, sgn * hi])
        jnt.CreateLowerLimitAttr(math.degrees(lo)); jnt.CreateUpperLimitAttr(math.degrees(hi))
        physx(jnt.GetPrim(), "PhysxJointAPI", {
            "armature": (Sdf.ValueTypeNames.Float, float(model.dof_armature[model.jnt_dofadr[j]])),
            "jointFriction": (Sdf.ValueTypeNames.Float, float(model.dof_frictionloss[model.jnt_dofadr[j]]))})
        drv = UsdPhysics.DriveAPI.Apply(jnt.GetPrim(), "angular")
        drv.CreateTypeAttr("force")
        drv.CreateStiffnessAttr(kp.get(j, 0.0)); drv.CreateDampingAttr(kv.get(j, 0.0))
        drv.CreateMaxForceAttr(float(abs(model.jnt_actfrcrange[j][1])) or 1e6)
        drv.CreateTargetPositionAttr(0.0)

    stage.GetRootLayer().Save()
    return stage

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default=None)
    ap.add_argument("--out", default=os.path.join(HERE, "g1_himalaya.usd"))
    ap.add_argument("--copy-to", default=os.path.join(HERE, "..", "g1_unitree.usd"), help="also write a copy here (shared file)")
    ap.add_argument("--no-gear", action="store_true")
    a = ap.parse_args()
    build(a.xml or fetch_menagerie(), a.out, gear=not a.no_gear)
    if a.copy_to:
        import shutil; shutil.copy(a.out, a.copy_to); print("copied to", a.copy_to)
    print("wrote", a.out)

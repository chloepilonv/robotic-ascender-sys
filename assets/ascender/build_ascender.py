#!/usr/bin/env python3
"""Turn the raw Tripo scan (climbing_tool_raw.usdz, 1 m tall, 982k verts) into a sim-ready ascender.

Output ascender.usd: Z-up, metres, origin at the bottom of the handle (carabiner hole), cam head up (+Z),
scaled to a real handled ascender (Petzl Ascension: 195 mm tall, 165 g), decimated by vertex clustering,
convex-hull collision + rigid-body mass. Deps: usd-core numpy trimesh
"""
import os, zipfile, tempfile, numpy as np, trimesh
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Gf, Sdf, Vt

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "climbing_tool_raw.usdz")
OUT = os.path.join(HERE, "ascender.usd")
HEIGHT_M, MASS_KG, CELL_M = 0.195, 0.165, 0.005   # real size / mass ; clustering cell on the 1 m model

def cluster(pts, tri, cell):
    q = np.floor(pts / cell).astype(np.int64)
    key = q[:, 0] * 73856093 ^ q[:, 1] * 19349663 ^ q[:, 2] * 83492791
    u, inv = np.unique(key, return_inverse=True)
    newp = np.zeros((len(u), 3)); cnt = np.zeros(len(u))
    np.add.at(newp, inv, pts); np.add.at(cnt, inv, 1); newp /= cnt[:, None]
    f = inv[tri]; f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    return newp, f

with tempfile.TemporaryDirectory() as tmp:
    zipfile.ZipFile(RAW).extractall(tmp)
    src = [f for f in os.listdir(tmp) if f.endswith((".usdc", ".usda", ".usd"))][0]
    st = Usd.Stage.Open(os.path.join(tmp, src))
    mesh = [p for p in st.Traverse() if p.IsA(UsdGeom.Mesh)][0]
    pts = np.array(UsdGeom.Mesh(mesh).GetPointsAttr().Get(), dtype=np.float64)
    tri = np.array(UsdGeom.Mesh(mesh).GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)

pts -= [(pts[:, 0].min() + pts[:, 0].max()) / 2, (pts[:, 1].min() + pts[:, 1].max()) / 2, pts[:, 2].min()]
v, f = cluster(pts, tri, CELL_M)
tm = trimesh.Trimesh(v, f, process=True)
v, f = tm.vertices * (HEIGHT_M / pts[:, 2].max()), tm.faces

stage = Usd.Stage.CreateNew(OUT)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage, 1.0)
root = UsdGeom.Xform.Define(stage, "/Ascender"); stage.SetDefaultPrim(root.GetPrim())
UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
mass = UsdPhysics.MassAPI.Apply(root.GetPrim()); mass.CreateMassAttr(MASS_KG)
mass.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, v.mean(0))))

mat = UsdShade.Material.Define(stage, "/Ascender/Looks/anodized_alu")
sh = UsdShade.Shader.Define(stage, "/Ascender/Looks/anodized_alu/Shader"); sh.CreateIdAttr("UsdPreviewSurface")
sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.85, 0.45, 0.05))  # orange anodised handle
sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.8); sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.35)
mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")

m = UsdGeom.Mesh.Define(stage, "/Ascender/visual")
m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v.astype(np.float32)))
m.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(f))); m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(f.astype(np.int32).ravel()))
m.CreateSubdivisionSchemeAttr("none")
m.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, v.min(0))), Gf.Vec3f(*map(float, v.max(0)))]))
UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)

hull = trimesh.Trimesh(v, f).convex_hull
c = UsdGeom.Mesh.Define(stage, "/Ascender/collision")
c.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(hull.vertices.astype(np.float32)))
c.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(hull.faces))); c.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(hull.faces.astype(np.int32).ravel()))
c.CreatePurposeAttr("guide"); UsdPhysics.CollisionAPI.Apply(c.GetPrim())
UsdPhysics.MeshCollisionAPI.Apply(c.GetPrim()).CreateApproximationAttr("convexHull")
stage.GetRootLayer().Save()
print(f"wrote {OUT}: {len(v)} verts, {len(f)} faces, size {np.round(v.max(0) - v.min(0), 3)} m, {MASS_KG} kg")

#!/usr/bin/env python3
"""Sim-ready ascender from the raw Tripo scan (climbing_tool_raw.usdz, 1 m tall, 982k verts, full PBR textures).

ascender.usd:
  /Ascender                RigidBodyAPI + MassAPI (165 g), Z-up, metres, origin = bottom of handle, +Z = cam head
  /Ascender/visual         REFERENCE to the usdz (/root) -> original mesh + basecolor/metallic/roughness/normal textures
  /Ascender/collision      convex hull (decimated), purpose=guide
Scaled to a real handled ascender (Petzl Ascension: 195 mm tall). Deps: usd-core numpy trimesh
"""
import os, zipfile, tempfile, numpy as np, trimesh
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Vt

HERE = os.path.dirname(os.path.abspath(__file__))
RAW, OUT = os.path.join(HERE, "climbing_tool_raw.usdz"), os.path.join(HERE, "ascender.usd")
HEIGHT_M, MASS_KG = 0.195, 0.165

with tempfile.TemporaryDirectory() as tmp:
    zipfile.ZipFile(RAW).extractall(tmp)
    src = [f for f in os.listdir(tmp) if f.endswith((".usdc", ".usda", ".usd"))][0]
    st = Usd.Stage.Open(os.path.join(tmp, src))
    mesh = [p for p in st.Traverse() if p.IsA(UsdGeom.Mesh)][0]
    pts = np.array(UsdGeom.Mesh(mesh).GetPointsAttr().Get(), dtype=np.float64)
    tri = np.array(UsdGeom.Mesh(mesh).GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    root_path = st.GetDefaultPrim().GetPath()

center = np.array([(pts[:, 0].min() + pts[:, 0].max()) / 2, (pts[:, 1].min() + pts[:, 1].max()) / 2, pts[:, 2].min()])
scale = HEIGHT_M / (pts[:, 2].max() - pts[:, 2].min())

stage = Usd.Stage.CreateNew(OUT)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage, 1.0)
root = UsdGeom.Xform.Define(stage, "/Ascender"); stage.SetDefaultPrim(root.GetPrim())
UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())

vis = UsdGeom.Xform.Define(stage, "/Ascender/visual")
vis.GetPrim().GetReferences().AddReference("./climbing_tool_raw.usdz", root_path)
x = UsdGeom.Xformable(vis.GetPrim())   # scale then re-centre: p' = (p - center) * scale
x.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))
x.AddTranslateOp().Set(Gf.Vec3d(*(-center)))

hull = trimesh.Trimesh((pts - center) * scale, tri).convex_hull
c = UsdGeom.Mesh.Define(stage, "/Ascender/collision")
c.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(hull.vertices.astype(np.float32)))
c.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(hull.faces)))
c.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(hull.faces.astype(np.int32).ravel()))
c.CreatePurposeAttr("guide"); UsdPhysics.CollisionAPI.Apply(c.GetPrim())
UsdPhysics.MeshCollisionAPI.Apply(c.GetPrim()).CreateApproximationAttr("convexHull")

mass = UsdPhysics.MassAPI.Apply(root.GetPrim()); mass.CreateMassAttr(MASS_KG)
mass.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, ((pts - center) * scale).mean(0))))
stage.GetRootLayer().Save()
print(f"wrote {OUT}: scale {scale:.4f}, size {np.round((pts.max(0) - pts.min(0)) * scale, 3)} m, hull {len(hull.faces)} faces")

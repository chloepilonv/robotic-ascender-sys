#!/usr/bin/env python3
"""Sim-ready ascender from the raw Tripo scan (climbing_tool_raw.usdz, 1 m tall, 982k verts, full PBR textures).

Only the CAM HEAD is kept (rope channel + locking mechanism); the handle loop and rubber grip are cut away
(everything below model z = CUT_Z). Textures are extracted next to this file and re-bound.

ascender.usd:
  /Ascender             RigidBodyAPI + MassAPI, Z-up, metres. Frame = full tool frame (origin at the old handle bottom,
                        +Z = up through the head) so mounts computed for the full tool still work.
  /Ascender/visual      Xform: mesh + Looks (UsdPreviewSurface with basecolor/metallic/roughness/normal
  /Ascender/collision   convex hull of the trimmed part, purpose=guide, invisible
Deps: usd-core numpy trimesh
"""
import os, zipfile, tempfile, shutil, numpy as np, trimesh
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Gf, Sdf, Vt

HERE = os.path.dirname(os.path.abspath(__file__))
RAW, OUT, TEXDIR = os.path.join(HERE, "climbing_tool_raw.usdz"), os.path.join(HERE, "ascender.usd"), os.path.join(HERE, "textures")
HEIGHT_M = 0.195          # full tool height (Petzl Ascension) -> sets the scale
CUT_Z = 0.65              # model-space (1 m tall) cut: keep z > CUT_Z  (cam head)
MASS_KG = 0.110           # cam head + body only (full tool = 165 g)

with tempfile.TemporaryDirectory() as tmp:
    zipfile.ZipFile(RAW).extractall(tmp)
    src = [f for f in os.listdir(tmp) if f.endswith((".usdc", ".usda", ".usd"))][0]
    st = Usd.Stage.Open(os.path.join(tmp, src))
    mesh = UsdGeom.Mesh([p for p in st.Traverse() if p.IsA(UsdGeom.Mesh)][0])
    pts = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)
    tri = np.array(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    st_uv = np.array(UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st").Get(), dtype=np.float32).reshape(-1, 3, 2)  # faceVarying
    os.makedirs(TEXDIR, exist_ok=True)
    for f in os.listdir(os.path.join(tmp, "textures")):
        shutil.copy(os.path.join(tmp, "textures", f), TEXDIR)

center = np.array([(pts[:, 0].min() + pts[:, 0].max()) / 2, (pts[:, 1].min() + pts[:, 1].max()) / 2, pts[:, 2].min()])
scale = HEIGHT_M / (pts[:, 2].max() - pts[:, 2].min())
P = (pts - center) * scale

keep = np.all(pts[tri][:, :, 2] > CUT_Z, axis=1)          # faces fully above the cut
tri_k, uv_k = tri[keep], st_uv[keep]
used = np.unique(tri_k); remap = -np.ones(len(P), dtype=np.int64); remap[used] = np.arange(len(used))
Pk, Fk = P[used], remap[tri_k]

stage = Usd.Stage.CreateNew(OUT)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage, 1.0)
root = UsdGeom.Xform.Define(stage, "/Ascender"); stage.SetDefaultPrim(root.GetPrim())
UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())

UsdGeom.Xform.Define(stage, "/Ascender/visual")
# material: UsdPreviewSurface + 4 texture readers
mat = UsdShade.Material.Define(stage, "/Ascender/visual/Looks/ascender_pbr")
sh = UsdShade.Shader.Define(stage, "/Ascender/visual/Looks/ascender_pbr/Surface"); sh.CreateIdAttr("UsdPreviewSurface")
uvr = UsdShade.Shader.Define(stage, "/Ascender/visual/Looks/ascender_pbr/uv"); uvr.CreateIdAttr("UsdPrimvarReader_float2")
uvr.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st"); uvr.CreateOutput("result", Sdf.ValueTypeNames.Float2)
for name, tex, chan, cs, typ in [("diffuseColor", "basecolor", "rgb", "sRGB", Sdf.ValueTypeNames.Color3f),
                                 ("metallic", "metallic", "r", "raw", Sdf.ValueTypeNames.Float),
                                 ("roughness", "roughness", "r", "raw", Sdf.ValueTypeNames.Float),
                                 ("normal", "normal", "rgb", "raw", Sdf.ValueTypeNames.Normal3f)]:
    t = UsdShade.Shader.Define(stage, f"/Ascender/visual/Looks/ascender_pbr/tex_{tex}"); t.CreateIdAttr("UsdUVTexture")
    t.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f"./textures/climbing_tool_3d_model_{tex}.JPEG")
    t.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(cs)
    t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(uvr.ConnectableAPI(), "result")
    if tex == "normal":
        t.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 2)); t.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, -1))
    t.CreateOutput(chan, typ)
    sh.CreateInput(name, typ).ConnectToSource(t.ConnectableAPI(), chan)
mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")

m = UsdGeom.Mesh.Define(stage, "/Ascender/visual/mesh")
m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(Pk.astype(np.float32)))
m.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(Fk))); m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(Fk.astype(np.int32).ravel()))
m.CreateSubdivisionSchemeAttr("none")
m.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, Pk.min(0))), Gf.Vec3f(*map(float, Pk.max(0)))]))
pv = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
pv.Set(Vt.Vec2fArray.FromNumpy(uv_k.reshape(-1, 2)))
UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)

hull = trimesh.Trimesh(Pk, Fk).convex_hull
c = UsdGeom.Mesh.Define(stage, "/Ascender/collision")
c.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(hull.vertices.astype(np.float32)))
c.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(hull.faces))); c.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(hull.faces.astype(np.int32).ravel()))
c.CreatePurposeAttr("guide"); c.CreateVisibilityAttr("invisible"); UsdPhysics.CollisionAPI.Apply(c.GetPrim())
UsdPhysics.MeshCollisionAPI.Apply(c.GetPrim()).CreateApproximationAttr("convexHull")

mass = UsdPhysics.MassAPI.Apply(root.GetPrim()); mass.CreateMassAttr(MASS_KG)
mass.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, Pk.mean(0))))
stage.GetRootLayer().Save()
print(f"wrote {OUT}: kept {len(Fk)}/{len(tri)} faces, {len(Pk)} verts, part spans z {Pk[:,2].min():.3f}..{Pk[:,2].max():.3f} m, {MASS_KG} kg")

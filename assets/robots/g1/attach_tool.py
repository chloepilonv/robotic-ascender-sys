#!/usr/bin/env python3
"""Create g1_unitree_ascender.usd: the dressed G1 holding the ascender in its right hand.

The tool is baked into `right_wrist_yaw_link` (visual + convex collision), its mass/CoM folded into the link's
MassAPI. No extra body/joint -> same 29-DoF articulation, Isaac Lab cfgs unchanged.
Grip: handle vertical in front of the palm (wrist +Z = up, cam head up), like on a fixed rope.
"""
import os, numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Gf, Sdf, Vt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "g1_himalaya.usd")
TOOL = os.path.join(HERE, "..", "..", "ascender", "ascender.usd")
OUT = os.path.join(HERE, "..", "g1_unitree_ascender.usd")
LINK = "/G1/right_wrist_yaw_link"
# tool origin (bottom of handle) in wrist_yaw frame: just in front of the rubber-hand paddle (x 0.087..0.132),
# handle spanning the palm height (z -0.07..0.05), head above.
TOOL_POS = Gf.Vec3d(0.145, 0.012, -0.075)
TOOL_ROT = Gf.Quatf(1, 0, 0, 0)  # tool Z (handle) = wrist Z; tool X (width) = wrist X

tool = Usd.Stage.Open(TOOL)
stage = Usd.Stage.Open(SRC)
link = stage.GetPrimAtPath(LINK)
tp = UsdGeom.Xform.Define(stage, LINK + "/tool_ascender")
xf = UsdGeom.Xformable(tp.GetPrim()); xf.AddTranslateOp().Set(TOOL_POS); xf.AddOrientOp().Set(TOOL_ROT)

mat = UsdShade.Material.Define(stage, "/G1/Looks/ascender_alu")
sh = UsdShade.Shader.Define(stage, "/G1/Looks/ascender_alu/Shader"); sh.CreateIdAttr("UsdPreviewSurface")
src_sh = UsdShade.Shader(tool.GetPrimAtPath("/Ascender/Looks/anodized_alu/Shader"))
for n, t in [("diffuseColor", Sdf.ValueTypeNames.Color3f), ("metallic", Sdf.ValueTypeNames.Float), ("roughness", Sdf.ValueTypeNames.Float)]:
    sh.CreateInput(n, t).Set(src_sh.GetInput(n).Get())
mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")

for name, collision in [("visual", False), ("collision", True)]:
    s = UsdGeom.Mesh(tool.GetPrimAtPath("/Ascender/" + name))
    d = UsdGeom.Mesh.Define(stage, tp.GetPath().AppendChild(name))
    d.CreatePointsAttr(s.GetPointsAttr().Get()); d.CreateFaceVertexCountsAttr(s.GetFaceVertexCountsAttr().Get())
    d.CreateFaceVertexIndicesAttr(s.GetFaceVertexIndicesAttr().Get()); d.CreateSubdivisionSchemeAttr("none")
    d.CreateExtentAttr(s.GetExtentAttr().Get())
    if collision:
        d.CreatePurposeAttr("guide"); UsdPhysics.CollisionAPI.Apply(d.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(d.GetPrim()).CreateApproximationAttr("convexHull")
    else:
        UsdShade.MaterialBindingAPI.Apply(d.GetPrim()).Bind(mat)

# fold tool mass into the link (parallel-axis on the diagonal inertia is small at 165 g; keep principal axes)
mass = UsdPhysics.MassAPI(link)
m0, c0 = mass.GetMassAttr().Get(), Gf.Vec3d(mass.GetCenterOfMassAttr().Get())
mt = UsdPhysics.MassAPI(tool.GetPrimAtPath("/Ascender")).GetMassAttr().Get()
ct = Gf.Vec3d(UsdPhysics.MassAPI(tool.GetPrimAtPath("/Ascender")).GetCenterOfMassAttr().Get()) + TOOL_POS
c = (c0 * m0 + ct * mt) / (m0 + mt)
mass.GetMassAttr().Set(float(m0 + mt)); mass.GetCenterOfMassAttr().Set(Gf.Vec3f(c))
I0 = np.array(mass.GetDiagonalInertiaAttr().Get()); r = np.array(ct - c); I0 += mt * (r.dot(r) - r * r)
mass.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*map(float, I0)))
link.SetCustomDataByKey("tool", "ascender")

stage.GetRootLayer().Export(OUT)
print(f"wrote {OUT}; {LINK} mass {m0:.3f} -> {m0 + mt:.3f} kg")

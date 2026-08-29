#!/usr/bin/env python3
"""Create g1_unitree_ascender.usd: the dressed G1 with the ascender AS its right end-effector.

The rubber hand is removed; the ascender is bolted to `right_wrist_yaw_link` where the hand was, handle along
the forearm (+X), cam head pointing outward. Visual + convex collision are baked into the link, its mass/CoM
folded into the link's MassAPI. No extra body/joint -> same 29-DoF articulation, Isaac Lab cfgs unchanged.
"""
import os, numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Gf, Sdf, Vt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "g1_himalaya.usd")
TOOL = os.path.join(HERE, "..", "..", "ascender", "ascender.usd")
OUT = os.path.join(HERE, "..", "g1_unitree_ascender.usd")
LINK = "/G1/right_wrist_yaw_link"
# tool origin (bottom of handle) at the end of the wrist link (its mesh ends at x=0.047); tool Z (handle -> cam head)
# rotated onto wrist +X (forearm axis): rotate 90 deg about Y. Tool width (X) -> wrist -Z, thickness (Y) -> wrist Y.
TOOL_POS = Gf.Vec3d(0.05, 0.0, 0.0)
TOOL_ROT = Gf.Quatf(0.7071068, 0.0, 0.7071068, 0.0)
HAND_X_MIN = 0.08  # the rubber-hand paddle lives at x 0.087..0.132 in the wrist frame; wrist link mesh ends at 0.047

tool = Usd.Stage.Open(TOOL)
stage = Usd.Stage.Open(SRC)
link = stage.GetPrimAtPath(LINK)
for child in list(stage.GetPrimAtPath(LINK + "/visuals").GetChildren()):  # drop the rubber hand: the tool replaces it
    if child.IsA(UsdGeom.Mesh):
        off = UsdGeom.Xformable(child).GetLocalTransformation().ExtractTranslation()
        if min(pt[0] for pt in UsdGeom.Mesh(child).GetPointsAttr().Get()) + off[0] > HAND_X_MIN:
            stage.RemovePrim(child.GetPath())
tp = UsdGeom.Xform.Define(stage, LINK + "/tool_ascender")
xf = UsdGeom.Xformable(tp.GetPrim()); xf.AddTranslateOp().Set(TOOL_POS); xf.AddOrientOp().Set(TOOL_ROT)
TOOL_ROT_D = Gf.Quatd(TOOL_ROT.GetReal(), *TOOL_ROT.GetImaginary())

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
ct = TOOL_ROT_D.Transform(Gf.Vec3d(UsdPhysics.MassAPI(tool.GetPrimAtPath("/Ascender")).GetCenterOfMassAttr().Get())) + TOOL_POS
c = (c0 * m0 + ct * mt) / (m0 + mt)
mass.GetMassAttr().Set(float(m0 + mt)); mass.GetCenterOfMassAttr().Set(Gf.Vec3f(c))
I0 = np.array(mass.GetDiagonalInertiaAttr().Get()); r = np.array(ct - c); I0 += mt * (r.dot(r) - r * r)
mass.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*map(float, I0)))
link.SetCustomDataByKey("tool", "ascender")

stage.GetRootLayer().Export(OUT)
print(f"wrote {OUT}; {LINK} mass {m0:.3f} -> {m0 + mt:.3f} kg")

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
# Tool base sunk 2 cm into the wrist link (its mesh ends at x=0.047) so it is visibly bolted on.
# Orientation: tool Z (handle -> cam head) onto wrist +X (forearm) = rotY(90), then rolled 90 deg about the forearm
# so the flat face of the ascender lies in the wrist X-Y plane (cam opening faces +/-Z).
TOOL_POS = Gf.Vec3d(0.03, 0.0, 0.0)
# Basis mapping (Gf is row-vector convention: rows = images of tool X, Y, Z in the wrist frame):
#   tool X (width)     -> wrist +Y
#   tool Y (thickness) -> wrist +Z   (flat face in the wrist X-Y plane)
#   tool Z (handle)    -> wrist +X   (along the forearm)
_R = Gf.Matrix3d(0, 1, 0,   0, 0, 1,   1, 0, 0)
_qd = _R.ExtractRotation().GetQuat(); TOOL_ROT = Gf.Quatf(_qd.GetReal(), *_qd.GetImaginary())
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

# visual: reference the textured ascender (usdz PBR material comes along); collision: convex hull baked in
vis = stage.DefinePrim(tp.GetPath().AppendChild("visual"))
vis.GetReferences().AddReference("../ascender/ascender.usd", "/Ascender/visual")
src_col = UsdGeom.Mesh(tool.GetPrimAtPath("/Ascender/collision"))
col = UsdGeom.Mesh.Define(stage, tp.GetPath().AppendChild("collision"))
col.CreatePointsAttr(src_col.GetPointsAttr().Get()); col.CreateFaceVertexCountsAttr(src_col.GetFaceVertexCountsAttr().Get())
col.CreateFaceVertexIndicesAttr(src_col.GetFaceVertexIndicesAttr().Get()); col.CreatePurposeAttr("guide")
UsdPhysics.CollisionAPI.Apply(col.GetPrim()); UsdPhysics.MeshCollisionAPI.Apply(col.GetPrim()).CreateApproximationAttr("convexHull")

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

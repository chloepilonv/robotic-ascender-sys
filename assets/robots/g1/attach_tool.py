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
# Mount: ascender UPRIGHT like on a fixed rope - cam head centred on the wrist joint, handle hanging below
# (wrist frame: X = forearm, Z = up when the arm is horizontal).
# Basis mapping (Gf is row-vector convention: rows = images of tool X, Y, Z in the wrist frame):
#   tool X (width)        -> wrist -X
#   tool Y (thickness)    -> wrist -Y
#   tool Z (handle->head) -> wrist +Z   (head up)
_R = Gf.Matrix3d(-1, 0, 0,   0, -1, 0,   0, 0, 1)   # 180 deg yaw about wrist Z
HEAD_Z = (0.06, 0.11)                        # cam-mechanism span along the tool axis (110 mm tool, cam in the top half)
TOOL_POS = Gf.Vec3d(0.08, 0.0, -(HEAD_Z[0] + HEAD_Z[1]) / 2)   # device centre 2 cm further out (device is 73 mm wide in X -> back face at x~0.044)
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
col.CreateFaceVertexIndicesAttr(src_col.GetFaceVertexIndicesAttr().Get()); col.CreatePurposeAttr("guide"); col.CreateVisibilityAttr("invisible")
UsdPhysics.CollisionAPI.Apply(col.GetPrim()); UsdPhysics.MeshCollisionAPI.Apply(col.GetPrim()).CreateApproximationAttr("convexHull")

# mounting flange: dark cylinder from the wrist link to the cam head so the tool reads as a bolted-on end-effector
fl_mat = UsdShade.Material.Define(stage, "/G1/Looks/flange_black")
fsh = UsdShade.Shader.Define(stage, "/G1/Looks/flange_black/Shader"); fsh.CreateIdAttr("UsdPreviewSurface")
fsh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.12, 0.12, 0.12))
fsh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5); fsh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.6)
fl_mat.CreateSurfaceOutput().ConnectToSource(fsh.ConnectableAPI(), "surface")
FLANGE_X0, FLANGE_X1, FLANGE_R = 0.040, TOOL_POS[0] + 0.006, 0.018   # overlaps wrist mesh (ends 0.047) and the head
fl = UsdGeom.Cylinder.Define(stage, tp.GetPath().GetParentPath().AppendChild("tool_flange"))
fl.CreateRadiusAttr(FLANGE_R); fl.CreateHeightAttr(FLANGE_X1 - FLANGE_X0); fl.CreateAxisAttr("X")
UsdGeom.Xformable(fl.GetPrim()).AddTranslateOp().Set(Gf.Vec3d((FLANGE_X0 + FLANGE_X1) / 2, 0, 0))
UsdShade.MaterialBindingAPI.Apply(fl.GetPrim()).Bind(fl_mat)

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

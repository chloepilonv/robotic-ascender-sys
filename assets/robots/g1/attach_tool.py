#!/usr/bin/env python3
"""Create g1_unitree_ascender.usd: the dressed G1 with the ascender AS its right end-effector.

The rubber hand is removed; the ascender is bolted to `right_wrist_yaw_link` where the hand was, handle along
the forearm (+X), cam head pointing outward. Visual + convex collision are baked into the link, its mass/CoM
folded into the link's MassAPI. No extra body/joint -> same 29-DoF articulation, Isaac Lab cfgs unchanged.
"""
import os, math, numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Gf, Sdf, Vt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "g1_himalaya.usd")
TOOL = os.path.join(HERE, "..", "..", "ascender", "ascender.usd")
OUT = os.path.join(HERE, "..", "g1_unitree_ascender.usd")
LINK = "/G1/right_wrist_yaw_link"
# Mount: the device's slanted riveted edge (tool +X side, ~26 deg from vertical) butts flat against the wrist end face.
# Frame: wrist X = forearm, Z = up when the arm is horizontal. Tool: X = width, Y = thickness, Z = up through the cam.
# 1) yaw 180 deg about Z so the slanted edge faces the wrist (-X)   2) tilt about wrist Y so that edge is vertical.
_edge = np.array([0.0379 - 0.0087, 0.0, 0.07 - 0.01])              # slanted edge direction in tool frame (from the mesh)
SLANT_DEG = math.degrees(math.atan2(_edge[0], _edge[2]))
_yaw = Gf.Matrix3d(-1, 0, 0,   0, -1, 0,   0, 0, 1)
_R = None
for _sgn in (1, -1):                                   # pick the tilt sign/order that makes the edge vertical (row-vector convention)
    for _cand in (_yaw * Gf.Matrix3d(Gf.Rotation(Gf.Vec3d(0, 1, 0), _sgn * SLANT_DEG)),
                  Gf.Matrix3d(Gf.Rotation(Gf.Vec3d(0, 1, 0), _sgn * SLANT_DEG)) * _yaw):
        _e = Gf.Vec3d(*_edge).GetNormalized() * _cand
        if abs(_e[0]) < 1e-3 and _e[2] > 0.99:
            _R = _cand
assert _R is not None, "no tilt made the edge vertical"
_ts = Usd.Stage.Open(TOOL)   # keep the stage alive while reading
_tv = np.array(UsdGeom.Mesh(_ts.GetPrimAtPath("/Ascender/visual/mesh")).GetPointsAttr().Get(), dtype=np.float64)
_tr = np.array([Gf.Vec3d(*v) * _R for v in _tv[::50]])           # rotated sample of the tool points (wrist frame, before translation)
EDGE_X, CAM_Z_TOOL = 0.036, 0.085                                  # edge 1 cm past the wrist mesh (ends x=0.026); cam centre at wrist z=0
_cam = Gf.Vec3d(0, 0, CAM_Z_TOOL) * _R
Z_UP = 0.025                                                       # raise the device along wrist Z
TOOL_POS = Gf.Vec3d(EDGE_X - _tr[:, 0].min(), 0.0, -_cam[2] + Z_UP)
_qd = _R.ExtractRotation().GetQuat(); TOOL_ROT = Gf.Quatf(_qd.GetReal(), *_qd.GetImaginary())
HAND_X_MIN = 0.08  # the rubber-hand paddle lives at x 0.087..0.132 in the wrist frame; wrist link mesh ends at 0.047

tool = Usd.Stage.Open(TOOL)
src = Usd.Stage.Open(SRC)                     # read-only: geometry queries
# Output = thin stage that REFERENCES g1_himalaya.usd and authors overrides (relative texture paths keep working)
stage = Usd.Stage.CreateNew(OUT)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage, 1.0)
g1 = stage.DefinePrim("/G1"); g1.GetReferences().AddReference(os.path.relpath(SRC, os.path.dirname(OUT)).replace(os.sep, "/"))
stage.SetDefaultPrim(g1)
link = stage.OverridePrim(LINK)
for child in list(src.GetPrimAtPath(LINK + "/visuals").GetChildren()):  # drop the rubber hand: the tool replaces it
    if child.IsA(UsdGeom.Mesh):
        off = UsdGeom.Xformable(child).GetLocalTransformation().ExtractTranslation()
        if min(pt[0] for pt in UsdGeom.Mesh(child).GetPointsAttr().Get()) + off[0] > HAND_X_MIN:
            stage.OverridePrim(child.GetPath()).SetActive(False)
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

# mounting block: dark rectangular bracket from inside the wrist link into the plain slanted zone below the rivets
bk_mat = UsdShade.Material(stage.GetPrimAtPath("/G1/Looks/metal"))   # same grey metal as the robot links
INSERT_TOOL_PT = Gf.Vec3d(0.0126, 0.0, 0.03)          # on the slanted edge, below the rivets (tool frame)
BRACKET_Z_UP = 0.01                                    # bracket raised along wrist Z
_ins = INSERT_TOOL_PT * _R + TOOL_POS                  # -> wrist frame
BLOCK_X0, BLOCK_X1, BLOCK_Y, BLOCK_Z = 0.012, _ins[0] + 0.012, 0.020, 0.024   # x span, width (Y), height (Z)
bk = UsdGeom.Cube.Define(stage, tp.GetPath().GetParentPath().AppendChild("tool_bracket")); bk.CreateSizeAttr(1.0)
bx = UsdGeom.Xformable(bk.GetPrim())
bx.AddTranslateOp().Set(Gf.Vec3d((BLOCK_X0 + BLOCK_X1) / 2, 0.0, _ins[2] + BRACKET_Z_UP)); bx.AddScaleOp().Set(Gf.Vec3f(BLOCK_X1 - BLOCK_X0, BLOCK_Y, BLOCK_Z))
UsdShade.MaterialBindingAPI.Apply(bk.GetPrim()).Bind(bk_mat)

# fold tool mass into the link (parallel-axis on the diagonal inertia is small at 165 g; keep principal axes)
mass_src = UsdPhysics.MassAPI(src.GetPrimAtPath(LINK))
m0, c0 = mass_src.GetMassAttr().Get(), Gf.Vec3d(mass_src.GetCenterOfMassAttr().Get())
mass = UsdPhysics.MassAPI(stage.GetPrimAtPath(LINK))   # composed prim: authoring writes overrides into OUT
mt = UsdPhysics.MassAPI(tool.GetPrimAtPath("/Ascender")).GetMassAttr().Get()
ct = TOOL_ROT_D.Transform(Gf.Vec3d(UsdPhysics.MassAPI(tool.GetPrimAtPath("/Ascender")).GetCenterOfMassAttr().Get())) + TOOL_POS
c = (c0 * m0 + ct * mt) / (m0 + mt)
mass.GetMassAttr().Set(float(m0 + mt)); mass.GetCenterOfMassAttr().Set(Gf.Vec3f(c))
I0 = np.array(mass_src.GetDiagonalInertiaAttr().Get()); r = np.array(ct - c); I0 += mt * (r.dot(r) - r * r)
mass.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*map(float, I0)))
link.SetCustomDataByKey("tool", "ascender")

stage.GetRootLayer().Save()
print(f"wrote {OUT} (references g1/g1_himalaya.usd); {LINK} mass {m0:.3f} -> {m0 + mt:.3f} kg")

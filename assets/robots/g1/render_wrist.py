"""Render the right wrist/tool: python render_wrist.py ../g1_unitree_ascender.usd rig.usda && usdrecord --camera threeq --renderer Metal rig.usda out.png (Apple USD tools)"""
import sys, os
from pxr import Usd, UsdGeom, UsdLux, Gf, Sdf
robot=os.path.abspath(sys.argv[1]); out=sys.argv[2]
st=Usd.Stage.CreateNew(out); UsdGeom.SetStageUpAxis(st,UsdGeom.Tokens.z)
r=st.DefinePrim('/World/G1'); r.GetReferences().AddReference(robot)
rs=Usd.Stage.Open(robot); xc=UsdGeom.XformCache(); M=xc.GetLocalToWorldTransform(rs.GetPrimAtPath('/G1/right_wrist_yaw_link'))
target=M.ExtractTranslation()+M.TransformDir(Gf.Vec3d(0.12,0,0)); fwd=M.TransformDir(Gf.Vec3d(1,0,0)); side=M.TransformDir(Gf.Vec3d(0,1,0)); up=M.TransformDir(Gf.Vec3d(0,0,1))
print('wrist X(forearm) world', fwd, 'wrist Y', side, 'wrist Z', up)
dome=UsdLux.DomeLight.Define(st,'/World/dome'); dome.CreateIntensityAttr(1.0)
for name,eye_dir,upv in [('side_from_negY',-side,up),('top_from_posZ',up,-fwd),('front_from_posX',fwd,up),('threeq',Gf.Vec3d(-side+up*0.6+fwd*0.5).GetNormalized(),up)]:
    cam=UsdGeom.Camera.Define(st,'/World/'+name); cam.CreateFocalLengthAttr(50); cam.CreateClippingRangeAttr(Gf.Vec2f(0.01,10))
    eye=target+eye_dir*0.45
    m=Gf.Matrix4d().SetLookAt(eye,target,upv).GetInverse()
    UsdGeom.Xformable(cam.GetPrim()).AddTransformOp().Set(m)
st.GetRootLayer().Save()

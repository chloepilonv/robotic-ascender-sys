#!/usr/bin/env python3
"""3 m x 3 m Himalaya test pad for Isaac Sim / Isaac Lab (Z-up, metres, origin = pad centre).

Layout (top view, +X = "north"):
      +Y
   ┌─────────┬─────────┐
   │ 10° slope│  ICE    │   ice: 1x1 m, mu=0.10
   │  (snow)  │ 1x1 m   │   slopes: 1 m run each, packed snow
   ├─────────┼─────────┤   wall: 1 m long, 1 m high rock face
   │ 40° slope│  WALL   │   rest: packed snow, mu=0.50, few-mm fractal bumps
   └─────────┴─────────┘  -Y
  -X                    +X
Centre (0,0) is left clear for the robot spawn.

Usage: python build_terrain.py [--out himalaya_3m.usd]   (deps: usd-core numpy)
"""
import argparse, math, os
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Gf, Sdf, Vt

HERE = os.path.dirname(os.path.abspath(__file__))
SIZE = 3.0
# name: (rgb, roughness, static mu, dynamic mu, restitution)
MATERIALS = {
    "packed_snow": ((0.92, 0.94, 0.97), 0.9, 0.50, 0.45, 0.0),
    "ice":         ((0.75, 0.88, 1.00), 0.05, 0.10, 0.08, 0.0),
    "rock":        ((0.35, 0.33, 0.31), 0.8, 0.80, 0.75, 0.0),
}
SLOPE_10, SLOPE_40 = 10.0, 40.0

def material(stage, root, name):
    rgb, rough, mu_s, mu_d, rest = MATERIALS[name]
    path = root.AppendChild("Looks").AppendChild(name)
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path.AppendChild("Shader"))
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(mu_s); pm.CreateDynamicFrictionAttr(mu_d); pm.CreateRestitutionAttr(rest)
    return mat

def bind(prim, mat):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)
    UsdShade.MaterialBindingAPI(prim).Bind(mat, materialPurpose="physics")

def collide(prim, approx="none"):
    UsdPhysics.CollisionAPI.Apply(prim)
    if prim.IsA(UsdGeom.Mesh):
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(approx)

def fractal_noise(n, octaves=4, seed=0):
    """Cheap value-noise: sum of bilinearly upsampled random lattices (sub-cell offsets avoid banding)."""
    rng = np.random.default_rng(seed); out = np.zeros((n, n))
    for o in range(octaves):
        k = 4 * 2 ** o
        lat = rng.random((k + 2, k + 2))
        xs = np.linspace(0, k, n) + rng.random() ; ys = np.linspace(0, k, n) + rng.random()
        xi, yi = np.floor(xs).astype(int).clip(0, k), np.floor(ys).astype(int).clip(0, k)
        fx, fy = (xs - xi)[:, None], (ys - yi)[None, :]
        v = (lat[xi][:, yi] * (1 - fx) * (1 - fy) + lat[xi + 1][:, yi] * fx * (1 - fy)
             + lat[xi][:, yi + 1] * (1 - fx) * fy + lat[xi + 1][:, yi + 1] * fx * fy)
        out += (v - 0.5) / 2 ** o
    return out / np.abs(out).max()

def grid_mesh(stage, path, x0, x1, y0, y1, n, zfn):
    xs, ys = np.linspace(x0, x1, n), np.linspace(y0, y1, n)
    X, Y = np.meshgrid(xs, ys, indexing="ij"); Z = zfn(X, Y)
    pts = np.stack([X, Y, Z], -1).reshape(-1, 3)
    idx = np.arange(n * n).reshape(n, n)
    a, b, c, d = idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]
    faces = np.stack([a, b, c, a, c, d], -1).reshape(-1, 3)   # CCW seen from +Z
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
    m.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.astype(np.int32).ravel()))
    m.CreateSubdivisionSchemeAttr("none")
    m.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, pts.min(0))), Gf.Vec3f(*map(float, pts.max(0)))]))
    return m

def box(stage, path, center, size, rot_y_deg=0.0):
    c = UsdGeom.Cube.Define(stage, path); c.CreateSizeAttr(1.0)
    x = UsdGeom.Xformable(c.GetPrim())
    x.AddTranslateOp().Set(Gf.Vec3d(*center))
    x.AddRotateYOp().Set(rot_y_deg)
    x.AddScaleOp().Set(Gf.Vec3f(*size))
    return c

def build(out):
    stage = Usd.Stage.CreateNew(out)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = Sdf.Path("/Terrain"); rp = UsdGeom.Xform.Define(stage, root).GetPrim(); stage.SetDefaultPrim(rp)
    mats = {k: material(stage, root, k) for k in MATERIALS}
    h = SIZE / 2

    # 1) packed snow ground, 3x3 m, +/-4 mm fractal bumps (n=121 -> 2.5 cm cells)
    noise = fractal_noise(121, seed=7) * 0.004
    xs = np.linspace(-h, h, 121)
    ground = grid_mesh(stage, root.AppendChild("packed_snow"), -h, h, -h, h, 121,
                       lambda X, Y: noise[np.searchsorted(xs, X.ravel()).clip(0, 120).reshape(X.shape),
                                          np.searchsorted(xs, Y.ravel()).clip(0, 120).reshape(Y.shape)])
    collide(ground.GetPrim()); bind(ground.GetPrim(), mats["packed_snow"])

    # 2) ice patch 1x1 m, NE quadrant, 1 cm slab sitting on the snow (flat, glossy)
    ice = box(stage, root.AppendChild("ice"), (0.75, 0.75, 0.005), (1.0, 1.0, 0.01))
    collide(ice.GetPrim()); bind(ice.GetPrim(), mats["ice"])

    # 3) 10 deg slope, NW quadrant: 1 m run along +X (rising towards -X edge), 1 m wide. Wedge = rotated box
    def wedge(name, deg, cx, cy, run=1.0, width=1.0):
        rise = run * math.tan(math.radians(deg)); L = run / math.cos(math.radians(deg))
        # box of length L rotated so its top face is the slope; thick enough to reach below ground
        thick = 0.3
        # centre of top face at (cx, cy, rise/2); box centre offset by thick/2 along the face normal
        n = np.array([math.sin(math.radians(deg)), 0, math.cos(math.radians(deg))])
        c = np.array([cx, cy, rise / 2]) - n * thick / 2
        w = box(stage, root.AppendChild(name), tuple(c), (L, width, thick), rot_y_deg=deg)  # +Y rot tilts +X end down
        collide(w.GetPrim()); bind(w.GetPrim(), mats["packed_snow"]); return w
    wedge("slope_10deg", SLOPE_10, -0.75, 0.75)
    wedge("slope_40deg", SLOPE_40, -0.75, -0.75)

    # 4) rock wall, SE quadrant: 1 m long (along Y), 0.15 m thick, 1 m high, face at x=1.0
    wall = box(stage, root.AppendChild("wall"), (1.075, -0.75, 0.5), (0.15, 1.0, 1.0))
    collide(wall.GetPrim()); bind(wall.GetPrim(), mats["rock"])

    # spawn marker (no collision) so the robot start is obvious in the viewport
    sp = UsdGeom.Xform.Define(stage, root.AppendChild("robot_spawn")); UsdGeom.Xformable(sp).AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
    stage.GetRootLayer().Save(); return stage

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(HERE, "himalaya_3m.usd"))
    a = ap.parse_args(); build(a.out); print("wrote", a.out)

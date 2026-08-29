#!/usr/bin/env python3
"""Load a Lhotse Face patch into MuJoCo as a heightfield and open the viewer.

    python mujoco_scene.py --list
    python mujoco_scene.py                        # default: real patch B
    python mujoco_scene.py --patch A              # another REAL location
    python mujoco_scene.py --patch B_slope45      # CURRICULUM (slope overridden)
    python mujoco_scene.py --headless --seconds 6

WHY hfield AND NOT A MESH GEOM:
MuJoCo replaces mesh geoms with their CONVEX HULL for collision. A terrain patch
would become a solid wedge -- concavities gone, contact wrong, and it looks
correct in the viewer while colliding wrongly. `hfield` collides against the
actual per-cell triangles.

Elevation is pushed in as float32 via model.hfield_data rather than a PNG: an
8-bit PNG quantises a ~20 m span into 8 cm steps, coarser than the 12 cm RMS
roughness we are trying to represent.
"""
import argparse, glob, json, os
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("--patch", default="B",
                help="patch name; searched in patches/real then patches/curriculum")
ap.add_argument("--list", action="store_true", help="list patches and exit")
ap.add_argument("--headless", action="store_true")
ap.add_argument("--no-ball", action="store_true")
ap.add_argument("--seconds", type=float, default=6.0)
ap.add_argument("--rope-height", type=float, default=0.60,
                help="rope standoff above the terrain surface, metres")
a = ap.parse_args()


def find(name):
    for sub in ("real", "curriculum"):
        p = f"{HERE}/patches/{sub}/{name}.npz"
        if os.path.exists(p):
            return p, f"{HERE}/patches/{sub}/{name}.json", sub
    legacy = f"{HERE}/data/lhotse_face_{name}.npz"
    if os.path.exists(legacy):
        return legacy, f"{HERE}/data/lhotse_face_{name}.json", "legacy"
    raise SystemExit(f"no patch '{name}'. Try --list")


if a.list:
    print("REAL  (location AND slope measured from Copernicus GLO-30):")
    for f in sorted(glob.glob(f"{HERE}/patches/real/*.json")):
        m = json.load(open(f))
        print(f"  {m['name']:<14} {m['approx_altitude_m']} m  "
              f"{m['real_slope_deg']:.1f} deg  aspect {m['downhill_aspect_deg']:.0f}  "
              f"({m['lat']:.5f}, {m['lon']:.5f})")
    print("\nCURRICULUM  (location real, SLOPE OVERRIDDEN -- training aid, not "
          "a measurement):")
    for f in sorted(glob.glob(f"{HERE}/patches/curriculum/*.json"),
                    key=lambda p: json.load(open(p))["applied_slope_deg"]):
        m = json.load(open(f))
        print(f"  {m['name']:<14} applied {m['applied_slope_deg']:.1f} deg  "
              f"(real here: {m['real_slope_deg']:.1f} deg, "
              f"{m['applied_slope_deg']-m['real_slope_deg']:+.1f})")
    print("\nIn EVERY patch, all detail finer than ~30 m is synthetic.")
    raise SystemExit(0)

npz, jsn, cls = find(a.patch)
Z = np.load(npz)["Z"].astype(np.float64)
M = json.load(open(jsn))
ny, nx = Z.shape
res = M["resolution_m"]
LX, LY = nx*res, ny*res
zmin, zmax = float(Z.min()), float(Z.max())
zspan = zmax - zmin
RH = a.rope_height

slope = M.get("applied_slope_deg", M.get("mean_slope_deg"))
real_slope = M.get("real_slope_deg", slope)
print(f"patch  {M['name'] if 'name' in M else a.patch}  [{cls}]")
print(f"  {nx} x {ny} @ {res*100:.0f} cm = {LX:.2f} x {LY:.2f} m")
print(f"  altitude {M['approx_altitude_m']} m   Z span {zspan:.2f} m")
if M.get("slope_is_real", True):
    print(f"  slope {slope:.1f} deg  [REAL - measured from the DEM]")
else:
    print(f"  slope {slope:.1f} deg  [SYNTHETIC OVERRIDE - real here is "
          f"{real_slope:.2f} deg]")
    print(f"  !! training aid; not a measurement of Everest at this angle")
print(f"  rope standoff {RH:.2f} m")

# rope: use stored waypoints if present, else lay a line up the fall line
if "rope_route_xyz" in M:
    rope = np.array(M["rope_route_xyz"])
else:
    u = (np.arange(nx)-nx/2)*res
    v = (np.arange(ny)-ny/2)*res
    ru = np.linspace(u[0]+1.0, u[-1]-1.0, 9)
    rv = 0.35*np.sin(np.linspace(0, 2.2, 9))
    rope = np.array([[uu, vv, Z[int(np.clip((vv-v[0])/res, 0, ny-1)),
                                 int(np.clip((uu-u[0])/res, 0, nx-1))]]
                     for uu, vv in zip(ru, rv)])

Zn = ((Z - zmin)/zspan).astype(np.float32)
caps = "\n".join(
    f'    <geom name="ropeseg{i}" type="capsule" size="0.025" '
    f'fromto="{rope[i][0]:.4f} {rope[i][1]:.4f} {rope[i][2]+RH:.4f} '
    f'{rope[i+1][0]:.4f} {rope[i+1][1]:.4f} {rope[i+1][2]+RH:.4f}" '
    f'rgba="0.85 0.08 0.05 1" contype="0" conaffinity="0"/>'
    for i in range(len(rope)-1))
ball = "" if a.no_ball else f"""
    <body name="ball" pos="{rope[-1][0]:.3f} {rope[-1][1]:.3f} {rope[-1][2]+RH+0.8:.3f}">
      <freejoint/>
      <geom name="ball" type="sphere" size="0.25" mass="5"
            rgba="0.95 0.45 0.05 1" friction="0.9 0.02 0.001"/>
    </body>"""

XML = f"""
<mujoco model="lhotse_{a.patch}">
  <compiler angle="degree"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
  <visual>
    <headlight ambient="0.35 0.35 0.4" diffuse="0.7 0.7 0.7"/>
    <map znear="0.02" zfar="200"/><quality shadowsize="4096"/>
  </visual>
  <asset>
    <hfield name="patch" nrow="{ny}" ncol="{nx}"
            size="{LX/2:.4f} {LY/2:.4f} {zspan:.4f} 1.0"/>
    <texture name="sky" type="skybox" builtin="gradient"
             rgb1="0.25 0.4 0.65" rgb2="0.05 0.08 0.15" width="256" height="256"/>
    <material name="snow" rgba="0.82 0.86 0.93 1" specular="0.25" shininess="0.1"/>
  </asset>
  <worldbody>
    <light name="sun" directional="true" pos="-8 -12 25" dir="0.35 0.5 -1"
           diffuse="0.9 0.9 0.88" castshadow="true"/>
    <geom name="terrain" type="hfield" hfield="patch" material="snow"
          pos="0 0 {zmin:.4f}" friction="0.9 0.02 0.001"/>
{caps}
{ball}
  </worldbody>
</mujoco>
"""
open(f"{HERE}/lhotse_face.xml", "w").write(XML)
model = mujoco.MjModel.from_xml_string(XML)
model.hfield_data[:] = Zn.ravel()
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

if a.headless:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    if bid >= 0:
        p0 = data.xpos[bid].copy()
        for _ in range(int(a.seconds/model.opt.timestep)):
            mujoco.mj_step(model, data)
        p1 = data.xpos[bid]
        print(f"  ball fell {p0[2]-p1[2]:.2f} m, moved "
              f"{np.hypot(*(p1[:2]-p0[:2])):.2f} m   finite="
              f"{np.isfinite(data.qpos).all()}")
else:
    import mujoco.viewer
    mujoco.viewer.launch(model, data)

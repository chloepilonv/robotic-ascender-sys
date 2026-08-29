#!/usr/bin/env python3
"""Load the Lhotse Face patch into MuJoCo as a heightfield and open the viewer.

WHY hfield AND NOT THE OBJ:
MuJoCo replaces mesh geoms with their CONVEX HULL for collision. A terrain patch
would become a solid wedge -- concavities gone, contact wrong. `hfield` collides
against the actual per-cell triangles, which is what terrain needs.

The elevation is pushed in as float32 via model.hfield_data rather than a PNG:
an 8-bit PNG quantises this patch's 20.5 m span into 8 cm steps, which is
coarser than the 12 cm RMS roughness we are trying to represent.

Usage:
    python -m terrain.mujoco_scene [--headless] [--no-ball] [--rope-height 0.6]
"""
import argparse, json, os
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--headless", action="store_true")
ap.add_argument("--no-ball", action="store_true")
ap.add_argument("--seconds", type=float, default=6.0)
ap.add_argument("--rope-height", type=float, default=0.60,
                help="rope standoff above the terrain surface, metres")
a = ap.parse_args()

Z = np.load(f"{HERE}/data/lhotse_face_B.npz")["Z"].astype(np.float64)  # (ny,nx) m
M = json.load(open(f"{HERE}/data/lhotse_face_B.json"))
ny, nx = Z.shape
res = M["resolution_m"]
LX, LY = nx*res, ny*res                       # 25 x 15 m
zmin, zmax = float(Z.min()), float(Z.max())
zspan = zmax - zmin
print(f"patch  {nx} x {ny} @ {res*100:.0f} cm = {LX:.2f} x {LY:.2f} m")
print(f"Z      {zmin:.2f} .. {zmax:.2f} m  (span {zspan:.2f} m)")
print(f"slope  {M['mean_slope_deg']:.1f} deg   altitude {M['approx_altitude_m']} m")

# MuJoCo hfield data is normalised 0..1 and scaled by size[2].
Zn = ((Z - zmin)/zspan).astype(np.float32)
# hfield rows run +Y; our grid row 0 is -Y already, so no flip needed.

rope = np.array(M["rope_route_xyz"])
RH = a.rope_height   # standoff above the surface
print(f"rope   standoff {RH:.2f} m above terrain")
rope_sites = "\n".join(
    f'      <site name="rope{i}" pos="{x:.4f} {y:.4f} {z+RH:.4f}" '
    f'size="0.05" rgba="0.85 0.08 0.05 1"/>'
    for i,(x,y,z) in enumerate(rope))
# thin capsules between consecutive waypoints = the future fixed rope
rope_caps = "\n".join(
    f'      <geom name="ropeseg{i}" type="capsule" size="0.025" '
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
<mujoco model="lhotse_face">
  <compiler angle="degree"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
  <visual>
    <headlight ambient="0.35 0.35 0.4" diffuse="0.7 0.7 0.7"/>
    <map znear="0.02" zfar="200"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>
    <hfield name="lhotse" nrow="{ny}" ncol="{nx}"
            size="{LX/2:.4f} {LY/2:.4f} {zspan:.4f} 1.0"/>
    <texture name="sky" type="skybox" builtin="gradient"
             rgb1="0.25 0.4 0.65" rgb2="0.05 0.08 0.15" width="256" height="256"/>
    <material name="snow" rgba="0.82 0.86 0.93 1" specular="0.25" shininess="0.1"/>
  </asset>
  <worldbody>
    <light name="sun" directional="true" pos="-8 -12 25" dir="0.35 0.5 -1"
           diffuse="0.9 0.9 0.88" specular="0.2 0.2 0.2" castshadow="true"/>
    <geom name="terrain" type="hfield" hfield="lhotse" material="snow"
          pos="0 0 {zmin:.4f}" friction="0.9 0.02 0.001"/>
{rope_caps}
{rope_sites}
{ball}
  </worldbody>
</mujoco>
"""
xml_path = f"{HERE}/lhotse_face.xml"
open(xml_path, "w").write(XML)
print(f"wrote {xml_path}")

model = mujoco.MjModel.from_xml_string(XML)
# push full-precision elevation in (bypasses 8-bit PNG quantisation)
model.hfield_data[:] = Zn.ravel()
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
print(f"model: {model.ngeom} geoms, hfield {model.hfield_nrow[0]}x{model.hfield_ncol[0]}")

if a.headless:
    n = int(a.seconds/model.opt.timestep)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    if bid >= 0:
        p0 = data.xpos[bid].copy()
        for _ in range(n):
            mujoco.mj_step(model, data)
        p1 = data.xpos[bid]
        print(f"ball {p0} -> {p1}")
        print(f"  fell {p0[2]-p1[2]:.3f} m, moved {np.hypot(*(p1[:2]-p0[:2])):.3f} m horizontally")
        print(f"  finite: {np.isfinite(data.qpos).all()}")
else:
    import mujoco.viewer
    mujoco.viewer.launch(model, data)

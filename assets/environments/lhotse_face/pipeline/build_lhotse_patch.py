#!/usr/bin/env python3
"""Build the robot-scale Lhotse Face patch.

REAL      : geographic location, macro slope, aspect, elevation profile
            (Copernicus GLO-30, ~30 m posting)
SYNTHETIC : all detail finer than ~30 m -- no such data exists for Everest

The macro surface is a plane fitted to the real DEM neighbourhood, so the real
slope and aspect are preserved exactly. Synthetic roughness is band-limited and
conservative, and the script reports slope before/after so the added detail can
be shown not to have changed the underlying incline.
"""
import numpy as np, json, rasterio, argparse
from rasterio.windows import Window

ap = argparse.ArgumentParser()
ap.add_argument("--candidate", default="B")
ap.add_argument("--length", type=float, default=25.0)   # along-slope, m
ap.add_argument("--width",  type=float, default=15.0)   # cross-slope, m
ap.add_argument("--res",    type=float, default=0.05)   # m per sample
ap.add_argument("--rough-rms", type=float, default=0.12, help="metres RMS")
ap.add_argument("--seed", type=int, default=20260829)
a = ap.parse_args()

DEM = "data/raw/Copernicus_DSM_30m_N27_E086.tif"
cands = {c["name"]: c for c in json.load(open("data/processed/everest/candidates.json"))}
C = cands[a.candidate]
lat0, lon0 = C["lat"], C["lon"]
M_LAT, M_LON = 110574.0, 111320.0*np.cos(np.radians(lat0))

# --- real macro geometry from the DEM -------------------------------------
with rasterio.open(DEM) as ds:
    r0, c0 = ds.index(lon0, lat0)
    h = 4                                    # +-4 cells ~ +-120 m
    A = ds.read(1, window=Window(c0-h, r0-h, 2*h+1, 2*h+1)).astype(np.float64)
    dx_m = abs(ds.transform.a)*M_LON
    dy_m = abs(ds.transform.e)*M_LAT

n = 2*h+1
xs = (np.arange(n)-h)*dx_m
ys = -(np.arange(n)-h)*dy_m           # +Y north
X, Y = np.meshgrid(xs, ys)
Amat = np.column_stack([X.ravel(), Y.ravel(), np.ones(X.size)])
coef, *_ = np.linalg.lstsq(Amat, A.ravel(), rcond=None)
gx, gy, z0 = coef
slope_real = np.degrees(np.arctan(np.hypot(gx, gy)))
aspect_real = (np.degrees(np.arctan2(-gx, -gy)) + 360) % 360   # downhill azimuth
print("=== REAL macro geometry (Copernicus GLO-30) ===")
print(f"  location        : {lat0:.5f} N, {lon0:.5f} E")
print(f"  elevation       : {z0:.0f} m")
print(f"  fitted slope    : {slope_real:.2f} deg   (plane over "
      f"{(2*h+1)*dx_m:.0f} x {(2*h+1)*dy_m:.0f} m)")
print(f"  downhill aspect : {aspect_real:.1f} deg from north")
print(f"  plane residual  : {np.sqrt(((Amat@coef - A.ravel())**2).mean()):.2f} m RMS")

# --- local frame: +X uphill along the fall line, +Y cross-slope ------------
dh = np.array([gx, gy]); dh /= np.linalg.norm(dh)     # uphill direction (grad)
nx_, ny_ = int(round(a.length/a.res)), int(round(a.width/a.res))
u = (np.arange(nx_)-nx_/2)*a.res                       # along-slope
v = (np.arange(ny_)-ny_/2)*a.res                       # cross-slope
U, V = np.meshgrid(u, v)
slope_rad = np.radians(slope_real)
Zmacro = U*np.tan(slope_rad)                           # real incline, exact

# --- synthetic band-limited roughness -------------------------------------
rng = np.random.default_rng(a.seed)
noise = rng.normal(size=(ny_, nx_))
from scipy.ndimage import gaussian_filter
octaves = [(0.6, 2.0), (0.3, 0.6), (0.1, 0.2)]         # (weight, corr length m)
rough = np.zeros_like(noise)
for wgt, corr in octaves:
    sm = gaussian_filter(rng.normal(size=(ny_, nx_)), corr/a.res, mode="reflect")
    sm /= (sm.std() + 1e-12)
    rough += wgt*sm
rough /= (rough.std() + 1e-12)
rough *= a.rough_rms
Z = Zmacro + rough

# --- verify the synthetic detail did not change the real slope ------------
def mean_slope(Zg, res):
    gyy, gxx = np.gradient(Zg, res, res)
    return float(np.degrees(np.arctan(np.hypot(gxx, gyy))).mean())
s_macro, s_final = mean_slope(Zmacro, a.res), mean_slope(Z, a.res)
# planar slope of the final surface (the thing that must not drift)
Am2 = np.column_stack([U.ravel(), V.ravel(), np.ones(U.size)])
cf2, *_ = np.linalg.lstsq(Am2, Z.ravel(), rcond=None)
s_plane = np.degrees(np.arctan(np.hypot(cf2[0], cf2[1])))
print("\n=== SYNTHETIC roughness ===")
print(f"  target RMS      : {a.rough_rms:.3f} m   actual {rough.std():.3f} m")
print(f"  amplitude range : {rough.min():+.3f} .. {rough.max():+.3f} m")
print(f"  correlation     : octaves at 2.0 / 0.6 / 0.2 m")
print(f"  seed            : {a.seed}  (reproducible)")
print(f"\n  planar slope  real {slope_real:.2f} -> final {s_plane:.2f} deg "
      f"(drift {s_plane-slope_real:+.3f})")
print(f"  cell-wise mean slope  macro {s_macro:.2f} -> final {s_final:.2f} deg")
print(f"    (cell-wise rises because roughness adds sub-metre relief; the"
      f" underlying incline is unchanged -- see planar slope above)")

# --- mesh + OBJ ------------------------------------------------------------
Vx, Vy, Vz = U.ravel(), V.ravel(), Z.ravel()
Vt = np.column_stack([Vx, Vy, Vz])
idx = np.arange(nx_*ny_).reshape(ny_, nx_)
tl = idx[:-1,:-1].ravel(); tr = idx[:-1,1:].ravel()
bl = idx[1:,:-1].ravel();  br = idx[1:,1:].ravel()
F = np.vstack([np.column_stack([tl, bl, br]), np.column_stack([tl, br, tr])])
print(f"\n=== MESH ===")
print(f"  samples    : {nx_} x {ny_}  @ {a.res*100:.0f} cm")
print(f"  vertices   : {len(Vt):,}   triangles : {len(F):,}")
print(f"  dimensions : {Vx.ptp() if hasattr(Vx,'ptp') else np.ptp(Vx):.2f} x "
      f"{np.ptp(Vy):.2f} x {np.ptp(Vz):.2f} m")
print(f"  vertical gain over patch : {np.ptp(Zmacro):.2f} m")

import os
# write next to this pipeline's environment, not a top-level assets/terrain
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.makedirs(OUTDIR, exist_ok=True)
obj = os.path.join(OUTDIR, f"lhotse_face_{a.candidate}.obj")
with open(obj, "w") as f:
    f.write(f"# Lhotse Face patch, candidate {a.candidate}\n")
    f.write(f"# REAL: location/slope/aspect from Copernicus GLO-30\n")
    f.write(f"# SYNTHETIC: all detail below ~30 m\n")
    f.write(f"# local frame: +X uphill along fall line, +Y cross-slope, +Z up, metres\n")
    np.savetxt(f, Vt, fmt="v %.5f %.5f %.5f")
    np.savetxt(f, F+1, fmt="f %d %d %d")
print(f"  wrote      : {obj} ({os.path.getsize(obj)/1e6:.1f} MB)")

# --- rope route waypoints (NOT rope physics) ------------------------------
rope_u = np.linspace(u[0]+1.0, u[-1]-1.0, 9)
rope_v = 0.35*np.sin(np.linspace(0, 2.2, 9))          # gentle real-world weave
rope = []
for uu, vv in zip(rope_u, rope_v):
    i = int(np.clip((uu-u[0])/a.res, 0, nx_-1))
    j = int(np.clip((vv-v[0])/a.res, 0, ny_-1))
    rope.append([float(uu), float(vv), float(Z[j, i])])   # raycast = grid lookup
rope = np.array(rope)
print(f"\n=== ROPE ROUTE (waypoints only, no physics) ===")
print(f"  {len(rope)} waypoints, {np.hypot(*(rope[-1,:2]-rope[0,:2])):.1f} m ground run, "
      f"{rope[-1,2]-rope[0,2]:+.2f} m gain")

meta = dict(
    source="Copernicus GLO-30 DEM (ESA), tile N27 E086",
    source_url="https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/",
    route="Everest South Col / Lhotse Face, Camp II -> Camp III",
    route_source="OpenStreetMap nodes (Camp 2S, Camp 3S, Camp 4S South Col)",
    route_registration_accuracy=(
        "Level 2 - real georeferenced DEM + real OSM camp coordinates; "
        "DEM validates within 8-23 m of OSM camp elevations. NOT a recorded "
        "GPS track (none found public); patch is on the correct face, at the "
        "correct altitude band, on the straight C2->C3 line."),
    candidate=a.candidate,
    lat=lat0, lon=lon0, approx_altitude_m=round(float(z0)),
    width_m=a.width, length_m=a.length,
    mean_slope_deg=round(float(slope_real), 2),
    downhill_aspect_deg=round(float(aspect_real), 1),
    vertical_gain_m=round(float(np.ptp(Zmacro)), 2),
    geometry_source="macro: real DEM plane fit; micro: synthetic",
    synthetic_refinement=True,
    synthetic_rms_m=a.rough_rms, synthetic_seed=a.seed,
    synthetic_octaves_m=[2.0, 0.6, 0.2],
    native_dem_resolution_m=[round(dx_m,2), round(dy_m,2)],
    real_dem_cells_in_patch=round((a.length/dx_m)*(a.width/dy_m), 3),
    resolution_m=a.res, vertices=int(len(Vt)), triangles=int(len(F)),
    rope_route_xyz=rope.tolist(),
)
json.dump(meta, open(os.path.join(OUTDIR, f"lhotse_face_{a.candidate}.json"),"w"), indent=2)
np.save(f"data/processed/everest/patch_{a.candidate}_Z.npy", Z)
print(f"  wrote      : {OUTDIR}/lhotse_face_{a.candidate}.json")

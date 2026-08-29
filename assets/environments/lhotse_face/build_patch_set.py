#!/usr/bin/env python3
"""Generate the training patch set.

TWO CLASSES, kept in separate directories because they are NOT equivalent
evidence about Everest:

  patches/real/        location AND macro slope both measured from the DEM.
                       Four distinct points along the real Lhotse Face.

  patches/curriculum/  location is real, but the SLOPE IS OVERRIDDEN to a
                       chosen value. These are training aids, not terrain
                       observations. Do not describe them as "the Lhotse Face
                       at N degrees" -- the Lhotse Face is not N degrees there.

In BOTH classes every detail finer than ~30 m is synthetic: GLO-30 cells here
are 27.3 x 30.7 m, so a 25 x 15 m patch is 0.36 of one cell. No public
sub-metre Everest terrain exists.

    python build_patch_set.py            # needs data/Copernicus_*.tif
    python fetch_data.py                 # to get it
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
DEM = os.path.join(HERE, "data", "Copernicus_DSM_30m_N27_E086.tif")
M_LAT = 110574.0

ap = argparse.ArgumentParser()
ap.add_argument("--length", type=float, default=25.0)
ap.add_argument("--width", type=float, default=15.0)
ap.add_argument("--res", type=float, default=0.05)
ap.add_argument("--rough-rms", type=float, default=0.12)
ap.add_argument("--n-real", type=int, default=4)
ap.add_argument("--curriculum-slopes", type=float, nargs="+",
                default=[25.0, 30.0, 35.0, 45.0, 50.0])
ap.add_argument("--base", default="B", help="which real patch the curriculum varies")
a = ap.parse_args()

R = json.load(open(f"{HERE}/data/route.json"))
CAMPS = R["camps"]
c2, c3 = CAMPS["Camp 2S"], CAMPS["Camp 3S"]

if not os.path.exists(DEM):
    raise SystemExit(f"missing {DEM}\nrun: python fetch_data.py")

ds = rasterio.open(DEM)
M_LON = 111320.0*np.cos(np.radians(27.975))
dx_m, dy_m = abs(ds.transform.a)*M_LON, abs(ds.transform.e)*M_LAT


def sample_slope(lat, lon, half=4):
    """Plane fit over a +-half-cell neighbourhood -> real slope, aspect, elev."""
    r0, c0 = ds.index(lon, lat)
    A = ds.read(1, window=Window(c0-half, r0-half, 2*half+1, 2*half+1)).astype(float)
    n = 2*half+1
    xs = (np.arange(n)-half)*dx_m
    ys = -(np.arange(n)-half)*dy_m
    X, Y = np.meshgrid(xs, ys)
    Am = np.column_stack([X.ravel(), Y.ravel(), np.ones(X.size)])
    coef, *_ = np.linalg.lstsq(Am, A.ravel(), rcond=None)
    gx, gy, z0 = coef
    return (float(np.degrees(np.arctan(np.hypot(gx, gy)))),
            float((np.degrees(np.arctan2(-gx, -gy)) + 360) % 360),
            float(z0),
            float(np.sqrt(((Am@coef - A.ravel())**2).mean())))


# ---- locate the steep band along the real C2 -> C3 line -------------------
prof = []
for k in range(121):
    f = k/120
    lat = c2[0] + f*(c3[0]-c2[0]); lon = c2[1] + f*(c3[1]-c2[1])
    s, asp, z, resid = sample_slope(lat, lon)
    prof.append(dict(frac=f, lat=lat, lon=lon, slope=s, aspect=asp,
                     elev=z, resid=resid))
steep = [p for p in prof if p["slope"] >= 28.0]
print(f"steep band (>=28 deg): frac {steep[0]['frac']:.2f}-{steep[-1]['frac']:.2f}, "
      f"elev {steep[0]['elev']:.0f}-{steep[-1]['elev']:.0f} m, {len(steep)} samples")

picks = [steep[int(round(f*(len(steep)-1)))]
         for f in np.linspace(0.08, 0.95, a.n_real)]
NAMES = "ABCDEFGH"


def roughness(ny, nx, res, rms, seed):
    rng = np.random.default_rng(seed)
    out = np.zeros((ny, nx))
    for wgt, corr in ((0.6, 2.0), (0.3, 0.6), (0.1, 0.2)):
        sm = gaussian_filter(rng.normal(size=(ny, nx)), corr/res, mode="reflect")
        out += wgt*sm/(sm.std()+1e-12)
    return out/(out.std()+1e-12)*rms


def build(name, lat, lon, real_slope, aspect, elev, resid, applied_slope,
          seed, outdir, slope_is_real):
    nx_, ny_ = int(round(a.length/a.res)), int(round(a.width/a.res))
    u = (np.arange(nx_)-nx_/2)*a.res
    v = (np.arange(ny_)-ny_/2)*a.res
    U, _ = np.meshgrid(u, v)
    Zmacro = U*np.tan(np.radians(applied_slope))
    rough = roughness(ny_, nx_, a.res, a.rough_rms, seed)
    Z = Zmacro + rough
    Am = np.column_stack([U.ravel(), np.meshgrid(u, v)[1].ravel(), np.ones(U.size)])
    cf, *_ = np.linalg.lstsq(Am, Z.ravel(), rcond=None)
    planar = float(np.degrees(np.arctan(np.hypot(cf[0], cf[1]))))
    os.makedirs(outdir, exist_ok=True)
    np.savez_compressed(f"{outdir}/{name}.npz", Z=Z.astype(np.float32))
    meta = dict(
        name=name,
        geometry_class=("real_location_real_slope" if slope_is_real
                        else "real_location_SYNTHETIC_slope"),
        slope_is_real=bool(slope_is_real),
        location_is_real=True,
        lat=lat, lon=lon, approx_altitude_m=round(elev),
        real_slope_deg=round(real_slope, 2),
        applied_slope_deg=round(applied_slope, 2),
        planar_slope_after_roughness_deg=round(planar, 2),
        downhill_aspect_deg=round(aspect, 1),
        plane_fit_residual_m=round(resid, 2),
        length_m=a.length, width_m=a.width, resolution_m=a.res,
        vertical_gain_m=round(float(np.ptp(Zmacro)), 2),
        synthetic_refinement=True, synthetic_rms_m=a.rough_rms,
        synthetic_seed=seed, synthetic_octaves_m=[2.0, 0.6, 0.2],
        native_dem_resolution_m=[round(dx_m, 2), round(dy_m, 2)],
        real_dem_cells_in_patch=round((a.length/dx_m)*(a.width/dy_m), 3),
        source="Copernicus GLO-30 (ESA) + OpenStreetMap camp nodes",
    )
    if not slope_is_real:
        meta["WARNING"] = (
            f"Slope overridden to {applied_slope:.1f} deg. The real slope at "
            f"this location is {real_slope:.2f} deg. Training aid only -- do "
            f"not present as measured Everest terrain.")
    json.dump(meta, open(f"{outdir}/{name}.json", "w"), indent=2)
    return meta


manifest = {"real": [], "curriculum": []}
print(f"\n=== REAL patches (location AND slope from the DEM) ===")
print(f"{'name':>6} {'lat':>10} {'lon':>10} {'alt m':>7} {'slope':>7} "
      f"{'aspect':>7} {'resid m':>8}")
base_meta = None
for i, p in enumerate(picks):
    nm = NAMES[i]
    s, asp, z, resid = p["slope"], p["aspect"], p["elev"], p["resid"]
    m = build(nm, p["lat"], p["lon"], s, asp, z, resid, s,
              20260829+i, f"{HERE}/patches/real", True)
    manifest["real"].append(m)
    if nm == a.base:
        base_meta = (p, s, asp, z, resid)
    print(f"{nm:>6} {p['lat']:10.5f} {p['lon']:10.5f} {z:7.0f} {s:6.1f}d "
          f"{asp:6.1f}d {resid:8.2f}")

p, s_real, asp, z, resid = base_meta
print(f"\n=== CURRICULUM patches (location {a.base}, SLOPE OVERRIDDEN) ===")
print(f"  real slope at this location: {s_real:.2f} deg")
print(f"{'name':>16} {'applied':>8} {'delta vs real':>14}")
for j, sl in enumerate(a.curriculum_slopes):
    nm = f"{a.base}_slope{int(round(sl))}"
    m = build(nm, p["lat"], p["lon"], s_real, asp, z, resid, sl,
              20270101+j, f"{HERE}/patches/curriculum", False)
    manifest["curriculum"].append(m)
    print(f"{nm:>16} {sl:7.1f}d {sl-s_real:+13.2f}d")

manifest["notes"] = {
    "real": "location and macro slope both measured from Copernicus GLO-30.",
    "curriculum": ("location real; slope OVERRIDDEN for training. Not a "
                   "measurement of Everest at that angle."),
    "both": ("all detail finer than ~30 m is synthetic in every patch; a "
             "25x15 m patch is 0.36 of one GLO-30 cell."),
}
json.dump(manifest, open(f"{HERE}/patches/manifest.json", "w"), indent=2)
print(f"\nwrote patches/manifest.json  "
      f"({len(manifest['real'])} real, {len(manifest['curriculum'])} curriculum)")
ds.close()

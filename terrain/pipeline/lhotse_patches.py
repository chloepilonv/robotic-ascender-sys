#!/usr/bin/env python3
"""Pick candidate ~20 m Lhotse Face patches along the Camp II -> Camp III route."""
import numpy as np, json, rasterio
from rasterio.windows import Window

DEM = "data/raw/Copernicus_DSM_30m_N27_E086.tif"
R = json.load(open("data/processed/everest/route.json"))
CAMPS = R["camps"]
PF = json.load(open("data/processed/everest/face_profile.json"))
prof = PF["profile"]

lat0 = 27.975
M_LAT, M_LON = 110574.0, 111320.0*np.cos(np.radians(lat0))

with rasterio.open(DEM) as ds:
    T0 = ds.transform
    dlat, dlon = abs(T0.e), abs(T0.a)
    dy_m, dx_m = dlat*M_LAT, dlon*M_LON
    print("=== SOURCE RESOLUTION REALITY CHECK (deliverable 8/9) ===")
    print(f"  Copernicus DEM cell : {dx_m:.2f} m (lon) x {dy_m:.2f} m (lat)")
    for L in (10, 20, 30):
        print(f"  a {L}x{L} m patch    : {L/dx_m:.2f} x {L/dy_m:.2f} cells "
              f"= {(L/dx_m)*(L/dy_m):.2f} cells, ~{2*(L/dx_m)*(L/dy_m):.2f} triangles")

    # steep section of the route
    steep = [p for p in prof if p["slope"] >= 25.0]
    print(f"\n  route samples with slope >= 25 deg: {len(steep)}/{len(prof)}"
          f"  (frac {steep[0]['frac']:.2f}..{steep[-1]['frac']:.2f},"
          f" elev {steep[0]['elev']:.0f}..{steep[-1]['elev']:.0f} m)")

    # three candidates spread across the steep band
    picks = [steep[int(f*(len(steep)-1))] for f in (0.15, 0.50, 0.90)]
    cands = []
    for k, p in enumerate(picks):
        name = "ABC"[k]
        lat, lon = p["lat"], p["lon"]
        # 3x3 cell neighbourhood (~90 m) for local statistics
        r0, c0 = ds.index(lon, lat)
        half = 2
        A = ds.read(1, window=Window(c0-half, r0-half, 2*half+1, 2*half+1)).astype(float)
        gy, gx = np.gradient(A, dy_m, dx_m)
        sl = np.degrees(np.arctan(np.hypot(gx, gy)))
        d_route = p["dist"]
        cands.append(dict(
            name=name, lat=lat, lon=lon, elev=p["elev"],
            slope_center=p["slope"], slope_mean=float(sl.mean()),
            slope_min=float(sl.min()), slope_max=float(sl.max()),
            dist_from_C2_m=d_route,
            frac=p["frac"],
            neighbourhood_m=(2*half+1)*dx_m,
            elev_range_local=float(A.max()-A.min()),
        ))

print("\n=== CANDIDATE PATCHES (Lhotse Face, Camp II -> Camp III) ===")
for c in cands:
    L = 20.0
    vgain = L*np.tan(np.radians(c["slope_mean"]))
    print(f"""
Candidate {c['name']}
  approx geographic area  : {c['lat']:.5f} N, {c['lon']:.5f} E
  position along route    : {c['dist_from_C2_m']:.0f} m from Camp 2S "
                            (frac {c['frac']:.2f} of the C2->C3 line)
  approx altitude         : {c['elev']:.0f} m
  patch dimensions        : 20 x 15 m (requested)
  mean slope              : {c['slope_mean']:.1f} deg
  slope range             : {c['slope_min']:.1f} - {c['slope_max']:.1f} deg
  vertical gain over 20 m : {vgain:.1f} m
  DEM cells in patch      : {(20/27.31)*(15/30.71):.2f}
  DEM triangles in patch  : {2*(20/27.31)*(15/30.71):.2f}
  est. geometric resolution: 27-31 m (Copernicus GLO-30)
  issues                  : patch is SMALLER THAN ONE DEM CELL -->
                            no real sub-30 m geometry exists here""")

json.dump(cands, open("data/processed/everest/candidates.json","w"), indent=2)
print("\nwrote candidates.json")

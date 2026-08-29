#!/usr/bin/env python3
"""Locate the Lhotse Face (Camp II -> Camp III) on the Copernicus 30 m DEM.

Works on the DEM's native EPSG:4326 grid and converts to metres with a local
equirectangular scale (accurate to <0.1% over this ~12 km area). This avoids a
reprojection step that silently misplaced the data.
"""
import numpy as np, json, rasterio
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

DEM = "data/raw/Copernicus_DSM_30m_N27_E086.tif"
R = json.load(open("data/processed/everest/route.json"))
CAMPS, PEAKS = R["camps"], R["peaks"]
ORDER = ["Base Camp","Camp 1S","Camp 2S","Camp 3S","Camp 4S (South Col)"]

lons = [c[1] for c in list(CAMPS.values())+list(PEAKS.values())]
lats = [c[0] for c in list(CAMPS.values())+list(PEAKS.values())]
pad = 0.015
with rasterio.open(DEM) as ds:
    win = rasterio.windows.from_bounds(min(lons)-pad, min(lats)-pad,
                                       max(lons)+pad, max(lats)+pad, ds.transform)
    # from_bounds returns a FRACTIONAL window; read() snaps to integer pixels
    # while window_transform() does not. Round first or the two disagree, which
    # in 60-degree terrain is a several-hundred-metre elevation error.
    win = win.round_offsets(op="floor").round_lengths(op="ceil")
    # Clamp to the tile. Base Camp sits at 27.9997 N and the pad pushes the
    # window past this tile's 28.0 N edge; read() then clips the array while
    # window_transform() keeps the out-of-bounds origin, shifting everything
    # by the overhang (53 rows here).
    win = win.intersection(rasterio.windows.Window(0, 0, ds.width, ds.height))
    Z = ds.read(1, window=win).astype(np.float64)
    T = ds.window_transform(win)
h, w = Z.shape
lat0 = (min(lats)+max(lats))/2
M_LAT = 110574.0
M_LON = 111320.0*np.cos(np.radians(lat0))
dlat, dlon = abs(T.e), abs(T.a)
dy_m, dx_m = dlat*M_LAT, dlon*M_LON
print(f"window {w} x {h}  cell {dx_m:.2f} m (lon) x {dy_m:.2f} m (lat)")
print(f"Z {Z.min():.0f} .. {Z.max():.0f} m   zeros {100*(Z==0).mean():.2f}%")

def ll_to_px(lat, lon):
    """Fractional pixel coords (for plotting). Integers land on pixel CORNERS."""
    c, r = ~T * (lon, lat)
    return c, r


def sample(lat, lon, A):
    """Sample a raster at lat/lon. MUST floor, not round: ~T puts integers on
    pixel corners, so rounding picks the neighbouring cell -- which on the
    Lhotse Face is a several-hundred-metre elevation error."""
    c, r = ~T * (lon, lat)
    j, i = int(np.floor(r)), int(np.floor(c))
    if 0 <= i < A.shape[1] and 0 <= j < A.shape[0]:
        return float(A[j, i])
    return float("nan")

# validate indexing against known elevations
print("\n=== indexing validation ===")
print(f"  {'feature':22} {'expect':>7} {'DEM':>7} {'diff':>6}")
for k in ORDER:
    lat, lon, ele = CAMPS[k]
    v = sample(lat, lon, Z)
    print(f"  {k:22} {ele:7.0f} {v:7.0f} {v-ele:+6.0f}")

gy, gx = np.gradient(Z, dy_m, dx_m)
slope = np.degrees(np.arctan(np.hypot(gx, gy)))
print(f"\nslope mean {slope.mean():.1f}  p95 {np.percentile(slope,95):.1f} deg")

# --- profile along the straight C2 -> C3 line ------------------------------
c2, c3 = CAMPS["Camp 2S"], CAMPS["Camp 3S"]
n = 80
print("\n=== Camp 2S -> Camp 3S profile ===")
print(f"{'frac':>5} {'dist m':>7} {'elev m':>7} {'slope':>7}")
tot = np.hypot((c3[1]-c2[1])*M_LON, (c3[0]-c2[0])*M_LAT)
prof = []
for k in range(n+1):
    f = k/n
    lat = c2[0]+f*(c3[0]-c2[0]); lon = c2[1]+f*(c3[1]-c2[1])
    e_, s_ = sample(lat, lon, Z), sample(lat, lon, slope)
    if np.isfinite(e_):
        prof.append(dict(frac=f, dist=f*tot, lat=lat, lon=lon,
                         elev=e_, slope=s_))
for p in prof[::8]:
    print(f"{p['frac']:5.2f} {p['dist']:7.0f} {p['elev']:7.0f} {p['slope']:7.1f}")
print(f"\ntotal C2->C3 straight-line distance: {tot:.0f} m")
json.dump(dict(profile=prof, dx_m=dx_m, dy_m=dy_m, tot=tot),
          open("data/processed/everest/face_profile.json","w"), indent=2)
np.save("data/processed/everest/dem_window.npy", Z)
json.dump(dict(transform=list(T)[:6], w=w, h=h, dx_m=dx_m, dy_m=dy_m),
          open("data/processed/everest/dem_window_meta.json","w"), indent=2)

# --- renders ---------------------------------------------------------------
def hillshade(Z, dx, dy, az=315, alt=45):
    gy, gx = np.gradient(Z, dy, dx)
    sl = np.arctan(np.hypot(gx, gy)); asp = np.arctan2(-gx, gy)
    a, e = np.radians(az), np.radians(alt)
    return np.sin(e)*np.cos(sl)+np.cos(e)*np.sin(sl)*np.cos(a-asp)
HS = hillshade(Z, dx_m, dy_m)
STROKE = [pe.withStroke(linewidth=2.5, foreground="black")]

def draw(ax, title, zoom=None):
    ax.imshow(HS, cmap="gray", origin="upper")
    ax.imshow(np.ma.masked_outside(slope, 30, 50), cmap="autumn", alpha=0.45,
              origin="upper", vmin=30, vmax=50)
    pts = [ll_to_px(CAMPS[k][0], CAMPS[k][1]) for k in ORDER]
    ax.plot([p[0] for p in pts],[p[1] for p in pts],"-o",color="deepskyblue",
            lw=2.4, ms=7, mec="black", label="South Col route (OSM camps)")
    for k,p in zip(ORDER,pts):
        ax.text(p[0]+3,p[1]-3,k.replace(" (South Col)",""),color="white",
                fontsize=9,weight="bold",path_effects=STROKE)
    for nm,(la,lo,el) in PEAKS.items():
        c,r = ll_to_px(la,lo); ax.plot(c,r,"r^",ms=10,mec="black")
        ax.text(c+3,r+9,f"{nm} {el:.0f}m",color="red",fontsize=9,
                weight="bold",path_effects=STROKE)
    ax.set_title(title, fontsize=12)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    if zoom: ax.set_xlim(zoom[0],zoom[1]); ax.set_ylim(zoom[3],zoom[2])

fig,ax = plt.subplots(figsize=(13,13*h/w))
draw(ax,"VIEW A - Everest South Col route, Copernicus 30 m DEM\n"
        "orange = slope 30-50 deg (fixed-rope terrain)")
plt.tight_layout(); plt.savefig("data/processed/everest/viewA_route.png",dpi=115)

p2 = ll_to_px(*c2[:2]); p3 = ll_to_px(*c3[:2]); m_=25
fig,ax = plt.subplots(figsize=(12,9))
draw(ax,"VIEW B - Camp II -> Lhotse Face -> Camp III",
     (min(p2[0],p3[0])-m_, max(p2[0],p3[0])+m_,
      min(p2[1],p3[1])-m_, max(p2[1],p3[1])+m_))
plt.tight_layout(); plt.savefig("data/processed/everest/viewB_lhotse.png",dpi=125)
print("wrote viewA_route.png + viewB_lhotse.png")

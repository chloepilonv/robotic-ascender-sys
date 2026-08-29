#!/usr/bin/env python3
"""Locate the Lhotse Face (Camp II -> Camp III) on the georeferenced Copernicus DEM."""
import numpy as np, json, rasterio
from rasterio.warp import transform as warp_transform
from pyproj import Transformer

DEM = "data/raw/Copernicus_DSM_30m_N27_E086.tif"
UTM = "EPSG:32645"          # UTM 45N covers 84-90E

CAMPS = {
 "Base Camp": (27.99966, 86.84879, 5364),
 "Camp 1S":   (27.98642, 86.87652, 6060),
 "Camp 2S":   (27.98156, 86.89914, 6400),
 "Camp 3S":   (27.96889, 86.91781, 7100),
 "Camp 4S (South Col)": (27.97337, 86.93025, 7920),
}
PEAKS = {
 "Everest": (27.98806, 86.92521, 8848.86),
 "Lhotse":  (27.96199, 86.93250, 8516),
 "Nuptse":  (27.96737, 86.88696, 7864),
}

with rasterio.open(DEM) as ds:
    print("=== Copernicus DEM 30 m, tile N27 E086 ===")
    print(f"  CRS        : {ds.crs}")
    print(f"  size       : {ds.width} x {ds.height}")
    print(f"  resolution : {abs(ds.transform.a)*3600:.2f} arcsec "
          f"({abs(ds.transform.a):.6f} deg)")
    print(f"  bounds     : lon {ds.bounds.left:.3f}..{ds.bounds.right:.3f}  "
          f"lat {ds.bounds.bottom:.3f}..{ds.bounds.top:.3f}")
    print(f"  nodata     : {ds.nodata}   dtype {ds.dtypes[0]}")

    # sample DEM elevation at each known point -> validates georeferencing
    print("\n=== DEM elevation at known OSM coordinates (validation) ===")
    print(f"  {'feature':22} {'OSM ele':>8} {'DEM':>8} {'diff':>7}")
    allpts = {**PEAKS, **CAMPS}
    rows = []
    for name, (lat, lon, ele) in allpts.items():
        v = list(ds.sample([(lon, lat)]))[0][0]
        rows.append((name, ele, float(v)))
        print(f"  {name:22} {ele:8.1f} {v:8.1f} {v-ele:+7.1f}")
    d = np.array([r[2]-r[1] for r in rows])
    print(f"  mean abs diff: {np.abs(d).mean():.1f} m   "
          f"(30 m DEM in extreme relief; summit smoothing expected)")

tf = Transformer.from_crs("EPSG:4326", UTM, always_xy=True)
print("\n=== route geometry in UTM 45N (metres) ===")
xy = {n: tf.transform(lon, lat) for n, (lat, lon, _) in CAMPS.items()}
names = list(CAMPS)
for a, b in zip(names, names[1:]):
    (xa, ya), (xb, yb) = xy[a], xy[b]
    dh = np.hypot(xb-xa, yb-ya); dv = CAMPS[b][2]-CAMPS[a][2]
    print(f"  {a:22} -> {b:22} {dh:7.0f} m horiz, {dv:+6.0f} m vert, "
          f"straight-line slope {np.degrees(np.arctan2(dv,dh)):5.1f} deg")

json.dump({"camps": CAMPS, "peaks": PEAKS, "utm": UTM,
           "camps_utm": {k: list(v) for k, v in xy.items()}},
          open("data/processed/everest/route.json", "w"), indent=2)
print("\nwrote data/processed/everest/route.json")

#!/usr/bin/env python3
"""Report everything we need to know about a DEM GeoTIFF before meshing it.

Usage: python3 inspect_dem.py ../data/raw/DEM_2024_DL_TA_11cm.tif
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import rasterio
from rasterio.crs import CRS


def human(n: float) -> str:
    return f"{n/1e6:.1f} M" if n >= 1e6 else f"{n:,.0f}"


def inspect(path: str, sample_step: int = 8) -> dict:
    with rasterio.open(path) as ds:
        t = ds.transform
        px, py = abs(t.a), abs(t.e)
        print("=" * 68)
        print(f"Source DEM: {os.path.basename(path)}")
        print("=" * 68)
        print(f"  driver          : {ds.driver}")
        print(f"  size            : {ds.width} x {ds.height} px "
              f"({human(ds.width*ds.height)} pixels)")
        print(f"  bands           : {ds.count}   dtype: {ds.dtypes[0]}")
        print(f"  CRS             : {ds.crs}")
        if ds.crs:
            print(f"    -> EPSG        : {ds.crs.to_epsg()}")
            print(f"    -> projected   : {ds.crs.is_projected}  "
                  f"(units: {ds.crs.linear_units if ds.crs.is_projected else 'degrees'})")
        print(f"  resolution      : {px:.4f} x {py:.4f} "
              f"{'m' if ds.crs and ds.crs.is_projected else 'deg'}")
        print(f"  nodata          : {ds.nodata}")
        b = ds.bounds
        print(f"  bounds          : X {b.left:.2f} .. {b.right:.2f}")
        print(f"                    Y {b.bottom:.2f} .. {b.top:.2f}")
        print(f"  extent          : {b.right-b.left:,.1f} x {b.top-b.bottom:,.1f} "
              f"{'m' if ds.crs and ds.crs.is_projected else 'deg'}")
        print(f"  blocks/tiled    : {ds.is_tiled}  block={ds.block_shapes[0]}")
        print(f"  compression     : {ds.profile.get('compress')}")
        print(f"  overviews (b1)  : {ds.overviews(1)}")

        # decimated read so we never pull 200 MB into RAM
        out_h = max(ds.height // sample_step, 1)
        out_w = max(ds.width // sample_step, 1)
        arr = ds.read(1, out_shape=(out_h, out_w), masked=True)
        valid = arr.compressed()
        cover = valid.size / arr.size * 100.0

        print(f"\n  --- elevation (decimated 1/{sample_step}, "
              f"{out_w}x{out_h}) ---")
        print(f"  valid coverage  : {cover:.1f}%  "
              f"(nodata: {100-cover:.1f}%)")
        if valid.size:
            print(f"  elevation range : {valid.min():.2f} .. {valid.max():.2f} m")
            print(f"  relief          : {valid.max()-valid.min():.2f} m")
            print(f"  mean / median   : {valid.mean():.2f} / "
                  f"{np.median(valid):.2f} m")
            for q in (1, 25, 50, 75, 99):
                print(f"    p{q:<3d}          : {np.percentile(valid, q):.2f} m")
            print(f"  non-finite      : "
                  f"{int((~np.isfinite(valid)).sum())}")
        return {"width": ds.width, "height": ds.height, "res": (px, py),
                "crs": ds.crs, "bounds": b, "nodata": ds.nodata}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--step", type=int, default=8,
                    help="decimation factor for the statistics read")
    a = ap.parse_args()
    inspect(a.path, a.step)

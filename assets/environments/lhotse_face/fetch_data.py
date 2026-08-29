#!/usr/bin/env python3
"""Download the source DEM the terrain pipeline needs.

The raster is NOT in git and does not need to be: it is 41 MB, published at a
stable public URL, and needed only if you re-run pipeline/. Git LFS would cost
quota and force every clone through it; a fetch is free and always current.

    python -m terrain.fetch_data
"""
import os, sys, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(HERE, "data")

FILES = {
    # Copernicus GLO-30 DEM, tile N27 E086 -- contains Everest, Lhotse, Nuptse.
    # ESA, free and open, no authentication.
    "Copernicus_DSM_30m_N27_E086.tif":
        "https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/"
        "Copernicus_DSM_COG_10_N27_00_E086_00_DEM/"
        "Copernicus_DSM_COG_10_N27_00_E086_00_DEM.tif",
}


def fetch(name, url):
    out = os.path.join(DEST, name)
    if os.path.exists(out):
        print(f"  {name}: already present ({os.path.getsize(out)/1e6:.1f} MB)")
        return out
    os.makedirs(DEST, exist_ok=True)
    print(f"  {name}: downloading...")
    part = out + ".part"
    with urllib.request.urlopen(url, timeout=120) as r, open(part, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk); done += len(chunk)
            if total:
                sys.stderr.write(f"\r    {done/1e6:6.1f} / {total/1e6:.1f} MB")
        sys.stderr.write("\n")
    os.replace(part, out)
    print(f"  {name}: done ({os.path.getsize(out)/1e6:.1f} MB)")
    return out


if __name__ == "__main__":
    print("Fetching terrain source data into terrain/data/")
    for n, u in FILES.items():
        fetch(n, u)
    print("\nNote: terrain/mujoco_scene.py does NOT need this -- the patch "
          "heightfield is committed. This is only for re-running pipeline/.")

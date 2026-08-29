#!/usr/bin/env python3
"""Regenerate the patch OBJ from the committed heightfield.

The OBJ is ~10 MB and would be the largest file in the repo, so only the
0.55 MB compressed heightfield is version-controlled. This rebuilds the mesh
byte-for-byte identically whenever you need it (Blender, rendering, etc.).

    python -m terrain.export_obj [-o lhotse_face_B.obj]
"""
import argparse, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--out", default=os.path.join(HERE, "lhotse_face_B.obj"))
a = ap.parse_args()

Z = np.load(f"{HERE}/data/lhotse_face_B.npz")["Z"].astype(np.float64)
M = json.load(open(f"{HERE}/data/lhotse_face_B.json"))
res = M["resolution_m"]
ny, nx = Z.shape
u = (np.arange(nx) - nx/2)*res
v = (np.arange(ny) - ny/2)*res
U, V = np.meshgrid(u, v)
Vt = np.column_stack([U.ravel(), V.ravel(), Z.ravel()])
idx = np.arange(nx*ny).reshape(ny, nx)
tl = idx[:-1, :-1].ravel(); tr = idx[:-1, 1:].ravel()
bl = idx[1:, :-1].ravel();  br = idx[1:, 1:].ravel()
F = np.vstack([np.column_stack([tl, bl, br]), np.column_stack([tl, br, tr])])
with open(a.out, "w") as f:
    f.write("# Lhotse Face patch (candidate B), regenerated from the heightfield\n")
    f.write("# REAL: location/slope/aspect from Copernicus GLO-30\n")
    f.write("# SYNTHETIC: all detail below ~30 m\n")
    f.write("# +X uphill along the fall line, +Y cross-slope, +Z up, metres\n")
    np.savetxt(f, Vt, fmt="v %.5f %.5f %.5f")
    np.savetxt(f, F+1, fmt="f %d %d %d")
print(f"wrote {a.out}  {len(Vt):,} verts  {len(F):,} tris  "
      f"{os.path.getsize(a.out)/1e6:.1f} MB")

#!/usr/bin/env python3
"""Crop the Sketchfab mesh at several sizes. How big must a crop be to be worth looking at?"""
import numpy as np, trimesh, json, os

M_PER_UNIT = 2176.0     # from Zmax=Everest 8848.86 m, Zmin~2800 m (see EVEREST_LHOTSE.md)
m = trimesh.load("data/processed/everest/everest.ply", process=False)
V = np.asarray(m.vertices); F = np.asarray(m.faces)
print(f"full mesh: {len(V):,} verts  {len(F):,} tris")

# pick the steepest well-populated spot as the crop centre
tri = V[F]; c = tri.mean(axis=1)
n = np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0])
n /= (np.linalg.norm(n, axis=1, keepdims=True)+1e-12)
slope = np.degrees(np.arccos(np.clip(np.abs(n[:,2]), 0, 1)))
# restrict to the interior so we don't land on the clipped western edge
lo, hi = V.min(axis=0), V.max(axis=0)
pad = 0.12*(hi-lo)
ok = ((c[:,0]>lo[0]+pad[0])&(c[:,0]<hi[0]-pad[0])&
      (c[:,1]>lo[1]+pad[1])&(c[:,1]<hi[1]-pad[1]))
# Target the 30-45 deg band (real Lhotse Face), NOT the global max: the
# steepest cells are near-vertical cliffs and crop to a useless thin sliver.
band = ok & (slope > 30) & (slope < 45)
gx = np.round((c[:,0]-lo[0])/(0.30/M_PER_UNIT*1000)).astype(int)
gy = np.round((c[:,1]-lo[1])/(0.30/M_PER_UNIT*1000)).astype(int)
key = gx*100003 + gy
uk, inv, cnt = np.unique(key[band], return_inverse=True, return_counts=True)
best = uk[np.argmax(cnt)]
sel0 = band & (key == best)
cx, cy = c[sel0,0].mean(), c[sel0,1].mean()
print(f"crop centre: densest 30-45 deg cluster, {sel0.sum()} faces")
print(f"crop centre: x={cx:.3f} y={cy:.3f}  mean slope {slope[sel0].mean():.1f} deg")

os.makedirs("data/processed/everest/crops", exist_ok=True)
print(f"\n{'crop size':>10} {'units':>9} {'verts':>8} {'tris':>8}   verdict")
rows=[]
for size_m in (4000, 1000, 250, 100, 25):
    half = (size_m/M_PER_UNIT)/2
    sel = ((c[:,0]>cx-half)&(c[:,0]<cx+half)&(c[:,1]>cy-half)&(c[:,1]<cy+half))
    nf = int(sel.sum())
    if nf:
        sub = m.submesh([np.nonzero(sel)[0]], append=True)
        nv = len(sub.vertices)
        p = f"data/processed/everest/crops/crop_{size_m}m.obj"
        # scale to real metres and drop to the local origin
        sub.vertices = (sub.vertices - sub.vertices.mean(axis=0))*M_PER_UNIT
        sub.export(p)
    else:
        nv = 0; p = None
    verdict = ("plenty of shape" if nf>2000 else
               "usable shape" if nf>200 else
               "a few facets" if nf>10 else
               "ESSENTIALLY FLAT - nothing to see")
    print(f"{size_m:>8} m {2*half:9.5f} {nv:8,} {nf:8,}   {verdict}")
    rows.append(dict(size_m=size_m, verts=nv, tris=nf, obj=p))
json.dump(rows, open("data/processed/everest/crops/ladder.json","w"), indent=2)

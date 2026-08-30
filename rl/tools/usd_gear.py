#!/usr/bin/env python3
"""Pull the mountaineering gear and ascender out of the project's G1 USD.

WHY A CONVERTER AND NOT "JUST USE THE USD"
`assets/robots/g1_unitree.usd` is an Isaac Sim asset. MuJoCo can *write* USD
(for rendering) but cannot load it as a model, so it can never be the robot in
an MJCF scene.

It also does not need to be. `assets/robots/g1/build_g1_usd.py` builds that USD
*from the MuJoCo Menagerie MJCF* -- the same MJCF this project already loads --
and then dresses it. So the kinematics, joint axes, masses and actuators are
already identical; the USD adds exactly two things:

  1. visual-only shells: jacket, boots, gaiters, logos (each a link's convex
     hull inflated along its normals) and the ascender tool on the right wrist
  2. BOOT_FRICTION = 0.8, against the stock menagerie 0.6

(1) is what this script recovers -- as OBJ plus a manifest naming the parent
link -- so the MuJoCo scene renders as *this* robot rather than a bare G1.
Because the shells are visual-only by construction they are added with
contype/conaffinity 0 and zero mass, so physics is bit-identical with or
without them.

(2) is a scene parameter, not an asset: see `IceParams` in climb_scene.py.
Note that setting it on geoms alone does nothing -- the G1 XML pins foot-floor
friction through an explicit <pair>, so BOOT_FRICTION never took effect in a
stock MuJoCo build of this robot.

    python -m rl.tools.usd_gear
    python -m rl.scripts.climb_scene --visual --gear --view
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_USD = os.path.join(REPO_ROOT, "assets", "robots", "g1_unitree.usd")
DEFAULT_OUT = os.path.join(REPO_ROOT, ".reference", "gear")
MANIFEST = "gear_manifest.json"

# The ascender's display mesh is ~972k vertices; MuJoCo builds a convex hull for
# every mesh at compile time, so use the decimated collision copy (~25k) for
# display too. It is the same object to within a millimetre at render scale.
_ASCENDER_DISPLAY = "collision"
_MAX_VERTS = 200_000


def _triangulate(counts, indices):
    """USD face-vertex streams -> triangle array, fanning any n-gons."""
    tris, k = [], 0
    for c in counts:
        f = indices[k : k + c]
        for i in range(1, c - 1):
            tris.append((f[0], f[i], f[i + 1]))
        k += c
    return np.asarray(tris, dtype=np.int64)


def _write_obj(path, verts, tris):
    with open(path, "w") as f:
        f.write("# extracted from g1_unitree.usd by rl/tools/usd_gear.py\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for t in tris:
            f.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")


def extract(usd_path: str = DEFAULT_USD, out_dir: str = DEFAULT_OUT) -> str:
    """Write gear OBJs plus a manifest; returns the manifest path."""
    from pxr import Usd, UsdGeom, UsdShade

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise FileNotFoundError(usd_path)
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    up = UsdGeom.GetStageUpAxis(stage)
    if up != "Z":
        raise ValueError(f"expected a Z-up stage to match MuJoCo, got {up}")
    os.makedirs(out_dir, exist_ok=True)

    entries = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        path = prim.GetPath().pathString
        parts = path.split("/")
        if len(parts) < 4:
            continue
        link, group = parts[2], parts[3]
        if group == "gear":
            name = parts[-1]
        elif group == "tool_ascender":
            if parts[-1] != _ASCENDER_DISPLAY:
                continue
            name = "ascender"
        else:
            continue

        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get()
        if pts is None:
            continue
        verts = np.asarray(pts, dtype=np.float64) * mpu
        if len(verts) > _MAX_VERTS:
            print(f"  skipping {path}: {len(verts)} verts over the {_MAX_VERTS} cap")
            continue
        tris = _triangulate(
            np.asarray(mesh.GetFaceVertexCountsAttr().Get()),
            np.asarray(mesh.GetFaceVertexIndicesAttr().Get()),
        )

        # Compose transforms from the link down to the mesh; the gear was
        # authored in link-local coordinates, but the ascender's collision copy
        # sits under a transformed group.
        xf = np.eye(4)
        node = prim
        while node.GetPath().pathString != f"/G1/{link}":
            if node.IsA(UsdGeom.Xformable):
                local = np.asarray(
                    UsdGeom.Xformable(node).GetLocalTransformation(), dtype=np.float64
                )
                xf = xf @ local        # USD is row-vector convention: v' = v @ M
            node = node.GetParent()
        verts = (np.c_[verts, np.ones(len(verts))] @ xf)[:, :3] * mpu / mpu

        rgba = [0.5, 0.5, 0.5, 1.0]
        dc = UsdGeom.Gprim(prim).GetDisplayColorAttr().Get()
        if dc:
            rgba[:3] = [float(c) for c in dc[0]]
        else:
            binding = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
            if binding:
                shader = binding.ComputeSurfaceSource()[0]
                if shader:
                    inp = shader.GetInput("diffuseColor")
                    if inp and inp.Get() is not None:
                        rgba[:3] = [float(c) for c in inp.Get()]

        obj = f"{link}__{name}.obj"
        _write_obj(os.path.join(out_dir, obj), verts, tris)
        entries.append(dict(body=link, name=name, obj=obj, rgba=rgba,
                            nverts=int(len(verts)), ntris=int(len(tris))))
        print(f"  {link:<26} {name:<16} {len(verts):>7} verts -> {obj}")

    manifest = os.path.join(out_dir, MANIFEST)
    with open(manifest, "w") as f:
        json.dump(dict(source=os.path.relpath(usd_path, REPO_ROOT),
                       meters_per_unit=mpu, items=entries), f, indent=1)
    print(f"wrote {manifest}  ({len(entries)} pieces)")
    return manifest


def load_manifest(out_dir: str = DEFAULT_OUT):
    """Return (dir, items) or (dir, []) if the gear has not been extracted."""
    path = os.path.join(out_dir, MANIFEST)
    if not os.path.exists(path):
        return out_dir, []
    with open(path) as f:
        return out_dir, json.load(f)["items"]


if __name__ == "__main__":
    sys.exit(0 if extract() else 0)

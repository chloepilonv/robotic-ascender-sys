# Ascender (handled rope clamp)

| File | What |
|---|---|
| `climbing_tool_raw.usdz` | raw Tripo scan, 1 m tall, 982k verts, textured (74 MB) — source only |
| `ascender.usd` | sim-ready: scaled to **195 mm / 165 g** (Petzl Ascension size), 114k faces, origin at bottom of handle, +Z = cam head up, convex-hull collision, `RigidBodyAPI` + `MassAPI` |
| `build_ascender.py` | raw → sim (vertex-clustering decimation, `usd-core numpy trimesh`) |

Used by `assets/robots/g1/attach_tool.py` → `assets/robots/g1_unitree_ascender.usd`.

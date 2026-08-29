# Ascender (handleless, Petzl Basic-style) — G1 right end-effector

| File | What |
|---|---|
| `headless_effector_raw.usdz` | raw Tripo scan, 1 m tall, 972k verts, textured (73 MB) — source only |
| `ascender.usd` | sim-ready: scaled to **110 x 73 mm**, ~100 g, textures re-bound from `textures/`, convex-hull collision (invisible), `RigidBodyAPI` + `MassAPI`. Frame: origin at the bottom (carabiner hole), +Z up through the cam, X = width, Y = thickness. |
| `textures/` | basecolor / metallic / roughness / normal JPEGs extracted from the usdz |
| `build_ascender.py` | raw → sim (`HEIGHT_M`, `MASS_KG`, optional `CUT_Z` / `SPUR_*` cuts) |

Used by `assets/robots/g1/attach_tool.py` → `assets/robots/g1_unitree_ascender.usd`.

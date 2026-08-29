# Ascender (handled rope clamp) — cam head only

| File | What |
|---|---|
| `climbing_tool_raw.usdz` | raw Tripo scan, full tool, 1 m tall, 982k verts, textured (74 MB) — source only |
| `ascender.usd` | sim-ready **cam head only** (rope channel + locking mechanism; handle loop and grip cut away at model z = 0.65). Scaled so the full tool would be **195 mm** (head ≈ 66 mm), **110 g**, textures re-bound from `textures/`, convex-hull collision (invisible), `RigidBodyAPI` + `MassAPI`. Frame = full-tool frame (origin at the old handle bottom, +Z up through the head). |
| `textures/` | basecolor / metallic / roughness / normal JPEGs extracted from the usdz |
| `build_ascender.py` | raw → sim (`CUT_Z`, `HEIGHT_M`, `MASS_KG` at the top) |

Used by `assets/robots/g1/attach_tool.py` → `assets/robots/g1_unitree_ascender.usd`.

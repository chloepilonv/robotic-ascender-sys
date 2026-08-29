# Ascender (handled rope clamp)

| File | What |
|---|---|
| `climbing_tool_raw.usdz` | raw Tripo scan, 1 m tall, 982k verts, textured (74 MB) — source only |
| `ascender.usd` | sim-ready: **references the usdz** (original mesh + basecolor/metallic/roughness/normal textures), scaled to **195 mm / 165 g** (Petzl Ascension size), origin at bottom of handle, +Z = cam head, convex-hull collision, `RigidBodyAPI` + `MassAPI`. Keep it next to the usdz. |
| `build_ascender.py` | raw → sim (`usd-core numpy trimesh`) |

Used by `assets/robots/g1/attach_tool.py` → `assets/robots/g1_unitree_ascender.usd`.

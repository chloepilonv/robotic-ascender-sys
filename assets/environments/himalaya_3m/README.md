# Himalaya 10 m test pad

`himalaya_3m.usd` — 10 m x 10 m physics test terrain, Z-up, metres, origin = centre (robot spawn).

```
      +Y
   ┌─────────┬─────────┐
   │ 10° slope│  ICE    │
   ├─────────┼─────────┤
   │ 40° slope│  WALL   │
   └─────────┴─────────┘  -Y
  -X                    +X
```

| Prim | What | Friction (static/dynamic) |
|---|---|---|
| `/Terrain/packed_snow` | 10x10 m mesh, ±4 mm fractal bumps, 5x5 m clear centre | 0.50 / 0.45 |
| `/Terrain/ice` | 1x1 m glossy slab at (2.5, 2.5) | 0.10 / 0.08 |
| `/Terrain/slope_10deg` | 2 m run x 2 m wide at (-2.5, 2.5), rises 0.35 m towards -X | snow |
| `/Terrain/slope_40deg` | 1.5 m run x 2 m wide at (-2.5, -2.5), rises 1.26 m towards -X | snow |
| `/Terrain/wall` | rock face at x=3.0, 2 m long, 1 m high, y in [-3.5,-1.5] | 0.80 / 0.75 |

All prims have `CollisionAPI`; materials carry both `UsdPreviewSurface` (look) and `PhysicsMaterialAPI` (grip).
Rebuild: `python build_terrain.py` (usd-core + numpy). Isaac Lab: `himalaya_3m_cfg.HIMALAYA_3M_TERRAIN_CFG`.

## Full scene
`assets/environments/himalaya_scene.py` builds the scene with the Isaac Sim API (terrain + G1 + lights + physics) — see `assets/environments/README.md`.

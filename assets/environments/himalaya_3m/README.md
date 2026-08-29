# Himalaya 3 m test pad

`himalaya_3m.usd` — 3 m x 3 m physics test terrain, Z-up, metres, origin = centre (robot spawn).

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
| `/Terrain/packed_snow` | 3x3 m mesh, ±4 mm fractal bumps | 0.50 / 0.45 |
| `/Terrain/ice` | 1x1 m glossy slab, NE quadrant | 0.10 / 0.08 |
| `/Terrain/slope_10deg` | 1 m run, rises to 0.18 m towards -X, NW | snow |
| `/Terrain/slope_40deg` | 1 m run, rises to 0.84 m towards -X, SW | snow |
| `/Terrain/wall` | rock face at x=1.0, 1 m long, 1 m high, SE | 0.80 / 0.75 |

All prims have `CollisionAPI`; materials carry both `UsdPreviewSurface` (look) and `PhysicsMaterialAPI` (grip).
Rebuild: `python build_terrain.py` (usd-core + numpy). Isaac Lab: `himalaya_3m_cfg.HIMALAYA_3M_TERRAIN_CFG`.

## Full scene
`assets/environments/himalaya_scene.py` builds the scene with the Isaac Sim API (terrain + G1 + lights + physics) — see `assets/environments/README.md`.

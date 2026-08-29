"""Isaac Lab TerrainImporterCfg for the 3 m Himalaya pad (assets/environments/himalaya_3m/himalaya_3m.usd)."""
import os
from isaaclab.terrains import TerrainImporterCfg

USD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "himalaya_3m.usd")

HIMALAYA_3M_TERRAIN_CFG = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="usd",
    usd_path=USD_PATH,
    collision_group=-1,
    # Friction lives on the USD materials (snow 0.5 / ice 0.1 / rock 0.8); physics_material here is
    # ignored for terrain_type="usd". Contact uses PhysX "average" combine mode with the boot material.
    debug_vis=False,
)
# Robot spawn: (0, 0, 0.79). Slopes rise towards -X, ice is at +X+Y, wall face at x=1.0 (y in [-1.25,-0.25]).

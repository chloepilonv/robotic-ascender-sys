"""World catalogue for the harness — the ClimbScene worlds only.

The four legacy `legacy_climb_env` worlds (the old flat tilted plane from
`rl/environment/climb_env.py`) are removed: that env is deleted, and training
now runs on `G1ClimbTerrain` (the same merged terrain). This module now
delegates the world definitions to `climb_worlds` and keeps the names
`runtime.py` has always imported (`WORLD_DEFINITIONS`, `describe_worlds`,
`world_names`, `resolve_world_name`, `ascender_geom_ids`).

The old world *names* (`free_0`, `climb_30`, ...) survive as aliases so
external callers (app/bms_ui's selftest) don't break; `resolve_world_name`
maps them to the closest ClimbScene world.
"""

from app.harness.chloe_worlds import CHLOE_WORLD_DEFINITIONS
from app.harness.climb_worlds import (
    CLIMB_WORLD_DEFINITIONS,
    DEFAULT_CLIMB_WORLD,
)

# The catalogue is the walking worlds THEN Chloe's two. Her worlds carry
# `kind: "chloe_ascender"`, which is the one thing `runtime.open_world` and
# `export_scene.open_world` branch on: a different plant (mjlab gains, one
# straight rope, her slope) and a different brain (her ONNX ascender policy
# instead of the walking network). Everything else in the harness -- the
# recorder, the pose stream, the snow, the sky, the flags, the hiker, the
# visibility dial -- treats them like any other world, because they present
# the same scene and episode surface.
WORLD_DEFINITIONS = dict(CLIMB_WORLD_DEFINITIONS)
WORLD_DEFINITIONS.update(CHLOE_WORLD_DEFINITIONS)

DEFAULT_WORLD_NAME = DEFAULT_CLIMB_WORLD

# Old names from the legacy_climb_env era -> the closest ClimbScene world.
WORLD_ALIASES = {
    "free_0": "flat_0",
    "climb_0": "flat_0",
    "free_30": "lhotse_B_free",
    "climb_30": "lhotse_B",
    # the legacy_* names from the intermediate rename, too
    "legacy_free_0": "flat_0",
    "legacy_climb_0": "flat_0",
    "legacy_free_30": "lhotse_B_free",
    "legacy_climb_30": "lhotse_B",
}


def resolve_world_name(name: str) -> str:
    """Accept an old name, return the current one. Unknown names pass through."""
    if name in WORLD_DEFINITIONS:
        return name
    resolved = WORLD_ALIASES.get(name)
    if resolved is not None:
        print(f"[worlds] {name!r} is an old name; using {resolved!r}", flush=True)
        return resolved
    return name


def world_names():
    return list(WORLD_DEFINITIONS)


def describe_worlds():
    """The rows `/api/worlds` serves to the map selector."""
    return [{
        "name": name,
        "label": definition["label"],
        "slope_degrees": definition["slope_degrees"],
        "rope": definition["rope"],
        "robot": definition["robot"],
        "kind": definition["kind"],
        "slope_provenance": definition.get("slope_provenance"),
        "description": definition["description"],
    } for name, definition in WORLD_DEFINITIONS.items()]


def ascender_geom_ids(model, meta):
    """Geoms of the ascender apparatus (carrier + visual rope segments).

    The runtime makes these transparent in "free" worlds. The merged scene
    marks the rope/carrier geoms with `GROUP_ROPE` (group 1 in
    `rl/environment/climb_scene.py`), so the ids are derived by group rather
    than by body name. Physics is unaffected either way: all apparatus geoms
    are `contype=0/conaffinity=0`.
    """
    return [
        i for i in range(model.ngeom) if model.geom_group[i] == 1  # GROUP_ROPE
    ]

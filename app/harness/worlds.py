"""World catalogue for the harness — the ClimbScene worlds only.

The four legacy `legacy_climb_env` worlds (the old flat tilted plane from
`rl/environment/climb_env.py`) are removed: that env is deleted, and training
now runs on `G1ClimbTerrain` (the same merged terrain). This module now
delegates everything to `climb_worlds`, keeping the names `runtime.py` has
always imported (`WORLD_DEFINITIONS`, `describe_worlds`, `world_names`,
`ascender_geom_ids`).
"""

from app.harness.climb_worlds import (
    CLIMB_WORLD_DEFINITIONS,
    DEFAULT_CLIMB_WORLD,
)

WORLD_DEFINITIONS = CLIMB_WORLD_DEFINITIONS
DEFAULT_WORLD_NAME = DEFAULT_CLIMB_WORLD

def world_names():
    return list(WORLD_DEFINITIONS)


def describe_worlds():
    return [
        dict(defn, name=name) for name, defn in WORLD_DEFINITIONS.items()
    ]


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

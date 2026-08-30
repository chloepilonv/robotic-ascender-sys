"""The four worlds the map selector offers, all built from THEIR env.

Nothing here rebuilds physics. A world is a `config_overrides` dict handed to
`rl.environment.climb_env.G1ClimbAscender` (via `team_env.load_team_model`),
plus one runtime flag -- whether the grip equality is active -- that is applied
to `MjData`, never to their model.

    name        slope   rope   what it shows
    climb_30    30 deg  on     the training task: slope + fixed line
    climb_5/8/10/12     on     the gentle ladder, same env at smaller slope_deg
    free_30     30 deg  OFF    can the walker handle the slope WITHOUT the rope?
    free_0       0 deg  OFF    the stock mels flat-walking baseline
    climb_0      0 deg  on     what the grip alone costs a walker, no slope

WHY THE ROPE FLAG IS NOT A CONFIG OVERRIDE: their `climb_config` has no
"disable the grip" key, and adding one would mean editing `rl/`. MuJoCo already
has the mechanism -- `data.eq_active[grip]` is the per-step enable for an
equality constraint -- so the "free" worlds simply run with it set to 0. The
model is byte-identical to the climbing one; only the constraint is off. The
carrier body and the visual rope stay exactly where their `_build_model` put
them (we hide the apparatus geoms so the picture is not confusing, which is a
render-time alpha change and nothing else).

THE MODEL CACHE. Each `G1ClimbAscender.__init__` re-compiles the MjSpec and
`mjx.put_model`s the result, so models are built lazily on first selection and
cached. MEASURED on this machine, warm: first world 1.64 s, second distinct
model 0.21 s, an already-built model 0.00 s -- 1.92 s to have all four. (An
earlier estimate of "~25 s" in this file was a COLD-START artifact: the very
first run in a fresh venv also clones mujoco_menagerie and compiles bytecode.
Do not quote it.) The cache key is the FULL frozen `config_overrides` dict, not
the slope alone: a key that omits any override that changes the model is how a
run silently reads a stale model. Because the rope flag is NOT an override, the
two 30-degree worlds share one model and the two 0-degree worlds share another
-- four worlds, two builds.

Inputs  : world name (str), one of WORLD_DEFINITIONS.
Outputs : (model, meta, definition) where model/meta are exactly what
          `team_env.load_team_model` returns and definition is the row below.
"""

import os
import time

import numpy as np

from app.harness import team_env

_HARNESS_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

# name -> the world. `config_overrides` goes straight to their env; `rope` is
# the MjData-level grip flag.
WORLD_DEFINITIONS = {
    "climb_30": {
        "label": "Climb 30° · rope",
        "slope_degrees": 30.0,
        "rope": True,
        "config_overrides": {"climb_config.slope_deg": 30.0},
        "description": "The training task: 30 degree slope, right palm on the"
                       " fixed line through the ascender.",
    },
    "climb_5": {
        "label": "Climb 5° · rope",
        "slope_degrees": 5.0,
        "rope": True,
        "config_overrides": {"climb_config.slope_deg": 5.0},
        "description": "5 degree slope with the fixed line: the gentle end of the ladder.",
    },
    "climb_8": {
        "label": "Climb 8° · rope",
        "slope_degrees": 8.0,
        "rope": True,
        "config_overrides": {"climb_config.slope_deg": 8.0},
        "description": "8 degree slope with the fixed line: the gentle end of the ladder.",
    },
    "climb_10": {
        "label": "Climb 10° · rope",
        "slope_degrees": 10.0,
        "rope": True,
        "config_overrides": {"climb_config.slope_deg": 10.0},
        "description": "10 degree slope with the fixed line: the gentle end of the ladder.",
    },
    "climb_12": {
        "label": "Climb 12° · rope",
        "slope_degrees": 12.0,
        "rope": True,
        "config_overrides": {"climb_config.slope_deg": 12.0},
        "description": "12 degree slope with the fixed line: the gentle end of the ladder.",
    },
    "free_30": {
        "label": "Free walk 30° · no rope",
        "slope_degrees": 30.0,
        "rope": False,
        "config_overrides": {"climb_config.slope_deg": 30.0},
        "description": "Same model and same slope, grip equality DEACTIVATED."
                       " Isolates how much of the behaviour is the slope.",
    },
    "free_0": {
        "label": "Free walk 0° · no rope",
        "slope_degrees": 0.0,
        "rope": False,
        "config_overrides": {"climb_config.slope_deg": 0.0},
        "description": "Flat ground, no rope: the stock mels walking baseline.",
    },
    "climb_0": {
        "label": "Climb 0° · rope",
        "slope_degrees": 0.0,
        "rope": True,
        "config_overrides": {"climb_config.slope_deg": 0.0},
        "description": "Flat ground with the grip on: what the fixed line alone"
                       " costs a walker, with the slope taken out.",
    },
}

# Every world above runs the stock Playground G1. The demo robot -- jacket,
# snow boots, ascender end-effector in place of the right hand -- is a second
# ROBOT VARIANT of the same worlds, built by pointing their `_build_model` at a
# different starting scene (app/harness/robot_variants.py). Same env class, same
# surgery, same everything else.
for _name, _definition in list(WORLD_DEFINITIONS.items()):
    _definition["robot"] = "bare"

PEMBA_WORLD_SOURCES = ("climb_30", "climb_12", "climb_8", "free_0")
for _source in PEMBA_WORLD_SOURCES:
    _base = WORLD_DEFINITIONS[_source]
    WORLD_DEFINITIONS[f"{_source}_pemba"] = {
        "label": f"{_base['label']} · Pemba G1",
        "slope_degrees": _base["slope_degrees"],
        "rope": _base["rope"],
        "config_overrides": dict(_base["config_overrides"]),
        "robot": "pemba",
        "description": (f"{_base['description']} Flown on the REAL demo robot"
                        " (jacket, snow boots, ascender instead of the right"
                        " hand) rather than the stock Playground G1."),
    }

DEFAULT_WORLD_NAME = "climb_30"


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
        "description": definition["description"],
    } for name, definition in WORLD_DEFINITIONS.items()]


def _cache_key(definition):
    """The ROBOT plus the FULL override set, frozen.

    The robot belongs in the key as much as the overrides do: `climb_30` and
    `climb_30_pemba` pass identical `config_overrides` and are different
    models. A key that omitted it would hand the jacketed world the bare
    robot's model and nothing would complain.
    """
    overrides = definition["config_overrides"]
    return (definition["robot"],) + tuple(sorted(
        (str(k), float(v) if isinstance(v, (int, float)) else v)
        for k, v in overrides.items()))


class WorldLibrary:
    """Lazy, cached loader for the four worlds' (model, meta) pairs."""

    def __init__(self, print_fingerprint=True, write_fingerprint=True):
        self._models_by_key = {}
        self.print_fingerprint = print_fingerprint
        self.write_fingerprint = write_fingerprint

    def is_cached(self, name) -> bool:
        return _cache_key(WORLD_DEFINITIONS[name]) in self._models_by_key

    def load(self, name, on_build_start=None):
        """(model, meta, definition). Builds on first use (~1.6 s), then cached.

        `on_build_start()` is called just before a build that will actually
        block, so the caller can tell the browser why the picture froze.
        """
        if name not in WORLD_DEFINITIONS:
            raise KeyError(f"unknown world {name!r}; have {world_names()}")
        definition = WORLD_DEFINITIONS[name]
        key = _cache_key(definition)
        if key not in self._models_by_key:
            if on_build_start is not None:
                on_build_start()
            build_started = time.time()
            print(f"[worlds] building {name} (robot {definition['robot']},"
                  f" overrides {definition['config_overrides']}) -- cached after",
                  flush=True)
            model, meta = team_env.load_team_model(
                config_overrides=dict(definition["config_overrides"]),
                robot=definition["robot"],
                print_fingerprint=self.print_fingerprint,
                write_fingerprint=self.write_fingerprint,
                # Each model gets its OWN fingerprint file: with one shared
                # path the last build would overwrite the evidence for the
                # others. The 30-degree jacketed model is the headline one, so
                # it takes the plain `fingerprint_pemba.json` name.
                fingerprint_path=os.path.join(
                    _HARNESS_DIRECTORY, fingerprint_filename(definition)),
            )
            self._models_by_key[key] = (model, meta)
            print(f"[worlds] built {name} in {time.time() - build_started:.2f} s",
                  flush=True)
        else:
            print(f"[worlds] {name}: model already built (shared with"
                  f" {self._siblings(name)}), no rebuild", flush=True)
        model, meta = self._models_by_key[key]
        return model, meta, definition

    def _siblings(self, name):
        key = _cache_key(WORLD_DEFINITIONS[name])
        return [other for other in WORLD_DEFINITIONS
                if other != name and _cache_key(WORLD_DEFINITIONS[other]) == key]


def fingerprint_filename(definition) -> str:
    """Which fingerprint file this world's model writes."""
    slope = definition["slope_degrees"]
    if definition["robot"] != "pemba":
        return f"fingerprint_slope_{slope:.0f}.json"
    return ("fingerprint_pemba.json" if slope == 30.0
            else f"fingerprint_pemba_slope_{slope:.0f}.json")


def ascender_geom_ids(model, meta):
    """Geoms belonging to the ascender apparatus (carrier + visual rope).

    DERIVED, not named. The carrier body is certain -- it is the body owning the
    slide joint. The rope body is best-effort: their `_build_model` creates both
    `rope` and `ascender_carrier` at the same world position, `line_pt`
    (climb_env.py:169 and :190), so a static (zero-dof) body sitting at
    `line_point_world` that is not the carrier is the rope. If the rope ever
    moves elsewhere this returns the carrier only, which costs a cosmetic
    hide-on-free-worlds and nothing else.
    """
    import mujoco
    carrier_body_id = meta["carrier_body_id"]
    line_point = np.asarray(meta["line_point_world"])
    body_ids = {carrier_body_id}
    for body_id in range(model.nbody):
        if body_id == carrier_body_id or model.body_dofnum[body_id] != 0:
            continue
        # 1e-5 m, not 1e-9: their `_line_pt` is a jax float32 array while
        # `body_pos` is float64, so an exact match never lands. 10 microns is
        # still far tighter than any two distinct bodies in this model.
        if np.linalg.norm(np.asarray(model.body_pos[body_id]) - line_point) < 1e-5:
            body_ids.add(body_id)
    geom_ids = [geom_id for geom_id in range(model.ngeom)
                if int(model.geom_bodyid[geom_id]) in body_ids]
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in sorted(body_ids)]
    print(f"[worlds] ascender apparatus derived: bodies {names},"
          f" {len(geom_ids)} geoms", flush=True)
    return geom_ids

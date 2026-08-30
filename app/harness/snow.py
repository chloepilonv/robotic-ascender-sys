"""Snow that looks like snow, and the moment a foot lands in it. Visual only.

NOTHING HERE TOUCHES PHYSICS. Two things are added to the compiled model — a
texture and a material — and one field is written on the terrain geom
(`geom_matid`). None of the three is read by the solver: a material is
appearance, and MuJoCo's contact model knows about `geom_friction`,
`geom_solref`, `condim` and the contact bitmasks, not about what the surface
looks like. The heightfield is never edited, so the ground the robot walks on is
bit-identical with the snow on or off. `PARITY.md` records the same-seed diff
that proves it.

WHY A TEXTURE AND NOT A COLOUR. `graphics.apply_alpine_look` whitens the terrain
geom by setting `geom_rgba`, which gives a flat sheet whose only relief is the
heightfield's own 12 cm roughness. Snow has structure below that: drifts a few
metres across, wind grain a few centimetres across, and the odd ice crystal
catching the sun. All three are cheap to generate and none of them can be a
heightfield, because a heightfield that fine would be a contact nightmare. The
grain earns its keep twice over: it is also the only thing on an otherwise
featureless white field that the guide's stereo block matcher can match on, so
this texture is part of the ROBOT'S SENSING, not only of the look.

WHERE THE FOOTSTEPS COME FROM. `TouchdownDetector` watches the foot geoms'
contacts with the terrain geom and reports the moment a foot lands after being
airborne. That event does two jobs at once, which is why it lives here rather
than in two places: it counts a step, and it goes out on the websocket as a
`foot_steps` event with the impact speed, so the page can play a crunch at the
right volume and drop a decal in the right place.

WHAT USED TO BE HERE, and why it is gone (2026-08-30, with the 2-D page). This
module also STAMPED each print into the texture's own pixels, faded them, and
pushed the whole texture back to every GL context with `mjr_uploadTexture` —
which only existed because a server-rendered JPEG was the only picture anyone
saw. `app/web/render3d.html` draws its own decals from the `foot_steps` events,
so the stamp, the fade and the upload are deleted. The detector that fires the
events is untouched, and so is the snow the eye cameras see.

Inputs  : the built `ClimbScene` (its spec, terrain and compiled model), and
          `MjData` after each control tick.
Outputs : `attach_snow` -> True once the snow texture is on the model, False if
          it could not be added. `TouchdownDetector.update(data, dt)` -> the
          landings on this tick, each a named map: {"foot": "left"|"right",
          "impact_speed_mps": float, "position_world": (3,) metres world frame,
          "yaw_radians": float}.
"""
from __future__ import annotations

import math

import numpy as np

# --------------------------------------------------------------- the texture
# Texels per metre of terrain. 64/m over 25 x 15 m is a 4.6 MB texture, which is
# generated once at world build and never touched again -- it used to be
# re-uploaded per footprint, and that upload was what made the number expensive.
TEXELS_PER_METER = 64.0
# ... but a 120 x 120 m sandbox at 64/m would be 7680 x 7680. The cap is on the
# TOTAL, and the resolution drops to fit -- the two are traded, never the size.
MAXIMUM_TEXELS = 1_700_000
MINIMUM_TEXTURE_SIDE = 256

# The snow's own colouring, before the alpine look's material tint is applied
# on top. Kept close to white with a cool cast: contrast between lit and shaded
# roughness is what sells a snowfield, and a bright flat white clips it away.
SNOW_BASE = (233, 238, 246)
DRIFT_AMPLITUDE = 9         # low-frequency shading, 0-255
GRAIN_AMPLITUDE = 7         # fine wind grain
SPARKLE_FRACTION = 0.0006   # ice crystals catching the sun
SPARKLE_BOOST = 22
# METRES, and it matters that they are metres: the first version wrote the
# coarse grid size in terms of the texel count and produced 4 cm blotches
# instead of 3 m drifts, which rendered as storm cloud rather than snowfield.
DRIFT_METERS = 3.5          # wavelength of the big shading
DRIFT_OCTAVE_RATIO = 3.0    # the second, finer drift layer

# --------------------------------------------------------------- touchdowns
AIRBORNE_TICKS_BEFORE_A_LANDING_COUNTS = 2   # debounce a scuffing foot

# Everything a recompile of HIS spec must leave where it was. ntex, nmat and
# the terrain geom's matid are DELIBERATELY absent: they are what changes.
STRUCTURAL_FIELDS = ("nq", "nv", "nu", "nbody", "njnt", "neq", "ngeom",
                     "nsite", "nsensor", "nkey")


def _structure(model) -> dict:
    signature = {name: int(getattr(model, name)) for name in STRUCTURAL_FIELDS}
    signature["jnt_qposadr"] = model.jnt_qposadr.tolist()
    signature["jnt_dofadr"] = model.jnt_dofadr.tolist()
    signature["actuator_target"] = model.actuator_trnid[:, 0].tolist()
    signature["body_mass"] = np.round(model.body_mass, 9).tolist()
    signature["geom_friction"] = np.round(model.geom_friction, 9).tolist()
    signature["geom_contype"] = model.geom_contype.tolist()
    signature["geom_conaffinity"] = model.geom_conaffinity.tolist()
    return signature


def texture_size_for(terrain) -> tuple:
    """(width, height) in texels for a terrain, at TEXELS_PER_METER or less."""
    length_x, length_y = terrain.size_xy
    scale = TEXELS_PER_METER
    texels = length_x * length_y * scale * scale
    if texels > MAXIMUM_TEXELS:
        scale *= math.sqrt(MAXIMUM_TEXELS / texels)
    width = max(MINIMUM_TEXTURE_SIDE, int(round(length_x * scale)))
    height = max(MINIMUM_TEXTURE_SIDE, int(round(length_y * scale)))
    return width, height


def snow_image(width: int, height: int, meters_x: float, meters_y: float,
               seed: int = 0) -> np.ndarray:
    """A snowfield, procedurally. -> (height, width, 3) uint8.

    Three scales, because that is what a real snowfield has:
      * DRIFTS -- metres across. Made by drawing white noise on a coarse grid
        whose cell count is the terrain's size in DRIFT WAVELENGTHS, then
        bilinearly blowing it up: a value-noise field, cheap, and smooth enough
        that the magnifying filter never shows the grid.
      * GRAIN -- centimetres. Full-resolution noise, blurred once so it is not
        single-pixel static (single-pixel noise aliases into a shimmer the
        moment the camera moves).
      * SPARKLE -- individual bright texels, six in ten thousand.
    All three are LUMINANCE offsets on one base colour, so the field cannot
    drift off-white however the amplitudes are dialled.
    """
    random = np.random.default_rng(seed)
    luminance = np.zeros((height, width), dtype=np.float32)

    for amplitude, wavelength_meters in (
            (DRIFT_AMPLITUDE, DRIFT_METERS),
            (0.5 * DRIFT_AMPLITUDE, DRIFT_METERS / DRIFT_OCTAVE_RATIO)):
        coarse_height = max(2, int(round(meters_y / wavelength_meters)))
        coarse_width = max(2, int(round(meters_x / wavelength_meters)))
        coarse = random.normal(0.0, 1.0, (coarse_height, coarse_width)).astype(np.float32)
        luminance += amplitude * _bilinear_upsample(coarse, height, width)

    grain = random.normal(0.0, 1.0, (height, width)).astype(np.float32)
    grain = 0.25 * (grain
                    + np.roll(grain, 1, 0) + np.roll(grain, -1, 0)
                    + np.roll(grain, 1, 1))
    luminance += GRAIN_AMPLITUDE * grain

    image = np.empty((height, width, 3), dtype=np.float32)
    image[:] = np.asarray(SNOW_BASE, dtype=np.float32)
    image += luminance[:, :, None]

    sparkle = random.random((height, width)) < SPARKLE_FRACTION
    image[sparkle] += SPARKLE_BOOST
    return np.clip(image, 0, 255).astype(np.uint8)


def _bilinear_upsample(coarse: np.ndarray, height: int, width: int) -> np.ndarray:
    """Blow a small grid up to (height, width) bilinearly. Pure numpy."""
    coarse_height, coarse_width = coarse.shape
    rows = np.linspace(0, coarse_height - 1, height, dtype=np.float32)
    columns = np.linspace(0, coarse_width - 1, width, dtype=np.float32)
    row0 = np.floor(rows).astype(int)
    column0 = np.floor(columns).astype(int)
    row1 = np.minimum(row0 + 1, coarse_height - 1)
    column1 = np.minimum(column0 + 1, coarse_width - 1)
    row_fraction = (rows - row0)[:, None]
    column_fraction = (columns - column0)[None, :]
    top = (coarse[np.ix_(row0, column0)] * (1 - column_fraction)
           + coarse[np.ix_(row0, column1)] * column_fraction)
    bottom = (coarse[np.ix_(row1, column0)] * (1 - column_fraction)
              + coarse[np.ix_(row1, column1)] * column_fraction)
    return top * (1 - row_fraction) + bottom * row_fraction


TEXTURE_NAME = "snow_ground"
MATERIAL_NAME = "snow_ground_material"
TERRAIN_GEOM_NAME = "floor"


def attach_snow(scene, seed: int = 0, verbose: bool = True):
    """Add the snow texture and material to HIS spec and recompile in place.

    Returns True once the snow is on the model, False if the scene has no
    terrain geom or the recompile moved anything structural. Idempotent: a
    cached scene re-opened for a second world already has it.
    """
    import mujoco

    model = scene.model
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, TEXTURE_NAME) >= 0:
        return True

    terrain_geom = None
    for geom in scene.spec.geoms:
        if geom.name == TERRAIN_GEOM_NAME:
            terrain_geom = geom
            break
    if terrain_geom is None:
        print(f"[snow] no {TERRAIN_GEOM_NAME!r} geom in this scene; snow off",
              flush=True)
        return False

    width, height = texture_size_for(scene.terrain)
    length_x, length_y = scene.terrain.size_xy
    image = snow_image(width, height, length_x, length_y, seed)

    before = _structure(model)
    texture = scene.spec.add_texture()
    texture.name = TEXTURE_NAME
    texture.type = mujoco.mjtTexture.mjTEXTURE_2D
    texture.builtin = mujoco.mjtBuiltin.mjBUILTIN_NONE
    texture.width, texture.height, texture.nchannel = width, height, 3
    texture.data = image.tobytes()

    material = scene.spec.add_material()
    material.name = MATERIAL_NAME
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = TEXTURE_NAME
    # ONE repeat across the whole patch, and texuniform off, so world (x, y) ->
    # texel is a single affine step with no wrapping to undo. That was needed
    # when prints were stamped into the pixels; it is kept because it is also
    # what stops the drift pattern tiling visibly across the slope.
    material.texrepeat = [1, 1]
    material.texuniform = False
    material.rgba = [1.0, 1.0, 1.0, 1.0]
    terrain_geom.material = MATERIAL_NAME

    recompiled = scene.spec.compile()
    after = _structure(recompiled)
    moved = [key for key in before if before[key] != after[key]]
    if moved:
        print(f"[snow] REFUSED: recompiling moved {moved}. Keeping the original"
              " model; the ground stays plain.", flush=True)
        return False

    recompiled.vis.global_.offwidth = model.vis.global_.offwidth
    recompiled.vis.global_.offheight = model.vis.global_.offheight
    scene.model = recompiled
    scene.data = mujoco.MjData(recompiled)
    scene.ascender.bind(recompiled, mujoco)
    scene.reset()
    if verbose:
        print(f"[snow] texture {width}x{height} over {length_x:.0f}x{length_y:.0f} m"
              f" = {width / length_x:.0f} texels/m,"
              f" {image.nbytes / 1e6:.1f} MB, uploaded once at build;"
              f" all {len(before)} structural fields unchanged", flush=True)
    return True


class TouchdownDetector:
    """When did a foot land? -> one event per landing, per control tick.

    A landing is "this foot touches the terrain now, and did not for at least
    `AIRBORNE_TICKS_BEFORE_A_LANDING_COUNTS` ticks before". The debounce is the
    whole reason this is a class and not a line: a foot that scuffs along the
    ground makes and breaks contact several times a second, and without it the
    page would machine-gun footstep sounds and the snow would be painted solid.

    Impact speed is the foot's DOWNWARD speed on the last tick it was still in
    the air, differenced from its own world height — the number a sound engine
    wants for volume, and the one that is gone by the time contact exists.

    Feet are identified by geom id, never by name: the jacketed robot's foot
    contacts are four unnamed spheres per ankle on the demo model and a single
    box on the Playground one, and `meta["foot_geom_ids"]` already holds
    whichever it is.
    """

    def __init__(self, model, foot_geom_ids, terrain_geom_id):
        import mujoco

        self.model = model
        self.terrain_geom_id = int(terrain_geom_id)
        # Group the foot geoms by the body that owns them, then label the two
        # bodies left/right off their y offset in the pelvis frame -- again, no
        # names. Body y > 0 is the robot's left.
        self.geoms_by_body = {}
        for geom_id in foot_geom_ids:
            body_id = int(model.geom_bodyid[geom_id])
            self.geoms_by_body.setdefault(body_id, []).append(int(geom_id))
        self.side_of_body = {}
        for body_id in self.geoms_by_body:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            if "left" in name:
                side = "left"
            elif "right" in name:
                side = "right"
            else:
                side = "left" if model.body_pos[body_id][1] >= 0 else "right"
            self.side_of_body[body_id] = side
        self.airborne_ticks = {body_id: 99 for body_id in self.geoms_by_body}
        self.previous_height = {body_id: None for body_id in self.geoms_by_body}
        self.descent_speed = {body_id: 0.0 for body_id in self.geoms_by_body}
        self.step_count = 0

    def describe(self) -> str:
        import mujoco
        parts = []
        for body_id, geoms in self.geoms_by_body.items():
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            parts.append(f"{self.side_of_body[body_id]}={name}({len(geoms)} geoms)")
        return ", ".join(parts)

    def update(self, data, dt_seconds: float) -> list:
        """One control tick. -> [{"foot", "impact_speed_mps", ...}, ...]."""
        touching = set()
        for index in range(data.ncon):
            contact = data.contact[index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if geom1 == self.terrain_geom_id:
                other = geom2
            elif geom2 == self.terrain_geom_id:
                other = geom1
            else:
                continue
            body_id = int(self.model.geom_bodyid[other])
            if body_id in self.geoms_by_body:
                touching.add(body_id)

        events = []
        for body_id in self.geoms_by_body:
            height = float(data.xpos[body_id][2])
            previous = self.previous_height[body_id]
            if previous is not None:
                self.descent_speed[body_id] = max(
                    0.0, (previous - height) / max(dt_seconds, 1e-9))
            self.previous_height[body_id] = height

            if body_id in touching:
                if self.airborne_ticks[body_id] >= AIRBORNE_TICKS_BEFORE_A_LANDING_COUNTS:
                    self.step_count += 1
                    rotation = np.asarray(data.xmat[body_id]).reshape(3, 3)
                    events.append({
                        "foot": self.side_of_body[body_id],
                        "impact_speed_mps": round(self.descent_speed[body_id], 3),
                        "position_world": np.asarray(data.xpos[body_id]).copy(),
                        "yaw_radians": math.atan2(rotation[1, 0], rotation[0, 0]),
                    })
                self.airborne_ticks[body_id] = 0
            else:
                self.airborne_ticks[body_id] += 1
        return events

"""Snow that looks like snow, and footprints left in it. Visual only.

NOTHING HERE TOUCHES PHYSICS. Two things are added to the compiled model — a
texture and a material — and one field is written on the terrain geom
(`geom_matid`). None of the three is read by the solver: a material is
appearance, and MuJoCo's contact model knows about `geom_friction`,
`geom_solref`, `condim` and the contact bitmasks, not about what the surface
looks like. The footprints are painted into the texture's own pixels and
re-uploaded to the GPU; the heightfield is never edited, so the ground the robot
walks on is bit-identical with the prints on or off. `PARITY.md` records the
same-seed diff that proves it.

WHY A TEXTURE AND NOT A COLOUR. `graphics.apply_alpine_look` whitens the terrain
geom by setting `geom_rgba`, which gives a flat sheet whose only relief is the
heightfield's own 12 cm roughness. Snow has structure below that: drifts a few
metres across, wind grain a few centimetres across, and the odd ice crystal
catching the sun. All three are cheap to generate and none of them can be a
heightfield, because a heightfield that fine would be a contact nightmare.

WHERE THE FOOTPRINTS COME FROM. `TouchdownDetector` watches the foot geoms'
contacts with the terrain geom and reports the moment a foot lands after being
airborne. That event does three jobs at once, which is why it lives here rather
than in three places: it stamps a print into the texture, it counts a step, and
it goes out on the websocket as a `foot_steps` event with the impact speed so
the page can play a crunch at the right volume.

Inputs  : the built `ClimbScene` (its spec, terrain and compiled model), and
          `MjData` after each control tick.
Outputs : `attach_snow` -> a `SnowGround` or None. `SnowGround.step(...)` ->
          the list of touchdown events for this tick, each a named map:
          {"foot": "left"|"right", "impact_speed_mps": float}.
"""
from __future__ import annotations

import math
import time

import numpy as np

# --------------------------------------------------------------- the texture
# Texels per metre of terrain. 64 makes a 26 cm footprint about 17 texels long,
# which is enough for the print to read as a shape rather than a smudge -- and
# the resolution is not free: `mjr_uploadTexture` replaces the WHOLE texture,
# so it is the number that sets the per-tick cost. MEASURED at 25 x 15 m: 80/m
# is a 7.2 MB texture and 6.52 ms to push to the main and eye contexts; 64/m is
# 4.6 MB and about two thirds of that.
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

# --------------------------------------------------------------- footprints
FOOTPRINT_LENGTH_METERS = 0.26
FOOTPRINT_WIDTH_METERS = 0.12
# MEASURED, not guessed: at 0.30 toward a (120,132,152) shadow the print was
# 15% darker than the snow, the drift noise is +/-14 on 233, and the prints were
# invisible in a rendered frame. 0.55 toward (70,84,110) puts a fresh core 39%
# below the surrounding snow, which reads at demo distance without looking
# painted on.
FOOTPRINT_DARKEN = 0.55     # how far toward the shadow colour a fresh core goes
FOOTPRINT_SHADOW = (70, 84, 110)
FOOTPRINT_SOFT_EDGE = 0.35  # fraction of the radius spent fading out
# At most this many texture uploads a second, per GL context. 6 rather than 10
# because the upload is the whole cost of this feature and a print that appears
# 170 ms after the foot lands is not a print anyone notices arriving late.
UPLOAD_HZ = 6.0
DECAY_INTERVAL_SECONDS = 3.0
DECAY_FACTOR = 0.90         # per decay pass; ~30 s to fade out of sight
MAXIMUM_LIVE_PRINTS = 400   # ring buffer; the oldest is erased outright

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

    Returns a `SnowGround`, or None if the scene has no terrain geom or the
    recompile moved anything structural. Idempotent: a cached scene re-opened
    for a second world already has it.
    """
    import mujoco

    model = scene.model
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, TEXTURE_NAME) >= 0:
        return SnowGround(scene, seed=seed, verbose=False)

    terrain_geom = None
    for geom in scene.spec.geoms:
        if geom.name == TERRAIN_GEOM_NAME:
            terrain_geom = geom
            break
    if terrain_geom is None:
        print(f"[snow] no {TERRAIN_GEOM_NAME!r} geom in this scene; snow off",
              flush=True)
        return None

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
    # ONE repeat across the whole patch, and texuniform off. Both matter for
    # the footprints, not for the look: it is what makes the map from a world
    # (x, y) to a texel a single affine step with no wrapping to undo.
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
        return None

    recompiled.vis.global_.offwidth = model.vis.global_.offwidth
    recompiled.vis.global_.offheight = model.vis.global_.offheight
    scene.model = recompiled
    scene.data = mujoco.MjData(recompiled)
    scene.ascender.bind(recompiled, mujoco)
    scene.reset()
    ground = SnowGround(scene, seed=seed, verbose=False)
    if verbose:
        print(f"[snow] texture {width}x{height} over {length_x:.0f}x{length_y:.0f} m"
              f" = {width / length_x:.0f} texels/m,"
              f" {image.nbytes / 1e6:.1f} MB; footprint"
              f" {FOOTPRINT_LENGTH_METERS * width / length_x:.0f} x"
              f" {FOOTPRINT_WIDTH_METERS * height / length_y:.0f} texels;"
              f" all {len(before)} structural fields unchanged", flush=True)
    return ground


class SnowGround:
    """The live texture: the snow underneath, the prints on top, the uploads.

    Two images are kept. `base` is the snowfield as generated and never
    changes. `live` is what the GPU has, and is always `base` with the current
    print alphas composited over it -- so a print can fade without accumulating
    rounding error, and erasing one is exact rather than approximate.

    A print is a soft ellipse written into an alpha channel (`np.maximum`, so
    overlapping steps do not stack into black). Fading is one multiply of that
    alpha every few seconds, followed by a re-composite of only the rectangle
    the prints actually occupy.
    """

    def __init__(self, scene, seed: int = 0, verbose: bool = True):
        import mujoco

        self.terrain = scene.terrain
        self.model = scene.model
        self.texture_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_TEXTURE, TEXTURE_NAME)
        self.available = self.texture_id >= 0
        if not self.available:
            return
        self.width = int(self.model.tex_width[self.texture_id])
        self.height = int(self.model.tex_height[self.texture_id])
        address = int(self.model.tex_adr[self.texture_id])
        # A VIEW into the model's own texture buffer, shaped like an image. This
        # is what makes a repaint one vectorised assignment instead of a Python
        # loop over rows: `tex_data` is contiguous, so the slice is a view and
        # writing through it writes the model.
        self.texture_view = self.model.tex_data[
            address:address + self.width * self.height * 3
        ].reshape(self.height, self.width, 3)
        # Regenerated from the seed rather than read back, so re-opening a
        # cached scene starts from clean snow instead of inheriting the last
        # episode's footprints.
        length_x, length_y = self.terrain.size_xy
        self.base = snow_image(self.width, self.height, length_x, length_y, seed)
        self.alpha = np.zeros((self.height, self.width), dtype=np.float32)
        self.texture_view[:] = self.base
        self.shadow = np.asarray(FOOTPRINT_SHADOW, dtype=np.float32)
        self.print_boxes = []          # (row0, row1, column0, column1), oldest first
        self.dirty = False
        self.last_upload_seconds = -1e9
        self.last_decay_seconds = 0.0
        self.upload_milliseconds = 0.0
        self.paint_milliseconds = 0.0
        self.step_count = 0

    def reset(self) -> None:
        """Clean snow again: a new episode should not walk into old prints."""
        if not self.available:
            return
        self.alpha[:] = 0.0
        self.texture_view[:] = self.base
        self.print_boxes.clear()
        self.dirty = True

    # ------------------------------------------------------------- mapping
    def world_to_texel(self, x_world: float, y_world: float) -> tuple:
        """World (x, y) -> (row, column) in the texture, or None if off-patch.

        The terrain geom is TILTED about world -y, so world x is not the
        heightfield's own coordinate: a local grid point (u, v, h) lands at
        world x = u*cos(s) - h*sin(s). `terrain.surface_z` already inverts that
        by fixed point, and this is the same three passes, kept here because it
        needs `u` rather than the height `u` produces.
        """
        terrain = self.terrain
        length_x, length_y = terrain.size_xy
        slope = terrain.slope_rad
        cosine, sine = math.cos(slope), math.sin(slope)
        u = x_world if terrain.baked else x_world / cosine
        if not terrain.baked:
            for _ in range(3):
                u = (x_world + float(terrain._sample(u, y_world)) * sine) / cosine
        column = (u + length_x / 2) / length_x * (self.width - 1)
        row = (y_world + length_y / 2) / length_y * (self.height - 1)
        if not (0 <= column < self.width and 0 <= row < self.height):
            return None
        return row, column

    # ---------------------------------------------------------- footprints
    def paint_footprint(self, x_world: float, y_world: float,
                        yaw_radians: float) -> bool:
        started = time.time()
        placed = self.world_to_texel(x_world, y_world)
        if placed is None:
            return False
        row_centre, column_centre = placed
        length_x, length_y = self.terrain.size_xy
        texels_per_meter_x = self.width / length_x
        texels_per_meter_y = self.height / length_y
        half_length = 0.5 * FOOTPRINT_LENGTH_METERS
        half_width = 0.5 * FOOTPRINT_WIDTH_METERS
        # The stamp's bounding box has to cover the rotated ellipse, so it is
        # sized by the LONGER semi-axis in both directions.
        reach_columns = int(math.ceil(half_length * texels_per_meter_x)) + 1
        reach_rows = int(math.ceil(half_length * texels_per_meter_y)) + 1
        row0 = max(0, int(row_centre) - reach_rows)
        row1 = min(self.height, int(row_centre) + reach_rows + 1)
        column0 = max(0, int(column_centre) - reach_columns)
        column1 = min(self.width, int(column_centre) + reach_columns + 1)
        if row1 <= row0 or column1 <= column0:
            return False

        rows = (np.arange(row0, row1, dtype=np.float32) - row_centre) / texels_per_meter_y
        columns = (np.arange(column0, column1, dtype=np.float32) - column_centre) / texels_per_meter_x
        delta_y, delta_x = np.meshgrid(rows, columns, indexing="ij")
        cosine, sine = math.cos(-yaw_radians), math.sin(-yaw_radians)
        along = delta_x * cosine - delta_y * sine       # foot's long axis
        across = delta_x * sine + delta_y * cosine
        radius = np.sqrt((along / half_length) ** 2 + (across / half_width) ** 2)
        # 1 inside the core, falling smoothly to 0 at the rim: a hard-edged
        # stamp reads as a decal, a soft one as a depression.
        strength = np.clip((1.0 - radius) / FOOTPRINT_SOFT_EDGE, 0.0, 1.0)

        window = self.alpha[row0:row1, column0:column1]
        np.maximum(window, strength, out=window)
        self.print_boxes.append((row0, row1, column0, column1))
        if len(self.print_boxes) > MAXIMUM_LIVE_PRINTS:
            oldest = self.print_boxes.pop(0)
            self.alpha[oldest[0]:oldest[1], oldest[2]:oldest[3]] = 0.0
            self._composite(*oldest)
        self._composite(row0, row1, column0, column1)
        self.dirty = True
        self.paint_milliseconds = (time.time() - started) * 1000.0
        return True

    def _composite(self, row0, row1, column0, column1) -> None:
        """live = base blended toward the shadow colour by alpha, in one box.

        Always recomputed FROM `base`, never from what is already on screen: a
        fade that repeatedly darkened the previous frame would ratchet the
        rounding one way and the snow would go grey.
        """
        base = self.base[row0:row1, column0:column1].astype(np.float32)
        weight = (self.alpha[row0:row1, column0:column1] * FOOTPRINT_DARKEN)[:, :, None]
        blended = base * (1.0 - weight) + self.shadow[None, None, :] * weight
        self.texture_view[row0:row1, column0:column1] = blended.astype(np.uint8)

    def decay(self, time_seconds: float) -> None:
        """Fade every print a little. Cheap, and only every few seconds."""
        if not self.print_boxes:
            return
        if time_seconds - self.last_decay_seconds < DECAY_INTERVAL_SECONDS:
            return
        self.last_decay_seconds = time_seconds
        row0 = min(box[0] for box in self.print_boxes)
        row1 = max(box[1] for box in self.print_boxes)
        column0 = min(box[2] for box in self.print_boxes)
        column1 = max(box[3] for box in self.print_boxes)
        self.alpha[row0:row1, column0:column1] *= DECAY_FACTOR
        self._composite(row0, row1, column0, column1)
        self.dirty = True

    # -------------------------------------------------------------- upload
    def upload(self, renderers, time_seconds: float, force=False) -> bool:
        """Push the changed texture to each renderer's GL context. -> uploaded?

        `mjr_uploadTexture` replaces the whole texture, so this is throttled to
        UPLOAD_HZ and skipped entirely when nothing was painted. Each renderer
        has its OWN `MjrContext` with its own GPU copy, so every one that will
        show the ground needs the call.
        """
        import mujoco

        if not self.dirty and not force:
            return False
        if not force and time_seconds - self.last_upload_seconds < 1.0 / UPLOAD_HZ:
            return False
        started = time.time()
        for renderer in renderers:
            context = getattr(renderer, "_mjr_context", None)
            gl_context = getattr(renderer, "_gl_context", None)
            if context is None or gl_context is None:
                continue
            gl_context.make_current()
            mujoco.mjr_uploadTexture(self.model, context, self.texture_id)
        self.upload_milliseconds = (time.time() - started) * 1000.0
        self.last_upload_seconds = time_seconds
        self.dirty = False
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

"""The alpine look: fog, a low sun, snow, shadows. Visual only.

NOTHING HERE TOUCHES PHYSICS. Every field written is either in `model.vis`
(MuJoCo's visualisation block), `model.geom_rgba` / `model.mat_*` (appearance),
`model.light_*` (illumination), or a render-time flag on `renderer.scene.flags`.
None of them is read by the solver. The parity story is unchanged and PARITY.md
says so; if that ever stops being true, this module is the first suspect and
`describe()` prints every value it set.

WHAT IS AND IS NOT POSSIBLE ON A COMPILED MODEL

A skybox is a TEXTURE, and textures are fixed at COMPILE time -- the merged
scene compiles with `ntex 2`, both `mjTEXTURE_2D` (the Everest logo and the
ascender's basecolour). Without a skybox texture `mjRND_SKYBOX` does nothing
and the background renders BLACK, which is what the first attempt here looked
like.

`build_scene` hands back `scene.spec`, so the texture can be added there and
the spec recompiled. That produces a new `MjModel`, which sounds like it would
invalidate every cached id -- but a texture is an ASSET, not structure, and
adding one moves nothing the physics or this harness addresses. MEASURED, not
assumed: all 14 structural fields (`nq nv nu nbody njnt neq ngeom nsite nsensor
nkey`, `jnt_qposadr`, `jnt_dofadr`, `body_mass`, actuator targets) come back
bit-identical, with only `ntex` changing 2 -> 3. `add_skybox` re-checks that
list every time and REFUSES to swap the model in if anything moved, so the day
this stops being true it fails loudly instead of silently corrupting a run.

Inputs  : the compiled `MjModel`, and the terrain size so fog distances are
          scaled to the map rather than hard-coded.
Outputs : mutates the model in place; `describe()` returns what it set.
"""

import numpy as np

# Pale blue-white: high-altitude haze, not a blue sky. Slightly blue-shifted
# grey so snow reads as white against it rather than merging into it.
FOG_COLOUR = (0.78, 0.84, 0.92, 1.0)
HAZE_COLOUR = (0.86, 0.90, 0.96, 1.0)

# Fog as a fraction of the map's diagonal: clear nearby, gone by the far edge.
FOG_START_FRACTION = 0.25
FOG_END_FRACTION = 1.35
HAZE_FRACTION = 0.25

# Exposure. The first pass set snow to 0.90 with ambient 0.42 and sun 1.00,
# which summed past 1.0 everywhere and clipped: the rendered face came back a
# featureless white sheet with LESS visible relief than the stock grey. Snow is
# bright but the camera is not; what sells it is CONTRAST between the lit and
# shaded sides of the roughness, so the total is kept near 1.0 and most of it
# is directional.
SNOW_RGBA = (0.82, 0.85, 0.90, 1.0)   # cool white, not paper white
SNOW_SPECULAR = 0.15                  # snow is matte; a shiny floor reads as ice
SNOW_SHININESS = 0.08
SNOW_REFLECTANCE = 0.05

# A low sun rakes the surface, which is the only thing that makes a heightfield
# read as terrain: overhead light flattens it into a grey sheet.
SUN_ELEVATION_DEGREES = 16.0
SUN_AZIMUTH_DEGREES = 215.0
SUN_DIFFUSE = (0.78, 0.75, 0.70)      # low sun, faintly warm
SUN_SPECULAR = (0.12, 0.12, 0.12)
SUN_AMBIENT = (0.00, 0.00, 0.00)

# Snow bounces a lot of light back up, so ambient is high and blue.
HEADLIGHT_AMBIENT = (0.20, 0.22, 0.26)
HEADLIGHT_DIFFUSE = (0.10, 0.11, 0.13)
HEADLIGHT_SPECULAR = (0.02, 0.02, 0.02)

SKY_ZENITH_RGB = (0.28, 0.44, 0.68)    # deep high-altitude blue overhead
SKY_HORIZON_RGB = (0.86, 0.90, 0.95)   # pale, matching the fog it fades into
SKY_TEXTURE_SIZE = 512

SHADOW_TEXTURE_SIZE = 4096
SHADOW_CLIP = 2.0
OFFSAMPLES = 8                        # MSAA for offscreen rendering

# Above this width, shadow rendering is dropped unless explicitly forced --
# measured, see PARITY.md. Below it shadows are free enough to keep.
SHADOW_MAXIMUM_WIDTH_PIXELS = 1280


def apply_alpine_look(model, terrain_size_meters=None, snow=True):
    """Dress a compiled model. Returns a dict of everything set."""
    diagonal = 40.0
    if terrain_size_meters is not None:
        diagonal = float(np.hypot(*terrain_size_meters))

    visual = model.vis
    visual.map.fogstart = FOG_START_FRACTION * diagonal
    visual.map.fogend = FOG_END_FRACTION * diagonal
    visual.map.haze = HAZE_FRACTION
    visual.rgba.fog[:] = FOG_COLOUR
    visual.rgba.haze[:] = HAZE_COLOUR

    visual.quality.shadowsize = SHADOW_TEXTURE_SIZE
    visual.quality.offsamples = OFFSAMPLES
    visual.map.shadowclip = SHADOW_CLIP

    visual.headlight.ambient[:] = HEADLIGHT_AMBIENT
    visual.headlight.diffuse[:] = HEADLIGHT_DIFFUSE
    visual.headlight.specular[:] = HEADLIGHT_SPECULAR
    visual.headlight.active = 1

    sun = _place_sun(model)
    floor = _make_it_snow(model) if snow else None

    return {
        "fog_start_meters": float(visual.map.fogstart),
        "fog_end_meters": float(visual.map.fogend),
        "fog_rgba": list(FOG_COLOUR), "haze_rgba": list(HAZE_COLOUR),
        "haze": float(visual.map.haze),
        "shadow_texture": int(visual.quality.shadowsize),
        "offsamples": int(visual.quality.offsamples),
        "terrain_diagonal_meters": diagonal,
        "sun": sun, "floor": floor,
    }


def _place_sun(model):
    """Point light 0 down the slope from low on the horizon, casting shadows."""
    if model.nlight == 0:
        return None
    elevation = np.radians(SUN_ELEVATION_DEGREES)
    azimuth = np.radians(SUN_AZIMUTH_DEGREES)
    direction = np.array([
        -np.cos(elevation) * np.cos(azimuth),
        -np.cos(elevation) * np.sin(azimuth),
        -np.sin(elevation),
    ])
    direction /= np.linalg.norm(direction)
    model.light_dir[0] = direction
    model.light_pos[0] = -400.0 * direction     # far away, so it reads parallel
    model.light_castshadow[0] = 1
    model.light_diffuse[0] = SUN_DIFFUSE
    model.light_specular[0] = SUN_SPECULAR
    model.light_ambient[0] = SUN_AMBIENT
    if hasattr(model, "light_directional"):
        model.light_directional[0] = 1
    return {"elevation_degrees": SUN_ELEVATION_DEGREES,
            "azimuth_degrees": SUN_AZIMUTH_DEGREES,
            "direction": direction.round(4).tolist(),
            "castshadow": True}


def _make_it_snow(model):
    """Whiten the terrain geom.

    The floor compiles with `matid -1` -- no material, just a grey `rgba` -- so
    the colour is set on the geom directly. A noise texture would need a
    compile-time asset for the same reason a skybox does; the heightfield's own
    12 cm roughness plus a raking sun already give the surface its texture, and
    that texture is real geometry rather than a painted-on pattern.
    """
    try:
        floor_id = model.geom("floor").id
    except KeyError:
        return None
    model.geom_rgba[floor_id] = SNOW_RGBA
    material_id = int(model.geom_matid[floor_id])
    if material_id >= 0:
        model.mat_rgba[material_id] = SNOW_RGBA
        model.mat_specular[material_id] = SNOW_SPECULAR
        model.mat_shininess[material_id] = SNOW_SHININESS
        model.mat_reflectance[material_id] = SNOW_REFLECTANCE
    return {"geom": "floor", "rgba": list(SNOW_RGBA),
            "material_id": material_id,
            "note": "geom rgba; no material on the floor geom"}


STRUCTURAL_FIELDS = ("nq", "nv", "nu", "nbody", "njnt", "neq", "ngeom",
                     "nsite", "nsensor", "nkey")


def _structure(model):
    """Everything a recompile must not move for the swap to be safe."""
    import numpy as np
    signature = {name: int(getattr(model, name)) for name in STRUCTURAL_FIELDS}
    signature["jnt_qposadr"] = model.jnt_qposadr.tolist()
    signature["jnt_dofadr"] = model.jnt_dofadr.tolist()
    signature["body_mass"] = np.round(model.body_mass, 9).tolist()
    signature["actuator_target"] = model.actuator_trnid[:, 0].tolist()
    return signature


def add_skybox(scene, verbose=True):
    """Add a gradient skybox to HIS spec, recompile, and swap it in safely.

    Refuses -- leaving the scene untouched -- if the recompile moved anything
    structural. Returns True if the sky was installed.
    """
    import mujoco

    model = scene.model
    if any(model.tex_type[i] == mujoco.mjtTexture.mjTEXTURE_SKYBOX
           for i in range(model.ntex)):
        return True                       # already has one; nothing to do

    before = _structure(model)
    texture = scene.spec.add_texture()
    texture.name = "alpine_sky"
    texture.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    texture.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    texture.width = texture.height = SKY_TEXTURE_SIZE
    texture.rgb1 = list(SKY_ZENITH_RGB)
    texture.rgb2 = list(SKY_HORIZON_RGB)

    recompiled = scene.spec.compile()
    after = _structure(recompiled)
    moved = [key for key in before if before[key] != after[key]]
    if moved:
        print(f"[graphics] skybox REFUSED: recompiling moved {moved}."
              " Keeping the original model; the sky stays black.", flush=True)
        return False

    # The offscreen framebuffer size lives in `vis.global_` and is set on the
    # COMPILED model by whoever makes the renderer, not in the spec -- so a
    # recompile resets it to MuJoCo's 640x480 default and the next 1920-wide
    # renderer raises "Image width 1920 > framebuffer width 640". Carry it over.
    recompiled.vis.global_.offwidth = model.vis.global_.offwidth
    recompiled.vis.global_.offheight = model.vis.global_.offheight

    # Ids are proven identical, so only the data buffer and the ascender's
    # model-bound addresses need remaking.
    scene.model = recompiled
    scene.data = mujoco.MjData(recompiled)
    scene.ascender.bind(recompiled, mujoco)
    scene.reset()
    if verbose:
        print(f"[graphics] skybox installed: gradient {SKY_ZENITH_RGB} ->"
              f" {SKY_HORIZON_RGB}, {recompiled.ntex} textures,"
              f" all {len(before)} structural fields unchanged", flush=True)
    return True


def apply_render_flags(renderer, shadows=True):
    """Render-time flags on the renderer's scene. -> what was enabled."""
    import mujoco
    flags = renderer.scene.flags
    enabled = {}
    for name, value in (("mjRND_FOG", True),
                        ("mjRND_HAZE", True),
                        ("mjRND_SHADOW", bool(shadows)),
                        ("mjRND_SKYBOX", True)):
        flag = getattr(mujoco.mjtRndFlag, name, None)
        if flag is None:
            continue
        flags[int(flag)] = 1 if value else 0
        enabled[name] = bool(value)
    return enabled


def shadows_affordable(width_pixels, forced=None):
    """Shadows are kept below the measured width where they stop being free."""
    if forced is not None:
        return bool(forced)
    return int(width_pixels) <= SHADOW_MAXIMUM_WIDTH_PIXELS

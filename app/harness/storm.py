"""VISIBILITY, as the ROBOT experiences it. A WHITE-OUT, not a snow shower.

THE POINT. The 3D page draws the weather. This file is the other half -- what a
white-out does to a machine that navigates by looking. It owns the ROBOT'S EYES;
the page's own fog lives in `app/web/render3d.html` and `three/world.js` and is
driven by the same knob and the same two endpoints, so the two agree.

**VISIBILITY IS ITS OWN DIAL** (user's ruling, 2026-08-30, and it replaced this
file's second version). Until then a `storm` switch turned the weather on and
the thickness came out of the WIND SPEED -- `100 m * exp(-wind / 6)` -- which
made two independent things one control: you could not have a still white-out or
a clear gale, and every wind experiment silently changed what the robot could
see. The knob is now a DISTANCE IN METRES and it is derived from nothing. Wind
still does force, flag, sound and gusting; visibility does only visibility.

**A WHITE-OUT IS FOG** (the earlier user ruling, which stands). The first
version painted dense flakes and wind-blown streaks over the eye images; it read
as particles slapping the lens, which is a windscreen effect, not a mountain.
What a white-out actually does is take DISTANCE away: the far slope dissolves
into white first, then the middle distance, and finally you cannot see the
person three metres in front of you. Everything here is distance-dependent, and
nothing sits on the lens.

  VISIBILITY  v, in metres, straight off the knob. CLEAR_VISIBILITY_METERS
              (100 m) means CLEAR and the eyes are not touched at all; the far
              end of the dial is MINIMUM_VISIBILITY_METERS (3 m), at which only
              the nearest couple of metres are readable.
  THE FOG     composited per pixel from the eye renderer's OWN DEPTH BUFFER:
              `out = colour*(1 - f) + white*f`, `f` ramping linearly from
              `FOG_START_FRACTION_OF_VISIBILITY * v` (nothing) to `v` (gone).
  SENSOR      mild Gaussian noise per eye, drawn INDEPENDENTLY for the left and
              the right. That is the one thing fog does not reproduce: a real
              pair of cameras staring into a low-contrast white field produces
              two different noise fields, and a block matcher has nothing to
              match. A couple of grey levels, not a texture. It scales with the
              WHITE-OUT SHARE, not with the wind, for the same reason as
              everything else here.

WHY THE FOG IS COMPOSITED AND NOT LEFT TO MuJoCo, which is the whole reason this
file is shaped the way it is. MuJoCo has linear GL fog, and it CANNOT BE MOVED
at runtime from Python. All four of these were measured, not assumed:

  * `model.vis.map.fogstart` / `fogend` are multiples of `model.stat.extent`,
    and `mjr_render(viewport, scene, context)` takes NO MODEL -- it cannot read
    them. Writing them mid-run changes the picture by **0.000**.
  * `context.fogStart` / `fogEnd` are metres and ARE writable from Python, but
    they only reach GL through `glFogf` inside `mjr_makeContext`. Writing them
    mid-run also changes the picture by **0.000**.
  * `mjr_makeContext` is not exposed in the Python bindings, so the context
    cannot be rebuilt to pick them up.
  * `context.fogRGBA` IS read every frame (**14.06** mean pixel change), which
    is exactly the trap: the fog COLOUR is live while the fog DISTANCE is frozen
    at whatever the model compiled with -- 2534 m on `flat_0`, i.e. no fog at
    all. Driving only the colour puts a flat white wash over the whole frame at
    every depth, near objects included. That is the "particles on the glass"
    look arrived at from the other direction, and it is what the first version
    of this file was really doing.

A depth-buffer composite has none of those problems, costs one extra render pass
per eye, and is the same arithmetic GL would have done.

NONE OF THIS IS PHYSICS. Nothing here writes to the model or to `MjData`: the
fog is arithmetic on two rendered arrays. `test_storm` section H is a same-seed
diff with the storm on and off, and it is 0.000e+00.

DETERMINISM. The noise comes from one `numpy.random.Generator` seeded from the
run's `--seed` and advanced once per eye per vision tick, so a replay at the
same seed sees the same grain. Nothing reads the wall clock.

Inputs  : a rendered RGB image and the `mujoco.Renderer` that produced it (its
          scene is still loaded, so the depth pass needs no second
          `update_scene`), plus the visibility in METRES.
Outputs : `StormVision.state()` -- what the page and the recorder are told;
          `degrade(image, renderer)` returns a new image, same shape and dtype.
"""
from __future__ import annotations

import math

import numpy as np

# ------------------------------------------------------------------- the fog
# THE TWO ENDS OF THE DIAL, and they are the ONLY numbers this file shares with
# the page (`CLEAR_VISIBILITY_METERS` / `MINIMUM_VISIBILITY_METERS` in
# app/web/three/world.js). If one moves, move the other, or the robot and the
# picture stop being in the same weather.
#
# 100 m is CLEAR: at or above it the eyes are handed back untouched, which is
# what the retired `storm = off` state was. 3 m is the floor rather than zero,
# because a visibility of nothing is a blank screen and not a white-out -- and
# because 3 m is already inside the follower's own FOLLOW/WAIT band, i.e. as
# blind as the demo can be and still be a demo.
CLEAR_VISIBILITY_METERS = 100.0
MINIMUM_VISIBILITY_METERS = 3.0
# Fog starts this fraction of the way out. Not zero: fog that begins at the lens
# is a flat wash over the whole frame, which is the look this is not.
FOG_START_FRACTION_OF_VISIBILITY = 0.15
# Snow-and-sky white. A white-out is white; the clear-weather pale blue-grey is
# haze, which is a different thing.
WHITEOUT_RGB = (247.0, 250.0, 253.0)
# Anything at or past this depth is sky or the far clip plane and is fully
# fogged whatever the visibility. It also keeps the ramp away from the 3219 m
# the depth buffer reports for background pixels.
FAR_DEPTH_METERS = 1000.0

# ---------------------------------------------------------------- the sensor
# Gaussian noise in grey levels of 255, per eye, drawn independently. Mild on
# purpose: this is a clean camera in a bad scene, not a broken camera. It exists
# because two real sensors staring into a low-contrast white field disagree,
# which is what leaves a block matcher nothing to match.
# It used to be `1.0 + 0.30 * wind`, i.e. 1.0 in still air and 7.0 in a 20 m/s
# gale. The endpoints are kept and the DRIVER is changed: 1.0 at 100 m of clear
# air and 7.0 in a 3 m white-out, on the same log share everything else uses.
# Grain that tracked the wind was the last place the two dials were still tied
# together.
SENSOR_NOISE_SIGMA_CLEAR = 1.0
SENSOR_NOISE_SIGMA_WHITEOUT = 7.0

_VISIBILITY_LOG_SPAN = math.log(CLEAR_VISIBILITY_METERS
                                / MINIMUM_VISIBILITY_METERS)


def clamp_visibility_meters(visibility_meters) -> float:
    """The knob, sanitised. -> metres in [MINIMUM, CLEAR].

    A missing or unparseable value reads as CLEAR, because the safe failure of a
    weather knob is good weather: a page that sends nothing must not blind the
    robot.
    """
    try:
        value = float(visibility_meters)
    except (TypeError, ValueError):
        return CLEAR_VISIBILITY_METERS
    if not math.isfinite(value):
        return CLEAR_VISIBILITY_METERS
    return max(MINIMUM_VISIBILITY_METERS,
               min(CLEAR_VISIBILITY_METERS, value))


def whiteout_share(visibility_meters) -> float:
    """How far down the dial we are. -> 0.0 at 100 m (clear), 1.0 at 3 m.

    LOGARITHMIC, because visibility is: 100 m to 50 m is barely a haze and 6 m
    to 3 m is the difference between navigating and not. The same expression
    runs in `whiteoutShare` (app/web/three/world.js) and sets the slider's own
    scale, so the page's fog, its flakes and the robot's grain all move together.
    """
    visibility = clamp_visibility_meters(visibility_meters)
    return max(0.0, min(1.0, math.log(CLEAR_VISIBILITY_METERS / visibility)
                             / _VISIBILITY_LOG_SPAN))


def fog_fraction(depth_meters, visibility) -> np.ndarray:
    """How white each pixel goes. -> (H, W) float32 in [0, 1].

    A linear ramp from `FOG_START_FRACTION_OF_VISIBILITY * visibility` (nothing)
    to `visibility` (gone) -- the same law GL's `GL_LINEAR` fog uses. Background
    pixels come back from the depth buffer at the far clip plane, 3219 m on this
    scene, and land at 1.0 like any other distant thing, which is why the sky
    whites out too.
    """
    start = FOG_START_FRACTION_OF_VISIBILITY * float(visibility)
    span = max(float(visibility) - start, 1e-3)
    depth = np.minimum(np.asarray(depth_meters, dtype=np.float32),
                       FAR_DEPTH_METERS)
    return np.clip((depth - start) / span, 0.0, 1.0)


def fog_image(image, depth_meters, visibility) -> np.ndarray:
    """Composite the white-out onto one rendered image. -> uint8 RGB.

    Inputs  : `image` (H, W, 3) uint8 RGB; `depth_meters` (H, W) float32 from
              the same renderer and the same scene; `visibility` in metres.
    Outputs : (H, W, 3) uint8, same shape. The input is not modified.
    """
    fraction = fog_fraction(depth_meters, visibility)[:, :, None]
    white = np.array(WHITEOUT_RGB, dtype=np.float32)
    blended = image.astype(np.float32) * (1.0 - fraction) + white * fraction
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)


def sensor_noise(image, visibility_meters, generator) -> np.ndarray:
    """Independent Gaussian grain on one eye. -> uint8 RGB, same shape.

    Called on EACH eye with the SAME generator, which is what makes the two
    grain fields different: the generator has advanced by the left eye's draws
    before the right asks for any. Identical noise in both eyes would sit at
    zero disparity and the matcher would happily match it.
    """
    share = whiteout_share(visibility_meters)
    sigma = (SENSOR_NOISE_SIGMA_CLEAR
             + (SENSOR_NOISE_SIGMA_WHITEOUT - SENSOR_NOISE_SIGMA_CLEAR) * share)
    if sigma <= 0.0:
        return image
    noisy = image.astype(np.float32) + generator.normal(0.0, sigma, image.shape)
    return np.clip(noisy, 0.0, 255.0).astype(np.uint8)


def render_depth(renderer) -> np.ndarray:
    """Depth for the scene the renderer has ALREADY drawn. -> (H, W) metres.

    No `update_scene` here, on purpose: `Renderer.render` draws whatever scene
    is currently loaded, so calling this straight after a colour render gives
    the depth of exactly that picture from exactly that camera, with no risk of
    the two describing different moments.
    """
    renderer.enable_depth_rendering()
    try:
        return renderer.render().copy()
    finally:
        renderer.disable_depth_rendering()


class StormVision:
    """The white-out on the robot's eyes, driven by one call a tick.

    Held by `runtime.run` and handed to `guide.StereoEyes` as its `degradation`
    hook, so the fog lands on the eye images BEFORE the block matcher and the
    detector ever see them -- the only placement that makes the degradation
    honest. Degrading a picture after the measurement is a special effect.

    Inputs  : a seed, and per tick the VISIBILITY in metres. Nothing about the
              wind reaches this class any more (user's ruling, 2026-08-30).
    Outputs : `state()` and `recorded()` for the websocket and the recorder;
              `degrade(image, renderer)` is the hook.
    """

    def __init__(self, seed=0):
        self.generator = np.random.default_rng(int(seed))
        self.visibility_meters = CLEAR_VISIBILITY_METERS
        self.fog_milliseconds = 0.0

    @property
    def enabled(self) -> bool:
        """Is the weather doing anything at all? -> bool.

        Kept as a READ-ONLY convenience for callers that want one word rather
        than a distance (the contact sheet's labels, `recorded`). It is a
        derived fact now, not a mode: there is no switch left to set.
        """
        return self.visibility_meters < CLEAR_VISIBILITY_METERS

    def update(self, visibility_meters) -> None:
        self.visibility_meters = clamp_visibility_meters(visibility_meters)

    def degrade(self, image, renderer=None, with_noise=True):
        """The hook. -> the image fogged and grained, or the image untouched.

        `renderer` is the one that drew `image`, and its scene must still be
        loaded. Without it the fog is skipped and only the grain is added, which
        is what happens if a caller has no depth to offer.

        AT CLEAR VISIBILITY THE IMAGE IS RETURNED UNTOUCHED -- not fogged with a
        100 m ramp and not grained. That identity is what makes "clear" a
        control arm in every table rather than a nearly-clear one, and the
        generator is not advanced either, so a clear run's random stream is the
        stream a run with no weather at all would have had.
        """
        if not self.enabled:
            return image
        import time
        started = time.time()
        if renderer is not None:
            image = fog_image(image, render_depth(renderer),
                              self.visibility_meters)
        self.fog_milliseconds = (time.time() - started) * 1000.0
        if with_noise:
            image = sensor_noise(image, self.visibility_meters, self.generator)
        return image

    def state(self) -> dict:
        """What the page reads. -> {"visibility_meters": metres}.

        ALWAYS A NUMBER, never null: the page's fog, its flakes and its slider
        are all functions of this one field, and a null would have each of them
        inventing its own idea of "clear".
        """
        return {"visibility_meters": round(self.visibility_meters, 2)}

    def recorded(self) -> dict:
        """Per-tick columns. `Recorder.append` stacks floats, so both are floats."""
        return {
            "visibility_meters": float(self.visibility_meters),
            "whiteout_share": float(whiteout_share(self.visibility_meters)),
        }

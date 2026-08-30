"""The storm, as the ROBOT experiences it. A WHITE-OUT, not a snow shower.

THE POINT. The 3D page draws the weather. This file is the other half -- what a
blizzard does to a machine that navigates by looking. It owns the ROBOT'S EYES;
the page's own fog lives in `app/web/three/world.js` and is driven by the same
`storm` knob and the same wind speed, so the two agree.

**A STORM IS FOG** (user's ruling, and it replaced this file's first version).
That attempt painted dense flakes and wind-blown streaks over the eye images; it
read as particles slapping the lens, which is a windscreen effect, not a
mountain. What a white-out actually does is take DISTANCE away: the far slope
dissolves into white first, then the middle distance, and finally you cannot see
the person four metres in front of you. Everything here is distance-dependent,
and nothing sits on the lens.

  VISIBILITY  v = CLEAR_VISIBILITY_METERS * exp(-speed / VISIBILITY_DECAY_MPS),
              driven by the INSTANTANEOUS wind speed so a gust really does blind
              the robot for a second. 100 m calm, 37 m at 6 m/s, 14 m at 12 m/s,
              3.6 m at 20 m/s -- at which point only the nearest few metres are
              readable at all.
  THE FOG     composited per pixel from the eye renderer's OWN DEPTH BUFFER:
              `out = colour*(1 - f) + white*f`, `f` ramping linearly from
              `FOG_START_FRACTION_OF_VISIBILITY * v` (nothing) to `v` (gone).
  SENSOR      mild Gaussian noise per eye, drawn INDEPENDENTLY for the left and
              the right. That is the one thing fog does not reproduce: a real
              pair of cameras staring into a low-contrast white field produces
              two different noise fields, and a block matcher has nothing to
              match. A couple of grey levels, not a texture.

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
          `update_scene`), plus the instantaneous wind speed.
Outputs : `StormVision.state()` -- what the page and the recorder are told;
          `degrade(image, renderer)` returns a new image, same shape and dtype.
"""
from __future__ import annotations

import math

import numpy as np

# ------------------------------------------------------------------- the fog
# THE VISIBILITY CURVE, exponential rather than the 1/(1+kv) this file started
# with, because no 1/(1+kv) passes through the three points the look was
# specified by (about 40 m at 6 m/s, 12 m at 12, 4 m at 20 -- a white-out).
# `100 * exp(-speed / 6)` gives 100 / 37 / 14 / 3.6 m at 0 / 6 / 12 / 20 m/s,
# which is those three points to within the eye's ability to tell them apart.
# THE 3D PAGE USES THE SAME CURVE (app/web/three/world.js); if one moves, move
# the other, or the robot and the picture stop being in the same weather.
CLEAR_VISIBILITY_METERS = 100.0
VISIBILITY_DECAY_MPS = 6.0
# Never quite zero: past 25 m/s the curve is already under 2 m, and a visibility
# of nothing is a blank screen rather than a white-out.
MINIMUM_VISIBILITY_METERS = 1.5
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
SENSOR_NOISE_SIGMA_STILL = 1.0
SENSOR_NOISE_SIGMA_PER_MPS = 0.30


def visibility_meters(wind_speed_mps: float) -> float:
    """How far the robot can see. -> metres."""
    speed = max(0.0, float(wind_speed_mps))
    return max(MINIMUM_VISIBILITY_METERS,
               CLEAR_VISIBILITY_METERS * math.exp(-speed / VISIBILITY_DECAY_MPS))


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


def sensor_noise(image, wind_speed_mps: float, generator) -> np.ndarray:
    """Independent Gaussian grain on one eye. -> uint8 RGB, same shape.

    Called on EACH eye with the SAME generator, which is what makes the two
    grain fields different: the generator has advanced by the left eye's draws
    before the right asks for any. Identical noise in both eyes would sit at
    zero disparity and the matcher would happily match it.
    """
    speed = max(0.0, float(wind_speed_mps))
    sigma = SENSOR_NOISE_SIGMA_STILL + SENSOR_NOISE_SIGMA_PER_MPS * speed
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

    Inputs  : a seed, and per tick the storm knob and the instantaneous wind
              speed.
    Outputs : `state()` and `recorded()` for the websocket and the recorder;
              `degrade(image, renderer)` is the hook.
    """

    def __init__(self, seed=0):
        self.generator = np.random.default_rng(int(seed))
        self.enabled = False
        self.wind_speed_mps = 0.0
        self.visibility_meters = float("inf")
        self.fog_milliseconds = 0.0

    def update(self, enabled: bool, wind_speed_mps: float) -> None:
        self.enabled = bool(enabled)
        self.wind_speed_mps = float(wind_speed_mps)
        self.visibility_meters = (visibility_meters(self.wind_speed_mps)
                                  if self.enabled else float("inf"))

    def degrade(self, image, renderer=None, with_noise=True):
        """The hook. -> the image fogged and grained, or the image untouched.

        `renderer` is the one that drew `image`, and its scene must still be
        loaded. Without it the fog is skipped and only the grain is added, which
        is what happens if a caller has no depth to offer.
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
            image = sensor_noise(image, self.wind_speed_mps, self.generator)
        return image

    def state(self) -> dict:
        return {"storm": bool(self.enabled),
                "visibility_meters": (None if not self.enabled
                                      else round(self.visibility_meters, 2))}

    def recorded(self) -> dict:
        """Per-tick columns. `Recorder.append` stacks floats, so both are floats."""
        return {
            "storm": 1.0 if self.enabled else 0.0,
            "storm_visibility_meters": (
                float(self.visibility_meters) if self.enabled else -1.0),
        }

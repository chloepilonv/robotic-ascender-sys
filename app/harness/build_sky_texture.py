"""Turn a real Everest photograph into the 3-D page's equirectangular sky.

WHY THIS SCRIPT EXISTS AT ALL. `app/web/three/stage.js` used to draw a Preetham
procedural sky: a physically-plausible gradient with nothing in it. On a
Himalayan face that reads as a studio backdrop -- the robot is on a snow slope
in the middle of a featureless blue dome. What sells the altitude is the SKYLINE:
Nuptse, Changtse, the Khumbu ridge, prayer flags. So the background became a
photograph.

THE SOURCE IS NOT A 360 PANORAMA, AND THAT IS THE WHOLE PROBLEM THIS SOLVES.
`Everest panorama from Kala Patthar.jpg` (Wikimedia Commons, Markrosenrosen,
CC BY-SA 3.0) is a 15-shot stitch covering 172 degrees horizontally and 36.18
degrees vertically -- a horizon BAND, not a sphere. Three things have to happen
before Three.js can use it as `scene.background`:

  1. SCALE IT HONESTLY. An equirectangular image is linear in angle on both
     axes: 4096 px across is 360 degrees, so one degree is 11.378 px on BOTH
     axes. The source is rescaled by its own aspect (it is close enough to the
     declared field of view that forcing the declared numbers would distort the
     peaks more than trusting the pixels does), and the resulting angular width
     is printed so the number is checked rather than assumed.

  2. FILL THE OTHER 188 DEGREES BY MIRRORING. The band is pasted once, then
     again flipped left-right. A mirror join duplicates the edge COLUMN rather
     than butting two different columns together, so both seams -- the one in
     the middle and the one that wraps at u=0 -- are continuous by construction.
     There is no seam to hunt for later, and the cost is that the far half of
     the world is Everest's reflection. On a slope where the camera mostly looks
     uphill, that is a trade worth making.

  3. EXTEND SKY AND GROUND TO THE POLES. Above the band each column fades from
     its own top pixel to a single deep zenith blue; below it, from its own
     bottom pixel to a dark moraine grey. Per-column means the join is smooth
     everywhere instead of only on average.

Finally the band is blended toward `world.js`'s FOG_COLOUR within a few degrees
of the horizon, because the terrain fades into exactly that colour at distance:
without it the fogged snow meets an unfogged photograph along a hard line.

Inputs  : --source, a wide Himalayan panorama (any aspect; JPEG/PNG).
Outputs : one JPEG, 4096x2048, sRGB, equirectangular, written to
          app/web/sky/. Horizontal angle 0 (texture u=0) points along the
          panorama's own left edge; `stage.js` yaws it into place at runtime.
          Every number that shaped the picture is printed to stdout.
"""

import argparse
import os

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

OUTPUT_WIDTH = 4096
OUTPUT_HEIGHT = 2048

# Where the panorama's vertical centre sits, in degrees of elevation. The
# Kala Patthar stitch was shot roughly level -- Everest's summit is about 14
# degrees up from a 5,645 m viewpoint and the Khumbu glacier runs below the
# horizon -- so the band straddles the equator almost symmetrically.
BAND_CENTRE_ELEVATION_DEGREES = -2.0

# The poles. The zenith is not a hand-picked blue: it is the photograph's own
# top-of-frame sky pushed darker and bluer by this factor, so the invented half
# of the sphere is the same weather as the real half. A guessed constant here
# is exactly how a composite starts to look like a composite.
ZENITH_FACTOR = (0.42, 0.52, 0.86)
# Below the band there is no photograph, and on a 38 degree face the camera DOES
# see down there whenever it swings downhill. The first pass filled it moraine
# grey and it read as a flat lead plate laid across the valley. Pale haze reads
# instead as the cloud sea that actually sits under Kala Patthar, and it is the
# same family of colour the terrain fogs into, so the join stops being a line.
NADIR_RGB = (176, 188, 204)

# The invented halves are built by extending the band's edge rows outward. Those
# rows carry the photograph's per-pixel noise, and stretching one pixel over 800
# rows turns that noise into vertical streaks -- very visible against a clear
# sky. Blurring the edge along u first is what removes them, and it costs
# nothing because the extension has no real detail to protect.
EDGE_BLUR_PIXELS = 161            # ~14 degrees of azimuth
# How far down the invented ground takes to reach NADIR_RGB, as a fraction of
# the rows below the band. Kept SHORT: the panorama's bottom rows are shadowed
# moraine, and stretching them gently over 70 degrees hung a dark grey skirt
# under the whole world -- very obvious the moment the camera swung downhill.
GROUND_FADE_FRACTION = 0.18

# world.js FOG_COLOUR = 0xbfd0e2. The terrain fades to this at distance, so the
# sky has to arrive at it too or the horizon is a cut line.
HAZE_RGB = (0xBF, 0xD0, 0xE2)
HAZE_STRENGTH = 0.55           # peak mix toward haze, at the horizon exactly
HAZE_FALLOFF_DEGREES = 6.5     # gaussian sigma; gone by ~15 degrees up

DEFAULT_OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "..", "web", "sky", "everest_kala_patthar_4k.jpg")


def _smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _blur_along_azimuth(row, width_pixels):
    """Box-blur a (W, 3) edge row, wrapping at u=0 so the seam stays seamless."""
    half = width_pixels // 2
    padded = np.concatenate([row[-half:], row, row[:half + 1]], axis=0)
    cumulative = np.concatenate([np.zeros((1, 3), np.float32),
                                 np.cumsum(padded, axis=0)], axis=0)
    span = 2 * half + 1
    return (cumulative[span:span + row.shape[0]] - cumulative[:row.shape[0]]) / span


def build(source_path, output_path, jpeg_quality=85,
          band_centre_degrees=BAND_CENTRE_ELEVATION_DEGREES):
    source = Image.open(source_path).convert("RGB")
    print(f"[sky] source {source_path}")
    print(f"[sky]   {source.width}x{source.height} px, aspect "
          f"{source.width / source.height:.3f}")

    half_width = OUTPUT_WIDTH // 2
    band_height = int(round(half_width * source.height / source.width))
    degrees_per_pixel = 360.0 / OUTPUT_WIDTH
    print(f"[sky] band  {half_width}x{band_height} px = "
          f"{half_width * degrees_per_pixel:.1f} deg wide x "
          f"{band_height * degrees_per_pixel:.1f} deg tall "
          f"(mirrored to fill 360)")

    band = np.asarray(source.resize((half_width, band_height), Image.LANCZOS),
                      dtype=np.float32)
    strip = np.concatenate([band, band[:, ::-1, :]], axis=1)
    assert strip.shape == (band_height, OUTPUT_WIDTH, 3)

    centre_row = OUTPUT_HEIGHT / 2.0 - band_centre_degrees / degrees_per_pixel
    band_top = int(round(centre_row - band_height / 2.0))
    band_bottom = band_top + band_height
    if band_top < 1 or band_bottom > OUTPUT_HEIGHT - 1:
        raise SystemExit(f"band rows {band_top}..{band_bottom} fall outside "
                         f"the {OUTPUT_HEIGHT}-row canvas")
    top_elevation = (OUTPUT_HEIGHT / 2.0 - band_top) * degrees_per_pixel
    bottom_elevation = (OUTPUT_HEIGHT / 2.0 - band_bottom) * degrees_per_pixel
    print(f"[sky] rows  {band_top}..{band_bottom} -> elevation "
          f"{top_elevation:+.1f} deg (top) to {bottom_elevation:+.1f} deg "
          f"(bottom), horizon at row {OUTPUT_HEIGHT // 2}")

    canvas = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.float32)
    canvas[band_top:band_bottom] = strip

    # --- sky above: each column fades from its own top pixels to one zenith blue
    top_edge = _blur_along_azimuth(strip[:8].mean(axis=0), EDGE_BLUR_PIXELS)
    zenith = np.clip(top_edge.mean(axis=0) * np.float32(ZENITH_FACTOR), 0, 255)
    print(f"[sky] zenith rgb {zenith.round(1).tolist()} (derived from the "
          f"photograph's own top-of-frame sky "
          f"{top_edge.mean(axis=0).round(1).tolist()})")
    rows_above = np.arange(band_top, dtype=np.float32)
    share = _smoothstep(rows_above / max(band_top - 1, 1))[:, None, None]
    canvas[:band_top] = (zenith[None, None, :] * (1.0 - share)
                         + top_edge[None, :, :] * share)

    # --- ground below: the same, toward moraine grey, and faster: nothing down
    # there is real, so the sooner it becomes one flat colour the better.
    bottom_edge = _blur_along_azimuth(strip[-8:].mean(axis=0), EDGE_BLUR_PIXELS)
    rows_below = np.arange(OUTPUT_HEIGHT - band_bottom, dtype=np.float32)
    reach = max((OUTPUT_HEIGHT - band_bottom) * GROUND_FADE_FRACTION, 1.0)
    share = _smoothstep(rows_below / reach)[:, None, None]
    canvas[band_bottom:] = (bottom_edge[None, :, :] * (1.0 - share)
                            + np.float32(NADIR_RGB)[None, None, :] * share)

    # --- horizon haze, so the photograph arrives at the terrain's fog colour
    elevation = (OUTPUT_HEIGHT / 2.0 - (np.arange(OUTPUT_HEIGHT) + 0.5)) * degrees_per_pixel
    haze = HAZE_STRENGTH * np.exp(-0.5 * (elevation / HAZE_FALLOFF_DEGREES) ** 2)
    canvas = (canvas * (1.0 - haze[:, None, None])
              + np.float32(HAZE_RGB)[None, None, :] * haze[:, None, None])
    print(f"[sky] haze  peak {HAZE_STRENGTH:.2f} toward rgb{HAZE_RGB} at the "
          f"horizon, sigma {HAZE_FALLOFF_DEGREES} deg")

    picture = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    picture.save(output_path, "JPEG", quality=jpeg_quality,
                 subsampling=0, optimize=True)

    # --- the numbers to read before believing any of this
    array = np.asarray(picture, dtype=np.float32)
    wrap_gap = np.abs(array[:, 0] - array[:, -1]).mean()
    middle_gap = np.abs(array[:, half_width - 1] - array[:, half_width]).mean()
    print(f"[sky] wrote {output_path}  {picture.width}x{picture.height}  "
          f"{os.path.getsize(output_path) / 1e6:.2f} MB  q{jpeg_quality}")
    print(f"[sky] seam  wrap column |delta| {wrap_gap:.2f}/255, mirror join "
          f"|delta| {middle_gap:.2f}/255 (both should be near zero)")
    print(f"[sky] mean  rgb {array.reshape(-1, 3).mean(axis=0).round(1).tolist()}"
          f"  band-mean rgb "
          f"{array[band_top:band_bottom].reshape(-1, 3).mean(axis=0).round(1).tolist()}")
    brightest = int(np.argmax(array[:band_top].mean(axis=(0, 2))))
    print(f"[sky] light brightest sky column {brightest} of {OUTPUT_WIDTH} "
          f"= texture azimuth {brightest * degrees_per_pixel:.0f} deg")
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", default=os.path.normpath(DEFAULT_OUTPUT))
    parser.add_argument("--quality", type=int, default=85)
    parser.add_argument("--band-centre-degrees", type=float,
                        default=BAND_CENTRE_ELEVATION_DEGREES)
    arguments = parser.parse_args()
    build(arguments.source, arguments.output, arguments.quality,
          arguments.band_centre_degrees)


if __name__ == "__main__":
    main()

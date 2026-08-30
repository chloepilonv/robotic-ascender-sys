# The sky

The 3-D page (`app/web/render3d.html`) used to draw a Preetham procedural sky
(`three/vendor/addons/objects/Sky.js`): correct physics, empty picture. What
sells altitude is the SKYLINE, and no analytic model has one. So the background
is now a photograph of the actual place.

## The asset

`everest_kala_patthar_4k.jpg` — 4096 x 2048 equirectangular, sRGB JPEG q85
(4:4:4), **0.61 MB**.

Built by `app/harness/build_sky_texture.py` from:

| | |
| --- | --- |
| **Title** | `Everest panorama from Kala Patthar.jpg` |
| **Source** | https://commons.wikimedia.org/wiki/File:Everest_panorama_from_Kala_Patthar.jpg |
| **File** | https://upload.wikimedia.org/wikipedia/commons/3/39/Everest_panorama_from_Kala_Patthar.jpg |
| **Author** | Markrosenrosen (Wikimedia Commons) |
| **Licence** | **CC BY-SA 3.0 Unported** — https://creativecommons.org/licenses/by-sa/3.0/ |
| **Date** | 12 October 2011 |
| **Original** | 12993 x 2839 px, 25.3 MB, a 15-shot stitch |
| **Declared field of view** | 172° horizontal x 36.18° vertical |
| **Shows** | Changtse, Everest, Nuptse, Lhotse, Khumbutse, Taboche, Kangtega, Thamserku, Ama Dablam, and prayer flags on Kala Patthar |

**ATTRIBUTION IS MANDATORY AND SHARE-ALIKE APPLIES.** CC BY-SA 3.0 is a
copyleft licence: the credit line above must travel with anything that ships
this image, and a modified version of the image is itself licensed CC BY-SA
3.0. The derived `everest_kala_patthar_4k.jpg` in this directory is such a
modified version — it is a crop, rescale, mirror and gradient-extension of
Markrosenrosen's photograph — so **it is CC BY-SA 3.0, not ours**. Nothing else
in this repository is affected; the licence attaches to the image file, not to
the code that reads it.

Poly Haven was checked first (CC0, no obligations) and rejected on looks: none
of its ~90 mountain/snow HDRIs is high-altitude. They are fields, forests,
frozen lakes and low hills — `alps_field`, `lago_disola`, `pizzo_pernice`,
`snowy_hillside` and the rest were rendered to a contact sheet and none of them
puts a snow peak on the horizon. A real Everest skyline under a share-alike
licence beat a licence-free European meadow.

## How the panorama became a sphere

The source is a horizon BAND, not a 360° sphere, so `build_sky_texture.py` does
three things (its module docstring has the full reasoning; the run prints every
number):

1. **Scales it honestly.** An equirectangular image is linear in angle, so
   4096 px = 360° and one degree is 11.378 px on both axes. Rescaling by the
   source's own aspect gives a band **180.0° wide x 39.3° tall**.
2. **Mirrors it to fill 360°.** The band is pasted once, then flipped
   left-right. A mirror join repeats the edge COLUMN instead of butting two
   different columns together, so both seams — the middle one and the one that
   wraps at u=0 — are continuous by construction. The build verifies this:
   `wrap column |delta| 0.00/255, mirror join |delta| 0.00/255`. The cost is
   that the far half of the world is Everest's reflection.
3. **Extends sky and ground to the poles.** Each column fades from its own edge
   pixels — blurred 161 px along azimuth first, or the extension is a field of
   vertical streaks — to a single zenith blue derived from the photograph's own
   top-of-frame sky, and downward to pale haze. The first pass used moraine
   grey below the horizon and it read as a flat lead plate across the valley;
   pale haze reads as the cloud sea that is actually under Kala Patthar, and it
   is the colour the terrain fogs into.

The band lands at rows 823–1270, i.e. elevation **+17.7° to −21.6°** with the
horizon at row 1024, and is blended toward `world.js`'s `FOG_COLOUR` (`#bfd0e2`)
within a few degrees of the horizon so the fogged terrain and the unfogged
photograph do not meet along a line.

Rebuild with:

    python -m app.harness.build_sky_texture --source <the original panorama>

## How the page uses it

`app/web/three/stage.js`:

* `scene.background` — the texture with `EquirectangularReflectionMapping`.
  Three.js re-projects it into a cube render target, so it is a true skybox at
  infinity, not a flat backdrop.
* `scene.environment` — the same texture, PMREM-filtered, at
  `environmentIntensity` **0.35**. Deliberately low: `world.js` records that an
  earlier pass at this scene summed past 1.0 everywhere and came back a white
  sheet, and image-based lighting is one more additive term in that sum. It is
  here for the cold blue bounce on the robot's metal.
* `scene.backgroundRotation` / `environmentRotation` = `(90°, 0, 150°)`. The
  90° about X is because Three.js samples an equirectangular map assuming Y is
  up while this whole app is Z-up. The 150° yaw aims Everest; see the constant's
  comment for why 150 and not the sun-exact 128.
* The scene's `FogExp2` is untouched, and the texture's own horizon haze is what
  blends it into the fog.

Cost, measured with `render3d_shots/fps.mjs`'s method (200 back-to-back draws
with a GPU fence, 1920x1080, Apple M4 Pro): procedural Sky **0.16 ms/frame**,
photographic sky **0.17 ms/frame**. Both sit at the display's 120 fps cap in
normal running.

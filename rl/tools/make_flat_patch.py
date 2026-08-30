#!/usr/bin/env python3
"""Add a slope-0 curriculum patch, so `--patch B_slope0` exists.

`assets/environments/lhotse_face/build_patch_set.py` builds every curriculum
patch as

    Z = U*tan(applied_slope) + roughness(seed, rms)

and the DEM is only consulted for the *location* metadata, which is identical
across the whole `B_slope*` family. At applied_slope = 0 the macro term is
identically zero, so a flat patch is just the roughness field -- reproducible
here without the 43 MB Copernicus raster, which is git-ignored.

The octave recipe (0.6/2.0 m, 0.3/0.6 m, 0.1/0.2 m, normalised to 0.12 m rms)
and the seed convention (20270101 + index) are copied from the generator, and
the location metadata is inherited from B_slope25 so the family stays
consistent. The result is honestly labelled: slope OVERRIDDEN, a training aid,
not a measurement of Everest at 0 degrees.

    python -m rl.tools.make_flat_patch
"""
import json
import os
import sys

import numpy as np

from rl.environment import terrain as T

CURRICULUM = os.path.join(T.PATCH_ROOT, "curriculum")
TEMPLATE = "B_slope25"
NAME = "B_slope0"
SEED = 20270106  # next after the 20270101..05 the generator used
LENGTH_M, WIDTH_M, RES = 25.0, 15.0, 0.05
ROUGH_RMS = 0.12


def main() -> int:
    meta = json.load(open(os.path.join(CURRICULUM, f"{TEMPLATE}.json")))
    nx, ny = int(round(LENGTH_M / RES)), int(round(WIDTH_M / RES))

    # Z = U*tan(0) + roughness == roughness.
    Z = T.synth_roughness(ny, nx, RES, ROUGH_RMS, SEED)
    _, planar, _ = T.deplane(Z, RES)

    meta.update(
        name=NAME,
        applied_slope_deg=0.0,
        planar_slope_after_roughness_deg=round(planar, 2),
        vertical_gain_m=0.0,
        synthetic_seed=SEED,
        WARNING=(
            "Slope overridden to 0.0 deg. The real slope at this location is "
            f"{meta['real_slope_deg']} deg. Training aid only -- this is the "
            "flat reference for checking a walking policy, not measured "
            "Everest terrain."
        ),
    )
    np.savez_compressed(os.path.join(CURRICULUM, f"{NAME}.npz"), Z=Z.astype(np.float32))
    json.dump(meta, open(os.path.join(CURRICULUM, f"{NAME}.json"), "w"), indent=2)

    tr = T.load_patch(NAME)
    print(f"wrote {CURRICULUM}/{NAME}.npz")
    print(f"  {tr.shape[1]}x{tr.shape[0]} @ {tr.res*100:.0f} cm, "
          f"slope {tr.slope_deg:.3f} deg, roughness rms {tr.rough.std():.3f} m")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Lhotse Face terrain as a MuJoCo heightfield, with slope and roughness separated.

WHY SEPARATE THEM
`assets/environments/lhotse_face/build_patch_set.py` builds every shipped patch as

    Z = U*tan(applied_slope) + roughness(seed, rms)

a pure plane plus a mean-zero fractal field. The two are additive *by
construction*, so a least-squares de-plane recovers `roughness` exactly
(verified: residual RMS 0.11 m on every patch, and cross-patch correlation
~0 because each patch draws a different seed). Slope therefore need not be
baked into the grid:

    hfield  <- roughness only (mean-zero)
    slope   <- quat on the terrain geom

MuJoCo heightfield geoms do respect orientation. The original
`rl/environment/climb_env.py` asserts the opposite ("the rough_terrain hfield
cannot be tilted") and falls back to tilting a plane; that assertion is wrong
-- a rotated hfield geom collides correctly, verified by dropping a ball on
one and watching it land and roll downhill.

Keeping slope in the quat is what makes slope randomisable per environment
under MJX: `geom_quat` is 4 floats per env, whereas batching a baked 300x500
grid would cost ~4.9 GB at 8192 envs.

TWO MODES
  separable (default)  roughness grid + quat slope. Randomisable.
  baked                a stored patch loaded verbatim, no rotation. Reproduces
                       the shipped patch exactly; use it to check that the
                       separable path has not drifted.

They differ only in where the roughness correlation length is measured: baked
patches define the 2.0/0.6/0.2 m octaves on the horizontal projection, the
separable path along the slope surface (a 1/cos(slope) stretch down the fall
line). Along-surface is the more natural definition for terrain roughness;
the difference is recorded here rather than hidden.

PROVENANCE -- READ BEFORE CLAIMING REALISM
The patch *slope angle and location* are measured from Copernicus GLO-30.
Everything finer than ~30 m is synthetic: a 25 x 15 m patch covers 0.447 of
one DEM cell. Randomising roughness explores a synthetic noise family, not
observed Everest micro-terrain.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
PATCH_ROOT = os.path.join(
    REPO_ROOT, "assets", "environments", "lhotse_face", "patches"
)

# The octave recipe from build_patch_set.roughness(), kept identical so the
# separable path reproduces the shipped patches' roughness statistics.
_OCTAVES = ((0.6, 2.0), (0.3, 0.6), (0.1, 0.2))
DEFAULT_RES_M = 0.05
DEFAULT_ROUGH_RMS = 0.12


def synth_roughness(ny: int, nx: int, res: float, rms: float, seed: int) -> np.ndarray:
    """Mean-zero fractal roughness normalised to `rms` metres.

    Identical to build_patch_set.roughness() for the same arguments, so a
    synthesised terrain is statistically the same object as a shipped patch.
    """
    rng = np.random.default_rng(seed)
    out = np.zeros((ny, nx))
    for wgt, corr in _OCTAVES:
        sm = gaussian_filter(rng.normal(size=(ny, nx)), corr / res, mode="reflect")
        out += wgt * sm / (sm.std() + 1e-12)
    return out / (out.std() + 1e-12) * rms


def find_patch(name: str) -> tuple[str, str, str]:
    """Locate a patch .npz/.json by name; searches real/ then curriculum/."""
    for sub in ("real", "curriculum"):
        npz = os.path.join(PATCH_ROOT, sub, f"{name}.npz")
        if os.path.exists(npz):
            return npz, os.path.join(PATCH_ROOT, sub, f"{name}.json"), sub
    have = {k: v for k, v in list_patches().items() if v}
    listing = "; ".join(f"{k}: {', '.join(v)}" for k, v in have.items())
    raise FileNotFoundError(
        f"no patch {name!r}. Available -- {listing}.\n"
        f"For flat ground use --patch B_slope0, or --slope 0 to synthesise it."
    )


def list_patches() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for sub in ("real", "curriculum"):
        d = os.path.join(PATCH_ROOT, sub)
        if os.path.isdir(d):
            out[sub] = sorted(
                f[:-4] for f in os.listdir(d) if f.endswith(".npz")
            )
    return out


def deplane(Z: np.ndarray, res: float) -> tuple[np.ndarray, float, float]:
    """Split a patch into (mean-zero roughness, slope_deg, aspect_rad)."""
    ny, nx = Z.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    A = np.c_[(xx * res).ravel(), (yy * res).ravel(), np.ones(Z.size)]
    coef, *_ = np.linalg.lstsq(A, Z.ravel(), rcond=None)
    gx, gy = float(coef[0]), float(coef[1])
    rough = (Z.ravel() - A @ coef).reshape(Z.shape)
    slope_deg = float(np.degrees(np.arctan(np.hypot(gx, gy))))
    return rough, slope_deg, float(np.arctan2(gy, gx))


@dataclass
class Terrain:
    """A terrain ready to be written into a MuJoCo model.

    `rough` is the mean-zero heightfield in metres. `slope_deg` is applied as a
    rotation of the terrain geom about world -Y, so the surface rises toward +x
    and the fall line is the world x axis.
    """

    rough: np.ndarray
    res: float
    slope_deg: float
    name: str = "terrain"
    source: str = "synthetic"
    meta: dict = field(default_factory=dict)
    baked: bool = False  # slope already in `rough`; do not rotate the geom

    @property
    def shape(self) -> tuple[int, int]:
        return self.rough.shape

    @property
    def size_xy(self) -> tuple[float, float]:
        ny, nx = self.rough.shape
        return nx * self.res, ny * self.res

    @property
    def z_span(self) -> float:
        return float(self.rough.max() - self.rough.min())

    @property
    def z_min(self) -> float:
        return float(self.rough.min())

    @property
    def slope_rad(self) -> float:
        return float(np.radians(self.slope_deg))

    def hfield_data(self) -> np.ndarray:
        """Grid normalised to [0, 1], as MuJoCo's hfield_data expects."""
        span = self.z_span or 1.0
        return ((self.rough - self.rough.min()) / span).astype(np.float32)

    def hfield_size(self) -> tuple[float, float, float, float]:
        """(radius_x, radius_y, elevation, base) for the <hfield> asset."""
        lx, ly = self.size_xy
        return (lx / 2, ly / 2, max(self.z_span, 1e-6), 1.0)

    def geom_quat(self) -> tuple[float, float, float, float]:
        """Rotation about -Y so the geom's local +x points uphill."""
        if self.baked:
            return (1.0, 0.0, 0.0, 0.0)
        h = self.slope_rad / 2.0
        return (float(np.cos(h)), 0.0, float(-np.sin(h)), 0.0)

    def geom_pos(self) -> tuple[float, float, float]:
        """Offset that restores true heights after hfield normalisation.

        `hfield_data` is stored as (grid - z_min)/z_span with elevation z_span,
        so the surface in the geom's local frame is `grid - z_min`. Lifting the
        geom by +z_min along its own z puts the surface back where it belongs.
        """
        if self.baked:
            return (0.0, 0.0, self.z_min)
        s = self.slope_rad
        return (-self.z_min * np.sin(s), 0.0, self.z_min * np.cos(s))

    # -- surface queries, world frame ----------------------------------
    @property
    def world_extent_x(self) -> float:
        """Half-width of the terrain along world x.

        Rotating the geom foreshortens the patch: a 25 m grid at 50 deg only
        spans 16 m of world x. Queries beyond this fall off the heightfield.
        """
        lx, _ = self.size_xy
        return lx / 2 if self.baked else lx / 2 * float(np.cos(self.slope_rad))

    def surface_z(self, x, y, iters: int = 3):
        """World-frame terrain height under (x, y). Vectorised.

        A local grid point (u, v, h) maps to world x = u*cos(s) - h*sin(s), so
        recovering u from x is implicit whenever the surface is not flat. Two or
        three fixed-point passes converge to well under a millimetre; ignoring
        the h*sin(s) term (as the first version did) leaves errors of ~0.2 m at
        38 deg, enough to hang the rope through the ground.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if self.baked:
            return self._sample(x, y)
        s = self.slope_rad
        cs, sn = np.cos(s), np.sin(s)
        u = x / cs
        for _ in range(iters):
            u = (x + self._sample(u, y) * sn) / cs
        return u * sn + self._sample(u, y) * cs

    def _sample(self, u, v):
        """Bilinear sample of the roughness grid, grid coordinates in metres."""
        ny, nx = self.rough.shape
        lx, ly = nx * self.res, ny * self.res
        fx = np.clip((np.asarray(u, float) + lx / 2) / self.res - 0.5, 0, nx - 1)
        fy = np.clip((np.asarray(v, float) + ly / 2) / self.res - 0.5, 0, ny - 1)
        x0, y0 = np.floor(fx).astype(int), np.floor(fy).astype(int)
        x1, y1 = np.minimum(x0 + 1, nx - 1), np.minimum(y0 + 1, ny - 1)
        tx, ty = fx - x0, fy - y0
        g = self.rough
        return (
            g[y0, x0] * (1 - tx) * (1 - ty)
            + g[y0, x1] * tx * (1 - ty)
            + g[y1, x0] * (1 - tx) * ty
            + g[y1, x1] * tx * ty
        )


def load_patch(name: str = "B") -> Terrain:
    """Load a shipped patch, splitting slope from roughness (separable mode)."""
    npz, jsn, sub = find_patch(name)
    meta = json.load(open(jsn))
    res = float(meta.get("resolution_m", DEFAULT_RES_M))
    rough, slope_deg, _ = deplane(np.load(npz)["Z"].astype(np.float64), res)
    return Terrain(rough, res, slope_deg, name, f"patch:{sub}/{name}", meta)


def load_patch_baked(name: str = "B") -> Terrain:
    """Load a shipped patch verbatim, slope baked into the grid."""
    npz, jsn, sub = find_patch(name)
    meta = json.load(open(jsn))
    res = float(meta.get("resolution_m", DEFAULT_RES_M))
    Z = np.load(npz)["Z"].astype(np.float64)
    _, slope_deg, _ = deplane(Z, res)
    return Terrain(
        Z, res, slope_deg, name, f"patch-baked:{sub}/{name}", meta, baked=True
    )


def make_terrain(
    slope_deg: float = 38.0,
    rough_rms: float = DEFAULT_ROUGH_RMS,
    seed: int = 0,
    length_m: float = 25.0,
    width_m: float = 15.0,
    res: float = DEFAULT_RES_M,
) -> Terrain:
    """Synthesise a terrain. This is the randomisation entry point.

    slope_deg  -> macro tilt          (the "slope" axis)
    rough_rms  -> roughness amplitude (the "surface variation" axis)
    seed       -> surface realisation (a fresh draw of the same noise family)
    """
    nx, ny = int(round(length_m / res)), int(round(width_m / res))
    return Terrain(
        rough=synth_roughness(ny, nx, res, rough_rms, seed),
        res=res,
        slope_deg=float(slope_deg),
        name=f"synth_s{slope_deg:g}_r{rough_rms:g}_{seed}",
        source="synthetic",
        meta=dict(applied_slope_deg=float(slope_deg), rough_rms=rough_rms, seed=seed),
    )

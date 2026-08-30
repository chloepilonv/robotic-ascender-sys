"""Fixed-rope route and the ascender that rides it.

THE MECHANISM
A real ascender is a one-way clamp: it slides freely up a fixed rope and jams
under load in the other direction, so the climber's hand can advance along the
line but never retreat, and a slip is arrested by the rope.

The original `climb_env.py` modelled this as a *slide joint* along a straight
cylinder. A slide joint has exactly one axis, so it cannot follow a rope that
drapes over terrain. This module replaces it with a driven attachment point:

    carrier = a MuJoCo mocap body
    grip    = connect equality, `right_palm` site <-> carrier site
    ratchet = each substep, project the palm onto the polyline to get its arc
              length s, clamp s to be non-decreasing, and write the carrier to
              polyline(s)

The equality then does the physics. Perpendicular to the rope it pulls the hand
onto the line (the hand can never leave the rope). Along the rope the carrier
tracks the hand, so upward motion is free. When the robot slips, s stays at its
high-water mark and the equality hauls the hand back up to it -- fall arrest,
which is the whole point of the device.

WHY MOCAP RATHER THAN A JOINT
A mocap body has zero degrees of freedom, so nq/nv are untouched. That removes
every `qpos[7:slide_qposadr]` slice, the phantom-joint trimming, and the eight
`_cost_*` overrides that the slide-joint version needed to hide its extra
coordinate; the observation stays natively 103-dimensional and the mels
baseline policy still loads. It also generalises for free: an arbitrary
polyline costs no more than a straight line.

JAX PORTABILITY
`project_arclen` and `point_at` are branch-free -- an argmin and a clipped
searchsorted over fixed-size arrays. Swapping `np` for `jnp` makes them
jit/vmap-safe inside an MJX `lax.scan`, the same place the old ratchet lived.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RopeRoute:
    """A polyline fixed rope, parameterised by arc length from its low end.

    `points` is (N+1, 3) in world coordinates, ordered uphill: index 0 is the
    bottom anchor, index N the top.
    """

    points: np.ndarray

    def __post_init__(self):
        self.points = np.asarray(self.points, dtype=float)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError(f"points must be (N+1, 3), got {self.points.shape}")
        if len(self.points) < 2:
            raise ValueError("a rope needs at least two waypoints")
        d = np.diff(self.points, axis=0)
        seg_len = np.linalg.norm(d, axis=1)
        if np.any(seg_len < 1e-9):
            raise ValueError("duplicate consecutive waypoints give a zero-length segment")
        self.seg = d                                   # (N, 3)
        self.seg_len = seg_len                         # (N,)
        self.cum = np.concatenate([[0.0], np.cumsum(seg_len)])  # (N+1,)

    @property
    def length(self) -> float:
        return float(self.cum[-1])

    @property
    def n_seg(self) -> int:
        return len(self.seg)

    def project_arclen(self, p) -> tuple[float, float]:
        """Arc length of the closest point on the rope to `p`, and the distance.

        Branch-free: clamp the parameter on every segment, then take an argmin.
        """
        p = np.asarray(p, dtype=float)
        rel = p[None, :] - self.points[:-1]                       # (N, 3)
        t = np.clip(
            np.einsum("ij,ij->i", rel, self.seg) / self.seg_len**2, 0.0, 1.0
        )
        closest = self.points[:-1] + t[:, None] * self.seg        # (N, 3)
        dist = np.linalg.norm(p[None, :] - closest, axis=1)       # (N,)
        i = int(np.argmin(dist))
        return float(self.cum[i] + t[i] * self.seg_len[i]), float(dist[i])

    def point_at(self, s) -> np.ndarray:
        """World point at arc length `s`, clamped to the rope's extent."""
        s = np.clip(s, 0.0, self.length)
        j = int(np.clip(np.searchsorted(self.cum, s, side="right") - 1, 0, self.n_seg - 1))
        u = (s - self.cum[j]) / self.seg_len[j]
        return self.points[j] + u * self.seg[j]

    def tangent_at(self, s) -> np.ndarray:
        """Unit tangent (pointing uphill) at arc length `s`."""
        s = np.clip(s, 0.0, self.length)
        j = int(np.clip(np.searchsorted(self.cum, s, side="right") - 1, 0, self.n_seg - 1))
        return self.seg[j] / self.seg_len[j]

    def translated(self, delta) -> "RopeRoute":
        return RopeRoute(self.points + np.asarray(delta, dtype=float)[None, :])

    def shifted_through(self, target) -> "RopeRoute":
        """Translate the rope so its closest point to `target` lands on it.

        Used to guarantee the reset pose starts with the palm exactly on the
        rope: a non-zero initial equality error would otherwise be yanked out
        by the stiff grip constraint on the first step.
        """
        target = np.asarray(target, dtype=float)
        s, _ = self.project_arclen(target)
        return self.translated(target - self.point_at(s))


def drape_route(
    terrain,
    n_waypoints: int = 9,
    standoff: float = 0.60,
    margin: float = 1.0,
    lateral_amp: float = 0.35,
    lateral_waves: float = 2.2,
    lateral_phase: float = 0.0,
) -> RopeRoute:
    """Lay a rope up the fall line, following the terrain at a fixed standoff.

    The fall line is world +x (terrain.surface_z rises toward +x). `standoff`
    is measured vertically above the surface -- the rope hangs off anchors, so
    it clears the ground rather than resting on it. `lateral_amp` gives the
    line a gentle sideways weave so the route is a genuine polyline and the
    ascender's segment handling is actually exercised.
    """
    lx, ly = terrain.size_xy
    half = lx / 2 - margin
    if half <= 0:
        raise ValueError(f"margin {margin} m too large for a {lx:.1f} m patch")
    x = np.linspace(-half, half, n_waypoints)
    y = lateral_amp * np.sin(np.linspace(0.0, lateral_waves, n_waypoints) + lateral_phase)
    y = np.clip(y, -ly / 2 + margin, ly / 2 - margin)
    z = terrain.surface_z(x, y) + standoff
    return RopeRoute(np.stack([x, y, z], axis=1))


class MocapAscender:
    """Drives the carrier mocap body along a rope, with the one-way ratchet.

    Owns only the arc-length state `s`, so it is cheap to reset and trivial to
    port: the MJX version carries `s` in the env's info dict instead.
    """

    def __init__(self, route: RopeRoute, s0: float = 0.0, ratchet: bool = True):
        self.route = route
        self.s = float(s0)
        self.s0 = float(s0)
        self.ratchet = ratchet

    def reset(self, s0: float | None = None) -> None:
        self.s = self.s0 if s0 is None else float(s0)

    def update(self, palm_xyz) -> tuple[np.ndarray, float]:
        """Advance the ratchet from the palm position; return (carrier_xyz, s).

        The clamp is what makes it an ascender rather than a bead on a wire.
        """
        s_raw, _ = self.route.project_arclen(palm_xyz)
        self.s = max(self.s, s_raw) if self.ratchet else s_raw
        self.s = float(np.clip(self.s, 0.0, self.route.length))
        return self.route.point_at(self.s), self.s

    @property
    def progress(self) -> float:
        """Arc length climbed since reset -- the natural RL progress reward."""
        return self.s - self.s0

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


class RopeCarrier:
    """A bead on a wire: free to slide along the rope, held onto it, one-way.

    WHY THE CARRIER NEEDS REAL DEGREES OF FREEDOM
    The obvious construction -- a zero-DOF mocap carrier written to the palm's
    projection each substep -- does not work, and fails silently. `connect` is an
    isotropic 3-DOF point constraint, so the palm cannot move relative to the
    carrier at all; its projection therefore never changes; so the carrier never
    moves. The hand ends up welded to a fixed point. Measured: driving the
    shoulder through 1.2 rad advanced the ascender 0.000 m.

    So the carrier is a real body with three slide joints. Each substep its
    PERPENDICULAR offset from the rope is removed and the perpendicular velocity
    cancelled, while the along-rope component is left to the dynamics -- the hand
    pulls the carrier up the line, exactly as a hand pulls an ascender.

    The ratchet then clamps the arc length non-decreasing, which is what makes it
    an ascender rather than a free bead: it slides up and jams under load.

    The carrier's joints are appended after the robot's, so `qpos[7:7+29]` and
    `qvel[6:6+29]` still address the robot alone -- slice with explicit bounds,
    never open-ended, or the carrier reads as three phantom joints.
    """

    def __init__(self, route: RopeRoute, s0: float = 0.0, ratchet: bool = True,
                 slide_friction: float = 8.0):
        """`slide_friction` is the dry (Coulomb) drag of the cam on the sheath,
        in newtons -- a constant resisting force, not proportional to speed.

        A real ascender takes a small steady push to run up a rope; it does not
        glide. 8 N is at the light end of a spring-loaded cam. Set 0 for a
        frictionless bead. It has no bearing on the downward direction: that is
        the ratchet, which is absolute.
        """
        self.route = route
        self.s = float(s0)
        self.s0 = float(s0)
        self.ratchet = ratchet
        self.slide_friction = float(slide_friction)
        self._slide_adr = None   # qpos address of the carrier's 3 slide joints
        self._dof_adr = None
        self._origin = None      # carrier body frame origin (slides are offsets)
        self._dv = 0.0           # velocity the friction removes per substep

    def bind(self, model, mujoco) -> None:
        """Locate the carrier's coordinates in a compiled model."""
        jid = model.joint("carrier_x").id
        self._slide_adr = int(model.jnt_qposadr[jid])
        self._dof_adr = int(model.jnt_dofadr[jid])
        body = model.body("rope_carrier").id
        self._origin = np.asarray(model.body_pos[body], dtype=float).copy()
        # Coulomb friction is applied on the velocity because that is where this
        # class already projects. Doing it here rather than through MuJoCo's
        # joint `frictionloss` keeps it isotropic: frictionloss sits on the three
        # world-axis slides, so its effective resistance would depend on which
        # way the rope happens to point.
        mass = float(model.body_mass[body]) or 1.0
        self._dv = self.slide_friction / mass * float(model.opt.timestep)

    def reset(self, s0: float | None = None) -> None:
        self.s = self.s0 if s0 is None else float(s0)

    def advance(self, pos) -> float:
        """Ratcheted arc length for a carrier at `pos`. Pure; no model needed."""
        s_raw, _ = self.route.project_arclen(pos)
        self.s = max(self.s, s_raw) if self.ratchet else s_raw
        self.s = float(np.clip(self.s, 0.0, self.route.length))
        return self.s

    def place(self, data) -> None:
        """Put the carrier at the current arc length and stop it dead."""
        a, v = self._slide_adr, self._dof_adr
        data.qpos[a : a + 3] = self.route.point_at(self.s) - self._origin
        data.qvel[v : v + 3] = 0.0

    def constrain(self, data) -> float:
        """Project the carrier back onto the rope and apply the ratchet.

        Position: the perpendicular offset is removed outright. Velocity: the
        perpendicular component is cancelled and, when ratcheting, the along-rope
        component is clamped non-negative -- so the hand can drag the carrier up
        but a slip cannot drag it back down.
        """
        a, v = self._slide_adr, self._dof_adr
        self.advance(data.qpos[a : a + 3] + self._origin)
        data.qpos[a : a + 3] = self.route.point_at(self.s) - self._origin
        tangent = self.route.tangent_at(self.s)
        along = float(data.qvel[v : v + 3] @ tangent)
        if self.ratchet:
            along = max(along, 0.0)
        if self._dv:
            # Dry friction: shave a fixed velocity increment off, never past zero.
            along = np.sign(along) * max(0.0, abs(along) - self._dv)
        data.qvel[v : v + 3] = along * tangent
        return self.s

    @property
    def progress(self) -> float:
        """Arc length climbed since reset -- the natural RL progress reward."""
        return self.s - self.s0


# The previous name. It was a mocap body, and it could not slide.
MocapAscender = RopeCarrier

"""Broadcast every body's world pose once per control tick, as bytes.

The JPEG stream ships a PICTURE of the scene (~40 kB a frame, 2 MB/s at 50 Hz)
and the browser can only show it. `app/web/render3d.html` draws the scene
itself, so it needs the geometry once (`app/harness/export_scene.py` writes the
GLB) and then only where everything IS. That is this file: 32 bodies x 7
floats, 916 bytes a tick, 46 kB/s -- one fiftieth of the JPEG stream, and it
buys a camera the server never has to know about.

THE SEAM. `Episode.physics_step_hooks` is a list of `callable(model, data) ->
dict | None` called after EVERY `mj_step`, i.e. at 500 Hz. A pose per physics
substep would be ten times more than any display can use, so the hook COUNTS
and only broadcasts on the last substep of each control tick -- the same instant
`runtime.run` hands `episode.data` to the renderer, so the poses and the JPEG
are the same moment of simulation, not neighbouring ones. The hook returns None
always, which is what keeps it out of the `latest_bms` channel that seam also
carries.

THE MESSAGE (little-endian throughout)

    offset  bytes  what
    0       4      b"POS0"
    4       4      uint32  world key -- FNV-1a 32 of the world's name. The page
                           drops any frame whose key is not the world it has
                           loaded, so the ~1.6 s a first-time world build takes
                           cannot paint one map's poses onto another's mesh.
    8       4      uint32  control tick
    12      4      uint32  nbody
    16      4      uint32  nfoot
    20      n*28   float32 [nbody x 7] xpos xyz then xquat wxyz, WORLD frame,
                           metres and a unit quaternion, in MuJoCo body-index
                           order. The sidecar's `bodies` array is the key from
                           that index to the GLB node name.
    ...     nfoot  uint8   1 while that foot has a live contact with anything.
                           The page paints a footprint decal on the 0 -> 1 edge;
                           it is read from `data.contact`, so it is the solver's
                           own answer, not a height threshold guessed in JS.

WHY WORLD FRAME AND NOT JOINT ANGLES. Joint angles would be smaller still (29
floats), but rebuilding world poses from them means reimplementing MuJoCo's
kinematic tree, its joint frames and the free joint in JavaScript -- three
places for the picture to silently disagree with the physics. `data.xpos` is
the answer MuJoCo already computed and the one its own renderer draws.

STALENESS, stated because it is a real one-substep offset: `mj_step` integrates
`qpos` last, so `xpos` trails it by 2 ms. `mjv_updateScene` reads the same
`xpos`, so the JPEG has exactly the same offset -- the two views agree with each
other, which is what matters here.
"""
import struct
import time

import numpy as np

POSE_MESSAGE_MAGIC = b"POS0"
POSE_HEADER_FORMAT = "<4sIIII"
POSE_HEADER_BYTES = struct.calcsize(POSE_HEADER_FORMAT)
FLOATS_PER_BODY = 7
# How often the cost report goes to stdout. The project rule is that anything
# claimed to be negligible has to have been MEASURED and PRINTED.
COST_REPORT_EVERY_TICKS = 500


def world_key(world_name: str) -> int:
    """FNV-1a 32 of the world name. Five lines here, five lines in the page."""
    value = 0x811C9DC5
    for byte in world_name.encode("utf-8"):
        value = ((value ^ byte) * 0x01000193) & 0xFFFFFFFF
    return value


def _foot_groups(model, meta):
    """[(label, {geom ids})] for the left and right feet, in that order.

    Derived from `meta["foot_geom_ids"]` -- the same list the friction knob
    writes to -- grouped by the BODY each geom hangs off, so it survives a foot
    that is four spheres in one world and one box in another.
    """
    import mujoco
    by_body = {}
    for geom_id in meta.get("foot_geom_ids", []):
        body_id = int(model.geom_bodyid[geom_id])
        by_body.setdefault(body_id, set()).add(int(geom_id))
    groups = []
    for body_id in sorted(by_body):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"
        groups.append((name, by_body[body_id]))
    # Left before right, so the page's two decal channels are stable across
    # worlds however the body ids happen to be ordered.
    groups.sort(key=lambda item: (0 if "left" in item[0] else 1, item[0]))
    return groups


class PoseStream:
    """One world's pose broadcaster. Install with `attach`; it needs no removal.

    A new episode gets a new PoseStream and the old one is dropped with the old
    episode's hook list, so nothing accumulates across map switches.
    """

    def __init__(self, model, meta, server, world_name, substeps):
        self.server = server
        self.world_name = world_name
        self.world_key = world_key(world_name)
        self.substeps = int(substeps)
        self.nbody = int(model.nbody)
        self.foot_groups = _foot_groups(model, meta)
        self.foot_names = [name for name, _ in self.foot_groups]
        self.foot_geom_sets = [geoms for _, geoms in self.foot_groups]
        self.substep_counter = 0
        self.tick = 0
        # Allocated once: this runs 50 times a second forever, and a fresh
        # (nbody, 7) array per tick is garbage the collector has to chase.
        self.poses = np.zeros((self.nbody, FLOATS_PER_BODY), dtype=np.float32)
        self.contacts = bytearray(len(self.foot_groups))
        self.header = struct.pack(
            POSE_HEADER_FORMAT, POSE_MESSAGE_MAGIC, self.world_key, 0,
            self.nbody, len(self.foot_groups))
        self.message_bytes = POSE_HEADER_BYTES + self.nbody * FLOATS_PER_BODY * 4 \
            + len(self.foot_groups)
        self.total_seconds = 0.0
        self.maximum_seconds = 0.0
        self.reported_ticks = 0
        print(f"[pose] {world_name}: {self.nbody} bodies,"
              f" {len(self.foot_groups)} feet {self.foot_names},"
              f" {self.message_bytes} B/tick ="
              f" {self.message_bytes * 50 / 1000:.0f} kB/s at 50 Hz", flush=True)

    # ------------------------------------------------------------- the hook
    def __call__(self, model, data):
        """`callable(model, data) -> None`, appended to physics_step_hooks."""
        self.substep_counter += 1
        if self.substep_counter < self.substeps:
            return None
        self.substep_counter = 0
        self.tick += 1
        started = time.perf_counter()

        self.poses[:, 0:3] = data.xpos
        self.poses[:, 3:7] = data.xquat
        self._read_contacts(data)
        self.server.broadcast(
            struct.pack(POSE_HEADER_FORMAT, POSE_MESSAGE_MAGIC, self.world_key,
                        self.tick, self.nbody, len(self.foot_groups))
            + self.poses.tobytes() + bytes(self.contacts))

        elapsed = time.perf_counter() - started
        self.total_seconds += elapsed
        self.maximum_seconds = max(self.maximum_seconds, elapsed)
        if self.tick - self.reported_ticks >= COST_REPORT_EVERY_TICKS:
            ticks = self.tick - self.reported_ticks
            print(f"[pose] {self.world_name}: {ticks} ticks,"
                  f" mean {self.total_seconds / ticks * 1e6:.0f} us,"
                  f" max {self.maximum_seconds * 1e6:.0f} us"
                  f"  (the control tick is 20000 us)", flush=True)
            self.reported_ticks = self.tick
            self.total_seconds = 0.0
            self.maximum_seconds = 0.0
        return None

    def _read_contacts(self, data):
        """1 while that foot is touching anything, straight off `data.contact`."""
        for index in range(len(self.contacts)):
            self.contacts[index] = 0
        contact_count = int(data.ncon)
        if not contact_count:
            return
        geom1 = np.asarray(data.contact.geom1[:contact_count])
        geom2 = np.asarray(data.contact.geom2[:contact_count])
        for index, geoms in enumerate(self.foot_geom_sets):
            for one, two in zip(geom1, geom2):
                if int(one) in geoms or int(two) in geoms:
                    self.contacts[index] = 1
                    break


def attach(episode, server, world_name):
    """Install a PoseStream on an episode. -> the stream, or None if it cannot.

    Best-effort by design: a broken pose stream must never take the walker down
    with it, exactly as the BMS plugin is best-effort on the same seam.
    """
    if server is None:
        return None
    try:
        stream = PoseStream(episode.model, episode.meta, server, world_name,
                            episode.substeps)
    except Exception as error:                # pragma: no cover - reporting only
        print(f"[pose] NOT attached: {type(error).__name__}: {error}."
              " The JPEG stream and app/web/index.html are unaffected.", flush=True)
        return None
    episode.physics_step_hooks.append(stream)
    return stream

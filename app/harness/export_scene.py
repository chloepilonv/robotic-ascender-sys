"""Freeze a harness world into a glTF binary the browser can render itself.

    ../.venv_everest/bin/python -m app.harness.export_scene --world lhotse_B
    ../.venv_everest/bin/python -m app.harness.export_scene --all

WHY THIS EXISTS. The harness used to push a JPEG per control tick: MuJoCo
rendered a chase camera on the laptop and the picture went down the socket. That
capped the look at whatever MuJoCo's fixed-function renderer does and at
whatever the encoder could carry, for 10-20 ms of a 20 ms control tick.
`app/web/render3d.html` draws the scene in WebGL instead, which needs the
GEOMETRY once (this file) and then only the POSES per tick
(`app/harness/pose_stream.py`) -- and since the 2-D page was retired
(2026-08-30) that is the ONLY way anything is drawn. Nothing here touches
physics: it reads a compiled `MjModel` and writes a file.

WHAT COMES OUT
    app/harness/scene_assets/<world>.glb    one node per MuJoCo BODY, named by
        body name, sitting at the body's world pose at reset. Every visible geom
        of that body is a CHILD node holding a mesh, placed at the geom's
        body-local `geom_pos` / `geom_quat`. So the page moves ONE node per body
        and the geoms come along -- exactly what a `[nbody x 7]` pose message
        can drive.
    app/harness/scene_assets/<world>.json   the sidecar: body index -> node
        name (the pose message is positional, so this is the key that decodes
        it), terrain bounds, the rope polyline, the spawn pose, the sun the
        recorded mp4 uses, and the foot bodies the page paints footprints from.

FRAME. MuJoCo is Z-up and so is this file -- no axis conversion anywhere. glTF
conventionally means Y-up, but nothing in the format requires it, and rotating
the world would mean rotating every streamed pose too. The page sets
`THREE.Object3D.DEFAULT_UP` to (0, 0, 1) instead, once.

WHAT IS SKIPPED, and why it is safe
    * geoms in group 3+ -- MuJoCo's own default view mask is [1,1,1,0,0,0], so
      these are already invisible in the JPEG. On the G1 that is the collision
      primitive set: capsules approximating limbs that already have visual
      meshes in group 2. Drawing both would double every limb.
    * geoms with alpha 0 -- `Episode._set_ascender_visible` hides the rope and
      carrier on rope-off worlds this way, and the exporter honours the same
      flag so a "free walk" world exports without a rope hanging in the air.
Neither is a physics change; both are what the existing render already shows.

THE SNOW SHELL (2026-08-30). Some worlds stand the robot on a perfectly smooth
physics ground -- `flat_free`'s zero-elevation heightfield, and the tilted PLANE
under every chloe/mrinal ascender world -- which draws as a featureless slab.
Those worlds get a second, DECORATIVE terrain mesh: the measured Lhotse relief
(`rl.environment.terrain.load_patch("B").rough`, mean-zero, RMS 0.1138 m) laid
2 cm above the ground in the ground geom's own frame, so it tilts with the
slope. The feet then sink into it while the robot really stands on the flat
plane -- which is what walking in powder looks like. It cannot touch physics:
this file only READS a compiled model and writes a display file the browser
draws. Detection is geometric (a plane, or a heightfield whose relief standard
deviation is under 1 cm), never a list of world names, so a rough world can
never pick it up by accident.

TERRAIN DECIMATION. The measured Lhotse patches are 25 x 15 m at 5 cm, which is
~300k triangles -- kept whole, because the roughness IS the terrain. The 120 m
sandbox at the same resolution would be 11.5M, so the grid is strided down
until it fits `--max-terrain-triangles` (default 500k) and the sidecar records
the stride that was used.
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

_REPOSITORY_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, _REPOSITORY_ROOT)

import mujoco  # noqa: E402

from app.harness import climb_worlds as climb_worlds_module  # noqa: E402
from app.harness import graphics as graphics_module  # noqa: E402
from app.harness import worlds as worlds_module  # noqa: E402

SCENE_ASSETS_DIRECTORY = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "scene_assets")

# MuJoCo's own default view mask is [1,1,1,0,0,0]: groups 0-2 draw, 3+ do not.
VISIBLE_GEOM_GROUPS = (0, 1, 2)
# A geom whose rgba is exactly this never had a colour set; the material (if
# any) is then the authority. Same rule mjv_updateScene uses.
MUJOCO_DEFAULT_GEOM_RGBA = (0.5, 0.5, 0.5, 1.0)
DEFAULT_MAXIMUM_TERRAIN_TRIANGLES = 500_000
# An infinite MuJoCo plane has size 0; give it something a camera can stand on.
INFINITE_PLANE_HALF_EXTENT_METERS = 120.0

# ------------------------------------------------------- the snow shell
# A world whose PHYSICS ground is a perfectly smooth plane (flat_free, and
# every chloe/mrinal ascender world, whose floor is one tilted plane) draws as
# a featureless slab. The shell is the measured Lhotse relief laid a couple of
# centimetres ABOVE that plane, VISUAL ONLY: it is a mesh in a .glb, the file
# the browser draws, and nothing here or downstream is ever compiled into
# MuJoCo. Physics is bit-identical with or without it -- the robot still stands
# on the flat plane, and the shell simply closes over its feet, which is what
# walking in powder looks like.
SNOW_SHELL_NODE_NAME = "snow_shell"
SNOW_SHELL_PATCH_NAME = "B"
# Mean-zero relief, so this is the height of the shell's MEAN plane above the
# physics plane. Crests then stand ~0.25 m proud and dips fall BELOW the floor,
# where the floor -- still drawn, still opaque -- hides them.
SNOW_SHELL_LIFT_METERS = 0.02
# 1.0 = the measured patch at full amplitude (RMS 0.1138 m).
SNOW_SHELL_AMPLITUDE_SCALE = 1.0
# The slab is 25 x 15 m and the climb runs uphill out of the spawn, so the slab
# is pushed this far along the route: the whole climb stays on snow.
SNOW_SHELL_FORWARD_OFFSET_METERS = 8.0
# The rim fades to the mean plane over this distance, so the boundary is a 2 cm
# lip rather than a row of crests sliced off in mid-air.
SNOW_SHELL_EDGE_TAPER_METERS = 1.5
# Decoration gets a smaller budget than the terrain it imitates. The patch is
# 300 x 500 at 5 cm = 298k triangles, which fits whole.
SNOW_SHELL_MAXIMUM_TRIANGLES = 300_000
# Below this relief standard deviation a ground geom is FLAT for our purposes:
# flat_free's floor is a heightfield whose elevation scale is exactly 0.
SMOOTH_TERRAIN_STANDARD_DEVIATION_METERS = 0.01

# glTF component types
UNSIGNED_INT = 5125
FLOAT = 5126
ARRAY_BUFFER = 34962
ELEMENT_ARRAY_BUFFER = 34963


# ----------------------------------------------------------------- primitives
# Every helper returns (positions (n,3) float32, normals (n,3) float32,
# indices (m,3) uint32) in the geom's own local frame, MuJoCo's convention:
# capsules and cylinders run along local +z, box sizes are HALF-extents.
def _unit_sphere(segments=20, rings=14):
    positions, normals, indices = [], [], []
    for ring in range(rings + 1):
        polar = np.pi * ring / rings
        for segment in range(segments + 1):
            azimuth = 2.0 * np.pi * segment / segments
            direction = np.array([np.sin(polar) * np.cos(azimuth),
                                  np.sin(polar) * np.sin(azimuth),
                                  np.cos(polar)])
            positions.append(direction)
            normals.append(direction)
    for ring in range(rings):
        for segment in range(segments):
            a = ring * (segments + 1) + segment
            b = a + segments + 1
            indices.append([a, b, a + 1])
            indices.append([a + 1, b, b + 1])
    return (np.asarray(positions, np.float32), np.asarray(normals, np.float32),
            np.asarray(indices, np.uint32))


def sphere_mesh(radius, segments=20, rings=14):
    positions, normals, indices = _unit_sphere(segments, rings)
    return positions * float(radius), normals, indices


def ellipsoid_mesh(radii, segments=20, rings=14):
    positions, normals, indices = _unit_sphere(segments, rings)
    radii = np.asarray(radii, np.float32)
    scaled = positions * radii
    # The normal of an ellipsoid is not the scaled sphere normal; it is the
    # gradient of x^2/a^2 + ..., i.e. the point divided by the radii SQUARED.
    gradient = scaled / np.maximum(radii * radii, 1e-12)
    lengths = np.linalg.norm(gradient, axis=1, keepdims=True)
    return scaled, (gradient / np.maximum(lengths, 1e-12)).astype(np.float32), indices


def capsule_mesh(radius, half_length, segments=20, rings=14):
    """A cylinder of half-height `half_length` capped by two hemispheres."""
    radius, half_length = float(radius), float(half_length)
    positions, normals, indices = [], [], []
    rows = []
    # bottom hemisphere: polar pi/2 -> pi, offset to -half_length
    for ring in range(rings // 2 + 1):
        polar = np.pi / 2 + (np.pi / 2) * ring / (rings // 2)
        rows.append((polar, -half_length))
    # top hemisphere: polar pi/2 -> 0, offset to +half_length. Reversed so the
    # rows run bottom-to-top and the strip between the two seams is the barrel.
    top = []
    for ring in range(rings // 2 + 1):
        polar = np.pi / 2 - (np.pi / 2) * ring / (rings // 2)
        top.append((polar, +half_length))
    rows = list(reversed(rows)) + top      # bottom pole ... equator, equator ... top pole
    for polar, offset in rows:
        for segment in range(segments + 1):
            azimuth = 2.0 * np.pi * segment / segments
            direction = np.array([np.sin(polar) * np.cos(azimuth),
                                  np.sin(polar) * np.sin(azimuth),
                                  np.cos(polar)])
            normals.append(direction)
            positions.append(direction * radius + np.array([0.0, 0.0, offset]))
    for row in range(len(rows) - 1):
        for segment in range(segments):
            a = row * (segments + 1) + segment
            b = a + segments + 1
            indices.append([a, a + 1, b])
            indices.append([a + 1, b + 1, b])
    return (np.asarray(positions, np.float32), np.asarray(normals, np.float32),
            np.asarray(indices, np.uint32))


def cylinder_mesh(radius, half_height, segments=24):
    radius, half_height = float(radius), float(half_height)
    positions, normals, indices = [], [], []
    for segment in range(segments + 1):
        azimuth = 2.0 * np.pi * segment / segments
        side = np.array([np.cos(azimuth), np.sin(azimuth), 0.0])
        positions.append(side * radius + [0, 0, -half_height])
        normals.append(side)
        positions.append(side * radius + [0, 0, +half_height])
        normals.append(side)
    for segment in range(segments):
        a = 2 * segment
        indices.append([a, a + 2, a + 1])
        indices.append([a + 1, a + 2, a + 3])
    for sign in (-1.0, +1.0):
        center = len(positions)
        positions.append([0.0, 0.0, sign * half_height])
        normals.append([0.0, 0.0, sign])
        for segment in range(segments + 1):
            azimuth = 2.0 * np.pi * segment / segments
            positions.append([radius * np.cos(azimuth), radius * np.sin(azimuth),
                              sign * half_height])
            normals.append([0.0, 0.0, sign])
        for segment in range(segments):
            a = center + 1 + segment
            triangle = [center, a, a + 1] if sign > 0 else [center, a + 1, a]
            indices.append(triangle)
    return (np.asarray(positions, np.float32), np.asarray(normals, np.float32),
            np.asarray(indices, np.uint32))


def box_mesh(half_extents):
    half_x, half_y, half_z = [float(v) for v in half_extents]
    faces = [
        ((+1, 0, 0), [(+half_x, -half_y, -half_z), (+half_x, +half_y, -half_z),
                      (+half_x, +half_y, +half_z), (+half_x, -half_y, +half_z)]),
        ((-1, 0, 0), [(-half_x, +half_y, -half_z), (-half_x, -half_y, -half_z),
                      (-half_x, -half_y, +half_z), (-half_x, +half_y, +half_z)]),
        ((0, +1, 0), [(+half_x, +half_y, -half_z), (-half_x, +half_y, -half_z),
                      (-half_x, +half_y, +half_z), (+half_x, +half_y, +half_z)]),
        ((0, -1, 0), [(-half_x, -half_y, -half_z), (+half_x, -half_y, -half_z),
                      (+half_x, -half_y, +half_z), (-half_x, -half_y, +half_z)]),
        ((0, 0, +1), [(-half_x, -half_y, +half_z), (+half_x, -half_y, +half_z),
                      (+half_x, +half_y, +half_z), (-half_x, +half_y, +half_z)]),
        ((0, 0, -1), [(-half_x, +half_y, -half_z), (+half_x, +half_y, -half_z),
                      (+half_x, -half_y, -half_z), (-half_x, -half_y, -half_z)]),
    ]
    positions, normals, indices = [], [], []
    for normal, corners in faces:
        base = len(positions)
        positions.extend(corners)
        normals.extend([normal] * 4)
        indices.append([base, base + 1, base + 2])
        indices.append([base, base + 2, base + 3])
    return (np.asarray(positions, np.float32), np.asarray(normals, np.float32),
            np.asarray(indices, np.uint32))


def plane_mesh(half_x, half_y):
    half_x = float(half_x) or INFINITE_PLANE_HALF_EXTENT_METERS
    half_y = float(half_y) or INFINITE_PLANE_HALF_EXTENT_METERS
    positions = np.asarray([[-half_x, -half_y, 0], [half_x, -half_y, 0],
                            [half_x, half_y, 0], [-half_x, half_y, 0]], np.float32)
    normals = np.tile(np.asarray([[0, 0, 1]], np.float32), (4, 1))
    indices = np.asarray([[0, 1, 2], [0, 2, 3]], np.uint32)
    return positions, normals, indices


def mujoco_mesh(model, mesh_id):
    """A compiled MuJoCo mesh, UNWELDED so its own normals survive.

    MuJoCo stores positions and normals in two independently indexed arrays
    (`mesh_face` into `mesh_vert`, `mesh_facenormal` into `mesh_normal`), which
    a single glTF index buffer cannot express. Expanding to three vertices per
    triangle is the honest translation: it keeps the hard edges MuJoCo draws
    (a re-welded smooth normal turns the G1's boots into blobs) and costs
    memory that is nothing beside the terrain.
    """
    vertex_start = int(model.mesh_vertadr[mesh_id])
    vertex_count = int(model.mesh_vertnum[mesh_id])
    face_start = int(model.mesh_faceadr[mesh_id])
    face_count = int(model.mesh_facenum[mesh_id])
    vertices = np.asarray(
        model.mesh_vert[vertex_start:vertex_start + vertex_count], np.float32)
    faces = np.asarray(model.mesh_face[face_start:face_start + face_count], np.int64)
    positions = vertices[faces.reshape(-1)]

    normals = None
    if hasattr(model, "mesh_facenormal") and hasattr(model, "mesh_normal"):
        normal_start = int(model.mesh_normaladr[mesh_id])
        face_normals = np.asarray(
            model.mesh_facenormal[face_start:face_start + face_count], np.int64)
        if face_normals.size and face_normals.max() >= 0:
            source = np.asarray(model.mesh_normal, np.float32)
            normals = source[normal_start + face_normals.reshape(-1)]
    if normals is None:
        triangles = positions.reshape(-1, 3, 3)
        face_normal = np.cross(triangles[:, 1] - triangles[:, 0],
                               triangles[:, 2] - triangles[:, 0])
        lengths = np.linalg.norm(face_normal, axis=1, keepdims=True)
        face_normal = face_normal / np.maximum(lengths, 1e-12)
        normals = np.repeat(face_normal, 3, axis=0).astype(np.float32)
    # Re-weld only where BOTH position and normal agree: hard edges keep their
    # duplicate vertices, smooth surfaces collapse back to one each. Measured on
    # lhotse_B this is the difference between a 46 MB and a 17 MB download, and
    # the picture is bit-identical because nothing merged had different data.
    attributes = np.concatenate(
        [positions.astype(np.float32), normals.astype(np.float32)], axis=1)
    _unique, first, inverse = np.unique(
        attributes, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first)                 # keep the original vertex order
    remap = np.empty(order.shape[0], np.int64)
    remap[order] = np.arange(order.shape[0])
    indices = remap[inverse.reshape(-1)].astype(np.uint32).reshape(-1, 3)
    kept = first[order]
    return (positions[kept].astype(np.float32), normals[kept].astype(np.float32),
            indices)


def grid_surface_mesh(sampled, x, y):
    """A regular height grid -> (positions (n,3), normals (n,3), indices (m,3)).

    `sampled[row, column]` is the height in metres at (`x[column]`, `y[row]`),
    all three in the SAME local frame; row runs along +y and column along +x,
    which is MuJoCo's heightfield convention. Shared by the terrain itself and
    by the decorative snow shell, so the two get identical triangulation and
    identical (analytic, central-difference) normals.
    """
    grid_x, grid_y = np.meshgrid(x, y)
    positions = np.stack([grid_x, grid_y, sampled], axis=-1).astype(np.float32)

    # Central differences on the SAMPLED grid, so the normals belong to the
    # triangles actually written rather than to a resolution nobody can see.
    gradient_y, gradient_x = np.gradient(sampled, y, x)
    normals = np.stack([-gradient_x, -gradient_y, np.ones_like(sampled)], axis=-1)
    normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-12)

    height, width = sampled.shape
    corner = (np.arange(height - 1)[:, None] * width + np.arange(width - 1)[None, :])
    indices = np.stack([
        np.stack([corner, corner + 1, corner + width], axis=-1),
        np.stack([corner + 1, corner + width + 1, corner + width], axis=-1),
    ], axis=-2).reshape(-1, 3).astype(np.uint32)
    return (positions.reshape(-1, 3), normals.reshape(-1, 3).astype(np.float32),
            indices)


def heightfield_mesh(model, hfield_id, maximum_triangles):
    """The terrain grid, with analytic normals. -> (positions, normals, indices, report)

    MuJoCo stores the field normalised to [0, 1] with the vertical scale in
    `hfield_size` = (radius_x, radius_y, elevation_z, base_z), and the geom's
    own pos/quat then places it. Row index runs along +y, column along +x.
    """
    rows = int(model.hfield_nrow[hfield_id])
    columns = int(model.hfield_ncol[hfield_id])
    address = int(model.hfield_adr[hfield_id])
    radius_x, radius_y, elevation_z, _base_z = [
        float(v) for v in model.hfield_size[hfield_id]]
    grid = np.asarray(
        model.hfield_data[address:address + rows * columns], np.float64
    ).reshape(rows, columns)

    stride = 1
    while ((rows - 1) // stride) * ((columns - 1) // stride) * 2 > maximum_triangles:
        stride += 1
    row_index = np.arange(0, rows, stride)
    column_index = np.arange(0, columns, stride)
    if row_index[-1] != rows - 1:
        row_index = np.append(row_index, rows - 1)
    if column_index[-1] != columns - 1:
        column_index = np.append(column_index, columns - 1)
    sampled = grid[np.ix_(row_index, column_index)] * elevation_z

    x = (2.0 * column_index / (columns - 1) - 1.0) * radius_x
    y = (2.0 * row_index / (rows - 1) - 1.0) * radius_y
    positions, normals, indices = grid_surface_mesh(sampled, x, y)

    height, width = sampled.shape
    report = {
        "rows": rows, "columns": columns, "stride": stride,
        "sampled_rows": int(height), "sampled_columns": int(width),
        "triangles": int(indices.shape[0]),
        "resolution_meters": [float(2 * radius_x / (columns - 1) * stride),
                              float(2 * radius_y / (rows - 1) * stride)],
        "half_extent_meters": [radius_x, radius_y],
        "elevation_meters": elevation_z,
    }
    return positions, normals, indices, report


# -------------------------------------------------------------- the snow shell
_SNOW_PATCH_CACHE = {}


def _snow_patch(patch_name):
    """The measured Lhotse patch, loaded once. -> rl.environment.terrain.Terrain

    `patch.rough` is the mean-zero relief grid with the macro slope already
    removed (25 x 15 m at 5 cm for patch B, RMS 0.1138 m). That grid IS the
    Lhotse ground shape; nothing here invents noise.
    """
    if patch_name not in _SNOW_PATCH_CACHE:
        from rl.environment import terrain as terrain_module
        _SNOW_PATCH_CACHE[patch_name] = terrain_module.load_patch(patch_name)
    return _SNOW_PATCH_CACHE[patch_name]


def terrain_relief_standard_deviation(model, geom_id):
    """How rough this ground geom really is, in metres. -> float or None

    0.0 for a PLANE (perfectly smooth by construction), and the standard
    deviation of the scaled elevation for a HFIELD -- which is 0 for
    flat_free, whose floor is a heightfield with elevation scale 0, and
    ~0.11 m for every measured Lhotse patch. None when the geom is not ground.
    """
    geom_type = int(model.geom_type[geom_id])
    if geom_type == mujoco.mjtGeom.mjGEOM_PLANE:
        return 0.0
    if geom_type != mujoco.mjtGeom.mjGEOM_HFIELD:
        return None
    hfield_id = int(model.geom_dataid[geom_id])
    rows = int(model.hfield_nrow[hfield_id])
    columns = int(model.hfield_ncol[hfield_id])
    address = int(model.hfield_adr[hfield_id])
    elevation_z = float(model.hfield_size[hfield_id][2])
    grid = np.asarray(
        model.hfield_data[address:address + rows * columns], np.float64)
    return float(np.std(grid * elevation_z))


def _smoothstep(edge_low, edge_high, value):
    ramp = np.clip((value - edge_low) / max(edge_high - edge_low, 1e-9), 0.0, 1.0)
    return ramp * ramp * (3.0 - 2.0 * ramp)


def snow_shell_mesh(center_local_xy, lift_meters=SNOW_SHELL_LIFT_METERS,
                    amplitude_scale=SNOW_SHELL_AMPLITUDE_SCALE,
                    patch_name=SNOW_SHELL_PATCH_NAME,
                    maximum_triangles=SNOW_SHELL_MAXIMUM_TRIANGLES,
                    edge_taper_meters=SNOW_SHELL_EDGE_TAPER_METERS):
    """The decorative snow surface. -> (positions, normals, indices, report)

    Built in the TERRAIN GEOM'S LOCAL FRAME -- the node carries that geom's own
    pos/quat, so on a tilted chloe world the whole slab tilts with the displayed
    slope for free and the lift stays perpendicular to the ground the robot
    stands on. `center_local_xy` slides the 25 x 15 m slab along the route.
    """
    patch = _snow_patch(patch_name)
    rough = np.asarray(patch.rough, np.float64)
    resolution = float(patch.res)
    rows, columns = rough.shape

    stride = 1
    while ((rows - 1) // stride) * ((columns - 1) // stride) * 2 > maximum_triangles:
        stride += 1
    row_index = np.arange(0, rows, stride)
    column_index = np.arange(0, columns, stride)
    if row_index[-1] != rows - 1:
        row_index = np.append(row_index, rows - 1)
    if column_index[-1] != columns - 1:
        column_index = np.append(column_index, columns - 1)

    x = (column_index - (columns - 1) / 2.0) * resolution + float(center_local_xy[0])
    y = (row_index - (rows - 1) / 2.0) * resolution + float(center_local_xy[1])
    sampled = rough[np.ix_(row_index, column_index)] * float(amplitude_scale)

    # Fade the rim back to the mean plane, so the slab ends in a 2 cm lip
    # instead of a row of crests sliced off in mid-air.
    distance_x = np.minimum(x - x[0], x[-1] - x)
    distance_y = np.minimum(y - y[0], y[-1] - y)
    window = (_smoothstep(0.0, edge_taper_meters, distance_y)[:, None]
              * _smoothstep(0.0, edge_taper_meters, distance_x)[None, :])
    relief = sampled * window
    heights = relief + float(lift_meters)

    positions, normals, indices = grid_surface_mesh(heights, x, y)
    report = {
        "patch": patch_name,
        "amplitude_scale": float(amplitude_scale),
        "lift_meters": float(lift_meters),
        "stride": int(stride),
        "resolution_meters": float(resolution * stride),
        "extent_meters": [float(x[-1] - x[0]), float(y[-1] - y[0])],
        "center_local_meters": [round(float(center_local_xy[0]), 4),
                                round(float(center_local_xy[1]), 4)],
        "relief_rms_meters": float(np.sqrt(np.mean(relief ** 2))),
        "relief_minimum_meters": float(relief.min()),
        "relief_maximum_meters": float(relief.max()),
        "triangles": int(indices.shape[0]),
        "vertices": int(positions.shape[0]),
    }
    return positions, normals, indices, report


def snow_shell_center_local(model, data, terrain_geom_id, half_extent_meters,
                            slab_extent_meters,
                            forward_offset=SNOW_SHELL_FORWARD_OFFSET_METERS):
    """Where to park the slab, in the terrain geom's local x/y. -> (x, y)

    Centred on the robot's spawn and pushed `forward_offset` along the route
    (the rope's far end when there is a rope, the local +x fall line when there
    is not -- the chloe robots gain ~6 m uphill in 15 s, so a slab centred on
    the spawn would run out from under them). Then clamped so the slab never
    hangs off the ground geom it decorates: on flat_free, whose floor is
    exactly 25 x 15 m, that pins it dead centre.
    """
    rotation = np.asarray(data.geom_xmat[terrain_geom_id], np.float64).reshape(3, 3)
    origin = np.asarray(data.geom_xpos[terrain_geom_id], np.float64)

    def to_local(point):
        return rotation.T @ (np.asarray(point, np.float64) - origin)

    spawn_local = to_local(data.qpos[0:3])
    forward = np.array([1.0, 0.0])
    polyline = rope_polyline(model, data)
    if len(polyline) >= 2:
        ends = [to_local(polyline[0])[:2], to_local(polyline[-1])[:2]]
        far = max(ends, key=lambda end: float(np.linalg.norm(end - spawn_local[:2])))
        direction = far - spawn_local[:2]
        if np.linalg.norm(direction) > 1e-6:
            forward = direction / np.linalg.norm(direction)

    center = spawn_local[:2] + forward * float(forward_offset)
    for axis in (0, 1):
        room = max(float(half_extent_meters[axis])
                   - 0.5 * float(slab_extent_meters[axis]), 0.0)
        center[axis] = float(np.clip(center[axis], -room, room))
    return center


# ------------------------------------------------------------- the glb writer
class GlbBuilder:
    """Minimal glTF 2.0 binary writer: positions, normals, indices, PBR, nodes.

    Deliberately hand-rolled rather than pulled from a library: the whole format
    we need is one buffer, one bufferView per accessor and a flat node list, and
    a dependency that has to be installed at a hackathon is a dependency that
    can fail at a hackathon.
    """

    def __init__(self):
        self.binary = bytearray()
        self.buffer_views = []
        self.accessors = []
        self.meshes = []
        self.materials = []
        self.nodes = []

    def _align(self):
        while len(self.binary) % 4:
            self.binary.append(0)

    def _add_view(self, data: bytes, target=None) -> int:
        self._align()
        offset = len(self.binary)
        self.binary.extend(data)
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
        if target is not None:
            view["target"] = target
        self.buffer_views.append(view)
        return len(self.buffer_views) - 1

    def _add_accessor(self, array, component_type, type_name, target,
                      with_bounds=False) -> int:
        view = self._add_view(array.tobytes(), target)
        accessor = {"bufferView": view, "componentType": component_type,
                    "count": int(array.shape[0]), "type": type_name}
        if with_bounds:
            accessor["min"] = [float(v) for v in array.min(axis=0)]
            accessor["max"] = [float(v) for v in array.max(axis=0)]
        self.accessors.append(accessor)
        return len(self.accessors) - 1

    def add_material(self, name, rgba, metallic, roughness) -> int:
        red, green, blue, alpha = [float(v) for v in rgba]
        material = {
            "name": name,
            "pbrMetallicRoughness": {
                "baseColorFactor": [red, green, blue, alpha],
                "metallicFactor": float(metallic),
                "roughnessFactor": float(roughness),
            },
            "doubleSided": True,
        }
        if alpha < 0.999:
            material["alphaMode"] = "BLEND"
        self.materials.append(material)
        return len(self.materials) - 1

    def add_mesh(self, name, positions, normals, indices, material) -> int:
        position_accessor = self._add_accessor(
            np.ascontiguousarray(positions, np.float32), FLOAT, "VEC3",
            ARRAY_BUFFER, with_bounds=True)
        normal_accessor = self._add_accessor(
            np.ascontiguousarray(normals, np.float32), FLOAT, "VEC3", ARRAY_BUFFER)
        index_accessor = self._add_accessor(
            np.ascontiguousarray(indices.reshape(-1), np.uint32),
            UNSIGNED_INT, "SCALAR", ELEMENT_ARRAY_BUFFER)
        self.meshes.append({
            "name": name,
            "primitives": [{
                "attributes": {"POSITION": position_accessor,
                               "NORMAL": normal_accessor},
                "indices": index_accessor,
                "material": material,
            }],
        })
        return len(self.meshes) - 1

    def add_node(self, name, translation=None, rotation_wxyz=None, mesh=None,
                 children=None) -> int:
        node = {"name": name}
        if translation is not None:
            node["translation"] = [float(v) for v in translation]
        if rotation_wxyz is not None:
            w, x, y, z = [float(v) for v in rotation_wxyz]
            node["rotation"] = [x, y, z, w]          # glTF is xyzw
        if mesh is not None:
            node["mesh"] = int(mesh)
        if children:
            node["children"] = [int(child) for child in children]
        self.nodes.append(node)
        return len(self.nodes) - 1

    def write(self, path, root_nodes, generator="app/harness/export_scene.py"):
        self._align()
        document = {
            "asset": {"version": "2.0", "generator": generator},
            "scene": 0,
            "scenes": [{"nodes": [int(node) for node in root_nodes]}],
            "nodes": self.nodes,
            "meshes": self.meshes,
            "materials": self.materials,
            "accessors": self.accessors,
            "bufferViews": self.buffer_views,
            "buffers": [{"byteLength": len(self.binary)}],
        }
        json_chunk = json.dumps(document, separators=(",", ":")).encode("utf-8")
        json_chunk += b" " * ((4 - len(json_chunk) % 4) % 4)
        binary_chunk = bytes(self.binary)
        total = 12 + 8 + len(json_chunk) + 8 + len(binary_chunk)
        with open(path, "wb") as handle:
            handle.write(struct.pack("<III", 0x46546C67, 2, total))
            handle.write(struct.pack("<II", len(json_chunk), 0x4E4F534A))
            handle.write(json_chunk)
            handle.write(struct.pack("<II", len(binary_chunk), 0x004E4942))
            handle.write(binary_chunk)
        return total


# ------------------------------------------------------------------ the scene
def _name_of(model, object_type, index, fallback):
    name = mujoco.mj_id2name(model, object_type, index)
    return name if name else fallback


def geom_appearance(model, geom_id):
    """(rgba, metallic, roughness) the way MuJoCo's own renderer resolves it."""
    material_id = int(model.geom_matid[geom_id])
    rgba = np.asarray(model.geom_rgba[geom_id], np.float64)
    metallic, roughness = 0.0, 0.85
    if material_id >= 0:
        rgba = np.asarray(model.mat_rgba[material_id], np.float64)
        shininess = float(model.mat_shininess[material_id])
        roughness = float(np.clip(1.0 - shininess, 0.05, 1.0))
        if hasattr(model, "mat_metallic"):
            value = float(model.mat_metallic[material_id])
            if value >= 0.0:
                metallic = float(np.clip(value, 0.0, 1.0))
        if hasattr(model, "mat_roughness"):
            value = float(model.mat_roughness[material_id])
            if value >= 0.0:
                roughness = float(np.clip(value, 0.05, 1.0))
    # An explicitly coloured geom overrides its material, same as mjv does.
    if not np.allclose(model.geom_rgba[geom_id], MUJOCO_DEFAULT_GEOM_RGBA):
        rgba = np.asarray(model.geom_rgba[geom_id], np.float64)
    return rgba, metallic, roughness


def geom_geometry(model, geom_id, maximum_terrain_triangles):
    """-> (positions, normals, indices, report) or None if we do not draw it."""
    geom_type = int(model.geom_type[geom_id])
    size = np.asarray(model.geom_size[geom_id], np.float64)
    types = mujoco.mjtGeom
    if geom_type == types.mjGEOM_PLANE:
        # A PLANE IS A TERRAIN TOO. The report is what tells the page which
        # nodes are ground -- it is what earns the snow/rock/ice shader and
        # the footprint decal canvas -- and a world whose ground is one flat
        # geom (Chloe's rope worlds) would otherwise get an untextured slab
        # and no footprints. Two rows and two columns, no relief: honest for a
        # plane, and the same shape `heightfield_mesh` returns.
        half_x = float(size[0]) or INFINITE_PLANE_HALF_EXTENT_METERS
        half_y = float(size[1]) or INFINITE_PLANE_HALF_EXTENT_METERS
        report = {
            "rows": 2, "columns": 2, "stride": 1,
            "sampled_rows": 2, "sampled_columns": 2, "triangles": 2,
            "resolution_meters": [2.0 * half_x, 2.0 * half_y],
            "half_extent_meters": [half_x, half_y],
            "elevation_meters": 0.0,
        }
        return plane_mesh(size[0], size[1]) + (report,)
    if geom_type == types.mjGEOM_HFIELD:
        positions, normals, indices, report = heightfield_mesh(
            model, int(model.geom_dataid[geom_id]), maximum_terrain_triangles)
        return positions, normals, indices, report
    if geom_type == types.mjGEOM_SPHERE:
        return sphere_mesh(size[0]) + (None,)
    if geom_type == types.mjGEOM_CAPSULE:
        return capsule_mesh(size[0], size[1]) + (None,)
    if geom_type == types.mjGEOM_ELLIPSOID:
        return ellipsoid_mesh(size[:3]) + (None,)
    if geom_type == types.mjGEOM_CYLINDER:
        return cylinder_mesh(size[0], size[1]) + (None,)
    if geom_type == types.mjGEOM_BOX:
        return box_mesh(size[:3]) + (None,)
    if geom_type in (types.mjGEOM_MESH, getattr(types, "mjGEOM_SDF", -1)):
        mesh_id = int(model.geom_dataid[geom_id])
        if mesh_id < 0:
            return None
        return mujoco_mesh(model, mesh_id) + (None,)
    return None


def rope_polyline(model, data):
    """The fixed line's centreline in world coordinates.

    `climb_scene.build_scene` lays the rope down as capsules named `ropeseg0..n`
    along the route, so the polyline is each capsule's two endpoints in order,
    de-duplicated where they meet. Empty on a world that has no such geoms.
    """
    segments = []
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if not name or not name.startswith("ropeseg"):
            continue
        try:
            order = int(name[len("ropeseg"):])
        except ValueError:
            continue
        half_length = float(model.geom_size[geom_id][1])
        center = np.asarray(data.geom_xpos[geom_id], np.float64)
        axis = np.asarray(data.geom_xmat[geom_id], np.float64).reshape(3, 3)[:, 2]
        segments.append((order, center - axis * half_length, center + axis * half_length))
    segments.sort(key=lambda item: item[0])
    points = []
    for _order, start, end in segments:
        if not points or np.linalg.norm(points[-1] - start) > 1e-6:
            points.append(start)
        points.append(end)
    return [[round(float(v), 5) for v in point] for point in points]


def open_world(world_name, plain_graphics=False, with_guide=True):
    """(model, data, meta, definition) at the world's reset pose.

    Uses the harness's own loaders, so a world exported here is the world the
    runtime opens -- including `apply_alpine_look`, which whitens the terrain
    geom. The skybox is deliberately NOT added: it is a MuJoCo texture asset
    the browser has no use for, and adding it recompiles the model.
    """
    world_name = worlds_module.resolve_world_name(world_name)
    if world_name not in worlds_module.WORLD_DEFINITIONS:
        raise KeyError(f"unknown world {world_name!r};"
                       f" have {worlds_module.world_names()}")
    definition = worlds_module.WORLD_DEFINITIONS[world_name]
    look = None
    if definition["kind"] in ("climb_scene", "chloe_ascender"):
        # Chloe's worlds build a different plant behind the same scene surface
        # (app/harness/chloe_worlds.py), so everything below -- the guide
        # surgery, the alpine look, the reset, the rope -- is shared verbatim.
        # Her ground is one PLANE rather than a heightfield; `geom_geometry`
        # reports a plane as terrain for exactly this reason, so the page still
        # gets its snow shader and its footprint canvas.
        if definition["kind"] == "chloe_ascender":
            from app.harness import chloe_worlds as chloe_worlds_module
            scene, meta, definition = chloe_worlds_module.ChloeSceneLibrary().load(world_name)
        else:
            scene, meta, definition = climb_worlds_module.ClimbSceneLibrary().load(world_name)
        # THE GUIDE'S SURGERY GOES FIRST, exactly as `runtime.open_world` does
        # it. It recompiles the spec and appends the guide's mocap body, which
        # takes nbody from 32 to 33 -- so an export that skipped it would hand
        # the page a scene one body short of every pose message. Best-effort:
        # `guide.py` is someone else's file and a broken import must not stop
        # the exporter, only cost the picture its human.
        if with_guide:
            try:
                from app.harness import guide as guide_module
                guide_module.attach_guide(scene)
            except Exception as error:        # pragma: no cover - reporting only
                print(f"[export] guide NOT attached: {type(error).__name__}:"
                      f" {error}. The scene exports without the human.", flush=True)
        model, data = scene.model, scene.data
        if not plain_graphics:
            look = graphics_module.apply_alpine_look(
                model, terrain_size_meters=scene.terrain.size_xy)
        # His reset: the keyframe, mj_forward, the carrier placed on the line.
        scene.reset()
        # `ClimbSceneEpisode._hide_rope_apparatus`, verbatim rule: a rope-off
        # world hides every group-1 geom (rope AND carrier), alpha only.
        if not definition["rope"]:
            from rl.environment import climb_scene as climb_scene_module
            for geom_id in range(model.ngeom):
                if int(model.geom_group[geom_id]) == climb_scene_module.GROUP_ROPE:
                    model.geom_rgba[geom_id, 3] = 0.0
    else:
        model, meta, definition = worlds_module.WorldLibrary(
            write_fingerprint=False).load(world_name)
        if not plain_graphics:
            look = graphics_module.apply_alpine_look(model)
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        data.qpos[:] = meta["keyframe_qpos"]
        data.qvel[:] = 0.0
        if not definition["rope"]:
            for geom_id in worlds_module.ascender_geom_ids(model, meta):
                model.geom_rgba[geom_id, 3] = 0.0
    mujoco.mj_forward(model, data)
    return model, data, meta, definition, look


def export_world(world_name, output_directory=SCENE_ASSETS_DIRECTORY,
                 maximum_terrain_triangles=DEFAULT_MAXIMUM_TERRAIN_TRIANGLES,
                 plain_graphics=False, with_guide=True,
                 snow_shell=True,
                 snow_shell_scale=SNOW_SHELL_AMPLITUDE_SCALE,
                 snow_shell_lift=SNOW_SHELL_LIFT_METERS):
    model, data, meta, definition, look = open_world(
        world_name, plain_graphics, with_guide)
    os.makedirs(output_directory, exist_ok=True)

    builder = GlbBuilder()
    material_cache, terrain_report, snow_report = {}, None, None
    terrain_node_names, rope_node_names = [], []
    body_nodes, root_nodes = [], []
    triangles_written = 0
    geoms_drawn = geoms_skipped = 0

    for body_id in range(model.nbody):
        body_name = _name_of(model, mujoco.mjtObj.mjOBJ_BODY, body_id, f"body_{body_id}")
        children = []
        for geom_id in range(model.ngeom):
            if int(model.geom_bodyid[geom_id]) != body_id:
                continue
            if int(model.geom_group[geom_id]) not in VISIBLE_GEOM_GROUPS:
                geoms_skipped += 1
                continue
            if float(model.geom_rgba[geom_id][3]) <= 0.0:
                geoms_skipped += 1
                continue
            geometry = geom_geometry(model, geom_id, maximum_terrain_triangles)
            if geometry is None:
                geoms_skipped += 1
                continue
            positions, normals, indices, report = geometry
            rgba, metallic, roughness = geom_appearance(model, geom_id)
            geom_name = _name_of(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id,
                                 f"geom_{geom_id}")
            key = (tuple(np.round(rgba, 4)), round(metallic, 3), round(roughness, 3))
            if key not in material_cache:
                material_id = int(model.geom_matid[geom_id])
                label = (_name_of(model, mujoco.mjtObj.mjOBJ_MATERIAL, material_id,
                                  f"material_{material_id}") if material_id >= 0
                         else f"geom_colour_{len(material_cache)}")
                material_cache[key] = builder.add_material(
                    label, rgba, metallic, roughness)
            # "__" and not "::": GLTFLoader runs every node name through
            # PropertyBinding.sanitizeNodeName, which strips anything outside
            # [\w-] -- so a colon silently vanishes and the sidecar's
            # `terrain_nodes` stops matching anything in the loaded scene. That
            # cost one debugging round; underscores survive.
            node_name = f"{body_name}__{geom_name}"
            mesh = builder.add_mesh(node_name, positions, normals, indices,
                                    material_cache[key])
            children.append(builder.add_node(
                node_name, translation=model.geom_pos[geom_id],
                rotation_wxyz=model.geom_quat[geom_id], mesh=mesh))
            triangles_written += int(indices.shape[0])
            geoms_drawn += 1
            if report is not None:
                terrain_report = dict(report, geom=geom_name, node=node_name)
                terrain_node_names.append(node_name)
                # THE SNOW SHELL, and it is DECORATION ONLY. A ground geom with
                # no relief of its own (a plane, or flat_free's zero-elevation
                # heightfield) draws as a featureless slab, so the measured
                # Lhotse shape goes on top of it as a second mesh in this .glb.
                # The file is display-only -- the page reads it, MuJoCo never
                # does -- so physics is bit-identical either way, and the flat
                # floor stays drawn underneath to hide the shell's dips.
                relief_deviation = terrain_relief_standard_deviation(model, geom_id)
                is_smooth = (relief_deviation is not None and relief_deviation
                             < SMOOTH_TERRAIN_STANDARD_DEVIATION_METERS)
                if snow_shell and is_smooth and snow_report is None:
                    patch = _snow_patch(SNOW_SHELL_PATCH_NAME)
                    slab_extent = [(patch.rough.shape[1] - 1) * float(patch.res),
                                   (patch.rough.shape[0] - 1) * float(patch.res)]
                    center_local = snow_shell_center_local(
                        model, data, geom_id, report["half_extent_meters"],
                        slab_extent)
                    (shell_positions, shell_normals, shell_indices,
                     snow_report) = snow_shell_mesh(
                        center_local, lift_meters=snow_shell_lift,
                        amplitude_scale=snow_shell_scale)
                    shell_node_name = f"{body_name}__{SNOW_SHELL_NODE_NAME}"
                    shell_mesh = builder.add_mesh(
                        shell_node_name, shell_positions, shell_normals,
                        shell_indices, material_cache[key])
                    children.append(builder.add_node(
                        shell_node_name, translation=model.geom_pos[geom_id],
                        rotation_wxyz=model.geom_quat[geom_id], mesh=shell_mesh))
                    triangles_written += int(shell_indices.shape[0])
                    # In `terrain_nodes` ON PURPOSE. app/web/three/world.js reads
                    # that list to hand a mesh the snow/rock shader, the
                    # footprint decals and castShadow=false -- which is exactly
                    # the treatment the shell needs to match the real terrain.
                    # The list drives no bounds: the footprint canvas and the
                    # camera's terrain bounds come from `sidecar.terrain`, which
                    # stays the REAL ground geom's report. The chase camera's
                    # height field does read every terrain mesh, so it now
                    # clears the drifts by a few centimetres.
                    terrain_node_names.append(shell_node_name)
                    snow_report = dict(snow_report, node=shell_node_name,
                                       over_geom=geom_name,
                                       ground_relief_std_meters=relief_deviation)
            if geom_name.startswith("ropeseg"):
                rope_node_names.append(node_name)

        node = builder.add_node(body_name, translation=data.xpos[body_id],
                                rotation_wxyz=data.xquat[body_id],
                                children=children)
        body_nodes.append({"index": body_id, "name": body_name, "node": body_name,
                           "geoms": len(children)})
        root_nodes.append(node)

    glb_path = os.path.join(output_directory, f"{world_name}.glb")
    total_bytes = builder.write(glb_path, root_nodes)

    pelvis_body = int(meta["pelvis_body_id"])
    foot_bodies = sorted({int(model.geom_bodyid[geom_id])
                          for geom_id in meta["foot_geom_ids"]})
    terrain_bounds = None
    if terrain_report is not None:
        # The hfield geom's own placement turns the local grid into world x/y.
        terrain_geom = int(mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, terrain_report["geom"]))
        center = np.asarray(data.geom_xpos[terrain_geom], np.float64)
        half_x, half_y = terrain_report["half_extent_meters"]
        terrain_bounds = {
            "center_world": [round(float(v), 4) for v in center],
            "half_extent_meters": [half_x, half_y],
            "elevation_meters": terrain_report["elevation_meters"],
        }

    sidecar = {
        "world": world_name,
        "label": definition["label"],
        "kind": definition["kind"],
        "slope_degrees": float(meta.get("slope_degrees", definition["slope_degrees"])),
        "rope_enabled": bool(definition["rope"]),
        "glb": f"/app/harness/scene_assets/{world_name}.glb",
        "nbody": int(model.nbody),
        "bodies": body_nodes,
        "pelvis_body": pelvis_body,
        "torso_body": int(meta["torso_body_id"]),
        "carrier_body": int(meta.get("carrier_body_id", -1)),
        "foot_bodies": [{"index": body_id,
                         "name": _name_of(model, mujoco.mjtObj.mjOBJ_BODY, body_id,
                                          f"body_{body_id}")}
                        for body_id in foot_bodies],
        "spawn": {
            "pelvis_position_world": [round(float(v), 5) for v in data.qpos[0:3]],
            "pelvis_quaternion_wxyz": [round(float(v), 6) for v in data.qpos[3:7]],
        },
        "terrain": (dict(terrain_report, **(terrain_bounds or {}))
                    if terrain_report else None),
        "terrain_nodes": terrain_node_names,
        # Display-only: the measured Lhotse relief laid over a smooth physics
        # ground. Null on worlds whose ground already has shape of its own.
        "snow_shell": snow_report,
        "rope": {
            "radius_meters": float(meta.get("rope_radius_meters", 0.025)),
            "polyline_world": rope_polyline(model, data) if definition["rope"] else [],
            "nodes": rope_node_names,
        },
        "sun": (look or {}).get("sun"),
        "fog": {"start_meters": (look or {}).get("fog_start_meters"),
                "end_meters": (look or {}).get("fog_end_meters")},
        "statistics": {
            "triangles": triangles_written,
            "geoms_drawn": geoms_drawn,
            "geoms_skipped": geoms_skipped,
            "glb_bytes": total_bytes,
            "materials": len(builder.materials),
            "meshes": len(builder.meshes),
        },
    }
    json_path = os.path.join(output_directory, f"{world_name}.json")
    with open(json_path, "w") as handle:
        json.dump(sidecar, handle, indent=1)

    # THE NUMBERS WE HAVE TO READ (project rule: print what you must ingest).
    print(f"[export] {world_name:<22} {total_bytes / 1e6:7.2f} MB"
          f"  {triangles_written:>8,} tris"
          f"  {geoms_drawn} geoms drawn / {geoms_skipped} skipped"
          f"  {model.nbody} bodies"
          f"  {len(builder.materials)} materials", flush=True)
    if terrain_report:
        print(f"[export]   terrain {terrain_report['rows']}x{terrain_report['columns']}"
              f" -> {terrain_report['sampled_rows']}x{terrain_report['sampled_columns']}"
              f" (stride {terrain_report['stride']},"
              f" {terrain_report['resolution_meters'][0] * 100:.1f} cm),"
              f" {terrain_report['triangles']:,} tris,"
              f" half-extent {terrain_report['half_extent_meters']} m,"
              f" relief {terrain_report['elevation_meters']:.2f} m", flush=True)
    if snow_report:
        print(f"[export]   snow shell (VISUAL ONLY, physics untouched):"
              f" patch {snow_report['patch']} x{snow_report['amplitude_scale']:.2f},"
              f" relief RMS {snow_report['relief_rms_meters']:.4f} m"
              f" (range {snow_report['relief_minimum_meters']:+.3f} to"
              f" {snow_report['relief_maximum_meters']:+.3f} m),"
              f" lift {snow_report['lift_meters']:.3f} m,"
              f" slab {snow_report['extent_meters'][0]:.2f} x"
              f" {snow_report['extent_meters'][1]:.2f} m at"
              f" {snow_report['resolution_meters'] * 100:.0f} cm centred"
              f" {snow_report['center_local_meters']} local,"
              f" {snow_report['triangles']:,} tris"
              f" (ground relief std"
              f" {snow_report['ground_relief_std_meters']:.4f} m)", flush=True)
    rope_points = len(sidecar["rope"]["polyline_world"])
    print(f"[export]   rope {rope_points} polyline points,"
          f" spawn pelvis {sidecar['spawn']['pelvis_position_world']},"
          f" slope {sidecar['slope_degrees']:.1f} deg", flush=True)
    if not rope_points and definition["rope"]:
        print("[export]   WARNING: rope world with no `ropeseg*` geoms found",
              flush=True)
    return glb_path, json_path, sidecar


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--world", default=worlds_module.DEFAULT_WORLD_NAME)
    parser.add_argument("--all", action="store_true",
                        help="every world in the map selector")
    parser.add_argument("--worlds", nargs="+", default=None,
                        help="an explicit list of world names")
    parser.add_argument("--output-directory", default=SCENE_ASSETS_DIRECTORY)
    parser.add_argument("--max-terrain-triangles", type=int,
                        default=DEFAULT_MAXIMUM_TERRAIN_TRIANGLES)
    parser.add_argument("--no-guide", action="store_true",
                        help="export without app/harness/guide.py's mocap human"
                             " (the runtime always attaches it, so this makes"
                             " the GLB one body short of the pose message)")
    parser.add_argument("--plain-graphics", action="store_true",
                        help="skip apply_alpine_look (raw MuJoCo colours)")
    parser.add_argument("--no-snow-shell", action="store_true",
                        help="do not lay the measured Lhotse relief over a"
                             " smooth physics ground (display only either way)")
    parser.add_argument("--snow-shell-scale", type=float,
                        default=SNOW_SHELL_AMPLITUDE_SCALE,
                        help="amplitude of that relief, 1.0 = as measured")
    parser.add_argument("--snow-shell-lift", type=float,
                        default=SNOW_SHELL_LIFT_METERS,
                        help="metres the shell's mean plane sits above the"
                             " physics ground")
    arguments = parser.parse_args(argv)

    if arguments.all:
        names = worlds_module.world_names()
    elif arguments.worlds:
        names = arguments.worlds
    else:
        names = [arguments.world]

    index = []
    for name in names:
        try:
            _glb, _json, sidecar = export_world(
                name, arguments.output_directory, arguments.max_terrain_triangles,
                arguments.plain_graphics, not arguments.no_guide,
                snow_shell=not arguments.no_snow_shell,
                snow_shell_scale=arguments.snow_shell_scale,
                snow_shell_lift=arguments.snow_shell_lift)
        except Exception as error:            # one broken world must not stop a sweep
            print(f"[export] {name}: FAILED {type(error).__name__}: {error}", flush=True)
            continue
        index.append({"world": name, "label": sidecar["label"],
                      "glb": sidecar["glb"],
                      "json": f"/app/harness/scene_assets/{name}.json",
                      "triangles": sidecar["statistics"]["triangles"],
                      "bytes": sidecar["statistics"]["glb_bytes"]})
    # THE INDEX IS THE MAP DROPDOWN. `render3d.html:listWorlds` shows a world
    # only if `index.json` names it, so an index rewritten from scratch by a
    # one-world export silently empties the selector of every other world --
    # which is exactly what `--world chloe_20` used to do. Merge instead:
    # worlds exported now replace their row, worlds exported earlier keep
    # theirs as long as their .glb is still on disk, and a world whose file
    # has been deleted drops out. `--all` is then a convenience, not a
    # requirement.
    index_path = os.path.join(arguments.output_directory, "index.json")
    os.makedirs(arguments.output_directory, exist_ok=True)
    merged = {entry["world"]: entry for entry in index}
    kept = 0
    if os.path.exists(index_path):
        try:
            with open(index_path) as handle:
                for entry in json.load(handle).get("scenes", []):
                    name = entry.get("world")
                    if name in merged or not name:
                        continue
                    if not os.path.exists(os.path.join(arguments.output_directory,
                                                       f"{name}.glb")):
                        print(f"[export] index: dropping {name}, its .glb is gone",
                              flush=True)
                        continue
                    merged[name] = entry
                    kept += 1
        except (ValueError, OSError) as error:
            print(f"[export] index unreadable ({error}); writing a fresh one",
                  flush=True)
    with open(index_path, "w") as handle:
        json.dump({"scenes": [merged[name] for name in sorted(merged)]},
                  handle, indent=1)
    print(f"[export] index.json: {len(index)} written now, {kept} kept from"
          f" earlier runs, {len(merged)} worlds in the map dropdown", flush=True)
    total = sum(entry["bytes"] for entry in index)
    print(f"[export] {len(index)}/{len(names)} worlds ->"
          f" {arguments.output_directory}  ({total / 1e6:.1f} MB total)", flush=True)
    return 0 if len(index) == len(names) else 1


if __name__ == "__main__":
    raise SystemExit(main())

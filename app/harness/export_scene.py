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

    report = {
        "rows": rows, "columns": columns, "stride": stride,
        "sampled_rows": int(height), "sampled_columns": int(width),
        "triangles": int(indices.shape[0]),
        "resolution_meters": [float(2 * radius_x / (columns - 1) * stride),
                              float(2 * radius_y / (rows - 1) * stride)],
        "half_extent_meters": [radius_x, radius_y],
        "elevation_meters": elevation_z,
    }
    return (positions.reshape(-1, 3), normals.reshape(-1, 3).astype(np.float32),
            indices, report)


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
        return plane_mesh(size[0], size[1]) + (None,)
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
    if definition["kind"] == "climb_scene":
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
                 plain_graphics=False, with_guide=True):
    model, data, meta, definition, look = open_world(
        world_name, plain_graphics, with_guide)
    os.makedirs(output_directory, exist_ok=True)

    builder = GlbBuilder()
    material_cache, terrain_report = {}, None
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
                arguments.plain_graphics, not arguments.no_guide)
        except Exception as error:            # one broken world must not stop a sweep
            print(f"[export] {name}: FAILED {type(error).__name__}: {error}", flush=True)
            continue
        index.append({"world": name, "label": sidecar["label"],
                      "glb": sidecar["glb"],
                      "json": f"/app/harness/scene_assets/{name}.json",
                      "triangles": sidecar["statistics"]["triangles"],
                      "bytes": sidecar["statistics"]["glb_bytes"]})
    index_path = os.path.join(arguments.output_directory, "index.json")
    os.makedirs(arguments.output_directory, exist_ok=True)
    with open(index_path, "w") as handle:
        json.dump({"scenes": index}, handle, indent=1)
    total = sum(entry["bytes"] for entry in index)
    print(f"[export] {len(index)}/{len(names)} worlds ->"
          f" {arguments.output_directory}  ({total / 1e6:.1f} MB total)", flush=True)
    return 0 if len(index) == len(names) else 1


if __name__ == "__main__":
    raise SystemExit(main())

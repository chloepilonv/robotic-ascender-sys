"""Blender headless: import the Sketchfab FBX, report stats, export PLY+GLB.

Run:  /Applications/Blender.app/Contents/MacOS/Blender -b -P scripts/fbx_convert.py -- <fbx> <outdir>
"""
import bpy, sys, os, json
from mathutils import Vector

argv = sys.argv[sys.argv.index("--")+1:]
fbx, outdir = argv[0], argv[1]
os.makedirs(outdir, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
try:
    bpy.ops.import_scene.fbx(filepath=fbx)
except Exception as e:
    print("FBX_IMPORT_ERROR:", e); raise

info = {"objects": [], "scene_unit_scale": bpy.context.scene.unit_settings.scale_length}
tot_v = tot_t = 0
lo = Vector((1e30,)*3); hi = Vector((-1e30,)*3)

for ob in bpy.context.scene.objects:
    if ob.type != 'MESH':
        info["objects"].append({"name": ob.name, "type": ob.type})
        continue
    me = ob.data
    me.calc_loop_triangles()
    nv, nt = len(me.vertices), len(me.loop_triangles)
    tot_v += nv; tot_t += nt
    for c in ob.bound_box:
        w = ob.matrix_world @ Vector(c)
        lo = Vector((min(lo[i], w[i]) for i in range(3)))
        hi = Vector((max(hi[i], w[i]) for i in range(3)))
    info["objects"].append({
        "name": ob.name, "type": "MESH", "vertices": nv, "triangles": nt,
        "uv_layers": [l.name for l in me.uv_layers],
        "materials": [m.name if m else None for m in me.materials],
        "matrix_world": [list(r) for r in ob.matrix_world],
        "scale": list(ob.scale), "rotation_euler": list(ob.rotation_euler),
    })

info["total_vertices"] = tot_v
info["total_triangles"] = tot_t
info["bbox_min"] = list(lo); info["bbox_max"] = list(hi)
info["bbox_size"] = [hi[i]-lo[i] for i in range(3)]

# textures / images
info["images"] = [{"name": im.name, "size": list(im.size), "filepath": im.filepath}
                  for im in bpy.data.images]

# any custom properties that might carry georeferencing
props = {}
for ob in bpy.context.scene.objects:
    d = {k: str(ob[k]) for k in ob.keys() if k not in ("_RNA_UI",)}
    if d: props[ob.name] = d
sc = {k: str(bpy.context.scene[k]) for k in bpy.context.scene.keys()
      if k not in ("_RNA_UI",)}
info["custom_props_objects"] = props
info["custom_props_scene"] = sc

json.dump(info, open(os.path.join(outdir, "fbx_info.json"), "w"), indent=2)
print("FBX_INFO_JSON_WRITTEN")
print(json.dumps({k: info[k] for k in
                  ("total_vertices","total_triangles","bbox_min","bbox_max",
                   "bbox_size","scene_unit_scale","images","custom_props_objects",
                   "custom_props_scene")}, indent=2))

# export for offline processing (PLY keeps it simple + fast to load in numpy)
bpy.ops.object.select_all(action='SELECT')
ply = os.path.join(outdir, "everest.ply")
try:
    bpy.ops.wm.ply_export(filepath=ply, export_selected_objects=False,
                          apply_modifiers=True)
except Exception:
    bpy.ops.export_mesh.ply(filepath=ply, use_selection=False)
print("PLY_WRITTEN", ply, os.path.getsize(ply))

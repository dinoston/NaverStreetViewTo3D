"""Render three neutral Blender previews for a mesh passed after `--`."""

import sys
from pathlib import Path

import bpy
from mathutils import Vector


mesh_path = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
output_dir = mesh_path.parent
output_prefix = mesh_path.stem

bpy.ops.wm.read_factory_settings(use_empty=True)
if mesh_path.suffix.lower() == ".ply":
    bpy.ops.wm.ply_import(filepath=str(mesh_path))
else:
    bpy.ops.wm.obj_import(filepath=str(mesh_path))
obj = bpy.context.selected_objects[0]

if mesh_path.suffix.lower() == ".ply" or not obj.data.materials:
    material = bpy.data.materials.new("Neutral building")
    material.diffuse_color = (0.46, 0.50, 0.56, 1.0)
    material.roughness = 0.82
    obj.data.materials.clear()
    obj.data.materials.append(material)

corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
minimum = Vector((min(p.x for p in corners), min(p.y for p in corners), min(p.z for p in corners)))
maximum = Vector((max(p.x for p in corners), max(p.y for p in corners), max(p.z for p in corners)))
center = (minimum + maximum) / 2
extent = maximum - minimum
size = max(extent)

bpy.ops.object.light_add(type="AREA", location=center + Vector((size, -size, size * 1.8)))
bpy.context.object.data.energy = 1800
bpy.context.object.data.shape = "DISK"
bpy.context.object.data.size = size
bpy.ops.object.light_add(type="SUN", location=center + Vector((-size, size, size)))
bpy.context.object.rotation_euler = (0.45, -0.55, -0.35)
bpy.context.object.data.energy = 2.0

bpy.ops.object.camera_add()
camera = bpy.context.object
camera.data.type = "ORTHO"
camera.data.ortho_scale = size * 1.18
bpy.context.scene.camera = camera

scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE_NEXT"
scene.render.resolution_x = 1280
scene.render.resolution_y = 900
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.film_transparent = False
if scene.world is None:
    scene.world = bpy.data.worlds.new("Preview World")
scene.world.color = (0.025, 0.025, 0.025)

views = {
    "front": Vector((1.4, -2.0, 0.65)),
    "side": Vector((-2.0, -0.25, 0.55)),
    "top": Vector((0.45, -0.65, 2.5)),
}
for name, direction in views.items():
    direction.normalize()
    camera.location = center + direction * size * 2.2
    camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath = str(output_dir / f"{output_prefix}_{name}.png")
    bpy.ops.render.render(write_still=True)

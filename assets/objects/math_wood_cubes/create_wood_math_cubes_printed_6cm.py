#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Create realistic wooden math/number cubes for RoboLab using Blender.

This script generates 16 cube assets (0-9, plus, minus, multiply, divide,
percent, equals), each approximately 7 cm per side, with bevelled/rounded
edges, procedural wood grain, and a colored printed label on one face.

Outputs:
  assets/objects/math_wood_cubes/*.usda
  assets/objects/math_wood_cubes/*.blend
  assets/objects/math_wood_cubes/previews/*.png
  assets/objects/math_wood_cubes/textures/*.png
  assets/objects/math_wood_cubes/README.md

Run from the repository root with:
  blender --background --python assets/objects/math_wood_cubes/create_wood_math_cubes_printed_6cm.py
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import bpy
    from mathutils import Vector
except ImportError as exc:  # pragma: no cover - script is intended for Blender
    raise SystemExit("This script must be run with Blender's Python: blender --background --python ...") from exc


REPO_ROOT = Path(__file__).resolve().parents[3]
ASSET_DIR = REPO_ROOT / "assets" / "objects" / "math_wood_cubes"
TEXTURE_DIR = ASSET_DIR / "textures"
PREVIEW_DIR = ASSET_DIR / "previews"
CATALOG_PATH = REPO_ROOT / "assets" / "objects" / "object_catalog.json"

CUBE_SIZE_M = 0.06
HALF = CUBE_SIZE_M / 2.0
BEVEL_M = 0.0045
LABEL_Z = HALF + 0.00045
LABEL_SIZE_M = CUBE_SIZE_M * 0.84
SPECK_RANGE_M = CUBE_SIZE_M * 0.37
MASS_KG = 0.06
WOOD_DENSITY_KG_M3 = 650.0
# Final assets are oriented with the printed face upward (+Z), which works well
# for tabletop benchmarks and object recognition. Rotate 90 deg about X in scene
# placement if a vertical/front-facing label is preferred.

# Very dark, saturated ink colors chosen for high contrast against warm wood.
SYMBOLS = [
    ("0", "0", "deep blue", "#00245C"),
    ("1", "1", "burgundy", "#4C0010"),
    ("2", "2", "forest green", "#003D1D"),
    ("3", "3", "eggplant purple", "#250341"),
    ("4", "4", "burnt umber", "#5A1C00"),
    ("5", "5", "deep teal", "#003B44"),
    ("6", "6", "dark plum", "#47102C"),
    ("7", "7", "dark ochre", "#473300"),
    ("8", "8", "near-black navy", "#080C14"),
    ("9", "9", "dark olive", "#24370C"),
    ("plus", "+", "dark cyan", "#07343C"),
    ("minus", "−", "dark magenta", "#450C3E"),
    ("multiply", "×", "dark crimson", "#5C0000"),
    ("divide", "÷", "midnight indigo", "#11102E"),
    ("percent", "%", "dark emerald", "#033526"),
    ("equals", "=", "dark slate", "#0F172A"),
]

# Use a broadly available bold font. Fall back to Blender's text font if absent.
FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]


def run_system_python(code: str, *args: str) -> None:
    """Run a short helper snippet with system Python (has Pillow in this env)."""
    python = shutil.which("python3") or "/usr/bin/python3"
    subprocess.run([python, "-c", code, *args], check=True)


def hex_to_rgb01(value: str) -> tuple[float, float, float]:
    value = value.strip().lstrip("#")
    return tuple(int(value[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def ensure_dirs() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    TEXTURE_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)


def create_label_texture(symbol_id: str, glyph: str, color_hex: str) -> Path:
    """Create transparent PNG label texture using Pillow via system Python."""
    out = TEXTURE_DIR / f"label_{symbol_id}.png"
    font_arg = next((p for p in FONT_PATHS if Path(p).exists()), "")

    helper = r'''
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import sys
out = Path(sys.argv[1])
glyph = sys.argv[2]
color_hex = sys.argv[3]
font_path = sys.argv[4]
W = H = 1024
img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
d = ImageDraw.Draw(img)
# For a toy-block printed look: bold, centered, with a soft ink edge.
font_size = 820 if glyph != "%" else 690
if glyph in {"−", "-"}:
    font_size = 850
if glyph in {"+", "×", "÷", "="}:
    font_size = 780
try:
    font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
except Exception:
    font = ImageFont.load_default()
bbox = d.textbbox((0, 0), glyph, font=font, stroke_width=10)
tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
x = (W - tw) / 2 - bbox[0]
y = (H - th) / 2 - bbox[1] - H * 0.035
# Subtle off-white underprint improves visibility on wood without looking like a decal sheet.
under = Image.new("RGBA", (W, H), (0, 0, 0, 0))
ud = ImageDraw.Draw(under)
ud.text((x, y), glyph, font=font, fill=(252, 247, 230, 190), stroke_width=20, stroke_fill=(252, 247, 230, 95))
under = under.filter(ImageFilter.GaussianBlur(1.0))
img.alpha_composite(under)
# Color ink; slight darker stroke acts like real printed edge bleed.
r = int(color_hex[1:3], 16); g = int(color_hex[3:5], 16); b = int(color_hex[5:7], 16)
shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
sd = ImageDraw.Draw(shadow)
sd.text((x + 4, y - 4), glyph, font=font, fill=(40, 25, 10, 70), stroke_width=10, stroke_fill=(40, 25, 10, 35))
shadow = shadow.filter(ImageFilter.GaussianBlur(2.0))
img.alpha_composite(shadow)
d = ImageDraw.Draw(img)
d.text((x, y), glyph, font=font, fill=(r, g, b, 245), stroke_width=8, stroke_fill=(max(r-50,0), max(g-50,0), max(b-50,0), 200))
# Restrict to a large centered circle-ish/rounded-square area: transparent outside remains.
img.save(out)
'''
    run_system_python(helper, str(out), glyph, color_hex, font_arg)
    return out


def create_wood_textures(seed: int = 7) -> tuple[Path, Path, Path]:
    """Return the real Wood095 PBR texture maps bundled with the generated assets."""
    base = TEXTURE_DIR / "Wood095_2K-PNG_Color.png"
    rough = TEXTURE_DIR / "Wood095_2K-PNG_Roughness.png"
    normal = TEXTURE_DIR / "Wood095_2K-PNG_NormalGL.png"
    missing = [p for p in (base, rough, normal) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing Wood095 texture map(s): "
            + ", ".join(str(p) for p in missing)
            + ". Extract Wood095_2K-PNG_Color/Roughness/NormalGL.png into assets/objects/math_wood_cubes/textures/."
        )
    return base, rough, normal


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for block in (bpy.data.meshes, bpy.data.materials, bpy.data.images, bpy.data.textures, bpy.data.curves):
        for item in list(block):
            if item.users == 0:
                block.remove(item)


def make_wood_material(base_tex: Path, rough_tex: Path, normal_tex: Path) -> bpy.types.Material:
    mat = bpy.data.materials.new("realistic_warm_wood")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf is None:
        return mat
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Roughness"].default_value = 0.68
    bsdf.inputs["Base Color"].default_value = (0.66, 0.48, 0.27, 1.0)

    def tex_node(path: Path, label: str, colorspace: str = "sRGB"):
        node = nodes.new(type="ShaderNodeTexImage")
        node.label = label
        node.image = bpy.data.images.load(str(path))
        node.image.colorspace_settings.name = colorspace
        return node

    base = tex_node(base_tex, "Wood095 base color", "sRGB")
    rough = tex_node(rough_tex, "Wood095 roughness", "Non-Color")
    normal = tex_node(normal_tex, "Wood095 OpenGL normal", "Non-Color")
    normal_map = nodes.new(type="ShaderNodeNormalMap")
    normal_map.inputs["Strength"].default_value = 0.18
    coord = nodes.new(type="ShaderNodeTexCoord")
    mapping = nodes.new(type="ShaderNodeMapping")
    # Wood095 is a 2:1 map with grain running horizontally. A mild vertical
    # stretch makes the grain fine enough for a 7 cm toy cube without obvious
    # repetition in Isaac Sim.
    mapping.inputs["Scale"].default_value = (1.35, 2.35, 1.0)
    links = mat.node_tree.links
    links.new(coord.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], base.inputs["Vector"])
    links.new(mapping.outputs["Vector"], rough.inputs["Vector"])
    links.new(mapping.outputs["Vector"], normal.inputs["Vector"])
    links.new(base.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(rough.outputs["Color"], bsdf.inputs["Roughness"])
    links.new(normal.outputs["Color"], normal_map.inputs["Color"])
    links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def make_label_material(symbol_id: str, label_tex: Path, color_hex: str) -> bpy.types.Material:
    mat = bpy.data.materials.new(f"printed_{symbol_id}")
    mat.use_nodes = True
    mat.blend_method = "BLEND"
    mat.use_screen_refraction = False
    mat.show_transparent_back = False
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf is None:
        return mat
    rgb = hex_to_rgb01(color_hex)
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Alpha"].default_value = 1.0
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Roughness"].default_value = 0.72
    tex = nodes.new(type="ShaderNodeTexImage")
    tex.label = "Transparent printed symbol"
    tex.image = bpy.data.images.load(str(label_tex))
    mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    mat.node_tree.links.new(tex.outputs["Alpha"], bsdf.inputs["Alpha"])
    return mat


def make_dark_edge_material() -> bpy.types.Material:
    mat = bpy.data.materials.new("slightly_darker_end_grain_and_chamfer")
    mat.diffuse_color = (0.42, 0.22, 0.08, 1.0)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.44, 0.24, 0.09, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.76
    return mat


def create_rounded_cube(name: str, wood_mat: bpy.types.Material, edge_mat: bpy.types.Material) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(size=CUBE_SIZE_M, location=(0, 0, 0))
    cube = bpy.context.object
    cube.name = name
    cube.data.name = f"{name}_beveled_body_mesh"
    cube.data.materials.append(wood_mat)
    cube.data.materials.append(edge_mat)

    bevel = cube.modifiers.new("8 mm rounded/chamfered toy-block edges", "BEVEL")
    bevel.width = BEVEL_M
    bevel.segments = 8
    bevel.affect = "EDGES"
    bevel.harden_normals = True
    bevel.profile = 0.5

    weighted = cube.modifiers.new("weighted normals for realistic highlights", "WEIGHTED_NORMAL")
    weighted.keep_sharp = True

    # Apply modifiers before USD export for maximum simulator compatibility.
    cube.select_set(True)
    bpy.context.view_layer.objects.active = cube
    for mod in list(cube.modifiers):
        bpy.ops.object.modifier_apply(modifier=mod.name)

    # UV unwrap the bevelled mesh for procedural wood texture export.
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=0.02)
    bpy.ops.object.mode_set(mode="OBJECT")

    return cube


def add_label_plane(name: str, glyph: str, mat: bpy.types.Material) -> bpy.types.Object:
    # Plane is smaller than the top face to leave a visible wooden border, but
    # large enough for the symbols to be readable in Designer thumbnails.
    bpy.ops.mesh.primitive_plane_add(size=LABEL_SIZE_M, location=(0, 0, LABEL_Z), rotation=(0, 0, 0))
    label = bpy.context.object
    label.name = f"{name}_printed_label"
    label.data.name = f"{name}_printed_label_mesh"
    label.data.materials.append(mat)
    # Add a tiny solidify thickness; it behaves as printed ink/paint, not just a mathematical plane.
    solid = label.modifiers.new("thin printed ink layer", "SOLIDIFY")
    solid.thickness = 0.00025
    solid.offset = 0.0
    bpy.context.view_layer.objects.active = label
    label.select_set(True)
    bpy.ops.object.modifier_apply(modifier=solid.name)
    return label


def add_small_imperfections(parent_name: str, wood_mat: bpy.types.Material, count: int, seed: int) -> list[bpy.types.Object]:
    """Add tiny darker/raised/dented specks and grain lines on non-label faces."""
    rng = random.Random(seed)
    objs: list[bpy.types.Object] = []
    speck_mat = bpy.data.materials.new(f"{parent_name}_natural_tiny_dark_specks")
    speck_mat.diffuse_color = (0.20, 0.10, 0.035, 1.0)
    speck_mat.use_nodes = True
    bsdf = speck_mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.22, 0.11, 0.04, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.85

    faces = ["front", "back", "left", "right", "bottom"]
    for i in range(count):
        face = rng.choice(faces)
            # Tiny, nearly flush flat printed/painted freckles. Keep them very subtle;
        # earlier raised cylinders read as dark pimples in Blender and Isaac Sim.
        radius = rng.uniform(0.00045, 0.0011)
        if face in {"front", "back"}:
            y = HALF + 0.00008 if face == "front" else -HALF - 0.00008
            x = rng.uniform(-SPECK_RANGE_M, SPECK_RANGE_M)
            z = rng.uniform(-SPECK_RANGE_M, SPECK_RANGE_M)
            rot = (math.radians(90), 0, 0)
            loc = (x, y, z)
        elif face in {"left", "right"}:
            x = -HALF - 0.00008 if face == "left" else HALF + 0.00008
            y = rng.uniform(-SPECK_RANGE_M, SPECK_RANGE_M)
            z = rng.uniform(-SPECK_RANGE_M, SPECK_RANGE_M)
            rot = (0, math.radians(90), 0)
            loc = (x, y, z)
        else:
            z = -HALF - 0.00008
            x = rng.uniform(-SPECK_RANGE_M, SPECK_RANGE_M)
            y = rng.uniform(-SPECK_RANGE_M, SPECK_RANGE_M)
            rot = (0, 0, 0)
            loc = (x, y, z)
        bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=radius, depth=0.000035, location=loc, rotation=rot)
        speck = bpy.context.object
        speck.name = f"{parent_name}_wood_speck_{i:02d}"
        speck.scale.x *= rng.uniform(1.0, 1.9)
        speck.scale.y *= rng.uniform(0.55, 1.0)
        speck.data.materials.append(speck_mat)
        objs.append(speck)
    return objs


def join_objects(name: str, objects: list[bpy.types.Object]) -> bpy.types.Object:
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.object.join()
    joined = bpy.context.object
    joined.name = name
    joined.data.name = f"{name}_mesh"
    return joined


def add_collision_proxy(root: bpy.types.Object, name: str) -> bpy.types.Object:
    """A hidden simple collider cube for physics engines; visual mesh remains bevelled."""
    bpy.ops.mesh.primitive_cube_add(size=CUBE_SIZE_M * 0.985, location=(0, 0, 0))
    col = bpy.context.object
    col.name = f"{name}_collision_box"
    col.display_type = "WIRE"
    col.hide_render = True
    col.hide_viewport = True
    mat = bpy.data.materials.new(f"{name}_collision_invisible")
    mat.diffuse_color = (0, 0, 0, 0)
    col.data.materials.append(mat)
    return col


def setup_lighting_camera() -> None:
    # Soft, low-power studio lighting. The original previews used a very bright
    # key plus a bright world color, which clipped highlights on the label/top
    # face in the Designer thumbnails. Keep the light large for soft shadows,
    # but lower intensity and let color management preserve highlights.
    bpy.ops.object.light_add(type="AREA", location=(-0.35, -0.45, 0.70))
    key = bpy.context.object
    key.name = "large_softbox_reflection_light"
    key.data.energy = 155
    key.data.size = 0.85
    bpy.ops.object.light_add(type="POINT", location=(0.25, 0.30, 0.35))
    fill = bpy.context.object
    fill.name = "small_warm_fill_light"
    fill.data.energy = 12
    fill.data.color = (1.0, 0.82, 0.62)

    bpy.ops.object.camera_add(location=(0.19, -0.22, 0.18), rotation=(math.radians(61), 0, math.radians(41)))
    cam = bpy.context.object
    bpy.context.scene.camera = cam
    cam.data.lens = 65
    cam.data.dof.use_dof = True
    cam.data.dof.focus_distance = 0.32
    cam.data.dof.aperture_fstop = 7.5


def fix_usd_asset_paths(path: Path, name: str) -> None:
    """Make USD asset/material paths robust when this file is used as a payload.

    Isaac Sim composes payload contents under the payload prim, so absolute
    material paths such as </_materials/foo> point outside the payload scope and
    get ignored. Keep texture paths explicitly relative to the asset file and
    make material relationships relative to the cube root prim.
    """
    text = path.read_text()
    text = text.replace("asset inputs:file = @textures/", "asset inputs:file = @./textures/")
    text = text.replace("</_materials/", f"</{name}/_materials/")
    text = text.replace('\n}\n\ndef Scope "_materials"\n{', '\n\n    def Scope "_materials"\n    {')

    # Moving Blender's exported top-level _materials scope under the payload root
    # consumes the root's original closing brace. Ensure the root prim is closed
    # exactly once afterwards; otherwise Isaac Sim fails to load payloads with a
    # USDA syntax error and the cubes disappear from composed scenes.
    balance = text.count("{") - text.count("}")
    if balance > 0:
        text = text.rstrip() + ("\n" + "}\n" * balance)
    elif balance < 0:
        raise RuntimeError(f"USD brace imbalance after path fix in {path}: {balance}")
    path.write_text(text)


def add_physics_metadata_to_usda(path: Path, name: str, description: str) -> None:
    """Patch Blender-exported USD with RoboLab-friendly physics/custom attrs."""
    text = path.read_text()
    # Root Xform header: add physics APIs and kind if not already present.
    pattern = rf'def Xform "{re.escape(name)}"\n\{{'
    replacement = (
        f'def Xform "{name}" (\n'
        f'    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]\n'
        f'    kind = "component"\n'
        f')\n'
        f'{{\n'
        f'    custom string class = "wooden_math_cube"\n'
        f'    custom string description = "{description}"\n'
        f'    bool physics:kinematicEnabled = 0\n'
        f'    float physics:mass = {MASS_KG}\n'
        f'    bool physics:rigidBodyEnabled = 1\n'
    )
    text, n = re.subn(pattern, replacement, text, count=1)
    if n != 1:
        raise RuntimeError(f"Could not patch root prim in {path}")

    # Make the main mesh collidable. The mesh name is deterministic after join.
    # Blender may write different metadata in the parenthesized block, so replace
    # the whole block immediately before the mesh body.
    mesh_pattern = rf'def Mesh "{re.escape(name)}_mesh" \([\s\S]*?\n    \)\n    \{{'
    mesh_repl = (
        f'def Mesh "{name}_mesh" (\n'
        f'        active = true\n'
        f'        prepend apiSchemas = ["MaterialBindingAPI", "PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]\n'
        f'    )\n'
        f'    {{\n'
        f'        uniform token physics:approximation = "convexHull"'
    )
    text, n = re.subn(mesh_pattern, mesh_repl, text, count=1)
    if n != 1:
        raise RuntimeError(f"Could not patch mesh collision API in {path}")

    # Add a simple physics material inside the root and bind it to the mesh for catalog friction extraction.
    insert = f'''

    def Material "PhysicsMaterial" (
        prepend apiSchemas = ["PhysicsMaterialAPI"]
    )
    {{
        float physics:density = {WOOD_DENSITY_KG_M3:g}
        float physics:dynamicFriction = 0.8
        float physics:restitution = 0.05
        float physics:staticFriction = 1.0
    }}
'''
    first_child = f'\n    def Mesh "{name}_mesh"'
    if first_child in text:
        text = text.replace(first_child, insert + first_child, 1)
    # The material binding relationship needs to be inside the mesh body. Put it after extent.
    extent_match = re.search(r'(def Mesh "' + re.escape(name) + r'_mesh"[\s\S]*?\n\s*float3\[\] extent = \[[^\n]+\]\n)', text)
    if extent_match:
        binding = f'            rel material:binding:physics = </{name}/PhysicsMaterial>\n'
        start, end = extent_match.span(1)
        if 'material:binding:physics' not in text[start : start + 800]:
            text = text[:end] + binding + text[end:]

    path.write_text(text)


def export_asset(symbol_id: str, glyph: str, color_name: str, color_hex: str, wood_tex: tuple[Path, Path, Path]) -> dict:
    clear_scene()
    name = f"wood_math_cube_{symbol_id}"
    desc = f"A realistic 7 cm chamfered wooden cube with the {glyph} symbol printed in {color_name} ink on one face."

    label_tex = create_label_texture(symbol_id, glyph, color_hex)
    wood_mat = make_wood_material(*wood_tex)
    edge_mat = make_dark_edge_material()
    label_mat = make_label_material(symbol_id, label_tex, color_hex)
    body = create_rounded_cube(name, wood_mat, edge_mat)
    label = add_label_plane(name, glyph, label_mat)
    # Do not add separate raised speck geometry: small dark cylinders looked like
    # pimples/protrusions in both Blender previews and Isaac Sim. Natural variation
    # now comes from the flat wood texture instead.
    joined = join_objects(name, [body, label])

    # Set origin in the center; asset extents remain about 0.07 m each side.
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    joined.location = (0, 0, 0)

    # Custom properties are exported as userProperties and also patched onto root for RoboLab catalog.
    joined["class"] = "wooden_math_cube"
    joined["description"] = desc

    setup_lighting_camera()

    # Render a quick preview PNG.
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.samples = 96
    bpy.context.scene.render.resolution_x = 900
    bpy.context.scene.render.resolution_y = 900
    bpy.context.scene.view_settings.view_transform = "AgX"
    bpy.context.scene.view_settings.look = "AgX - Medium Low Contrast"
    bpy.context.scene.view_settings.exposure = -0.45
    bpy.context.scene.view_settings.gamma = 1.0
    bpy.context.scene.world.color = (0.42, 0.44, 0.48)
    preview_path = PREVIEW_DIR / f"{name}.png"
    bpy.context.scene.render.filepath = str(preview_path)
    bpy.ops.render.render(write_still=True)

    blend_path = ASSET_DIR / f"{name}.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))

    # Export as ASCII USDA first so we can add physics metadata in a transparent,
    # simulator-friendly way, then keep a .usd copy for consistency with the catalog.
    usd_path = ASSET_DIR / f"{name}.usda"
    bpy.ops.wm.usd_export(
        filepath=str(usd_path),
        root_prim_path="/",
        selected_objects_only=False,
        export_animation=False,
        export_lights=False,
        export_cameras=False,
        export_materials=True,
        generate_preview_surface=True,
        export_textures_mode="NEW",
        relative_paths=True,
        export_custom_properties=True,
    )
    add_physics_metadata_to_usda(usd_path, name, desc)
    fix_usd_asset_paths(usd_path, name)

    return {
        "name": name,
        "usd_path": f"assets/objects/math_wood_cubes/{name}.usda",
        "prim_path": f"/{name}",
        "position": [0.0, 0.0, 0.0],
        "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        "payload": [],
        "rigid_body": True,
        "static_body": False,
        "class": "wooden_math_cube",
        "description": desc,
        "dims": [CUBE_SIZE_M, CUBE_SIZE_M, CUBE_SIZE_M],
        "mass": MASS_KG,
        "density": WOOD_DENSITY_KG_M3,
        "dynamic_friction": 0.8,
        "static_friction": 1.0,
        "restitution": 0.05,
        "dataset": "math_wood_cubes",
    }


def update_catalog(entries: list[dict]) -> bool:
    """Update RoboLab's optional object catalog when one is available."""
    if not CATALOG_PATH.exists():
        return False
    catalog = json.loads(CATALOG_PATH.read_text())
    names = {entry["name"] for entry in entries}
    catalog = [obj for obj in catalog if obj.get("name") not in names and obj.get("dataset") != "math_wood_cubes"]
    catalog.extend(entries)
    CATALOG_PATH.write_text(json.dumps(catalog, indent=2) + "\n")
    return True


def write_readme(entries: list[dict]) -> None:
    lines = [
        "# Math Wood Cubes",
        "",
        "Blender-generated 7 cm wooden benchmark cubes with chamfered/rounded edges, procedural wood grain, and colored printed symbols on the top face.",
        "",
        f"- Size: approximately {CUBE_SIZE_M:.2f} × {CUBE_SIZE_M:.2f} × {CUBE_SIZE_M:.2f} m",
        f"- Mass metadata: {MASS_KG:.2f} kg each",
        "- Physics metadata: rigid body, collision mesh, static friction 1.0, dynamic friction 0.8, restitution 0.05",
        "- Class: `wooden_math_cube`",
        "- Source generator: `assets/objects/math_wood_cubes/create_wood_math_cubes_printed_6cm.py`",
        "",
        "| Asset | Preview | Symbol | USD |",
        "| :-- | :-- | :-- | :-- |",
    ]
    for entry, (_, glyph, _, _) in zip(entries, SYMBOLS):
        name = entry["name"]
        lines.append(f"| `{name}` | <img src=\"previews/{name}.png\" width=\"120\"> | `{glyph}` | `{name}.usda` |")
    (ASSET_DIR / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ensure_dirs()
    wood_tex = create_wood_textures()
    entries: list[dict] = []
    for symbol_id, glyph, color_name, color_hex in SYMBOLS:
        entries.append(export_asset(symbol_id, glyph, color_name, color_hex, wood_tex))
    catalog_updated = update_catalog(entries)
    write_readme(entries)
    print(f"Created {len(entries)} wooden math cubes in {ASSET_DIR}")
    if catalog_updated:
        print(f"Updated catalog: {CATALOG_PATH}")
    else:
        print(f"Skipped optional RoboLab catalog (not present): {CATALOG_PATH}")


if __name__ == "__main__":
    main()

bl_info = {
    "name": "Color ID UV Splitter",
    "author": "",
    "version": (20, 2, 0),
    "blender": (5, 0, 0),
    "location": "3D View > Sidebar > Color ID",
    "description": "Vectorized Base Color classification for large meshes. Select source colors and assign output ID materials.",
    "category": "Material",
}

import bpy
import numpy as np
from mathutils import Vector
from bpy_extras import view3d_utils


MAX_RULES = 10

OUTPUT_COLORS = [
    ("Red",     (1.0, 0.0, 0.0, 1.0)),
    ("Green",   (0.0, 1.0, 0.0, 1.0)),
    ("Blue",    (0.0, 0.0, 1.0, 1.0)),
    ("Yellow",  (1.0, 1.0, 0.0, 1.0)),
    ("Magenta", (1.0, 0.0, 1.0, 1.0)),
    ("Cyan",    (0.0, 1.0, 1.0, 1.0)),
    ("Orange",  (1.0, 0.35, 0.0, 1.0)),
    ("Purple",  (0.55, 0.0, 1.0, 1.0)),
    ("Lime",    (0.55, 1.0, 0.0, 1.0)),
    ("Pink",    (1.0, 0.15, 0.55, 1.0)),
    ("White",   (1.0, 1.0, 1.0, 1.0)),
    ("Black",   (0.0, 0.0, 0.0, 1.0)),
]


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def srgb_to_linear_array(rgb):
    rgb = np.clip(rgb, 0.0, 1.0)
    return np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )


def image_to_numpy(image):
    width, height = image.size[:2]
    if width <= 0 or height <= 0:
        raise RuntimeError("Base Color image has no valid pixels.")

    channels = image.channels
    pixels = np.empty(width * height * channels, dtype=np.float32)
    image.pixels.foreach_get(pixels)

    pixels = pixels.reshape((height, width, channels))

    if channels >= 3:
        return pixels[:, :, :3].copy()

    raise RuntimeError("Base Color image has fewer than 3 channels.")


def find_basecolor_image(material):
    if not material or not material.use_nodes or not material.node_tree:
        return None, None

    nodes = material.node_tree.nodes
    bsdf_nodes = [n for n in nodes if n.type == "BSDF_PRINCIPLED"]

    for bsdf in bsdf_nodes:
        socket = bsdf.inputs.get("Base Color")
        if not socket:
            continue

        # Walk upstream through the Base Color network.
        queue = [link.from_node for link in socket.links]
        visited = set()

        while queue:
            node = queue.pop(0)

            if node in visited:
                continue
            visited.add(node)

            if node.type == "TEX_IMAGE" and node.image:
                return node.image, node

            for inp in node.inputs:
                for link in inp.links:
                    queue.append(link.from_node)

    # Fallback: use image node whose name/image name looks like BaseColor.
    for node in nodes:
        if node.type != "TEX_IMAGE" or not node.image:
            continue

        text = (node.name + " " + node.image.name).lower()
        if (
            "basecolor" in text
            or "base color" in text
            or "base_color" in text
            or "albedo" in text
        ):
            return node.image, node

    return None, None


def find_uv_layer(mesh, image_node):
    if not mesh.uv_layers:
        return None

    if image_node:
        vector = image_node.inputs.get("Vector")
        if vector:
            queue = [link.from_node for link in vector.links]
            visited = set()

            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)

                if node.type == "UVMAP" and node.uv_map:
                    if node.uv_map in mesh.uv_layers:
                        return mesh.uv_layers[node.uv_map]

                for inp in node.inputs:
                    for link in inp.links:
                        queue.append(link.from_node)

    return mesh.uv_layers.active


def get_active_material(obj):
    if not obj.material_slots:
        return None

    # The reference V16 expects a single original material. For this version,
    # the active material is used as the default Base Color source.
    if obj.active_material:
        return obj.active_material

    return obj.material_slots[0].material


# ---------------------------------------------------------------------------
# Texture selection / eyedropper
# ---------------------------------------------------------------------------

def pick_face_under_mouse(context, event):
    region = context.region
    rv3d = context.region_data
    if not rv3d:
        return None

    coord = (event.mouse_region_x, event.mouse_region_y)

    origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
    direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)

    depsgraph = context.evaluated_depsgraph_get()
    best = None

    for obj in context.visible_objects:
        if obj.type != "MESH" or obj.hide_get():
            continue

        evaluated = obj.evaluated_get(depsgraph)

        try:
            inv = evaluated.matrix_world.inverted()
            local_origin = inv @ origin
            local_direction = inv.to_3x3() @ direction

            hit, location, normal, face_index = evaluated.ray_cast(
                local_origin,
                local_direction.normalized(),
            )
        except Exception:
            continue

        if not hit or face_index < 0:
            continue

        world_location = evaluated.matrix_world @ location
        distance = (world_location - origin).length

        if best is None or distance < best["distance"]:
            best = {
                "object": obj,
                "face_index": face_index,
                "distance": distance,
            }

    return best


def sample_basecolor_at_face(obj, face_index):
    mesh = obj.data
    poly = mesh.polygons[face_index]

    material = None
    if poly.material_index < len(obj.material_slots):
        material = obj.material_slots[poly.material_index].material

    if material is None:
        material = get_active_material(obj)

    image, image_node = find_basecolor_image(material)

    if image is None:
        return None, None, None

    uv_layer = find_uv_layer(mesh, image_node)
    if uv_layer is None:
        return None, material, image

    # Average the face UVs for a stable click sample.
    uv_sum = Vector((0.0, 0.0))
    count = len(poly.loop_indices)

    if count == 0:
        return None, material, image

    for loop_index in poly.loop_indices:
        uv_sum += uv_layer.data[loop_index].uv

    uv = uv_sum / count

    pixels = image_to_numpy(image)
    h, w = pixels.shape[:2]

    u = uv.x % 1.0
    v = uv.y % 1.0

    x = min(w - 1, max(0, int(u * (w - 1))))
    y = min(h - 1, max(0, int(v * (h - 1))))

    color = pixels[y, x]
    return tuple(float(x) for x in color), material, image


class CID_OT_pick_color(bpy.types.Operator):
    bl_idname = "cid17.pick_color"
    bl_label = "Pick Base Color"
    bl_description = "Click a mesh surface and sample its Base Color texture"

    rule_index: bpy.props.IntProperty(default=0)

    def invoke(self, context, event):
        if context.area.type != "VIEW_3D":
            self.report({"ERROR"}, "Run the picker from a 3D View.")
            return {"CANCELLED"}

        context.window_manager.modal_handler_add(self)
        context.area.header_text_set(
            "Base Color Picker: LEFT CLICK surface | ESC to cancel"
        )
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type in {"ESC", "RIGHTMOUSE"}:
            context.area.header_text_set(None)
            return {"CANCELLED"}

        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            hit = pick_face_under_mouse(context, event)

            if not hit:
                self.report({"WARNING"}, "No mesh surface was hit.")
                return {"RUNNING_MODAL"}

            color, material, image = sample_basecolor_at_face(
                hit["object"],
                hit["face_index"],
            )

            if color is None:
                context.area.header_text_set(None)
                self.report(
                    {"ERROR"},
                    "Could not find a readable Base Color texture on the clicked face.",
                )
                return {"CANCELLED"}

            rules = context.window_manager.cid17_rules
            if self.rule_index >= len(rules):
                context.area.header_text_set(None)
                return {"CANCELLED"}

            item = rules[self.rule_index]
            item.source_color = (*color[:3], 1.0)
            item.source_material = material.name if material else ""
            item.source_texture = image.name if image else ""

            context.area.header_text_set(None)
            self.report(
                {"INFO"},
                "Picked Base Color: "
                + ", ".join(f"{v:.3f}" for v in color[:3]),
            )
            return {"FINISHED"}

        return {"RUNNING_MODAL"}


# ---------------------------------------------------------------------------
# Rule management
# ---------------------------------------------------------------------------

class CID_OT_add_rule(bpy.types.Operator):
    bl_idname = "cid17.add_rule"
    bl_label = "Add Color"
    bl_description = "Add a new color classification rule"

    def execute(self, context):
        rules = context.window_manager.cid17_rules

        if len(rules) >= MAX_RULES:
            self.report({"WARNING"}, "Maximum of 10 colors.")
            return {"CANCELLED"}

        item = rules.add()
        item.name = f"Color {len(rules)}"
        item.output_id = OUTPUT_COLORS[(len(rules) - 1) % len(OUTPUT_COLORS)][0]
        return {"FINISHED"}


class CID_OT_remove_rule(bpy.types.Operator):
    bl_idname = "cid17.remove_rule"
    bl_label = "Remove Color"
    bl_description = "Remove this color rule"

    index: bpy.props.IntProperty(default=0)

    def execute(self, context):
        rules = context.window_manager.cid17_rules

        if len(rules) <= 1:
            self.report({"WARNING"}, "At least one color must remain.")
            return {"CANCELLED"}

        if 0 <= self.index < len(rules):
            rules.remove(self.index)

        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Fast vectorized classification
# ---------------------------------------------------------------------------

def build_texture_and_uv_cache(obj):
    """
    Load each material's Base Color image once and collect polygon UV samples
    using Blender foreach_get. This is the large-mesh path.

    It intentionally avoids:
      - Python per-face material/node discovery
      - image.pixels access per face
      - adjacency creation for the entire mesh
    """
    mesh = obj.data
    poly_count = len(mesh.polygons)

    if poly_count == 0:
        raise RuntimeError("Mesh has no polygons.")

    # Material slot indices for every polygon.
    poly_material = np.empty(poly_count, dtype=np.int32)
    mesh.polygons.foreach_get("material_index", poly_material)

    # First loop index for every polygon.
    loop_start = np.empty(poly_count, dtype=np.int32)
    mesh.polygons.foreach_get("loop_start", loop_start)

    # UV data for all loops.
    uv_layer = mesh.uv_layers.active
    if uv_layer is None:
        raise RuntimeError("Mesh has no UV map.")

    uv_data = np.empty(len(mesh.loops) * 2, dtype=np.float32)
    uv_layer.data.foreach_get("uv", uv_data)
    uv_data = uv_data.reshape((-1, 2))

    # Keep the same sampling strategy as the reference V16:
    # first UV of each polygon. This is extremely fast and stable.
    poly_uv = uv_data[loop_start]

    # Clamp instead of relying on slow per-face Python operations.
    poly_uv = np.clip(poly_uv, 0.0, 1.0)

    # One texture per material slot.
    material_images = {}
    texture_arrays = {}

    for slot_index, slot in enumerate(obj.material_slots):
        material = slot.material
        if not material:
            continue

        image, image_node = find_basecolor_image(material)
        if image is None:
            continue

        if image.name not in texture_arrays:
            texture_arrays[image.name] = image_to_numpy(image)

        material_images[slot_index] = image.name

    # If the active material is not represented by a slot, add it logically.
    if not material_images and obj.active_material:
        image, _ = find_basecolor_image(obj.active_material)
        if image:
            texture_arrays[image.name] = image_to_numpy(image)
            material_images[0] = image.name

    if not material_images:
        raise RuntimeError(
            "No Base Color Image Texture was found on the mesh materials."
        )

    # Allocate the final sampled RGB array.
    rgb = np.zeros((poly_count, 3), dtype=np.float32)

    # Group polygons by material slot so every texture is indexed in one
    # vectorized operation.
    for slot_index, image_name in material_images.items():
        mask = (poly_material == slot_index)
        indices = np.flatnonzero(mask)

        if indices.size == 0:
            continue

        pixels = texture_arrays[image_name]
        h, w = pixels.shape[:2]

        uv = poly_uv[indices]

        x = np.clip(
            (uv[:, 0] * (w - 1)).astype(np.int32),
            0,
            w - 1,
        )
        y = np.clip(
            (uv[:, 1] * (h - 1)).astype(np.int32),
            0,
            h - 1,
        )

        rgb[indices] = pixels[y, x, :3]

    return rgb


def classify_faces(rgb, rules):
    """
    Vectorized rule matching.

    A face is assigned by the first rule that matches it. This makes rule
    order useful for overlapping color ranges: put the more specific colors
    first.
    """
    count = rgb.shape[0]
    assignment = np.full(count, -1, dtype=np.int16)
    unassigned = np.ones(count, dtype=bool)

    rgb_linear = srgb_to_linear_array(rgb)

    for rule_index, rule in enumerate(rules):
        target = np.asarray(rule["color"][:3], dtype=np.float32)
        target_linear = srgb_to_linear_array(target.reshape(1, 3))[0]

        delta = rgb_linear - target_linear
        distance = np.sqrt(np.sum(delta * delta, axis=1))

        match = (distance <= rule["range"]) & unassigned

        assignment[match] = rule_index
        unassigned[match] = False

        if not np.any(unassigned):
            break

    return assignment


def build_face_adjacency(mesh):
    """Build undirected face adjacency from shared mesh edges."""
    nfaces = len(mesh.polygons)
    nloops = len(mesh.loops)

    verts = np.empty(nloops, dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", verts)

    starts = np.empty(nfaces, dtype=np.int64)
    totals = np.empty(nfaces, dtype=np.int32)
    mesh.polygons.foreach_get("loop_start", starts)
    mesh.polygons.foreach_get("loop_total", totals)

    owners = np.repeat(np.arange(nfaces, dtype=np.int32), totals)
    li = np.arange(nloops, dtype=np.int64)

    starts_l = np.repeat(starts, totals)
    totals_l = np.repeat(totals, totals)
    nxt = li + 1
    last = (li - starts_l + 1) >= totals_l
    nxt[last] = starts_l[last]

    lo = np.minimum(verts, verts[nxt])
    hi = np.maximum(verts, verts[nxt])

    keys = np.empty(nloops, dtype=[("a", np.int32), ("b", np.int32)])
    keys["a"], keys["b"] = lo, hi
    order = np.argsort(keys, kind="mergesort")
    keys = keys[order]
    owners = owners[order]

    same = (
        (keys[1:]["a"] == keys[:-1]["a"]) &
        (keys[1:]["b"] == keys[:-1]["b"]) &
        (keys[1:]["a"] != keys[1:]["b"])
    )
    if not np.any(same):
        return np.empty(0, np.int32), np.empty(0, np.int32)

    u = owners[:-1][same]
    v = owners[1:][same]
    good = u != v
    return u[good], v[good]


def local_majority_smooth(assignment, mesh, passes):
    """Remove isolated classification noise using strict neighbor majority."""
    passes = int(passes)
    if passes <= 0 or assignment.size == 0:
        return assignment

    u, v = build_face_adjacency(mesh)
    if u.size == 0:
        return assignment

    n = assignment.size
    src = np.concatenate((u, v))
    dst = np.concatenate((v, u))
    order = np.argsort(src, kind="quicksort")
    src, dst = src[order], dst[order]

    counts = np.bincount(src, minlength=n).astype(np.int64)
    offsets = np.empty(n + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])

    result = assignment.copy()

    for _ in range(passes):
        new_result = result.copy()

        for f in np.flatnonzero(result >= 0):
            lo, hi = offsets[f], offsets[f + 1]
            if lo == hi:
                continue

            labels = result[dst[lo:hi]]
            labels = labels[labels >= 0]
            if labels.size < 2:
                continue

            vals, cnts = np.unique(labels, return_counts=True)
            best_i = int(np.argmax(cnts))
            best_label = int(vals[best_i])
            best_count = int(cnts[best_i])

            # Strict majority only: ties and weak minorities stay unchanged.
            if best_label != int(result[f]) and best_count * 2 > labels.size:
                new_result[f] = best_label

        if np.array_equal(new_result, result):
            break
        result = new_result

    return result


def get_or_create_id_material(name):
    color = dict(OUTPUT_COLORS)[name]
    material_name = "CID17_" + name

    mat = bpy.data.materials.get(material_name)

    if mat is None:
        mat = bpy.data.materials.new(material_name)

    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")

    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Roughness"].default_value = 1.0

    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    mat.diffuse_color = color

    return mat


def assign_material_indices(obj, assignment, rules):
    mesh = obj.data

    # Reuse/create output material slots.
    slot_for_rule = {}

    for rule_index, rule in enumerate(rules):
        if not np.any(assignment == rule_index):
            continue

        material = get_or_create_id_material(rule["output_id"])

        slot_index = None
        for i, slot in enumerate(obj.material_slots):
            if slot.material == material:
                slot_index = i
                break

        if slot_index is None:
            mesh.materials.append(material)
            slot_index = len(obj.material_slots) - 1

        slot_for_rule[rule_index] = slot_index

    # Material index array is generated in NumPy and written in one call.
    output_indices = np.empty(len(mesh.polygons), dtype=np.int32)

    # Preserve original assignment for unmatched faces.
    mesh.polygons.foreach_get("material_index", output_indices)

    for rule_index, slot_index in slot_for_rule.items():
        output_indices[assignment == rule_index] = slot_index

    mesh.polygons.foreach_set("material_index", output_indices)
    mesh.update()


class CID_OT_generate(bpy.types.Operator):
    bl_idname = "cid17.generate"
    bl_label = "Generate Color IDs"
    bl_description = "Classify Base Color and assign ID materials using NumPy"

    def execute(self, context):
        obj = context.active_object

        if not obj or obj.type != "MESH":
            self.report({"ERROR"}, "Select a mesh object.")
            return {"CANCELLED"}

        rules = context.window_manager.cid17_rules

        if not rules:
            self.report({"ERROR"}, "Add at least one color rule.")
            return {"CANCELLED"}

        rule_data = [
            {
                "color": tuple(item.source_color),
                "range": float(item.color_range),
                "output_id": item.output_id,
            }
            for item in rules
        ]

        try:
            self.report(
                {"INFO"},
                f"Reading Base Color for {len(obj.data.polygons):,} faces...",
            )

            rgb = build_texture_and_uv_cache(obj)

            assignment = classify_faces(
                rgb,
                rule_data,
            )

            smooth_passes = max(
                [int(item.smooth_passes) for item in rules],
                default=0,
            )
            assignment = local_majority_smooth(
                assignment,
                obj.data,
                smooth_passes,
            )

            matched = assignment >= 0

            if not np.any(matched):
                self.report(
                    {"WARNING"},
                    "No faces matched the selected color ranges.",
                )
                return {"CANCELLED"}

            assign_material_indices(
                obj,
                assignment,
                rule_data,
            )

            counts = [
                int(np.count_nonzero(assignment == i))
                for i in range(len(rule_data))
            ]

            summary = ", ".join(
                f"{rules[i].output_id}: {counts[i]:,}"
                for i in range(len(rules))
                if counts[i] > 0
            )

            self.report(
                {"INFO"},
                f"Done. Matched {int(np.count_nonzero(matched)):,} faces. {summary}",
            )

        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        return {"FINISHED"}




# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

class CID17_Rule(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty(default="Color")

    source_color: bpy.props.FloatVectorProperty(
        name="Source Color",
        subtype="COLOR",
        size=4,
        min=0.0,
        max=1.0,
        default=(0.5, 0.5, 0.5, 1.0),
    )

    source_material: bpy.props.StringProperty(default="")
    source_texture: bpy.props.StringProperty(default="")

    color_range: bpy.props.FloatProperty(
        name="Color Range",
        description="Linear RGB distance allowed around the selected source color",
        default=0.10,
        min=0.0,
        max=1.5,
        soft_min=0.0,
        soft_max=0.5,
    )

    min_region_faces: bpy.props.IntProperty(
        name="Min Region Faces",
        description="Reserved for region cleanup; current large-mesh pass does not build full adjacency",
        default=20,
        min=1,
        max=1000000,
    )

    smooth_passes: bpy.props.IntProperty(
        name="Smooth Passes",
        description="Number of local-majority cleanup passes",
        default=1,
        min=0,
        max=5,
    )

    output_id: bpy.props.EnumProperty(
        name="Output ID",
        items=[
            (name, name, "Output " + name)
            for name, _ in OUTPUT_COLORS
        ],
        default="Red",
    )


# ---------------------------------------------------------------------------
# UI -- kept intentionally close to the v6 layout
# ---------------------------------------------------------------------------

class CID17_PT_panel(bpy.types.Panel):
    bl_label = "Color ID Extractor"
    bl_idname = "CID17_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Color ID"

    def draw_mapping(self, layout, item, index):
        box = layout.box()

        header = box.row()
        header.label(text=f"Color {index + 1}", icon="MATERIAL")

        remove = header.operator(
            "cid17.remove_rule",
            text="",
            icon="X",
        )
        remove.index = index

        row = box.row(align=True)
        row.prop(item, "source_color", text="Source Color")

        pick = row.operator(
            "cid17.pick_color",
            text="Pick Base Color",
            icon="EYEDROPPER",
        )
        pick.rule_index = index

        row = box.row(align=True)
        row.label(text="Color Range")
        row.prop(item, "color_range", text="", slider=True)

        row = box.row(align=True)
        row.label(text="Min Region Faces")
        row.prop(item, "min_region_faces", text="")

        row = box.row(align=True)
        row.label(text="Smooth Passes")
        row.prop(item, "smooth_passes", text="")

        row = box.row(align=True)
        row.label(text="Output ID")
        row.prop(item, "output_id", text="")

        if item.source_material:
            box.label(
                text="Material: " + item.source_material,
                icon="MATERIAL",
            )

        if item.source_texture:
            box.label(
                text="Texture: " + item.source_texture,
                icon="IMAGE_DATA",
            )

    def draw(self, context):
        wm = context.window_manager
        rules = wm.cid17_rules

        if len(rules) == 0:
            item = rules.add()
            item.name = "Color 1"
            item.output_id = "Red"

        layout = self.layout

        row = layout.row()
        row.label(text="Color Mappings")
        row.label(text=f"{len(rules)} / {MAX_RULES}")

        for index, item in enumerate(rules):
            self.draw_mapping(layout, item, index)

        row = layout.row()
        row.enabled = len(rules) < MAX_RULES
        row.operator(
            "cid17.add_rule",
            text="Add Color",
            icon="ADD",
        )

        layout.separator()

        row = layout.row()
        row.scale_y = 1.35
        row.operator(
            "cid17.generate",
            text="Generate Color IDs",
            icon="MATERIAL",
        )

        layout.separator()

        info = layout.box()
        info.label(text="Large Mesh Mode")
        info.label(text="NumPy / foreach_get")
        info.label(text="Base Color driven")
        info.label(text="UV and geometry unchanged")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

CLASSES = (
    CID17_Rule,
    CID_OT_pick_color,
    CID_OT_add_rule,
    CID_OT_remove_rule,
    CID_OT_generate,
    CID17_PT_panel,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)

    # Store temporary rule data on WindowManager instead of Scene.
    # This avoids Blender 5.2's restricted Scene datablock context during
    # legacy add-on enable/reload.
    bpy.types.WindowManager.cid17_rules = bpy.props.CollectionProperty(
        type=CID17_Rule,
    )


def unregister():
    if hasattr(bpy.types.WindowManager, "cid17_rules"):
        del bpy.types.WindowManager.cid17_rules

    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()

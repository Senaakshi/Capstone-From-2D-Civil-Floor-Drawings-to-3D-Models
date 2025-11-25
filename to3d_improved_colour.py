#!/usr/bin/env python3
import os, json
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
import trimesh

# --- units and base heights (tweak as you like) ---
PX_TO_M  = 0.01   # 1 px = 1 cm
WALL_H_BASE   = 3.0    # wall height
FLOOR_T  = 0.15   # floor slab thickness
DOOR_H   = 2.1    # door opening height
WIN_SILL = 0.9    # window sill height above floor
WIN_HEAD = 2.1    # window head height above floor

# --- colors (RGBA 0–255) ------------------------------------
# Light-brown walls, blue glass windows, beige floor
COLOR_WALL   = [120, 210, 255, 140]     # light brown walls (semi-opaque)
#COLOR_WINDOW = [120, 210, 255, 140]   # blue/teal glass (semi-transparent)
COLOR_FLOOR  = [235, 230, 220, 255]   # soft beige floor (opaque)


def apply_color(mesh, rgba):
    """
    Attach per-vertex RGBA colors to a trimesh mesh.
    """
    rgba = np.array(rgba, dtype=np.uint8)
    n = len(mesh.vertices)
    colors = np.tile(rgba, (n, 1))
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=colors)
    return mesh


def poly_from_points(pts):
    """
    Convert list of (x,y) points → valid shapely Polygon, or None if invalid/small.
    """
    if len(pts) < 3:
        return None
    try:
        poly = Polygon(pts)
        if not poly.is_valid:
            poly = poly.buffer(0)  # fix self-intersections
        if poly.is_empty or poly.area < 10:
            return None
        return poly
    except Exception:
        return None


def load_polygons(json_path):
    """
    Load polygons from the polygons.json created by the improved UNet script.
    Expects "class" to be one of: wall, room, door, window.
    """
    with open(json_path, "r") as f:
        items = json.load(f)

    walls, rooms, doors, windows = [], [], [], []

    for it in items:
        cls = it.get("class", "")
        pts = it.get("points", [])
        poly = poly_from_points(pts)
        if poly is None:
            continue

        if cls == "wall":
            walls.append(poly)
        elif cls == "room":
            rooms.append(poly)
        elif cls == "door":
            doors.append(poly)
        elif cls == "window":
            windows.append(poly)

    return walls, rooms, doors, windows


def iter_polys(geom):
    """Yield Polygon(s) from Polygon/MultiPolygon/GeometryCollection."""
    if geom is None or geom.is_empty:
        return
    gt = geom.geom_type
    if gt == "Polygon":
        yield geom
    elif gt in ("MultiPolygon", "GeometryCollection"):
        for g in geom.geoms:
            if g.is_empty:
                continue
            if g.geom_type in ("Polygon", "MultiPolygon", "GeometryCollection"):
                yield from iter_polys(g)
            # ignore non-area types


def extrude(poly, height_m, z0=0.0, scale=1.0, engine="triangle", color=None):
    """
    Extrude a 2D Polygon into a 3D mesh using trimesh.
    - poly: shapely Polygon in pixel coordinates
    - height_m: extrusion height in meters
    - z0: base elevation
    - scale: px→m
    - color: optional RGBA list to color the mesh
    """
    # scale pixel coords to meters first
    ext = np.array(poly.exterior.coords)
    scaled_ext = [(x * scale, y * scale) for x, y in ext]

    holes = []
    for r in poly.interiors:
        r_coords = np.array(r.coords)
        holes.append([(x * scale, y * scale) for x, y in r_coords])

    poly_scaled = Polygon(scaled_ext, holes if holes else None)

    # Use explicit engine for robustness
    mesh = trimesh.creation.extrude_polygon(poly_scaled, height_m, engine=engine)
    mesh.apply_translation([0, 0, z0])

    if color is not None:
        apply_color(mesh, color)

    return mesh


# ---------- auto height scaling -------------------------------------------

def estimate_wall_scale_factor(wall_polys, ref_thickness_px=30.0,
                               min_scale=0.4, max_scale=1.8):
    """
    Estimate a wall height scale factor based on average wall thickness in pixels.

    If walls are drawn very thick in pixels, the factor will be < 1
    (so wall height shrinks). If walls are thin, factor will be > 1.

    ref_thickness_px: thickness (in px) at which scale_factor ~ 1.
    min_scale/max_scale: clamp to avoid crazy extremes.
    """
    thicknesses = []

    for poly in wall_polys:
        minx, miny, maxx, maxy = poly.bounds
        # approximate wall thickness as smallest side of bounding box
        th = min(maxx - minx, maxy - miny)
        if th > 1:
            thicknesses.append(th)

    if not thicknesses:
        print("[info] Could not estimate wall thickness; using base height.")
        return 1.0

    avg_th = float(np.mean(thicknesses))
    if avg_th <= 0:
        return 1.0

    # If avg thickness is bigger than reference, scale down height
    raw_scale = ref_thickness_px / avg_th
    scale_factor = max(min(raw_scale, max_scale), min_scale)

    print(f"[auto-scale] avg wall thickness: {avg_th:.2f}px, "
          f"scale_factor: {scale_factor:.3f}")
    return scale_factor


# ---------- main build functions -----------------------------------------

def build_mesh(json_path, out_path):
    """
    Full pipeline:
      - Load wall, room, door, window polygons.
      - Auto-scale wall height based on wall thickness.
      - Subtract doors/windows from walls to create openings.
      - Extrude:
          * floors from rooms (soft beige)
          * walls with openings (light brown)
          * window panels as glass (blue, between sill & head)
      - Export GLB with vertex colors.
    """
    walls, rooms, doors, windows = load_polygons(json_path)

    wall_union = unary_union(walls) if walls else None
    room_union = unary_union(rooms) if rooms else None
    door_union = unary_union(doors) if doors else None
    win_union  = unary_union(windows) if windows else None

    # Auto-scale wall height based on wall thickness
    if walls:
        scale_factor = estimate_wall_scale_factor(walls)
    else:
        scale_factor = 1.0
    wall_height = WALL_H_BASE * scale_factor

    # Combine door + window footprints as openings in walls
    openings = None
    if door_union is not None and not door_union.is_empty:
        openings = door_union
    if win_union is not None and not win_union.is_empty:
        openings = win_union if openings is None else openings.union(win_union)

    meshes = []

    # --- floors from room polygons (colored floor slab) ---
    if room_union is not None and not room_union.is_empty:
        for rp in iter_polys(room_union):
            floor_mesh = extrude(
                rp,
                FLOOR_T,
                z0=0.0,
                scale=PX_TO_M,
                color=COLOR_FLOOR,
            )
            meshes.append(floor_mesh)
    else:
        print(f"[warn] No room polygons found for floors in {json_path}.")

    # --- walls, subtracting doors+windows as openings (light brown) ---
    if wall_union is not None and not wall_union.is_empty:
        walls_with_openings = wall_union
        if openings is not None and not openings.is_empty:
            # small buffer to make subtraction robust
            walls_with_openings = walls_with_openings.difference(openings.buffer(0.01))

        for wp in iter_polys(walls_with_openings):
            wall_mesh = extrude(
                wp,
                wall_height,
                z0=FLOOR_T,
                scale=PX_TO_M,
                color=COLOR_WALL,
            )
            meshes.append(wall_mesh)
    else:
        print(f"[warn] No wall polygons found in {json_path}.")

    # --- glass window panels (blue/teal, semi-transparent) ---
    #if win_union is not None and not win_union.is_empty:
    #    win_height = WIN_HEAD - WIN_SILL
    #    for wp in iter_polys(win_union):
    #        win_mesh = extrude(
    #            wp,
    #            win_height,
    #            z0=WIN_SILL,       # bottom at sill, top at head
    #            scale=PX_TO_M,
    #            color=COLOR_WINDOW,
    #        )
    #        meshes.append(win_mesh)
    #else:
    #    print(f"[info] No window polygons found in {json_path} (no glass panels created).")

    if not meshes:
        raise RuntimeError(f"No geometry to export (check polygons JSON: {json_path}).")

    # Combine all meshes into one mesh; vertex colors stay attached
    scene_mesh = trimesh.util.concatenate(meshes)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    scene_mesh.export(out_path)
    print("3D saved →", out_path)


def build_mesh_folder(json_dir, out_dir="out_3d"):
    """
    Batch version:
      - Looks for all *_polygons.json in json_dir
      - Builds a GLB for each one
      - Saves GLBs into out_dir
    """
    json_dir = Path(json_dir)
    out_dir = Path(out_dir)
    if not json_dir.exists():
        raise FileNotFoundError(f"JSON folder not found: {json_dir}")

    json_files = sorted(json_dir.glob("*_polygons.json"))
    if not json_files:
        print(f"No *_polygons.json files found in {json_dir}")
        return

    print(f"Found {len(json_files)} polygons files in {json_dir}\n")

    out_dir.mkdir(parents=True, exist_ok=True)

    for i, jp in enumerate(json_files, 1):
        # derive a clean stem for GLB name
        stem = jp.stem
        if stem.endswith("_polygons"):
            stem = stem[:-9]  # drop the suffix "_polygons"
        glb_path = out_dir / f"{stem}.glb"

        print(f"[{i}/{len(json_files)}] {jp.name} → {glb_path.name}")
        try:
            build_mesh(str(jp), str(glb_path))
        except Exception as e:
            print(f"  [error] Failed on {jp.name}: {e}")

    print("\nDONE: exported all available models to", out_dir)


if __name__ == "__main__":
    # --- OPTION 1: single file ---
    # jp  = "pred_unet_improved/0_1_polygons.json"
    # out = "out_3d/0_1.glb"
    # build_mesh(jp, out)

    # --- OPTION 2: whole folder of polygons JSONs ---
    build_mesh_folder(
        json_dir="/Users/senakshikrishnamurthy/Desktop/DS/606 capstone/Capstone-From-2D-Civil-Floor-Drawings-to-3D-Models/pred_unet_improved",
        out_dir="out_3d_improved_colour"
    )

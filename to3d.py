#!/usr/bin/env python3
import os, json
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
import trimesh

PX_TO_M = 0.01   # 1 px = 1 cm (tweak)
WALL_H  = 3.0
FLOOR_T = 0.15

def poly_from_points(pts):
    if len(pts) < 3: return None
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
    with open(json_path, "r") as f:
        items = json.load(f)
    walls, rooms = [], []
    for it in items:
        poly = poly_from_points(it["points"])
        if poly is None: 
            continue
        if it["class"] == "wall":
            walls.append(poly)
        elif it["class"] == "room":
            rooms.append(poly)
    return walls, rooms

def iter_polys(geom):
    """Yield Polygon(s) from Polygon/MultiPolygon/GeometryCollection."""
    if geom is None or geom.is_empty:
        return
    gt = geom.geom_type
    if gt == "Polygon":
        yield geom
    elif gt == "MultiPolygon" or gt == "GeometryCollection":
        for g in geom.geoms:
            if g.is_empty: 
                continue
            if g.geom_type in ("Polygon","MultiPolygon","GeometryCollection"):
                yield from iter_polys(g)
            # ignore non-area types (LineString, Point)

def extrude(poly, height_m, z0=0.0, scale=1.0):
    # scale pixel coords to meters first
    ext = np.array(poly.exterior.coords)
    scaled_ext = [(x*scale, y*scale) for x,y in ext]
    holes = []
    for r in poly.interiors:
        holes.append([(x*scale, y*scale) for x,y in np.array(r.coords)])
    poly_scaled = Polygon(scaled_ext, holes if holes else None)
    mesh = trimesh.creation.extrude_polygon(poly_scaled, height_m)
    mesh.apply_translation([0,0,z0])
    return mesh

def build_mesh(json_path, out_path):
    walls, rooms = load_polygons(json_path)

    wall_union  = unary_union(walls) if walls else None
    room_union  = unary_union(rooms) if rooms else None

    meshes = []

    # floors
    for rp in iter_polys(room_union):
        meshes.append(extrude(rp, FLOOR_T, z0=0.0, scale=PX_TO_M))

    # walls
    for wp in iter_polys(wall_union):
        meshes.append(extrude(wp, WALL_H, z0=FLOOR_T, scale=PX_TO_M))

    if not meshes:
        raise RuntimeError("No geometry to export (check polygons).")

    scene = trimesh.util.concatenate(meshes)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    scene.export(out_path)
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
    # jp  = "pred_unet_improved/5_16_polygons.json"
    # out = "out_3d/5_16.glb"
    # build_mesh(jp, out)

    # --- OPTION 2: whole folder of polygons JSONs ---
    build_mesh_folder(
        json_dir="/Users/senakshikrishnamurthy/Desktop/DS/606 capstone/Capstone-From-2D-Civil-Floor-Drawings-to-3D-Models/pred_unet",   # folder where *_polygons.json live
        out_dir="out_3d"                # folder to store GLBs
    )


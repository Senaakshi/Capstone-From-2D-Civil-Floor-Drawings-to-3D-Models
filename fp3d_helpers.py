# fp3d_helpers.py
# Build 3D walls/doors/windows from UNet masks with modern Trimesh APIs.

import numpy as np
import cv2
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import trimesh

# -----------------------------
# Utilities
# -----------------------------
def _as_valid(poly: Polygon) -> Polygon:
    try:
        # fixes self-intersections / ring orientation
        p = poly.buffer(0)
        return p if isinstance(p, Polygon) else poly
    except Exception:
        return poly

def mask_to_polygons(mask: np.ndarray, class_id: int, min_area: float = 20.0):
    """
    Convert a single-channel mask (H, W) into Shapely polygons for given class_id.
    Removes tiny areas and fixes polygon validity.
    """
    m = (mask == class_id).astype(np.uint8) * 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for cnt in contours:
        if len(cnt) < 3:
            continue
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        pts = cnt[:, 0, :].astype(float)
        poly = Polygon(pts)
        if not poly.is_valid:
            poly = _as_valid(poly)
        if poly.is_valid and poly.area >= min_area:
            polys.append(poly)
    if not polys:
        return []
    mp = unary_union(polys)
    if isinstance(mp, Polygon):
        return [mp]
    if isinstance(mp, MultiPolygon):
        return [g for g in mp.geoms if g.is_valid and g.area >= min_area]
    return []

def extrude_polygon_to_prism(poly: Polygon, height: float) -> trimesh.Trimesh:
    """
    Robust extrusion using trimesh.creation.extrude_polygon (works on current Trimesh).
    """
    from trimesh.creation import extrude_polygon
    # Ensure valid polygon (holes handled by Shapely)
    if not poly.is_valid:
        poly = _as_valid(poly)
    try:
        mesh = extrude_polygon(poly, height)
        # Ensure watertightness where possible
        if hasattr(mesh, "merge_vertices"):
            mesh.merge_vertices()
        return mesh
    except Exception as e:
        print("Extrusion failed:", e)
        return trimesh.Trimesh()

def _safe_concat(meshes):
    meshes = [m for m in meshes if isinstance(m, trimesh.Trimesh) and not m.is_empty]
    if not meshes:
        return None
    if len(meshes) == 1:
        return meshes[0]
    try:
        return trimesh.util.concatenate(meshes)
    except Exception as e:
        print("Concatenate failed:", e)
        return meshes[0]

# -----------------------------
# Scene assembly
# -----------------------------
def build_scene_from_masks(mask: np.ndarray,
                           wall_h=3.0,
                           door_h=2.1,
                           win_sill=0.9,
                           win_head=2.1,
                           inset_ratio=0.98):
    """
    Build a scene with walls (solids), doors (panels), and windows (glass panes).
    Classes: 1=wall, 2=window, 3=door (0=background).
    Returns (parts_dict, trimesh.Scene)
    """
    # 1) Polygons per class
    wall_polys = mask_to_polygons(mask, 1)
    win_polys  = mask_to_polygons(mask, 2)
    door_polys = mask_to_polygons(mask, 3)

    # 2) Extrude
    wall_meshes = [extrude_polygon_to_prism(p, wall_h) for p in wall_polys]
    walls = _safe_concat(wall_meshes)

    door_meshes = [extrude_polygon_to_prism(p, door_h) for p in door_polys]
    doors = _safe_concat(door_meshes)
    if doors is not None and inset_ratio is not None:
        try:
            doors.apply_scale(inset_ratio)
        except Exception:
            pass

    win_meshes = []
    for p in win_polys:
        pane = extrude_polygon_to_prism(p, max(0.0, win_head - win_sill))
        if not pane.is_empty:
            # Lift to sill height and inset slightly so it doesn't z-fight with walls
            try:
                pane.apply_translation([0, 0, win_sill])
                if inset_ratio is not None:
                    pane.apply_scale(inset_ratio)
            except Exception:
                pass
            win_meshes.append(pane)
    windows = _safe_concat(win_meshes)

    # 3) Optional: boolean cut (depends on available boolean engine; safe fallback)
    walls_cut = walls
    try:
        if walls is not None and (doors is not None or windows is not None):
            to_sub = [m for m in [doors, windows] if m is not None and not m.is_empty]
            if to_sub:
                subtract = _safe_concat(to_sub)
                # Try trimesh boolean; will raise if no engine (e.g., cork/openSCAD not present)
                walls_cut = trimesh.boolean.difference(walls, subtract)
    except Exception as e:
        # Fallback: keep walls as-is and just overlay doors/windows
        print("Boolean difference skipped:", e)
        walls_cut = walls

    parts = {
        "walls":   walls_cut if walls_cut is not None else trimesh.Trimesh(),
        "doors":   doors if doors is not None else trimesh.Trimesh(),
        "windows": windows if windows is not None else trimesh.Trimesh(),
    }

    # 4) Scene
    scene = trimesh.Scene()
    for name in ["walls", "doors", "windows"]:
        m = parts[name]
        if isinstance(m, trimesh.Trimesh) and not m.is_empty:
            scene.add_geometry(m, geom_name=name)
    return parts, scene

# -----------------------------
# PyVista bridge (for nice colors)
# -----------------------------
def colorize_scene_for_pyvista(parts):
    """
    Convert trimesh parts to PyVista PolyData with colors & opacity.
    Returns dict: name -> (pv.PolyData, color_rgb, opacity)
    """
    try:
        import pyvista as pv
    except Exception as e:
        raise RuntimeError(f"PyVista not installed: {e}")

    def to_pv(mesh: trimesh.Trimesh):
        if mesh is None or mesh.is_empty:
            return None
        v = mesh.vertices
        # faces are a flat (n*3,) array of vertex indices; PyVista needs [3, i, j, k, 3, ...]
        f = np.hstack([np.full((len(mesh.faces), 1), 3, dtype=np.int64), mesh.faces.reshape(-1, 3)]).ravel()
        surf = pv.PolyData(v, f)
        surf.clean(inplace=True)
        return surf

    out = {}
    w = to_pv(parts.get("walls"))
    d = to_pv(parts.get("doors"))
    g = to_pv(parts.get("windows"))

    if w is not None: out["walls"]   = (w, (200, 200, 200), 1.0)  # light gray, opaque
    if d is not None: out["doors"]   = (d, (150, 100, 50), 1.0)   # wood-ish
    if g is not None: out["windows"] = (g, (0, 200, 255), 0.2)    # teal glass

    return out

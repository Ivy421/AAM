"""
model_postprocess.py

Purpose
-------
Repair depression/model.stl before boolean union with the loft gripper.
The output mesh is required to satisfy, as much as possible:
    - watertight == True
    - is_volume == True
    - winding_consistent == True

Recommended run order:
    Depression_Completion.py
        -> repairModel_mesh.py
        -> model_postprocess.py
        -> depression_grip+loft.py

Notes
-----
1. The script first tries normal mesh repair with trimesh / PyMeshLab.
2. If the repaired mesh is still not a valid closed volume, it uses a voxel
   solidification fallback. This sacrifices a little surface detail but is much
   more reliable for boolean union.
3. If OVERWRITE_INPUT_MODEL is True, the original model.stl is backed up and
   replaced by the repaired result, so depression_grip+loft.py can keep using
   INPUT_STL = model.stl directly.
"""

import os
import shutil
import traceback
from pathlib import Path
import trimesh
import numpy as np

# =========================================================
# User config
# =========================================================
depression_dir = r"E:/HKUSTGZ/AAM/construction/data/completion_result/depression"
INPUT_MESH = os.path.join(depression_dir, "model.stl")
OUTPUT_MESH = os.path.join(depression_dir, "model_processed.stl")

# Remove tiny disconnected fragments whose surface area is smaller than this
# ratio of the total mesh area. Set 0.0 to keep all components.
MIN_COMPONENT_AREA_RATIO = 1e-5

# Final visualization. Keep False for batch execution.
VISUALIZE = False

# =========================================================
# Basic utilities
# =========================================================
def _as_mesh(obj):
    """Convert a trimesh load result, possibly a Scene, into one Trimesh."""
    import trimesh

    if isinstance(obj, trimesh.Scene):
        geoms = [g for g in obj.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if len(geoms) == 0:
            raise RuntimeError("Loaded scene contains no mesh geometry")
        return trimesh.util.concatenate(geoms)
    if isinstance(obj, trimesh.Trimesh):
        return obj
    raise TypeError(f"Unsupported mesh type: {type(obj)}")


def load_trimesh_mesh(path):
    #mesh = _as_mesh(trimesh.load(path, force="mesh"))
    mesh = trimesh.load(path, force="mesh")
    if mesh.vertices is None or mesh.faces is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise RuntimeError(f"Empty mesh: {path}")
    return mesh


def clean_trimesh(mesh, keep_large_components=True):
    """Basic cleanup without changing the intended shape too much."""
    import trimesh

    mesh = mesh.copy()
    mesh.remove_infinite_values()
    #mesh.remove_duplicate_faces()
    #mesh.remove_degenerate_faces()
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices(digits_vertex=8)

    if keep_large_components:
        parts = mesh.split(only_watertight=False)
        if len(parts) > 1:
            total_area = float(sum(max(p.area, 0.0) for p in parts))
            if total_area > 0:
                kept = [p for p in parts if p.area >= MIN_COMPONENT_AREA_RATIO * total_area]
            else:
                kept = [max(parts, key=lambda p: len(p.faces))]

            if len(kept) == 0:
                kept = [max(parts, key=lambda p: len(p.faces))]
            mesh = trimesh.util.concatenate(kept)

    mesh.remove_unreferenced_vertices()
    return mesh


def force_consistent_positive_volume(mesh):
    """Fix winding/normals and make volume positive when possible."""
    mesh = mesh.copy()
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fix_inversion(mesh)

    try:
        if mesh.volume < 0:
            mesh.invert()
            trimesh.repair.fix_normals(mesh)
    except Exception:
        pass

    mesh.remove_unreferenced_vertices()
    return mesh


def mesh_status(mesh, name="mesh"):
    print(f"\n========== {name} status ==========")
    print("vertices:", len(mesh.vertices))
    print("faces:", len(mesh.faces))
    print("extents:", mesh.extents)
    print("area:", float(mesh.area))
    try:
        print("volume:", float(mesh.volume))
    except Exception as e:
        print("volume: <failed>", e)
    print("watertight:", bool(mesh.is_watertight))
    print("volume flag:", bool(mesh.is_volume))
    print("winding consistent:", bool(mesh.is_winding_consistent))
    try:
        print("euler_number:", mesh.euler_number)
    except Exception:
        pass


def is_valid_volume(mesh):
    return bool(mesh.is_watertight and mesh.is_volume and mesh.is_winding_consistent)



# =========================================================
# trimesh repair
# =========================================================
def repair_with_trimesh(input_path):
    print("\n=== Strategy 1: trimesh direct repair ===")
    mesh = load_trimesh_mesh(input_path)
    mesh_status(mesh, "input")
    mesh = clean_trimesh(mesh)

    # Fill simple boundary holes. This only works for relatively small/simple holes,
    # but is shape-preserving and should be tried before remeshing.
    try:
        filled = trimesh.repair.fill_holes(mesh)
        print("fill_holes result:", filled)
    except Exception as e:
        print("fill_holes skipped:", e)

    mesh = force_consistent_positive_volume(mesh)

    return is_valid_volume(mesh), mesh

# =========================================================
# Optional Open3D visualization
# =========================================================
def visualize_mesh(mesh_path, window_name="Mesh Viewer"):
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if mesh.is_empty():
        raise RuntimeError(f"读取失败或网格为空: {mesh_path}")
    mesh.compute_vertex_normals()

    print(f"[Visualize] {mesh_path}")
    print(f"vertices = {len(mesh.vertices)}, faces = {len(mesh.triangles)}")

    o3d.visualization.draw_geometries(
        [mesh],
        window_name=window_name,
        mesh_show_back_face=True,
    )


# =========================================================
# Main
# =========================================================
def main():
    if not os.path.exists(INPUT_MESH):
        raise FileNotFoundError(f"找不到输入文件: {INPUT_MESH}")

    strategies = [("trimesh direct repair", lambda: repair_with_trimesh(INPUT_MESH))]

    final_mesh = None
    final_strategy = None

    for name, func in strategies:
        try:
            ok, mesh = func()
            if mesh is not None:
                final_mesh = mesh
                #final_strategy = name
            if ok:
                print(f"\n[SUCCESS] Valid volume produced by: {name}")
                break
            else:
                print(f"\n[NOT VALID] {name} did not produce a valid volume. Trying next strategy...")
        except Exception as e:
            print(f"\n[FAILED] {name}: {e}")
            traceback.print_exc()

    if final_mesh is None:
        raise RuntimeError("trimesh 修复策略都失败，未能生成 mesh")

    final_mesh = force_consistent_positive_volume(clean_trimesh(final_mesh))
    final_mesh.export(OUTPUT_MESH)
    mesh_status(final_mesh, "FINAL OUTPUT")

    if not is_valid_volume(final_mesh):
        raise RuntimeError(
            "输出 mesh 仍不满足 watertight / volume / winding consistent。"
            "建议减小 VOXEL_PITCH_MM，例如 0.20 或 0.15，或检查输入 model.stl 是否严重自交。")

    print("Saved processed mesh:", OUTPUT_MESH)

    if VISUALIZE:
        visualize_mesh(OUTPUT_MESH, window_name="Processed watertight STL")


if __name__ == "__main__":
    main()
    mesh = trimesh.load(depression_dir + '/model_processed.stl',force = 'mesh', process = True )
    mesh = force_consistent_positive_volume(clean_trimesh(mesh))
    print("watertight:", bool(mesh.is_watertight))
    print("volume flag:", bool(mesh.is_volume))
    print("winding consistent:", bool(mesh.is_winding_consistent))

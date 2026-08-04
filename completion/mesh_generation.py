"""
mesh_generation.py

Generate and postprocess the depression repair mesh in one step:
    model.pcd -> model.stl -> model_processed.stl

This merges the useful parts of repairModel_mesh.py and
depression_model_postprocess.py into one depression-focused script.
"""

import argparse
import os
import traceback

import numpy as np
import open3d as o3d
import pymeshlab
import trimesh


DEFAULT_DEPRESSION_DIR = r"E:/HKUSTGZ/AAM/construction/data/completion_result/depression"
MIN_COMPONENT_AREA_RATIO = 1e-5


def _as_mesh(obj):
    if isinstance(obj, trimesh.Scene):
        geoms = [g for g in obj.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if len(geoms) == 0:
            raise RuntimeError("Loaded scene contains no mesh geometry")
        return trimesh.util.concatenate(geoms)
    if isinstance(obj, trimesh.Trimesh):
        return obj
    raise TypeError(f"Unsupported mesh type: {type(obj)}")


def load_trimesh_mesh(path):
    mesh = _as_mesh(trimesh.load(path, force="mesh"))
    if mesh.vertices is None or mesh.faces is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise RuntimeError(f"Empty mesh: {path}")
    return mesh


def clean_trimesh(mesh, keep_large_components=True):
    mesh = mesh.copy()
    mesh.remove_infinite_values()
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
    except Exception as exc:
        print("volume: <failed>", exc)
    print("watertight:", bool(mesh.is_watertight))
    print("volume flag:", bool(mesh.is_volume))
    print("winding consistent:", bool(mesh.is_winding_consistent))
    try:
        print("euler_number:", mesh.euler_number)
    except Exception:
        pass


def is_valid_volume(mesh):
    return bool(mesh.is_watertight and mesh.is_volume and mesh.is_winding_consistent)


def point_cloud_to_stl(
    pcd_path,
    stl_path,
    alpha=0.005,
    normal_radius=0.002,
    normal_max_nn=30,
    smooth_iterations=3,
    max_hole_size=200,
    visualize=False,
):
    repair_block_pcd = o3d.io.read_point_cloud(pcd_path)
    if len(repair_block_pcd.points) == 0:
        raise RuntimeError(f"Empty point cloud: {pcd_path}")

    print("\n========== Generate mesh from repair point cloud ==========")
    print("input pcd:", pcd_path)
    print("point count:", len(repair_block_pcd.points))

    repair_block_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius,
            max_nn=normal_max_nn,
        )
    )

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(repair_block_pcd, alpha)
    if len(mesh.triangles) == 0:
        raise RuntimeError("Alpha shape reconstruction failed; mesh is empty. Try increasing alpha.")

    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()

    print("initial mesh vertices:", len(mesh.vertices))
    print("initial mesh triangles:", len(mesh.triangles))

    if smooth_iterations > 0:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=smooth_iterations)
        mesh.compute_vertex_normals()

    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.triangles)

    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=v, face_matrix=f), "repair_block")
    ms.meshing_close_holes(maxholesize=max_hole_size)
    ms.meshing_remove_connected_component_by_diameter(mincomponentdiag=pymeshlab.PercentageValue(1.0))

    m = ms.current_mesh()
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(m.vertex_matrix())
    mesh.triangles = o3d.utility.Vector3iVector(m.face_matrix())
    mesh.compute_vertex_normals()

    os.makedirs(os.path.dirname(stl_path), exist_ok=True)
    if not o3d.io.write_triangle_mesh(stl_path, mesh):
        raise RuntimeError(f"Failed to write STL: {stl_path}")

    print("saved raw STL:", stl_path)
    print("final raw mesh vertices:", len(mesh.vertices))
    print("final raw mesh triangles:", len(mesh.triangles))

    if visualize:
        mesh_show = o3d.geometry.TriangleMesh(mesh)
        mesh_show.paint_uniform_color([0.7, 0.7, 0.7])
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
        o3d.visualization.draw_geometries([mesh_show, frame], window_name="Generated repair mesh")

    return stl_path


def repair_with_trimesh(input_path):
    print("\n========== Postprocess mesh with trimesh ==========")
    mesh = load_trimesh_mesh(input_path)
    mesh_status(mesh, "input")
    mesh = clean_trimesh(mesh)

    try:
        filled = trimesh.repair.fill_holes(mesh)
        print("fill_holes result:", filled)
    except Exception as exc:
        print("fill_holes skipped:", exc)

    mesh = force_consistent_positive_volume(mesh)
    return is_valid_volume(mesh), mesh


def postprocess_stl(input_mesh, output_mesh, require_valid_volume=True, visualize=False):
    if not os.path.exists(input_mesh):
        raise FileNotFoundError(f"Input mesh not found: {input_mesh}")

    final_mesh = None
    try:
        ok, mesh = repair_with_trimesh(input_mesh)
        final_mesh = mesh
        if ok:
            print("\n[SUCCESS] Valid volume produced by trimesh direct repair.")
        else:
            print("\n[NOT VALID] trimesh direct repair did not produce a fully valid volume.")
    except Exception as exc:
        print(f"\n[FAILED] trimesh direct repair: {exc}")
        traceback.print_exc()

    if final_mesh is None:
        raise RuntimeError("Mesh repair failed; no output mesh was generated.")

    final_mesh = force_consistent_positive_volume(clean_trimesh(final_mesh))
    os.makedirs(os.path.dirname(output_mesh), exist_ok=True)
    final_mesh.export(output_mesh)
    mesh_status(final_mesh, "FINAL OUTPUT")

    if require_valid_volume and not is_valid_volume(final_mesh):
        raise RuntimeError(
            "Output mesh is still not watertight / volume / winding consistent. "
            "Try increasing alpha, increasing max hole size, or inspect model.stl for self-intersections."
        )

    print("saved processed STL:", output_mesh)

    if visualize:
        mesh_o3d = o3d.io.read_triangle_mesh(output_mesh)
        if mesh_o3d.is_empty():
            raise RuntimeError(f"Failed to read processed mesh: {output_mesh}")
        mesh_o3d.compute_vertex_normals()
        o3d.visualization.draw_geometries(
            [mesh_o3d],
            window_name="Processed repair mesh",
            mesh_show_back_face=True,
        )

    return output_mesh


def parse_args():
    parser = argparse.ArgumentParser(description="Generate and postprocess depression repair mesh.")
    parser.add_argument("--completion-dir", type=str, default=DEFAULT_DEPRESSION_DIR, help="Directory containing model.pcd.")
    parser.add_argument("--pcd", type=str, default=None, help="Input repair point cloud path. Defaults to completion-dir/model.pcd.")
    parser.add_argument("--raw-stl", type=str, default=None, help="Output raw STL path. Defaults to completion-dir/model.stl.")
    parser.add_argument("--processed-stl", type=str, default=None, help="Output processed STL path. Defaults to completion-dir/model_processed.stl.")
    parser.add_argument("--alpha", type=float, default=0.005, help="Alpha shape reconstruction parameter.")
    parser.add_argument("--normal-radius", type=float, default=0.002, help="Normal estimation search radius.")
    parser.add_argument("--normal-max-nn", type=int, default=30, help="Normal estimation max neighbors.")
    parser.add_argument("--smooth-iterations", type=int, default=3, help="Taubin smoothing iterations.")
    parser.add_argument("--max-hole-size", type=int, default=200, help="PyMeshLab close-holes max hole size.")
    parser.add_argument("--allow-invalid-volume", action="store_true", help="Do not fail if processed STL is not a valid closed volume.")
    parser.add_argument("--visualize", action="store_true", help="Visualize raw and processed meshes.")
    return parser.parse_args()


def main():
    args = parse_args()
    completion_dir = args.completion_dir
    pcd_path = args.pcd or os.path.join(completion_dir, "model.pcd")
    raw_stl = args.raw_stl or os.path.join(completion_dir, "model.stl")
    processed_stl = args.processed_stl or os.path.join(completion_dir, "model_processed.stl")

    point_cloud_to_stl(
        pcd_path=pcd_path,
        stl_path=raw_stl,
        alpha=args.alpha,
        normal_radius=args.normal_radius,
        normal_max_nn=args.normal_max_nn,
        smooth_iterations=args.smooth_iterations,
        max_hole_size=args.max_hole_size,
        visualize=args.visualize,
    )
    postprocess_stl(
        input_mesh=raw_stl,
        output_mesh=processed_stl,
        require_valid_volume=not args.allow_invalid_volume,
        visualize=args.visualize,
    )


if __name__ == "__main__":
    main()

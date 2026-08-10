"""Visualize base_T_full and repair geometry against the fix surface."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

sys.path.append('E:/HKUSTGZ/AAM')

DEFAULT_RUN_DIR = Path(r"E:\HKUSTGZ\AAM\data\runs\20260806_143628")
PLANE_DISTANCE_M = 0.003


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument(
        "--plane-distance-m", type=float, default=PLANE_DISTANCE_M
    )
    return parser.parse_args()


def load_with_global_normal(path, color, plane_distance, delta_xy):
    if not path.is_file():
        raise FileNotFoundError(path)

    cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points)
    points[:, :2] -= delta_xy
    if len(points) < 3:
        raise RuntimeError(f"Point cloud needs at least 3 points: {path}")

    plane, inliers = cloud.segment_plane(
        distance_threshold=plane_distance,
        ransac_n=3,
        num_iterations=2000,
    )
    normal = np.asarray(plane[:3], dtype=float)
    normal /= np.linalg.norm(normal)
    if normal[0] > 0.0:
        normal = -normal
    center = points.mean(axis=0)
    cloud.paint_uniform_color(color)

    print(f"{path.name}: {len(points)} points")
    print("global normal:", normal)
    print("plane inliers:", len(inliers), "/", len(points))
    print("bbox min [m]:", points.min(axis=0))
    print("bbox max [m]:", points.max(axis=0))
    return cloud, center, normal


def normal_line(center, normal, length, color):
    return o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(
            np.vstack([center, center + length * normal])
        ),
        lines=o3d.utility.Vector2iVector([[0, 1]]),
    ).paint_uniform_color(color)


def print_outermost_curve_points(path, plane_distance, delta_xy):
    cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points).copy()
    points[:, :2] -= delta_xy

    plane, _ = cloud.segment_plane(
        distance_threshold=plane_distance,
        ransac_n=3,
        num_iterations=2000,
    )
    normal = np.asarray(plane[:3], dtype=float)
    normal /= np.linalg.norm(normal)
    if normal[0] > 0.0:
        normal = -normal

    order = np.argsort(points @ normal)[::-1]
    outermost_three = points[order[:3]]
    outermost_ten_mean = points[order[:10]].mean(axis=0)

    print("fix_points_curve global outward normal:", normal)
    print("Outermost 3 points in base_after_mark1_motion [m]:")
    for index, point in enumerate(outermost_three, start=1):
        print(f"  {index}:", point)
    print(
        "Mean of outermost 10 points in base_after_mark1_motion [m]:",
        outermost_ten_mean,
    )


def visualize(items, repair_cloud, base_T_full, achieved_origin):
    clouds = [item[0] for item in items]
    all_points = np.vstack([np.asarray(cloud.points) for cloud in clouds])
    diagonal = np.linalg.norm(all_points.max(axis=0) - all_points.min(axis=0))
    frame_size = max(0.02, 0.25 * diagonal)
    normal_length = max(0.03, 0.35 * diagonal)
    base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=frame_size
    )
    object_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=frame_size
    ).transform(base_T_full)
    point_radius = max(0.004, 0.04 * diagonal)
    object_origin = o3d.geometry.TriangleMesh.create_sphere(
        radius=point_radius * 2.0 / 3.0
    )
    object_origin.translate(base_T_full[:3, 3])
    object_origin.paint_uniform_color([1.0, 0.0, 1.0])
    achieved_point = o3d.geometry.TriangleMesh.create_sphere(
        radius=point_radius * 0.5
    )
    achieved_point.translate(achieved_origin)
    achieved_point.paint_uniform_color([1.0, 1.0, 0.0])

    viewer = o3d.visualization.Visualizer()
    viewer.create_window(
        window_name=(
            "fine_fuse (gray) | predicted repair (green) | "
            "target origin (magenta) | achieved origin (yellow)"
        ),
        width=1280,
        height=800,
    )
    for cloud, center, normal, normal_color in items:
        viewer.add_geometry(cloud)
        viewer.add_geometry(
            normal_line(center, normal, normal_length, normal_color)
        )
    viewer.add_geometry(repair_cloud)
    #viewer.add_geometry(base_frame)
    #viewer.add_geometry(object_frame)
    viewer.add_geometry(object_origin)
    viewer.add_geometry(achieved_point)

    render = viewer.get_render_option()
    render.background_color = np.array([0.03, 0.03, 0.03])
    render.point_size = 2
    viewer.run()
    viewer.destroy_window()


def endpose_to_transform_mm(endpose):
    endpose = np.asarray(endpose, dtype=float).reshape(6)
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler(
        "xyz", endpose[3:], degrees=True
    ).as_matrix()
    transform[:3, 3] = endpose[:3]
    return transform


def achieved_full_origin(depression_dir, pickplace_dir):
    try:
        from pick_and_place import depression_endpose as dep
    except ModuleNotFoundError:
        import depression_endpose as dep

    result = json.loads(
        (pickplace_dir / "pick_place_endpose.json").read_text(
            encoding="utf-8"
        )
    )
    orientation_meta = np.load(
        depression_dir / "orientation_meta.npz", allow_pickle=True
    )
    full_height_mm = (
        float(orientation_meta["full_box_z_height"]) * dep.UNIT_SCALE
    )

    theta = np.deg2rad(dep.PRINT_YAW_DEG)
    base_T_printer = np.eye(4)
    base_T_printer[:3, :3] = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta), np.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ])
    base_T_printer[:3, 3] = dep.PRINTER_CENTER_BASE_MM

    printer_T_full = np.eye(4)
    printer_T_full[2, 3] = full_height_mm / 2.0

    base_T_end_grab = endpose_to_transform_mm(result["grab_endpose"])
    base_T_end_fix = endpose_to_transform_mm(result["fix_endpose"])
    end_grab_T_full = (
        np.linalg.inv(base_T_end_grab)
        @ base_T_printer
        @ printer_T_full
    )
    base_T_full_achieved = base_T_end_fix @ end_grab_T_full
    return base_T_full_achieved[:3, 3] / 1000.0


def main():
    args = parse_args()
    depression_dir = (
        args.run_dir.expanduser().resolve() / "completion" / "depression"
    )
    pickplace_dir = args.run_dir.expanduser().resolve() / "pickplace"
    orientation_meta = np.load(
        depression_dir / "orientation_meta.npz", allow_pickle=True
    )
    base_T_full = np.asarray(
        orientation_meta["base_T_full"], dtype=float
    ).copy()

    motion = json.loads(
        (pickplace_dir / "mark1_motion.json").read_text(encoding="utf-8")
    )
    delta_xy = (
        np.asarray(motion["final_delta_base_m"], dtype=float)
        if motion["move"] else np.zeros(2)
    )
    base_T_full[:2, 3] -= delta_xy
    achieved_origin = achieved_full_origin(depression_dir, pickplace_dir)

    print("Target Full origin from orientation_meta [m]:", base_T_full[:3, 3])
    print("Achieved Full origin from fix endpose [m]:", achieved_origin)
    print(
        "Achieved - target Full origin [mm]:",
        (achieved_origin - base_T_full[:3, 3]) * 1000.0,
    )

    print_outermost_curve_points(
        depression_dir / "fix_points_curve.pcd",
        args.plane_distance_m,
        delta_xy,
    )

    fine_fuse = load_with_global_normal(
        args.run_dir.expanduser().resolve()
        / "construction"
        / "fine_scan"
        / "fine_fuse.pcd",
        [0.65, 0.65, 0.65],
        args.plane_distance_m,
        delta_xy,
    )

    # fix_points = load_with_global_normal(
    #     depression_dir / "fix_points.pcd",
    #     [1.0, 0.45, 0.0],
    #     args.plane_distance_m,
    #     delta_xy,
    # )
    # fix_curve = load_with_global_normal(
    #     depression_dir / "fix_points_curve.pcd",
    #     [0.0, 0.85, 1.0],
    #     args.plane_distance_m,
    #     delta_xy,
    # )

    repair_cloud = o3d.io.read_point_cloud(
        str(depression_dir / "model_oriented.pcd")
    )
    repair_cloud.transform(base_T_full)
    repair_cloud.paint_uniform_color([0.1, 0.9, 0.2])

    print("Mark1 delta XY [m]:", delta_xy)
    print("Corrected base_T_full:\n", base_T_full)
    print("base_T_full origin [m]:", base_T_full[:3, 3])

    visualize(
        [
            (*fine_fuse, [1.0, 1.0, 0.0]),
            # (*fix_points, [1.0, 1.0, 0.0]),
            # (*fix_curve, [1.0, 0.0, 1.0]),
        ],
        repair_cloud,
        base_T_full,
        achieved_origin,
    )


if __name__ == "__main__":
    main()

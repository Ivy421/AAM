"""Fit a fixed-size square plate to SAM3-derived point clouds.

Pipeline:
    SAM3 plate cloud -> RANSAC plane -> robust 2-D square fit -> plate centre

The input clouds are produced by ``pick_and_place/plate_calibration.py`` and
must be in the robot base frame.  The fit deliberately ignores the outermost
percentiles, so small tabs/extensions, noisy edge points and missing pixels do
not directly become plate boundaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


o3d: Any = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / "data/plate_calibration"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "plate_fit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument(
        "--clouds", nargs="+",
        default=["plate4_plate_base.pcd", "plate5_plate_base.pcd"],
        help="Base-frame PCD filenames, relative to --input-dir unless absolute.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--plate-size-mm", type=float, default=180.0)
    parser.add_argument("--plane-threshold-mm", type=float, default=3.0)
    parser.add_argument("--ransac-n", type=int, default=3)
    parser.add_argument("--ransac-iterations", type=int, default=3000)
    parser.add_argument("--voxel-size-mm", type=float, default=1.0)
    parser.add_argument(
        "--trim-percent", type=float, default=2.0,
        help="Ignore this percentage at each projected edge during robust fitting.",
    )
    parser.add_argument("--angle-step-deg", type=float, default=0.25)
    parser.add_argument("--validation-height-mm", type=float, default=50.0)
    parser.add_argument("--visualize", action="store_true")
    return parser.parse_args()


def resolve_cloud(input_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (input_dir / path).resolve()


def load_and_merge(paths: list[Path], voxel_size: float) -> o3d.geometry.PointCloud:
    merged = o3d.geometry.PointCloud()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        cloud = o3d.io.read_point_cloud(str(path))
        if len(cloud.points) == 0:
            raise RuntimeError(f"Empty point cloud: {path}")
        print(f"Loaded {path.name}: {len(cloud.points)} points")
        merged += cloud
    if voxel_size > 0:
        merged = merged.voxel_down_sample(voxel_size)
    if len(merged.points) < 100:
        raise RuntimeError("Too few merged points for plate fitting.")
    return merged


def fit_plane(
    cloud: o3d.geometry.PointCloud,
    threshold: float,
    ransac_n: int,
    iterations: int,
) -> tuple[o3d.geometry.PointCloud, np.ndarray, np.ndarray, float]:
    model, indices = cloud.segment_plane(
        distance_threshold=threshold,
        ransac_n=ransac_n,
        num_iterations=iterations,
    )
    plane_cloud = cloud.select_by_index(indices)
    if len(plane_cloud.points) < 100:
        raise RuntimeError("RANSAC found too few plate-plane inliers.")
    normal = np.asarray(model[:3], dtype=float)
    normal /= np.linalg.norm(normal)
    offset = float(model[3]) / np.linalg.norm(np.asarray(model[:3], dtype=float))
    # The validation point must be above the plate in the robot base frame.
    if np.dot(normal, np.array([0.0, 0.0, 1.0])) < 0:
        normal, offset = -normal, -offset
    points = np.asarray(plane_cloud.points)
    origin = points.mean(axis=0)
    origin -= (np.dot(normal, origin) + offset) * normal
    return plane_cloud, origin, normal, offset


def plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Start with the base axis least parallel to the plane normal.
    helper = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(helper, normal)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    u = helper - np.dot(helper, normal) * normal
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    v /= np.linalg.norm(v)
    return u, v


def robust_square_fit(
    points_3d: np.ndarray,
    plane_origin: np.ndarray,
    plane_u: np.ndarray,
    plane_v: np.ndarray,
    plate_size: float,
    trim_percent: float,
    angle_step_deg: float,
) -> dict[str, np.ndarray | float]:
    if not 0.0 <= trim_percent < 25.0:
        raise ValueError("--trim-percent must be in [0, 25).")
    if angle_step_deg <= 0 or angle_step_deg >= 45:
        raise ValueError("--angle-step-deg must be in (0, 45).")

    relative = points_3d - plane_origin
    xy = np.column_stack((relative @ plane_u, relative @ plane_v))
    lo, hi = trim_percent, 100.0 - trim_percent
    expected_span = plate_size * (hi - lo) / 100.0
    best: tuple[float, float, np.ndarray, np.ndarray] | None = None

    # A square repeats every 90 degrees. Robust percentile widths suppress
    # isolated edge extensions and missing/noisy extreme points.
    for angle_deg in np.arange(0.0, 90.0, angle_step_deg):
        angle = np.deg2rad(angle_deg)
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)],
             [np.sin(angle), np.cos(angle)]],
            dtype=float,
        )
        local = xy @ rotation
        bounds_lo = np.percentile(local, lo, axis=0)
        bounds_hi = np.percentile(local, hi, axis=0)
        spans = bounds_hi - bounds_lo
        # Prefer an orientation whose two robust spans agree with the known
        # 180 mm square. A small area term resolves nearly equal candidates.
        size_error = np.sum((spans - expected_span) ** 2)
        square_error = (spans[0] - spans[1]) ** 2
        score = size_error + 0.5 * square_error + 0.01 * np.prod(spans)
        if best is None or score < best[0]:
            best = (float(score), angle, bounds_lo, bounds_hi)

    assert best is not None
    score, angle, bounds_lo, bounds_hi = best
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)],
         [np.sin(angle), np.cos(angle)]],
        dtype=float,
    )
    center_local = (bounds_lo + bounds_hi) / 2.0
    center_xy = center_local @ rotation.T
    x_axis = plane_u * rotation[0, 0] + plane_v * rotation[1, 0]
    y_axis = plane_u * rotation[0, 1] + plane_v * rotation[1, 1]
    x_axis /= np.linalg.norm(x_axis)
    y_axis /= np.linalg.norm(y_axis)

    # Resolve the square's 90/180-degree ambiguity deterministically: choose
    # the in-plane fitted axis most aligned with base +X, then make it positive.
    if abs(np.dot(y_axis, [1.0, 0.0, 0.0])) > abs(np.dot(x_axis, [1.0, 0.0, 0.0])):
        x_axis, y_axis = y_axis, -x_axis
    if np.dot(x_axis, [1.0, 0.0, 0.0]) < 0:
        x_axis, y_axis = -x_axis, -y_axis
    center_3d = plane_origin + center_xy[0] * plane_u + center_xy[1] * plane_v
    measured_spans = bounds_hi - bounds_lo
    return {
        "center": center_3d,
        "x_axis": x_axis,
        "y_axis": y_axis,
        "measured_robust_spans": measured_spans,
        "angle_deg": float(np.rad2deg(angle)),
        "fit_score": score,
    }


def make_fit_geometry(
    center: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    normal: np.ndarray,
    plate_size: float,
    validation_height: float,
) -> tuple[o3d.geometry.LineSet, o3d.geometry.PointCloud]:
    half = plate_size / 2.0
    corners = np.array(
        [center - half * x_axis - half * y_axis,
         center + half * x_axis - half * y_axis,
         center + half * x_axis + half * y_axis,
         center - half * x_axis + half * y_axis],
    )
    validation = center + validation_height * normal
    points = np.vstack((corners, center, validation))
    lines = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(
            [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5]]
        ),
    )
    lines.colors = o3d.utility.Vector3dVector(
        [[1.0, 0.0, 0.0]] * 4 + [[0.0, 0.0, 1.0]]
    )
    key_points = o3d.geometry.PointCloud()
    key_points.points = o3d.utility.Vector3dVector(points)
    key_points.colors = o3d.utility.Vector3dVector(
        [[1.0, 0.0, 0.0]] * 4 + [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    return lines, key_points


def main() -> None:
    global o3d
    args = parse_args()
    try:
        import open3d as open3d_module
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Open3D is required to read and fit the PCD files. Run this script "
            "in the project's point-cloud environment."
        ) from exc
    o3d = open3d_module
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cloud_paths = [resolve_cloud(input_dir, value) for value in args.clouds]
    plate_size = args.plate_size_mm / 1000.0
    voxel_size = args.voxel_size_mm / 1000.0
    threshold = args.plane_threshold_mm / 1000.0
    validation_height = args.validation_height_mm / 1000.0

    merged = load_and_merge(cloud_paths, voxel_size)
    plane_cloud, plane_origin, normal, offset = fit_plane(
        merged, threshold, args.ransac_n, args.ransac_iterations
    )
    plane_cloud.paint_uniform_color([0.1, 0.75, 0.2])
    fit = robust_square_fit(
        np.asarray(plane_cloud.points), plane_origin, *plane_basis(normal),
        plate_size, args.trim_percent, args.angle_step_deg,
    )
    center = np.asarray(fit["center"])
    x_axis = np.asarray(fit["x_axis"])
    # Recompute Y so the saved frame is exactly right-handed.
    y_axis = np.cross(normal, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    arm_base_t_plate = np.eye(4)
    arm_base_t_plate[:3, :3] = np.column_stack((x_axis, y_axis, normal))
    arm_base_t_plate[:3, 3] = center
    validation_point = center + validation_height * normal

    np.save(output_dir / "arm_base_T_plate.npy", arm_base_t_plate)
    np.save(output_dir / "arm_base_P_plate_center.npy", center)
    np.save(output_dir / "arm_base_P_plate_validation.npy", validation_point)
    o3d.io.write_point_cloud(str(output_dir / "plate_plane_inliers.pcd"), plane_cloud)
    lines, key_points = make_fit_geometry(
        center, x_axis, y_axis, normal, plate_size, validation_height
    )
    o3d.io.write_line_set(str(output_dir / "plate_fitted_square.ply"), lines)
    o3d.io.write_point_cloud(str(output_dir / "plate_fit_keypoints.pcd"), key_points)

    spans = np.asarray(fit["measured_robust_spans"])
    result = {
        "coordinate_frame": "arm_base",
        "source_clouds": [str(path) for path in cloud_paths],
        "plate_size_m": plate_size,
        "plane_threshold_m": threshold,
        "plane_inliers": len(plane_cloud.points),
        "plane_normal": normal.tolist(),
        "plane_equation": [*normal.tolist(), offset],
        "robust_trim_percent_each_side": args.trim_percent,
        "measured_robust_spans_m": spans.tolist(),
        "search_angle_deg": fit["angle_deg"],
        "fit_score": fit["fit_score"],
        "arm_base_T_plate": arm_base_t_plate.tolist(),
        "arm_base_P_plate_center": center.tolist(),
        "validation_height_m": validation_height,
        "arm_base_P_plate_validation": validation_point.tolist(),
        "axis_ambiguity_note": (
            "Square geometry cannot identify the physical +X/+Y corner. "
            "X is selected deterministically toward arm-base +X."
        ),
    }
    with (output_dir / "plate_fit_result.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    print(f"RANSAC plane inliers: {len(plane_cloud.points)} / {len(merged.points)}")
    print(f"Robust measured spans: {spans * 1000.0} mm")
    print(f"Plate center in arm base: {center} m")
    print(f"Validation point (+{args.validation_height_mm:g} mm): {validation_point} m")
    print(f"Results written to {output_dir}")
    if args.visualize:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.05, origin=center
        )
        frame.rotate(arm_base_t_plate[:3, :3], center=center)
        o3d.visualization.draw_geometries(
            [plane_cloud, lines, key_points, frame],
            window_name="Plate plane and robust 180 mm square fit",
        )


if __name__ == "__main__":
    main()

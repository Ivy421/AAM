"""Generate two continuous sponge-brush glue trajectories from fix_points_curve.pcd."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.interpolate import splprep, splev


PATH_FRACTIONS = (0.25, 0.75)
BAND_HALF_WIDTH_M = 0.003
PCA_BIN_SIZE_M = 0.0006
TRAJECTORY_SPACING_M = 0.002
MIN_BIN_POINTS = 3
BSPLINE_RMS_SMOOTH_M = 0.0005
DENSE_SPLINE_SAMPLES = 4000


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fix-points", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--meta",
        type=Path,
        default=None,
        help="completion meta.npz; default: sibling meta.npz of fix_points_curve.pcd",
    )
    return parser.parse_args()


def normalize(vector):
    vector = np.asarray(vector, dtype=float)
    return vector / (np.linalg.norm(vector) + 1e-12)


def load_completion_frame(fix_points_path, meta_path):
    meta_path = meta_path or (fix_points_path.parent / "meta.npz")
    meta = np.load(meta_path, allow_pickle=True)
    return {
        "origin": np.asarray(meta["top_plane_center"], dtype=float),
        "u_axis": normalize(meta["u_axis"]),
        "v_axis": normalize(meta["v_axis"]),
        "n_axis": normalize(meta["n_axis"]),
        "meta_path": meta_path,
    }


def project_to_uvz(points, frame):
    vectors = points - frame["origin"]
    return np.column_stack(
        [
            vectors @ frame["u_axis"],
            vectors @ frame["v_axis"],
            vectors @ frame["n_axis"],
        ]
    )


def pca_curve_direction(uv):
    center = uv.mean(axis=0)
    centered = uv - center
    covariance = centered.T @ centered / max(len(uv), 1)
    _, eigenvectors = np.linalg.eigh(covariance)
    direction = normalize(eigenvectors[:, -1])

    # Keep a deterministic direction for trajectory ordering.
    if abs(direction[0]) >= abs(direction[1]):
        if direction[0] < 0.0:
            direction = -direction
    elif direction[1] < 0.0:
        direction = -direction

    return center, direction


def build_binned_centers(points, uv, bin_size):
    uv_center, direction = pca_curve_direction(uv)
    s = (uv - uv_center) @ direction

    s_min = float(s.min())
    bin_ids = np.floor((s - s_min) / bin_size).astype(int)

    centers = []
    center_s = []
    for bin_id in np.unique(bin_ids):
        ids = np.where(bin_ids == bin_id)[0]
        if len(ids) < MIN_BIN_POINTS:
            continue
        centers.append(np.median(points[ids], axis=0))
        center_s.append(float(np.median(s[ids])))

    order = np.argsort(center_s)
    centers = np.asarray(centers, dtype=float)[order]
    center_s = np.asarray(center_s, dtype=float)[order]
    return centers, center_s, direction


def fit_bspline(points):
    k = min(3, len(points) - 1)
    smooth = len(points) * BSPLINE_RMS_SMOOTH_M ** 2
    tck, _ = splprep(points.T, s=smooth, k=k)
    return tck, k, smooth


def sample_bspline_by_arc_length(tck, spacing):
    u_dense = np.linspace(0.0, 1.0, DENSE_SPLINE_SAMPLES)
    dense_points = np.column_stack(splev(u_dense, tck))

    step = np.linalg.norm(np.diff(dense_points, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(step)])
    total_length = float(arc[-1])

    sample_arc = np.arange(0.0, total_length + 1e-12, spacing)
    sample_u = np.interp(sample_arc, arc, u_dense)
    sampled_points = np.column_stack(splev(sample_u, tck))
    return sampled_points, sample_arc, total_length


def build_trajectory(points, uvz, target_z, reverse=False):
    band_mask = np.abs(uvz[:, 2] - target_z) <= BAND_HALF_WIDTH_M
    band_points = points[band_mask]
    band_uv = uvz[band_mask, :2]

    binned_centers, center_s, pca_direction = build_binned_centers(
        band_points,
        band_uv,
        PCA_BIN_SIZE_M,
    )

    tck, spline_order, smooth = fit_bspline(binned_centers)
    trajectory_points, arc, total_length = sample_bspline_by_arc_length(
        tck,
        TRAJECTORY_SPACING_M,
    )

    if reverse:
        trajectory_points = trajectory_points[::-1]
        arc = total_length - arc[::-1]

    return {
        "target_z_m": float(target_z),
        "band_point_count": int(len(band_points)),
        "pca_direction_uv": np.round(pca_direction, 9).tolist(),
        "bin_size_m": PCA_BIN_SIZE_M,
        "binned_center_count": int(len(binned_centers)),
        "binned_centers_base_m": np.round(binned_centers, 9).tolist(),
        "bspline_order": int(spline_order),
        "bspline_smoothing": float(smooth),
        "trajectory_spacing_m": TRAJECTORY_SPACING_M,
        "trajectory_length_m": round(total_length, 9),
        "trajectory_point_count": int(len(trajectory_points)),
        "trajectory_points_base_m": np.round(trajectory_points, 9).tolist(),
        "trajectory_points_base_mm": np.round(trajectory_points * 1000.0, 3).tolist(),
        "arc_length_m": np.round(arc, 9).tolist(),
    }


def save_outputs(out_dir, fix_points_path, frame, z_min, z_max, trajectories, runtime_seconds):
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "glue_brush_continue_trajectory.json"
    npz_path = out_dir / "glue_brush_continue_trajectory.npz"
    pcd_path = out_dir / "glue_brush_continue_trajectory.pcd"

    payload = {
        "coordinate_frame": "base",
        "position_unit": "m",
        "source_fix_points": str(fix_points_path),
        "source_meta": str(frame["meta_path"]),
        "runtime_seconds": round(runtime_seconds, 6),
        "parameters": {
            "path_fractions": list(PATH_FRACTIONS),
            "band_half_width_m": BAND_HALF_WIDTH_M,
            "pca_bin_size_m": PCA_BIN_SIZE_M,
            "trajectory_spacing_m": TRAJECTORY_SPACING_M,
            "min_bin_points": MIN_BIN_POINTS,
            "bspline_rms_smooth_m": BSPLINE_RMS_SMOOTH_M,
        },
        "completion_frame": {
            "origin_base_m": np.round(frame["origin"], 9).tolist(),
            "u_axis": np.round(frame["u_axis"], 9).tolist(),
            "v_axis": np.round(frame["v_axis"], 9).tolist(),
            "n_axis": np.round(frame["n_axis"], 9).tolist(),
        },
        "fix_depth_range_m": [float(z_min), float(z_max)],
        "fix_depth_m": float(z_max - z_min),
        "trajectory_count": len(trajectories),
        "trajectories": trajectories,
    }

    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    np.savez(
        npz_path,
        trajectory_0_base_m=np.asarray(
            trajectories[0]["trajectory_points_base_m"],
            dtype=float,
        ),
        trajectory_1_base_m=np.asarray(
            trajectories[1]["trajectory_points_base_m"],
            dtype=float,
        ),
        target_z_m=np.asarray(
            [item["target_z_m"] for item in trajectories],
            dtype=float,
        ),
        n_axis=frame["n_axis"],
        u_axis=frame["u_axis"],
        v_axis=frame["v_axis"],
        origin=frame["origin"],
    )

    all_points = np.vstack(
        [
            np.asarray(item["trajectory_points_base_m"], dtype=float)
            for item in trajectories
        ]
    )
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(all_points)
    o3d.io.write_point_cloud(str(pcd_path), cloud)

    return json_path, npz_path, pcd_path


def main():
    args = parse_args()
    start_time = time.perf_counter()

    source_cloud = o3d.io.read_point_cloud(str(args.fix_points))
    points = np.asarray(source_cloud.points, dtype=float)

    frame = load_completion_frame(args.fix_points, args.meta)
    uvz = project_to_uvz(points, frame)

    z_min = float(uvz[:, 2].min())
    z_max = float(uvz[:, 2].max())
    depth = z_max - z_min

    target_z = [z_min + fraction * depth for fraction in PATH_FRACTIONS]
    trajectories = [
        build_trajectory(points, uvz, target_z[0], reverse=False),
        build_trajectory(points, uvz, target_z[1], reverse=True),
    ]

    runtime_seconds = time.perf_counter() - start_time
    paths = save_outputs(
        args.out_dir,
        args.fix_points,
        frame,
        z_min,
        z_max,
        trajectories,
        runtime_seconds,
    )

    print(f"Fix depth: {depth:.6f} m")
    print(f"Trajectory 1 points: {trajectories[0]['trajectory_point_count']}")
    print(f"Trajectory 2 points: {trajectories[1]['trajectory_point_count']}")
    for path in paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()

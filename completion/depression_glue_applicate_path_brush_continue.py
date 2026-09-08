"""Fit and save two continuous sponge-brush B-spline paths."""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.interpolate import splprep


PATH_FRACTIONS = (0.25, 0.75)
BAND_HALF_WIDTH_M = 0.003
PCA_BIN_SIZE_M = 0.0006
MIN_BIN_POINTS = 3
BSPLINE_RMS_SMOOTH_M = 0.0005


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

    if abs(direction[0]) >= abs(direction[1]):
        if direction[0] < 0.0:
            direction = -direction
    elif direction[1] < 0.0:
        direction = -direction

    return center, direction


def build_binned_centers(points, uv):
    uv_center, direction = pca_curve_direction(uv)
    s = (uv - uv_center) @ direction
    bin_ids = np.floor(
        (s - float(s.min())) / PCA_BIN_SIZE_M
    ).astype(int)

    centers = []
    center_s = []

    for bin_id in np.unique(bin_ids):
        ids = np.where(bin_ids == bin_id)[0]
        if len(ids) < MIN_BIN_POINTS:
            continue

        centers.append(np.median(points[ids], axis=0))
        center_s.append(float(np.median(s[ids])))

    if len(centers) < 2:
        raise RuntimeError("Not enough binned centers to fit B-spline.")

    order = np.argsort(center_s)
    return np.asarray(centers, dtype=float)[order], direction


def fit_bspline(points):
    k = min(3, len(points) - 1)
    smooth = len(points) * BSPLINE_RMS_SMOOTH_M ** 2
    tck, _ = splprep(points.T, s=smooth, k=k)
    t, c, k = tck

    return {
        "t": np.asarray(t, dtype=float),
        "c": np.asarray(c, dtype=float),
        "k": int(k),
        "smoothing": float(smooth),
    }


def build_bspline(points, uvz, target_z):
    mask = np.abs(uvz[:, 2] - target_z) <= BAND_HALF_WIDTH_M
    band_points = points[mask]
    band_uv = uvz[mask, :2]

    if len(band_points) == 0:
        raise RuntimeError(
            f"No points found around target_z={target_z:.6f} m."
        )

    binned_centers, pca_direction = build_binned_centers(
        band_points,
        band_uv,
    )
    spline = fit_bspline(binned_centers)

    return {
        "target_z": float(target_z),
        "pca_direction": pca_direction,
        "binned_centers": binned_centers,
        **spline,
    }


def save_bspline(out_dir, frame, z_min, z_max, splines):
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "glue_brush_bspline.npz"

    np.savez(
        output_path,
        path0_t=splines[0]["t"],
        path0_c=splines[0]["c"],
        path0_k=splines[0]["k"],
        path0_target_z=splines[0]["target_z"],
        path0_pca_direction=splines[0]["pca_direction"],
        path0_binned_centers_base_m=splines[0]["binned_centers"],
        path1_t=splines[1]["t"],
        path1_c=splines[1]["c"],
        path1_k=splines[1]["k"],
        path1_target_z=splines[1]["target_z"],
        path1_pca_direction=splines[1]["pca_direction"],
        path1_binned_centers_base_m=splines[1]["binned_centers"],
        origin=frame["origin"],
        u_axis=frame["u_axis"],
        v_axis=frame["v_axis"],
        n_axis=frame["n_axis"],
        fix_depth_range_m=np.asarray([z_min, z_max], dtype=float),
        path_fractions=np.asarray(PATH_FRACTIONS, dtype=float),
        band_half_width_m=BAND_HALF_WIDTH_M,
        pca_bin_size_m=PCA_BIN_SIZE_M,
        bspline_rms_smooth_m=BSPLINE_RMS_SMOOTH_M,
    )

    return output_path


def main():
    args = parse_args()

    source_cloud = o3d.io.read_point_cloud(str(args.fix_points))
    points = np.asarray(source_cloud.points, dtype=float)

    frame = load_completion_frame(args.fix_points, args.meta)
    uvz = project_to_uvz(points, frame)

    z_min = float(uvz[:, 2].min())
    z_max = float(uvz[:, 2].max())
    depth = z_max - z_min

    target_z = [
        z_min + fraction * depth
        for fraction in PATH_FRACTIONS
    ]

    splines = [
        build_bspline(points, uvz, target_z[0]),
        build_bspline(points, uvz, target_z[1]),
    ]

    output_path = save_bspline(
        args.out_dir,
        frame,
        z_min,
        z_max,
        splines,
    )

    print(f"Fix depth: {depth:.6f} m")
    print(
        "B-spline centers: "
        f"{len(splines[0]['binned_centers'])}, "
        f"{len(splines[1]['binned_centers'])}"
    )
    print(f"Saved: {output_path.resolve()}")


if __name__ == "__main__":
    main()

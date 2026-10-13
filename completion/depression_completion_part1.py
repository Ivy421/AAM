import os
import json
import argparse
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import open3d as o3d
from scipy import ndimage
from scipy.signal import savgol_filter

o3d.utility.random.seed(274)


class CompletionConfig:
    # I/O paths, same style as previous scripts.
    fine_pcd_path: str = r"E:/HKUSTGZ/AAM/data/temp/fine_scan/fine_fuse.pcd"
    corner_json_path: str = r"E:/HKUSTGZ/AAM/construction/data/coarse_scan/corner_mapping_result.json"
    output_dir: str = r"E:/HKUSTGZ/AAM/data/temp/completion_result"

    # Set None to read from corner_json_path. Or manually set:
    # "max_u_max_v" / "max_u_min_v".
    corner_mode: Optional[str] = "max_u_max_v"

    # Plane extraction.
    plane_voxel_size: float = 0.0008
    plane_distance_threshold: float = 0.002
    plane_num_iterations: int = 3000

    # UV grid.
    grid_res: float = 0.0008
    grid_pad: float = 0.000

    # Boundary contour extraction.
    edge_bin_size: float = 0.001
    boundary_min_bin_points: int = 5
    boundary_outer_quantile: float = 0.95
    boundary_keep_tail_fraction: float = 0.85
    boundary_roi_restriction: float = 0.15
    min_boundary_points: int = 20

    # Boundary support enhancement for sloped/slide-like edge points.
    # Top plane is still used for occupancy masks; these points are only used
    # to extract the outer boundary for ROI turn detection.
    use_rest_points_for_boundary: bool = True
    boundary_support_z_min: float = -0.003
    boundary_support_z_max: float = 0.010
    boundary_side_band_inward: float = 0.050
    boundary_side_band_outward: float = 0.015

    # Boundary denoise / smooth.
    rolling_window: int = 9
    rolling_jump_th: float = 0.010
    smooth_window: int = 31
    smooth_polyorder: int = 2

    # Turn detection by two-segment piecewise line fitting.
    baseline_fraction: float = 0.35
    turn_exclude_ratio: float = 0.05
    turn_score_smooth_window: int = 11
    min_turn_deviation: float = 0.0035 #0.0015
    turn_search_tail_min_ratio: float = 0.0
    turn_search_tail_max_ratio: float = 0.70
    twofit_min_segment_points: int = 12
    twofit_min_angle_deg: float = 8.0
    twofit_min_inward_offset: float = 0.002
    twofit_regularization: float = 0.2

    # Convert turn points to corner ROI.
    roi_margin_ratio: float = 1.10
    roi_min_frac: float = 0.03
    roi_max_frac: float = 0.80

    # Top occupancy / defect mask smoothing.
    top_occ_close_iter: int = 1
    top_occ_dilate_iter: int = 0
    defect_close_iter: int = 2
    min_component_area_pixels: int = 30

    # Flood fill / layered completion.
    layer_step: float = 0.001
    band_width: float = 0.0007
    thres_points_num: int = 20
    max_bad_layers: int = 5
    max_area_ratio: float = 0.70
    min_area_pixels: int = 20
    max_search_depth: float = 0.20
    wall_endpoint_extend_pixels: int = 8

    # Column depth post-filter.
    enable_column_depth_filter: bool = False
    min_column_layers: int = 3

    # Other.
    inward_vec: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    visualize: bool = True


@dataclass
class PlaneData:
    pcd_raw: o3d.geometry.PointCloud
    plane_model: np.ndarray
    plane_pcd: o3d.geometry.PointCloud
    rest_pcd: o3d.geometry.PointCloud
    origin: np.ndarray
    u_axis: np.ndarray
    v_axis: np.ndarray
    n_axis: np.ndarray
    top_points: np.ndarray
    points_raw: np.ndarray
    top_uv: np.ndarray
    z_bottom: float


@dataclass
class RoiData:
    u_side: str
    v_side: str
    boundary_support_uv: np.ndarray
    near_top_rest_uv: np.ndarray
    near_top_rest_points: np.ndarray
    u_side_support_uv: np.ndarray
    v_side_support_uv: np.ndarray
    u_line_raw: np.ndarray
    v_line_raw: np.ndarray
    u_line: np.ndarray
    v_line: np.ndarray
    turn_idx_u: int
    turn_idx_v: int
    turn_point_u: np.ndarray
    turn_point_v: np.ndarray
    score_u: np.ndarray
    score_v: np.ndarray
    deviation_u: np.ndarray
    deviation_v: np.ndarray
    roi_frac_u: float
    roi_frac_v: float


@dataclass
class GridData:
    u_min: float
    v_min: float
    u_max: float
    v_max: float
    u_grid: np.ndarray
    v_grid: np.ndarray
    H: int
    W: int
    top_occ_mask: np.ndarray
    top_occ_smooth: np.ndarray
    corner_roi: np.ndarray
    domain_mask: np.ndarray
    defect_mask: np.ndarray
    top_defect_margin_points: np.ndarray
    top_defect_margin_uv: np.ndarray
    top_plane_center: np.ndarray


@dataclass
class CompletionData:
    repair_volume_points: np.ndarray
    repair_point_center: np.ndarray
    repair_layers: List[dict]
    side_points: Dict[str, np.ndarray]
    side_n_mark: Dict[str, np.ndarray]
    defect_surface_points: np.ndarray
    fix_points: np.ndarray
    fix_points_curve: np.ndarray
    fix_npz_data: Dict[str, np.ndarray]
    debug_records: List[dict]


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    return v / (np.linalg.norm(v) + 1e-12)


def keep_largest_cluster(
    pcd: o3d.geometry.PointCloud,
    eps: float = 0.01,
    min_points: int = 40,
):
    labels = np.asarray(
        pcd.cluster_dbscan(
            eps=eps,
            min_points=min_points,
            print_progress=False,
        )
    )

    valid = labels >= 0
    if not np.any(valid):
        raise RuntimeError("No valid cluster found in pcd_raw.")

    cluster_ids, counts = np.unique(
        labels[valid],
        return_counts=True,
    )

    largest_id = cluster_ids[np.argmax(counts)]
    keep_idx = np.where(labels == largest_id)[0]

    pcd_largest = pcd.select_by_index(keep_idx.tolist())

    print(
        f"[Raw PCD DBSCAN] clusters={len(cluster_ids)}, "
        f"largest={largest_id}, "
        f"points={len(pcd_largest.points)} / {len(pcd.points)}"
    )

    return pcd_largest


def resolve_corner_mode(cfg: CompletionConfig) -> str:
    allowed = {"max_u_max_v", "max_u_min_v"}
    if cfg.corner_mode is not None:
        corner_mode = cfg.corner_mode
    else:
        with open(cfg.corner_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        corner_mode = data.get("corner_mode", "max_u_max_v")

    return corner_mode


def find_plane(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
    distance_threshold: float,
    num_iterations: int,
):
    #pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd_down=pcd
    plane_model, inliers = pcd_down.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=3,
        num_iterations=num_iterations,
    )
    plane_pcd = pcd_down.select_by_index(inliers)
    rest_pcd = pcd_down.select_by_index(inliers, invert=True)
    return np.asarray(plane_model, dtype=float), plane_pcd, rest_pcd


def build_plane_basis(points: np.ndarray, normal: np.ndarray):
    n_axis = normalize(normal)
    origin = points.mean(axis=0)

    pts = points - origin
    pts_plane = pts - np.outer(pts @ n_axis, n_axis)
    cov = np.cov(pts_plane.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]

    axis1 = normalize(eigvecs[:, order[0]] - np.dot(eigvecs[:, order[0]], n_axis) * n_axis)
    axis2 = normalize(eigvecs[:, order[1]] - np.dot(eigvecs[:, order[1]], n_axis) * n_axis)

    world_x = np.array([1.0, 0.0, 0.0])
    u_axis = axis1 if abs(np.dot(axis1, world_x)) >= abs(np.dot(axis2, world_x)) else axis2

    # Keep same convention as previous Depression_Completion.py.
    #要求 u_axis指向机械臂base的方向
    if np.dot(world_x, u_axis) > 0:
        u_axis = -u_axis

    v_axis = normalize(np.cross(n_axis, u_axis))
    return origin, u_axis, v_axis


def project_to_uv(points: np.ndarray, origin: np.ndarray, u_axis: np.ndarray, v_axis: np.ndarray):
    vec = points - origin
    u = vec @ u_axis
    v = vec @ v_axis
    return np.column_stack([u, v])


def project_to_uvz(
    points: np.ndarray,
    origin: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    z_axis: np.ndarray,
):
    vec = points - origin
    u = vec @ u_axis
    v = vec @ v_axis
    z = vec @ z_axis
    return np.column_stack([u, v, z])


def uv_to_3d(uv: np.ndarray, origin: np.ndarray, u_axis: np.ndarray, v_axis: np.ndarray):
    uv = np.asarray(uv, dtype=float)
    return origin + uv[:, 0:1] * u_axis + uv[:, 1:2] * v_axis


def uvz_to_3d(
    uv: np.ndarray,
    z: np.ndarray,
    origin: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    z_axis: np.ndarray,
):
    uv = np.asarray(uv, dtype=float)
    z = np.asarray(z, dtype=float).reshape(-1, 1)
    return origin + uv[:, 0:1] * u_axis + uv[:, 1:2] * v_axis + z * z_axis


def uv_to_grid_index(uv: np.ndarray, u_min: float, v_min: float, grid_res: float, W: int, H: int):
    xs = ((uv[:, 0] - u_min) / grid_res).astype(int)
    ys = ((uv[:, 1] - v_min) / grid_res).astype(int)
    valid = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    return xs, ys, valid


def mask_to_uv_points(mask: np.ndarray, u_min: float, v_min: float, grid_res: float):
    ys, xs = np.where(mask)
    u = u_min + (xs + 0.5) * grid_res
    v = v_min + (ys + 0.5) * grid_res
    return np.column_stack([u, v])


def make_colored_pcd(points: np.ndarray, color):
    pcd = o3d.geometry.PointCloud()
    points = np.asarray(points)
    if len(points) > 0:
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color(color)
    return pcd


def make_sphere(center: np.ndarray, radius: float, color):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(np.asarray(center, dtype=float).reshape(3))
    sphere.paint_uniform_color(color)
    return sphere


def get_target_sides_from_corner_mode(corner_mode: str):
    if corner_mode == "max_u_max_v":
        return "max_u", "max_v"
    if corner_mode == "max_u_min_v":
        return "max_u", "min_v"
    raise ValueError(f"Unsupported corner_mode: {corner_mode}")


def extract_binned_boundary_points(
    uv: np.ndarray,
    side: str,
    bin_size: float,
    min_bin_points: int,
    outer_quantile: float,
) -> np.ndarray:
    """
    Extract one outer boundary point per bin.

    Compared with direct max/min, using an outer quantile is more robust to isolated outliers.
    """
    pts = []
    uv = np.asarray(uv, dtype=float)

    if side in ["max_u", "min_u"]:
        coord = uv[:, 1]  # bin along v
        bins = np.arange(coord.min(), coord.max() + bin_size, bin_size)
        for i in range(len(bins) - 1):
            mask = (coord >= bins[i]) & (coord < bins[i + 1])
            bin_pts = uv[mask]
            if len(bin_pts) < min_bin_points:
                continue
            v_mid = np.median(bin_pts[:, 1])
            if side == "min_u":
                u_outer = np.quantile(bin_pts[:, 0], 1.0 - outer_quantile)
            else:
                u_outer = np.quantile(bin_pts[:, 0], outer_quantile)
            pts.append([u_outer, v_mid])

    elif side in ["min_v", "max_v"]:
        coord = uv[:, 0]  # bin along u
        bins = np.arange(coord.min(), coord.max() + bin_size, bin_size)
        for i in range(len(bins) - 1):
            mask = (coord >= bins[i]) & (coord < bins[i + 1])
            bin_pts = uv[mask]
            if len(bin_pts) < min_bin_points:
                continue
            u_mid = np.median(bin_pts[:, 0])
            if side == "min_v":
                v_outer = np.quantile(bin_pts[:, 1], 1.0 - outer_quantile)
            else:
                v_outer = np.quantile(bin_pts[:, 1], outer_quantile)
            pts.append([u_mid, v_outer])
    else:
        raise ValueError(f"Unknown side: {side}")

    return np.asarray(pts, dtype=float)


def filter_support_points_near_top_side(
    support_uv: np.ndarray,
    top_side_points: np.ndarray,
    side: str,
    inward_band: float,
    outward_band: float,
) -> np.ndarray:
    """
    Keep support points in a side-specific corridor around the top-only boundary.

    This prevents max_u and max_v/min_v extraction from competing for points in
    the same global support cloud near the damaged corner.
    """
    support_uv = np.asarray(support_uv, dtype=float)
    top_side_points = np.asarray(top_side_points, dtype=float)

    if len(support_uv) == 0 or len(top_side_points) < 2:
        return support_uv

    inward_band = max(float(inward_band), 0.0)
    outward_band = max(float(outward_band), 0.0)

    if side in ["max_u", "min_u"]:
        order = np.argsort(top_side_points[:, 1])
        ref_main = top_side_points[order, 1]
        ref_side = top_side_points[order, 0]
        valid_main = (support_uv[:, 1] >= ref_main.min()) & (support_uv[:, 1] <= ref_main.max())
        ref = np.interp(support_uv[:, 1], ref_main, ref_side)

        if side == "max_u":
            side_mask = (support_uv[:, 0] >= ref - inward_band) & (support_uv[:, 0] <= ref + outward_band)
        else:
            side_mask = (support_uv[:, 0] <= ref + inward_band) & (support_uv[:, 0] >= ref - outward_band)

    elif side in ["max_v", "min_v"]:
        order = np.argsort(top_side_points[:, 0])
        ref_main = top_side_points[order, 0]
        ref_side = top_side_points[order, 1]
        valid_main = (support_uv[:, 0] >= ref_main.min()) & (support_uv[:, 0] <= ref_main.max())
        ref = np.interp(support_uv[:, 0], ref_main, ref_side)

        if side == "max_v":
            side_mask = (support_uv[:, 1] >= ref - inward_band) & (support_uv[:, 1] <= ref + outward_band)
        else:
            side_mask = (support_uv[:, 1] <= ref + inward_band) & (support_uv[:, 1] >= ref - outward_band)

    else:
        raise ValueError(f"Unknown side: {side}")

    filtered = support_uv[valid_main & side_mask]
    return filtered if len(filtered) >= 2 else support_uv


def extract_side_specific_boundary_points(
    top_uv: np.ndarray,
    support_uv: np.ndarray,
    side: str,
    cfg: CompletionConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract one boundary side from a side-specific support corridor.

    Returns:
        candidates: binned side boundary points used downstream.
        side_support_uv: support points kept in the side corridor.
        top_side_candidates: top-only reference boundary used to build corridor.
    """
    top_side_candidates = extract_binned_boundary_points(
        top_uv,
        side=side,
        bin_size=cfg.edge_bin_size,
        min_bin_points=cfg.boundary_min_bin_points,
        outer_quantile=cfg.boundary_outer_quantile,
    )

    side_support_uv = filter_support_points_near_top_side(
        support_uv=support_uv,
        top_side_points=top_side_candidates,
        side=side,
        inward_band=cfg.boundary_side_band_inward,
        outward_band=cfg.boundary_side_band_outward,
    )

    candidates = extract_binned_boundary_points(
        side_support_uv,
        side=side,
        bin_size=cfg.edge_bin_size,
        min_bin_points=cfg.boundary_min_bin_points,
        outer_quantile=cfg.boundary_outer_quantile,
    )

    if len(candidates) < cfg.min_boundary_points and len(top_side_candidates) > len(candidates):
        candidates = top_side_candidates

    return candidates, side_support_uv, top_side_candidates


def sort_boundary_from_far_to_defect(points: np.ndarray, side: str, corner_mode: str) -> np.ndarray:
    """
    Sort boundary line so that the line starts from the stable/far side
    and ends at the damaged corner.
    """
    if len(points) == 0:
        return points

    if side == "max_u":
        # max_u boundary extends along v.
        if corner_mode == "max_u_max_v":
            order = np.argsort(points[:, 1])      # min_v -> max_v
        elif corner_mode == "max_u_min_v":
            order = np.argsort(points[:, 1])[::-1]  # max_v -> min_v
        else:
            raise ValueError(f"Unsupported corner_mode: {corner_mode}")
    elif side in ["max_v", "min_v"]:
        # max_v / min_v boundary extends along u. Defect is always at max_u.
        order = np.argsort(points[:, 0])           # min_u -> max_u
    else:
        raise ValueError(f"Unsupported side: {side}")

    return points[order]


def select_near_corner_segment(points: np.ndarray, side: str, cfg: CompletionConfig) -> np.ndarray:
    """
    Keep only a near-corner boundary segment, controlled by physical length.
    The input points must be sorted from far side to damaged corner.
    """
    if len(points) <= 2:
        return points

    if side == "max_u":
        physical_len = float(points[:, 1].max() - points[:, 1].min())
    elif side in ["max_v", "min_v"]:
        physical_len = float(points[:, 0].max() - points[:, 0].min())
    else:
        raise ValueError(f"Unsupported side: {side}")

    if physical_len <= 1e-12:
        return points

    frac = min(cfg.boundary_roi_restriction / physical_len, 1.0)
    n_keep = int(np.ceil(len(points) * frac))
    n_keep = int(np.clip(n_keep, cfg.min_boundary_points, len(points)))
    return points[-n_keep:]


def keep_boundary_tail(points: np.ndarray, keep_fraction: float) -> np.ndarray:
    """
    Keep the tail portion of a sorted boundary line.

    The boundary line is sorted from the far/stable side to the damaged corner,
    so this removes the far-side trailing part before ROI turn detection.
    """
    if len(points) <= 2:
        return points

    keep_fraction = float(np.clip(keep_fraction, 0.0, 1.0))
    n_keep = int(np.ceil(len(points) * keep_fraction))
    n_keep = int(np.clip(n_keep, 1, len(points)))
    return points[-n_keep:]


def robust_filter_boundary_line(points: np.ndarray, side: str, window: int, jump_th: float) -> np.ndarray:
    """
    Lightweight rolling-median filter.
    The line must already be sorted from far side to damaged corner.
    """
    if len(points) <= window:
        return points

    check_dim = 0 if side == "max_u" else 1
    kept = []
    for p in points:
        if len(kept) < window:
            kept.append(p)
            continue
        ref = np.median(np.asarray(kept[-window:])[:, check_dim])
        if abs(p[check_dim] - ref) <= jump_th:
            kept.append(p)

    filtered = np.asarray(kept, dtype=float)
    return filtered if len(filtered) >= 2 else points


def smooth_boundary_line(points: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    n = len(points)
    if n < 5:
        return points

    window = min(window, n if n % 2 == 1 else n - 1)
    if window <= polyorder:
        window = polyorder + 3
        if window % 2 == 0:
            window += 1
    window = min(window, n if n % 2 == 1 else n - 1)
    if window < 5:
        return points

    u_smooth = savgol_filter(points[:, 0], window, polyorder)
    v_smooth = savgol_filter(points[:, 1], window, polyorder)
    return np.column_stack([u_smooth, v_smooth])


def _arc_length(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.empty(0)
    d = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])


def _safe_savgol_1d(values: np.ndarray, window: int, polyorder: int = 2) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 5:
        return values
    window = min(window, n if n % 2 == 1 else n - 1)
    if window <= polyorder:
        window = polyorder + 3
        if window % 2 == 0:
            window += 1
    window = min(window, n if n % 2 == 1 else n - 1)
    if window < 5:
        return values
    return savgol_filter(values, window, polyorder)


def find_turn_by_piecewise_linear_fit(
    points: np.ndarray,
    side: str,
    cfg: CompletionConfig,
):
    """
    Find the boundary turn point by trying every two-line split.

    The line is sorted from far/stable side to damaged corner.
    For each split:
        - fit the stable boundary line before the split
        - fit the damaged/defect trend line after the split
        - accept splits with enough angle change and inward offset
        - choose the accepted split with the smallest total residual
    """
    points = np.asarray(points, dtype=float)
    n = len(points)
    if n < 5:
        return max(0, n - 1), np.zeros(n), np.zeros(n)

    if side == "max_u":
        main = points[:, 1]
        normal = points[:, 0]
        inward_sign = -1.0  # inward means u becomes smaller
    elif side == "max_v":
        main = points[:, 0]
        normal = points[:, 1]
        inward_sign = -1.0  # inward means v becomes smaller
    elif side == "min_v":
        main = points[:, 0]
        normal = points[:, 1]
        inward_sign = 1.0   # inward means v becomes larger
    else:
        raise ValueError(f"Unsupported side: {side}")

    tail_min = float(np.clip(cfg.turn_search_tail_min_ratio, 0.0, 1.0))
    tail_max = float(np.clip(cfg.turn_search_tail_max_ratio, tail_min + 1e-6, 1.0))
    min_seg = int(np.clip(cfg.twofit_min_segment_points, 3, max(3, n // 2)))
    start = max(min_seg, int(np.floor(n * (1.0 - tail_max))))
    end = min(n - min_seg, int(np.ceil(n * (1.0 - tail_min))), int(n * (1.0 - cfg.turn_exclude_ratio)))
    end = max(start + 1, end)

    single_a, single_b = np.polyfit(main, normal, deg=1)
    single_residual = normal - (single_a * main + single_b)
    single_sse = float(np.sum(single_residual ** 2))

    score = np.zeros(n, dtype=float)
    candidates = []
    fallback = None

    for split in range(start, end):
        if split < min_seg or (n - split) < min_seg:
            continue

        a1, b1 = np.polyfit(main[:split], normal[:split], deg=1)
        a2, b2 = np.polyfit(main[split:], normal[split:], deg=1)

        pred1 = a1 * main[:split] + b1
        pred2 = a2 * main[split:] + b2
        res1 = normal[:split] - pred1
        res2 = normal[split:] - pred2
        total_sse = float(np.sum(res1 ** 2) + np.sum(res2 ** 2))

        angle = abs(np.arctan(a2) - np.arctan(a1))
        angle = min(angle, np.pi - angle)
        angle_deg = float(np.degrees(angle))

        stable_pred_after = a1 * main[split:] + b1
        inward_after = inward_sign * (normal[split:] - stable_pred_after)
        inward_after = np.maximum(inward_after, 0.0)
        inward_offset = float(np.median(inward_after))

        improvement = max(single_sse - total_sse, 0.0)
        score[split] = improvement / (single_sse + 1e-12)

        penalty = float(cfg.twofit_regularization) * split / max(n - 1, 1)
        record = {
            "split": split,
            "total_sse": total_sse + penalty,
            "raw_sse": total_sse,
            "angle_deg": angle_deg,
            "inward_offset": inward_offset,
            "a1": a1,
            "b1": b1,
            "a2": a2,
            "b2": b2,
            "score": score[split],
        }

        if fallback is None or record["score"] > fallback["score"]:
            fallback = record

        if (
            angle_deg >= cfg.twofit_min_angle_deg and
            inward_offset >= cfg.twofit_min_inward_offset
        ):
            candidates.append(record)

    if len(candidates) > 0:
        max_candidate_score = max(r["score"] for r in candidates)
        good_candidates = [
            r for r in candidates
            if r["score"] >= 0.85 * max_candidate_score
        ]
        best = min(good_candidates, key=lambda r: r["split"])
    elif fallback is not None:
        best = fallback
    else:
        best = {"split": max(0, min(n - 1, start)), "a1": single_a, "b1": single_b}

    idx = int(np.clip(best["split"], 0, n - 1))
    baseline = best["a1"] * main + best["b1"]
    raw_dev = inward_sign * (normal - baseline)
    inward_dev = np.maximum(raw_dev, 0.0)
    inward_dev = _safe_savgol_1d(inward_dev, cfg.turn_score_smooth_window, polyorder=2)

    return idx, score, inward_dev


def compute_corner_roi_frac_from_turns(
    uv: np.ndarray,
    u_line: np.ndarray,
    v_line: np.ndarray,
    turn_idx_u: int,
    turn_idx_v: int,
    corner_mode: str,
    margin_ratio: float,
    min_frac: float,
    max_frac: float,
):
    u_min_all, v_min_all = uv.min(axis=0)
    u_max_all, v_max_all = uv.max(axis=0)

    total_u = u_max_all - u_min_all
    total_v = v_max_all - v_min_all

    # u_line is max_u boundary and mainly varies along v, so it gives defect length along v.
    # v_line is max_v/min_v boundary and mainly varies along u, so it gives defect length along u.
    turn_u_on_v_line = v_line[turn_idx_v, 0]
    turn_v_on_u_line = u_line[turn_idx_u, 1]

    if corner_mode == "max_u_min_v":
        len_u_defect = u_max_all - turn_u_on_v_line
        len_v_defect = turn_v_on_u_line - v_min_all
    elif corner_mode == "max_u_max_v":
        len_u_defect = u_max_all - turn_u_on_v_line
        len_v_defect = v_max_all - turn_v_on_u_line
    else:
        raise ValueError(f"Unsupported corner_mode: {corner_mode}")

    frac_u = len_u_defect / (total_u + 1e-12)
    frac_v = len_v_defect / (total_v + 1e-12)

    frac_u = float(np.clip(frac_u * margin_ratio, min_frac, max_frac))
    frac_v = float(np.clip(frac_v * margin_ratio, min_frac, max_frac))
    return frac_u, frac_v


def extract_boundary_support_uv(cfg: CompletionConfig, plane: PlaneData):
    """
    Build uv points for boundary extraction.

    The top plane points define the occupancy mask later. Near-top rest points
    are added only here so sloped/slide-like edge points can support the outer
    boundary used by turn detection.
    """
    top_uv = plane.top_uv

    if not cfg.use_rest_points_for_boundary:
        return top_uv, np.empty((0, 2)), np.empty((0, 3))

    rest_points = np.asarray(plane.rest_pcd.points)
    if len(rest_points) == 0:
        return top_uv, np.empty((0, 2)), np.empty((0, 3))

    rest_uvz = project_to_uvz(
        rest_points,
        plane.origin,
        plane.u_axis,
        plane.v_axis,
        plane.n_axis,
    )
    rest_uv = rest_uvz[:, :2]
    rest_z = rest_uvz[:, 2]

    near_top_mask = (
        (rest_z >= cfg.boundary_support_z_min) &
        (rest_z <= cfg.boundary_support_z_max)
    )

    near_top_rest_uv = rest_uv[near_top_mask]
    near_top_rest_points = rest_points[near_top_mask]

    if len(near_top_rest_uv) > 0:
        boundary_support_uv = np.vstack([top_uv, near_top_rest_uv])
    else:
        boundary_support_uv = top_uv

    return boundary_support_uv, near_top_rest_uv, near_top_rest_points


def estimate_roi_from_boundary(cfg: CompletionConfig, plane: PlaneData, corner_mode: str) -> RoiData:
    top_uv = plane.top_uv
    boundary_support_uv, near_top_rest_uv, near_top_rest_points = extract_boundary_support_uv(cfg, plane)
    u_side, v_side = get_target_sides_from_corner_mode(corner_mode)

    u_candidates, u_side_support_uv, u_top_candidates = extract_side_specific_boundary_points(
        top_uv=top_uv,
        support_uv=boundary_support_uv,
        side=u_side,
        cfg=cfg,
    )
    v_candidates, v_side_support_uv, v_top_candidates = extract_side_specific_boundary_points(
        top_uv=top_uv,
        support_uv=boundary_support_uv,
        side=v_side,
        cfg=cfg,
    )

    u_sorted = sort_boundary_from_far_to_defect(u_candidates, u_side, corner_mode)
    v_sorted = sort_boundary_from_far_to_defect(v_candidates, v_side, corner_mode)

    u_sorted = keep_boundary_tail(u_sorted, cfg.boundary_keep_tail_fraction)
    v_sorted = keep_boundary_tail(v_sorted, cfg.boundary_keep_tail_fraction)

    u_line_raw = select_near_corner_segment(u_sorted, u_side, cfg)
    v_line_raw = select_near_corner_segment(v_sorted, v_side, cfg)

    u_line = robust_filter_boundary_line(u_line_raw, u_side, cfg.rolling_window, cfg.rolling_jump_th)
    v_line = robust_filter_boundary_line(v_line_raw, v_side, cfg.rolling_window, cfg.rolling_jump_th)

    u_line = smooth_boundary_line(u_line, cfg.smooth_window, cfg.smooth_polyorder)
    v_line = smooth_boundary_line(v_line, cfg.smooth_window, cfg.smooth_polyorder)

    turn_idx_u, score_u, deviation_u = find_turn_by_piecewise_linear_fit(u_line, u_side, cfg)
    turn_idx_v, score_v, deviation_v = find_turn_by_piecewise_linear_fit(v_line, v_side, cfg)

    roi_frac_u, roi_frac_v = compute_corner_roi_frac_from_turns(
        uv=top_uv,
        u_line=u_line,
        v_line=v_line,
        turn_idx_u=turn_idx_u,
        turn_idx_v=turn_idx_v,
        corner_mode=corner_mode,
        margin_ratio=cfg.roi_margin_ratio,
        min_frac=cfg.roi_min_frac,
        max_frac=cfg.roi_max_frac,
    )

    return RoiData(
        u_side=u_side,
        v_side=v_side,
        boundary_support_uv=boundary_support_uv,
        near_top_rest_uv=near_top_rest_uv,
        near_top_rest_points=near_top_rest_points,
        u_side_support_uv=u_side_support_uv,
        v_side_support_uv=v_side_support_uv,
        u_line_raw=u_line_raw,
        v_line_raw=v_line_raw,
        u_line=u_line,
        v_line=v_line,
        turn_idx_u=turn_idx_u,
        turn_idx_v=turn_idx_v,
        turn_point_u=u_line[turn_idx_u],
        turn_point_v=v_line[turn_idx_v],
        score_u=score_u,
        score_v=score_v,
        deviation_u=deviation_u,
        deviation_v=deviation_v,
        roi_frac_u=roi_frac_u,
        roi_frac_v=roi_frac_v,
    )


def reanchor_frame_to_defect_turn_intersection(plane: PlaneData, roi: RoiData):
    """
    Keep the top-plane PCA axes unchanged, but move the UV origin to the
    intersection defined by the two TwoFit defect turn points.

    U coordinate comes from the V-side turn point; V coordinate comes from
    the U-side turn point:
        u0 = turn_point_v[0]
        v0 = turn_point_u[1]
    """
    uv_shift = np.array([roi.turn_point_u[0], roi.turn_point_v[1]], dtype=float)
    fill_origin = plane.origin + uv_shift[0] * plane.u_axis + uv_shift[1] * plane.v_axis

    plane.origin = fill_origin
    plane.top_uv = project_to_uv(plane.top_points, plane.origin, plane.u_axis, plane.v_axis)

    def shift_uv(arr):
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0:
            return arr.copy()
        return arr - uv_shift

    roi.boundary_support_uv = shift_uv(roi.boundary_support_uv)
    roi.near_top_rest_uv = shift_uv(roi.near_top_rest_uv)
    roi.u_side_support_uv = shift_uv(roi.u_side_support_uv)
    roi.v_side_support_uv = shift_uv(roi.v_side_support_uv)
    roi.u_line_raw = shift_uv(roi.u_line_raw)
    roi.v_line_raw = shift_uv(roi.v_line_raw)
    roi.u_line = shift_uv(roi.u_line)
    roi.v_line = shift_uv(roi.v_line)
    roi.turn_point_u = shift_uv(roi.turn_point_u)
    roi.turn_point_v = shift_uv(roi.turn_point_v)

    print("\n========== Filling frame re-anchored ==========")
    print("fill origin world:", plane.origin)
    print("turn intersection in new UV: [0, 0]")
    print("u_axis unchanged:", plane.u_axis)
    print("v_axis unchanged:", plane.v_axis)
    return plane, roi


def build_domain_mask_from_fill_origin(
    u_grid: np.ndarray,
    v_grid: np.ndarray,
    corner_mode: str,
):
    """Build the fill domain from the TwoFit turn-point intersection at UV=(0,0)."""
    U, V = np.meshgrid(u_grid, v_grid)
    tol = 0.5 * min(abs(u_grid[1] - u_grid[0]) if len(u_grid) > 1 else 0.0,
                    abs(v_grid[1] - v_grid[0]) if len(v_grid) > 1 else 0.0)
    if corner_mode == "max_u_max_v":
        return (U <= tol) & (V <= tol)
    if corner_mode == "max_u_min_v":
        return (U <= tol) & (V >= -tol)
    raise ValueError(f"Unsupported corner_mode: {corner_mode}")


def build_domain_mask_from_turn_box(
    u_grid: np.ndarray,
    v_grid: np.ndarray,
    roi: RoiData,
):
    """
    Finite defect ROI defined by:
        O  = filling origin = (0, 0)
        P1 = U-side defect turn point
        P2 = V-side defect turn point

    After frame re-anchoring:
        P1 = (0, dv)
        P2 = (du, 0)

    The fourth corner is:
        P3 = (du, dv)
    """

    U, V = np.meshgrid(u_grid, v_grid)

    p1 = np.asarray(roi.turn_point_u, dtype=float)
    p2 = np.asarray(roi.turn_point_v, dtype=float)

    du = float(p2[0])
    dv = float(p1[1])

    u_lo = min(0.0, du)
    u_hi = max(0.0, du)

    v_lo = min(0.0, dv)
    v_hi = max(0.0, dv)

    tol = 0.5 * min(
        abs(u_grid[1] - u_grid[0]) if len(u_grid) > 1 else 0.0,
        abs(v_grid[1] - v_grid[0]) if len(v_grid) > 1 else 0.0,
    )

    domain_mask = (
        (U >= u_lo - tol)
        & (U <= u_hi + tol)
        & (V >= v_lo - tol)
        & (V <= v_hi + tol)
    )

    print("\n========== Finite defect ROI ==========")
    print("origin O:", [0.0, 0.0])
    print("P1 turn U:", p1)
    print("P2 turn V:", p2)
    print("P3:", [du, dv])
    print(f"U range: [{u_lo:.4f}, {u_hi:.4f}] m")
    print(f"V range: [{v_lo:.4f}, {v_hi:.4f}] m")
    print("ROI pixels:", int(domain_mask.sum()))

    return domain_mask


def build_corner_roi(H: int, W: int, mode: str, frac_u: float, frac_v: float):
    roi = np.zeros((H, W), dtype=bool)
    h = max(1, int(round(H * frac_v)))
    w = max(1, int(round(W * frac_u)))

    if mode == "max_u_min_v":
        roi[:h, W - w:] = True
    elif mode == "max_u_max_v":
        roi[H - h:, W - w:] = True
    else:
        raise ValueError(f"Unsupported corner_mode: {mode}")
    return roi


def keep_largest_component(mask: np.ndarray, min_area: int = 1) -> np.ndarray:
    label_mask, num = ndimage.label(mask)
    if num == 0:
        return np.zeros_like(mask, dtype=bool)

    labels = np.arange(1, num + 1)
    areas = np.asarray(ndimage.sum(mask, label_mask, index=labels))
    valid = labels[areas >= min_area]
    if len(valid) == 0:
        return np.zeros_like(mask, dtype=bool)

    best_label = valid[np.argmax(areas[valid - 1])]
    return label_mask == best_label


def extract_top_defect_margin(
    defect_mask: np.ndarray,
    top_occ_mask: np.ndarray,
    u_min: float,
    v_min: float,
    grid_res: float,
    origin: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
):
    defect_near = ndimage.binary_dilation(defect_mask, iterations=1)
    margin_mask = top_occ_mask & defect_near
    margin_uv = mask_to_uv_points(margin_mask, u_min, v_min, grid_res)
    if len(margin_uv) > 0:
        margin_points = uv_to_3d(margin_uv, origin, u_axis, v_axis)
    else:
        margin_points = np.empty((0, 3))
    return margin_points, margin_uv


def load_and_prepare_plane(cfg: CompletionConfig) -> PlaneData:
    pcd_raw = o3d.io.read_point_cloud(cfg.fine_pcd_path)
    pcd_raw = keep_largest_cluster(pcd_raw)

    points_raw = np.asarray(pcd_raw.points)
    plane_model, plane_pcd, rest_pcd = find_plane(
        pcd_raw,
        voxel_size=cfg.plane_voxel_size,
        distance_threshold=cfg.plane_distance_threshold,
        num_iterations=cfg.plane_num_iterations,
    )

    n_axis = plane_model[:3]
    inward_vec = np.asarray(cfg.inward_vec, dtype=float)  # inward_vec=(0,0,1)
    #要求n_axis垂直朝下指向object内部
    if np.dot(n_axis, inward_vec) > 0:
        n_axis = -n_axis
    n_axis = normalize(n_axis)

    top_points = np.asarray(plane_pcd.points)
    origin, u_axis, v_axis = build_plane_basis(top_points, n_axis)
    top_uv = project_to_uv(top_points, origin, u_axis, v_axis)

    rest_points = np.asarray(rest_pcd.points)
    rest_local = project_to_uvz(rest_points,origin,u_axis, v_axis,n_axis,)
    rest_z = rest_local[:, 2]
    rest_z = rest_z[np.isfinite(rest_z)]
    z_bottom = float(np.quantile(rest_z, 0.98))
    print(f"z_bottom Q98: {z_bottom:.6f} m ({z_bottom * 1000:.2f} mm)")

    return PlaneData(
        pcd_raw=pcd_raw,
        plane_model=plane_model,
        plane_pcd=plane_pcd,
        rest_pcd=rest_pcd,
        origin=origin,
        u_axis=u_axis,
        v_axis=v_axis,
        n_axis=n_axis,
        top_points=top_points,
        points_raw=points_raw,
        top_uv=top_uv,
        z_bottom=z_bottom
    )


def build_grid_and_defect_masks(
    cfg: CompletionConfig,
    plane: PlaneData,
    roi: RoiData,
    corner_mode: str,
) -> GridData:
    top_uv = plane.top_uv

    u_min = float(top_uv[:, 0].min() - cfg.grid_pad)
    u_max = float(top_uv[:, 0].max() + cfg.grid_pad)
    v_min = float(top_uv[:, 1].min() - cfg.grid_pad)
    v_max = float(top_uv[:, 1].max() + cfg.grid_pad)

    u_grid = np.arange(u_min, u_max + cfg.grid_res, cfg.grid_res)
    v_grid = np.arange(v_min, v_max + cfg.grid_res, cfg.grid_res)
    W, H = len(u_grid), len(v_grid)

    top_occ_mask = np.zeros((H, W), dtype=bool)
    xs, ys, valid = uv_to_grid_index(top_uv, u_min, v_min, cfg.grid_res, W, H)
    top_occ_mask[ys[valid], xs[valid]] = True

    top_occ_smooth = ndimage.binary_closing(top_occ_mask, iterations=cfg.top_occ_close_iter)
    if cfg.top_occ_dilate_iter > 0:
        top_occ_smooth = ndimage.binary_dilation(top_occ_smooth, iterations=cfg.top_occ_dilate_iter)

    # The fill origin is the intersection of the two defect turn coordinates.
    # PCA axes are unchanged. The domain starts from UV=(0,0) and extends
    # toward the damaged object corner according to corner_mode.
    #domain_mask = build_domain_mask_from_fill_origin(u_grid, v_grid, corner_mode)

    domain_mask = build_domain_mask_from_turn_box(u_grid, v_grid,roi,)
    corner_roi = domain_mask.copy()

    defect_mask = domain_mask & (~top_occ_smooth)
    defect_mask = ndimage.binary_closing(defect_mask, iterations=cfg.defect_close_iter)
    defect_mask = keep_largest_component(defect_mask, min_area=cfg.min_component_area_pixels)

    top_defect_margin_points, top_defect_margin_uv = extract_top_defect_margin(
        defect_mask=defect_mask,
        top_occ_mask=top_occ_smooth,
        u_min=u_min,
        v_min=v_min,
        grid_res=cfg.grid_res,
        origin=plane.origin,
        u_axis=plane.u_axis,
        v_axis=plane.v_axis,
    )

    defect_uv = mask_to_uv_points(defect_mask, u_min, v_min, cfg.grid_res)
    if len(defect_uv) > 0:
        repair_top_points = uv_to_3d(defect_uv, plane.origin, plane.u_axis, plane.v_axis)
        top_plane_center = np.median(repair_top_points, axis=0)
    else:
        top_plane_center = plane.top_points.mean(axis=0)

    print("\n========== Grid and defect mask ==========")
    print("grid H, W:", H, W)
    print("corner_roi area:", int(corner_roi.sum()))
    print("defect_mask area:", int(defect_mask.sum()))
    print("top_defect_margin point num:", len(top_defect_margin_points))

    return GridData(
        u_min=u_min,
        v_min=v_min,
        u_max=u_max,
        v_max=v_max,
        u_grid=u_grid,
        v_grid=v_grid,
        H=H,
        W=W,
        top_occ_mask=top_occ_mask,
        top_occ_smooth=top_occ_smooth,
        corner_roi=corner_roi,
        domain_mask=domain_mask,
        defect_mask=defect_mask,
        top_defect_margin_points=top_defect_margin_points,
        top_defect_margin_uv=top_defect_margin_uv,
        top_plane_center=top_plane_center,
    )


def visualize_filling_start_point(plane: PlaneData):
    raw_vis = o3d.geometry.PointCloud(plane.pcd_raw)
    raw_vis.paint_uniform_color([0.7, 0.7, 0.7])

    top_vis = o3d.geometry.PointCloud(plane.plane_pcd)
    top_vis.paint_uniform_color([1.0, 0.0, 0.0])

    rest_vis = o3d.geometry.PointCloud(plane.rest_pcd)
    rest_vis.paint_uniform_color([1.0, 0.6, 0.0])

    start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.004)
    start_sphere.translate(plane.origin)
    start_sphere.paint_uniform_color([0.0, 1.0, 0.0])
    start_sphere.compute_vertex_normals()

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)

    T = np.eye(4)
    T[:3, :3] = np.column_stack([
        plane.u_axis,
        plane.v_axis,
        plane.n_axis,
    ])
    T[:3, 3] = plane.origin
    frame.transform(T)

    print("\n========== Filling Start Point ==========")
    print("origin xyz:", plane.origin)
    print("u_axis:", plane.u_axis)
    print("v_axis:", plane.v_axis)
    print("n_axis:", plane.n_axis)

    o3d.visualization.draw_geometries([
        top_vis,
        rest_vis,
        start_sphere,
        frame,
    ])


PART1_STATE_NAME = "_part1_state.npz"
PART1_RAW_PCD_NAME = "_part1_raw.pcd"
PART1_PLANE_PCD_NAME = "_part1_plane.pcd"
PART1_REST_PCD_NAME = "_part1_rest.pcd"


def save_part1_state(cfg: CompletionConfig, plane: PlaneData, roi: RoiData, grid: GridData, corner_mode: str):
    os.makedirs(cfg.output_dir, exist_ok=True)
    o3d.io.write_point_cloud(os.path.join(cfg.output_dir, PART1_RAW_PCD_NAME), plane.pcd_raw)
    o3d.io.write_point_cloud(os.path.join(cfg.output_dir, PART1_PLANE_PCD_NAME), plane.plane_pcd)
    o3d.io.write_point_cloud(os.path.join(cfg.output_dir, PART1_REST_PCD_NAME), plane.rest_pcd)
    np.savez(
        os.path.join(cfg.output_dir, PART1_STATE_NAME),
        corner_mode=np.asarray(corner_mode),
        plane_model=plane.plane_model,
        origin=plane.origin,
        u_axis=plane.u_axis,
        v_axis=plane.v_axis,
        n_axis=plane.n_axis,
        z_bottom=np.asarray(plane.z_bottom),
        top_points=plane.top_points,
        points_raw=plane.points_raw,
        top_uv=plane.top_uv,
        u_side=np.asarray(roi.u_side),
        v_side=np.asarray(roi.v_side),
        boundary_support_uv=roi.boundary_support_uv,
        near_top_rest_uv=roi.near_top_rest_uv,
        near_top_rest_points=roi.near_top_rest_points,
        u_side_support_uv=roi.u_side_support_uv,
        v_side_support_uv=roi.v_side_support_uv,
        u_line_raw=roi.u_line_raw,
        v_line_raw=roi.v_line_raw,
        u_line=roi.u_line,
        v_line=roi.v_line,
        turn_idx_u=np.asarray(roi.turn_idx_u),
        turn_idx_v=np.asarray(roi.turn_idx_v),
        turn_point_u=roi.turn_point_u,
        turn_point_v=roi.turn_point_v,
        score_u=roi.score_u,
        score_v=roi.score_v,
        deviation_u=roi.deviation_u,
        deviation_v=roi.deviation_v,
        roi_frac_u=np.asarray(roi.roi_frac_u),
        roi_frac_v=np.asarray(roi.roi_frac_v),
        u_min=np.asarray(grid.u_min), v_min=np.asarray(grid.v_min), u_max=np.asarray(grid.u_max), v_max=np.asarray(grid.v_max),
        u_grid=grid.u_grid, v_grid=grid.v_grid, H=np.asarray(grid.H), W=np.asarray(grid.W),
        top_occ_mask=grid.top_occ_mask, top_occ_smooth=grid.top_occ_smooth, corner_roi=grid.corner_roi,
        domain_mask=grid.domain_mask, defect_mask=grid.defect_mask,
        top_defect_margin_points=grid.top_defect_margin_points, top_defect_margin_uv=grid.top_defect_margin_uv,
        top_plane_center=grid.top_plane_center,
    )
    print("\n========== Part 1 done ==========")
    print("saved:", os.path.join(cfg.output_dir, PART1_STATE_NAME))


def load_part1_state(output_dir: str):
    data = np.load(os.path.join(output_dir, PART1_STATE_NAME), allow_pickle=True)
    pcd_raw = o3d.io.read_point_cloud(os.path.join(output_dir, PART1_RAW_PCD_NAME))
    plane_pcd = o3d.io.read_point_cloud(os.path.join(output_dir, PART1_PLANE_PCD_NAME))
    rest_pcd = o3d.io.read_point_cloud(os.path.join(output_dir, PART1_REST_PCD_NAME))
    plane = PlaneData(
        pcd_raw=pcd_raw, plane_model=data["plane_model"], plane_pcd=plane_pcd, rest_pcd=rest_pcd,
        origin=data["origin"], u_axis=data["u_axis"], v_axis=data["v_axis"], n_axis=data["n_axis"],
        top_points=data["top_points"], points_raw=data["points_raw"], top_uv=data["top_uv"],z_bottom=float(data["z_bottom"]),
    )
    roi = RoiData(
        u_side=str(data["u_side"]), v_side=str(data["v_side"]), boundary_support_uv=data["boundary_support_uv"],
        near_top_rest_uv=data["near_top_rest_uv"], near_top_rest_points=data["near_top_rest_points"],
        u_side_support_uv=data["u_side_support_uv"], v_side_support_uv=data["v_side_support_uv"],
        u_line_raw=data["u_line_raw"], v_line_raw=data["v_line_raw"], u_line=data["u_line"], v_line=data["v_line"],
        turn_idx_u=int(data["turn_idx_u"]), turn_idx_v=int(data["turn_idx_v"]),
        turn_point_u=data["turn_point_u"], turn_point_v=data["turn_point_v"], score_u=data["score_u"], score_v=data["score_v"],
        deviation_u=data["deviation_u"], deviation_v=data["deviation_v"], roi_frac_u=float(data["roi_frac_u"]), roi_frac_v=float(data["roi_frac_v"]),
        
    )
    grid = GridData(
        u_min=float(data["u_min"]), v_min=float(data["v_min"]), u_max=float(data["u_max"]), v_max=float(data["v_max"]),
        u_grid=data["u_grid"], v_grid=data["v_grid"], H=int(data["H"]), W=int(data["W"]),
        top_occ_mask=data["top_occ_mask"].astype(bool), top_occ_smooth=data["top_occ_smooth"].astype(bool),
        corner_roi=data["corner_roi"].astype(bool), domain_mask=data["domain_mask"].astype(bool), defect_mask=data["defect_mask"].astype(bool),
        top_defect_margin_points=data["top_defect_margin_points"], top_defect_margin_uv=data["top_defect_margin_uv"], top_plane_center=data["top_plane_center"],
    )
    return plane, roi, grid, str(data["corner_mode"])



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-segment line-fit ROI depression completion.")
    parser.add_argument("--pcd", type=str, default=None, help="Input fine_fuse.pcd path.")
    parser.add_argument("--corner-json", type=str, default=None, help="corner_mapping_result.json path.")
    parser.add_argument("--corner-mode", type=str, default=None, choices=["max_u_max_v", "max_u_min_v"], help="Override corner mode.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory.")
    parser.add_argument("--roi-restriction", type=float, default=None, help="Near-corner boundary physical length for ROI turn detection.")
    parser.add_argument("--boundary-support-z-min", type=float, default=None, help="Min rest_z for near-top boundary support.")
    parser.add_argument("--boundary-support-z-max", type=float, default=None, help="Max rest_z for near-top boundary support.")
    parser.add_argument("--boundary-side-band-inward", type=float, default=None, help="Inward width of each side-specific boundary support corridor.")
    parser.add_argument("--boundary-side-band-outward", type=float, default=None, help="Outward width of each side-specific boundary support corridor.")
    parser.add_argument("--twofit-min-segment-points", type=int, default=None, help="Minimum points on each side of a candidate two-line split.")
    parser.add_argument("--twofit-min-angle-deg", type=float, default=None, help="Minimum angle change between stable and defect trend lines.")
    parser.add_argument("--twofit-min-inward-offset", type=float, default=None, help="Minimum median inward offset after a candidate split.")
    parser.add_argument("--twofit-regularization", type=float, default=None, help="Optional split-index penalty added to two-fit SSE.")
    parser.add_argument("--no-rest-boundary", action="store_true", help="Use only top plane points for ROI boundary extraction.")
    parser.add_argument("--enable-column-filter", action="store_true", help="Enable column depth post-filter.")
    parser.add_argument("--min-column-layers", type=int, default=None, help="Min layers for column depth filter.")
    parser.add_argument("--wall-endpoint-extend-pixels", type=int, default=None, help="Short normal extension length for skeleton wall endpoints.")
    parser.add_argument("--no-vis", action="store_true", help="Disable Open3D visualization.")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> CompletionConfig:
    cfg = CompletionConfig()
    if args.pcd is not None:
        cfg.fine_pcd_path = args.pcd
    if args.corner_json is not None:
        cfg.corner_json_path = args.corner_json
        if args.corner_mode is None:
            cfg.corner_mode = None
    if args.corner_mode is not None:
        cfg.corner_mode = args.corner_mode
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.roi_restriction is not None:
        cfg.boundary_roi_restriction = args.roi_restriction
    if args.boundary_support_z_min is not None:
        cfg.boundary_support_z_min = args.boundary_support_z_min
    if args.boundary_support_z_max is not None:
        cfg.boundary_support_z_max = args.boundary_support_z_max
    if args.boundary_side_band_inward is not None:
        cfg.boundary_side_band_inward = args.boundary_side_band_inward
    if args.boundary_side_band_outward is not None:
        cfg.boundary_side_band_outward = args.boundary_side_band_outward
    if args.twofit_min_segment_points is not None:
        cfg.twofit_min_segment_points = args.twofit_min_segment_points
    if args.twofit_min_angle_deg is not None:
        cfg.twofit_min_angle_deg = args.twofit_min_angle_deg
    if args.twofit_min_inward_offset is not None:
        cfg.twofit_min_inward_offset = args.twofit_min_inward_offset
    if args.twofit_regularization is not None:
        cfg.twofit_regularization = args.twofit_regularization
    if args.no_rest_boundary:
        cfg.use_rest_points_for_boundary = False
    if args.enable_column_filter:
        cfg.enable_column_depth_filter = True
    if args.min_column_layers is not None:
        cfg.min_column_layers = args.min_column_layers
    if args.wall_endpoint_extend_pixels is not None:
        cfg.wall_endpoint_extend_pixels = args.wall_endpoint_extend_pixels
    if args.no_vis:
        cfg.visualize = False
    return cfg


def run_part1(cfg: CompletionConfig):
    corner_mode = resolve_corner_mode(cfg)
    plane = load_and_prepare_plane(cfg)
    roi = estimate_roi_from_boundary(cfg, plane, corner_mode)
    plane, roi = reanchor_frame_to_defect_turn_intersection(plane, roi)
    grid = build_grid_and_defect_masks(cfg, plane, roi, corner_mode)
    save_part1_state(cfg, plane, roi, grid, corner_mode)


if __name__ == "__main__":
    run_part1(config_from_args(parse_args()))

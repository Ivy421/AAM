"""
depression_completion_twoFit.py

Two-segment line-fit ROI depression point-cloud completion pipeline.

支持的 CORNER_MODE:
    - max_u_max_v
    - max_u_min_v

主流程：
    1. 读取 fine_fuse.pcd，并只拟合一次顶面平面。
    2. 基于顶面平面建立局部 uvz 坐标系。
    3. 仅使用 top plane 点投影得到 top_uv，并提取目标角相邻的两条边界轮廓。
    4. 按 corner_mode 选取靠近缺陷角的一段边界线。
    5. 对边界线做轻量去噪和平滑。
    6. 用“一维偏离基线最大变化点”寻找两条边界的转折点。
    7. 根据两个转折点计算 roi_frac_u / roi_frac_v。
    8. 根据 corner_mode 和 roi_frac 生成 corner_roi。
    9. 在 corner_roi 内生成顶面 defect_mask。
    10. 使用逐层 barrier + flood fill 方法生成补全体点云。
    11. 可选后处理：按 uv 列统计 z 方向层数，删除层数少于 min_column_layers 的列。
    12. 保存 repair model、side planes、top margin，以及 fix/contact masks。
"""

import os
import json
import argparse
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import open3d as o3d
from scipy import ndimage
from scipy.signal import savgol_filter

# Keep Open3D RANSAC deterministic.
o3d.utility.random.seed(274)


# ============================================================
# Configuration
# ============================================================
@dataclass
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


# ============================================================
# Data containers
# ============================================================
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


# ============================================================
# Basic geometry utilities
# ============================================================
def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    return v / (np.linalg.norm(v) + 1e-12)


def resolve_corner_mode(cfg: CompletionConfig) -> str:
    allowed = {"max_u_max_v", "max_u_min_v"}
    if cfg.corner_mode is not None:
        corner_mode = cfg.corner_mode
    else:
        with open(cfg.corner_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        corner_mode = data.get("corner_mode", "max_u_max_v")

    if corner_mode not in allowed:
        raise ValueError(f"Wrong corner mode {corner_mode}, from corner_mapping_result.json.")
    return corner_mode


def find_plane(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
    distance_threshold: float,
    num_iterations: int,
):
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
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


# ============================================================
# Boundary ROI estimation
# ============================================================
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


# ============================================================
# Mask utilities
# ============================================================
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


def skeletonize_zhang_suen(mask: np.ndarray, max_iter: int = 200) -> np.ndarray:
    """
    Thin a 2D binary mask to a one-pixel skeleton.

    This local implementation avoids adding a skimage dependency.
    """
    img = np.asarray(mask, dtype=bool).copy()
    if img.sum() == 0:
        return img

    for _ in range(max_iter):
        changed = False
        for step in (0, 1):
            padded = np.pad(img, 1, mode="constant", constant_values=False)
            p2 = padded[:-2, 1:-1]
            p3 = padded[:-2, 2:]
            p4 = padded[1:-1, 2:]
            p5 = padded[2:, 2:]
            p6 = padded[2:, 1:-1]
            p7 = padded[2:, :-2]
            p8 = padded[1:-1, :-2]
            p9 = padded[:-2, :-2]

            neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
            neighbor_count = sum(neighbors)
            transitions = np.zeros_like(img, dtype=np.int16)
            for a, b in zip(neighbors, neighbors[1:] + neighbors[:1]):
                transitions += (~a & b)

            if step == 0:
                side_rule = ~(p2 & p4 & p6) & ~(p4 & p6 & p8)
            else:
                side_rule = ~(p2 & p4 & p8) & ~(p2 & p6 & p8)

            remove = (
                img
                & (neighbor_count >= 2)
                & (neighbor_count <= 6)
                & (transitions == 1)
                & side_rule
            )
            if np.any(remove):
                img[remove] = False
                changed = True

        if not changed:
            break
    return img


def draw_grid_line(mask: np.ndarray, p0: np.ndarray, p1: np.ndarray):
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    dist = np.abs(p1 - p0)
    n = int(max(dist[0], dist[1])) + 1
    if n <= 1:
        y, x = np.round(p0).astype(int)
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
            mask[y, x] = True
        return

    ys = np.round(np.linspace(p0[0], p1[0], n)).astype(int)
    xs = np.round(np.linspace(p0[1], p1[1], n)).astype(int)
    valid = (ys >= 0) & (ys < mask.shape[0]) & (xs >= 0) & (xs < mask.shape[1])
    mask[ys[valid], xs[valid]] = True


def bridge_wall_components(
    mask: np.ndarray,
    domain_mask: np.ndarray,
    min_area: int = 2,
    max_bridge_pixels: int = 18,
) -> np.ndarray:
    """
    Keep the main broken wall chain and bridge nearby connected components.

    Raw wall slices are often broken into many small components. Keeping only
    the largest one loses most of the wall, so this starts from the largest
    component and repeatedly connects the nearest component within a limited
    pixel gap.
    """
    mask = np.asarray(mask, dtype=bool) & np.asarray(domain_mask, dtype=bool)
    label_mask, num = ndimage.label(mask)
    if num == 0:
        return np.zeros_like(mask, dtype=bool)

    labels = np.arange(1, num + 1)
    areas = np.asarray(ndimage.sum(mask, label_mask, index=labels))
    keep_labels = labels[areas >= min_area]
    if len(keep_labels) == 0:
        keep_labels = np.asarray([labels[int(np.argmax(areas))]])

    start_label = keep_labels[int(np.argmax(areas[keep_labels - 1]))]
    connected = label_mask == start_label
    remaining = set(int(label) for label in keep_labels if label != start_label)

    while len(remaining) > 0:
        _, nearest = ndimage.distance_transform_edt(~connected, return_indices=True)
        best_label = None
        best_point = None
        best_target = None
        best_dist = np.inf

        for label in list(remaining):
            coords = np.column_stack(np.where(label_mask == label))
            if len(coords) == 0:
                remaining.remove(label)
                continue

            d2 = np.sum((coords - nearest[:, coords[:, 0], coords[:, 1]].T) ** 2, axis=1)
            idx = int(np.argmin(d2))
            dist = float(np.sqrt(d2[idx]))
            if dist < best_dist:
                best_dist = dist
                best_label = label
                best_point = coords[idx]
                best_target = nearest[:, best_point[0], best_point[1]]

        if best_label is None or best_dist > max_bridge_pixels:
            break

        draw_grid_line(connected, best_point, best_target)
        connected |= label_mask == best_label
        remaining.remove(best_label)

    return connected & domain_mask


def select_curve_endpoints(curve_mask: np.ndarray) -> np.ndarray:
    coords = np.column_stack(np.where(curve_mask))
    if len(coords) <= 2:
        return coords

    padded = np.pad(curve_mask, 1, mode="constant", constant_values=False)
    neighbor_count = np.zeros_like(curve_mask, dtype=np.int16)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            neighbor_count += padded[1 + dy:1 + dy + curve_mask.shape[0], 1 + dx:1 + dx + curve_mask.shape[1]]

    endpoint_mask = curve_mask & (neighbor_count <= 1)
    endpoints = np.column_stack(np.where(endpoint_mask))
    if len(endpoints) >= 2:
        coords = endpoints

    d2 = np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=2)
    i, j = np.unravel_index(np.argmax(d2), d2.shape)
    return np.asarray([coords[i], coords[j]], dtype=int)


def get_domain_side_coords(domain_mask: np.ndarray, side: str) -> np.ndarray:
    ys, xs = np.where(domain_mask)
    if len(xs) == 0:
        return np.empty((0, 2), dtype=int)

    if side == "min_u":
        side_mask = xs == xs.min()
    elif side == "max_u":
        side_mask = xs == xs.max()
    elif side == "min_v":
        side_mask = ys == ys.min()
    elif side == "max_v":
        side_mask = ys == ys.max()
    else:
        raise ValueError(f"Unknown domain side: {side}")

    return np.column_stack([ys[side_mask], xs[side_mask]])


def closest_side_point(point: np.ndarray, side_coords: np.ndarray) -> np.ndarray:
    d2 = np.sum((side_coords - point[None, :]) ** 2, axis=1)
    return side_coords[int(np.argmin(d2))]


def seal_curve_to_domain_border(
    curve_mask: np.ndarray,
    domain_mask: np.ndarray,
    corner_mode: str,
) -> np.ndarray:
    curve = np.asarray(curve_mask, dtype=bool) & np.asarray(domain_mask, dtype=bool)
    if curve.sum() == 0:
        return curve

    if corner_mode == "max_u_max_v":
        target_sides = ("min_u", "min_v")
    elif corner_mode == "max_u_min_v":
        target_sides = ("min_u", "max_v")
    else:
        raise ValueError(f"Unsupported corner_mode: {corner_mode}")

    side_a = get_domain_side_coords(domain_mask, target_sides[0])
    side_b = get_domain_side_coords(domain_mask, target_sides[1])
    if len(side_a) == 0 or len(side_b) == 0:
        return curve

    sealed = curve.copy()
    endpoints = select_curve_endpoints(curve)
    if len(endpoints) == 1:
        target = closest_side_point(endpoints[0], side_a)
        draw_grid_line(sealed, endpoints[0], target)
        return sealed & domain_mask

    e0, e1 = endpoints[0], endpoints[1]
    e0_a = closest_side_point(e0, side_a)
    e0_b = closest_side_point(e0, side_b)
    e1_a = closest_side_point(e1, side_a)
    e1_b = closest_side_point(e1, side_b)

    cost_ab = np.sum((e0 - e0_a) ** 2) + np.sum((e1 - e1_b) ** 2)
    cost_ba = np.sum((e0 - e0_b) ** 2) + np.sum((e1 - e1_a) ** 2)
    if cost_ab <= cost_ba:
        draw_grid_line(sealed, e0, e0_a)
        draw_grid_line(sealed, e1, e1_b)
    else:
        draw_grid_line(sealed, e0, e0_b)
        draw_grid_line(sealed, e1, e1_a)

    return sealed & domain_mask


def estimate_endpoint_tangent(curve_mask: np.ndarray, endpoint: np.ndarray, k: int = 9) -> np.ndarray:
    coords = np.column_stack(np.where(curve_mask))
    if len(coords) < 2:
        return np.array([0.0, 1.0])

    endpoint = np.asarray(endpoint, dtype=float)
    d2 = np.sum((coords - endpoint[None, :]) ** 2, axis=1)
    nearest = coords[np.argsort(d2)[:min(k, len(coords))]].astype(float)
    centered = nearest - nearest.mean(axis=0)
    if len(nearest) < 2 or np.linalg.norm(centered) < 1e-12:
        farthest = coords[int(np.argmax(d2))].astype(float)
        tangent = farthest - endpoint
    else:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        tangent = vh[0]

    norm = np.linalg.norm(tangent)
    if norm < 1e-12:
        return np.array([0.0, 1.0])
    return tangent / norm


def desired_endpoint_normal(tangent: np.ndarray, corner_mode: str) -> np.ndarray:
    tangent = np.asarray(tangent, dtype=float)
    horizontal = abs(tangent[1]) >= abs(tangent[0])

    if horizontal:
        if corner_mode == "max_u_max_v":
            desired = np.array([1.0, 0.0])   # max_v in grid row direction.
        elif corner_mode == "max_u_min_v":
            desired = np.array([-1.0, 0.0])  # min_v in grid row direction.
        else:
            raise ValueError(f"Unsupported corner_mode: {corner_mode}")
    else:
        if corner_mode in ["max_u_max_v", "max_u_min_v"]:
            desired = np.array([0.0, 1.0])   # max_u in grid col direction.
        else:
            raise ValueError(f"Unsupported corner_mode: {corner_mode}")

    n1 = np.array([-tangent[1], tangent[0]])
    n2 = -n1
    normal = n1 if np.dot(n1, desired) >= np.dot(n2, desired) else n2
    return normal / (np.linalg.norm(normal) + 1e-12)


def extend_skeleton_endpoints_by_normal(
    curve_mask: np.ndarray,
    domain_mask: np.ndarray,
    corner_mode: str,
    extend_pixels: int,
) -> np.ndarray:
    curve = np.asarray(curve_mask, dtype=bool) & np.asarray(domain_mask, dtype=bool)
    if curve.sum() == 0 or extend_pixels <= 0:
        return curve

    extended = curve.copy()
    endpoints = select_curve_endpoints(curve)
    for endpoint in endpoints:
        tangent = estimate_endpoint_tangent(curve, endpoint)
        normal = desired_endpoint_normal(tangent, corner_mode)
        target = np.asarray(endpoint, dtype=float) + normal * float(extend_pixels)
        draw_grid_line(extended, endpoint, target)

    return extended & domain_mask


def build_skeleton_barrier_from_raw(
    raw_barrier_mask: np.ndarray,
    domain_mask: np.ndarray,
    corner_mode: str,
    endpoint_extend_pixels: int,
):
    """
    Convert a thick/noisy per-layer wall point mask into a curve barrier.

    Returns:
        barrier_mask: a thin blocking wall used by flood fill.
        curve_mask: one-pixel skeleton curve saved for debugging/inspection.
        fix_wall_mask: original skeleton wall used for fix point extraction.
    """
    raw = np.asarray(raw_barrier_mask, dtype=bool) & np.asarray(domain_mask, dtype=bool)
    if raw.sum() == 0:
        empty = np.zeros_like(raw, dtype=bool)
        return empty, empty, empty

    structure = np.ones((3, 3), dtype=bool)
    wall_band = ndimage.binary_closing(raw, structure=structure, iterations=1)
    wall_band = bridge_wall_components(wall_band, domain_mask)
    wall_band = ndimage.binary_closing(wall_band, structure=structure, iterations=1)
    wall_band = wall_band & domain_mask
    if wall_band.sum() == 0:
        wall_band = keep_largest_component(raw, min_area=1)

    curve_mask = skeletonize_zhang_suen(wall_band)
    curve_mask = curve_mask & domain_mask
    if curve_mask.sum() == 0:
        curve_mask = wall_band

    skeleton_mask = curve_mask.copy()
    extended_skeleton_mask = extend_skeleton_endpoints_by_normal(
        skeleton_mask,
        domain_mask,
        corner_mode,
        endpoint_extend_pixels,
    )
    barrier_mask = ndimage.binary_dilation(extended_skeleton_mask, structure=structure, iterations=1)
    barrier_mask = barrier_mask & domain_mask
    fix_wall_mask = ndimage.binary_dilation(skeleton_mask, structure=structure, iterations=1)
    fix_wall_mask = fix_wall_mask & domain_mask
    return barrier_mask, skeleton_mask, fix_wall_mask


def extract_first_contact_curve_mask(
    repair_mask: np.ndarray,
    barrier_mask: np.ndarray,
    corner_mode: str,
) -> np.ndarray:
    """
    Keep one flood-fill contact curve per layer.

    The curve is the first ring outside the flood-fill repair area that touches
    the processed barrier. This avoids saving the whole thick barrier band.
    """
    _ = corner_mode
    repair_mask = repair_mask.astype(bool)
    repair_front = ndimage.binary_dilation(repair_mask, iterations=1) & (~repair_mask)
    return repair_front & barrier_mask.astype(bool)


def get_seed_cell(corner_mode: str, H: int, W: int, domain_mask: np.ndarray):
    if corner_mode == "max_u_min_v":
        target = np.array([0, W - 1])
    elif corner_mode == "max_u_max_v":
        target = np.array([H - 1, W - 1])
    else:
        raise ValueError(f"Unsupported corner_mode: {corner_mode}")

    if domain_mask[target[0], target[1]]:
        return tuple(target)

    ys, xs = np.where(domain_mask)
    coords = np.column_stack([ys, xs])
    d = np.linalg.norm(coords - target[None, :], axis=1)
    return tuple(coords[np.argmin(d)])


def flood_fill_from_corner(barrier_mask: np.ndarray, domain_mask: np.ndarray, seed):
    allowed = domain_mask & (~barrier_mask)
    if not allowed[seed]:
        ys, xs = np.where(allowed)
        if len(xs) == 0:
            return np.zeros_like(domain_mask, dtype=bool)
        coords = np.column_stack([ys, xs])
        d = np.linalg.norm(coords - np.array(seed)[None, :], axis=1)
        seed = tuple(coords[np.argmin(d)])

    seed_mask = np.zeros_like(domain_mask, dtype=bool)
    seed_mask[seed] = True
    return ndimage.binary_propagation(seed_mask, mask=allowed)


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


def get_side_normal(corner_mode: str, u_axis: np.ndarray, v_axis: np.ndarray):
    side_n_mark = {}
    if corner_mode == "max_u_max_v":
        side_n_mark["max_u_outward"] = u_axis
        side_n_mark["max_v_outward"] = v_axis
    elif corner_mode == "max_u_min_v":
        side_n_mark["max_u_outward"] = u_axis
        side_n_mark["min_v_outward"] = -v_axis
    else:
        raise ValueError(f"Unsupported corner_mode: {corner_mode}")
    return side_n_mark


def extract_side_plane_points(
    repair_points: np.ndarray,
    origin: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    n_axis: np.ndarray,
    corner_mode: str,
    u_min: float,
    u_max: float,
    v_min: float,
    v_max: float,
    side_tol: float,
):
    repair_local = project_to_uvz(repair_points, origin, u_axis, v_axis, n_axis)
    repair_u = repair_local[:, 0]
    repair_v = repair_local[:, 1]

    u_side, v_side = get_target_sides_from_corner_mode(corner_mode)
    target_sides = [u_side, v_side]

    boundary_dict = {
        "min_u": u_min,
        "max_u": u_max,
        "min_v": v_min,
        "max_v": v_max,
    }
    coord_dict = {
        "min_u": repair_u,
        "max_u": repair_u,
        "min_v": repair_v,
        "max_v": repair_v,
    }

    side_points = {}
    for side_name in target_sides:
        boundary = boundary_dict[side_name]
        coord = coord_dict[side_name]
        mask = np.abs(coord - boundary) <= side_tol
        side_points[side_name] = repair_points[mask]
    return side_points


def filter_repair_layers_by_column_depth(repair_layers: List[dict], min_column_layers: int):
    if len(repair_layers) == 0:
        return repair_layers, None, None

    layer_stack = np.stack([layer["mask"].astype(bool) for layer in repair_layers], axis=0)
    column_count_map = layer_stack.sum(axis=0)
    column_keep_mask = column_count_map >= min_column_layers

    filtered_layers = []
    before_points = 0
    after_points = 0
    for layer in repair_layers:
        new_layer = dict(layer)
        old_mask = layer["mask"].astype(bool)
        before_points += int(old_mask.sum())
        new_mask = old_mask & column_keep_mask
        new_layer["mask"] = new_mask
        after_points += int(new_mask.sum())
        if new_mask.sum() > 0:
            filtered_layers.append(new_layer)

    return filtered_layers, column_keep_mask, column_count_map


def filter_repair_layers_by_planar_width(repair_layers: List[dict], min_width_pixels: int):
    """
    Remove thin planar tails inside each flood-fill layer.

    Column-depth filtering removes uv columns with too few z layers, but a long
    thin tail can still survive if it exists in many layers. This filter works
    per 2D layer: a small opening removes strips narrower than min_width_pixels,
    then the largest remaining connected region is kept.
    """
    min_width_pixels = int(max(1, min_width_pixels))
    structure = np.ones((min_width_pixels, min_width_pixels), dtype=bool)

    filtered_layers = []
    before_points = 0
    after_points = 0

    for layer in repair_layers:
        new_layer = dict(layer)
        old_mask = layer["mask"].astype(bool)
        before_points += int(old_mask.sum())

        new_mask = ndimage.binary_opening(old_mask, structure=structure)
        if new_mask.sum() > 0:
            new_mask = keep_largest_component(new_mask, min_area=1)

        new_layer["mask"] = new_mask
        after_points += int(new_mask.sum())

        if new_mask.sum() > 0:
            filtered_layers.append(new_layer)


    return filtered_layers


def filter_repair_layers_by_uv_column_support(repair_layers: List[dict], min_points_per_line: int = 3):
    """
    Remove whole u/v grid lines that are never sufficiently supported in any layer.

    For each fixed u index, count points along v in every z layer. If no layer has
    more than min_points_per_line points on that u line, remove that u line from
    all layers. Do the same for each fixed v index along u.
    """
    if len(repair_layers) == 0:
        return repair_layers, None, None

    min_points_per_line = int(max(0, min_points_per_line))
    layer_stack = np.stack([layer["mask"].astype(bool) for layer in repair_layers], axis=0)

    # mask shape per layer is (H, W): rows are v bins, columns are u bins.
    u_line_counts = layer_stack.sum(axis=1)  # (layers, W), fixed u, count along v.
    v_line_counts = layer_stack.sum(axis=2)  # (layers, H), fixed v, count along u.

    u_keep = (u_line_counts > min_points_per_line).any(axis=0)
    v_keep = (v_line_counts > min_points_per_line).any(axis=0)
    uv_keep_mask = v_keep[:, None] & u_keep[None, :]

    filtered_layers = []
    before_points = 0
    after_points = 0

    for layer in repair_layers:
        new_layer = dict(layer)
        old_mask = layer["mask"].astype(bool)
        before_points += int(old_mask.sum())

        new_mask = old_mask & uv_keep_mask
        new_layer["mask"] = new_mask
        after_points += int(new_mask.sum())

        if new_mask.sum() > 0:
            filtered_layers.append(new_layer)

    print("\n========== UV column support filter ==========")
    print("min_points_per_line:", min_points_per_line)
    print("repair points before filter:", before_points)
    print("repair points after filter:", after_points)
    print("kept u lines:", int(u_keep.sum()), "/", int(len(u_keep)))
    print("kept v lines:", int(v_keep.sum()), "/", int(len(v_keep)))

    return filtered_layers, u_keep, v_keep


# ============================================================
# Pipeline steps
# ============================================================
def load_and_prepare_plane(cfg: CompletionConfig) -> PlaneData:
    pcd_raw = o3d.io.read_point_cloud(cfg.fine_pcd_path)
    if len(pcd_raw.points) == 0:
        raise RuntimeError(f"Empty point cloud: {cfg.fine_pcd_path}")

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

    corner_roi = build_corner_roi(
        H=H,
        W=W,
        mode=corner_mode,
        frac_u=roi.roi_frac_u,
        frac_v=roi.roi_frac_v,
    )
    domain_mask = corner_roi.copy()

    defect_mask = domain_mask & (~top_occ_smooth)
    defect_mask = ndimage.binary_closing(defect_mask, iterations=cfg.defect_close_iter)
    defect_mask = keep_largest_component(defect_mask, min_area=cfg.min_component_area_pixels)

    if defect_mask.sum() == 0:
        raise RuntimeError("No defect_mask generated. Check boundary turn detection or corner_mode.")

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


def extract_defect_surface_points(cfg: CompletionConfig, plane: PlaneData, grid: GridData):
    rest_points = np.asarray(plane.rest_pcd.points)
    rest_local = project_to_uvz(rest_points, plane.origin, plane.u_axis, plane.v_axis, plane.n_axis)
    rest_uv = rest_local[:, :2]
    rest_z = rest_local[:, 2]

    xs, ys, valid_grid = uv_to_grid_index(
        rest_uv,
        grid.u_min,
        grid.v_min,
        cfg.grid_res,
        grid.W,
        grid.H,
    )

    valid_domain = np.zeros(len(rest_points), dtype=bool)
    valid_domain[valid_grid] = grid.domain_mask[ys[valid_grid], xs[valid_grid]]

    defect_surface_mask = valid_grid & valid_domain & (rest_z > -0.003)
    return (
        rest_points[defect_surface_mask],
        rest_uv[defect_surface_mask],
        rest_z[defect_surface_mask],
    )


def run_layered_flood_fill(
    cfg: CompletionConfig,
    plane: PlaneData,
    grid: GridData,
    corner_mode: str,
) -> CompletionData:
    defect_surface_points, defect_surface_uv, defect_surface_z = extract_defect_surface_points(cfg, plane, grid)

    domain_mask = grid.domain_mask
    domain_area = domain_mask.sum()
    seed = get_seed_cell(corner_mode, grid.H, grid.W, domain_mask)

    repair_layers = [{
        "z": 0.0,
        "mask": grid.defect_mask.copy(),
        "type": "top_defect_mask",
        "point_num": -1,
        "area_ratio": grid.defect_mask.sum() / (domain_area + 1e-12),
    }]

    bad_barrier_count = 0
    low_points_count = 0
    debug_records = []

    barrier_raw_masks = []
    barrier_curve_masks = []
    barrier_processed_masks = []
    fix_surface_masks = []
    barrier_z_values = []
    barrier_valid_flags = []
    barrier_area_ratios = []
    barrier_slice_nums = []
    fix_points_all = []
    fix_points_curve_all = []
    fix_surface_curve_masks = []

    z = cfg.layer_step
    while z <= cfg.max_search_depth:
        slice_mask = np.abs(defect_surface_z - z) <= cfg.band_width
        slice_uv = defect_surface_uv[slice_mask]
        slice_points = defect_surface_points[slice_mask]
        slice_num = len(slice_uv)
        low_points = slice_num < cfg.thres_points_num

        if low_points:
            low_points_count += 1
        else:
            low_points_count = 0

        valid_barrier = False
        repair_mask_z = None
        area_ratio = np.nan

        if not low_points:
            sx, sy, svalid = uv_to_grid_index(
                slice_uv,
                grid.u_min,
                grid.v_min,
                cfg.grid_res,
                grid.W,
                grid.H,
            )

            raw_barrier_mask = np.zeros((grid.H, grid.W), dtype=bool)
            raw_barrier_mask[sy[svalid], sx[svalid]] = True
            raw_barrier_mask = raw_barrier_mask & domain_mask

            barrier_mask, barrier_curve_mask, fix_wall_mask = build_skeleton_barrier_from_raw(
                raw_barrier_mask,
                domain_mask,
                corner_mode,
                cfg.wall_endpoint_extend_pixels,
            )

            repair_mask_z = flood_fill_from_corner(barrier_mask, domain_mask, seed)
            repair_area = repair_mask_z.sum()
            area_ratio = repair_area / (domain_area + 1e-12)
            valid_barrier = (repair_area >= cfg.min_area_pixels) and (area_ratio < cfg.max_area_ratio)

            fix_surface_mask = fix_wall_mask & ndimage.binary_dilation(repair_mask_z, iterations=2)
            fix_surface_mask = fix_surface_mask & domain_mask

            barrier_raw_masks.append(raw_barrier_mask.copy())
            barrier_curve_masks.append(barrier_curve_mask.copy())
            barrier_processed_masks.append(barrier_mask.copy())
            fix_surface_masks.append(fix_surface_mask.copy())
            barrier_z_values.append(z)
            barrier_valid_flags.append(valid_barrier)
            barrier_area_ratios.append(area_ratio)
            barrier_slice_nums.append(slice_num)

            fix_uv = mask_to_uv_points(fix_surface_mask, grid.u_min, grid.v_min, cfg.grid_res)
            if len(fix_uv) > 0:
                fix_z_arr = np.full(len(fix_uv), z)
                fix_pts = uvz_to_3d(fix_uv, fix_z_arr, plane.origin, plane.u_axis, plane.v_axis, plane.n_axis)
                fix_points_all.append(fix_pts)

        if valid_barrier:
            bad_barrier_count = 0
            fix_surface_curve_mask = extract_first_contact_curve_mask(
                repair_mask_z,
                fix_wall_mask,
                corner_mode,
            )
            fix_surface_curve_masks.append(fix_surface_curve_mask.copy())
            curve_uv = mask_to_uv_points(fix_surface_curve_mask, grid.u_min, grid.v_min, cfg.grid_res)
            if len(curve_uv) > 0:
                curve_z_arr = np.full(len(curve_uv), z)
                curve_pts = uvz_to_3d(
                    curve_uv,
                    curve_z_arr,
                    plane.origin,
                    plane.u_axis,
                    plane.v_axis,
                    plane.n_axis,
                )
                fix_points_curve_all.append(curve_pts)
            repair_layers.append({
                "z": z,
                "mask": repair_mask_z.copy(),
                "type": "slice_floodfill",
                "point_num": slice_num,
                "area_ratio": area_ratio,
            })
            print(f"[有效] z={z:.4f} m, slice points={slice_num}, area ratio={area_ratio:.3f}")
        else:
            bad_barrier_count += 1
            print(f"[无效] z={z:.4f} m, slice points={slice_num}, low_points={low_points}, area ratio={area_ratio}")

        debug_records.append({
            "z": z,
            "slice_num": slice_num,
            "low_points": low_points,
            "valid_barrier": valid_barrier,
            "area_ratio": area_ratio,

        })

        if bad_barrier_count >= cfg.max_bad_layers:
            print(f"\n停止：连续 {cfg.max_bad_layers} 层无法形成有效阻挡")
            break
        if low_points_count >= cfg.max_bad_layers:
            print(f"\n停止：连续 {cfg.max_bad_layers} 层缺陷点数量少于 {cfg.thres_points_num}")
            break

        z += cfg.layer_step

    column_keep_mask = np.ones_like(grid.domain_mask, dtype=bool)
    column_count_map = np.zeros_like(grid.domain_mask, dtype=np.int32)

    repair_layers, column_keep_mask, column_count_map = filter_repair_layers_by_column_depth(
        repair_layers,
        min_column_layers=cfg.min_column_layers,
    )
    if len(repair_layers) == 0:
        raise RuntimeError("No repair layers left after column depth filter. Try reducing cfg.min_column_layers.")

    repair_layers, uv_column_u_keep, uv_column_v_keep = filter_repair_layers_by_uv_column_support(
        repair_layers,
        min_points_per_line=3,
    )
    if len(repair_layers) == 0:
        raise RuntimeError("No repair layers left after UV column support filter.")

    all_layer_points = []
    for layer in repair_layers:
        uv_layer = mask_to_uv_points(layer["mask"], grid.u_min, grid.v_min, cfg.grid_res)
        if len(uv_layer) == 0:
            continue
        z_arr = np.full(len(uv_layer), layer["z"])
        pts_3d = uvz_to_3d(uv_layer, z_arr, plane.origin, plane.u_axis, plane.v_axis, plane.n_axis)
        all_layer_points.append(pts_3d)

    if len(all_layer_points) == 0:
        raise RuntimeError("No repair points generated.")

    repair_volume_points = np.vstack(all_layer_points)
    repair_point_center = repair_volume_points.mean(axis=0)

    side_points = extract_side_plane_points(
        repair_points=repair_volume_points,
        origin=plane.origin,
        u_axis=plane.u_axis,
        v_axis=plane.v_axis,
        n_axis=plane.n_axis,
        corner_mode=corner_mode,
        u_min=grid.u_min,
        u_max=grid.u_max,
        v_min=grid.v_min,
        v_max=grid.v_max,
        side_tol=cfg.layer_step,
    )
    side_n_mark = get_side_normal(corner_mode, plane.u_axis, plane.v_axis)

    fix_points = np.vstack(fix_points_all) if len(fix_points_all) > 0 else np.empty((0, 3))

    if len(fix_surface_masks) > 0:
        barrier_raw_masks_arr = np.asarray(barrier_raw_masks, dtype=bool)
        barrier_curve_masks_arr = np.asarray(barrier_curve_masks, dtype=bool)
        barrier_processed_masks_arr = np.asarray(barrier_processed_masks, dtype=bool)
        fix_surface_masks_arr = np.asarray(fix_surface_masks, dtype=bool)
    else:
        barrier_raw_masks_arr = np.zeros((0, grid.H, grid.W), dtype=bool)
        barrier_curve_masks_arr = np.zeros((0, grid.H, grid.W), dtype=bool)
        barrier_processed_masks_arr = np.zeros((0, grid.H, grid.W), dtype=bool)
        fix_surface_masks_arr = np.zeros((0, grid.H, grid.W), dtype=bool)
    if len(fix_surface_curve_masks) > 0:
        fix_surface_curve_masks_arr = np.asarray(fix_surface_curve_masks, dtype=bool)
    else:
        fix_surface_curve_masks_arr = np.zeros((0, grid.H, grid.W), dtype=bool)

    fix_points_curve = (
        np.vstack(fix_points_curve_all)
        if len(fix_points_curve_all) > 0
        else np.empty((0, 3))
    )

    fix_npz_data = {
        "barrier_raw_masks": barrier_raw_masks_arr,
        "barrier_curve_masks": barrier_curve_masks_arr,
        "barrier_processed_masks": barrier_processed_masks_arr,
        "fix_surface_masks": fix_surface_masks_arr,
        "fix_surface_curve_masks": fix_surface_curve_masks_arr,
        "barrier_z_values": np.asarray(barrier_z_values),
        "barrier_valid_flags": np.asarray(barrier_valid_flags, dtype=bool),
        "barrier_area_ratios": np.asarray(barrier_area_ratios),
        "barrier_slice_nums": np.asarray(barrier_slice_nums),
        "grid_res": np.asarray(cfg.grid_res),
        "u_min": np.asarray(grid.u_min),
        "v_min": np.asarray(grid.v_min),
        "u_max": np.asarray(grid.u_max),
        "v_max": np.asarray(grid.v_max),
        "H": np.asarray(grid.H),
        "W": np.asarray(grid.W),
        "origin": plane.origin,
        "u_axis": plane.u_axis,
        "v_axis": plane.v_axis,
        "n_axis": plane.n_axis,
        "domain_mask": grid.domain_mask,
        "corner_roi": grid.corner_roi,
        "top_defect_mask": grid.defect_mask,
        "column_keep_mask": column_keep_mask,
        "column_count_map": column_count_map,
        "min_column_layers": np.asarray(cfg.min_column_layers),
        "uv_column_min_points_per_line": np.asarray(3),
        "uv_column_u_keep": np.asarray(uv_column_u_keep, dtype=bool),
        "uv_column_v_keep": np.asarray(uv_column_v_keep, dtype=bool),
        "wall_endpoint_extend_pixels": np.asarray(cfg.wall_endpoint_extend_pixels),
    }

    print("\n有效层数:", len(repair_layers))
    print("补全体点云数量:", len(repair_volume_points))
    print("最大有效深度:", max([layer["z"] for layer in repair_layers]), "m")
    print("缺陷凹陷面候选点数量:", len(defect_surface_points))

    return CompletionData(
        repair_volume_points=repair_volume_points,
        repair_point_center=repair_point_center,
        repair_layers=repair_layers,
        side_points=side_points,
        side_n_mark=side_n_mark,
        defect_surface_points=defect_surface_points,
        fix_points=fix_points,
        fix_points_curve=fix_points_curve,
        fix_npz_data=fix_npz_data,
        debug_records=debug_records,
    )


def estimate_defect_world_y(top_defect_margin_points: np.ndarray) -> str:
    left_count = np.sum(top_defect_margin_points[:, 1] > 0)
    right_count = np.sum(top_defect_margin_points[:, 1] < 0)
    return "left" if left_count > right_count else "right"


def save_results(
    cfg: CompletionConfig,
    plane: PlaneData,
    roi: RoiData,
    grid: GridData,
    completion: CompletionData,
    corner_mode: str,
):
    os.makedirs(cfg.output_dir, exist_ok=True)

    repair_model_pcd = make_colored_pcd(completion.repair_volume_points, [1.0, 0.0, 0.0])
    top_plane_pcd = o3d.geometry.PointCloud(plane.plane_pcd)
    top_plane_pcd.paint_uniform_color([0.6, 0.6, 0.6])
    top_margin_pcd = make_colored_pcd(grid.top_defect_margin_points, [1.0, 0.0, 0.0])
    fix_pcd = make_colored_pcd(completion.fix_points, [1.0, 0.5, 0.0])
    fix_curve_pcd = make_colored_pcd(completion.fix_points_curve, [1.0, 0.0, 0.0])

    u_side_name, v_side_name = get_target_sides_from_corner_mode(corner_mode)
    u_side_pcd = make_colored_pcd(completion.side_points.get(u_side_name, np.empty((0, 3))), [0.0, 1.0, 0.0])
    v_side_pcd = make_colored_pcd(completion.side_points.get(v_side_name, np.empty((0, 3))), [0.0, 0.0, 1.0])

    o3d.io.write_point_cloud(os.path.join(cfg.output_dir, "model.pcd"), repair_model_pcd)
    o3d.io.write_point_cloud(os.path.join(cfg.output_dir, "u_side_plane.pcd"), u_side_pcd)
    o3d.io.write_point_cloud(os.path.join(cfg.output_dir, "v_side_plane.pcd"), v_side_pcd)
    o3d.io.write_point_cloud(os.path.join(cfg.output_dir, "top_plane.pcd"), top_plane_pcd)
    o3d.io.write_point_cloud(os.path.join(cfg.output_dir, "top_defect_margin.pcd"), top_margin_pcd)
    o3d.io.write_point_cloud(os.path.join(cfg.output_dir, "fix_points.pcd"), fix_pcd)
    o3d.io.write_point_cloud(os.path.join(cfg.output_dir, "fix_points_curve.pcd"), fix_curve_pcd)

    defect_world_y = estimate_defect_world_y(grid.top_defect_margin_points)

    np.savez(
        os.path.join(cfg.output_dir, "meta.npz"),
        side_n_mark=completion.side_n_mark,
        repair_point_center=completion.repair_point_center,
        n_axis=plane.n_axis,
        u_axis=plane.u_axis,
        v_axis=plane.v_axis,
        top_plane_model=plane.plane_model,
        top_plane_center=grid.top_plane_center,
        defect_world_y=defect_world_y,
        corner_mode=corner_mode,
        roi_frac_u=np.asarray(roi.roi_frac_u),
        roi_frac_v=np.asarray(roi.roi_frac_v),
        turn_point_u=roi.turn_point_u,
        turn_point_v=roi.turn_point_v,
        u_side=roi.u_side,
        v_side=roi.v_side,
    )

    np.savez(
        os.path.join(cfg.output_dir, "top_defect_margin.npz"),
        top_defect_margin_points=grid.top_defect_margin_points,
        top_defect_margin_uv=grid.top_defect_margin_uv,
    )

    np.savez(os.path.join(cfg.output_dir, "fix_mask.npz"), **completion.fix_npz_data)

    np.savez(
        os.path.join(cfg.output_dir, "debug_masks.npz"),
        top_occ_mask=grid.top_occ_mask,
        top_occ_smooth=grid.top_occ_smooth,
        corner_roi=grid.corner_roi,
        domain_mask=grid.domain_mask,
        defect_mask=grid.defect_mask,
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
    )

    print("\n========== depression_completion_twoFit done ==========")
    print("corner_mode:", corner_mode)
    print("output_dir:", cfg.output_dir)
    print("saved: model.pcd, u_side_plane.pcd, v_side_plane.pcd")
    print("saved: top_plane.pcd, top_defect_margin.pcd")
    print("saved: meta.npz, top_defect_margin.npz")
    print("saved: fix_mask.npz, fix_points.pcd, debug_masks.npz")


def visualize_results(cfg: CompletionConfig, plane: PlaneData, completion: CompletionData):
    if not cfg.visualize:
        return
    repair_model_pcd = make_colored_pcd(completion.repair_volume_points, [1.0, 0.0, 0.0])
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
    o3d.visualization.draw_geometries([plane.pcd_raw, repair_model_pcd, frame])
    o3d.visualization.draw_geometries([repair_model_pcd])



def run_completion(cfg: CompletionConfig):
    corner_mode = resolve_corner_mode(cfg)
    plane = load_and_prepare_plane(cfg)
    roi = estimate_roi_from_boundary(cfg, plane, corner_mode)
    grid = build_grid_and_defect_masks(cfg, plane, roi, corner_mode)
    completion = run_layered_flood_fill(cfg, plane, grid, corner_mode)
    save_results(cfg, plane, roi, grid, completion, corner_mode)
    visualize_results(cfg, plane, completion)


# ============================================================
# CLI
# ============================================================
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


if __name__ == "__main__":
    run_completion(config_from_args(parse_args()))

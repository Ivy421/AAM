"""
depression_completion_boundary.py

支持的 CORNER_MODE:
    - max_u_max_v
    - max_u_min_v

主流程：
    1. 读取 fine_fuse.pcd，并只拟合一次顶面平面。
    2. 基于顶面平面建立局部 uvz 坐标系。
    3. 将 rest_pcd 投影到 uvz 坐标系，并提取靠近顶面的滑坡/侧棱点。
    4. 将 top_uv 和 near_top_rest_uv 合并，作为边界拟合的支持点 boundary_support_uv。
    5. 根据 corner_mode，自适应选取远离缺陷角的边界候选点：30-90%。
    6. 使用 RANSAC 拟合 max_u 边界线，以及 max_v / min_v 边界线。
    7. 由两条边界线外推得到理想缺陷角点 ideal corner。
    8. 根据两条拟合边界线生成理想平面区域 ideal_inner_mask。
    9. 在 ideal_inner_mask 中寻找最靠近 ideal corner 的缺失顶面连通域。
    10. 将该缺失连通域的 bbox 外扩，生成干净连续的局部补全区域 domain_mask。
    11. 使用逐层 barrier + flood fill 方法生成补全体点云。
    12. 后处理：按 uv 列统计 z 方向层数，删除层数少于 min_column_layers 的列。
    13. 保存 repair model、side planes、top margin，以及 fix/contact masks。

    """

import os
import json
import argparse
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import open3d as o3d
from scipy import ndimage

# Keep RANSAC deterministic in Open3D.
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
    plane_voxel_size: float = 0.001
    plane_distance_threshold: float = 0.003
    plane_num_iterations: int = 3000

    # UV grid.
    grid_res: float = 0.0008
    grid_pad: float = 0.006

    # Boundary candidate extraction and RANSAC fitting.
    boundary_bin_size: float = 0.001
    boundary_min_bin_points: int = 5
    # Boundary points used for RANSAC are selected by normalized distance
    # from the damaged corner along each fitted edge.
    # 0.0 = closest to damaged corner, 1.0 = farthest from damaged corner.
    # Keep the middle-far segment [0.30, 0.90].
    boundary_away_low: float = 0.30
    boundary_away_high: float = 0.90

    # Boundary support enhancement for ICP sliding/sloped edge.
    # Top plane is still used for coordinate frame and top occupancy mask.
    # Boundary fitting uses top_uv + near-top rest_pcd points in this z range.
    use_rest_points_for_boundary: bool = True
    boundary_support_z_min: float = -0.003
    boundary_support_z_max: float = 0.01

    ransac_trials: int = 500
    ransac_residual_thresh: float = 0.002
    ransac_min_inliers: int = 20
    boundary_line_margin: float = 0.002  #0.002

    # Top occupancy smoothing.
    top_occ_close_iter: int = 2
    top_occ_dilate_iter: int = 0 # 1
    defect_close_iter: int = 2
    min_component_area_pixels: int = 30
    bbox_margin: float = 0.015 #0.008

    # Flood fill / layered completion.
    layer_step: float = 0.001
    band_width: float = 0.0007
    thres_points_num: int = 50
    max_bad_layers: int = 5
    barrier_dilate_iter: int = 4
    barrier_close_iter: int = 2
    max_area_ratio: float = 0.60
    min_area_pixels: int = 20
    max_search_depth: float = 0.150

    # Column depth post-filter.
    enable_column_depth_filter: bool = True
    min_column_layers: int = 5
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
class BoundaryData:
    # max_u boundary: u = a * v + b
    max_u_line: Tuple[float, float]
    max_u_inliers: np.ndarray
    max_u_candidates: np.ndarray

    # paired v boundary: v = a * u + b; side is max_v or min_v.
    edge_v_line: Tuple[float, float]
    edge_v_side: str
    edge_v_inliers: np.ndarray
    edge_v_candidates: np.ndarray

    # Boundary fitting support data.
    # boundary_support_uv = top_uv + near_top_rest_uv.
    # near_top_rest_points keeps original 3D sloped/rest points for debug visualization.
    boundary_support_uv: np.ndarray
    near_top_rest_uv: np.ndarray
    near_top_rest_points: np.ndarray

    ideal_corner_uv: np.ndarray
    ideal_inner_mask: Optional[np.ndarray] = None


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
    ideal_inner_mask: np.ndarray
    defect_candidate: np.ndarray
    defect_component: np.ndarray
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
    raw_fix_surface_points: np.ndarray
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
        raise ValueError(f"Only {allowed} are supported, got: {corner_mode}")
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


def make_uv_line_set(
    uv_points: np.ndarray,
    origin: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    color,
):
    line = o3d.geometry.LineSet()
    uv_points = np.asarray(uv_points, dtype=float)
    if len(uv_points) < 2:
        return line

    pts_3d = uv_to_3d(uv_points, origin, u_axis, v_axis)
    lines = [[i, i + 1] for i in range(len(pts_3d) - 1)]
    line.points = o3d.utility.Vector3dVector(pts_3d)
    line.lines = o3d.utility.Vector2iVector(lines)
    line.colors = o3d.utility.Vector3dVector(np.tile(np.asarray(color, dtype=float), (len(lines), 1)))
    return line


def make_sphere(center: np.ndarray, radius: float, color):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(np.asarray(center, dtype=float).reshape(3))
    sphere.paint_uniform_color(color)
    return sphere


# ============================================================
# Boundary extraction and robust line fitting
# ============================================================
def extract_binned_boundary_points(
    uv: np.ndarray,
    side: str,
    bin_size: float,
    min_bin_points: int,
) -> np.ndarray:
    pts = []

    if side in ["u_min", "u_max"]:
        coord = uv[:, 1]  # bin along v
        bins = np.arange(coord.min(), coord.max() + bin_size, bin_size)
        for i in range(len(bins) - 1):
            mask = (coord >= bins[i]) & (coord < bins[i + 1])
            bin_pts = uv[mask]
            if len(bin_pts) < min_bin_points:
                continue
            idx = np.argmin(bin_pts[:, 0]) if side == "u_min" else np.argmax(bin_pts[:, 0])
            pts.append(bin_pts[idx])

    elif side in ["min_v", "max_v"]:
        coord = uv[:, 0]  # bin along u
        bins = np.arange(coord.min(), coord.max() + bin_size, bin_size)
        for i in range(len(bins) - 1):
            mask = (coord >= bins[i]) & (coord < bins[i + 1])
            bin_pts = uv[mask]
            if len(bin_pts) < min_bin_points:
                continue
            idx = np.argmin(bin_pts[:, 1]) if side == "min_v" else np.argmax(bin_pts[:, 1])
            pts.append(bin_pts[idx])

    else:
        raise ValueError(f"Unknown side: {side}")

    return np.asarray(pts, dtype=float)


def ransac_fit_y_from_x(
    x: np.ndarray,
    y: np.ndarray,
    residual_thresh: float,
    trials: int,
    min_inliers: int,
    random_seed: int = 274,
):
    """
    Fit y = a*x + b using simple RANSAC.
    Returns: (a, b), inlier_mask
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    n = len(x)
    if n < 2:
        raise RuntimeError("Not enough points for line fitting.")

    rng = np.random.default_rng(random_seed)
    best_inliers = None
    best_count = -1
    best_error = np.inf

    for _ in range(trials):
        ids = rng.choice(n, size=2, replace=False)
        dx = x[ids[1]] - x[ids[0]]
        if abs(dx) < 1e-12:
            continue
        a = (y[ids[1]] - y[ids[0]]) / dx
        b = y[ids[0]] - a * x[ids[0]]
        residual = np.abs(y - (a * x + b))
        inliers = residual <= residual_thresh
        count = int(inliers.sum())
        error = float(np.mean(residual[inliers])) if count > 0 else np.inf
        if (count > best_count) or (count == best_count and error < best_error):
            best_count = count
            best_error = error
            best_inliers = inliers

    if best_inliers is None or best_count < max(2, min_inliers):
        # Fallback to all points if RANSAC cannot find enough inliers.
        best_inliers = np.ones(n, dtype=bool)

    a, b = np.polyfit(x[best_inliers], y[best_inliers], deg=1)
    residual = np.abs(y - (a * x + b))
    inliers = residual <= residual_thresh
    if inliers.sum() >= max(2, min_inliers):
        a, b = np.polyfit(x[inliers], y[inliers], deg=1)
    else:
        inliers = best_inliers

    return (float(a), float(b)), inliers


def _select_candidate_range_away_from_defect(
    candidates: np.ndarray,
    coord_idx: int,
    defect_at_max_side: bool,
    away_low: float,
    away_high: float,
) -> np.ndarray:
    """
    Select boundary candidates by distance from the damaged corner.

    The selected range is defined on a normalized 1D distance along the boundary:
        0.0 = closest to damaged corner
        1.0 = farthest from damaged corner

    For example, away_low=0.30 and away_high=0.90 keeps the middle-far
    30%-90% boundary segment, and excludes both the damaged-corner end
    and the far endpoint, which is often noisy.
    """
    if len(candidates) == 0:
        return candidates

    if not (0.0 <= away_low < away_high <= 1.0):
        raise ValueError(
            f"Invalid boundary away range: [{away_low}, {away_high}]. "
            "Require 0 <= low < high <= 1."
        )

    coord = candidates[:, coord_idx]

    if defect_at_max_side:
        # Defect is at the larger coordinate end.
        # away=0.30~0.90 corresponds to coordinate percentile 10%~70%.
        q_low = 1.0 - away_high
        q_high = 1.0 - away_low
    else:
        # Defect is at the smaller coordinate end.
        # away=0.30~0.90 corresponds to coordinate percentile 30%~90%.
        q_low = away_low
        q_high = away_high

    c_low, c_high = np.quantile(coord, [q_low, q_high])
    mask = (coord >= c_low) & (coord <= c_high)
    filtered = candidates[mask]

    # If too aggressive, fallback to all candidates.
    if len(filtered) < 2:
        return candidates

    return filtered


def adaptive_far_filter_for_boundary(
    candidates: np.ndarray,
    top_uv: np.ndarray,
    corner_mode: str,
    boundary_side: str,
    away_low: float,
    away_high: float,
) -> np.ndarray:
    """
    Select the 30%-90% boundary segment away from the damaged corner.

    This function does not use a fixed physical ROI. It uses the extracted
    binned boundary candidates themselves and keeps only the requested
    normalized distance range from the defect corner.
    """
    if len(candidates) == 0:
        return candidates

    if boundary_side == "max_u":
        # max_u boundary is parameterized along v.
        # max_u_max_v: damaged end is max_v.
        # max_u_min_v: damaged end is min_v.
        if corner_mode == "max_u_max_v":
            return _select_candidate_range_away_from_defect(
                candidates, coord_idx=1, defect_at_max_side=True,
                away_low=away_low, away_high=away_high,
            )
        if corner_mode == "max_u_min_v":
            return _select_candidate_range_away_from_defect(
                candidates, coord_idx=1, defect_at_max_side=False,
                away_low=away_low, away_high=away_high,
            )
        raise ValueError(f"Unsupported corner_mode: {corner_mode}")

    if boundary_side in ["max_v", "min_v"]:
        # max_v/min_v boundary is parameterized along u.
        # Both supported corner modes are damaged at max_u.
        return _select_candidate_range_away_from_defect(
            candidates, coord_idx=0, defect_at_max_side=True,
            away_low=away_low, away_high=away_high,
        )

    raise ValueError(f"Unsupported boundary_side: {boundary_side}")


def extract_boundary_support_uv(cfg: CompletionConfig, plane: PlaneData):
    """
    Build boundary support uv points for RANSAC boundary fitting.

    Top plane inliers are still used for local frame and top occupancy mask.
    For boundary fitting, add near-top rest_pcd points because ICP fusion may create
    a sloped/slide-like edge whose outer boundary is not included in top plane inliers.

    rest_z is measured along plane.n_axis in the same uvz frame.
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


def fit_boundary_lines(cfg: CompletionConfig, plane: PlaneData, corner_mode: str) -> BoundaryData:
    top_uv = plane.top_uv
    boundary_support_uv, near_top_rest_uv, near_top_rest_points = extract_boundary_support_uv(cfg, plane)

    max_u_candidates_all = extract_binned_boundary_points(
        boundary_support_uv, "u_max", cfg.boundary_bin_size, cfg.boundary_min_bin_points
    )
    max_u_candidates = adaptive_far_filter_for_boundary(
        max_u_candidates_all,
        top_uv,
        corner_mode,
        "max_u",
        cfg.boundary_away_low,
        cfg.boundary_away_high,
    )
    # max_u line: u = a * v + b
    max_u_line, max_u_inliers = ransac_fit_y_from_x(
        x=max_u_candidates[:, 1],
        y=max_u_candidates[:, 0],
        residual_thresh=cfg.ransac_residual_thresh,
        trials=cfg.ransac_trials,
        min_inliers=cfg.ransac_min_inliers,
    )

    if corner_mode == "max_u_max_v":
        edge_v_side = "max_v"
    elif corner_mode == "max_u_min_v":
        edge_v_side = "min_v"
    else:
        raise ValueError(f"Unsupported corner_mode: {corner_mode}")

    edge_v_candidates_all = extract_binned_boundary_points(
        boundary_support_uv, edge_v_side, cfg.boundary_bin_size, cfg.boundary_min_bin_points
    )
    edge_v_candidates = adaptive_far_filter_for_boundary(
        edge_v_candidates_all,
        top_uv,
        corner_mode,
        edge_v_side,
        cfg.boundary_away_low,
        cfg.boundary_away_high,
    )
    # v-side line: v = a * u + b
    edge_v_line, edge_v_inliers = ransac_fit_y_from_x(
        x=edge_v_candidates[:, 0],
        y=edge_v_candidates[:, 1],
        residual_thresh=cfg.ransac_residual_thresh,
        trials=cfg.ransac_trials,
        min_inliers=cfg.ransac_min_inliers,
    )

    ideal_corner_uv = intersect_uv_lines(max_u_line, edge_v_line)

    print("\n========== Boundary fitting ==========")
    print("corner_mode:", corner_mode)
    print("max_u line: u = %.6f * v + %.6f" % max_u_line)
    print("%s line: v = %.6f * u + %.6f" % (edge_v_side, edge_v_line[0], edge_v_line[1]))
    print("ideal_corner_uv:", ideal_corner_uv)
    print("boundary away range:", (cfg.boundary_away_low, cfg.boundary_away_high))
    print("boundary support uv count:", len(boundary_support_uv))
    print("near-top rest support count:", len(near_top_rest_uv))
    print("support z range:", (cfg.boundary_support_z_min, cfg.boundary_support_z_max))
    print("max_u candidates/inliers:", len(max_u_candidates), int(max_u_inliers.sum()))
    print(f"{edge_v_side} candidates/inliers:", len(edge_v_candidates), int(edge_v_inliers.sum()))

    return BoundaryData(
        max_u_line=max_u_line,
        max_u_inliers=max_u_inliers,
        max_u_candidates=max_u_candidates,
        edge_v_line=edge_v_line,
        edge_v_side=edge_v_side,
        edge_v_inliers=edge_v_inliers,
        edge_v_candidates=edge_v_candidates,
        boundary_support_uv=boundary_support_uv,
        near_top_rest_uv=near_top_rest_uv,
        near_top_rest_points=near_top_rest_points,
        ideal_corner_uv=ideal_corner_uv,
    )


def intersect_uv_lines(max_u_line: Tuple[float, float], edge_v_line: Tuple[float, float]) -> np.ndarray:
    """
    max_u line: u = a1 * v + b1
    v-side line: v = a2 * u + b2
    """
    a1, b1 = max_u_line
    a2, b2 = edge_v_line
    denom = 1.0 - a1 * a2
    if abs(denom) < 1e-9:
        raise RuntimeError("Boundary lines are nearly parallel in uv equation form.")
    v = (a2 * b1 + b2) / denom
    u = a1 * v + b1
    return np.array([u, v], dtype=float)


# ============================================================
# Mask utilities
# ============================================================
def build_ideal_inner_mask_from_lines(
    u_grid: np.ndarray,
    v_grid: np.ndarray,
    boundary: BoundaryData,
    corner_mode: str,
    line_margin: float,
) -> np.ndarray:
    U, V = np.meshgrid(u_grid, v_grid)
    a_u, b_u = boundary.max_u_line
    a_v, b_v = boundary.edge_v_line

    u_edge = a_u * V + b_u
    v_edge = a_v * U + b_v

    # Inside object relative to fitted max_u side.
    inside_u = U <= (u_edge + line_margin)

    if corner_mode == "max_u_max_v":
        inside_v = V <= (v_edge + line_margin)
    elif corner_mode == "max_u_min_v":
        inside_v = V >= (v_edge - line_margin)
    else:
        raise ValueError(f"Unsupported corner_mode: {corner_mode}")

    return inside_u & inside_v


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


def uv_to_nearest_grid_cell(uv: np.ndarray, u_min: float, v_min: float, grid_res: float, W: int, H: int) -> np.ndarray:
    x = int(round((uv[0] - u_min) / grid_res))
    y = int(round((uv[1] - v_min) / grid_res))
    x = int(np.clip(x, 0, W - 1))
    y = int(np.clip(y, 0, H - 1))
    return np.array([y, x], dtype=int)


def select_component_closest_to_corner(
    defect_candidate: np.ndarray,
    corner_cell: np.ndarray,
    min_area: int,
) -> np.ndarray:
    label_mask, num = ndimage.label(defect_candidate)
    if num == 0:
        return np.zeros_like(defect_candidate, dtype=bool)

    labels = np.arange(1, num + 1)
    best_label = None
    best_dist = np.inf
    best_area = -1

    for label in labels:
        ys, xs = np.where(label_mask == label)
        area = len(xs)
        if area < min_area:
            continue
        coords = np.column_stack([ys, xs])
        d = np.min(np.linalg.norm(coords - corner_cell[None, :], axis=1))

        # Primary: closest to ideal corner. Secondary: larger area.
        if (d < best_dist) or (np.isclose(d, best_dist) and area > best_area):
            best_label = label
            best_dist = d
            best_area = area

    if best_label is None:
        return np.zeros_like(defect_candidate, dtype=bool)

    return label_mask == best_label


def bbox_mask_from_component(component_mask: np.ndarray, H: int, W: int, margin_pix: int) -> np.ndarray:
    ys, xs = np.where(component_mask)
    if len(xs) == 0:
        raise RuntimeError("No defect component found. Check boundary fitting or top_occ smoothing.")

    y0 = max(int(ys.min()) - margin_pix, 0)
    y1 = min(int(ys.max()) + margin_pix, H - 1)
    x0 = max(int(xs.min()) - margin_pix, 0)
    x1 = min(int(xs.max()) + margin_pix, W - 1)

    mask = np.zeros((H, W), dtype=bool)
    mask[y0:y1 + 1, x0:x1 + 1] = True
    return mask


def get_seed_cell_from_corner_uv(corner_uv: np.ndarray, grid: GridData, domain_mask: np.ndarray):
    target = uv_to_nearest_grid_cell(corner_uv, grid.u_min, grid.v_min, grid.u_grid[1] - grid.u_grid[0], grid.W, grid.H)

    if domain_mask[target[0], target[1]]:
        return tuple(target)

    ys, xs = np.where(domain_mask)
    if len(xs) == 0:
        raise RuntimeError("domain_mask is empty, cannot choose flood-fill seed.")

    coords = np.column_stack([ys, xs])
    d = np.linalg.norm(coords - target[None, :], axis=1)
    seed = coords[np.argmin(d)]
    return tuple(seed)


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


def get_target_sides_from_corner_mode(corner_mode: str):
    if corner_mode == "max_u_max_v":
        return ["max_u", "max_v"]
    if corner_mode == "max_u_min_v":
        return ["max_u", "min_v"]
    raise ValueError(f"Unsupported corner_mode: {corner_mode}")


def get_side_normal(corner_mode: str, u_axis: np.ndarray, v_axis: np.ndarray):
    # Keep naming compatible with previous downstream code.
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
    u_min_domain: float,
    u_max_domain: float,
    v_min_domain: float,
    v_max_domain: float,
    side_tol: float,
):
    repair_local = project_to_uvz(repair_points, origin, u_axis, v_axis, n_axis)
    repair_u = repair_local[:, 0]
    repair_v = repair_local[:, 1]

    target_sides = get_target_sides_from_corner_mode(corner_mode)
    boundary_dict = {
        "min_u": u_min_domain,
        "max_u": u_max_domain,
        "min_v": v_min_domain,
        "max_v": v_max_domain,
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


def filter_repair_layers_by_column_depth(
    repair_layers: List[dict],
    min_column_layers: int = 5,
):
    """
    Filter generated repair volume by vertical column continuity.

    A column means the same uv grid cell through all z layers.
    If one column has fewer than min_column_layers occupied layers,
    remove all repair points in this uv column.

    This is useful for removing shallow overflow caused by boundary over-expansion.
    """
    if len(repair_layers) == 0:
        return repair_layers, None, None

    layer_stack = np.stack(
        [layer["mask"].astype(bool) for layer in repair_layers],
        axis=0,
    )  # shape = [num_layers, H, W]

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

    print("\n========== Column depth filter ==========")
    print("min_column_layers:", min_column_layers)
    print("repair points before filter:", before_points)
    print("repair points after filter:", after_points)
    print("kept columns:", int(column_keep_mask.sum()))

    return filtered_layers, column_keep_mask, column_count_map

# ============================================================
# Pipeline steps
# ============================================================
def load_and_prepare_plane(cfg: CompletionConfig) -> PlaneData:
    pcd_raw = o3d.io.read_point_cloud(cfg.fine_pcd_path)
    if len(pcd_raw.points) == 0:
        raise RuntimeError(f"Empty point cloud: {cfg.fine_pcd_path}")

    points_raw = np.asarray(pcd_raw.points)

    # Find plane only once, so top plane and rest cloud are consistent.
    plane_model, plane_pcd, rest_pcd = find_plane(
        pcd_raw,
        voxel_size=cfg.plane_voxel_size,
        distance_threshold=cfg.plane_distance_threshold,
        num_iterations=cfg.plane_num_iterations,
    )

    n_axis = plane_model[:3]
    inward_vec = np.asarray(cfg.inward_vec, dtype=float)
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
    boundary: BoundaryData,
    corner_mode: str,
) -> GridData:
    # Grid canvas must include top points and extrapolated ideal corner.
    top_uv = plane.top_uv
    corner_uv = boundary.ideal_corner_uv

    u_min = float(min(top_uv[:, 0].min(), corner_uv[0]) - cfg.grid_pad)
    u_max = float(max(top_uv[:, 0].max(), corner_uv[0]) + cfg.grid_pad)
    v_min = float(min(top_uv[:, 1].min(), corner_uv[1]) - cfg.grid_pad)
    v_max = float(max(top_uv[:, 1].max(), corner_uv[1]) + cfg.grid_pad)

    u_grid = np.arange(u_min, u_max + cfg.grid_res, cfg.grid_res)
    v_grid = np.arange(v_min, v_max + cfg.grid_res, cfg.grid_res)
    W, H = len(u_grid), len(v_grid)

    top_occ_mask = np.zeros((H, W), dtype=bool)
    xs, ys, valid = uv_to_grid_index(top_uv, u_min, v_min, cfg.grid_res, W, H)
    top_occ_mask[ys[valid], xs[valid]] = True

    top_occ_smooth = ndimage.binary_closing(top_occ_mask, iterations=cfg.top_occ_close_iter)
    if cfg.top_occ_dilate_iter > 0:
        top_occ_smooth = ndimage.binary_dilation(top_occ_smooth, iterations=cfg.top_occ_dilate_iter)

    ideal_inner_mask = build_ideal_inner_mask_from_lines(
        u_grid=u_grid,
        v_grid=v_grid,
        boundary=boundary,
        corner_mode=corner_mode,
        line_margin=cfg.boundary_line_margin,
    )
    boundary.ideal_inner_mask = ideal_inner_mask

    # Missing top candidates inside the ideal model footprint.
    defect_candidate = ideal_inner_mask & (~top_occ_smooth)
    defect_candidate = ndimage.binary_closing(defect_candidate, iterations=cfg.defect_close_iter)

    corner_cell = uv_to_nearest_grid_cell(corner_uv, u_min, v_min, cfg.grid_res, W, H)
    defect_component = select_component_closest_to_corner(
        defect_candidate,
        corner_cell=corner_cell,
        min_area=cfg.min_component_area_pixels,
    )

    margin_pix = max(1, int(round(cfg.bbox_margin / cfg.grid_res)))
    bbox_mask = bbox_mask_from_component(defect_component, H=H, W=W, margin_pix=margin_pix)

    # Final domain is a clean local bbox clipped by fitted ideal boundary lines.
    domain_mask = bbox_mask & ideal_inner_mask

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
        ideal_inner_mask=ideal_inner_mask,
        defect_candidate=defect_candidate,
        defect_component=defect_component,
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
    boundary: BoundaryData,
    grid: GridData,
    corner_mode: str,
) -> CompletionData:
    defect_surface_points, defect_surface_uv, defect_surface_z = extract_defect_surface_points(cfg, plane, grid)

    domain_mask = grid.domain_mask
    domain_area = domain_mask.sum()
    seed = get_seed_cell_from_corner_uv(boundary.ideal_corner_uv, grid, domain_mask)

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
    fix_surface_masks = []
    barrier_z_values = []
    barrier_valid_flags = []
    barrier_area_ratios = []
    barrier_slice_nums = []
    fix_points_all = []
    raw_fix_surface_points_all = []

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

            barrier_mask = ndimage.binary_dilation(raw_barrier_mask, iterations=cfg.barrier_dilate_iter)
            barrier_mask = ndimage.binary_closing(barrier_mask, iterations=cfg.barrier_close_iter)
            barrier_mask = barrier_mask & domain_mask

            repair_mask_z = flood_fill_from_corner(barrier_mask, domain_mask, seed)
            repair_area = repair_mask_z.sum()
            area_ratio = repair_area / (domain_area + 1e-12)
            valid_barrier = (repair_area >= cfg.min_area_pixels) and (area_ratio < cfg.max_area_ratio)

            # Save repair-side contact margin for glue planning.
            fix_surface_mask = barrier_mask & ndimage.binary_dilation(repair_mask_z, iterations=1)
            fix_surface_mask = fix_surface_mask & domain_mask

            raw_fix_valid = np.zeros(len(slice_points), dtype=bool)
            raw_fix_valid[svalid] = fix_surface_mask[sy[svalid], sx[svalid]]
            if np.any(raw_fix_valid):
                raw_fix_surface_points_all.append(slice_points[raw_fix_valid])

            barrier_raw_masks.append(raw_barrier_mask.copy())
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
            "bad_barrier_count": bad_barrier_count,
            "low_points_count": low_points_count,
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

    if cfg.enable_column_depth_filter:
        repair_layers, column_keep_mask, column_count_map = filter_repair_layers_by_column_depth(
            repair_layers,
            min_column_layers=cfg.min_column_layers,
        )

        if len(repair_layers) == 0:
            raise RuntimeError(
                "No repair layers left after column depth filter. "
                "Try reducing cfg.min_column_layers."
            )

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

    uv_domain = mask_to_uv_points(grid.domain_mask, grid.u_min, grid.v_min, cfg.grid_res)
    u_min_domain, v_min_domain = uv_domain.min(axis=0)
    u_max_domain, v_max_domain = uv_domain.max(axis=0)

    side_points = extract_side_plane_points(
        repair_points=repair_volume_points,
        origin=plane.origin,
        u_axis=plane.u_axis,
        v_axis=plane.v_axis,
        n_axis=plane.n_axis,
        corner_mode=corner_mode,
        u_min_domain=u_min_domain,
        u_max_domain=u_max_domain,
        v_min_domain=v_min_domain,
        v_max_domain=v_max_domain,
        side_tol=cfg.layer_step,
    )
    side_n_mark = get_side_normal(corner_mode, plane.u_axis, plane.v_axis)

    fix_points = np.vstack(fix_points_all) if len(fix_points_all) > 0 else np.empty((0, 3))
    raw_fix_surface_points = (
        np.unique(np.vstack(raw_fix_surface_points_all), axis=0)
        if len(raw_fix_surface_points_all) > 0
        else np.empty((0, 3))
    )

    if len(fix_surface_masks) > 0:
        barrier_raw_masks_arr = np.asarray(barrier_raw_masks, dtype=bool)
        fix_surface_masks_arr = np.asarray(fix_surface_masks, dtype=bool)
    else:
        barrier_raw_masks_arr = np.zeros((0, grid.H, grid.W), dtype=bool)
        fix_surface_masks_arr = np.zeros((0, grid.H, grid.W), dtype=bool)

    fix_npz_data = {
        "barrier_raw_masks": barrier_raw_masks_arr,
        "fix_surface_masks": fix_surface_masks_arr,
        # Backward-compatible name. It stores repair-side fix/contact margin.
        "barrier_processed_masks": fix_surface_masks_arr,
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
        "ideal_inner_mask": grid.ideal_inner_mask,
        "defect_candidate": grid.defect_candidate,
        "defect_component": grid.defect_component,
        "top_defect_mask": grid.defect_mask,
        "ideal_corner_uv": boundary.ideal_corner_uv,
        "max_u_line": np.asarray(boundary.max_u_line),
        "edge_v_line": np.asarray(boundary.edge_v_line),
        "edge_v_side": np.asarray(boundary.edge_v_side),
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
        raw_fix_surface_points=raw_fix_surface_points,
        fix_npz_data=fix_npz_data,
        debug_records=debug_records,
    )


def estimate_defect_world_y(top_defect_margin_points: np.ndarray) -> str:
    if len(top_defect_margin_points) == 0:
        return "unknown"
    left_count = np.sum(top_defect_margin_points[:, 1] > 0)
    right_count = np.sum(top_defect_margin_points[:, 1] < 0)
    return "left" if left_count > right_count else "right"


def save_results(
    cfg: CompletionConfig,
    plane: PlaneData,
    boundary: BoundaryData,
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
    raw_fix_surface_pcd = make_colored_pcd(completion.raw_fix_surface_points, [1.0, 0.0, 1.0])

    target_side_names = get_target_sides_from_corner_mode(corner_mode)
    u_side_name = target_side_names[0]
    v_side_name = target_side_names[1]

    u_side_pcd = make_colored_pcd(completion.side_points.get(u_side_name, np.empty((0, 3))), [0.0, 1.0, 0.0])
    v_side_pcd = make_colored_pcd(completion.side_points.get(v_side_name, np.empty((0, 3))), [0.0, 0.0, 1.0])

    o3d.io.write_point_cloud(os.path.join(cfg.output_dir, "model.pcd"), repair_model_pcd)
    o3d.io.write_point_cloud(os.path.join(cfg.output_dir, "u_side_plane.pcd"), u_side_pcd)
    o3d.io.write_point_cloud(os.path.join(cfg.output_dir, "v_side_plane.pcd"), v_side_pcd)
    o3d.io.write_point_cloud(os.path.join(cfg.output_dir, "top_plane.pcd"), top_plane_pcd)
    o3d.io.write_point_cloud(os.path.join(cfg.output_dir, "top_defect_margin.pcd"), top_margin_pcd)
    o3d.io.write_point_cloud(os.path.join(cfg.output_dir, "fix_points.pcd"), fix_pcd)
    o3d.io.write_point_cloud(os.path.join(cfg.output_dir, "raw_fix_surface_points.pcd"), raw_fix_surface_pcd)

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
        max_u_line=np.asarray(boundary.max_u_line),
        edge_v_line=np.asarray(boundary.edge_v_line),
        edge_v_side=boundary.edge_v_side,
        ideal_corner_uv=boundary.ideal_corner_uv,
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
        ideal_inner_mask=grid.ideal_inner_mask,
        defect_candidate=grid.defect_candidate,
        defect_component=grid.defect_component,
        domain_mask=grid.domain_mask,
        defect_mask=grid.defect_mask,
        boundary_support_uv=boundary.boundary_support_uv,
        near_top_rest_uv=boundary.near_top_rest_uv,
        near_top_rest_points=boundary.near_top_rest_points,
        max_u_candidates=boundary.max_u_candidates,
        max_u_inliers=boundary.max_u_inliers,
        edge_v_candidates=boundary.edge_v_candidates,
        edge_v_inliers=boundary.edge_v_inliers,
        max_u_line=np.asarray(boundary.max_u_line),
        edge_v_line=np.asarray(boundary.edge_v_line),
        ideal_corner_uv=boundary.ideal_corner_uv,
    )

    print("\n========== depression_completion_boundary_slide done ==========")
    print("corner_mode:", corner_mode)
    print("output_dir:", cfg.output_dir)
    print("saved: model.pcd, u_side_plane.pcd, v_side_plane.pcd")
    print("saved: top_plane.pcd, top_defect_margin.pcd")
    print("saved: meta.npz, top_defect_margin.npz")
    print("saved: fix_mask.npz, fix_points.pcd, debug_masks.npz")


def visualize_boundary_fit(cfg: CompletionConfig, plane: PlaneData, boundary: BoundaryData):
    if not cfg.visualize:
        return

    top_plane_pcd = o3d.geometry.PointCloud(plane.plane_pcd)
    top_plane_pcd.paint_uniform_color([0.65, 0.65, 0.65])

    near_top_rest_support = make_colored_pcd(
        boundary.near_top_rest_points,
        [0.7, 0.0, 1.0],
    )

    max_u_candidates = make_colored_pcd(
        uv_to_3d(boundary.max_u_candidates, plane.origin, plane.u_axis, plane.v_axis),
        [1.0, 0.55, 0.0],
    )
    max_u_inliers = make_colored_pcd(
        uv_to_3d(boundary.max_u_candidates[boundary.max_u_inliers], plane.origin, plane.u_axis, plane.v_axis),
        [1.0, 0.0, 0.0],
    )
    edge_v_candidates = make_colored_pcd(
        uv_to_3d(boundary.edge_v_candidates, plane.origin, plane.u_axis, plane.v_axis),
        [0.0, 0.75, 1.0],
    )
    edge_v_inliers = make_colored_pcd(
        uv_to_3d(boundary.edge_v_candidates[boundary.edge_v_inliers], plane.origin, plane.u_axis, plane.v_axis),
        [0.0, 0.0, 1.0],
    )

    v_samples = np.linspace(plane.top_uv[:, 1].min(), plane.top_uv[:, 1].max(), 80)
    max_u_samples = np.column_stack([
        boundary.max_u_line[0] * v_samples + boundary.max_u_line[1],
        v_samples,
    ])
    max_u_line = make_uv_line_set(max_u_samples, plane.origin, plane.u_axis, plane.v_axis, [1.0, 0.0, 0.0])

    u_samples = np.linspace(plane.top_uv[:, 0].min(), plane.top_uv[:, 0].max(), 80)
    edge_v_samples = np.column_stack([
        u_samples,
        boundary.edge_v_line[0] * u_samples + boundary.edge_v_line[1],
    ])
    edge_v_line = make_uv_line_set(edge_v_samples, plane.origin, plane.u_axis, plane.v_axis, [0.0, 0.0, 1.0])

    ideal_corner_xyz = uv_to_3d(
        boundary.ideal_corner_uv.reshape(1, 2),
        plane.origin,
        plane.u_axis,
        plane.v_axis,
    )[0]
    corner_sphere = make_sphere(ideal_corner_xyz, radius=0.003, color=[1.0, 1.0, 0.0])
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)

    print("\n========== Visualize boundary RANSAC ==========")
    print("gray/raw: original point cloud")
    print("purple: near-top rest/sloped support points used for boundary support")
    print("orange/red: max_u candidates/inliers and fitted line")
    print("cyan/blue: edge_v candidates/inliers and fitted line")
    print("yellow sphere: ideal fitted corner")
    o3d.visualization.draw_geometries([
        plane.pcd_raw,
        near_top_rest_support,
        max_u_candidates,
        max_u_inliers,
        edge_v_candidates,
        edge_v_inliers,
        max_u_line,
        edge_v_line,
        corner_sphere,
        frame,
    ], window_name="RANSAC boundary lines vs top point cloud")


def visualize_results(cfg: CompletionConfig, plane: PlaneData, boundary: BoundaryData, completion: CompletionData):
    if not cfg.visualize:
        return
    repair_model_pcd = make_colored_pcd(completion.repair_volume_points, [1.0, 0.0, 0.0])
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
    o3d.visualization.draw_geometries([plane.pcd_raw, repair_model_pcd, frame])


def run_completion(cfg: CompletionConfig):
    corner_mode = resolve_corner_mode(cfg)
    plane = load_and_prepare_plane(cfg)
    boundary = fit_boundary_lines(cfg, plane, corner_mode)
    visualize_boundary_fit(cfg, plane, boundary)
    grid = build_grid_and_defect_masks(cfg, plane, boundary, corner_mode)
    completion = run_layered_flood_fill(cfg, plane, boundary, grid, corner_mode)
    save_results(cfg, plane, boundary, grid, completion, corner_mode)
    visualize_results(cfg, plane, boundary, completion)


# ============================================================
# CLI
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Boundary-line based depression completion.")
    parser.add_argument("--pcd", type=str, default=None, help="Input fine_fuse.pcd path.")
    parser.add_argument("--corner-json", type=str, default=None, help="corner_mapping_result.json path.")
    parser.add_argument("--corner-mode", type=str, default=None, choices=["max_u_max_v", "max_u_min_v"], help="Override corner mode.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory.")
    parser.add_argument("--boundary-away-low", type=float, default=None, help="Lower normalized distance from damaged corner for boundary candidates, default 0.30.")
    parser.add_argument("--boundary-away-high", type=float, default=None, help="Upper normalized distance from damaged corner for boundary candidates, default 0.90.")
    parser.add_argument("--boundary-support-z-min", type=float, default=None, help="Min rest_z for near-top boundary support, default -0.003 m.")
    parser.add_argument("--boundary-support-z-max", type=float, default=None, help="Max rest_z for near-top boundary support, default 0.020 m.")
    parser.add_argument("--no-rest-boundary", action="store_true", help="Use only top plane points for boundary fitting.")
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
    if args.boundary_away_low is not None:
        cfg.boundary_away_low = args.boundary_away_low
    if args.boundary_away_high is not None:
        cfg.boundary_away_high = args.boundary_away_high
    if args.boundary_support_z_min is not None:
        cfg.boundary_support_z_min = args.boundary_support_z_min
    if args.boundary_support_z_max is not None:
        cfg.boundary_support_z_max = args.boundary_support_z_max
    if args.no_rest_boundary:
        cfg.use_rest_points_for_boundary = False
    if args.no_vis:
        cfg.visualize = False
    return cfg


if __name__ == "__main__":
    run_completion(config_from_args(parse_args()))

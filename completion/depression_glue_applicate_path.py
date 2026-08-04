import argparse
import json
import os

import numpy as np
import open3d as o3d
from scipy import ndimage
from scipy.spatial import cKDTree


# ======================
# User parameters
# ======================
FIXPOINT_CHOICE = "fix_points"  # "fix_points" or "raw_fix_surface_points"
FIXPOINT_CHOICES = {
    "fix_points": "fix_points.pcd",
    "raw_fix_surface_points": "raw_fix_surface_points.pcd",
}
if FIXPOINT_CHOICE not in FIXPOINT_CHOICES:
    raise ValueError(
        f"FIXPOINT_CHOICE must be one of {tuple(FIXPOINT_CHOICES)}, got {FIXPOINT_CHOICE!r}"
    )

NUM_DEPTH_LAYERS = 1
DOT_SPACING = 0.03              # m, spacing between dispensing dots
SURFACE_OFFSET = 0.0003         # m, offset to missing/free side to avoid scraping
MAX_ORDER_POINTS = 900          # cap points for nearest-neighbor ordering
MIN_MASK_COMPONENT_POINTS = 3
LOCAL_NORMAL_RADII = (0.005, 0.008, 0.012, 0.020)  # m
LOCAL_NORMAL_MIN_NEIGHBORS = 10
LOCAL_NORMAL_K_FALLBACK = 30

SAVE_NPZ = True
SAVE_PCD = True
SAVE_JSON = True


def parse_args():
    parser = argparse.ArgumentParser(description="Generate glue dispensing dots and local normals.")
    parser.add_argument("--fix-mask", required=True, help="twoFit output fix_mask.npz.")
    parser.add_argument(
        "--fix-points",
        required=True,
        help="twoFit output surface PCD selected by --fixpoint-choice.",
    )
    parser.add_argument("--out-dir", required=True, help="Directory for generated dot files.")
    parser.add_argument(
        "--fixpoint-choice",
        choices=tuple(FIXPOINT_CHOICES),
        default=FIXPOINT_CHOICE,
        help="Surface point cloud used to generate dots and normals.",
    )
    return parser.parse_args()


# ======================
# Basic geometry helpers
# ======================
def normalize(v):
    return v / (np.linalg.norm(v) + 1e-12)


def estimate_and_save_fix_plane_normal(
    fix_points_path,
    output_path,
    distance_threshold=0.002,
    num_iterations=3000,
):
    """Fit the dominant fix plane and orient its normal toward -world_x."""
    pcd = o3d.io.read_point_cloud(fix_points_path)
    points = np.asarray(pcd.points)
    if len(points) < 3:
        raise RuntimeError(f"fix_points.pcd needs at least 3 points: {fix_points_path}")

    plane_model, _ = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=3,
        num_iterations=num_iterations,
    )
    normal = normalize(np.asarray(plane_model[:3], dtype=float))
    world_x = np.array([1.0, 0.0, 0.0])
    dot_world_x = float(np.dot(normal, world_x))
    if abs(dot_world_x) <= 1e-9:
        raise RuntimeError(
            "The fitted normal is perpendicular to world_x; its dot product "
            "cannot be made strictly negative by flipping the normal."
        )
    if dot_world_x > 0.0:
        normal = -normal

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {"n_fix_plane": [round(float(value), 10) for value in normal]},
            f,
            indent=2,
            ensure_ascii=False,
        )
    return normal


def inward_dot_normals(dot_groups, fix_points_path, fix_plane_normal):
    """Fit one local PCA normal per dot from neighboring fix_points.pcd points."""
    global_inward_normal = -normalize(np.asarray(fix_plane_normal, dtype=float).reshape(3))
    world_x = np.array([1.0, 0.0, 0.0])
    if float(np.dot(global_inward_normal, world_x)) <= 0.0:
        global_inward_normal = -global_inward_normal

    fix_pcd = o3d.io.read_point_cloud(str(fix_points_path))
    fix_points = np.asarray(fix_pcd.points, dtype=float)
    if len(fix_points) < 3:
        raise RuntimeError(f"fix_points.pcd needs at least 3 points: {fix_points_path}")
    tree = cKDTree(fix_points)

    normal_groups = []
    neighbor_count_groups = []
    fallback_groups = []
    fallback_k = min(LOCAL_NORMAL_K_FALLBACK, len(fix_points))

    for points in dot_groups:
        group_normals = []
        group_neighbor_counts = []
        group_fallbacks = []
        for dot in np.asarray(points, dtype=float):
            neighbor_ids = []
            for radius in LOCAL_NORMAL_RADII:
                neighbor_ids = tree.query_ball_point(dot, radius)
                if len(neighbor_ids) >= LOCAL_NORMAL_MIN_NEIGHBORS:
                    break

            used_fallback = False
            if len(neighbor_ids) < LOCAL_NORMAL_MIN_NEIGHBORS:
                _, neighbor_ids = tree.query(dot, k=fallback_k)
                neighbor_ids = np.atleast_1d(neighbor_ids).astype(int).tolist()

            neighbors = fix_points[neighbor_ids]
            centered = neighbors - neighbors.mean(axis=0)
            covariance = centered.T @ centered / max(len(neighbors), 1)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            local_normal = normalize(eigenvectors[:, np.argmin(eigenvalues)])

            if float(np.dot(local_normal, world_x)) < 0.0:
                local_normal = -local_normal
            if float(np.dot(local_normal, world_x)) <= 1e-9:
                local_normal = global_inward_normal.copy()
                used_fallback = True

            group_normals.append(local_normal)
            group_neighbor_counts.append(len(neighbor_ids))
            group_fallbacks.append(used_fallback)

        normal_groups.append(np.asarray(group_normals, dtype=float).reshape(-1, 3))
        neighbor_count_groups.append(np.asarray(group_neighbor_counts, dtype=int))
        fallback_groups.append(np.asarray(group_fallbacks, dtype=bool))

    return global_inward_normal, normal_groups, neighbor_count_groups, fallback_groups


def cells_to_3d(cells_yx, z, u_min, v_min, grid_res, origin, u_axis, v_axis, n_axis):
    cells_yx = np.asarray(cells_yx, dtype=float)
    if len(cells_yx) == 0:
        return np.empty((0, 3))

    u = u_min + (cells_yx[:, 1] + 0.5) * grid_res
    v = v_min + (cells_yx[:, 0] + 0.5) * grid_res
    return origin + u[:, None] * u_axis + v[:, None] * v_axis + z * n_axis


def largest_component(mask):
    labels, num = ndimage.label(mask)
    if num == 0:
        return mask
    areas = ndimage.sum(mask, labels, index=np.arange(1, num + 1))
    return labels == (np.argmax(areas) + 1)


def inner_boundary(mask, erode_iter=1):
    eroded = ndimage.binary_erosion(mask, iterations=erode_iter)
    return mask & (~eroded)


def mask_components(mask, min_points=3):
    labels, num = ndimage.label(mask)
    components = []

    for label_id in range(1, num + 1):
        cells = np.column_stack(np.where(labels == label_id))
        if len(cells) >= min_points:
            components.append(cells)

    return components


def order_points_nn(cells_yx, max_points=900):
    pts = np.asarray(cells_yx, dtype=float)
    if len(pts) == 0:
        return pts

    if len(pts) > max_points:
        center = pts.mean(axis=0)
        _, _, vh = np.linalg.svd(pts - center, full_matrices=False)
        s = (pts - center) @ vh[0]
        order = np.argsort(s)
        pts = pts[order[np.linspace(0, len(order) - 1, max_points).astype(int)]]

    start = np.argmin(pts[:, 0] + pts[:, 1])
    ordered = [pts[start]]
    remain = np.ones(len(pts), dtype=bool)
    remain[start] = False
    cur = pts[start]

    for _ in range(len(pts) - 1):
        ids = np.where(remain)[0]
        nxt = ids[np.argmin(np.linalg.norm(pts[ids] - cur, axis=1))]
        cur = pts[nxt]
        ordered.append(cur)
        remain[nxt] = False

    return np.asarray(ordered)


def sample_ordered_dot_cells(ordered_cells, grid_res, spacing):
    pts = np.asarray(ordered_cells, dtype=float)
    if len(pts) <= 1:
        return pts

    ds = np.linalg.norm(np.diff(pts, axis=0), axis=1) * grid_res
    s = np.concatenate([[0.0], np.cumsum(ds)])
    total = s[-1]
    if total <= spacing:
        return pts[[0]]

    targets = np.arange(0.0, total + 1e-12, spacing)
    ids = np.searchsorted(s, targets, side="left")
    ids = np.clip(ids, 0, len(pts) - 1)
    return pts[np.unique(ids)]


def free_side_vector_yx(mask, top_defect_mask):
    mask_yx = np.column_stack(np.where(mask))
    defect_yx = np.column_stack(np.where(top_defect_mask))
    if len(mask_yx) == 0 or len(defect_yx) == 0:
        return np.array([0.0, 0.0])
    return normalize(defect_yx.mean(axis=0) - mask_yx.mean(axis=0))


def sample_mask_grid_cells(mask, grid_res, spacing):
    cells = np.column_stack(np.where(mask))
    if len(cells) == 0:
        return np.empty((0, 2))

    spacing_cells = spacing / grid_res
    y_min, x_min = cells.min(axis=0)

    bin_y = np.floor((cells[:, 0] - y_min) / spacing_cells).astype(int)
    bin_x = np.floor((cells[:, 1] - x_min) / spacing_cells).astype(int)
    bin_ids = np.column_stack([bin_y, bin_x])

    sampled = []
    for by, bx in np.unique(bin_ids, axis=0):
        in_bin = (bin_y == by) & (bin_x == bx)
        bin_cells = cells[in_bin]
        bin_center = np.array([
            y_min + (by + 0.5) * spacing_cells,
            x_min + (bx + 0.5) * spacing_cells
        ])
        sampled.append(bin_cells[np.argmin(np.linalg.norm(bin_cells - bin_center, axis=1))])

    return np.asarray(sampled, dtype=float)


def sample_component_centerline_cells(cells_yx, grid_res, spacing):
    cells = np.asarray(cells_yx, dtype=float)
    if len(cells) <= 1:
        return cells

    center = cells.mean(axis=0)
    _, _, vh = np.linalg.svd(cells - center, full_matrices=False)
    main_axis = vh[0]
    s = (cells - center) @ main_axis
    spacing_cells = spacing / grid_res

    if s.max() - s.min() <= spacing_cells:
        idx = np.argmin(np.linalg.norm(cells - center, axis=1))
        return cells[[idx]]

    targets = np.arange(s.min(), s.max() + 1e-12, spacing_cells)
    sampled = []
    half = spacing_cells / 2

    for target in targets:
        in_bin = np.abs(s - target) <= half
        if np.any(in_bin):
            bin_cells = cells[in_bin]
            bin_center = bin_cells.mean(axis=0)
            sampled.append(bin_cells[np.argmin(np.linalg.norm(bin_cells - bin_center, axis=1))])
        else:
            sampled.append(cells[np.argmin(np.abs(s - target))])

    sampled = np.asarray(sampled)
    _, unique_ids = np.unique(np.round(sampled).astype(int), axis=0, return_index=True)
    return sampled[np.sort(unique_ids)]


def centerline_dots_from_mask(mask, z, params):
    grid_res = params["grid_res"]
    components = mask_components(
        mask,
        min_points=params["min_mask_component_points"]
    )
    if len(components) == 0:
        return np.empty((0, 3))

    dot_cells = []
    for component_yx in components:
        dots = sample_component_centerline_cells(
            component_yx,
            grid_res=grid_res,
            spacing=params["dot_spacing"]
        )
        if len(dots) > 0:
            dot_cells.append(dots)

    if len(dot_cells) == 0:
        return np.empty((0, 3))

    dots = np.vstack(dot_cells)
    return cells_to_3d(
        dots, z,
        params["u_min"], params["v_min"], grid_res,
        params["origin"], params["u_axis"], params["v_axis"], params["n_axis"]
    )


def grid_dots_from_mask(mask, z, params):
    grid_res = params["grid_res"]
    dots = sample_mask_grid_cells(
        mask,
        grid_res=grid_res,
        spacing=params["dot_spacing"]
    )

    return cells_to_3d(
        dots, z,
        params["u_min"], params["v_min"], grid_res,
        params["origin"], params["u_axis"], params["v_axis"], params["n_axis"]
    )


def build_depth_layer_dots(layer_mask, z, top_defect_mask, params):
    return grid_dots_from_mask(
        layer_mask,
        z,
        params
    )


def group_colors(dot_groups, dot_info):
    depth_palette = [
        np.array([0.85, 0.10, 0.10]),
        np.array([1.00, 0.55, 0.05]),
        np.array([0.10, 0.45, 0.85]),
    ]
    colors = []
    depth_i = 0
    for group, info in zip(dot_groups, dot_info):
        if info["type"] == "depth_barrier_dots":
            color = depth_palette[depth_i % len(depth_palette)]
            depth_i += 1
        else:
            color = np.array([0.6, 0.2, 0.8])
        colors.append(np.tile(color, (len(group), 1)))
    return np.vstack(colors)


def select_dispensing_layer_ids(valid_ids, num_layers):
    candidate_ids = valid_ids[2:-1]
    if len(candidate_ids) == 0:
        raise RuntimeError("No valid layers available from the 3rd layer to the penultimate layer.")

    pick_count = min(num_layers, len(candidate_ids))
    pick_pos = np.linspace(0, len(candidate_ids) - 1, pick_count).astype(int)
    return candidate_ids[pick_pos][::-1]       # deep to shallow


# ======================
# Main dot generation
# ======================
def main():
    args = parse_args()
    fix_mask_path = os.path.abspath(args.fix_mask)
    fix_points_path = os.path.abspath(args.fix_points)
    out_dir = os.path.abspath(args.out_dir)
    fix_normal_path = os.path.join(out_dir, "n_fix_plane.json")

    if not os.path.isfile(fix_mask_path):
        raise FileNotFoundError(fix_mask_path)
    if not os.path.isfile(fix_points_path):
        raise FileNotFoundError(fix_points_path)

    print("FIXPOINT_CHOICE:", args.fixpoint_choice)
    print("Surface point cloud:", fix_points_path)
    n_fix_plane = estimate_and_save_fix_plane_normal(fix_points_path, fix_normal_path)
    data = np.load(fix_mask_path, allow_pickle=True)

    barrier_masks = data["barrier_processed_masks"].astype(bool)
    z_values = data["barrier_z_values"]
    valid_flags = data["barrier_valid_flags"].astype(bool)
    top_defect_mask = data["top_defect_mask"].astype(bool)

    params = {
        "grid_res": float(data["grid_res"]),
        "u_min": float(data["u_min"]),
        "v_min": float(data["v_min"]),
        "origin": data["origin"],
        "u_axis": data["u_axis"],
        "v_axis": data["v_axis"],
        "n_axis": data["n_axis"],
        "surface_offset": SURFACE_OFFSET,
        "dot_spacing": DOT_SPACING,
        "max_order_points": MAX_ORDER_POINTS,
        "min_mask_component_points": MIN_MASK_COMPONENT_POINTS,
    }

    valid_ids = np.where(valid_flags)[0]
    selected_ids = select_dispensing_layer_ids(valid_ids, NUM_DEPTH_LAYERS)

    dot_groups = []
    dot_info = []

    for layer_id in selected_ids:
        dots = build_depth_layer_dots(
            barrier_masks[layer_id],
            float(z_values[layer_id]),
            top_defect_mask,
            params
        )
        dot_groups.append(dots)
        dot_info.append({
            "type": "depth_barrier_dots",
            "layer_id": int(layer_id),
            "z": float(z_values[layer_id]),
            "point_num": int(len(dots))
        })

    all_points = np.vstack(dot_groups)
    dot_group_ids = np.concatenate([np.full(len(p), i, dtype=int) for i, p in enumerate(dot_groups)])
    inward_normal, dot_normal_groups, normal_neighbor_count_groups, normal_fallback_groups = (
        inward_dot_normals(dot_groups, fix_points_path, n_fix_plane)
    )
    all_normals = np.vstack(dot_normal_groups)
    all_normal_neighbor_counts = np.concatenate(normal_neighbor_count_groups)
    all_normal_fallback_flags = np.concatenate(normal_fallback_groups)

    os.makedirs(out_dir, exist_ok=True)
    out_npz = os.path.join(out_dir, "glue_applicate_dots.npz")
    out_pcd = os.path.join(out_dir, "glue_applicate_dots.pcd")
    out_json = os.path.join(out_dir, "glue_applicate_dots_info.json")

    if SAVE_NPZ:
        np.savez(
            out_npz,
            dot_groups=np.asarray(dot_groups, dtype=object),
            all_points=all_points,
            dot_normal_groups=np.asarray(dot_normal_groups, dtype=object),
            all_normals=all_normals,
            inward_normal=inward_normal,
            fixpoint_choice=np.asarray(args.fixpoint_choice),
            surface_point_cloud=np.asarray(fix_points_path),
            normal_neighbor_count_groups=np.asarray(normal_neighbor_count_groups, dtype=object),
            all_normal_neighbor_counts=all_normal_neighbor_counts,
            normal_fallback_groups=np.asarray(normal_fallback_groups, dtype=object),
            all_normal_fallback_flags=all_normal_fallback_flags,
            dot_group_ids=dot_group_ids,
            selected_layer_ids=selected_ids,
            selected_layer_z=z_values[selected_ids],
            dot_info=np.asarray(dot_info, dtype=object),
            surface_offset=SURFACE_OFFSET,
            dot_spacing=DOT_SPACING
        )

    if SAVE_PCD:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points)
        pcd.normals = o3d.utility.Vector3dVector(all_normals)
        pcd.colors = o3d.utility.Vector3dVector(group_colors(dot_groups, dot_info))
        o3d.io.write_point_cloud(out_pcd, pcd)

    if SAVE_JSON:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(dot_info, f, indent=2, ensure_ascii=False)

    print("Selected depth layer ids:", selected_ids)
    print("Selected depth z values:", z_values[selected_ids])
    print("Generated dot groups:", len(dot_groups))
    print("Generated dispensing dots:", len(all_points))
    print("Fix plane normal:", n_fix_plane.tolist())
    print("Saved:", fix_normal_path)
    print("Saved:", out_npz)
    print("Saved:", out_pcd)
    print("Saved:", out_json)


if __name__ == "__main__":
    main()

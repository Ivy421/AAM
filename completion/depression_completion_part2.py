import os
import argparse
import numpy as np
import open3d as o3d
from scipy import ndimage

from depression_completion_part1 import *


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


PART2_STATE_NAME = "_part2_barriers.npz"
PART2_DEFECT_PCD_NAME = "_part2_defect_surface.pcd"


def build_layer_barrier_candidates(cfg: CompletionConfig, plane: PlaneData, grid: GridData, corner_mode: str):
    defect_surface_points, defect_surface_uv, defect_surface_z = extract_defect_surface_points(cfg, plane, grid)
    o3d.io.write_point_cloud(os.path.join(cfg.output_dir, PART2_DEFECT_PCD_NAME), make_colored_pcd(defect_surface_points, [1.0, 0.5, 0.0]))

    z_values = []
    slice_nums = []
    low_points_flags = []
    raw_masks = []
    curve_masks = []
    processed_masks = []
    fix_wall_masks = []

    z = cfg.layer_step
    while z <= cfg.max_search_depth:
        slice_mask = np.abs(defect_surface_z - z) <= cfg.band_width
        slice_uv = defect_surface_uv[slice_mask]
        slice_num = len(slice_uv)
        low_points = slice_num < cfg.thres_points_num

        raw_barrier_mask = np.zeros((grid.H, grid.W), dtype=bool)
        barrier_curve_mask = np.zeros((grid.H, grid.W), dtype=bool)
        barrier_mask = np.zeros((grid.H, grid.W), dtype=bool)
        fix_wall_mask = np.zeros((grid.H, grid.W), dtype=bool)

        if not low_points:
            sx, sy, svalid = uv_to_grid_index(slice_uv, grid.u_min, grid.v_min, cfg.grid_res, grid.W, grid.H)
            raw_barrier_mask[sy[svalid], sx[svalid]] = True
            raw_barrier_mask &= grid.domain_mask
            barrier_mask, barrier_curve_mask, fix_wall_mask = build_skeleton_barrier_from_raw(
                raw_barrier_mask, grid.domain_mask, corner_mode, cfg.wall_endpoint_extend_pixels,
            )

        z_values.append(z)
        slice_nums.append(slice_num)
        low_points_flags.append(low_points)
        raw_masks.append(raw_barrier_mask)
        curve_masks.append(barrier_curve_mask)
        processed_masks.append(barrier_mask)
        fix_wall_masks.append(fix_wall_mask)
        z += cfg.layer_step

    np.savez(
        os.path.join(cfg.output_dir, PART2_STATE_NAME),
        z_values=np.asarray(z_values), slice_nums=np.asarray(slice_nums), low_points_flags=np.asarray(low_points_flags, dtype=bool),
        raw_masks=np.asarray(raw_masks, dtype=bool), curve_masks=np.asarray(curve_masks, dtype=bool),
        processed_masks=np.asarray(processed_masks, dtype=bool), fix_wall_masks=np.asarray(fix_wall_masks, dtype=bool),
    )
    print("\n========== Part 2 done ==========")
    print("defect surface points:", len(defect_surface_points))
    print("candidate layers:", len(z_values))
    print("saved:", os.path.join(cfg.output_dir, PART2_STATE_NAME))


def load_part2_state(output_dir: str):
    data = np.load(os.path.join(output_dir, PART2_STATE_NAME), allow_pickle=True)
    defect_pcd = o3d.io.read_point_cloud(os.path.join(output_dir, PART2_DEFECT_PCD_NAME))
    return data, np.asarray(defect_pcd.points)




def run_part2(cfg: CompletionConfig):
    plane, roi, grid, corner_mode = load_part1_state(cfg.output_dir)
    build_layer_barrier_candidates(cfg, plane, grid, corner_mode)


if __name__ == "__main__":
    run_part2(config_from_args(parse_args()))

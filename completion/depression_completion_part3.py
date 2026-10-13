import os
import argparse
import numpy as np
import open3d as o3d
from scipy import ndimage

from depression_completion_part1 import *
from depression_completion_part2 import load_part2_state


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


def get_seed_cell_from_fill_origin(grid: GridData, domain_mask: np.ndarray):
    """Return the valid domain cell nearest the Part-1 filling origin UV=(0,0)."""
    target_x = int(np.argmin(np.abs(grid.u_grid)))
    target_y = int(np.argmin(np.abs(grid.v_grid)))
    target = np.array([target_y, target_x], dtype=int)

    if domain_mask[target[0], target[1]]:
        return tuple(target)

    ys, xs = np.where(domain_mask)
    if len(xs) == 0:
        raise RuntimeError("domain_mask is empty; cannot select Flood Fill seed.")
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

def extend_last_valid_layer_to_bottom(
    repair_layers: List[dict],
    z_bottom: float,
    layer_step: float,
    domain_area: int,
):
    if len(repair_layers) == 0:
        return repair_layers, np.empty(0, dtype=float), np.nan

    z_bottom = float(z_bottom)
    layer_step = float(layer_step)

    last_valid_layer = repair_layers[-1]
    z_stop = float(last_valid_layer["z"])
    stop_mask = last_valid_layer["mask"].astype(bool).copy()

    extension_z_values = []

    if z_bottom <= z_stop + 1e-12:
        return repair_layers, np.empty(0, dtype=float), z_stop

    area_ratio = stop_mask.sum() / (domain_area + 1e-12)
    z_fill = z_stop + layer_step

    while z_fill < z_bottom - 1e-12:
        repair_layers.append({
            "z": float(z_fill),
            "mask": stop_mask.copy(),
            "type": "bottom_depth_extension",
            "point_num": -1,
            "area_ratio": area_ratio,
        })
        extension_z_values.append(float(z_fill))
        z_fill += layer_step

    repair_layers.append({
        "z": z_bottom,
        "mask": stop_mask.copy(),
        "type": "bottom_depth_extension",
        "point_num": -1,
        "area_ratio": area_ratio,
    })

    extension_z_values.append(z_bottom)

    return repair_layers, np.asarray(extension_z_values), z_stop


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
    repair_model_pcd = make_colored_pcd(completion.repair_volume_points, [0, 1, 0.0])
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
    o3d.visualization.draw_geometries([plane.pcd_raw, repair_model_pcd, frame])
    o3d.visualization.draw_geometries([ repair_model_pcd])


def run_layered_flood_fill_from_saved_barriers(cfg: CompletionConfig, plane: PlaneData, grid: GridData, corner_mode: str) -> CompletionData:
    barrier_data, defect_surface_points = load_part2_state(cfg.output_dir)
    domain_mask = grid.domain_mask
    domain_area = domain_mask.sum()
    seed = get_seed_cell_from_fill_origin(grid, domain_mask)

    repair_layers = [{"z": 0.0, "mask": grid.defect_mask.copy(), "type": "top_defect_mask", "point_num": -1, "area_ratio": grid.defect_mask.sum() / (domain_area + 1e-12)}]
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
    fix_surface_curve_masks = []
    fix_surface_curve_z_values = []

    for i, z in enumerate(barrier_data["z_values"]):
        z = float(z)
        slice_num = int(barrier_data["slice_nums"][i])
        low_points = bool(barrier_data["low_points_flags"][i])
        if low_points: low_points_count += 1
        else: low_points_count = 0

        valid_barrier = False
        repair_mask_z = None
        area_ratio = np.nan

        if not low_points:
            raw_barrier_mask = barrier_data["raw_masks"][i].astype(bool)
            barrier_curve_mask = barrier_data["curve_masks"][i].astype(bool)
            barrier_mask = barrier_data["processed_masks"][i].astype(bool)
            fix_wall_mask = barrier_data["fix_wall_masks"][i].astype(bool)

            repair_mask_z = flood_fill_from_corner(barrier_mask, domain_mask, seed)
            repair_area = repair_mask_z.sum()
            area_ratio = repair_area / (domain_area + 1e-12)
            valid_barrier = (repair_area >= cfg.min_area_pixels) and (area_ratio < cfg.max_area_ratio)

            fix_surface_mask = fix_wall_mask & ndimage.binary_dilation(repair_mask_z, iterations=2)
            fix_surface_mask &= domain_mask

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
            fix_surface_curve_mask = extract_first_contact_curve_mask(repair_mask_z, fix_wall_mask, corner_mode)
            fix_surface_curve_masks.append(fix_surface_curve_mask.copy())
            fix_surface_curve_z_values.append(z)
            repair_layers.append({"z": z, "mask": repair_mask_z.copy(), "type": "slice_floodfill", "point_num": slice_num, "area_ratio": area_ratio})
            print(f"[有效] z={z:.4f} m, slice points={slice_num}, area ratio={area_ratio:.3f}")
        else:
            bad_barrier_count += 1
            print(f"[无效] z={z:.4f} m, slice points={slice_num}, low_points={low_points}, area ratio={area_ratio}")

        debug_records.append({"z": z, "slice_num": slice_num, "low_points": low_points, "valid_barrier": valid_barrier, "area_ratio": area_ratio})
        if bad_barrier_count >= cfg.max_bad_layers:
            print(f"\n停止：连续 {cfg.max_bad_layers} 层无法形成有效阻挡")
            break
        if low_points_count >= cfg.max_bad_layers:
            print(f"\n停止：连续 {cfg.max_bad_layers} 层缺陷点数量少于 {cfg.thres_points_num}")
            break


    repair_layers, bottom_extension_z_values, z_stop = extend_last_valid_layer_to_bottom(repair_layers=repair_layers, z_bottom=plane.z_bottom,layer_step=cfg.layer_step,domain_area=domain_area,)
    repair_layers, column_keep_mask, column_count_map = filter_repair_layers_by_column_depth(repair_layers, min_column_layers=cfg.min_column_layers)

    repair_layers, uv_column_u_keep, uv_column_v_keep = filter_repair_layers_by_uv_column_support(repair_layers, min_points_per_line=3)
    if len(repair_layers) == 0: raise RuntimeError("No repair layers left after UV column support filter.")

    fix_surface_curve_masks_raw = [mask.copy() for mask in fix_surface_curve_masks]
    uv_keep_mask = uv_column_v_keep[:, None] & uv_column_u_keep[None, :]
    postprocess_keep_mask = column_keep_mask & uv_keep_mask
    filtered_curve_masks = []
    fix_points_curve_all = []

    for curve_mask, curve_z in zip(fix_surface_curve_masks_raw, fix_surface_curve_z_values):
        matching_layer = next((layer for layer in repair_layers if np.isclose(layer["z"], curve_z, atol=cfg.layer_step * 0.25)), None)
        if matching_layer is None:
            filtered_mask = np.zeros_like(curve_mask, dtype=bool)
        else:
            layer_support = ndimage.binary_dilation(matching_layer["mask"].astype(bool), iterations=1)
            filtered_mask = curve_mask.astype(bool) & postprocess_keep_mask & layer_support
        filtered_curve_masks.append(filtered_mask)
        curve_uv = mask_to_uv_points(filtered_mask, grid.u_min, grid.v_min, cfg.grid_res)
        if len(curve_uv) > 0:
            curve_z_arr = np.full(len(curve_uv), curve_z)
            fix_points_curve_all.append(uvz_to_3d(curve_uv, curve_z_arr, plane.origin, plane.u_axis, plane.v_axis, plane.n_axis))

    fix_surface_curve_masks = filtered_curve_masks
    raw_curve_points = sum(mask.sum() for mask in fix_surface_curve_masks_raw)
    filtered_curve_points = sum(mask.sum() for mask in fix_surface_curve_masks)
    print("\n========== Fix curve post-filter ==========")
    print("curve points before filter:", int(raw_curve_points))
    print("curve points after filter:", int(filtered_curve_points))

    all_layer_points = []
    for layer in repair_layers:
        uv_layer = mask_to_uv_points(layer["mask"], grid.u_min, grid.v_min, cfg.grid_res)
        if len(uv_layer) == 0: continue
        z_arr = np.full(len(uv_layer), layer["z"])
        all_layer_points.append(uvz_to_3d(uv_layer, z_arr, plane.origin, plane.u_axis, plane.v_axis, plane.n_axis))
    if len(all_layer_points) == 0: raise RuntimeError("No repair points generated.")

    repair_volume_points = np.vstack(all_layer_points)
    repair_point_center = repair_volume_points.mean(axis=0)
    side_points = extract_side_plane_points(repair_volume_points, plane.origin, plane.u_axis, plane.v_axis, plane.n_axis, corner_mode, grid.u_min, grid.u_max, grid.v_min, grid.v_max, cfg.layer_step)
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
        fix_surface_curve_masks_raw_arr = np.asarray(fix_surface_curve_masks_raw, dtype=bool)
        fix_surface_curve_masks_arr = np.asarray(fix_surface_curve_masks, dtype=bool)
    else:
        fix_surface_curve_masks_raw_arr = np.zeros((0, grid.H, grid.W), dtype=bool)
        fix_surface_curve_masks_arr = np.zeros((0, grid.H, grid.W), dtype=bool)

    fix_points_curve = np.vstack(fix_points_curve_all) if len(fix_points_curve_all) > 0 else np.empty((0, 3))
    fix_npz_data = {
        "barrier_raw_masks": barrier_raw_masks_arr, "barrier_curve_masks": barrier_curve_masks_arr,
        "barrier_processed_masks": barrier_processed_masks_arr, "fix_surface_masks": fix_surface_masks_arr,
        "fix_surface_curve_masks_raw": fix_surface_curve_masks_raw_arr, "fix_surface_curve_masks": fix_surface_curve_masks_arr,
        "fix_surface_curve_z_values": np.asarray(fix_surface_curve_z_values), "barrier_z_values": np.asarray(barrier_z_values),
        "barrier_valid_flags": np.asarray(barrier_valid_flags, dtype=bool), "barrier_area_ratios": np.asarray(barrier_area_ratios),
        "barrier_slice_nums": np.asarray(barrier_slice_nums), "grid_res": np.asarray(cfg.grid_res),
        "u_min": np.asarray(grid.u_min), "v_min": np.asarray(grid.v_min), "u_max": np.asarray(grid.u_max), "v_max": np.asarray(grid.v_max),
        "H": np.asarray(grid.H), "W": np.asarray(grid.W), "origin": plane.origin, "u_axis": plane.u_axis, "v_axis": plane.v_axis, "n_axis": plane.n_axis,
        "domain_mask": grid.domain_mask, "corner_roi": grid.corner_roi, "top_defect_mask": grid.defect_mask,
        "column_keep_mask": column_keep_mask, "column_count_map": column_count_map, "min_column_layers": np.asarray(cfg.min_column_layers),
        "uv_column_min_points_per_line": np.asarray(3), "uv_column_u_keep": np.asarray(uv_column_u_keep, dtype=bool),
        "uv_column_v_keep": np.asarray(uv_column_v_keep, dtype=bool), "wall_endpoint_extend_pixels": np.asarray(cfg.wall_endpoint_extend_pixels),
    }

    print("\n有效层数:", len(repair_layers))
    print("补全体点云数量:", len(repair_volume_points))
    print("最大有效深度:", max([layer["z"] for layer in repair_layers]), "m")
    print("缺陷凹陷面候选点数量:", len(defect_surface_points))
    return CompletionData(repair_volume_points, repair_point_center, repair_layers, side_points, side_n_mark, defect_surface_points, fix_points, fix_points_curve, fix_npz_data, debug_records)




def run_part3(cfg: CompletionConfig):
    plane, roi, grid, corner_mode = load_part1_state(cfg.output_dir)
    completion = run_layered_flood_fill_from_saved_barriers(cfg, plane, grid, corner_mode)
    save_results(cfg, plane, roi, grid, completion, corner_mode)
    visualize_results(cfg, plane, completion)



if __name__ == "__main__":
    run_part3(config_from_args(parse_args()))

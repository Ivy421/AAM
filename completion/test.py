import json

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


RUN_DIR = "/home/smmg/AAM/data/runs/20260812_104525"
LAYER_ID = 12  # 0-based layer index. Set 12 to inspect the 12th saved z slice index.
FIX_PCD = f"{RUN_DIR}/completion/depression/fix_points_curve.pcd"
RAW_FIX_PCD = f"{RUN_DIR}/completion/depression/fix_points.pcd"
SEG_JSON = f"{RUN_DIR}/completion/depression/glue_brush_adaptive_segments.json"
PCD_RAW= f"{RUN_DIR}/construction/fine_scan/fine_fuse.pcd"
FIX_MASK = f"{RUN_DIR}/completion/depression/fix_mask.npz"
repair_block =  f"{RUN_DIR}/completion/depression/model.pcd"

mask_meta = np.load(FIX_MASK, allow_pickle=True)


def make_colored_pcd(points, color):
    pcd = o3d.geometry.PointCloud()
    points = np.asarray(points)
    if len(points) > 0:
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color(color)
    return pcd


def mask_to_points(mask, z):
    grid_res = float(mask_meta["grid_res"])
    u_min = float(mask_meta["u_min"])
    v_min = float(mask_meta["v_min"])
    origin = np.asarray(mask_meta["origin"], dtype=float)
    u_axis = np.asarray(mask_meta["u_axis"], dtype=float)
    v_axis = np.asarray(mask_meta["v_axis"], dtype=float)
    n_axis = np.asarray(mask_meta["n_axis"], dtype=float)

    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.empty((0, 3))
    u = u_min + (xs + 0.5) * grid_res
    v = v_min + (ys + 0.5) * grid_res
    return origin + u[:, None] * u_axis + v[:, None] * v_axis + float(z) * n_axis


def mask_stack_to_pcd(mask_name, color, valid_only=False):
    masks = mask_meta[mask_name].astype(bool)
    z_values = np.asarray(mask_meta["barrier_z_values"], dtype=float)
    valid_flags = np.asarray(mask_meta["barrier_valid_flags"], dtype=bool)
    pts_all = []
    for i, mask in enumerate(masks):
        if i >= len(z_values):
            break
        if valid_only and i < len(valid_flags) and not valid_flags[i]:
            continue
        pts = mask_to_points(mask, z_values[i])
        if len(pts) > 0:
            pts_all.append(pts)
    pcd = o3d.geometry.PointCloud()
    if len(pts_all) > 0:
        pcd.points = o3d.utility.Vector3dVector(np.vstack(pts_all))
        pcd.paint_uniform_color(color)
    return pcd

fixpcd = o3d.io.read_point_cloud(FIX_PCD)
rawfix_pcd = o3d.io.read_point_cloud(RAW_FIX_PCD)
pcd_raw = o3d.io.read_point_cloud(PCD_RAW)
repair_block_pcd = o3d.io.read_point_cloud(repair_block) 
fixpcd.paint_uniform_color([0.0, 1.0, 0.0])
pcd_raw.paint_uniform_color([0.0, 0.0, 1.0])

with open(SEG_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

barrier_raw_pcd = mask_stack_to_pcd("barrier_raw_masks", [1.0, 0.4, 0.0], valid_only=False)
barrier_curve_pcd = mask_stack_to_pcd("barrier_curve_masks", [1.0, 0.0, 1.0], valid_only=False)
barrier_processed_pcd = mask_stack_to_pcd("barrier_processed_masks", [1.0, 1.0, 0.0], valid_only=False)
fix_surface_pcd = mask_stack_to_pcd("fix_surface_masks", [0.0, 1.0, 1.0], valid_only=False)
fix_curve_mask_pcd = mask_stack_to_pcd("fix_surface_curve_masks", [1.0, 0.0, 1.0], valid_only=False)

masks_len = len(mask_meta["barrier_raw_masks"])
layer_id = int(np.clip(LAYER_ID, 0, masks_len - 1))
raw_wall_layer = mask_meta["barrier_raw_masks"][layer_id].astype(bool)
skeleton_layer = mask_meta["barrier_curve_masks"][layer_id].astype(bool)
final_wall_layer = mask_meta["barrier_processed_masks"][layer_id].astype(bool)
z_value = float(mask_meta["barrier_z_values"][layer_id])

plt.figure(f"Layer {layer_id} skeleton and final wall")
plt.imshow(raw_wall_layer, cmap="gray", origin="lower", alpha=0.45)
wy, wx = np.where(final_wall_layer)
sy, sx = np.where(skeleton_layer)
plt.scatter(wx, wy, s=5, c="yellow", label="final wall / barrier")
plt.scatter(sx, sy, s=7, c="magenta", label="skeleton curve")
plt.title(f"Layer {layer_id}, z={z_value:.4f} m: skeleton + final wall")
plt.legend()
plt.axis("equal")
plt.tight_layout()
plt.show()

skeleton_layer_pcd = make_colored_pcd(mask_to_points(skeleton_layer, z_value), [1.0, 0.0, 1.0])
final_wall_layer_pcd = make_colored_pcd(mask_to_points(final_wall_layer, z_value), [1.0, 1.0, 0.0])

geometries = [
    skeleton_layer_pcd,
    #final_wall_layer_pcd,
    fixpcd,
    #rawfix_pcd,
    pcd_raw,
    #repair_block_pcd

    #barrier_raw_pcd,
    #barrier_curve_pcd,
    #barrier_processed_pcd,
    #fix_surface_pcd,
    #fix_curve_mask_pcd,
]
for segment in data["segments"]:
    center_m = np.asarray(segment["center_point_base_mm"], dtype=float) / 1000.0
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
    sphere.translate(center_m)
    sphere.paint_uniform_color([1.0, 0.0, 0.0])
    geometries.append(sphere)

o3d.visualization.draw_geometries(geometries)

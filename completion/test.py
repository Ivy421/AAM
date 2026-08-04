import json

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


RUN_DIR = "E:/HKUSTGZ/AAM_MASTER/data/runs/20260801_153620"
FIX_PCD = f"{RUN_DIR}/completion/depression/fix_points_curve.pcd"
RAW_FIX_PCD = f"{RUN_DIR}/completion/depression/fix_points.pcd"
SEG_JSON = f"{RUN_DIR}/completion/depression/glue_brush_adaptive_segments.json"
PCD_RAW= f"{RUN_DIR}/construction/fine_scan/fine_fuse.pcd"
FIX_MASK = f"{RUN_DIR}/completion/depression/fix_mask.npz"

mask_meta = np.load(FIX_MASK, allow_pickle=True)


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
fixpcd.paint_uniform_color([0.0, 1.0, 0.0])
pcd_raw.paint_uniform_color([0.0, 0.0, 1.0])

with open(SEG_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

barrier_raw_pcd = mask_stack_to_pcd("barrier_raw_masks", [1.0, 0.4, 0.0], valid_only=False)
barrier_processed_pcd = mask_stack_to_pcd("barrier_processed_masks", [1.0, 1.0, 0.0], valid_only=False)
fix_surface_pcd = mask_stack_to_pcd("fix_surface_masks", [0.0, 1.0, 1.0], valid_only=False)
fix_curve_mask_pcd = mask_stack_to_pcd("fix_surface_curve_masks", [1.0, 0.0, 1.0], valid_only=False)

layer_id = 10  # 13th layer from top plane, 0-based index.
barrier_layer = mask_meta["barrier_raw_masks"][layer_id].astype(bool)  #barrier_raw_masks  barrier_processed_masks
fix_surface_layer = mask_meta["barrier_processed_masks"][layer_id].astype(bool)  #fix_surface_masks
z_value = float(mask_meta["barrier_z_values"][layer_id])

plt.figure("Layer 13 masks")
plt.imshow(barrier_layer, cmap="gray", origin="lower")
fy, fx = np.where(fix_surface_layer)
plt.scatter(fx, fy, s=4, c="red", label="fix_surface_mask")
plt.title(f"Layer 13, z={z_value:.4f} m: barrier mask + fix surface mask")
plt.legend()
plt.axis("equal")
plt.tight_layout()
plt.show()

geometries = [
    #fixpcd,
    #rawfix_pcd,
    pcd_raw,

    barrier_raw_pcd,
    #barrier_processed_pcd,
    #fix_surface_pcd,
    #fix_curve_mask_pcd,
]
for segment in data["segments"]:
    center_m = np.asarray(segment["center_point_base_mm"], dtype=float) / 1000.0
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.006)
    sphere.translate(center_m)
    sphere.paint_uniform_color([1.0, 0.0, 0.0])
    geometries.append(sphere)

o3d.visualization.draw_geometries(geometries)

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


# =========================
# Paths / parameters
# =========================
PROJECT_ROOT = Path(os.getenv("AAM_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
DATA_DIR = PROJECT_ROOT / "construction" / "data" / "coarse_scan"

COARSE_POINT_FILE = DATA_DIR / "coarse_point_result.npz"
COARSE_ICP_RESULT_FILE = DATA_DIR / "coarse_icp_result.json"
COARSE_FUSE_PCD_FILE = DATA_DIR / "coarse_fuse.pcd"
PNG_SEQUENCE_FILE = DATA_DIR / "coarse_png_sequence.json"

# 注意：你当前文件名里是 coner，不是 corner，这里保留兼容
CORNER_RESULT_FILE = DATA_DIR / "corner_mapping_result.json"
CORNER_RESULT_FILE_FALLBACK = DATA_DIR / "corner_mapping_result.json"
CORNER_LABEL_MAPPING_FILE = DATA_DIR / "corner_label_mapping.json"

# Camera intrinsic path，用于把 3D object points 投影到 target image 上
CAMERA_CONFIG_PATH = PROJECT_ROOT / "config" / "calibration" / "right_camera" / "camera_config.npy"

OUTPUT_ROI_JSON = DATA_DIR / "defect_roi_result.json"

# 选取离缺陷 corner pixel 最近的 object points 数量，要求 10-20 个，这里默认 15
NEAREST_K = 15
MIN_REQUIRED_POINTS = 10

# ROI cube: 150 x 150 x 150 mm = 0.15 m
ROI_BOX_SIZE_M = 0.150

# 可视化
VISUALIZE = False


def configure_paths(args):
    global DATA_DIR, COARSE_POINT_FILE, COARSE_ICP_RESULT_FILE, COARSE_FUSE_PCD_FILE
    global PNG_SEQUENCE_FILE, CORNER_RESULT_FILE, CORNER_RESULT_FILE_FALLBACK
    global CORNER_LABEL_MAPPING_FILE, CAMERA_CONFIG_PATH, OUTPUT_ROI_JSON, VISUALIZE

    if args.run_dir:
        DATA_DIR = Path(args.run_dir) / "construction" / "coarse_scan"
        CAMERA_CONFIG_PATH = Path(args.run_dir) / "camera_config.npy"

    if args.coarse_scan_dir:
        DATA_DIR = Path(args.coarse_scan_dir)
    if args.camera_config:
        CAMERA_CONFIG_PATH = Path(args.camera_config)

    COARSE_POINT_FILE = DATA_DIR / "coarse_point_result.npz"
    COARSE_ICP_RESULT_FILE = DATA_DIR / "coarse_icp_result.json"
    COARSE_FUSE_PCD_FILE = DATA_DIR / "coarse_fuse.pcd"
    PNG_SEQUENCE_FILE = DATA_DIR / "coarse_png_sequence.json"
    CORNER_RESULT_FILE = DATA_DIR / "corner_mapping_result.json"
    CORNER_RESULT_FILE_FALLBACK = DATA_DIR / "corner_mapping_result.json"
    CORNER_LABEL_MAPPING_FILE = DATA_DIR / "corner_label_mapping.json"
    OUTPUT_ROI_JSON = Path(args.output_json) if args.output_json else DATA_DIR / "defect_roi_result.json"
    VISUALIZE = bool(args.visualize)

    DATA_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Basic IO
# =========================
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_camera_intrinsic(path):
    config = np.load(path, allow_pickle=True).item()
    intrinsic = config["color_intrinsic"]

    # RealSense style dict: {fx, fy, ppx, ppy}
    if isinstance(intrinsic, dict):
        fx = float(intrinsic["fx"])
        fy = float(intrinsic["fy"])
        cx = float(intrinsic["ppx"])
        cy = float(intrinsic["ppy"])
        return fx, fy, cx, cy

    # 3x3 matrix style
    intrinsic = np.asarray(intrinsic, dtype=float)
    if intrinsic.shape == (3, 3):
        fx = float(intrinsic[0, 0])
        fy = float(intrinsic[1, 1])
        cx = float(intrinsic[0, 2])
        cy = float(intrinsic[1, 2])
        return fx, fy, cx, cy

    raise ValueError(f"Unsupported color_intrinsic format: {type(intrinsic)}, shape={getattr(intrinsic, 'shape', None)}")


def load_points_collection(npz_path):
    """
    支持两种保存方式：
    1. object array: points_collection=[Nx4, Mx4, ...]
    2. packed array: all_points + offsets
    """
    with np.load(npz_path, allow_pickle=True) as meta:
        keys = set(meta.files)

        if "points_collection" in keys:
            arr = meta["points_collection"]
            points_collection = [np.asarray(p, dtype=float) for p in arr]
        elif "all_points" in keys and "offsets" in keys:
            all_points = np.asarray(meta["all_points"], dtype=float)
            offsets = np.asarray(meta["offsets"], dtype=np.int64)
            points_collection = [all_points[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]
        else:
            raise KeyError("Cannot find points_collection or all_points/offsets in npz file.")

        if "bcT_collection" in keys:
            bcT_arr = meta["bcT_collection"]
            bcT_collection = [np.asarray(T, dtype=float) for T in bcT_arr]
        else:
            bcT_collection = None

    return points_collection, bcT_collection


def load_sequence_names():
    if PNG_SEQUENCE_FILE.exists():
        names = load_json(PNG_SEQUENCE_FILE)
        return list(names)

    # fallback: 根据 coarse_scan_*.png 排序
    png_files = sorted(DATA_DIR.glob("*.png"))
    return [p.stem for p in png_files]


# =========================
# Target frame / corner label
# =========================
def resolve_target_frame(icp_result, sequence_names):
    """
    优先使用 coarse_icp_result.json 中的 target_name，避免 target_index 的 0/1-based 混淆。
    """
    target_name = icp_result.get("target_name", None)
    target_index = icp_result.get("target_index", None)

    if target_name is not None and target_name in sequence_names:
        return int(sequence_names.index(target_name)), target_name

    if target_index is not None:
        target_index = int(target_index)
        if 0 <= target_index < len(sequence_names):
            return target_index, sequence_names[target_index]

    raise ValueError(
        f"Cannot resolve target frame. target_name={target_name}, "
        f"target_index={target_index}, sequence_names={sequence_names}"
    )


def load_defect_corner_pixel():
    corner_file = CORNER_RESULT_FILE if CORNER_RESULT_FILE.exists() else CORNER_RESULT_FILE_FALLBACK
    corner_result = load_json(corner_file)
    label = corner_result["label"]

    label_mapping = load_json(CORNER_LABEL_MAPPING_FILE)
    if label not in label_mapping:
        raise KeyError(f"Label {label} not found in {CORNER_LABEL_MAPPING_FILE}")

    info = label_mapping[label]
    pixel = info.get("corner_pixel", info.get("pixel", None))
    if pixel is None:
        raise KeyError(f"Label {label} has no corner_pixel / pixel field.")

    # corner_label_mapping.json 里的 pixel 是：
    # raw_pixel * VIS_SCALE + BORDER
    LABEL_VIS_SCALE = 1.2
    LABEL_BORDER = 120.0

    pixel_label_img = np.asarray(pixel, dtype=float)
    pixel_raw_img = (pixel_label_img - LABEL_BORDER) / LABEL_VIS_SCALE

    return label, pixel_raw_img, corner_result, info

# =========================
# Geometry
# =========================
def points_to_xyz(points):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Invalid points shape: {points.shape}")
    points = points[:, :3]
    valid = np.all(np.isfinite(points), axis=1)
    return points[valid]


def project_base_points_to_image(points_base, T_base_cam, image_shape, intrinsic):
    """
    points_base: Nx3 in world/base coordinate
    T_base_cam: 4x4 camera pose in base coordinate
    returns:
        uv_valid: Mx2
        points_valid: Mx3 corresponding base points
    """
    H, W = image_shape[:2]
    fx, fy, cx, cy = intrinsic

    T_cam_base = np.linalg.inv(T_base_cam)
    points_h = np.column_stack([points_base, np.ones(len(points_base))])
    points_cam = (T_cam_base @ points_h.T).T[:, :3]

    X = points_cam[:, 0]
    Y = points_cam[:, 1]
    Z = points_cam[:, 2]

    valid_z = np.isfinite(Z) & (Z > 1e-6)
    if not np.any(valid_z):
        return np.zeros((0, 2), dtype=float), np.zeros((0, 3), dtype=float)

    X = X[valid_z]
    Y = Y[valid_z]
    Z = Z[valid_z]
    points_base_valid = points_base[valid_z]

    u = fx * X / Z + cx
    v = fy * Y / Z + cy
    uv = np.column_stack([u, v])

    in_img = (
        (uv[:, 0] >= 0) & (uv[:, 0] < W) &
        (uv[:, 1] >= 0) & (uv[:, 1] < H)
    )

    return uv[in_img], points_base_valid[in_img]


#def nearest_object_points_to_corner(points_base, T_base_cam, image_path, corner_pixel, intrinsic, k=NEAREST_K):
    img = cv2.imread(str(image_path))

    uv, points_valid = project_base_points_to_image(
        points_base=points_base,
        T_base_cam=T_base_cam,
        image_shape=img.shape,
        intrinsic=intrinsic,
    )

    if len(uv) == 0:
        raise RuntimeError("No target-frame object points can be projected into the target image.")

    dist_px = np.linalg.norm(uv - corner_pixel.reshape(1, 2), axis=1)
    order = np.argsort(dist_px)
    k = min(int(k), len(order))

    nearest_idx = order[:k]
    nearest_points = points_valid[nearest_idx]
    nearest_uv = uv[nearest_idx]
    nearest_dist_px = dist_px[nearest_idx]

    if len(nearest_points) < MIN_REQUIRED_POINTS:
        print(
            f"WARNING: only {len(nearest_points)} nearest points found, "
            f"less than MIN_REQUIRED_POINTS={MIN_REQUIRED_POINTS}."
        )

    center = nearest_points.mean(axis=0)
    return center, nearest_points, nearest_uv, nearest_dist_px
#

def nearest_object_points_to_corner(points_base, T_base_cam, image_path, corner_pixel, intrinsic, k=NEAREST_K):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read target image: {image_path}")

    uv, points_valid = project_base_points_to_image(
        points_base=points_base,
        T_base_cam=T_base_cam,
        image_shape=img.shape,
        intrinsic=intrinsic,
    )

    if len(uv) == 0:
        raise RuntimeError("No target-frame object points can be projected into the target image.")

    dist_px = np.linalg.norm(uv - corner_pixel.reshape(1, 2), axis=1)
    order = np.argsort(dist_px)
    k = min(int(k), len(order))

    nearest_idx = order[:k]
    nearest_points = points_valid[nearest_idx]
    nearest_uv = uv[nearest_idx]
    nearest_dist_px = dist_px[nearest_idx]

    if len(nearest_points) < MIN_REQUIRED_POINTS:
        print(
            f"WARNING: only {len(nearest_points)} nearest points found, "
            f"less than MIN_REQUIRED_POINTS={MIN_REQUIRED_POINTS}."
        )

    # 3D ROI center: 最近点的三维平均
    center = nearest_points.mean(axis=0)

    # =========================
    # 2D visualization in target frame
    # =========================
    vis = img.copy()

    # 1. 可视化 corner pixel, 红色
    cx, cy = [int(round(v)) for v in corner_pixel]
    cv2.circle(vis, (cx, cy), 9, (0, 0, 255), -1)
    cv2.circle(vis, (cx, cy), 13, (255, 255, 255), 2)
    cv2.putText(
        vis,
        "C corner",
        (cx + 12, cy - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
    )

    # 2. 可视化 nearest_uv, 蓝色
    for i, (u, v) in enumerate(nearest_uv):
        x = int(round(float(u)))
        y = int(round(float(v)))
        cv2.circle(vis, (x, y), 5, (255, 0, 0), -1)
        cv2.circle(vis, (x, y), 7, (255, 255, 255), 1)

        if i < 5:
            cv2.putText(
                vis,
                f"n{i}",
                (x + 6, y + 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 0, 0),
                1,
            )

    # 3. 可视化 object center 投影点, 绿色
    # 这里用整份 points_base 的 percentile bbox center 估计 object center
    valid_points = np.asarray(points_base, dtype=float)
    valid_points = valid_points[np.all(np.isfinite(valid_points), axis=1)]
    obj_low = np.percentile(valid_points[:, :3], 2.0, axis=0)
    obj_high = np.percentile(valid_points[:, :3], 98.0, axis=0)
    object_center = 0.5 * (obj_low + obj_high)

    object_uv, _ = project_base_points_to_image(
        points_base=object_center.reshape(1, 3),
        T_base_cam=T_base_cam,
        image_shape=img.shape,
        intrinsic=intrinsic,
    )

    if len(object_uv) > 0:
        ox, oy = [int(round(v)) for v in object_uv[0]]
        cv2.circle(vis, (ox, oy), 9, (0, 255, 0), -1)
        cv2.circle(vis, (ox, oy), 13, (255, 255, 255), 2)
        cv2.putText(
            vis,
            "object center",
            (ox + 12, oy - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 180, 0),
            2,
        )

    # 4. 可视化 ROI center 投影点, 黄色
    center_uv, _ = project_base_points_to_image(
        points_base=center.reshape(1, 3),
        T_base_cam=T_base_cam,
        image_shape=img.shape,
        intrinsic=intrinsic,
    )

    if len(center_uv) > 0:
        rx, ry = [int(round(v)) for v in center_uv[0]]
        cv2.circle(vis, (rx, ry), 9, (0, 255, 255), -1)
        cv2.circle(vis, (rx, ry), 13, (0, 0, 0), 2)
        cv2.putText(
            vis,
            "ROI center",
            (rx + 12, ry + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 180, 180),
            2,
        )

    if VISUALIZE:
        cv2.imshow("target frame nearest_uv visualization", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        vis_path = DATA_DIR / "defect_roi_projection_vis.png"
        cv2.imwrite(str(vis_path), vis)

    return center, nearest_points, nearest_uv, nearest_dist_px

def make_axis_aligned_cube(center, size_m=ROI_BOX_SIZE_M):
    """
    Axis-aligned ROI cube in world/base coordinate.
    corner order:
        [xmin,ymin,zmin], [xmin,ymin,zmax], [xmin,ymax,zmin], [xmin,ymax,zmax],
        [xmax,ymin,zmin], [xmax,ymin,zmax], [xmax,ymax,zmin], [xmax,ymax,zmax]
    """
    center = np.asarray(center, dtype=float)
    half = float(size_m) / 2.0

    xs = [center[0] - half, center[0] + half]
    ys = [center[1] - half, center[1] + half]
    zs = [center[2] - half, center[2] + half]

    corners = []
    for x in xs:
        for y in ys:
            for z in zs:
                corners.append([x, y, z])

    return np.asarray(corners, dtype=float)


def make_cube_lineset(corners, color=(1.0, 0.0, 0.0)):
    # corners are ordered by nested loops x,y,z
    lines = [
        [0, 1], [0, 2], [0, 4],
        [3, 1], [3, 2], [3, 7],
        [5, 1], [5, 4], [5, 7],
        [6, 2], [6, 4], [6, 7],
    ]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for _ in lines])
    return line_set


def make_points_pcd(points, color=(0.0, 1.0, 0.0)):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float))
    pcd.paint_uniform_color(color)
    return pcd


# =========================
# Main
# =========================
def main():
    icp_result = load_json(COARSE_ICP_RESULT_FILE)
    sequence_names = load_sequence_names()
    target_index, target_name = resolve_target_frame(icp_result, sequence_names)
    target_image_path = DATA_DIR / f"{target_name}.png"
    print('11111::::   ', target_image_path)
    label, corner_pixel, corner_result, label_info = load_defect_corner_pixel()
    intrinsic = load_camera_intrinsic(CAMERA_CONFIG_PATH)

    points_collection, bcT_collection = load_points_collection(COARSE_POINT_FILE)
    if target_index >= len(points_collection):
        raise IndexError(f"target_index={target_index} out of range for points_collection length={len(points_collection)}")
    if bcT_collection is None or target_index >= len(bcT_collection):
        raise IndexError("bcT_collection missing or target_index out of range.")

    target_points_base = points_to_xyz(points_collection[target_index])
    T_base_cam_target = np.asarray(bcT_collection[target_index], dtype=float)

    fuse_pcd = o3d.io.read_point_cloud(COARSE_FUSE_PCD_FILE)
    fuse_point = np.asarray(fuse_pcd.points)
    roi_center, nearest_points, nearest_uv, nearest_dist_px = nearest_object_points_to_corner(
        #points_base=target_points_base,
        points_base=  fuse_point,
        T_base_cam=T_base_cam_target,
        image_path=target_image_path,
        corner_pixel=corner_pixel,
        intrinsic=intrinsic,
        k=NEAREST_K,
    )

    cube_corners = make_axis_aligned_cube(roi_center, size_m=ROI_BOX_SIZE_M)

    result = {
        "data_dir": str(DATA_DIR),
        "target_index": int(target_index),
        "target_name": target_name,
        "target_image_path": str(target_image_path),
        "defect_label": label,
        "corner_mode": corner_result.get("corner_mode", None),
        "corner_pixel_xy": [round(float(v), 3) for v in corner_pixel],
        "nearest_k": int(len(nearest_points)),
        "nearest_pixel_distances_px": [round(float(v), 3) for v in nearest_dist_px.tolist()],
        "defect_roi_center_world_m": [round(float(v), 6) for v in roi_center],
        "defect_roi_size_m": [ROI_BOX_SIZE_M, ROI_BOX_SIZE_M, ROI_BOX_SIZE_M],
        "defect_roi_size_mm": [ROI_BOX_SIZE_M * 1000.0] * 3,
        "defect_roi_cube_corners_world_m": [
            [round(float(v), 6) for v in corner] for corner in cube_corners
        ],
    }

    save_json(OUTPUT_ROI_JSON, result)

    print("========== Defect ROI result ==========")
    print(f"target_index: {target_index}")
    print(f"target_name:  {target_name}")
    print(f"label:        {label}")
    print(f"corner_pixel: {corner_pixel.tolist()}")
    print(f"nearest_k:    {len(nearest_points)}")
    print(f"ROI center [m]: {roi_center}")
    print("ROI cube corners [m]:")
    print(cube_corners)
    print(f"Saved result: {OUTPUT_ROI_JSON}")

    if VISUALIZE:
        geoms = []
        fused = o3d.io.read_point_cloud(str(COARSE_FUSE_PCD_FILE))
        fused.paint_uniform_color([0.65, 0.65, 0.65])
        geoms.append(fused)
        geoms.append(make_cube_lineset(cube_corners, color=(1.0, 0.0, 0.0)))
        geoms.append(make_points_pcd(nearest_points, color=(0.0, 0.0, 1.0)))
        geoms.append(make_points_pcd(roi_center.reshape(1, 3), color=(1.0, 0.0, 0.0)))

        o3d.visualization.draw_geometries(
            geoms,
            window_name="Defect ROI cube from corner pixel",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--coarse-scan-dir", type=Path, default=None)
    parser.add_argument("--camera-config", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--visualize", action="store_true")
    configure_paths(parser.parse_args())
    main()

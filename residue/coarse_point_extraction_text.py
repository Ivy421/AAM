import os
import gc
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch
from scipy.spatial.transform import Rotation as R

# Project import path
sys.path.append(r"/home/smmg/AAM")
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# =========================
# Fixed paths / parameters
# =========================
DATA_DIR = Path(r"/home/smmg/AAM/construction/data")
FIRST_STEM = "test"

# 第一帧数据
FIRST_IMAGE_PATH = DATA_DIR / f"{FIRST_STEM}.png"
FIRST_DEPTH_PATH = DATA_DIR / f"{FIRST_STEM}.npy"
FIRST_ENDPOSE_PATH = DATA_DIR / f"{FIRST_STEM}.json"
FIRST_POINT_RESULT_PATH = DATA_DIR / "point_result.npz"

# 后续 coarse scan 帧数据：每帧都有 png / npy / json
COARSE_SCAN_DIR = DATA_DIR / "coarse_scan"

# 输出
OUTPUT_POINT_FILE = COARSE_SCAN_DIR / "coarse_point_result.npz"
OUTPUT_SEQ_FILE = COARSE_SCAN_DIR / "coarse_png_sequence.json"
OUTPUT_DEBUG_FILE = COARSE_SCAN_DIR / "coarse_prompt_debug.json"
VIS_DIR = COARSE_SCAN_DIR / "vis_prompt_mask"

# 标定
CONFIG_DIR = Path(r"/home/smmg/AAM/config/calibration/right_camera")
CAMERA_CONFIG_PATH = CONFIG_DIR / "camera_config.npy"
HAND_EYE_PATH = CONFIG_DIR / "ecT.npy"       # T_ee_cam

# endpose json 单位换算
POSITION_SCALE = 1e-6     # raw position -> meter
ANGLE_SCALE = 1e-3        # raw angle -> degree
EULER_ORDER = "xyz"

# 深度过滤，单位 m
MIN_DEPTH_M = 0.10
MAX_DEPTH_M = 1.50

# SAM3 prompt 设置
BOX_MARGIN_PX = 25
MAX_PRIOR_PROJECT_POINTS = 8000
CONFIDENCE_THRESHOLD = 0.30
USE_BOX_PROMPT = True
USE_POINT_PROMPT = True
SAVE_VISUALIZATION = True

# 点云滤波参数
VOXEL_SIZE_M = 0.001
STAT_NB_NEIGHBORS = 50
STAT_STD_RATIO = 1.0
RADIUS_NB_POINTS = 50
RADIUS_M = 0.015


# =========================
# Basic utilities
# =========================
def load_camera_config(camera_config_path):
    config = np.load(camera_config_path, allow_pickle=True).item()
    color_intrinsic = config["color_intrinsic"]
    depth_scale = float(config.get("depth_scale", 0.001))
    return color_intrinsic, depth_scale


def load_matrix(path):
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.shape == () and arr.dtype == object:
        arr = arr.item()
    arr = np.asarray(arr, dtype=float)
    if arr.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix from {path}, got {arr.shape}")
    return arr


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_endpose_json(path):
    """Parse robot endpose json into T_base_ee."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    pose = {}
    if isinstance(raw, list):
        for item in raw:
            pose.update(item)
    elif isinstance(raw, dict):
        pose = raw
    else:
        raise ValueError("Endpose JSON must be a dict or a list of single-key dicts.")

    x = float(pose["x"]) * POSITION_SCALE
    y = float(pose["y"]) * POSITION_SCALE
    z = float(pose["z"]) * POSITION_SCALE
    rx = float(pose["rx"]) * ANGLE_SCALE
    ry = float(pose["ry"]) * ANGLE_SCALE
    rz = float(pose["rz"]) * ANGLE_SCALE

    T = np.eye(4)
    T[:3, :3] = R.from_euler(EULER_ORDER, [rx, ry, rz], degrees=True).as_matrix()
    T[:3, 3] = [x, y, z]
    return T


def transform_points(T, points_xyz):
    points_xyz = np.asarray(points_xyz, dtype=float)
    points_h = np.column_stack([points_xyz, np.ones(len(points_xyz))])
    return (T @ points_h.T).T[:, :3]


def depth_to_meter(depth, depth_scale):
    """Convert depth array to meter as robustly as possible."""
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

    if depth.dtype == np.uint16 or depth.dtype == np.uint32:
        return depth.astype(np.float32) * depth_scale

    depth = depth.astype(np.float32)
    valid = depth[depth > 0]
    if len(valid) == 0:
        return depth

    # 如果 float 深度最大值很大，通常说明仍是 mm。
    if float(np.nanmax(valid)) > 10.0:
        return depth / 1000.0

    return depth


# =========================
# First-frame prior
# =========================
def load_first_object_points():
    """
    Load first-frame object point cloud from point_result.npz.
    Expected format:
        points_collection[0] -> object points, usually Nx4 homogeneous in base coordinate.
    """
    if not FIRST_POINT_RESULT_PATH.exists():
        raise FileNotFoundError(f"Cannot find first-frame point result: {FIRST_POINT_RESULT_PATH}")

    meta = np.load(FIRST_POINT_RESULT_PATH, allow_pickle=True)
    points_collection = meta["points_collection"]

    points = points_collection[0]
    points = np.asarray(points, dtype=float)

    if points.ndim != 2 or points.shape[1] < 3:
        points = points.reshape(-1, points.shape[-1])

    points = points[:, :3]
    points = points[np.all(np.isfinite(points), axis=1)]

    if len(points) == 0:
        raise ValueError("points_collection[0] contains no valid points.")

    return points


def estimate_object_center_and_keypoints(points_base, percentile=2.0):
    """
    Use first-frame object point cloud as 3D prior.
    object_center is bbox percentile center.
    keypoints are center + a few small offsets in base-aligned local coordinates.
    """
    low = np.percentile(points_base, percentile, axis=0)
    high = np.percentile(points_base, 100.0 - percentile, axis=0)
    center = 0.5 * (low + high)
    size = high - low

    # 这些点都是 3D 先验点，用于重投影成 SAM3 positive point prompts。
    # 以 object_center 为主，再加几个靠近中心的小偏移点，避免单点落在遮挡/边缘上。
    offsets = np.array([
        [0.0, 0.0, 0.0],
        [0.20 * size[0], 0.0, 0.0],
        [-0.20 * size[0], 0.0, 0.0],
        [0.0, 0.20 * size[1], 0.0],
        [0.0, -0.20 * size[1], 0.0],
    ])
    keypoints = center[None, :] + offsets

    return center, size, keypoints


# =========================
# Projection prompt generation
# =========================
def project_points_base_to_image(points_base, T_base_cam, color_intrinsic, image_shape):
    """
    Reproject base-coordinate 3D points to current image.

    T_base_cam: current camera pose in base coordinate.
    points_base: Nx3.
    Returns:
        uv: Mx2 pixel coordinates
        z:  M depth in current camera coordinate
    """
    H, W = image_shape[:2]
    fx = float(color_intrinsic["fx"])
    fy = float(color_intrinsic["fy"])
    cx = float(color_intrinsic["ppx"])
    cy = float(color_intrinsic["ppy"])

    T_cam_base = np.linalg.inv(T_base_cam)
    points_cam = transform_points(T_cam_base, points_base)

    X = points_cam[:, 0]
    Y = points_cam[:, 1]
    Z = points_cam[:, 2]

    valid = np.isfinite(Z) & (Z > MIN_DEPTH_M) & (Z < MAX_DEPTH_M)
    if not np.any(valid):
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]

    u = fx * X / Z + cx
    v = fy * Y / Z + cy
    uv = np.column_stack([u, v])

    in_img = (
        (uv[:, 0] >= 0) & (uv[:, 0] < W) &
        (uv[:, 1] >= 0) & (uv[:, 1] < H)
    )

    return uv[in_img].astype(np.float32), Z[in_img].astype(np.float32)


def make_projected_box(uv, image_shape, margin_px=BOX_MARGIN_PX):
    if len(uv) == 0:
        return None

    H, W = image_shape[:2]
    x0 = max(0.0, float(np.min(uv[:, 0])) - margin_px)
    y0 = max(0.0, float(np.min(uv[:, 1])) - margin_px)
    x1 = min(float(W - 1), float(np.max(uv[:, 0])) + margin_px)
    y1 = min(float(H - 1), float(np.max(uv[:, 1])) + margin_px)

    if x1 <= x0 + 2 or y1 <= y0 + 2:
        return None

    return [x0, y0, x1, y1]


def make_point_prompts(object_keypoints_base, object_points_base, T_base_cam, color_intrinsic, image_shape):
    """
    Main positive prompt is the projected object_center.
    If the center/keypoints are not visible, fallback to median of projected prior points.
    """
    key_uv, _ = project_points_base_to_image(
        object_keypoints_base, T_base_cam, color_intrinsic, image_shape
    )

    if len(key_uv) > 0:
        # 去除过近重复点，避免多个 prompt 落在同一像素附近。
        pts = []
        for p in key_uv:
            if all(np.linalg.norm(p - q) > 8.0 for q in pts):
                pts.append(p)
        return np.asarray(pts[:5], dtype=np.float32)

    # fallback: use projected prior point cloud median
    prior_uv, _ = project_points_base_to_image(
        object_points_base, T_base_cam, color_intrinsic, image_shape
    )
    if len(prior_uv) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    median_uv = np.median(prior_uv, axis=0)
    return median_uv.reshape(1, 2).astype(np.float32)


def build_prompts_from_prior(object_points_base, object_keypoints_base, T_base_cam, color_intrinsic, image_shape):
    """Build point prompt and optional box prompt by reprojecting first-frame 3D prior."""
    # 为了生成 box，投影第一帧 object point cloud。
    points_for_box = object_points_base
    if len(points_for_box) > MAX_PRIOR_PROJECT_POINTS:
        idx = np.linspace(0, len(points_for_box) - 1, MAX_PRIOR_PROJECT_POINTS).astype(int)
        points_for_box = points_for_box[idx]

    prior_uv, _ = project_points_base_to_image(
        points_for_box, T_base_cam, color_intrinsic, image_shape
    )

    box = make_projected_box(prior_uv, image_shape) if USE_BOX_PROMPT else None
    points = make_point_prompts(
        object_keypoints_base, object_points_base, T_base_cam, color_intrinsic, image_shape
    ) if USE_POINT_PROMPT else np.zeros((0, 2), dtype=np.float32)

    return points, box, prior_uv


# =========================
# SAM3 mask and point cloud
# =========================
def _to_numpy(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _squeeze_masks(masks):
    masks = _to_numpy(masks)
    if masks is None:
        return None

    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]

    return masks


def _boxes_from_masks(masks):
    masks = _squeeze_masks(masks)

    if masks is None or len(masks) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    boxes = []
    for m in masks:
        m = np.squeeze(m).astype(bool)
        ys, xs = np.where(m)

        if len(xs) == 0 or len(ys) == 0:
            boxes.append([0, 0, 0, 0])
        else:
            boxes.append([xs.min(), ys.min(), xs.max(), ys.max()])

    return np.asarray(boxes, dtype=np.float32)


def init_sam3_model(confidence_threshold=CONFIDENCE_THRESHOLD):
    """
    只在程序开始时加载一次 SAM3。
    后续所有图片都复用这个 model / processor。
    """
    torch.cuda.empty_cache()
    gc.collect()

    try:
        model = build_sam3_image_model(enable_inst_interactivity=True)
    except TypeError:
        model = build_sam3_image_model()

    model.eval()
    processor = Sam3Processor(model, confidence_threshold=confidence_threshold)

    return model, processor


def sam3_point_box_predict(model, processor, image_path, point_prompts, box_prompt):
    """
    使用已经加载好的 SAM3 model / processor 做 point + box 分割。
    不在这里重新 build_sam3_image_model。
    """
    image = Image.open(str(image_path)).convert("RGB")

    point_arg = None
    point_labels = None

    if point_prompts is not None and len(point_prompts) > 0:
        point_arg = np.asarray(point_prompts, dtype=np.float32)
        if point_arg.ndim == 1:
            point_arg = point_arg[None, :]
        point_labels = np.ones((len(point_arg),), dtype=np.int32)

    # 如果没有 point，但是有 box，则用 box 中心补一个 positive point
    if point_arg is None and box_prompt is not None:
        x0, y0, x1, y1 = [float(v) for v in box_prompt]
        point_arg = np.array([[(x0 + x1) / 2.0, (y0 + y1) / 2.0]], dtype=np.float32)
        point_labels = np.ones((1,), dtype=np.int32)

    box_arg = None
    if box_prompt is not None:
        box_arg = np.asarray(box_prompt, dtype=np.float32).reshape(1, 4)

    if point_arg is None and box_arg is None:
        raise ValueError("No valid point or box prompt for SAM3.")

    with torch.inference_mode():
        inference_state = processor.set_image(image)

        masks, scores, _ = model.predict_inst(
            inference_state,
            point_coords=point_arg,
            point_labels=point_labels,
            box=box_arg,
            multimask_output=True,
        )

    masks = _squeeze_masks(masks)
    scores = _to_numpy(scores)
    boxes = _boxes_from_masks(masks)

    del inference_state
    torch.cuda.empty_cache()
    gc.collect()

    return masks, boxes, scores

def choose_best_mask(masks, boxes, scores, point_prompts):
    """Prefer masks covering projected positive points; then use SAM score."""
    masks = np.asarray(masks)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]

    if len(masks) == 0:
        return None, None, None, None

    scores_arr = np.asarray(scores).reshape(-1) if scores is not None else np.zeros((len(masks),))
    best_idx = 0
    best_metric = -1e9

    for i, m in enumerate(masks):
        m_bool = np.squeeze(m).astype(bool)
        cover_count = 0
        for x, y in np.asarray(point_prompts).reshape(-1, 2):
            xi = int(round(float(x)))
            yi = int(round(float(y)))
            if 0 <= yi < m_bool.shape[0] and 0 <= xi < m_bool.shape[1] and m_bool[yi, xi]:
                cover_count += 1

        score = float(scores_arr[i]) if i < len(scores_arr) else 0.0
        metric = cover_count * 10.0 + score
        if metric > best_metric:
            best_metric = metric
            best_idx = i

    mask = np.squeeze(masks[best_idx]).astype(bool)
    box = np.asarray(boxes[best_idx]).astype(float).tolist() if boxes is not None and len(boxes) > best_idx else None
    score = float(scores_arr[best_idx]) if best_idx < len(scores_arr) else 0.0
    return mask, box, score, int(best_idx)


def get_mask_by_projected_prompts(sam3_model, sam3_processor, image_path, point_prompts, box_prompt):
    """
    Use SAM3 point + box prompts generated by 3D prior reprojection.
    这里不再调用 positioning()，而是复用已经加载好的 SAM3 模型。
    """
    if len(point_prompts) == 0 and box_prompt is None:
        raise ValueError("No valid projected prompt. Cannot segment this frame.")

    masks, boxes, scores = sam3_point_box_predict(
        model=sam3_model,
        processor=sam3_processor,
        image_path=image_path,
        point_prompts=point_prompts,
        box_prompt=box_prompt,
    )

    return choose_best_mask(masks, boxes, scores, point_prompts)


def mask_to_points_base(image_path, depth_path, mask, T_base_cam, color_intrinsic, depth_scale):
    depth_raw = np.load(depth_path)
    depth = depth_to_meter(depth_raw, depth_scale)
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    H, W = depth.shape[:2]
    if mask.shape != depth.shape:
        mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)

    fx = float(color_intrinsic["fx"])
    fy = float(color_intrinsic["fy"])
    cx = float(color_intrinsic["ppx"])
    cy = float(color_intrinsic["ppy"])

    valid_depth = (depth > MIN_DEPTH_M) & (depth < MAX_DEPTH_M) & np.isfinite(depth)
    valid_mask = mask.astype(bool) & valid_depth

    v_valid, u_valid = np.nonzero(valid_mask)
    if len(u_valid) == 0:
        return np.zeros((0, 4), dtype=float), 0, 0

    z = depth[v_valid, u_valid]
    x = (u_valid - cx) * z / fx
    y = (v_valid - cy) * z / fy
    points_cam = np.column_stack([x, y, z])

    raw_count = len(points_cam)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_cam)
    pcd.colors = o3d.utility.Vector3dVector(img_rgb[v_valid, u_valid] / 255.0)

    if len(pcd.points) > 10:
        pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE_M)

    if len(pcd.points) > STAT_NB_NEIGHBORS:
        _, ind = pcd.remove_statistical_outlier(
            nb_neighbors=STAT_NB_NEIGHBORS,
            std_ratio=STAT_STD_RATIO,
            print_progress=False,
        )
        pcd = pcd.select_by_index(ind)

    if len(pcd.points) > RADIUS_NB_POINTS:
        _, ind = pcd.remove_radius_outlier(
            nb_points=RADIUS_NB_POINTS,
            radius=RADIUS_M,
            print_progress=False,
        )
        pcd = pcd.select_by_index(ind)

    points_cam_filtered = np.asarray(pcd.points)
    if len(points_cam_filtered) == 0:
        return np.zeros((0, 4), dtype=float), raw_count, 0

    points_base = transform_points(T_base_cam, points_cam_filtered)
    points_base_h = np.column_stack([points_base, np.ones(len(points_base))])
    return points_base_h, raw_count, len(points_base_h)


# =========================
# Visualization
# =========================
def save_prompt_visualization(image_path, depth_path, mask, box_prompt, point_prompts, selected_box, selected_score, save_path):
    img = cv2.imread(str(image_path))
    if img is None:
        return

    vis = img.copy()
    if mask is not None:
        m = mask.astype(bool)
        if m.shape[:2] != vis.shape[:2]:
            m = cv2.resize(m.astype(np.uint8), (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        red = np.zeros_like(vis)
        red[m] = (0, 0, 255)
        vis = cv2.addWeighted(vis, 1.0, red, 0.45, 0)

    if box_prompt is not None:
        x0, y0, x1, y1 = [int(round(v)) for v in box_prompt]
        cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 255, 0), 2)
        cv2.putText(vis, "projected box", (x0, max(0, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    if selected_box is not None:
        x0, y0, x1, y1 = [int(round(v)) for v in selected_box]
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(vis, f"SAM box {selected_score:.3f}", (x0, min(vis.shape[0] - 5, y1 + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    for i, (x, y) in enumerate(point_prompts):
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        cv2.circle(vis, (xi, yi), 5, (0, 255, 255), -1)
        cv2.putText(vis, f"p{i}", (xi + 6, yi - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), vis)


# =========================
# Per-frame / main extraction
# =========================
def process_one_frame(
    name,
    object_points_base,
    object_keypoints_base,
    color_intrinsic,
    depth_scale,
    T_ee_cam,
    sam3_model,
    sam3_processor,
):
    image_path = COARSE_SCAN_DIR / f"{name}.png"
    depth_path = COARSE_SCAN_DIR / f"{name}.npy"
    endpose_path = COARSE_SCAN_DIR / f"{name}.json"

    if not image_path.exists() or not depth_path.exists() or not endpose_path.exists():
        raise FileNotFoundError(f"Missing png/npy/json for frame: {name}")

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    T_base_ee = parse_endpose_json(endpose_path)
    T_base_cam = T_base_ee @ T_ee_cam

    point_prompts, box_prompt, prior_uv = build_prompts_from_prior(
        object_points_base=object_points_base,
        object_keypoints_base=object_keypoints_base,
        T_base_cam=T_base_cam,
        color_intrinsic=color_intrinsic,
        image_shape=img.shape,
    )

    mask, selected_box, selected_score, selected_idx = get_mask_by_projected_prompts(
        sam3_model=sam3_model,
        sam3_processor=sam3_processor,
        image_path=image_path,
        point_prompts=point_prompts,
        box_prompt=box_prompt  )

    if mask is None:
        return np.zeros((0, 4), dtype=float), T_base_cam, {
            "name": name,
            "success": False,
            "reason": "SAM3 returned no mask",
        }

    points_base_h, raw_count, filtered_count = mask_to_points_base(
        image_path=image_path,
        depth_path=depth_path,
        mask=mask,
        T_base_cam=T_base_cam,
        color_intrinsic=color_intrinsic,
        depth_scale=depth_scale,
    )

    if SAVE_VISUALIZATION:
        save_prompt_visualization(
            image_path=image_path,
            depth_path=depth_path,
            mask=mask,
            box_prompt=box_prompt,
            point_prompts=point_prompts,
            selected_box=selected_box,
            selected_score=selected_score if selected_score is not None else 0.0,
            save_path=VIS_DIR / f"{name}_prompt_mask.png",
        )

    debug = {
        "name": name,
        "success": bool(filtered_count > 0),
        "point_prompts_xy": [[round(float(x), 2), round(float(y), 2)] for x, y in point_prompts],
        "box_prompt_xyxy": [round(float(v), 2) for v in box_prompt] if box_prompt is not None else None,
        "projected_prior_points_count": int(len(prior_uv)),
        "selected_mask_idx": selected_idx,
        "selected_score": round(float(selected_score), 4) if selected_score is not None else None,
        "selected_box_xyxy": [round(float(v), 2) for v in selected_box] if selected_box is not None else None,
        "raw_points_count": int(raw_count),
        "filtered_points_count": int(filtered_count),
    }

    return points_base_h, T_base_cam, debug


def extract_coarse_points():
    torch.cuda.empty_cache()
    gc.collect()

    COARSE_SCAN_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    color_intrinsic, depth_scale = load_camera_config(CAMERA_CONFIG_PATH)
    T_ee_cam = load_matrix(HAND_EYE_PATH)
    sam3_model, sam3_processor = init_sam3_model()

    # 第一帧先验：points_collection[0]
    object_points_base = load_first_object_points()
    object_center, object_size, object_keypoints_base = estimate_object_center_and_keypoints(object_points_base)

    png_files = sorted(COARSE_SCAN_DIR.glob("*.png"))
    png_names = [p.stem for p in png_files]

    points_collection = []
    bcT_collection = []
    valid_png_names = []
    debug_records = []

    print("========== First-frame object prior ==========")
    print(f"object_center_base_m: {object_center}")
    print(f"object_size_base_m:   {object_size}")
    print(f"coarse frames: {len(png_names)}")

    for name in png_names:
        print(f"\nProcessing coarse frame: {name}.png")
        try:
            points_base_h, T_base_cam, debug = process_one_frame(
                name=name,
                object_points_base=object_points_base,
                object_keypoints_base=object_keypoints_base,
                color_intrinsic=color_intrinsic,
                depth_scale=depth_scale,
                T_ee_cam=T_ee_cam,
                sam3_model=sam3_model,
                sam3_processor=sam3_processor,
            )
        except Exception as e:
            print(f"  FAILED: {e}")
            debug_records.append({"name": name, "success": False, "reason": str(e)})
            continue

        debug_records.append(debug)

        if len(points_base_h) == 0:
            print("  No valid object points extracted.")
            continue

        points_collection.append(points_base_h)
        bcT_collection.append(T_base_cam)
        valid_png_names.append(name)

        print(
            f"  success={debug['success']}, "
            f"raw={debug['raw_points_count']}, "
            f"filtered={debug['filtered_points_count']}, "
            f"prompts={debug['point_prompts_xy']}"
        )

    np.savez(
        OUTPUT_POINT_FILE,
        points_collection=np.asarray(points_collection, dtype=object),
        bcT_collection=np.asarray(bcT_collection, dtype=object),
    )

    save_json(OUTPUT_SEQ_FILE, valid_png_names)
    save_json(OUTPUT_DEBUG_FILE, {
        "object_center_base_m": [round(float(v), 6) for v in object_center],
        "object_size_base_m": [round(float(v), 6) for v in object_size],
        "valid_png_names": valid_png_names,
        "records": debug_records,
    })

    print("\n========== Coarse point extraction result ==========")
    print(f"Saved point cloud npz: {OUTPUT_POINT_FILE}")
    print(f"Saved png sequence:    {OUTPUT_SEQ_FILE}")
    print(f"Saved debug json:      {OUTPUT_DEBUG_FILE}")
    print(f"Valid frames: {len(valid_png_names)} / {len(png_names)}")


if __name__ == "__main__":
    extract_coarse_points()

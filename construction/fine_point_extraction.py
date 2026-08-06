import gc
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

PROJECT_ROOT = Path(os.getenv("AAM_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
sys.path.append(str(PROJECT_ROOT))


# =========================
# Paths / parameters
# =========================
DATA_DIR = PROJECT_ROOT / "construction" / "data"
FINE_SCAN_DIR = DATA_DIR / "fine_scan"
ROI_JSON = DATA_DIR / "coarse_scan" / "defect_roi_result.json"
FINE_SCANPOSE_PATH = FINE_SCAN_DIR / "fine_scanpose.json"
FIRST_POINT_RESULT_PATH = DATA_DIR / "initial_frame_point.npz"
SAFE_PROMPT_ALPHA = 0.5   # 0.3~0.6 都可以；越大越靠近 object center

OUTPUT_POINT_FILE = FINE_SCAN_DIR / "fine_point_result.npz"
OUTPUT_SEQ_FILE = FINE_SCAN_DIR / "fine_png_sequence.json"
OUTPUT_DEBUG_FILE = FINE_SCAN_DIR / "fine_prompt_debug.json"
VIS_DIR = FINE_SCAN_DIR / "vis_prompt_mask"

CONFIG_DIR = PROJECT_ROOT / "config" / "calibration" / "right_camera"
CAMERA_CONFIG_PATH = CONFIG_DIR / "camera_config.npy"
HAND_EYE_PATH = CONFIG_DIR / "ecT.npy"       # T_ee_cam

POSITION_SCALE = 1e-6     # raw position -> meter
ANGLE_SCALE = 1e-3        # raw angle -> degree
EULER_ORDER = "xyz"

MIN_DEPTH_M = 0.10
MAX_DEPTH_M = 1.50

CONFIDENCE_THRESHOLD = 0.30
TEXT_PROMPT = "a black rectangular-shaped wooden block"
SAVE_VISUALIZATION = True

VOXEL_SIZE_M = 0.001
STAT_NB_NEIGHBORS = 50
STAT_STD_RATIO = 1.0
RADIUS_NB_POINTS = 50
RADIUS_M = 0.015


def configure_paths(args):
    global DATA_DIR, FINE_SCAN_DIR, ROI_JSON, FINE_SCANPOSE_PATH, FIRST_POINT_RESULT_PATH
    global OUTPUT_POINT_FILE, OUTPUT_SEQ_FILE, OUTPUT_DEBUG_FILE, VIS_DIR
    global CAMERA_CONFIG_PATH, HAND_EYE_PATH, TEXT_PROMPT

    if args.run_dir:
        DATA_DIR = Path(args.run_dir) / "construction"
        FINE_SCAN_DIR = DATA_DIR / "fine_scan"
        ROI_JSON = DATA_DIR / "coarse_scan" / "defect_roi_result.json"
        FINE_SCANPOSE_PATH = FINE_SCAN_DIR / "fine_scanpose.json"
        FIRST_POINT_RESULT_PATH = DATA_DIR / "initial_frame_point.npz"
        CAMERA_CONFIG_PATH = Path(args.run_dir) / "camera_config.npy"
        HAND_EYE_PATH = PROJECT_ROOT / "config" / "calibration" / "right_camera" / "ecT.npy"

    if args.fine_scan_dir:
        FINE_SCAN_DIR = Path(args.fine_scan_dir)
    if args.roi_json:
        ROI_JSON = Path(args.roi_json)
    if args.first_point_result:
        FIRST_POINT_RESULT_PATH = Path(args.first_point_result)
    if args.output_npz:
        OUTPUT_POINT_FILE = Path(args.output_npz)
    else:
        OUTPUT_POINT_FILE = FINE_SCAN_DIR / "fine_point_result.npz"
    if args.camera_config:
        CAMERA_CONFIG_PATH = Path(args.camera_config)
    if args.hand_eye:
        HAND_EYE_PATH = Path(args.hand_eye)
    if args.text_prompt:
        TEXT_PROMPT = args.text_prompt

    OUTPUT_SEQ_FILE = FINE_SCAN_DIR / "fine_png_sequence.json"
    OUTPUT_DEBUG_FILE = FINE_SCAN_DIR / "fine_prompt_debug.json"
    FINE_SCANPOSE_PATH = FINE_SCAN_DIR / "fine_scanpose.json"
    VIS_DIR = FINE_SCAN_DIR / "vis_prompt_mask"
    FINE_SCAN_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Basic utilities
# =========================
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_camera_config(path):
    config = np.load(path, allow_pickle=True).item()
    color_intrinsic = config["color_intrinsic"]
    depth_scale = float(config.get("depth_scale", 0.001))
    return color_intrinsic, depth_scale


def load_matrix(path):
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.shape == () and arr.dtype == object:
        arr = arr.item()
    return np.asarray(arr, dtype=float)


def parse_endpose_json(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    pose = {}
    if isinstance(raw, list):
        for item in raw:
            pose.update(item)
    else:
        pose = raw

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
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    if depth.dtype == np.uint16 or depth.dtype == np.uint32:
        return depth.astype(np.float32) * depth_scale

    depth = depth.astype(np.float32)
    valid = depth[depth > 0]
    if len(valid) == 0:
        return depth
    if float(np.nanmax(valid)) > 10.0:
        return depth / 1000.0
    return depth


def load_safe_prompt_point():
    """
    prompt point 不直接用 defect center，
    而是把 defect center 往 object center 拉回一段距离。
    """
    roi_data = load_json(ROI_JSON)
    defect_center = np.asarray(
        roi_data["defect_roi_center_world_m"],
        dtype=float
    ).reshape(3)

    meta = np.load(FIRST_POINT_RESULT_PATH, allow_pickle=True)
    object_center = np.asarray(meta["object_center"], dtype=float).reshape(-1)[:3]

    safe_prompt = defect_center + SAFE_PROMPT_ALPHA * (object_center - defect_center)

    return (
        defect_center.reshape(1, 3),
        object_center.reshape(1, 3),
        safe_prompt.reshape(1, 3),
    )


def load_fine_prompt_targets(default_prompt_base):
    scanpose_records = load_json(FINE_SCANPOSE_PATH)
    prompt_targets = {}
    for i, rec in enumerate(scanpose_records, start=1):
        name = f"fine_scan_{i}"
        target = rec.get("look_target_base_m", default_prompt_base.reshape(3))
        prompt_targets[name] = {
            "prompt_point_base": np.asarray(target, dtype=float).reshape(1, 3),
            "cube_name": rec.get("cube_name", ""),
            "is_dev_view": bool(rec.get("is_dev_view", False)),
        }
    return prompt_targets


# =========================
# Projection prompt
# =========================
def project_points_base_to_image(points_base, T_base_cam, color_intrinsic, image_shape):
    H, W = image_shape[:2]
    fx = float(color_intrinsic["fx"])
    fy = float(color_intrinsic["fy"])
    cx = float(color_intrinsic["ppx"])
    cy = float(color_intrinsic["ppy"])

    T_cam_base = np.linalg.inv(T_base_cam)
    points_cam = transform_points(T_cam_base, points_base)

    X, Y, Z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
    valid = np.isfinite(Z) & (Z > MIN_DEPTH_M) & (Z < MAX_DEPTH_M)
    if not np.any(valid):
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    X, Y, Z = X[valid], Y[valid], Z[valid]
    u = fx * X / Z + cx
    v = fy * Y / Z + cy
    uv = np.column_stack([u, v])

    in_img = (
        (uv[:, 0] >= 0) & (uv[:, 0] < W) &
        (uv[:, 1] >= 0) & (uv[:, 1] < H)
    )
    return uv[in_img].astype(np.float32), Z[in_img].astype(np.float32)


def make_point_prompt(prompt_point_base, T_base_cam, color_intrinsic, image_shape):
    uv, _ = project_points_base_to_image(
        prompt_point_base,
        T_base_cam,
        color_intrinsic,
        image_shape,
    )
    if len(uv) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return uv[:1].astype(np.float32)


def fine_frame_sort_key(path):
    stem = path.stem
    try:
        return (0, int(stem.rsplit("_", 1)[-1]))
    except ValueError:
        return (1, stem)


# =========================
# SAM3 text + point selection
# =========================
def _to_numpy(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        x = x.detach()
        if x.dtype == torch.bfloat16 or x.dtype == torch.float16:
            x = x.float()
        return x.cpu().numpy()
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
    for mask in masks:
        ys, xs = np.where(np.squeeze(mask).astype(bool))
        boxes.append([xs.min(), ys.min(), xs.max(), ys.max()] if len(xs) else [0, 0, 0, 0])
    return np.asarray(boxes, dtype=np.float32)


def init_sam3_model():
    torch.cuda.empty_cache()
    gc.collect()
    try:
        model = build_sam3_image_model(enable_inst_interactivity=True)
    except TypeError:
        model = build_sam3_image_model()
    model.eval()
    processor = Sam3Processor(model, confidence_threshold=CONFIDENCE_THRESHOLD)
    return model, processor


def sam3_text_predict(model, processor, image_path, text_prompt=TEXT_PROMPT):
    image = Image.open(str(image_path)).convert("RGB")
    with torch.inference_mode():
        state = processor.set_image(image)
        output = processor.set_text_prompt(state=state, prompt=text_prompt)

    masks = _squeeze_masks(output["masks"])
    boxes = _to_numpy(output["boxes"])
    scores = _to_numpy(output["scores"])

    del state, output
    torch.cuda.empty_cache()
    gc.collect()
    return masks, boxes, scores


def sam3_point_predict(model, processor, image_path, point_prompts):
    image = Image.open(str(image_path)).convert("RGB")
    point_coords = np.asarray(point_prompts, dtype=np.float32).reshape(-1, 2)
    point_labels = np.ones((len(point_coords),), dtype=np.int32)

    with torch.inference_mode():
        state = processor.set_image(image)
        masks, scores, _ = model.predict_inst(
            state,
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )

    masks = _squeeze_masks(masks)
    boxes = _boxes_from_masks(masks)
    scores = _to_numpy(scores)

    del state
    torch.cuda.empty_cache()
    gc.collect()
    return masks, boxes, scores


def choose_best_mask(masks, boxes, scores, point_prompts):
    masks = np.asarray(masks)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if len(masks) == 0:
        return None, None, None, None

    scores_arr = np.asarray(scores).reshape(-1) if scores is not None else np.zeros((len(masks),))
    best_idx, best_metric = 0, -1e9

    for i, m in enumerate(masks):
        m_bool = np.squeeze(m).astype(bool)
        cover_count = 0
        for x, y in np.asarray(point_prompts).reshape(-1, 2):
            xi, yi = int(round(float(x))), int(round(float(y)))
            if 0 <= yi < m_bool.shape[0] and 0 <= xi < m_bool.shape[1] and m_bool[yi, xi]:
                cover_count += 1

        score = float(scores_arr[i]) if i < len(scores_arr) else 0.0
        metric = cover_count * 10.0 + score
        if metric > best_metric:
            best_idx, best_metric = i, metric

    mask = np.squeeze(masks[best_idx]).astype(bool)
    box = np.asarray(boxes[best_idx]).astype(float).tolist() if boxes is not None and len(boxes) > best_idx else None
    score = float(scores_arr[best_idx]) if best_idx < len(scores_arr) else 0.0
    return mask, box, score, int(best_idx)


def get_mask_by_point_then_text(model, processor, image_path, point_prompts):
    masks, boxes, scores = sam3_point_predict(model, processor, image_path, point_prompts)
    mask, box, score, idx = choose_best_mask(masks, boxes, scores, point_prompts)
    if mask is not None:
        return mask, box, score, idx, "point"

    masks, boxes, scores = sam3_text_predict(model, processor, image_path, TEXT_PROMPT)
    mask, box, score, idx = choose_best_mask(masks, boxes, scores, point_prompts)
    return mask, box, score, idx, "text_fallback"


# =========================
# Mask to point cloud
# =========================
def mask_to_points_base(image_path, depth_path, mask, T_base_cam, color_intrinsic, depth_scale):
    depth_raw = np.load(depth_path)
    depth = depth_to_meter(depth_raw, depth_scale)

    img_bgr = cv2.imread(str(image_path))
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
        _, ind = pcd.remove_statistical_outlier(nb_neighbors=STAT_NB_NEIGHBORS, std_ratio=STAT_STD_RATIO, print_progress=False)
        pcd = pcd.select_by_index(ind)
    if len(pcd.points) > RADIUS_NB_POINTS:
        _, ind = pcd.remove_radius_outlier(nb_points=RADIUS_NB_POINTS, radius=RADIUS_M, print_progress=False)
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
def save_prompt_visualization(image_path, mask, point_prompts, selected_box, selected_score, save_path):
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

    if selected_box is not None:
        x0, y0, x1, y1 = [int(round(v)) for v in selected_box]
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(vis, f"SAM box {selected_score:.3f}", (x0, min(vis.shape[0] - 5, y1 + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    for i, (x, y) in enumerate(point_prompts):
        xi, yi = int(round(float(x))), int(round(float(y)))
        cv2.circle(vis, (xi, yi), 5, (0, 255, 255), -1)
        cv2.putText(vis, f"p{i}", (xi + 6, yi - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), vis)


# =========================
# Per-frame / main
# =========================
def process_one_frame(name, prompt_point_base, prompt_meta, color_intrinsic, depth_scale, T_ee_cam, model, processor):
    image_path = FINE_SCAN_DIR / f"{name}.png"
    depth_path = FINE_SCAN_DIR / f"{name}.npy"
    endpose_path = FINE_SCAN_DIR / f"{name}.json"

    img = cv2.imread(str(image_path))
    T_base_ee = parse_endpose_json(endpose_path)
    T_base_cam = T_base_ee @ T_ee_cam

    point_prompts = make_point_prompt(
        prompt_point_base=prompt_point_base,
        T_base_cam=T_base_cam,
        color_intrinsic=color_intrinsic,
        image_shape=img.shape,
    )

    if len(point_prompts) == 0:
        return np.zeros((0, 4), dtype=float), T_base_cam, {
            "name": name,
            "success": False,
            "reason": "defect center projected outside image",
            "cube_name": prompt_meta.get("cube_name", ""),
            "is_dev_view": bool(prompt_meta.get("is_dev_view", False)),
            "prompt_point_base_m": [round(float(v), 6) for v in prompt_point_base.reshape(3)],
        }

    mask, selected_box, selected_score, selected_idx, segmentation_mode = get_mask_by_point_then_text(
        model=model,
        processor=processor,
        image_path=image_path,
        point_prompts=point_prompts,
    )

    if mask is None:
        return np.zeros((0, 4), dtype=float), T_base_cam, {
            "name": name,
            "success": False,
            "reason": "SAM3 returned no mask",
            "segmentation_mode": segmentation_mode,
            "cube_name": prompt_meta.get("cube_name", ""),
            "is_dev_view": bool(prompt_meta.get("is_dev_view", False)),
            "prompt_point_base_m": [round(float(v), 6) for v in prompt_point_base.reshape(3)],
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
            mask=mask,
            point_prompts=point_prompts,
            selected_box=selected_box,
            selected_score=selected_score if selected_score is not None else 0.0,
            save_path=VIS_DIR / f"{name}_prompt_mask.png",
        )

    debug = {
        "name": name,
        "success": bool(filtered_count > 0),
        "segmentation_mode": segmentation_mode,
        "cube_name": prompt_meta.get("cube_name", ""),
        "is_dev_view": bool(prompt_meta.get("is_dev_view", False)),
        "prompt_point_base_m": [round(float(v), 6) for v in prompt_point_base.reshape(3)],
        "text_prompt": TEXT_PROMPT,
        "point_prompts_xy": [[round(float(x), 2), round(float(y), 2)] for x, y in point_prompts],
        "selected_mask_idx": selected_idx,
        "selected_score": round(float(selected_score), 4) if selected_score is not None else None,
        "selected_box_xyxy": [round(float(v), 2) for v in selected_box] if selected_box is not None else None,
        "raw_points_count": int(raw_count),
        "filtered_points_count": int(filtered_count),
    }
    return points_base_h, T_base_cam, debug


def extract_fine_points():
    torch.cuda.empty_cache()
    gc.collect()

    VIS_DIR.mkdir(parents=True, exist_ok=True)
    color_intrinsic, depth_scale = load_camera_config(CAMERA_CONFIG_PATH)
    T_ee_cam = load_matrix(HAND_EYE_PATH)
    defect_center_base, object_center_base, safe_prompt_base = load_safe_prompt_point()

    sam3_model, sam3_processor = init_sam3_model()

    png_files = sorted(FINE_SCAN_DIR.glob("*.png"), key=fine_frame_sort_key)
    png_names = [p.stem for p in png_files]

    points_collection = []
    bcT_collection = []
    valid_png_names = []
    debug_records = []

    for name in png_names:
        print(f"\nProcessing fine frame: {name}.png")
        prompt_meta = {
            "prompt_point_base": object_center_base,
            "cube_name": "",
            "is_dev_view": False,
        }
        try:
            points_base_h, T_base_cam, debug = process_one_frame(
                name=name,
                prompt_point_base=prompt_meta["prompt_point_base"],
                prompt_meta=prompt_meta,
                color_intrinsic=color_intrinsic,
                depth_scale=depth_scale,
                T_ee_cam=T_ee_cam,
                model=sam3_model,
                processor=sam3_processor,
            )
        except Exception as e:
            print(f"  FAILED: {e}")
            debug_records.append({"name": name, "success": False, "reason": str(e)})
            continue

        debug_records.append(debug)
        if len(points_base_h) == 0:
            print("  No valid fine points extracted.")
            continue

        points_collection.append(points_base_h)
        bcT_collection.append(T_base_cam)
        valid_png_names.append(name)

        print(
            f"  success={debug['success']}, "
            f"raw={debug['raw_points_count']}, "
            f"filtered={debug['filtered_points_count']}, "
            f"prompt={debug['point_prompts_xy']}"
        )

    points_collection_obj = np.empty(len(points_collection), dtype=object)
    for i, pts in enumerate(points_collection):
        points_collection_obj[i] = np.asarray(pts, dtype=np.float64)

    bcT_collection_arr = np.asarray(bcT_collection, dtype=np.float64)
    lengths = np.asarray([len(pts) for pts in points_collection], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    all_points = np.concatenate([np.asarray(pts, dtype=np.float64) for pts in points_collection], axis=0) if len(points_collection) else np.zeros((0, 4))

    np.savez(
        OUTPUT_POINT_FILE,
        points_collection=points_collection_obj,
        bcT_collection=bcT_collection_arr,
        all_points=all_points,
        offsets=offsets,
        valid_png_names=np.asarray(valid_png_names, dtype=object),
        defect_roi_center_world_m=defect_center_base.reshape(3),
        object_center_world_m=object_center_base.reshape(3),
        safe_prompt_world_m=safe_prompt_base.reshape(3),
        safe_prompt_alpha=SAFE_PROMPT_ALPHA,
        prompt_targets_world_m=np.asarray(
            [object_center_base.reshape(3) for _ in valid_png_names],
            dtype=np.float64,
        ),
    )

    save_json(OUTPUT_SEQ_FILE, valid_png_names)
    save_json(OUTPUT_DEBUG_FILE, {
    "defect_roi_center_world_m": [round(float(v), 6) for v in defect_center_base.reshape(3)],
    "object_center_world_m": [round(float(v), 6) for v in object_center_base.reshape(3)],
    "safe_prompt_world_m": [round(float(v), 6) for v in safe_prompt_base.reshape(3)],
    "safe_prompt_alpha": SAFE_PROMPT_ALPHA,
    "prompt_source": "object_center",
    "valid_png_names": valid_png_names,
    "records": debug_records,
    })

    print("\n========== Fine point extraction result ==========")
    print(f"Saved point cloud npz: {OUTPUT_POINT_FILE}")
    print(f"Saved png sequence:    {OUTPUT_SEQ_FILE}")
    print(f"Saved debug json:      {OUTPUT_DEBUG_FILE}")
    print(f"Valid frames: {len(valid_png_names)} / {len(png_names)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--fine-scan-dir", type=Path, default=None)
    parser.add_argument("--roi-json", type=Path, default=None)
    parser.add_argument("--first-point-result", type=Path, default=None)
    parser.add_argument("--output-npz", type=Path, default=None)
    parser.add_argument("--camera-config", type=Path, default=None)
    parser.add_argument("--hand-eye", type=Path, default=None)
    parser.add_argument("--text-prompt", default=None)
    configure_paths(parser.parse_args())
    extract_fine_points()

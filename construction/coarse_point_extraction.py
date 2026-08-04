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

PROJECT_ROOT = Path(os.getenv("AAM_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
sys.path.append(str(PROJECT_ROOT))
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# =========================
# Fixed paths / parameters
# =========================
DATA_DIR = PROJECT_ROOT / "construction" / "data"
FIRST_STEM = "front"

FIRST_POINT_RESULT_PATH = DATA_DIR / "initial_frame_point.npz"
COARSE_SCAN_DIR = DATA_DIR / "coarse_scan"

OUTPUT_POINT_FILE = COARSE_SCAN_DIR / "coarse_point_result.npz"
OUTPUT_SEQ_FILE = COARSE_SCAN_DIR / "coarse_png_sequence.json"
OUTPUT_DEBUG_FILE = COARSE_SCAN_DIR / "coarse_prompt_debug.json"
VIS_DIR = COARSE_SCAN_DIR / "vis_prompt_mask"

CONFIG_DIR = PROJECT_ROOT / "config" / "calibration" / "right_camera"
CAMERA_CONFIG_PATH = CONFIG_DIR / "camera_config.npy"
HAND_EYE_PATH = CONFIG_DIR / "ecT.npy"

POSITION_SCALE = 1e-6
ANGLE_SCALE = 1e-3
EULER_ORDER = "xyz"

MIN_DEPTH_M = 0.10
MAX_DEPTH_M = 1.50

CONFIDENCE_THRESHOLD = 0.30
TEXT_PROMPT = "a dark wooden block"
SAVE_VISUALIZATION = True

# mask quality
MIN_VALID_DEPTH_POINTS = 500
MAX_VALID_DEPTH_POINTS = 100000

# point cloud filter
VOXEL_SIZE_M = 0.001
STAT_NB_NEIGHBORS = 50
STAT_STD_RATIO = 1.0
RADIUS_NB_POINTS = 50
RADIUS_M = 0.015

# largest cluster
KEEP_LARGEST_CLUSTER = True
CLUSTER_EPS = 0.015
CLUSTER_MIN_POINTS = 30


def configure_paths(args):
    global DATA_DIR, FIRST_POINT_RESULT_PATH, COARSE_SCAN_DIR, OUTPUT_POINT_FILE
    global OUTPUT_SEQ_FILE, OUTPUT_DEBUG_FILE, VIS_DIR, CAMERA_CONFIG_PATH, HAND_EYE_PATH
    global TEXT_PROMPT

    if args.run_dir:
        run_dir = Path(args.run_dir)
        DATA_DIR = run_dir / "construction"
        FIRST_POINT_RESULT_PATH = DATA_DIR / "initial_frame_point.npz"
        COARSE_SCAN_DIR = DATA_DIR / "coarse_scan"
        OUTPUT_POINT_FILE = COARSE_SCAN_DIR / "coarse_point_result.npz"
        OUTPUT_SEQ_FILE = COARSE_SCAN_DIR / "coarse_png_sequence.json"
        OUTPUT_DEBUG_FILE = COARSE_SCAN_DIR / "coarse_prompt_debug.json"
        VIS_DIR = COARSE_SCAN_DIR / "vis_prompt_mask"
        CAMERA_CONFIG_PATH = run_dir / "camera_config.npy"
        HAND_EYE_PATH = PROJECT_ROOT / "config" / "calibration" / "right_camera" / "ecT.npy"

    if args.coarse_scan_dir:
        COARSE_SCAN_DIR = Path(args.coarse_scan_dir)
    if args.first_point_result:
        FIRST_POINT_RESULT_PATH = Path(args.first_point_result)
    if args.output_npz:
        OUTPUT_POINT_FILE = Path(args.output_npz)
    if args.camera_config:
        CAMERA_CONFIG_PATH = Path(args.camera_config)
    if args.hand_eye:
        HAND_EYE_PATH = Path(args.hand_eye)
    if args.text_prompt:
        TEXT_PROMPT = args.text_prompt

    OUTPUT_SEQ_FILE = COARSE_SCAN_DIR / "coarse_png_sequence.json"
    OUTPUT_DEBUG_FILE = COARSE_SCAN_DIR / "coarse_prompt_debug.json"
    VIS_DIR = COARSE_SCAN_DIR / "vis_prompt_mask"
    COARSE_SCAN_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_POINT_FILE.parent.mkdir(parents=True, exist_ok=True)


# =========================
# Basic utilities
# =========================
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


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


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


# =========================
# First-frame object center prior
# =========================
def load_first_object_points():
    meta = np.load(FIRST_POINT_RESULT_PATH, allow_pickle=True)
    object_center = meta['object_center']
    object_bbox = meta['object_bbox']

    return  object_center, object_bbox



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


def make_object_center_prompt(object_center_base, T_base_cam, color_intrinsic, image_shape):
    uv, _ = project_points_base_to_image(
        object_center_base,
        T_base_cam,
        color_intrinsic,
        image_shape,
    )
    if len(uv) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return uv[:1].astype(np.float32)


# =========================
# SAM3 segmentation
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
    for m in masks:
        m = np.squeeze(m).astype(bool)
        ys, xs = np.where(m)
        if len(xs) == 0:
            boxes.append([0, 0, 0, 0])
        else:
            boxes.append([xs.min(), ys.min(), xs.max(), ys.max()])
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


def choose_highest_score_mask(masks, boxes, scores):
    masks = _squeeze_masks(masks)
    if masks is None or len(masks) == 0:
        return None, None, None, None

    if boxes is None or len(boxes) < len(masks):
        boxes = _boxes_from_masks(masks)

    scores_arr = np.asarray(scores).reshape(-1) if scores is not None else np.zeros((len(masks),))
    if len(scores_arr) < len(masks):
        scores_arr = np.pad(scores_arr, (0, len(masks) - len(scores_arr)), constant_values=0)

    best_idx = int(np.argmax(scores_arr))
    mask = np.squeeze(masks[best_idx]).astype(bool)
    box = np.asarray(boxes[best_idx]).astype(float).tolist()
    score = float(scores_arr[best_idx])

    return mask, box, score, best_idx


def sam3_text_predict(model, processor, image_path, text_prompt=TEXT_PROMPT):
    image = Image.open(str(image_path)).convert("RGB")

    with torch.inference_mode():
        state = processor.set_image(image)
        output = processor.set_text_prompt(state=state, prompt=text_prompt)

    masks = _squeeze_masks(output["masks"])
    boxes = _to_numpy(output.get("boxes", None))
    scores = _to_numpy(output.get("scores", None))

    del state, output
    torch.cuda.empty_cache()
    gc.collect()

    return choose_highest_score_mask(masks, boxes, scores)


def sam3_point_predict(model, processor, image_path, point_prompts):
    if point_prompts is None or len(point_prompts) == 0:
        return None, None, None, None

    image = Image.open(str(image_path)).convert("RGB")
    point_arg = np.asarray(point_prompts, dtype=np.float32)
    if point_arg.ndim == 1:
        point_arg = point_arg[None, :]
    point_labels = np.ones((len(point_arg),), dtype=np.int32)

    with torch.inference_mode():
        state = processor.set_image(image)
        masks, scores, _ = model.predict_inst(
            state,
            point_coords=point_arg,
            point_labels=point_labels,
            multimask_output=True,
        )

    boxes = _boxes_from_masks(masks)

    del state
    torch.cuda.empty_cache()
    gc.collect()

    return choose_highest_score_mask(masks, boxes, scores)


def get_mask_text_then_point_fallback(model, processor, image_path, point_prompts):
    # 主分割：只用 text prompt，不使用 point 选择 mask
    mask, box, score, idx = sam3_text_predict(model, processor, image_path, TEXT_PROMPT)
    if mask is not None:
        return mask, box, score, idx, "text"

    # fallback：只有 text 没有返回 mask 时，才用 object center point prompt
    mask, box, score, idx = sam3_point_predict(model, processor, image_path, point_prompts)
    return mask, box, score, idx, "point_fallback"


# =========================
# Point cloud filtering
# =========================
def keep_largest_cluster(pcd, eps=CLUSTER_EPS, min_points=CLUSTER_MIN_POINTS):
    if len(pcd.points) < min_points:
        return pcd

    labels = np.asarray(
        pcd.cluster_dbscan(
            eps=eps,
            min_points=min_points,
            print_progress=False,
        )
    )

    if labels.size == 0 or labels.max() < 0:
        return pcd

    largest_label = max(
        range(labels.max() + 1),
        key=lambda i: np.sum(labels == i)
    )

    keep_idx = np.where(labels == largest_label)[0]
    return pcd.select_by_index(keep_idx)


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
    raw_count = len(u_valid)

    if raw_count < MIN_VALID_DEPTH_POINTS or raw_count > MAX_VALID_DEPTH_POINTS:
        return np.zeros((0, 4), dtype=float), raw_count, 0

    z = depth[v_valid, u_valid]
    x = (u_valid - cx) * z / fx
    y = (v_valid - cy) * z / fy
    points_cam = np.column_stack([x, y, z])

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

    if KEEP_LARGEST_CLUSTER:
        pcd = keep_largest_cluster(pcd)

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
        cv2.putText(vis, f"SAM box {selected_score:.3f}", (x0, min(vis.shape[0] - 5, y1 + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    for i, (x, y) in enumerate(point_prompts):
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        cv2.circle(vis, (xi, yi), 5, (0, 255, 255), -1)
        cv2.putText(vis, f"p{i}", (xi + 6, yi - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), vis)


# =========================
# Per-frame extraction
# =========================
def process_one_frame(
    name,
    object_center_base,
    color_intrinsic,
    depth_scale,
    T_ee_cam,
    sam3_model,
    sam3_processor,
):
    image_path = COARSE_SCAN_DIR / f"{name}.png"
    depth_path = COARSE_SCAN_DIR / f"{name}.npy"
    endpose_path = COARSE_SCAN_DIR / f"{name}.json"

    img = cv2.imread(str(image_path))
    T_base_ee = parse_endpose_json(endpose_path)
    T_base_cam = T_base_ee @ T_ee_cam

    point_prompts = make_object_center_prompt(
        object_center_base=object_center_base,
        T_base_cam=T_base_cam,
        color_intrinsic=color_intrinsic,
        image_shape=img.shape,
    )

    mask, selected_box, selected_score, selected_idx, seg_mode = get_mask_text_then_point_fallback(
        model=sam3_model,
        processor=sam3_processor,
        image_path=image_path,
        point_prompts=point_prompts,
    )

    if mask is None:
        return np.zeros((0, 4), dtype=float), T_base_cam, {
            "name": name,
            "success": False,
            "segmentation_mode": seg_mode,
            "reason": "SAM3 returned no mask",
            "point_prompts_xy": [[round(float(x), 2), round(float(y), 2)] for x, y in point_prompts],
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

    success = bool(filtered_count > 0)
    reason = "ok" if success else "valid depth point count out of range or empty after filtering"

    debug = {
        "name": name,
        "success": success,
        "segmentation_mode": seg_mode,
        "text_prompt": TEXT_PROMPT,
        "point_prompts_xy": [[round(float(x), 2), round(float(y), 2)] for x, y in point_prompts],
        "selected_mask_idx": selected_idx,
        "selected_score": round(float(selected_score), 4) if selected_score is not None else None,
        "selected_box_xyxy": [round(float(v), 2) for v in selected_box] if selected_box is not None else None,
        "raw_points_count": int(raw_count),
        "filtered_points_count": int(filtered_count),
        "min_valid_depth_points": MIN_VALID_DEPTH_POINTS,
        "max_valid_depth_points": MAX_VALID_DEPTH_POINTS,
        "cluster_eps": CLUSTER_EPS,
        "cluster_min_points": CLUSTER_MIN_POINTS,
        "reason": reason,
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

    object_center, object_size = load_first_object_points()
    object_center_base = object_center.reshape(1,3)

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
                object_center_base=object_center_base,
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
            print(f"  skipped: {debug.get('reason', '')}, raw={debug.get('raw_points_count', 0)}")
            continue

        points_collection.append(points_base_h)
        bcT_collection.append(T_base_cam)
        valid_png_names.append(name)

        print(
            f"  success={debug['success']}, "
            f"mode={debug['segmentation_mode']}, "
            f"raw={debug['raw_points_count']}, "
            f"filtered={debug['filtered_points_count']}, "
            f"prompts={debug['point_prompts_xy']}"
        )

    points_collection_obj = np.empty(len(points_collection), dtype=object)
    for i, pts in enumerate(points_collection):
        points_collection_obj[i] = np.asarray(pts, dtype=np.float64)

    np.savez(
        OUTPUT_POINT_FILE,
        points_collection=points_collection_obj,
        bcT_collection=np.asarray(bcT_collection, dtype=np.float64),
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--coarse-scan-dir", dest="coarse_scan_dir", type=Path, default=None)
    parser.add_argument("--first-point-result", dest="first_point_result", type=Path, default=None)
    parser.add_argument("--output-npz", type=Path, default=None)
    parser.add_argument("--camera-config", type=Path, default=None)
    parser.add_argument("--hand-eye", type=Path, default=None)
    parser.add_argument("--text-prompt", default=None)
    configure_paths(parser.parse_args())
    extract_coarse_points()

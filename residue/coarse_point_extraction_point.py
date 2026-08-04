import gc
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R

sys.path.append(r"/home/smmg/AAM")
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# =========================
# Paths / parameters
# =========================
DATA_DIR = Path(r"/home/smmg/AAM/construction/data")

FIRST_POINT_RESULT_PATH = DATA_DIR / "initial_frame_point.npz"
FALLBACK_FIRST_POINT_RESULT_PATH = DATA_DIR / "point_result.npz"

COARSE_SCAN_DIR = DATA_DIR / "coarse_scan"

OUTPUT_POINT_FILE = COARSE_SCAN_DIR / "coarse_point_result.npz"
OUTPUT_SEQ_FILE = COARSE_SCAN_DIR / "coarse_png_sequence.json"
OUTPUT_DEBUG_FILE = COARSE_SCAN_DIR / "coarse_prompt_debug.json"
VIS_DIR = COARSE_SCAN_DIR / "vis_prompt_mask_point"

CONFIG_DIR = Path(r"/home/smmg/AAM/config/calibration/right_camera")
CAMERA_CONFIG_PATH = CONFIG_DIR / "camera_config.npy"
HAND_EYE_PATH = CONFIG_DIR / "ecT.npy"       # T_ee_cam

POSITION_SCALE = 1e-6     # raw position -> meter
ANGLE_SCALE = 1e-3        # raw angle -> degree
EULER_ORDER = "xyz"

MIN_DEPTH_M = 0.10
MAX_DEPTH_M = 1.50

CONFIDENCE_THRESHOLD = 0.30
SAVE_VISUALIZATION = True

VOXEL_SIZE_M = 0.001
STAT_NB_NEIGHBORS = 50
STAT_STD_RATIO = 1.0
RADIUS_NB_POINTS = 50
RADIUS_M = 0.015


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
    raw = load_json(path)

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
    if len(valid) > 0 and float(np.nanmax(valid)) > 10.0:
        return depth / 1000.0

    return depth


# =========================
# First-frame object center
# =========================
def load_first_object_center():
    point_path = FIRST_POINT_RESULT_PATH
    if not point_path.exists():
        point_path = FALLBACK_FIRST_POINT_RESULT_PATH

    meta = np.load(point_path, allow_pickle=True)

    if "object_center" in meta:
        object_center = np.asarray(meta["object_center"], dtype=float).reshape(-1)[:3]
    else:
        points = np.asarray(meta["points_collection"][0], dtype=float)
        points = points.reshape(-1, points.shape[-1])[:, :3]
        points = points[np.all(np.isfinite(points), axis=1)]
        low = np.percentile(points, 2.0, axis=0)
        high = np.percentile(points, 98.0, axis=0)
        object_center = 0.5 * (low + high)

    return object_center.reshape(1, 3)


# =========================
# Projection
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


def make_point_prompt(object_center_base, T_base_cam, color_intrinsic, image_shape):
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
# SAM3 point prompt
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


def boxes_from_masks(masks):
    masks = _squeeze_masks(masks)
    boxes = []

    if masks is None or len(masks) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    for m in masks:
        ys, xs = np.where(np.squeeze(m).astype(bool))
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


def sam3_point_predict(model, processor, image_path, point_prompts):
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
            box=None,
            multimask_output=True,
        )

    masks = _squeeze_masks(masks)
    scores = _to_numpy(scores)
    boxes = boxes_from_masks(masks)

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
            best_idx = i
            best_metric = metric

    mask = np.squeeze(masks[best_idx]).astype(bool)
    box = np.asarray(boxes[best_idx], dtype=float).tolist()
    score = float(scores_arr[best_idx]) if best_idx < len(scores_arr) else 0.0

    return mask, box, score, int(best_idx)


def get_mask_by_point_prompt(model, processor, image_path, point_prompts):
    masks, boxes, scores = sam3_point_predict(
        model=model,
        processor=processor,
        image_path=image_path,
        point_prompts=point_prompts,
    )
    return choose_best_mask(masks, boxes, scores, point_prompts)


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
        pcd = pcd.voxel_down_sample(VOXEL_SIZE_M)

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
def save_prompt_visualization(image_path, mask, point_prompts, selected_box, selected_score, save_path):
    img = cv2.imread(str(image_path))
    if img is None:
        return

    vis = img.copy()

    if mask is not None:
        m = mask.astype(bool)
        if m.shape[:2] != vis.shape[:2]:
            m = cv2.resize(
                m.astype(np.uint8),
                (vis.shape[1], vis.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        red = np.zeros_like(vis)
        red[m] = (0, 0, 255)
        vis = cv2.addWeighted(vis, 1.0, red, 0.45, 0)

    if selected_box is not None:
        x0, y0, x1, y1 = [int(round(v)) for v in selected_box]
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(
            vis,
            f"SAM box {selected_score:.3f}",
            (x0, min(vis.shape[0] - 5, y1 + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    for i, (x, y) in enumerate(point_prompts):
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        cv2.circle(vis, (xi, yi), 6, (0, 255, 255), -1)
        cv2.putText(
            vis,
            f"p{i}",
            (xi + 8, yi - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), vis)


# =========================
# Per-frame / main
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

    point_prompts = make_point_prompt(
        object_center_base=object_center_base,
        T_base_cam=T_base_cam,
        color_intrinsic=color_intrinsic,
        image_shape=img.shape,
    )

    if len(point_prompts) == 0:
        return np.zeros((0, 4), dtype=float), T_base_cam, {
            "name": name,
            "success": False,
            "reason": "object center projected outside image",
        }

    mask, selected_box, selected_score, selected_idx = get_mask_by_point_prompt(
        model=sam3_model,
        processor=sam3_processor,
        image_path=image_path,
        point_prompts=point_prompts,
    )

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
            mask=mask,
            point_prompts=point_prompts,
            selected_box=selected_box,
            selected_score=selected_score if selected_score is not None else 0.0,
            save_path=VIS_DIR / f"{name}_point_mask.png",
        )

    debug = {
        "name": name,
        "success": bool(filtered_count > 0),
        "prompt_type": "point_only_object_center",
        "point_prompts_xy": [[round(float(x), 2), round(float(y), 2)] for x, y in point_prompts],
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
    object_center_base = load_first_object_center()

    sam3_model, sam3_processor = init_sam3_model()

    png_names = [p.stem for p in sorted(COARSE_SCAN_DIR.glob("*.png"))]

    points_collection = []
    bcT_collection = []
    valid_png_names = []
    debug_records = []

    print("========== Coarse point extraction: point-only ==========")
    print(f"object_center_base_m: {object_center_base.reshape(3)}")
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
            print("  No valid object points extracted.")
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
    all_points = (
        np.concatenate([np.asarray(pts, dtype=np.float64) for pts in points_collection], axis=0)
        if len(points_collection)
        else np.zeros((0, 4), dtype=np.float64)
    )

    np.savez(
        OUTPUT_POINT_FILE,
        points_collection=points_collection_obj,
        bcT_collection=bcT_collection_arr,
        all_points=all_points,
        offsets=offsets,
        valid_png_names=np.asarray(valid_png_names, dtype=object),
        object_center_base_m=object_center_base.reshape(3),
    )

    save_json(OUTPUT_SEQ_FILE, valid_png_names)
    save_json(OUTPUT_DEBUG_FILE, {
        "prompt_type": "point_only_object_center",
        "object_center_base_m": [round(float(v), 6) for v in object_center_base.reshape(3)],
        "valid_png_names": valid_png_names,
        "records": debug_records,
    })

    print("\n========== Result ==========")
    print(f"Saved point cloud npz: {OUTPUT_POINT_FILE}")
    print(f"Saved png sequence:    {OUTPUT_SEQ_FILE}")
    print(f"Saved debug json:      {OUTPUT_DEBUG_FILE}")
    print(f"Valid frames: {len(valid_png_names)} / {len(png_names)}")


if __name__ == "__main__":
    extract_coarse_points()

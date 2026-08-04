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
STEM = "front"

IMAGE_PATH = DATA_DIR / f"{STEM}.png"
DEPTH_PATH = DATA_DIR / f"{STEM}.npy"
ENDPOSE_PATH = DATA_DIR / f"{STEM}.json"

# output dir
OUTPUT_POINT_FILE = DATA_DIR / "initial_frame_point.npz"   # keep user's requested name

CONFIG_DIR = PROJECT_ROOT / "config" / "calibration" / "right_camera"
CAMERA_CONFIG_PATH = CONFIG_DIR / "camera_config.npy"
HAND_EYE_PATH = CONFIG_DIR / "ecT.npy"       # T_ee_cam

TEXT_PROMPT = "a black rectagular-shaped block in the middle of image"
CONFIDENCE_THRESHOLD = 0.30

POSITION_SCALE = 1e-6     # raw position -> meter
ANGLE_SCALE = 1e-3        # raw angle -> degree
EULER_ORDER = "xyz"

MIN_DEPTH_M = 0.10
MAX_DEPTH_M = 1.50

VOXEL_SIZE_M = 0.001
STAT_NB_NEIGHBORS = 50
STAT_STD_RATIO = 1.0
RADIUS_NB_POINTS = 50
RADIUS_M = 0.015

SAVE_VISUALIZATION = True
VIS_PATH = DATA_DIR / "intial_frame_mask_vis.png"


def configure_paths(args):
    global DATA_DIR, STEM, IMAGE_PATH, DEPTH_PATH, ENDPOSE_PATH, OUTPUT_POINT_FILE
    global CAMERA_CONFIG_PATH, HAND_EYE_PATH, VIS_PATH, TEXT_PROMPT

    if args.run_dir:
        run_dir = Path(args.run_dir)
        DATA_DIR = run_dir / "construction"
        rough_dir = run_dir / "perception" / "rough_screening"
        IMAGE_PATH = rough_dir / "front.png"
        DEPTH_PATH = rough_dir / "front.npy"
        ENDPOSE_PATH = rough_dir / "front.json"
        OUTPUT_POINT_FILE = DATA_DIR / "initial_frame_point.npz"
        CAMERA_CONFIG_PATH = run_dir / "camera_config.npy"
        HAND_EYE_PATH = PROJECT_ROOT / "config" / "calibration" / "right_camera" / "ecT.npy"
        VIS_PATH = DATA_DIR / "initial_frame_mask_vis.png"

    if args.image_path:
        IMAGE_PATH = Path(args.image_path)
    if args.depth_path:
        DEPTH_PATH = Path(args.depth_path)
    if args.endpose_path:
        ENDPOSE_PATH = Path(args.endpose_path)
    if args.output_npz:
        OUTPUT_POINT_FILE = Path(args.output_npz)
    if args.camera_config:
        CAMERA_CONFIG_PATH = Path(args.camera_config)
    if args.hand_eye:
        HAND_EYE_PATH = Path(args.hand_eye)
    if args.text_prompt:
        TEXT_PROMPT = args.text_prompt

    DATA_DIR.mkdir(parents=True, exist_ok=True)


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
    arr = np.asarray(arr, dtype=float)
    return arr


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

def estimate_target(points_base, percentile=2.0):
    """Robust object center and bbox using percentiles."""
    low = np.percentile(points_base, percentile, axis=0)
    high = np.percentile(points_base, 100.0 - percentile, axis=0)
    center = 0.5 * (low + high)
    size = high - low
    return center, size



# =========================
# SAM3 point segmentation
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


def init_sam3_model():
    torch.cuda.empty_cache()
    gc.collect()
    model = build_sam3_image_model(enable_inst_interactivity=True)
    model.eval()
    processor = Sam3Processor(model, confidence_threshold=CONFIDENCE_THRESHOLD)
    return model, processor


def mask_to_box(mask):
    ys, xs = np.where(mask.astype(bool))
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def segment_by_point(model, processor, image_path):
    image = Image.open(str(image_path)).convert("RGB")

    with torch.inference_mode():
        state = processor.set_image(image)
        masks, scores, _ = model.predict_inst(
            state,
            point_coords=np.asarray([[320, 240]], dtype=np.float32),
            point_labels=np.asarray([1], dtype=np.int32),
            multimask_output=False,
        )

    masks = _squeeze_masks(masks)
    scores_arr = np.asarray(_to_numpy(scores)).reshape(-1)
    best_idx = int(np.argmax(scores_arr))
    best_mask = np.squeeze(masks[best_idx]).astype(bool)
    best_box = mask_to_box(best_mask)
    best_score = float(scores_arr[best_idx])

    del state
    torch.cuda.empty_cache()
    gc.collect()

    return best_mask, best_box, best_score


# =========================
# Point cloud generation
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

    points_base = transform_points(T_base_cam, points_cam)
    points_base_h = np.column_stack([points_base, np.ones(len(points_base))])
    return points_base_h, len(points_base_h)


def save_mask_visualization(image_path, mask, box, score, save_path):
    img = cv2.imread(str(image_path))
    if img is None:
        return

    vis = img.copy()
    m = mask.astype(bool)
    if m.shape[:2] != vis.shape[:2]:
        m = cv2.resize(m.astype(np.uint8), (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)

    red = np.zeros_like(vis)
    red[m] = (0, 0, 255)
    vis = cv2.addWeighted(vis, 1.0, red, 0.45, 0)

    if box is not None:
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(vis, f"{score:.3f}", (x0, max(0, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imwrite(str(save_path), vis)


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--image-path", type=Path, default=None)
    parser.add_argument("--depth-path", type=Path, default=None)
    parser.add_argument("--endpose-path", type=Path, default=None)
    parser.add_argument("--output-npz", type=Path, default=None)
    parser.add_argument("--camera-config", type=Path, default=None)
    parser.add_argument("--hand-eye", type=Path, default=None)
    parser.add_argument("--text-prompt", default=None)
    args = parser.parse_args()
    configure_paths(args)

    color_intrinsic, depth_scale = load_camera_config(CAMERA_CONFIG_PATH)
    T_ee_cam = load_matrix(HAND_EYE_PATH)
    T_base_ee = parse_endpose_json(ENDPOSE_PATH)
    T_base_cam = T_base_ee @ T_ee_cam

    model, processor = init_sam3_model()
    mask, box, score = segment_by_point(model, processor, IMAGE_PATH)

    points_base_h, points__num = mask_to_points_base(
        image_path=IMAGE_PATH,
        depth_path=DEPTH_PATH,
        mask=mask,
        T_base_cam=T_base_cam,
        color_intrinsic=color_intrinsic,
        depth_scale=depth_scale,
    )

    center, size = estimate_target(points_base_h)

    # =========================
    # visualize object center projection on test.png
    # =========================
    fx = float(color_intrinsic["fx"])
    fy = float(color_intrinsic["fy"])
    cx = float(color_intrinsic["ppx"])
    cy = float(color_intrinsic["ppy"])

    T_cam_base = np.linalg.inv(T_base_cam)
    center_cam = T_cam_base @ np.array([center[0], center[1], center[2], 1.0])

    X, Y, Z = center_cam[:3]
    u = int(round(fx * X / Z + cx))
    v = int(round(fy * Y / Z + cy))

    img = cv2.imread(str(IMAGE_PATH))
    cv2.circle(img, (u, v), 8, (0, 0, 255), -1)
    cv2.circle(img, (u, v), 13, (255, 255, 255), 2)
    cv2.putText(img, "object center", (u + 10, v - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    CENTER_VIS_PATH = DATA_DIR / "object_center_projection_vis.png"
    cv2.imwrite(str(CENTER_VIS_PATH), img)
    print(f"saved object center projection vis: {CENTER_VIS_PATH}")

    np.savez(
        OUTPUT_POINT_FILE,
        points_collection=points_base_h[None, :, :],
        bcT_collection=T_base_cam[None, :, :],
        object_center = np.asarray(center)[:3],
        object_bbox = np.asarray(size)[:3]

    )

    if SAVE_VISUALIZATION:
        save_mask_visualization(IMAGE_PATH, mask, box, score, VIS_PATH)

    print("========== Initial frame point result ==========")
    print('object center:', center)
    print(f"SAM score: {score:.4f}")
    print(f"points number: {points__num}")
    print(f"saved npz: {OUTPUT_POINT_FILE}")
    if SAVE_VISUALIZATION:
        print(f"saved vis: {VIS_PATH}")


if __name__ == "__main__":
    main()

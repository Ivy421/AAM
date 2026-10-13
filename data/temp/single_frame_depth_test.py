import gc
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch
from PIL import Image


# ============================================================
# Hard parameters
# ============================================================
PROJECT_ROOT = Path( r"/home/smmg/AAM")
DATA_ROOT = Path(r"/home/smmg/AAM/data")

# Default paths
SAM3_PATH = PROJECT_ROOT
CAMERA_CONFIG_PATH = PROJECT_ROOT / "config" / "calibration" / "right_camera" / "camera_config.npy"

# Single-frame RGB-D data
IMAGE_PATH = DATA_ROOT / "deformtest2.png"
DEPTH_PATH = DATA_ROOT / "deformtest2.npy"

# SAM3 text prompt
TEXT_PROMPT = "a long black board placed on the green table"
CONFIDENCE_THRESHOLD = 0.30

# Only reject invalid / clearly out-of-range depth.
# No voxel downsampling, statistical filtering, radius filtering,
# clustering, ICP, or base-frame transformation is used.
MIN_DEPTH_M = 0.10
MAX_DEPTH_M = 1.50

MASK_VIS_PATH = DATA_ROOT / "sam3_mask_vis.png"
PCD_PATH = DATA_ROOT / "object_camera_frame.pcd"


# ============================================================
# SAM3 import
# ============================================================
if str(SAM3_PATH) not in sys.path:
    sys.path.insert(0, str(SAM3_PATH))

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# ============================================================
# Basic utilities
# ============================================================
def load_camera_config(path):
    config = np.load(path, allow_pickle=True).item()
    color_intrinsic = config["color_intrinsic"]
    depth_scale = float(config.get("depth_scale", 0.001))
    return color_intrinsic, depth_scale


def depth_to_meter(depth_raw, depth_scale):
    depth_raw = np.nan_to_num(depth_raw, nan=0.0, posinf=0.0, neginf=0.0)

    if depth_raw.dtype == np.uint16 or depth_raw.dtype == np.uint32:
        return depth_raw.astype(np.float32) * depth_scale

    depth = depth_raw.astype(np.float32)
    valid = depth[depth > 0]

    if len(valid) == 0:
        return depth

    if float(np.nanmax(valid)) > 10.0:
        return depth / 1000.0

    return depth


def to_numpy(x):
    if x is None:
        return None

    if torch.is_tensor(x):
        x = x.detach()

        if x.dtype == torch.bfloat16 or x.dtype == torch.float16:
            x = x.float()

        return x.cpu().numpy()

    return np.asarray(x)


def squeeze_masks(masks):
    masks = to_numpy(masks)

    if masks is None:
        return None

    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]

    return masks


# ============================================================
# SAM3 text segmentation
# ============================================================
def init_sam3():
    torch.cuda.empty_cache()
    gc.collect()

    model = build_sam3_image_model()
    model.eval()

    processor = Sam3Processor(
        model,
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )

    return model, processor


def segment_object_by_text(model, processor, image_path, text_prompt):
    image = Image.open(str(image_path)).convert("RGB")

    with torch.inference_mode():
        state = processor.set_image(image)
        output = processor.set_text_prompt(
            state=state,
            prompt=text_prompt,
        )

    masks = squeeze_masks(output["masks"])
    boxes = to_numpy(output.get("boxes", None))
    scores = to_numpy(output.get("scores", None))

    if masks is None or len(masks) == 0:
        raise RuntimeError("SAM3 returned no mask.")

    scores = np.asarray(scores).reshape(-1) if scores is not None else np.zeros(len(masks))
    best_idx = int(np.argmax(scores))

    mask = np.squeeze(masks[best_idx]).astype(bool)
    box = boxes[best_idx].tolist() if boxes is not None and len(boxes) > best_idx else None
    score = float(scores[best_idx])

    del state, output
    torch.cuda.empty_cache()
    gc.collect()

    return mask, box, score


# ============================================================
# Masked single-frame depth -> CAMERA-frame point cloud
# ============================================================
def masked_depth_to_camera_pcd(image_path, depth_path, mask, color_intrinsic, depth_scale):
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read RGB image: {image_path}")

    depth_raw = np.load(depth_path)
    depth_m = depth_to_meter(depth_raw, depth_scale)

    if depth_m.ndim != 2:
        raise ValueError(f"Depth must be HxW, got {depth_m.shape}")

    H, W = depth_m.shape

    if mask.shape != (H, W):
        mask = cv2.resize(
            mask.astype(np.uint8),
            (W, H),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    if image_bgr.shape[:2] != (H, W):
        image_bgr = cv2.resize(image_bgr, (W, H), interpolation=cv2.INTER_LINEAR)

    valid_depth = (
        np.isfinite(depth_m)
        & (depth_m > MIN_DEPTH_M)
        & (depth_m < MAX_DEPTH_M)
    )

    valid = mask & valid_depth
    v, u = np.nonzero(valid)

    if len(u) == 0:
        raise RuntimeError("No valid depth points inside the SAM3 mask.")

    fx = float(color_intrinsic["fx"])
    fy = float(color_intrinsic["fy"])
    cx = float(color_intrinsic["ppx"])
    cy = float(color_intrinsic["ppy"])

    z = depth_m[v, u]
    x = (u.astype(np.float32) - cx) * z / fx
    y = (v.astype(np.float32) - cy) * z / fy

    # RealSense camera frame:
    # x -> right, y -> down, z -> forward
    points_camera = np.column_stack([x, y, z])

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    colors = image_rgb[v, u].astype(np.float64) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_camera.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd, valid


# ============================================================
# Visualization
# ============================================================
def save_mask_visualization(image_path, mask, box, score, save_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(f"Cannot read RGB image: {image_path}")

    if mask.shape != image.shape[:2]:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    overlay = image.copy()
    overlay[mask] = (0, 0, 255)
    vis = cv2.addWeighted(image, 0.65, overlay, 0.35, 0)

    if box is not None:
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(
            vis,
            f"SAM3 score={score:.3f}",
            (x0, max(20, y0 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )

    cv2.imwrite(str(save_path), vis)


def visualize_camera_pcd(pcd):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.05,
        origin=[0.0, 0.0, 0.0],
    )

    o3d.visualization.draw_geometries(
        [pcd, frame],
        window_name="Single-frame SAM3 object point cloud - camera frame",
        width=1280,
        height=900,
    )


# ============================================================
# Main
# ============================================================
def main():
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    print("========== Single-frame depth test ==========")
    print(f"Image:         {IMAGE_PATH}")
    print(f"Depth:         {DEPTH_PATH}")
    print(f"Camera config: {CAMERA_CONFIG_PATH}")
    print(f"Text prompt:   {TEXT_PROMPT}")

    color_intrinsic, depth_scale = load_camera_config(CAMERA_CONFIG_PATH)

    model, processor = init_sam3()

    mask, box, score = segment_object_by_text(
        model=model,
        processor=processor,
        image_path=IMAGE_PATH,
        text_prompt=TEXT_PROMPT,
    )

    print(f"SAM3 score:    {score:.4f}")
    print(f"Mask pixels:   {int(np.count_nonzero(mask))}")

    pcd, valid = masked_depth_to_camera_pcd(
        image_path=IMAGE_PATH,
        depth_path=DEPTH_PATH,
        mask=mask,
        color_intrinsic=color_intrinsic,
        depth_scale=depth_scale,
    )

    print(f"Valid points:  {len(pcd.points)}")
    print("Frame:         camera frame only")
    print("Filtering:     none")

    #save_mask_visualization(
    #    image_path=IMAGE_PATH,
    #    mask=mask,
    #    box=box,
    #    score=score,
    #    save_path=MASK_VIS_PATH,
    #)

    o3d.io.write_point_cloud(str(PCD_PATH), pcd)
    visualize_camera_pcd(pcd)


if __name__ == "__main__":
    main()

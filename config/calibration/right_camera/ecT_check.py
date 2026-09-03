"""Quick ecT validation by cross-frame reprojection.

Data:
    /home/smmg/AAM/config/calibration/right_camera/1.png
    /home/smmg/AAM/config/calibration/right_camera/1.npy
    /home/smmg/AAM/config/calibration/right_camera/1.json
    /home/smmg/AAM/config/calibration/right_camera/2.png
    /home/smmg/AAM/config/calibration/right_camera/2.npy
    /home/smmg/AAM/config/calibration/right_camera/2.json

Workflow:
    1. Use SAM3 text prompt to segment the black wooden rectangular board in 1.png.
    2. Back-project mask depth from 1.npy to camera-1 frame.
    3. Transform camera-1 points to robot base frame with base_T_end_1 @ ecT.
    4. Save the base-frame point cloud as 1.pcd.
    5. Reproject 1.pcd to 2.png using base_T_end_2 @ ecT.
    6. Save/show the reprojection visualization.
"""

import gc
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


DATA_DIR = Path("/home/smmg/AAM/config/calibration/right_camera")

IMG1_PATH = DATA_DIR / "2.png"
DEPTH1_PATH = DATA_DIR / "2.npy"
POSE1_PATH = DATA_DIR / "2.json"

IMG2_PATH = DATA_DIR / "1.png"
DEPTH2_PATH = DATA_DIR / "1.npy"  # kept for data completeness, not used
POSE2_PATH = DATA_DIR / "1.json"

CAMERA_CONFIG_PATH = DATA_DIR / "camera_config.npy"
ECT_PATH = DATA_DIR / "ecT_20260813.npy"

OUTPUT_PCD_PATH = DATA_DIR / "2.pcd"
OUTPUT_VIS_PATH = DATA_DIR / "2_pcd_reproject_to_1.png"
OUTPUT_MASK_PATH = DATA_DIR / "2_sam3_mask.png"

TEXT_PROMPT = "a cardboard box"
POINT_ALPHA = 0.35
POINT_RADIUS = 2
DRAW_STEP = 1


def load_camera_intrinsic(config_path: Path):
    config = np.load(config_path, allow_pickle=True)
    if isinstance(config, np.ndarray) and config.shape == ():
        config = config.item()

    intr = config["color_intrinsic"] if isinstance(config, dict) and "color_intrinsic" in config else config
    return {
        "fx": float(intr["fx"]),
        "fy": float(intr["fy"]),
        "ppx": float(intr.get("ppx", intr.get("cx"))),
        "ppy": float(intr.get("ppy", intr.get("cy"))),
        "depth_scale": float(config.get("depth_scale", 0.001)) if isinstance(config, dict) else 0.001,
    }


def piper_endpose_json_to_T(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        pose = json.load(f)

    if isinstance(pose, list) and len(pose) == 6 and all(isinstance(v, dict) for v in pose):
        xyz_raw = np.array([pose[0]["x"], pose[1]["y"], pose[2]["z"]], dtype=float)
        rpy_raw = np.array([pose[3]["rx"], pose[4]["ry"], pose[5]["rz"]], dtype=float)
    elif isinstance(pose, dict):
        xyz_raw = np.array([pose["x"], pose["y"], pose["z"]], dtype=float)
        rpy_raw = np.array([pose["rx"], pose["ry"], pose["rz"]], dtype=float)
    else:
        arr = np.asarray(pose, dtype=float).reshape(6)
        xyz_raw = arr[:3]
        rpy_raw = arr[3:]

    xyz_m = xyz_raw / 1_000_000.0 if np.max(np.abs(xyz_raw)) > 10000.0 else xyz_raw / 1000.0
    rpy_deg = rpy_raw / 1000.0 if np.max(np.abs(rpy_raw)) > 360.0 else rpy_raw

    T = np.eye(4)
    T[:3, :3] = R.from_euler("xyz", rpy_deg, degrees=True).as_matrix()
    T[:3, 3] = xyz_m
    return T


def transform_points(T: np.ndarray, points: np.ndarray):
    points_h = np.column_stack([points, np.ones(len(points))])
    return (T @ points_h.T).T[:, :3]


def sam3_text_mask(image_path: Path, text_prompt: str):
    torch.cuda.empty_cache()
    gc.collect()

    model = build_sam3_image_model()
    processor = Sam3Processor(model)

    image = Image.open(image_path).convert("RGB")
    state = processor.set_image(image)
    output = processor.set_text_prompt(state=state, prompt=text_prompt)

    masks = output["masks"].detach().cpu().numpy()
    scores = output["scores"].detach().cpu().numpy()

    masks = np.squeeze(masks)
    if masks.ndim == 2:
        mask = masks > 0
    else:
        mask = masks[int(np.argmax(scores))] > 0

    cv2.imwrite(str(OUTPUT_MASK_PATH), mask.astype(np.uint8) * 255)
    return mask

def depth_mask_to_camera_points(mask: np.ndarray, depth: np.ndarray, intr: dict):
    depth_m = depth.astype(float) * intr["depth_scale"]
    ys, xs = np.where(mask & (depth_m > 0))
    z = depth_m[ys, xs]
    x = (xs.astype(float) - intr["ppx"]) / intr["fx"] * z
    y = (ys.astype(float) - intr["ppy"]) / intr["fy"] * z
    return np.column_stack([x, y, z])


def ndarray_to_pcd(points: np.ndarray, save_path: Path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    o3d.io.write_point_cloud(str(save_path), pcd)


def project_base_points_to_image(points_base: np.ndarray, base_T_camera: np.ndarray, intr: dict):
    camera_T_base = np.linalg.inv(base_T_camera)
    points_cam = transform_points(camera_T_base, points_base)

    x = points_cam[:, 0]
    y = points_cam[:, 1]
    z = points_cam[:, 2]

    u = intr["fx"] * x / z + intr["ppx"]
    v = intr["fy"] * y / z + intr["ppy"]
    return np.column_stack([u, v]), z


def draw_projection(color_bgr: np.ndarray, uv: np.ndarray):
    h, w = color_bgr.shape[:2]
    vis = color_bgr.copy()
    overlay = vis.copy()

    for u, v in uv[::DRAW_STEP].astype(int):
        if 0 <= u < w and 0 <= v < h:
            cv2.circle(overlay, (u, v), POINT_RADIUS, (255, 255, 0), -1, cv2.LINE_AA)

    return cv2.addWeighted(overlay, POINT_ALPHA, vis, 1.0 - POINT_ALPHA, 0)


def main():
    intr = load_camera_intrinsic(CAMERA_CONFIG_PATH)
    ecT = np.load(ECT_PATH)
    ecT[:3,3] += np.array([  -0.002  , -0.003  , 0.001  ])

    mask1 = sam3_text_mask(IMG1_PATH, TEXT_PROMPT)
    depth1 = np.load(DEPTH1_PATH)
    points_cam1 = depth_mask_to_camera_points(mask1, depth1, intr)
    base_T_end1 = piper_endpose_json_to_T(POSE1_PATH)
    base_T_camera1 = base_T_end1 @ ecT
    points_base1 = transform_points(base_T_camera1, points_cam1)
    ndarray_to_pcd(points_base1, OUTPUT_PCD_PATH)

    pcd = o3d.io.read_point_cloud(str(OUTPUT_PCD_PATH))
    points_base = np.asarray(pcd.points)

    points_base = o3d.io.read_point_cloud('/home/smmg/AAM/config/calibration/right_camera/2.pcd')
    points_base = np.asarray(points_base.points)

    base_T_end2 = piper_endpose_json_to_T(POSE2_PATH)
    base_T_camera2 = base_T_end2 @ ecT
    uv2, z2 = project_base_points_to_image(points_base, base_T_camera2, intr)

    color2 = cv2.imread(str(IMG2_PATH), cv2.IMREAD_COLOR)
    vis = draw_projection(color2, uv2)

    cv2.imwrite(str(OUTPUT_VIS_PATH), vis)
    cv2.imshow("1.pcd reprojected to 2.png", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("saved pcd:", OUTPUT_PCD_PATH)
    print("saved mask:", OUTPUT_MASK_PATH)
    print("saved visualization:", OUTPUT_VIS_PATH)
    print("point count:", len(points_base))


if __name__ == "__main__":
    main()

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

# Project import path
sys.path.append(r"E:/HKUSTGZ/AAM")
from Piper.endpose_reachability import reachability_test


# =========================
# Fixed paths / parameters
# =========================
DATA_DIR = Path(r"E:/HKUSTGZ/AAM/construction/data")
STEM = "test"

IMAGE_PATH = DATA_DIR / f"{STEM}.png"   # If your file name is test.npg by typo, rename it to test.png.
DEPTH_PATH = DATA_DIR / f"{STEM}.npy"
ENDPOSE_PATH = DATA_DIR / f"{STEM}.json"
OUTPUT_PATH = DATA_DIR / "coarse_scanpose.json"

CAMERA_CONFIG_PATH = Path(r"E:/HKUSTGZ/AAM/config/calibration/right_camera/camera_config.npy")
HAND_EYE_PATH = Path(r"E:/HKUSTGZ/AAM/config/calibration/right_camera/ecT.npy")

# Optional first-frame mask. If this file exists, it will be used directly.
MASK_PATH = DATA_DIR / f"{STEM}_mask.npy"

# Data unit conversion
DEPTH_SCALE = 1e-3       # raw depth -> meter, usually uint16 mm -> m
POSITION_SCALE = 1e-6    # json position raw unit -> meter
ANGLE_SCALE = 1e-3       # json angle raw unit -> degree
ANGLE_UNIT = "deg"
EULER_ORDER = "xyz"

# Coarse scan cube settings
RADIUS_SCALE = 3
RADIUS_MIN = 0.20
RADIUS_MAX = 1
CUBE_SIZE_M = np.array([0.15, 0.15, 0.15])  # x, y, z cube size in base axes
GRID_NUM_PER_AXIS = 5                       # 3x3x3 points per cube
ROLL_LIST_DEG = [0.0, -30.0, 30.0]

# Local axes are the same as robot base axes: x forward, y left, z up.
BASE_X = np.array([1.0, 0.0, 0.0])
BASE_Y = np.array([0.0, 1.0, 0.0])
BASE_Z = np.array([0.0, 0.0, 1.0])


# =========================
# Basic utilities
# =========================
def normalize(v, eps=1e-9):
    v = np.asarray(v, dtype=float).reshape(3)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError(f"Cannot normalize near-zero vector: {v}")
    return v / n


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
    T[:3, :3] = R.from_euler(EULER_ORDER, [rx, ry, rz], degrees=(ANGLE_UNIT == "deg")).as_matrix()
    T[:3, 3] = [x, y, z]
    return T


def load_camera_config(path):
    config = np.load(path, allow_pickle=True).item()
    color_intrinsic = config["color_intrinsic"]
    return color_intrinsic


def load_matrix(path):
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.shape == () and arr.dtype == object:
        arr = arr.item()
    arr = np.asarray(arr, dtype=float)
    if arr.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix from {path}, got {arr.shape}")
    return arr


def depth_to_points(depth, fx, fy, cx, cy, mask=None):
    z = depth.astype(float) * DEPTH_SCALE
    valid = np.isfinite(z) & (z > 0)

    # Basic depth range filter, adjust if needed.
    valid &= (z > 0.10) & (z < 1.20)

    if mask is not None:
        valid &= mask.astype(bool)

    v, u = np.nonzero(valid)
    if len(u) == 0:
        raise ValueError("No valid depth points found.")

    zz = z[v, u]
    xx = (u - cx) * zz / fx
    yy = (v - cy) * zz / fy
    return np.column_stack([xx, yy, zz])


def transform_points(T, points):
    points_h = np.column_stack([points, np.ones(len(points))])
    return (T @ points_h.T).T[:, :3]


def estimate_target(points_base, percentile=2.0):
    low = np.percentile(points_base, percentile, axis=0)
    high = np.percentile(points_base, 100.0 - percentile, axis=0)
    center = 0.5 * (low + high)
    size = high - low
    return center, size


def matrix_to_endpose_mm_deg(T_base_ee):
    xyz_mm = T_base_ee[:3, 3] * 1000.0
    rpy_deg = R.from_matrix(T_base_ee[:3, :3]).as_euler(EULER_ORDER, degrees=True)
    return [round(float(v), 2) for v in np.concatenate([xyz_mm, rpy_deg])]


# =========================
# Mask acquisition
# =========================
def find_image_path():
    if IMAGE_PATH.exists():
        return IMAGE_PATH
    for ext in [".jpg", ".jpeg", ".bmp", ".npg"]:
        p = DATA_DIR / f"{STEM}{ext}"
        if p.exists():
            return p
    return IMAGE_PATH


def get_first_frame_mask(image_path, depth_shape):
    """
    Priority:
    1. Use test_mask.npy if it exists.
    2. Try SAM3/positioning if project function is available.
    3. Fallback to None, meaning whole valid depth image is used.
    """
    if MASK_PATH.exists():
        mask = np.load(MASK_PATH, allow_pickle=True)
        return np.squeeze(mask).astype(bool)

    try:
        from AI_models.sam3 import positioning
        mask, box, score = positioning(str(image_path), "a black cube")
        best = int(np.argmax(score)) if len(score) > 0 else 0
        mask = np.squeeze(np.asarray(mask[best])).astype(bool)
        print("Using SAM3/positioning mask from first frame.")
        return mask
    except Exception as e:
        print("WARNING: failed to get SAM3 mask. Use whole valid depth image instead.")
        print("Reason:", e)
        return None


# =========================
# Coarse cube pose generation
# =========================
def rotate_about_z(v, yaw_deg):
    return R.from_euler("z", yaw_deg, degrees=True).as_matrix() @ v


def get_front_direction(object_center, p_cam0):
    """
    Front direction is defined by the first-frame camera position:
        object_center -> first camera position.
    Its horizontal projection is used for stable left/right/up cube definition.
    """
    v = normalize(p_cam0 - object_center)
    v_xy = np.array([v[0], v[1], 0.0])
    if np.linalg.norm(v_xy) < 1e-6:
        v_xy = -BASE_X  # fallback: camera is on robot side of object
    return normalize(v_xy)


def build_cube_centers(object_center, p_cam0, radius):
    front = get_front_direction(object_center, p_cam0)

    # 45-degree upward viewing direction means camera position has both front offset and positive z offset.
    front_up_45 = normalize(np.cos(np.deg2rad(45.0)) * front + np.sin(np.deg2rad(45.0)) * BASE_Z)
    left_up_45 = normalize(np.cos(np.deg2rad(45.0)) * rotate_about_z(front, 45.0) + np.sin(np.deg2rad(45.0)) * BASE_Z)
    right_up_45 = normalize(np.cos(np.deg2rad(45.0)) * rotate_about_z(front, -45.0) + np.sin(np.deg2rad(45.0)) * BASE_Z)
    top = BASE_Z

    cube_defs = [
        ("front", front),
        ("front_up_45", front_up_45),
        ("top_down", top),
        ("left_up_45", left_up_45),
        ("right_up_45", right_up_45),
    ]

    return [(name, object_center + radius * direction, direction) for name, direction in cube_defs]


def look_at_camera_pose_with_roll(p_cam, target, roll_deg):
    """
    Build T_base_cam. Assumption: camera +Z axis is optical/depth direction.
    """
    z_cam = normalize(target - p_cam)
    up_hint = BASE_Z.copy()

    # Avoid singularity when optical axis is almost parallel to up_hint.
    if abs(float(np.dot(up_hint, z_cam))) > 0.95:
        up_hint = BASE_Y.copy()

    x_cam = normalize(np.cross(up_hint, z_cam))
    y_cam = normalize(np.cross(z_cam, x_cam))

    R_base_cam = np.column_stack([x_cam, y_cam, z_cam])
    R_roll = R.from_euler("z", roll_deg, degrees=True).as_matrix()
    R_base_cam = R_base_cam @ R_roll

    T = np.eye(4)
    T[:3, :3] = R_base_cam
    T[:3, 3] = p_cam
    return T


def sample_points_in_cube(cube_center):
    half = CUBE_SIZE_M / 2.0
    xs = np.linspace(-half[0], half[0], GRID_NUM_PER_AXIS)
    ys = np.linspace(-half[1], half[1], GRID_NUM_PER_AXIS)
    zs = np.linspace(-half[2], half[2], GRID_NUM_PER_AXIS)

    pts = []
    for dx in xs:
        for dy in ys:
            for dz in zs:
                pts.append(cube_center + np.array([dx, dy, dz]))

    # Try cube center first, then points closer to center first.
    pts = sorted(pts, key=lambda p: np.linalg.norm(p - cube_center))
    return pts


def find_best_pose_in_cube(cube_name, cube_center, object_center, T_ee_cam):
    inv_T_ee_cam = np.linalg.inv(T_ee_cam)
    best = None

    for p_cam in sample_points_in_cube(cube_center):
        for roll in ROLL_LIST_DEG:
            T_base_cam = look_at_camera_pose_with_roll(p_cam, object_center, roll)
            T_base_ee = T_base_cam @ inv_T_ee_cam
            endpose = matrix_to_endpose_mm_deg(T_base_ee)

            result = reachability_test(endpose)
            if not result.get("reachable", False):
                continue

            dist_to_center = float(np.linalg.norm(p_cam - cube_center))
            roll_abs = abs(float(roll))
            path_ok = bool(result.get("path_ok", False))

            score = 0.0
            score += 1000.0 if path_ok else 0.0
            score -= dist_to_center * 100.0
            score -= roll_abs * 0.01

            record = {
                "cube_name": cube_name,
                "roll": float(roll),
                "endpose": endpose,
                "joint_degrees": [round(float(v), 2) for v in result.get("joint_degrees", result.get("q_solution_deg", []))],
                "reachable": bool(result.get("reachable", False)),
                "path_ok": path_ok,
                "pos_err_mm": round(float(result.get("pos_err_mm", 0.0)), 4),
                "rot_err_deg": round(float(result.get("rot_err_deg", 0.0)), 4),
                "score": score,
            }

            if best is None or record["score"] > best["score"]:
                best = record

    return best


# =========================
# Main
# =========================
def main():
    image_path = find_image_path()
    depth = np.load(DEPTH_PATH)

    color_intrinsic = load_camera_config(CAMERA_CONFIG_PATH)
    fx = color_intrinsic["fx"]
    fy = color_intrinsic["fy"]
    cx = color_intrinsic["ppx"]
    cy = color_intrinsic["ppy"]

    T_base_ee0 = parse_endpose_json(ENDPOSE_PATH)
    T_ee_cam = load_matrix(HAND_EYE_PATH)
    T_base_cam0 = T_base_ee0 @ T_ee_cam

    points_base_meta = np.load('E:/HKUSTGZ/AAM/construction/data/frame_point_result.npz', allow_pickle=True)
    points_base = points_base_meta['points_collection'].reshape(23319, 4)
    points_base = points_base[:,:3]

    object_center, bbox_size = estimate_target(points_base)
    bbox_diag = float(np.linalg.norm(bbox_size))
    radius = float(np.clip(RADIUS_SCALE * bbox_diag, RADIUS_MIN, RADIUS_MAX))
    p_cam0 = T_base_cam0[:3, 3]

    cube_centers = build_cube_centers(object_center, p_cam0, radius)

    scan_records = []
    for name, center, direction in cube_centers:
        print(f"Searching cube: {name}, center={center}")
        rec = find_best_pose_in_cube(name, center, object_center, T_ee_cam)
        if rec is None:
            print(f"WARNING: no reachable pose found in cube {name}")
            continue

        rec["idx"] = len(scan_records)
        rec["radius"] = round(radius, 4)
        rec["cube_center_base_m"] = [round(float(v), 4) for v in center]
        rec["object_center_base_m"] = [round(float(v), 4) for v in object_center]
        scan_records.append(rec)
        print(f"  Found reachable pose: idx={rec['idx']}, path_ok={rec['path_ok']}, roll={rec['roll']}")

    save_json(OUTPUT_PATH, scan_records)

    print("\n========== Coarse scan result ==========")
    print(f"Object center [m]: {object_center}")
    print(f"BBox size [m]: {bbox_size}")
    print(f"BBox diag [m]: {bbox_diag:.4f}")
    print(f"Radius [m]: {radius:.4f}")
    print(f"Saved {len(scan_records)} coarse scan poses to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

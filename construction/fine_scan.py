"""
fine_scan.py

Plan fine-scan viewpoints around defect ROI center.

Input:
  E:/HKUSTGZ/AAM/construction/data/coarse_scan/defect_roi_result.json
  E:/HKUSTGZ/AAM/construction/data/coarse_scan/coarse_point_result.npz
  E:/HKUSTGZ/AAM/construction/data/coarse_scan/coarse_icp_result.json
  E:/HKUSTGZ/AAM/config/calibration/right_camera/ecT.npy

Output:
  E:/HKUSTGZ/AAM/construction/data/coarse_scan/fine_scanpose.json

Each fine-scan cube is a small camera-position search region.
The optimizer searches robot joint angles so that:
  1. camera optical center is inside / near the cube
  2. camera +Z optical axis looks at defect_center
  3. joints stay inside safe bounds
"""

import json
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

PROJECT_ROOT = Path(os.getenv("AAM_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
sys.path.append(str(PROJECT_ROOT))
from Piper.endpose_reachability_safe import (
    load_arm_model,
    get_safe_bounds,
    frame_pose,
    DEFAULT_EE_FRAME,
)


# =========================
# Paths / parameters
# =========================
DATA_DIR = PROJECT_ROOT / "construction" / "data"
COARSE_SCAN_DIR = DATA_DIR / "coarse_scan"

DEFECT_ROI_JSON = COARSE_SCAN_DIR / "defect_roi_result.json"
COARSE_POINT_FILE = COARSE_SCAN_DIR / "coarse_point_result.npz"
COARSE_ICP_RESULT_FILE = COARSE_SCAN_DIR / "coarse_icp_result.json"
PNG_SEQUENCE_FILE = COARSE_SCAN_DIR / "coarse_png_sequence.json"
HAND_EYE_PATH = PROJECT_ROOT / "config" / "calibration" / "right_camera" / "ecT.npy"

OUTPUT_PATH = DATA_DIR / "fine_scan/fine_scanpose.json"

# Fine scan geometry
# 60 mm cube for each candidate camera-position region.
CUBE_SIZE_M = np.array([0.06, 0.06, 0.06])
CUBE_TOL_M = 0.03

# Camera-to-defect distance for fine scan.
FINE_RADIUS_SCALE = 1.0
FINE_RADIUS_MIN = 0.12
FINE_RADIUS_MAX = 0.28
DEV_LOOK_OFFSET_M = 0.05

# Optimization settings
N_RANDOM_SEEDS = 40
RANDOM_SEED = 1
MAX_NFEV = 600
POS_SCALE = 0.005
LOOK_ANGLE_TOL_DEG = 10.0
LOOK_SCALE = np.deg2rad(3.0)
JOINT_LIMIT_FALLBACK_DEG = 2.0
JOINT_LIMIT_WARN_DEG = 5.0
JOINT_LIMIT_PENALTY_PER_DEG = 80.0

# Base axes
BASE_X = np.array([1.0, 0.0, 0.0])
BASE_Y = np.array([0.0, 1.0, 0.0])
BASE_Z = np.array([0.0, 0.0, 1.0])

EULER_ORDER = "xyz"


def configure_paths(args):
    global DATA_DIR, COARSE_SCAN_DIR, DEFECT_ROI_JSON, COARSE_POINT_FILE
    global COARSE_ICP_RESULT_FILE, PNG_SEQUENCE_FILE, HAND_EYE_PATH, OUTPUT_PATH

    if args.run_dir:
        DATA_DIR = Path(args.run_dir) / "construction"
        COARSE_SCAN_DIR = DATA_DIR / "coarse_scan"
        HAND_EYE_PATH = PROJECT_ROOT / "config" / "calibration" / "right_camera" / "ecT.npy"

    if args.coarse_scan_dir:
        COARSE_SCAN_DIR = Path(args.coarse_scan_dir)
    if args.output_json:
        OUTPUT_PATH = Path(args.output_json)
    else:
        OUTPUT_PATH = DATA_DIR / "fine_scan" / "fine_scanpose.json"
    if args.hand_eye:
        HAND_EYE_PATH = Path(args.hand_eye)

    DEFECT_ROI_JSON = COARSE_SCAN_DIR / "defect_roi_result.json"
    COARSE_POINT_FILE = COARSE_SCAN_DIR / "coarse_point_result.npz"
    COARSE_ICP_RESULT_FILE = COARSE_SCAN_DIR / "coarse_icp_result.json"
    PNG_SEQUENCE_FILE = COARSE_SCAN_DIR / "coarse_png_sequence.json"
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# =========================
# Basic utilities
# =========================
def normalize(v, eps=1e-9):
    v = np.asarray(v, dtype=float).reshape(3)
    n = np.linalg.norm(v)
    if n < eps:
        return np.array([1.0, 0.0, 0.0])
    return v / n


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_matrix(path):
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.shape == () and arr.dtype == object:
        arr = arr.item()
    arr = np.asarray(arr, dtype=float)
    return arr


def load_points_and_poses(npz_path):
    """Load points_collection and bcT_collection from coarse_point_result.npz."""
    with np.load(npz_path, allow_pickle=True) as meta:
        points_collection = [np.asarray(p, dtype=float) for p in meta["points_collection"]]
        bcT_collection = [np.asarray(T, dtype=float) for T in meta["bcT_collection"]]
    return points_collection, bcT_collection


def resolve_target_index(icp_result, sequence_names):
    target_name = icp_result.get("target_name", None)
    target_index = icp_result.get("target_index", None)

    if target_name is not None and target_name in sequence_names:
        return sequence_names.index(target_name), target_name

    if target_index is not None:
        target_index = int(target_index)
        return target_index, sequence_names[target_index]

    raise ValueError("Cannot resolve target frame from coarse_icp_result.json")


def matrix_to_endpose_mm_deg(T_base_ee):
    xyz_mm = T_base_ee[:3, 3] * 1000.0
    rpy_deg = R.from_matrix(T_base_ee[:3, :3]).as_euler(EULER_ORDER, degrees=True)
    return [round(float(v), 2) for v in np.concatenate([xyz_mm, rpy_deg])]


def rotate_about_z(v, yaw_deg):
    return R.from_euler("z", yaw_deg, degrees=True).as_matrix() @ v


# =========================
# Fine cube planning
# =========================
def get_front_direction(defect_center, target_camera_position):
    """
    Define local front direction from defect_center to target camera.
    Project to base XY plane so yaw offsets are stable.
    """
    front = np.asarray(target_camera_position - defect_center, dtype=float)
    front[2] = 0.0
    if np.linalg.norm(front) < 1e-6:
        front = -BASE_X.copy()
    return normalize(front)


def spherical_direction(front, yaw_deg, elevation_deg):
    """
    Local spherical direction.
    yaw: rotate around base Z relative to front.
    elevation: raise from XY plane toward +Z.
    """
    yaw_dir = normalize(rotate_about_z(front, yaw_deg))
    elev = np.deg2rad(elevation_deg)
    direction = np.cos(elev) * yaw_dir + np.sin(elev) * BASE_Z
    return normalize(direction)


def build_fine_cube_centers(defect_center, target_camera_position, radius):
    """
    7 local fine-scan cubes around defect_center.

    Design:
      - front_mid / front_up: cover defect from the target/front direction.
      - left/right front oblique: cover both front-side edges.
      - left/right side oblique: cover side surfaces near the defect corner.
      - high_front: a steeper top-front view to reduce occlusion.
    """
    front = get_front_direction(defect_center, target_camera_position)

    views = [
        # name, yaw_deg, elevation_deg, radius_scale
        ("front_low",       0.0,    0.0,    1.00),
        ("front_mid",       0.0,    25.0,   1.00),
        ("front_up",        0.0,    50.0,   1.00),

        ("left_front_up",   45.0,   30.0,   1.00),
        ("left_front_side", 60.0,   0.0,    1.0),
        ("left_side_mid",   90.0,   10.0,    1.05),
        ("left_side_dev",  90.0,   20.0,    1.05),
        ("left_side_up_dev",  90.0,   30,    1.05),

        ("right_front_up",  -45.0,  30.0,   1.00),
        ("right_front_side",-60.0,  0.0,    1.0),
        ("right_side_mid",  -90.0,  10.0,    1.05),
        ("right_side_dev", -90.0,  20.0,    1.05),
        ("right_side_up_dev", -90.0,  30,    1.05),        


    ]

    cube_records = []
    dev_names = {
        "left_side_dev",
        "right_side_dev",
        "left_side_up_dev",
        "right_side_up_dev",
    }
    dev_center = defect_center + DEV_LOOK_OFFSET_M * BASE_X

    for name, yaw, elev, r_scale in views:
        direction = spherical_direction(front, yaw, elev)
        cube_center = defect_center + radius * r_scale * direction
        look_target = dev_center if name in dev_names else defect_center
        cube_records.append((name, cube_center, direction, yaw, elev, radius * r_scale, look_target, name in dev_names))

    return cube_records


def compute_fine_radius(roi_result):
    defect_bbox_size = np.asarray(roi_result["defect_roi_size_m"], dtype=float)
    bbox_diag = float(np.linalg.norm(defect_bbox_size))
    radius = float(np.clip(FINE_RADIUS_SCALE * bbox_diag, FINE_RADIUS_MIN, FINE_RADIUS_MAX))
    return radius, bbox_diag, defect_bbox_size


# =========================
# Optimization helpers
# =========================
def cube_outside_residual(p, cube_center):
    half = CUBE_SIZE_M / 2.0
    outside = np.maximum(np.abs(p - cube_center) - half, 0.0)
    return outside / POS_SCALE


def look_at_residual(T_base_cam, look_target):
    p_cam = T_base_cam[:3, 3]
    z_cam = T_base_cam[:3, 2]
    target_dir = normalize(look_target - p_cam)
    return (z_cam - target_dir) / LOOK_SCALE


def camera_look_angle_deg(T_base_cam, look_target):
    p_cam = T_base_cam[:3, 3]
    z_cam = normalize(T_base_cam[:3, 2])
    target_dir = normalize(look_target - p_cam)
    dot_value = float(np.clip(np.dot(z_cam, target_dir), -1.0, 1.0))
    return float(np.rad2deg(np.arccos(dot_value)))


def is_inside_cube(p, cube_center):
    half = CUBE_SIZE_M / 2.0 + CUBE_TOL_M
    return bool(np.all(np.abs(p - cube_center) <= half))


def q_to_cam_pose(model, data, frame_id, q, T_ee_cam):
    T_base_ee_pin = frame_pose(model, data, q, frame_id)

    T_base_ee = np.eye(4)
    T_base_ee[:3, :3] = T_base_ee_pin.rotation
    T_base_ee[:3, 3] = T_base_ee_pin.translation

    T_base_cam = T_base_ee @ T_ee_cam
    return T_base_ee, T_base_cam


def joint_limit_margins_deg(q, lb, ub):
    margins_rad = np.minimum(q - lb, ub - q)
    return np.rad2deg(margins_rad)


def pull_back_from_joint_limits(q, lb, ub, margin_deg=JOINT_LIMIT_FALLBACK_DEG):
    margin = np.deg2rad(margin_deg)
    lower = lb + margin
    upper = ub - margin
    too_narrow = lower > upper
    center = 0.5 * (lb + ub)
    lower = np.where(too_narrow, center, lower)
    upper = np.where(too_narrow, center, upper)

    q_safe = np.clip(q, lower, upper)
    adjusted = bool(np.any(np.abs(q_safe - q) > 1e-10))
    return q_safe, adjusted


def joint_limit_score_penalty(min_margin_deg):
    if min_margin_deg >= JOINT_LIMIT_WARN_DEG:
        return 0.0
    return (JOINT_LIMIT_WARN_DEG - min_margin_deg) * JOINT_LIMIT_PENALTY_PER_DEG


def optimize_one_cube(cube_info, look_target, model, frame_id, T_ee_cam, lb, ub, seed_qs):
    cube_name, cube_center, direction, yaw_deg, elev_deg, radius, _, is_dev_view = cube_info

    best = None
    W_LOOK = 0.98
    W_CUBE = 0.02

    cube_scale = np.maximum(CUBE_SIZE_M / 2.0, 1e-6)
    look_scale = max(2.0 * np.sin(np.deg2rad(LOOK_ANGLE_TOL_DEG) / 2.0), 1e-6)

    for q0 in seed_qs:
        data = model.createData()

        def residual(q):
            _, T_base_cam = q_to_cam_pose(model, data, frame_id, q, T_ee_cam)
            p_cam = T_base_cam[:3, 3]
            r_cube = cube_outside_residual(p_cam, cube_center) / cube_scale
            r_look = look_at_residual(T_base_cam, look_target) / look_scale
            return np.concatenate([
                np.sqrt(W_CUBE) * r_cube,
                np.sqrt(W_LOOK) * r_look,
            ])

        res = least_squares(
            residual,
            q0,
            bounds=(lb, ub),
            max_nfev=MAX_NFEV,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
        )

        q_raw = res.x
        raw_limit_margins_deg = joint_limit_margins_deg(q_raw, lb, ub)
        q, joint_limit_adjusted = pull_back_from_joint_limits(q_raw, lb, ub)
        limit_margins_deg = joint_limit_margins_deg(q, lb, ub)
        min_limit_margin_deg = float(np.min(limit_margins_deg))

        T_base_ee, T_base_cam = q_to_cam_pose(model, data, frame_id, q, T_ee_cam)
        p_cam = T_base_cam[:3, 3]

        inside = is_inside_cube(p_cam, cube_center)
        look_angle = camera_look_angle_deg(T_base_cam, look_target)
        cube_err_m = float(np.linalg.norm(np.maximum(np.abs(p_cam - cube_center) - CUBE_SIZE_M / 2.0, 0.0)))
        feasible = inside and (look_angle < LOOK_ANGLE_TOL_DEG)

        score = 0.0
        score += 10000.0 if feasible else 0.0
        score -= 250.0 * look_angle
        score -= 600.0 * cube_err_m
        score -= 1.0 * float(np.linalg.norm(p_cam - cube_center))
        score -= joint_limit_score_penalty(float(np.min(raw_limit_margins_deg)))

        rec = {
            "cube_name": cube_name,
            "success": bool(feasible),
            "least_squares_success": bool(res.success),
            "joint_limit_adjusted": joint_limit_adjusted,
            "min_joint_limit_margin_deg": round(min_limit_margin_deg, 4),
            "raw_min_joint_limit_margin_deg": round(float(np.min(raw_limit_margins_deg)), 4),
            "look_angle_deg": round(float(look_angle), 4),
            "cube_outside_err_mm": round(cube_err_m * 1000.0, 4),
            "camera_position_base_m": [round(float(v), 5) for v in p_cam],
            "cube_center_base_m": [round(float(v), 5) for v in cube_center],
            "look_target_base_m": [round(float(v), 5) for v in look_target],
            "is_dev_view": bool(is_dev_view),
            "direction_from_defect_center": [round(float(v), 5) for v in direction],
            "yaw_deg": round(float(yaw_deg), 2),
            "elevation_deg": round(float(elev_deg), 2),
            "radius_m": round(float(radius), 4),
            "cube_size_m": [round(float(v), 5) for v in CUBE_SIZE_M],
            "endpose": matrix_to_endpose_mm_deg(T_base_ee),
            "joint_degrees": [round(float(v), 2) for v in np.rad2deg(q)],
            "raw_joint_degrees": [round(float(v), 2) for v in np.rad2deg(q_raw)],
            "jointctrl_args": [int(round(float(v) * 1000.0)) for v in np.rad2deg(q)],
            "score": float(score),
        }

        if best is None or rec["score"] > best["score"]:
            best = rec

        if feasible:
            break

    return best


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--coarse-scan-dir", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--hand-eye", type=Path, default=None)
    args = parser.parse_args()
    configure_paths(args)

    roi_result = load_json(DEFECT_ROI_JSON)
    defect_center = np.asarray(roi_result["defect_roi_center_world_m"], dtype=float)
    fine_radius, defect_bbox_diag, defect_bbox_size = compute_fine_radius(roi_result)

    icp_result = load_json(COARSE_ICP_RESULT_FILE)
    sequence_names = load_json(PNG_SEQUENCE_FILE)
    target_index, target_name = resolve_target_index(icp_result, sequence_names)

    _, bcT_collection = load_points_and_poses(COARSE_POINT_FILE)
    T_base_cam_target = np.asarray(bcT_collection[target_index], dtype=float)
    target_camera_position = T_base_cam_target[:3, 3]

    T_ee_cam = load_matrix(HAND_EYE_PATH)

    fine_cubes = build_fine_cube_centers(
        defect_center=defect_center,
        target_camera_position=target_camera_position,
        radius=fine_radius,
    )

    model = load_arm_model()
    frame_id = model.getFrameId(DEFAULT_EE_FRAME)
    lb, ub = get_safe_bounds(model)

    rng = np.random.default_rng(RANDOM_SEED)
    q_neutral = np.clip(pin.neutral(model), lb, ub)
    q_zero = np.clip(np.zeros(model.nq), lb, ub)
    seed_qs = [q_neutral, q_zero]
    for _ in range(N_RANDOM_SEEDS):
        seed_qs.append(rng.uniform(lb, ub))

    records = []
    for i, cube_info in enumerate(fine_cubes):
        name, cube_center, _, yaw, elev, radius, look_target, _ = cube_info
        print(f"{name}: yaw={yaw}, elev={elev}")

        rec = optimize_one_cube(
            cube_info=cube_info,
            look_target=look_target,
            model=model,
            frame_id=frame_id,
            T_ee_cam=T_ee_cam,
            lb=lb,
            ub=ub,
            seed_qs=seed_qs,
        )
        rec["idx"] = i
        rec["defect_center_base_m"] = [round(float(v), 6) for v in defect_center]
        rec["dev_center_base_m"] = [round(float(v), 6) for v in defect_center + DEV_LOOK_OFFSET_M * BASE_X]
        rec["dev_look_offset_m"] = DEV_LOOK_OFFSET_M
        rec["defect_bbox_size_m"] = [round(float(v), 6) for v in defect_bbox_size]
        rec["defect_bbox_diag_m"] = round(float(defect_bbox_diag), 6)
        rec["fine_radius_scale"] = FINE_RADIUS_SCALE
        rec["fine_radius_m"] = round(float(fine_radius), 6)
        rec["target_frame_name_for_front"] = target_name
        records.append(rec)

        print(
            f"  success={rec['success']}, "
            f"look_angle={rec['look_angle_deg']} deg, "
            f"cube_err={rec['cube_outside_err_mm']} mm"
        )

    save_json(OUTPUT_PATH, records)

    print("\n========== Fine scan result ==========")
    print(f"Defect center [m]: {defect_center}")
    print(f"Defect bbox size [m]: {defect_bbox_size}")
    print(f"Defect bbox diag [m]: {defect_bbox_diag:.4f}")
    print(f"Target frame used for front direction: {target_name}")
    print(f"Fine radius [m]: {fine_radius:.4f}")
    print(f"Fine cube size [m]: {CUBE_SIZE_M}")
    print(f"Saved {len(records)} fine scan poses to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

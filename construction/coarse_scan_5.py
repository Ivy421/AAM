"""
优化方向：
1. 五个cube的选择，俯视点太极端
2. radius_scale的制定：若某个cube不可达，迭代修改该参数搜索可达点
3. 对于关节角抵达limit，go_Zero函数调用有问题，要不就跑两次enbale + go_zero, 要不就limit的关节角+-1度
"""



import cv2
import json
import sys
from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

# Project import path
sys.path.append("/home/smmg/AAM")
from Piper.endpose_reachability import *

# =========================
# Fixed paths / parameters
# =========================
DATA_DIR = Path(r"/home/smmg/AAM/construction/data")
STEM = "test"

ENDPOSE_PATH = DATA_DIR / f"{STEM}.json"
OUTPUT_PATH = DATA_DIR / "coarse_scan/coarse_scanpose.json"

# Object point cloud from the first frame.
# Assumption: points_collection[2] is already expressed in robot base coordinate, unit: meter.
META_PATH = Path(r"/home/smmg/AAM/construction/data/initial_frame_point.npz")

# Hand-eye: T_ee_cam
HAND_EYE_PATH = Path(r"/home/smmg/AAM/config/calibration/right_camera/ecT.npy")

# Unit conversion for current robot endpose json
POSITION_SCALE = 1e-6    # raw position -> meter
ANGLE_SCALE = 1e-3       # raw angle -> degree
ANGLE_UNIT = "deg"
EULER_ORDER = "xyz"

# Coarse scan geometry
RADIUS_SCALE = 2.0
RADIUS_MIN = 0.20
RADIUS_MAX = 1
CUBE_SIZE_M = np.array([0.15, 0.15, 0.15])   # 15 x 15 x 15 mm

# Optimization settings
N_RANDOM_SEEDS = 30
RANDOM_SEED = 0
MAX_NFEV = 500
POS_SCALE = 0.005                 # 5 mm normalization for cube outside distance
LOOK_SCALE = np.deg2rad(3.0)      # about 3 deg normalization for look-at error
LOOK_ANGLE_TOL_DEG = 10          # accepted optical-axis error
CUBE_TOL_M = 0.02                # allow 2 mm numerical tolerance outside cube

# Local object coordinate is translated base coordinate:
# origin = object_center, axes are same as robot base: x forward, y left, z up.
BASE_X = np.array([1.0, 0.0, 0.0])
BASE_Y = np.array([0.0, 1.0, 0.0])
BASE_Z = np.array([0.0, 0.0, 1.0])


def normalize(v, eps=1e-9):
    v = np.asarray(v, dtype=float).reshape(3)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError(f"Cannot normalize near-zero vector: {v}")
    return v / n


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_matrix(path):
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.shape == () and arr.dtype == object:
        arr = arr.item()
    arr = np.asarray(arr, dtype=float)
    if arr.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix from {path}, got {arr.shape}")
    return arr


def parse_endpose_json(path):
    """Parse first-frame robot endpose json into T_base_ee."""
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

def load_initial_prior():
    meta = np.load(META_PATH, allow_pickle=True)

    points = np.asarray(meta["points_collection"][0], dtype=float)
    points = points[:, :3]

    T_base_cam0 = np.asarray(meta["bcT_collection"][0], dtype=float)
    object_center = np.asarray(meta['object_center'])
    object_center = object_center[:3]
    bbox_size = meta['object_bbox']


    return points, T_base_cam0,object_center,bbox_size


def matrix_to_endpose_mm_deg(T_base_ee):
    xyz_mm = T_base_ee[:3, 3] * 1000.0
    rpy_deg = R.from_matrix(T_base_ee[:3, :3]).as_euler(EULER_ORDER, degrees=True)
    return [round(float(v), 2) for v in np.concatenate([xyz_mm, rpy_deg])]


def rotate_about_z(v, yaw_deg):
    return R.from_euler("z", yaw_deg, degrees=True).as_matrix() @ v


def get_front_direction(object_center, p_cam0):
    """
    front direction: object_center -> first camera optical center,
    projected onto the object local z=0 plane.
    Since object local axes are same as base axes, this is the base XY plane.
    """
    v = np.asarray(p_cam0 - object_center, dtype=float)
    v[2] = 0.0
    if np.linalg.norm(v) < 1e-6:
        v = -BASE_X.copy()
    return normalize(v)


def build_cube_centers(object_center, p_cam0, radius):
    front = get_front_direction(object_center, p_cam0)
    front_up= normalize(np.cos(np.deg2rad(45.0)) * front + np.sin(np.deg2rad(45.0)) * BASE_Z)
    #先把 front 在水平面内绕 z 轴旋转 +45°，得到左前方方向。然后再向上抬高 45°。
    #left_up_45 = normalize(np.cos(np.deg2rad(45.0)) * rotate_about_z(front, 45.0) + np.sin(np.deg2rad(45.0)) * BASE_Z)
    #left_up_45 = normalize(np.cos(np.deg2rad(30.0)) * rotate_about_z(front, 30.0) + np.sin(np.deg2rad(45.0)) * BASE_Z)
    left_up = normalize(
    np.cos(np.deg2rad(45.0)) * rotate_about_z(front, 30.0)
    + np.sin(np.deg2rad(45.0)) * BASE_Z)
    right_up = normalize(np.cos(np.deg2rad(30.0)) * rotate_about_z(front, -30.0) + np.sin(np.deg2rad(45.0)) * BASE_Z)
    #up = normalize(  BASE_Z.copy() * np.cos(np.deg2rad(30.0)) )
    up = normalize(
    np.cos(np.deg2rad(70.0)) * front
    + np.sin(np.deg2rad(70.0)) * BASE_Z)

    cube_defs = [
        ("front", front),
        ("upper_front_45", front_up),
        ("up", up),
        ("left_up_45", left_up),
        ("right_up_45", right_up),
    ]

    return [(name, object_center + radius * direction, direction) for name, direction in cube_defs]


def cube_outside_residual(p, cube_center):
    """
    Residual is zero if p is inside the cube.
    If p is outside, residual is the axis-wise outside distance.
    """
    half = CUBE_SIZE_M / 2.0
    outside = np.maximum(np.abs(p - cube_center) - half, 0.0)
    return outside / POS_SCALE


def look_at_residual(T_base_cam, object_center):
    """Camera +Z axis should point from camera position to object_center."""
    p_cam = T_base_cam[:3, 3]
    z_cam = T_base_cam[:3, 2]
    target_dir = normalize(object_center - p_cam)

    # z_cam - target_dir is 0 only when the camera optical axis exactly looks at object_center.
    return (z_cam - target_dir) / LOOK_SCALE


def camera_look_angle_deg(T_base_cam, object_center):
    p_cam = T_base_cam[:3, 3]
    z_cam = normalize(T_base_cam[:3, 2])
    target_dir = normalize(object_center - p_cam)
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


def optimize_one_cube(cube_name, cube_center, object_center, model, frame_id, T_ee_cam, lb, ub, seed_qs):
    """
    Directly optimize joint angles q.

    Goal:
    1. camera optical center lies inside / near the cube
    2. camera +Z optical axis looks at object_center
    3. q stays inside URDF joint limits through least_squares bounds

    Residual design:
    - cube residual is normalized by cube half size
    - look residual is normalized by look-angle tolerance
    - look-at is more important: W_LOOK=0.99, W_CUBE=0.01
    """
    best = None

    # Weights in the final squared objective:
    # objective = 0.8 * ||look_norm||^2 + 0.2 * ||cube_norm||^2
    W_LOOK = 0.9
    W_CUBE = 0.1

    # least_squares minimizes sum of squared residuals,
    # so multiply residuals by sqrt(weight)
    SQRT_W_LOOK = np.sqrt(W_LOOK)
    SQRT_W_CUBE = np.sqrt(W_CUBE)

    # Normalize cube residual from meter to dimensionless.
    # cube_outside_residual is usually in meter.
    cube_scale = CUBE_SIZE_M / 2.0
    cube_scale = np.maximum(cube_scale, 1e-6)

    # Normalize look residual.
    # If look_at_residual is direction vector error:
    # ||z_cam - target_dir|| = 2 * sin(angle / 2)
    look_tol_rad = np.deg2rad(LOOK_ANGLE_TOL_DEG)
    look_scale = 2.0 * np.sin(look_tol_rad / 2.0)
    look_scale = max(look_scale, 1e-6)

    for q0 in seed_qs:
        data = model.createData()

        def residual(q):
            _, T_base_cam = q_to_cam_pose(model, data, frame_id, q, T_ee_cam)
            p_cam = T_base_cam[:3, 3]

            r_cube = cube_outside_residual(p_cam, cube_center)
            r_look = look_at_residual(T_base_cam, object_center)

            # Normalize
            r_cube_norm = r_cube / cube_scale
            r_look_norm = r_look / look_scale

            # Weighted residual
            return np.concatenate([
                SQRT_W_CUBE * r_cube_norm,
                SQRT_W_LOOK * r_look_norm,
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

        q = res.x
        T_base_ee, T_base_cam = q_to_cam_pose(model, data, frame_id, q, T_ee_cam)
        p_cam = T_base_cam[:3, 3]

        inside = is_inside_cube(p_cam, cube_center)
        look_angle = camera_look_angle_deg(T_base_cam, object_center)
        cube_err_m = float(
            np.linalg.norm(
                np.maximum(np.abs(p_cam - cube_center) - CUBE_SIZE_M / 2.0, 0.0)
            )
        )

        # Score: feasible first, then prioritize look angle, then cube error.
        feasible = inside and (look_angle < LOOK_ANGLE_TOL_DEG)

        record = {
            "cube_name": cube_name,
            "success": bool(feasible),
            "least_squares_success": bool(res.success),
            "look_angle_deg": round(look_angle, 4),
            "cube_outside_err_mm": round(cube_err_m * 1000.0, 4),
            "camera_position_base_m": [round(float(v), 5) for v in p_cam],
            "cube_center_base_m": [round(float(v), 5) for v in cube_center],
            "endpose": matrix_to_endpose_mm_deg(T_base_ee),
            "joint_degrees": [round(float(v), 2) for v in np.rad2deg(q)],
        }

        if best is None or record["score"] > best["score"]:
            best = record

        # Early stop if a good feasible solution is found.
        if feasible:
            break

    return best


def main():
    # 1. Load first-frame object point cloud in base coordinate.
    points_base, T_base_cam0 , object_center, bbox_size = load_initial_prior()
    #object_center, bbox_size = estimate_target(points_base)

    bbox_diag = float(np.linalg.norm(bbox_size))
    radius = float(np.clip(RADIUS_SCALE * bbox_diag, RADIUS_MIN, RADIUS_MAX))

    # 2. Load first-frame camera pose. Used only to define front direction.
    T_base_ee0 = parse_endpose_json(ENDPOSE_PATH)
    T_ee_cam = load_matrix(HAND_EYE_PATH)
    T_base_cam0 = T_base_ee0 @ T_ee_cam
    p_cam0 = T_base_cam0[:3, 3]

    # 3. Build five cube centers.
    cube_centers = build_cube_centers(object_center, p_cam0, radius)

    # 4. Load Piper model and joint limits.
    model = load_arm_model()
    data_tmp = model.createData()
    if not model.existFrame(DEFAULT_EE_FRAME):
        raise RuntimeError(f"Cannot find EE frame: {DEFAULT_EE_FRAME}")
    frame_id = model.getFrameId(DEFAULT_EE_FRAME)
    lb, ub = get_clean_bounds(model)

    # 5. Prepare joint seeds. Random seeds are used because no current joint state is assumed.
    rng = np.random.default_rng(RANDOM_SEED)
    q_neutral = np.clip(pin.neutral(model), lb, ub)
    q_zero = np.clip(np.zeros(model.nq), lb, ub)
    seed_qs = [q_neutral, q_zero]
    for _ in range(N_RANDOM_SEEDS):
        seed_qs.append(rng.uniform(lb, ub))

    # 6. Optimize one joint solution for each cube.
    scan_records = []
    for name, cube_center, direction in cube_centers:
        print(f"Optimizing cube: {name}, center={cube_center}")
        rec = optimize_one_cube(
            cube_name=name,
            cube_center=cube_center,
            object_center=object_center,
            model=model,
            frame_id=frame_id,
            T_ee_cam=T_ee_cam,
            lb=lb,
            ub=ub,
            seed_qs=seed_qs,
        )

        rec["idx"] = len(scan_records)
        rec["radius"] = round(radius, 4)
        rec["object_center_base_m"] = [round(float(v), 5) for v in object_center]
        rec["cube_size_m"] = [round(float(v), 5) for v in CUBE_SIZE_M]
        scan_records.append(rec)

        print(
            f"  success={rec['success']}, "
            f"look_angle={rec['look_angle_deg']} deg, "
            f"cube_err={rec['cube_outside_err_mm']} mm"
        )

    save_json(OUTPUT_PATH, scan_records)

    print("\n========== Coarse scan result ==========")
    print(f"Object center [m]: {object_center}")
    print(f"BBox size [m]: {bbox_size}")
    print(f"BBox diag [m]: {bbox_diag:.4f}")
    print(f"Radius [m]: {radius:.4f}")
    print(f"Saved {len(scan_records)} coarse scan poses to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

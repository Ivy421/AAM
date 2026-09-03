"""
Eye-in-hand hand-eye calibration.

Input folder format:
    data_dir/N.png
    data_dir/N.json

Each json saves robot endpose: x, y, z, rx, ry, rz
Output:
    ecT_20260727.csv
    ecT_20260727.npy
"""

import os
import csv
import cv2
import json
import numpy as np
from scipy.spatial.transform import Rotation as R


# ============================================================
# 0. User settings
# ============================================================

data_dir = "/home/smmg/AAM/config/calibration/right_camera/data/image_20260813"
camera_config_path = "/home/smmg/AAM/config/calibration/right_camera/camera_config.npy"

output_ecT_csv = os.path.join(data_dir, "ecT_20260813.csv")
output_ecT_npy = os.path.join(data_dir, "ecT_20260813.npy")

PATTERN_SIZE = (  10 , 7  )       # chessboard inner corners: (cols, rows)
SQUARE_SIZE_M = 0.015     # square size in meter

POS_SCALE = 1e-6            # x/y/z: 10^-6 m -> m
ANGLE_SCALE = 1e-3          # rx/ry/rz: 0.001 deg -> deg
FROM_EULER_ANGLE = "xyz"

PNP_MEAN_TH_PX = 0.5
PNP_MAX_TH_PX = 1.


# ============================================================
# 1. Basic utilities
# ============================================================

def load_endpose_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    pose = {}
    if isinstance(raw, list):
        for item in raw:
            pose.update(item)
    else:
        pose = raw

    xyz = np.array([pose["x"], pose["y"], pose["z"]], dtype=np.float64) * POS_SCALE
    rpy = np.array([pose["rx"], pose["ry"], pose["rz"]], dtype=np.float64) * ANGLE_SCALE

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_euler(FROM_EULER_ANGLE, rpy, degrees=True).as_matrix()
    T[:3, 3] = xyz
    return T


def load_calibration_pairs(folder_path):
    json_paths = []
    for name in os.listdir(folder_path):
        stem, ext = os.path.splitext(name)
        if ext.lower() == ".json" and stem.isdigit():
            json_paths.append((int(stem), os.path.join(folder_path, name)))

    pairs = []
    for idx, json_path in sorted(json_paths, key=lambda item: item[0]):
        image_path = os.path.join(folder_path, f"{idx}.png")
        if os.path.exists(image_path):
            pairs.append({
                "index": idx,
                "image_path": image_path,
                "base_T_ee": load_endpose_json(json_path),
            })
    return pairs


def load_camera_intrinsic(config_path):
    config = np.load(config_path, allow_pickle=True).item()
    intr = config["color_intrinsic"]

    K = np.array([
        [intr["fx"], 0.0, intr["ppx"]],
        [0.0, intr["fy"], intr["ppy"]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    dist = np.zeros((5, 1), dtype=np.float64)   # keep hard-coded zero distortion
    return K, dist


def make_chessboard_object_points(pattern_size, square_size_m):
    cols, rows = pattern_size
    objp = np.zeros((rows * cols, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size_m
    return objp


def detect_target_pose(image_path, K, dist, objp, pattern_size):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    flags = (
        cv2.CALIB_CB_NORMALIZE_IMAGE |
        cv2.CALIB_CB_EXHAUSTIVE |
        cv2.CALIB_CB_ACCURACY
    )

    found, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags)
    if not found:
        print(f"[REJECT] {os.path.basename(image_path)} | chessboard not found")
        return None

    corners = corners.reshape(-1, 2).astype(np.float64)

    ok, rvec, tvec = cv2.solvePnP(
        objp,
        corners,
        K,
        dist,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        print(f"[REJECT] {os.path.basename(image_path)} | solvePnP failed")
        return None

    projected, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    projected = projected.reshape(-1, 2)
    reproj_err = np.linalg.norm(projected - corners, axis=1)
    reproj_mean_px = float(np.mean(reproj_err))
    reproj_max_px = float(np.max(reproj_err))

    if reproj_mean_px >= PNP_MEAN_TH_PX or reproj_max_px >= PNP_MAX_TH_PX:
        print(
            f"[REJECT] {os.path.basename(image_path)} | "
            f"pnp_mean={reproj_mean_px:.4f}px, pnp_max={reproj_max_px:.4f}px"
        )
        return None

    R_target2cam, _ = cv2.Rodrigues(rvec)

    cam_T_target = np.eye(4, dtype=np.float64)
    cam_T_target[:3, :3] = R_target2cam
    cam_T_target[:3, 3] = tvec.reshape(3)

    return {
        "cam_T_target": cam_T_target,
        "corners": corners,
        "pnp_mean_px": reproj_mean_px,
        "pnp_max_px": reproj_max_px,
    }


def transform_points(T, points):
    points_h = np.column_stack([points, np.ones(len(points))])
    return (T @ points_h.T).T[:, :3]


def project_camera_points(points_cam, K):
    x = points_cam[:, 0]
    y = points_cam[:, 1]
    z = points_cam[:, 2]

    u = K[0, 0] * x / z + K[0, 2]
    v = K[1, 1] * y / z + K[1, 2]
    return np.column_stack([u, v])


def calc_cross_reprojection(records, objp, K, ecT):
    all_errors = []

    for src in records:
        base_T_target_src = src["base_T_ee"] @ ecT @ src["cam_T_target"]
        points_base = transform_points(base_T_target_src, objp.astype(np.float64))

        for dst in records:
            if src["index"] == dst["index"]:
                continue

            base_T_cam_dst = dst["base_T_ee"] @ ecT
            cam_T_base_dst = np.linalg.inv(base_T_cam_dst)
            points_cam_dst = transform_points(cam_T_base_dst, points_base)
            uv = project_camera_points(points_cam_dst, K)

            err = np.linalg.norm(uv - dst["corners"], axis=1)
            all_errors.extend(err.tolist())

    all_errors = np.asarray(all_errors, dtype=np.float64)
    return float(np.mean(all_errors)), float(np.max(all_errors))


def save_matrix_csv(path, T):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in T:
            writer.writerow(row)


def print_matrix(name, T):
    print(f"\n{name}:")
    np.set_printoptions(precision=8, suppress=True)
    print(T)


# ============================================================
# 2. Main hand-eye calibration
# ============================================================

def main():
    print("========== Eye-in-hand hand-eye calibration ==========")

    K, dist = load_camera_intrinsic(camera_config_path)
    print("\nCamera intrinsic K:")
    print(K)
    print("\nDistortion coeffs:")
    print(dist.reshape(-1))

    calibration_pairs = load_calibration_pairs(data_dir)
    print(f"\nLoaded calibration pairs: {len(calibration_pairs)}")

    objp = make_chessboard_object_points(PATTERN_SIZE, SQUARE_SIZE_M)

    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []
    records = []

    for pair in calibration_pairs:
        result = detect_target_pose(
            image_path=pair["image_path"],
            K=K,
            dist=dist,
            objp=objp,
            pattern_size=PATTERN_SIZE,
        )
        if result is None:
            continue

        base_T_ee = pair["base_T_ee"]
        cam_T_target = result["cam_T_target"]

        R_gripper2base.append(base_T_ee[:3, :3])
        t_gripper2base.append(base_T_ee[:3, 3].reshape(3, 1))
        R_target2cam.append(cam_T_target[:3, :3])
        t_target2cam.append(cam_T_target[:3, 3].reshape(3, 1))

        records.append({
            "index": pair["index"],
            "base_T_ee": base_T_ee,
            "cam_T_target": cam_T_target,
            "corners": result["corners"],
        })

        print(
            f"[OK] {pair['index']}.png | "
            f"pnp_mean={result['pnp_mean_px']:.4f}px, "
            f"pnp_max={result['pnp_max_px']:.4f}px"
        )

    if len(records) < 4:
        raise RuntimeError(f"Valid calibration pairs are too few: {len(records)}")

    print(f"\nValid pairs used: {len(records)}")
    print("Used image indices:", [r["index"] for r in records])

    R_cam2ee, t_cam2ee = cv2.calibrateHandEye(
        R_gripper2base=R_gripper2base,
        t_gripper2base=t_gripper2base,
        R_target2cam=R_target2cam,
        t_target2cam=t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI,
    )

    ecT = np.eye(4, dtype=np.float64)
    ecT[:3, :3] = R_cam2ee
    ecT[:3, 3] = t_cam2ee.reshape(3)

    print_matrix("ecT = end_effector_T_camera", ecT)

    cross_mean, cross_max = calc_cross_reprojection(records, objp, K, ecT)
    print(f"\ncross_reproj_mean_px: {cross_mean:.4f}")
    print(f"cross_reproj_max_px:  {cross_max:.4f}")

    save_matrix_csv(output_ecT_csv, ecT)
    np.save(output_ecT_npy, ecT)

    print("\nSaved:")
    print(output_ecT_csv)
    print(output_ecT_npy)
    print("\nDone.")


if __name__ == "__main__":
    main()

"""
手眼标定用这份代码
输入：image/x.png
endpose.csv 该数据直接保存机械臂末端endpose信息
"""

import os, csv, cv2, json
import numpy as np
from scipy.spatial.transform import Rotation as R

# ============================================================
# 0. User settings
# ============================================================

data_dir = '/home/smmg/AAM/config/alignment/test_data'
camera_config_path = '/home/smmg/AAM/config/calibration/right_camera/camera_config.npy'

output_ecT_csv = os.path.join(data_dir, "ecT_20260727.csv")
output_ecT_npy = os.path.join(data_dir, "ecT_20260727.npy")
output_used_pairs_csv = os.path.join(data_dir, "used_pairs_20260727.csv")

# 棋盘格参数：必须按你的标定板修改
PATTERN_SIZE = (9,6)       # chessboard inner corners: (cols, rows)
SQUARE_SIZE_M = 0.0235       # square size in meter

# endpose2.csv 单位
POS_SCALE = 1e-6            # x/y/z: 10^-6 m -> m
ANGLE_SCALE = 1e-3          # rx/ry/rz: 0.001 deg -> deg

FROM_EULER_ANGLE = "xyz"    # 用户要求


# ============================================================
# 1. Basic utilities
# ============================================================

def is_float(x):
    try:
        float(str(x).strip())
        return True
    except Exception:
        return False


def load_endposes(csv_path):
    """
    读取 endpose2.csv

    输入格式:
        x, y, z, rx, ry, rz

    单位:
        x/y/z: 10^-6 m
        rx/ry/rz: 0.001 degree

    输出:
        list of 4x4 matrix, each is base_T_ee
    """
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))

    rows = [r for r in rows if len(r) > 0 and any(str(c).strip() for c in r)]
    if len(rows) == 0:
        raise ValueError(f"Empty CSV: {csv_path}")

    first = rows[0]
    has_header = any(not is_float(c) for c in first)

    if has_header:
        headers = [str(c).strip().lower() for c in first]
        data_rows = rows[1:]

        required = ["x", "y", "z", "rx", "ry", "rz"]
        for k in required:
            if k not in headers:
                raise ValueError(f"Missing column '{k}' in {csv_path}. Existing headers: {headers}")

        idx = {k: headers.index(k) for k in required}

        values = []
        for r in data_rows:
            values.append([
                float(r[idx["x"]]),
                float(r[idx["y"]]),
                float(r[idx["z"]]),
                float(r[idx["rx"]]),
                float(r[idx["ry"]]),
                float(r[idx["rz"]]),
            ])
    else:
        values = []
        for r in rows:
            if len(r) < 6:
                continue
            values.append([float(r[i]) for i in range(6)])

    base_T_ee_list = []

    for i, row in enumerate(values):
        x, y, z, rx, ry, rz = row

        t = np.array([x, y, z], dtype=np.float64) * POS_SCALE
        euler_deg = np.array([rx, ry, rz], dtype=np.float64) * ANGLE_SCALE

        rot = R.from_euler(FROM_EULER_ANGLE, euler_deg, degrees=True)
        R_mat = rot.as_matrix()

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_mat
        T[:3, 3] = t

        base_T_ee_list.append(T)

    return base_T_ee_list


def load_endpose_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    pose = {}
    if isinstance(raw, list):
        for item in raw:
            pose.update(item)
    else:
        pose = raw

    x = float(pose["x"])
    y = float(pose["y"])
    z = float(pose["z"])
    rx = float(pose["rx"])
    ry = float(pose["ry"])
    rz = float(pose["rz"])

    t = np.array([x, y, z], dtype=np.float64) * POS_SCALE
    euler_deg = np.array([rx, ry, rz], dtype=np.float64) * ANGLE_SCALE

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_euler(FROM_EULER_ANGLE, euler_deg, degrees=True).as_matrix()
    T[:3, 3] = t
    return T


def load_calibration_pairs(folder_path):
    json_paths = []
    for name in os.listdir(folder_path):
        stem, ext = os.path.splitext(name)
        if ext.lower() == ".json" and stem.isdigit():
            json_paths.append((int(stem), os.path.join(folder_path, name)))

    json_paths = sorted(json_paths, key=lambda item: item[0])

    pairs = []
    for idx, json_path in json_paths:
        image_path = os.path.join(folder_path, f"{idx}.png")
        if not os.path.exists(image_path):
            print(f"[WARN] Image missing for endpose: {image_path}")
            continue
        pairs.append({
            "index": idx,
            "image_path": image_path,
            "base_T_ee": load_endpose_json(json_path),
        })

    return pairs


def load_camera_intrinsic(camera_config_path):
    """
    加载 camera_config.npy

    用户给定格式:
        config = np.load(camera_config_path, allow_pickle=True).item()
        color_intrinsic = config['color_intrinsic']
        fx = color_intrinsic['fx']
        fy = color_intrinsic['fy']
        cx = color_intrinsic['ppx']
        cy = color_intrinsic['ppy']
    """
    config = np.load(camera_config_path, allow_pickle=True).item()

    color_intrinsic = config["color_intrinsic"]
    fx = color_intrinsic["fx"]
    fy = color_intrinsic["fy"]
    cx = color_intrinsic["ppx"]
    cy = color_intrinsic["ppy"]

    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    # 如果你的 npy 里保存了畸变参数，会自动尝试读取；
    # 如果没有，就默认无畸变。
    dist = None

    possible_keys = [
        "color_distortion",
        "color_distortion_coeffs",
        "dist_coeffs",
        "distortion",
        "D"
    ]

    for k in possible_keys:
        if k in config:
            dist = np.asarray(config[k], dtype=np.float64).reshape(-1, 1)
            break

    if dist is None:
        dist = np.zeros((5, 1), dtype=np.float64)

    return K, dist


def make_chessboard_object_points(pattern_size, square_size_m):
    """
    生成棋盘格世界坐标点，位于标定板坐标系 z=0 平面。
    pattern_size = (cols, rows)
    """
    cols, rows = pattern_size

    objp = np.zeros((rows * cols, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size_m

    return objp


def detect_target_pose(image_path, K, dist, objp, pattern_size):
    """
    对单张图片检测棋盘格，并用 solvePnP 计算 camera_T_target.

    OpenCV solvePnP 返回:
        rvec, tvec: target coordinate -> camera coordinate
        即 cam_T_target
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"[WARN] Failed to read image: {image_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_NORMALIZE_IMAGE
        + cv2.CALIB_CB_FAST_CHECK
    )

    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)

    if not found:
        print(f"[WARN] Chessboard not found: {os.path.basename(image_path)}")
        return None

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        50,
        1e-6
    )

    corners_subpix = cv2.cornerSubPix(
        gray,
        corners,
        winSize=(11, 11),
        zeroZone=(-1, -1),
        criteria=criteria
    )

    ok, rvec, tvec = cv2.solvePnP(
        objp,
        corners_subpix,
        K,
        dist,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not ok:
        print(f"[WARN] solvePnP failed: {os.path.basename(image_path)}")
        return None

    R_target2cam, _ = cv2.Rodrigues(rvec)

    cam_T_target = np.eye(4, dtype=np.float64)
    cam_T_target[:3, :3] = R_target2cam
    cam_T_target[:3, 3] = tvec.reshape(3)

    return cam_T_target


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

    # ------------------------------------------------------------
    # Load camera intrinsic
    # ------------------------------------------------------------
    K, dist = load_camera_intrinsic(camera_config_path)

    print("\nCamera intrinsic K:")
    print(K)
    print("\nDistortion coeffs:")
    print(dist.reshape(-1))

    # ------------------------------------------------------------
    # Load image/endpose pairs from test_data/N.png + N.json
    # ------------------------------------------------------------
    calibration_pairs = load_calibration_pairs(data_dir)

    print(f"\nLoaded calibration pairs: {len(calibration_pairs)}")

    # ------------------------------------------------------------
    # Detect target pose in camera frame
    # ------------------------------------------------------------
    objp = make_chessboard_object_points(PATTERN_SIZE, SQUARE_SIZE_M)

    R_gripper2base = []
    t_gripper2base = []

    R_target2cam = []
    t_target2cam = []

    used_pairs = []

    for pair in calibration_pairs:
        image_path = pair["image_path"]
        base_T_ee = pair["base_T_ee"]
        image_index = pair["index"]

        cam_T_target = detect_target_pose(
            image_path=image_path,
            K=K,
            dist=dist,
            objp=objp,
            pattern_size=PATTERN_SIZE
        )

        if cam_T_target is None:
            continue

        # OpenCV calibrateHandEye 输入:
        # R_gripper2base: ee -> base, 即 base_T_ee
        # R_target2cam: target -> camera, 即 cam_T_target
        R_gripper2base.append(base_T_ee[:3, :3])
        t_gripper2base.append(base_T_ee[:3, 3].reshape(3, 1))

        R_target2cam.append(cam_T_target[:3, :3])
        t_target2cam.append(cam_T_target[:3, 3].reshape(3, 1))

        used_pairs.append(image_index)

        print(f"[OK] Use pair: image {image_index}.png")

    if len(used_pairs) < 4:
        raise RuntimeError(
            f"Valid calibration pairs are too few: {len(used_pairs)}. "
            f"At least 4 valid views are recommended."
        )

    print(f"\nValid pairs used: {len(used_pairs)}")
    print("Used image indices:", used_pairs)

    # ------------------------------------------------------------
    # Hand-eye calibration
    # ------------------------------------------------------------
    # 返回:
    # R_cam2gripper, t_cam2gripper
    # 即 ee_T_cam，也就是用户要求的 ecT
    R_cam2ee, t_cam2ee = cv2.calibrateHandEye(
        R_gripper2base=R_gripper2base,
        t_gripper2base=t_gripper2base,
        R_target2cam=R_target2cam,
        t_target2cam=t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    ecT = np.eye(4, dtype=np.float64)
    ecT[:3, :3] = R_cam2ee
    ecT[:3, 3] = t_cam2ee.reshape(3)

    print_matrix("ecT = end_effector_T_camera", ecT)

    # ------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------
    save_matrix_csv(output_ecT_csv, ecT)
    np.save(output_ecT_npy, ecT)

    with open(output_used_pairs_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_index", "image_name"])
        for i in used_pairs:
            writer.writerow([i, f"{i}.png"])

    print("\nSaved:")
    print(output_ecT_csv)
    print(output_ecT_npy)
    print(output_used_pairs_csv)

    print("\nDone.")


if __name__ == "__main__":
    main()

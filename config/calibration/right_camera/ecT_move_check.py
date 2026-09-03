import json
from pathlib import Path
import sys
sys.path.append('/home/smmg/AAM')
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from Piper.piper_ctrl import connect_piper
from Piper.endpose_reachability_safe import reachability_test

# ============================================================
# User settings
# ============================================================

DATA_DIR = Path('/home/smmg/AAM/config/calibration/right_camera')
IMAGE_PATH = DATA_DIR / 'chessboard.png'
CAMERA_CONFIG_PATH = DATA_DIR / 'camera_config.npy'
ECT_PATH = DATA_DIR / 'ecT_20260727.npy'

PATTERN_SIZE = ( 10 , 7 )       # chessboard inner corners: (cols, rows)
SQUARE_SIZE_M = 0.015     # meter
TARGET_CORNER = ( 4,4  )      # 1-based: (col, row), e.g. (2,2)
Z_OFFSET_M = 0.15         # flange target z = corner z + 150 mm

ARM_NAME = 'r_piper'
FROM_EULER = 'xyz'
OUTPUT_JSON = DATA_DIR / 'ecT_move_check_endpose.json'
OUTPUT_VIS = DATA_DIR / 'ecT_move_check_vis.png'


# ============================================================
# Utilities
# ============================================================

def load_camera_intrinsic(path):
    config = np.load(path, allow_pickle=True).item()
    intr = config['color_intrinsic']
    K = np.array([
        [intr['fx'], 0.0, intr['ppx']],
        [0.0, intr['fy'], intr['ppy']],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    dist = np.zeros((5, 1), dtype=np.float64)
    return K, dist


def make_chessboard_object_points(pattern_size, square_size_m):
    cols, rows = pattern_size
    objp = np.zeros((rows * cols, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size_m
    return objp


def detect_chessboard_pose(image_path, K, dist):
    img = cv2.imread(str(image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    flags = cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY
    found, corners = cv2.findChessboardCornersSB(gray, PATTERN_SIZE, flags)

    objp = make_chessboard_object_points(PATTERN_SIZE, SQUARE_SIZE_M)
    ok, rvec, tvec = cv2.solvePnP(objp, corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)

    cam_R_board, _ = cv2.Rodrigues(rvec)
    cam_T_board = np.eye(4, dtype=np.float64)
    cam_T_board[:3, :3] = cam_R_board
    cam_T_board[:3, 3] = tvec.reshape(3)

    return img, corners.reshape(-1, 2), cam_T_board, objp


def get_flange_T_base(arm_name):
    piper = connect_piper(arm_name, with_gripper=False)
    try:
        pose = piper.get_endpose()
    finally:
        piper.disconnect()

    x, y, z, roll, pitch, yaw = pose
    xyz_m = np.array([x,y,z]) / 1000
    rpy_rad = np.deg2rad([roll, pitch, yaw])
    base_T_flange = np.eye(4, dtype=np.float64)
    base_T_flange[:3, :3] = R.from_euler(FROM_EULER, rpy_rad, degrees=False).as_matrix()
    base_T_flange[:3, 3] = xyz_m
    return base_T_flange, np.array([x, y, z, roll, pitch, yaw], dtype=np.float64)


def transform_point(T, p):
    ph = np.r_[p, 1.0]
    return (T @ ph)[:3]


def save_result(path, endpose_m_rad, endpose_mm_deg, corner_base_m, corner_pixel, ik_result):
    data = {
        'target_corner_1based_col_row': list(TARGET_CORNER),
        'corner_base_m': corner_base_m.tolist(),
        'corner_pixel_uv': corner_pixel.tolist(),
        'flange_endpose_m_rad': endpose_m_rad.tolist(),
        'flange_endpose_mm_deg': endpose_mm_deg.tolist(),
        'joint degrees' : ik_result['joint_degrees']
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ============================================================
# Main
# ============================================================

def main():
    K, dist = load_camera_intrinsic(CAMERA_CONFIG_PATH)
    ecT = np.load(ECT_PATH)                 # flange_T_camera

    img, corners, cam_T_board, objp = detect_chessboard_pose(IMAGE_PATH, K, dist)
    base_T_flange, current_flange_pose = get_flange_T_base(ARM_NAME)
    base_T_camera = base_T_flange @ ecT

    col_1based, row_1based = TARGET_CORNER
    col = col_1based - 1
    row = row_1based - 1
    cols, _ = PATTERN_SIZE
    corner_idx = row * cols + col

    corner_board = objp[corner_idx].astype(np.float64)
    corner_cam = transform_point(cam_T_board, corner_board)
    corner_base = transform_point(base_T_camera, corner_cam)

    current_yaw = current_flange_pose[5]
    target_rpy = np.array([np.pi, 0.0, current_yaw], dtype=np.float64)   # flange z-axis vertical downward
    target_xyz = corner_base + np.array([0.0, 0.0, Z_OFFSET_M])

    endpose_m_rad = np.r_[target_xyz, target_rpy]
    endpose_mm_deg = np.r_[target_xyz * 1000.0, np.degrees(target_rpy)]
    ik_result = reachability_test(endpose_mm_deg)

    vis = img.copy()
    u, v = corners[corner_idx]
    cv2.circle(vis, (int(u), int(v)), 8, (0, 0, 255), -1)
    cv2.putText(vis, f'corner {TARGET_CORNER}', (int(u) + 10, int(v) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imwrite(str(OUTPUT_VIS), vis)

    save_result(OUTPUT_JSON, endpose_m_rad, endpose_mm_deg, corner_base, np.array([u, v]), ik_result)

    print('\nSelected chessboard corner:', TARGET_CORNER)
    print('corner pixel [u, v]:', np.array([u, v]))
    print('corner base [m]:', corner_base)
    print('\nTarget flange endpose [m, rad]:')
    print(endpose_m_rad)
    print('\nTarget flange endpose [mm, deg]:')
    print(endpose_mm_deg)
    print('\nIK reachable:', ik_result['reachable'])
    print('\nJoint degrees:')
    print(ik_result['joint_degrees'])
    print('IK pos_err_mm:', ik_result['pos_err_mm'])
    print('IK rot_err_deg:', ik_result['rot_err_deg'])



    print('\nSaved:', OUTPUT_JSON)
    print('Saved:', OUTPUT_VIS)


if __name__ == '__main__':
    main()

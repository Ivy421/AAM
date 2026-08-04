#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clear Mark1 -> Piper arm-base calibration script using a checkerboard.

This version keeps the coordinate-transform math as direct as possible.
It replaces the previous ArUco marker detection with checkerboard corner
extraction + one PnP pose per image.

Coordinate notation
-------------------
T_X_Y maps a homogeneous point from frame Y to frame X:

    p_X = T_X_Y @ p_Y

Frames used here:

    O : odom/world frame from Mark1 odometry
    B : Mark1 base frame
    C : camera optical frame
    A : Piper arm base frame
    H : checkerboard frame

Known for each image i:

    T_O_B_i : from odom.csv, Mark1 base pose in odom
    T_C_H_i : from checkerboard solvePnP, checkerboard pose in camera optical frame

Unknown during optimization:

    T_B_C : camera pose in Mark1 base frame, i.e. ^mark1_base T_camera
    T_O_H : fixed checkerboard pose in odom frame

Core constraint for every valid image i:

    T_O_H = T_O_B_i @ T_B_C @ T_C_H_i

The optimizer searches for T_B_C and T_O_H such that this equation is as
consistent as possible over all images.

After T_B_C is estimated, use the known Piper arm-base -> camera transform:

    T_A_C = ^arm_base T_camera

Then:

    T_B_A = T_B_C @ inv(T_A_C)

which is the desired Mark1 base -> Piper arm base transform.
"""

from __future__ import annotations

import glob
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R


# =============================================================================
# 0. HARD PARAMETERS
# =============================================================================

DATA_DIR = Path(r"E:/HKUSTGZ/AAM/config/calibration/mark1/data1")
ODOM_CSV = DATA_DIR / "odom.csv"
OUT_DIR = DATA_DIR / "calibration_result"

# Checkerboard parameters.
# OpenCV pattern size means number of INNER CORNERS: (columns, rows).
# For your checkerboard: 9 x 6 inner corners, square side length = 0.026 m.
CHECKERBOARD_SIZE = (9, 6)
SQUARE_SIZE_M = 0.026

CAMERA_NPY = Path(r"E:/HKUSTGZ/AAM/config/calibration/right_camera/camera_config.npy")
ARM_BASE_TO_CAMERA_NPY = Path(r"E:\HKUSTGZ\AAM\config\calibration\mark1\bcT.npy")

YAW_UNIT = "rad"

MAX_REPROJ_ERROR_PX = 2.0
ROTATION_RESIDUAL_WEIGHT = 0.05
MAX_OPTIMIZATION_EVALS = 3000


# =============================================================================
# 1. Basic SE(3) utilities
# =============================================================================

def T_inv(T: np.ndarray) -> np.ndarray:
    """Inverse of a rigid 4x4 transform."""
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ T[:3, 3]
    return out


def vec6_to_T(v: np.ndarray) -> np.ndarray:
    """Convert [rx, ry, rz, tx, ty, tz] to a 4x4 transform."""
    T = np.eye(4)
    T[:3, :3] = R.from_rotvec(v[:3]).as_matrix()
    T[:3, 3] = v[3:6]
    return T


def T_to_vec6(T: np.ndarray) -> np.ndarray:
    """Convert a 4x4 transform to [rx, ry, rz, tx, ty, tz]."""
    v = np.zeros(6)
    v[:3] = R.from_matrix(T[:3, :3]).as_rotvec()
    v[3:6] = T[:3, 3]
    return v


def se3_residual(T_error: np.ndarray, rot_weight: float = ROTATION_RESIDUAL_WEIGHT) -> np.ndarray:
    """
    Convert a pose error transform to a 6D residual.

    T_error should be close to identity.
    Translation residual is in meters.
    Rotation residual is rotvec in radians, scaled by rot_weight.
    """
    trans = T_error[:3, 3]
    rotvec = R.from_matrix(T_error[:3, :3]).as_rotvec()
    return np.r_[trans, rot_weight * rotvec]


def pose2d_to_T_O_B(x: float, y: float, yaw: float) -> np.ndarray:
    """
    Convert Mark1 planar odom pose into T_O_B = ^odom T_mark1_base.

    Assumption: ROS standard base frame:
        x forward, y left, z up, yaw positive counter-clockwise around z.
    """
    c = math.cos(yaw)
    s = math.sin(yaw)
    T = np.eye(4)
    T[:3, :3] = np.array(
        [[c, -s, 0.0],
         [s,  c, 0.0],
         [0.0, 0.0, 1.0]],
        dtype=float,
    )
    T[:3, 3] = [x, y, 0.0]
    return T


def mean_transform(T_list: List[np.ndarray]) -> np.ndarray:
    """Simple average transform, used only to initialize fixed checkerboard pose T_O_H."""
    T_mean = np.eye(4)
    T_mean[:3, 3] = np.mean([T[:3, 3] for T in T_list], axis=0)
    T_mean[:3, :3] = R.from_matrix(np.stack([T[:3, :3] for T in T_list])).mean().as_matrix()
    return T_mean


def print_transform(name: str, T: np.ndarray) -> None:
    xyz = T[:3, 3]
    rpy = R.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True)
    print(f"\n{name}")
    print(np.array2string(T, precision=8, suppress_small=False))
    print(f"translation xyz [m]        = {xyz}")
    print(f"roll pitch yaw [deg, xyz] = {rpy}")


def save_transform(out_dir: Path, name: str, T: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{name}.npy", T)
    np.savetxt(out_dir / f"{name}.txt", T, fmt="%.10f")


# =============================================================================
# 2. Read camera intrinsics and Mark1 odom
# =============================================================================

def load_camera_intrinsics(camera_npy: Path) -> Tuple[np.ndarray, np.ndarray]:
    config = np.load(str(camera_npy), allow_pickle=True).item()
    color_intrinsic = config["color_intrinsic"]

    fx = color_intrinsic["fx"]
    fy = color_intrinsic["fy"]
    cx = color_intrinsic["ppx"]
    cy = color_intrinsic["ppy"]

    K = np.array(
        [[fx, 0.0, cx],
         [0.0, fy, cy],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    D = np.asarray(color_intrinsic["coeffs"], dtype=np.float64)
    return K, D


def read_odom_csv(odom_csv: Path) -> Dict[int, np.ndarray]:
    """
    Read odom.csv and return:

        sample_index -> T_O_B

    sample_id like sample_000011 will be matched to 11_Color.png.
    """
    df = pd.read_csv(str(odom_csv))
    df.columns = [c.strip() for c in df.columns]

    if not {"x", "y", "yaw"}.issubset(df.columns):
        raise ValueError(f"odom.csv must contain x, y, yaw. Existing columns: {list(df.columns)}")

    if "sample_id" in df.columns:
        sample_indices = []
        for i, sid in enumerate(df["sample_id"].astype(str)):
            match = re.search(r"(\d+)", sid)
            sample_indices.append(int(match.group(1)) if match else i + 1)
    else:
        sample_indices = list(range(1, len(df) + 1))

    yaw_scale = math.pi / 180.0  if YAW_UNIT.lower().startswith("deg") else 1

    T_map: Dict[int, np.ndarray] = {}
    for sample_idx, (_, row) in zip(sample_indices, df.iterrows()):
        x = float(row["x"])
        y = float(row["y"])
        yaw = float(row["yaw"]) * yaw_scale
        T_map[int(sample_idx)] = pose2d_to_T_O_B(x, y, yaw)

    return T_map


# =============================================================================
# 3. Detect checkerboard and compute T_C_H by PnP
# =============================================================================

@dataclass
class Observation:
    sample_idx: int
    image_name: str
    T_O_B: np.ndarray      # known: ^odom T_mark1_base from odom.csv
    T_C_H: np.ndarray      # known: ^camera T_checkerboard from PnP
    reproj_error_px: float
    detected_corners: int


def image_index(path: str) -> Optional[int]:
    """Extract the number in names like 11_Color.png."""
    name = Path(path).name
    match = re.search(r"(\d+)(?=[^\d]*_?Color\.(png|jpg|jpeg|bmp)$)", name, re.IGNORECASE)
    if not match:
        match = re.search(r"(\d+)", name)
    return int(match.group(1)) if match else None


def checkerboard_object_points() -> np.ndarray:
    """
    Build checkerboard object points in H frame.

    For CHECKERBOARD_SIZE = (9, 6), OpenCV expects 9 columns and 6 rows of
    inner corners. The checkerboard plane is z=0.
    """
    cols, rows = CHECKERBOARD_SIZE
    obj = np.zeros((rows * cols, 3), dtype=np.float64)
    grid_x, grid_y = np.meshgrid(np.arange(cols), np.arange(rows))
    obj[:, 0] = grid_x.reshape(-1) * SQUARE_SIZE_M
    obj[:, 1] = grid_y.reshape(-1) * SQUARE_SIZE_M
    obj[:, 2] = 0.0
    return obj


def find_checkerboard_corners(gray: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
    """
    Detect checkerboard inner corners.

    Prefer findChessboardCornersSB because it is usually more stable.
    Fallback to classic findChessboardCorners + cornerSubPix.
    """
    flags_sb = 0
    if hasattr(cv2, "CALIB_CB_EXHAUSTIVE"):
        flags_sb |= cv2.CALIB_CB_EXHAUSTIVE
    if hasattr(cv2, "CALIB_CB_ACCURACY"):
        flags_sb |= cv2.CALIB_CB_ACCURACY

    if hasattr(cv2, "findChessboardCornersSB"):
        ok, corners = cv2.findChessboardCornersSB(gray, CHECKERBOARD_SIZE, flags=flags_sb)
        if ok:
            return True, corners.astype(np.float64)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_SIZE, flags=flags)
    if not ok:
        return False, None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    corners = cv2.cornerSubPix(gray, corners, winSize=(11, 11), zeroZone=(-1, -1), criteria=criteria)
    return True, corners.astype(np.float64)


def solve_checkerboard_pnp(corners_2d: np.ndarray, K: np.ndarray, D: np.ndarray) -> Tuple[bool, np.ndarray, float]:
    """
    Checkerboard PnP.

    Input:
        2D checkerboard inner corners in image pixels.

    Output:
        T_C_H = ^camera T_checkerboard.
    """
    object_points_H = checkerboard_object_points()
    image_points = corners_2d.reshape(-1, 2).astype(np.float64)

    if image_points.shape[0] != object_points_H.shape[0]:
        return False, np.eye(4), float("inf")

    ok, rvec, tvec = cv2.solvePnP(
        object_points_H,
        image_points,
        K,
        D,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return False, np.eye(4), float("inf")

    projected, _ = cv2.projectPoints(object_points_H, rvec, tvec, K, D)
    projected = projected.reshape(-1, 2)
    reproj_error = float(np.sqrt(np.mean(np.sum((projected - image_points) ** 2, axis=1))))

    T_C_H = np.eye(4)
    T_C_H[:3, :3] = cv2.Rodrigues(rvec)[0]
    T_C_H[:3, 3] = tvec.reshape(3)
    return True, T_C_H, reproj_error


def collect_observations(K: np.ndarray, D: np.ndarray) -> Tuple[List[Observation], List[str]]:
    """
    Build all valid checkerboard observations.

    Each valid image gives one equation:

        T_O_H = T_O_B_i @ T_B_C @ T_C_H_i
    """
    T_O_B_by_sample = read_odom_csv(ODOM_CSV)

    image_paths = sorted(
        glob.glob(str(DATA_DIR / "*_Color.png")),
        key=lambda p: (image_index(p) or 10**9, p),
    )
    if not image_paths:
        raise FileNotFoundError(f"No *_Color.png images found in {DATA_DIR}")

    observations: List[Observation] = []
    rejected: List[str] = []

    for image_path in image_paths:
        sample_idx = image_index(image_path)
        image_name = Path(image_path).name

        if sample_idx is None:
            rejected.append(f"{image_name}: cannot parse image index")
            continue
        if sample_idx not in T_O_B_by_sample:
            rejected.append(f"{image_name}: no matching odom sample {sample_idx}")
            continue

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            rejected.append(f"{image_name}: cannot read image")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ok, corners = find_checkerboard_corners(gray)
        if not ok or corners is None:
            rejected.append(f"{image_name}: checkerboard not detected")
            continue

        ok, T_C_H, reproj_error = solve_checkerboard_pnp(corners, K, D)
        if not ok or not np.isfinite(reproj_error):
            rejected.append(f"{image_name}: checkerboard PnP failed")
            continue
        if reproj_error > MAX_REPROJ_ERROR_PX:
            rejected.append(f"{image_name}: checkerboard reproj={reproj_error:.2f}px")
            continue

        observations.append(
            Observation(
                sample_idx=sample_idx,
                image_name=image_name,
                T_O_B=T_O_B_by_sample[sample_idx],
                T_C_H=T_C_H,
                reproj_error_px=reproj_error,
                detected_corners=int(corners.reshape(-1, 2).shape[0]),
            )
        )

    return observations, rejected


# =============================================================================
# 4. Initial guess for T_B_C
# =============================================================================

def initial_T_B_C_from_checkerboard(observations: List[Observation]) -> np.ndarray:
    """
    Initial guess for T_B_C using OpenCV hand-eye calibration.

    This is NOT the final answer. It is only used to initialize least_squares.

    In OpenCV naming:
        gripper = Mark1 base B
        robot base = odom O
        camera = camera C
        target = checkerboard H

    Output is T_B_C = ^mark1_base T_camera.
    """
    if len(observations) < 4:
        print("[WARN] Less than 4 valid checkerboard observations. Use identity as initial T_B_C.")
        return np.eye(4)

    R_gripper2base = []   # ^odom R_mark1_base = R_O_B
    t_gripper2base = []   # ^odom t_mark1_base = t_O_B
    R_target2cam = []     # ^camera R_checkerboard = R_C_H
    t_target2cam = []     # ^camera t_checkerboard = t_C_H

    for obs in observations:
        R_gripper2base.append(obs.T_O_B[:3, :3])
        t_gripper2base.append(obs.T_O_B[:3, 3])
        R_target2cam.append(obs.T_C_H[:3, :3])
        t_target2cam.append(obs.T_C_H[:3, 3])

    R_C_to_B, t_C_to_B = cv2.calibrateHandEye(
        R_gripper2base,
        t_gripper2base,
        R_target2cam,
        t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI,
    )

    T_B_C = np.eye(4)
    T_B_C[:3, :3] = np.asarray(R_C_to_B, dtype=float)
    T_B_C[:3, 3] = np.asarray(t_C_to_B, dtype=float).reshape(3)

    print(f"[INFO] Initial T_B_C from checkerboard hand-eye, observations={len(observations)}")
    return T_B_C


# =============================================================================
# 5. Least-squares optimization
# =============================================================================

def pack_variables(T_B_C: np.ndarray, T_O_H: np.ndarray) -> np.ndarray:
    """Optimization vector: x = [vec(T_B_C), vec(T_O_H)]."""
    return np.r_[T_to_vec6(T_B_C), T_to_vec6(T_O_H)]


def unpack_variables(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    T_B_C = vec6_to_T(x[:6])
    T_O_H = vec6_to_T(x[6:12])
    return T_B_C, T_O_H


def build_initial_variables(observations: List[Observation]) -> np.ndarray:
    """
    Initialize T_B_C and checkerboard pose T_O_H.

    Given an initial T_B_C, each observation predicts:

        T_O_H_pred = T_O_B @ T_B_C @ T_C_H

    Average all predictions to initialize T_O_H.
    """
    T_B_C_init = initial_T_B_C_from_checkerboard(observations)
    predictions = [obs.T_O_B @ T_B_C_init @ obs.T_C_H for obs in observations]
    T_O_H_init = mean_transform(predictions)
    return pack_variables(T_B_C_init, T_O_H_init)


def calibration_residuals(x: np.ndarray, observations: List[Observation]) -> np.ndarray:
    """
    Residual for the core equation:

        T_O_H = T_O_B_i @ T_B_C @ T_C_H_i

    For each observation, compute:

        T_error = inv(T_O_H) @ T_O_B_i @ T_B_C @ T_C_H_i

    Perfect calibration gives T_error = I.
    """
    T_B_C, T_O_H = unpack_variables(x)

    residual_blocks = []
    for obs in observations:
        T_O_H_pred = obs.T_O_B @ T_B_C @ obs.T_C_H
        T_error = T_inv(T_O_H) @ T_O_H_pred
        residual_blocks.append(se3_residual(T_error))

    return np.concatenate(residual_blocks)


def estimate_T_B_C(observations: List[Observation]) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Estimate T_B_C = ^mark1_base T_camera."""
    if len(observations) == 0:
        raise RuntimeError("No valid checkerboard observations. Check board size, square size, images, and odom matching.")

    x0 = build_initial_variables(observations)

    result = least_squares(
        calibration_residuals,
        x0,
        args=(observations,),
        max_nfev=MAX_OPTIMIZATION_EVALS,
    )
    if not result.success:
        print(f"[WARN] least_squares did not fully converge: {result.message}")

    T_B_C, T_O_H = unpack_variables(result.x)
    residual_table = compute_residual_table(observations, T_B_C, T_O_H)
    return T_B_C, T_O_H, residual_table


def compute_residual_table(
    observations: Iterable[Observation],
    T_B_C: np.ndarray,
    T_O_H: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for obs in observations:
        T_O_H_pred = obs.T_O_B @ T_B_C @ obs.T_C_H
        T_error = T_inv(T_O_H) @ T_O_H_pred

        trans_error_m = float(np.linalg.norm(T_error[:3, 3]))
        rot_error_deg = float(np.linalg.norm(R.from_matrix(T_error[:3, :3]).as_rotvec()) * 180.0 / math.pi)

        rows.append(
            {
                "sample_idx": obs.sample_idx,
                "image": obs.image_name,
                "detected_corners": obs.detected_corners,
                "reproj_error_px": obs.reproj_error_px,
                "se3_trans_residual_m": trans_error_m,
                "se3_rot_residual_deg": rot_error_deg,
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# 6. Final chain: Mark1 base -> arm base
# =============================================================================

def compute_T_B_A(T_B_C: np.ndarray) -> np.ndarray:
    """
    Compute the final desired transform:

        T_B_A = T_B_C @ inv(T_A_C)

    where:
        T_B_C = ^mark1_base T_camera, estimated by this script
        T_A_C = ^arm_base T_camera, loaded from bcT.npy
    """
    T_A_C = np.load(str(ARM_BASE_TO_CAMERA_NPY))
    if T_A_C.shape != (4, 4):
        raise ValueError(f"bcT.npy must be 4x4, got shape {T_A_C.shape}")

    T_B_A = T_B_C @ T_inv(T_A_C)
    return T_B_A


# =============================================================================
# 7. Main pipeline
# =============================================================================

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[INFO] Mathematical chain:")
    print("       T_O_H = T_O_B @ T_B_C @ T_C_H")
    print("       T_B_A = T_B_C @ inv(T_A_C)")
    print(f"[INFO] Checkerboard inner corners: {CHECKERBOARD_SIZE}")
    print(f"[INFO] Checkerboard square size [m]: {SQUARE_SIZE_M}")

    K, D = load_camera_intrinsics(CAMERA_NPY)
    print("\n[INFO] Camera intrinsics K:\n", K)
    print("[INFO] Distortion D:", D)

    observations, rejected = collect_observations(K, D)
    print(f"\n[INFO] valid checkerboard observations: {len(observations)}")
    print(f"[INFO] valid samples: {len(set(o.sample_idx for o in observations))}")

    with open(OUT_DIR / "rejected_observations.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(rejected))
        f.write("\n")

    T_B_C, T_O_H, residual_table = estimate_T_B_C(observations)

    T_C_B = T_inv(T_B_C)
    T_B_A = compute_T_B_A(T_B_C)
    T_A_B = T_inv(T_B_A)

    save_transform(OUT_DIR, "T_mark1_camera", T_B_C)
    save_transform(OUT_DIR, "T_camera_mark1", T_C_B)
    save_transform(OUT_DIR, "T_odom_checkerboard", T_O_H)
    save_transform(OUT_DIR, "T_mark1_armbase", T_B_A)
    save_transform(OUT_DIR, "T_armbase_mark1", T_A_B)

    residual_table.to_csv(OUT_DIR / "final_observation_residuals.csv", index=False)

    print_transform("T_mark1_camera  (^mark1_base T_camera)", T_B_C)
    print_transform("T_odom_checkerboard  (^odom T_checkerboard)", T_O_H)
    print_transform("T_mark1_armbase  (^mark1_base T_arm_base)", T_B_A)
    print_transform("T_armbase_mark1  (^arm_base T_mark1_base)", T_A_B)

    print("\n[INFO] residual summary:")
    print(residual_table[["se3_trans_residual_m", "se3_rot_residual_deg", "reproj_error_px"]].describe())
    print(f"\n[INFO] outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()

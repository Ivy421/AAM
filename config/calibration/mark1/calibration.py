#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clear Mark1 -> Piper arm-base calibration script.

This version intentionally keeps the flow simple and close to the math.
It removes the repeated robust/outlier optimization loop and keeps only the
necessary coordinate transforms.

Coordinate notation
-------------------
T_X_Y maps a homogeneous point from frame Y to frame X:

    p_X = T_X_Y @ p_Y

Frames used here:

    O : odom/world frame from Mark1 odometry
    B : Mark1 base frame
    C : camera optical frame
    A : Piper arm base frame
    G : one ArUco marker/tag frame

Known for each image i:

    T_O_B_i : from odom.csv, Mark1 base pose in odom
    T_C_G_i : from solvePnP, marker pose in camera optical frame

Unknown during optimization:

    T_B_C   : camera pose in Mark1 base frame, i.e. ^mark1_base T_camera
    T_O_G_k : fixed marker k pose in odom frame

Core constraint for every observation of marker k in image i:

    T_O_G_k = T_O_B_i @ T_B_C @ T_C_G_i

The optimizer searches for T_B_C and all T_O_G_k such that this equation is
as consistent as possible over all images.

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
from collections import Counter
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
#    These values are kept from your original script.
# =============================================================================

DATA_DIR = Path(r"E:/HKUSTGZ/AAM/config/calibration/mark1/data1")
ODOM_CSV = DATA_DIR / "odom.csv"
OUT_DIR = DATA_DIR / "calibration_result"

TAG_SIZE_M = 0.0242
ARUCO_DICTIONARY = "DICT_4X4_1000"

CAMERA_NPY = Path(r"E:/HKUSTGZ/AAM/config/calibration/right_camera/camera_config.npy")
ARM_BASE_TO_CAMERA_NPY = Path(r"E:\HKUSTGZ\AAM\config\calibration\mark1\bcT.npy")

YAW_UNIT = "rad"

EDGE_MARGIN_PX = 8.0
MIN_MARKER_AREA_PX2 = 400.0
MAX_REPROJ_ERROR_PX = 3.0
MIN_OBS_PER_TAG = 2

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
    """Simple average transform, used only to initialize each fixed tag pose T_O_G."""
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
    Read odom.csv and return a dictionary:

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

    yaw_scale = math.pi / 180.0 if YAW_UNIT.lower().startswith("deg") else 1.0

    T_map: Dict[int, np.ndarray] = {}
    for sample_idx, (_, row) in zip(sample_indices, df.iterrows()):
        x = float(row["x"])
        y = float(row["y"])
        yaw = float(row["yaw"]) * yaw_scale
        T_map[int(sample_idx)] = pose2d_to_T_O_B(x, y, yaw)

    return T_map


# =============================================================================
# 3. Detect ArUco markers and compute T_C_G by PnP
# =============================================================================

@dataclass
class Observation:
    sample_idx: int
    image_name: str
    tag_id: int
    T_O_B: np.ndarray      # known: ^odom T_mark1_base from odom.csv
    T_C_G: np.ndarray      # known: ^camera T_tag from PnP
    reproj_error_px: float
    marker_area_px2: float


def image_index(path: str) -> Optional[int]:
    """Extract the number in names like 11_Color.png."""
    name = Path(path).name
    match = re.search(r"(\d+)(?=[^\d]*_?Color\.(png|jpg|jpeg|bmp)$)", name, re.IGNORECASE)
    if not match:
        match = re.search(r"(\d+)", name)
    return int(match.group(1)) if match else None


def get_aruco_dictionary(name: str):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("cv2.aruco not found. Please install opencv-contrib-python.")
    aruco = cv2.aruco
    if not hasattr(aruco, name):
        raise ValueError(f"Unknown ArUco dictionary: {name}")
    return aruco.getPredefinedDictionary(getattr(aruco, name))


def detect_markers(gray: np.ndarray):
    aruco = cv2.aruco
    dictionary = get_aruco_dictionary(ARUCO_DICTIONARY)
    params = aruco.DetectorParameters()
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.minMarkerPerimeterRate = 0.02
    params.maxMarkerPerimeterRate = 4.0

    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(dictionary, params)
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = aruco.detectMarkers(gray, dictionary, parameters=params)

    return corners, ids


def polygon_area(points_2d: np.ndarray) -> float:
    pts = points_2d.reshape(-1, 2)
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def solve_marker_pnp(corners_2d: np.ndarray, K: np.ndarray, D: np.ndarray) -> Tuple[bool, np.ndarray, float]:
    """
    Single marker PnP.

    Input:
        2D marker corners in image pixels.

    Output:
        T_C_G = ^camera T_tag.
    """
    s = TAG_SIZE_M

    # OpenCV ArUco corner order:
    # top-left, top-right, bottom-right, bottom-left.
    tag_points_G = np.array(
        [[-s / 2.0,  s / 2.0, 0.0],
         [ s / 2.0,  s / 2.0, 0.0],
         [ s / 2.0, -s / 2.0, 0.0],
         [-s / 2.0, -s / 2.0, 0.0]],
        dtype=np.float64,
    )
    image_points = corners_2d.reshape(4, 2).astype(np.float64)

    flag = cv2.SOLVEPNP_IPPE_SQUARE if hasattr(cv2, "SOLVEPNP_IPPE_SQUARE") else cv2.SOLVEPNP_ITERATIVE
    ok, rvec, tvec = cv2.solvePnP(tag_points_G, image_points, K, D, flags=flag)
    if not ok:
        return False, np.eye(4), float("inf")

    projected, _ = cv2.projectPoints(tag_points_G, rvec, tvec, K, D)
    projected = projected.reshape(4, 2)
    reproj_error = float(np.sqrt(np.mean(np.sum((projected - image_points) ** 2, axis=1))))

    T_C_G = np.eye(4)
    T_C_G[:3, :3] = cv2.Rodrigues(rvec)[0]
    T_C_G[:3, 3] = tvec.reshape(3)
    return True, T_C_G, reproj_error


def collect_observations(K: np.ndarray, D: np.ndarray) -> Tuple[List[Observation], List[str]]:
    """
    Build all observations.

    Each valid observation gives one equation:

        T_O_G_k = T_O_B_i @ T_B_C @ T_C_G_i
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

        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners_list, ids = detect_markers(gray)

        if ids is None or len(ids) == 0:
            rejected.append(f"{image_name}: no ArUco marker detected")
            continue

        for corners, tag_id_array in zip(corners_list, ids.reshape(-1)):
            tag_id = int(tag_id_array)
            pts = np.asarray(corners, dtype=float).reshape(4, 2)

            near_edge = (
                pts[:, 0].min() < EDGE_MARGIN_PX or
                pts[:, 0].max() > width - EDGE_MARGIN_PX or
                pts[:, 1].min() < EDGE_MARGIN_PX or
                pts[:, 1].max() > height - EDGE_MARGIN_PX
            )
            if near_edge:
                rejected.append(f"{image_name}: tag {tag_id} too close to edge")
                continue

            area = polygon_area(pts)
            if area < MIN_MARKER_AREA_PX2:
                rejected.append(f"{image_name}: tag {tag_id} too small, area={area:.1f}")
                continue

            ok, T_C_G, reproj_error = solve_marker_pnp(pts, K, D)
            if not ok or not np.isfinite(reproj_error):
                rejected.append(f"{image_name}: tag {tag_id} PnP failed")
                continue
            if reproj_error > MAX_REPROJ_ERROR_PX:
                rejected.append(f"{image_name}: tag {tag_id} reproj={reproj_error:.2f}px")
                continue

            observations.append(
                Observation(
                    sample_idx=sample_idx,
                    image_name=image_name,
                    tag_id=tag_id,
                    T_O_B=T_O_B_by_sample[sample_idx],
                    T_C_G=T_C_G,
                    reproj_error_px=reproj_error,
                    marker_area_px2=area,
                )
            )

    # Keep only markers observed at least MIN_OBS_PER_TAG times.
    counts = Counter(o.tag_id for o in observations)
    observations = [o for o in observations if counts[o.tag_id] >= MIN_OBS_PER_TAG]

    return observations, rejected


# =============================================================================
# 4. Initial guess for T_B_C
# =============================================================================

def initial_T_B_C_from_repeated_tag(observations: List[Observation]) -> np.ndarray:
    """
    Initial guess for T_B_C using OpenCV hand-eye calibration.

    This is NOT the final answer. It is only used to initialize least_squares.

    In OpenCV naming:
        gripper = Mark1 base B
        robot base = odom O
        camera = camera C
        target = tag G

    Output is T_B_C = ^mark1_base T_camera.
    """
    counts = Counter(o.tag_id for o in observations)
    if not counts:
        return np.eye(4)

    best_tag_id, n = counts.most_common(1)[0]
    tag_obs = [o for o in observations if o.tag_id == best_tag_id]
    if n < 4:
        print("[WARN] Less than 4 observations for the best tag. Use identity as initial T_B_C.")
        return np.eye(4)

    R_gripper2base = []   # ^odom R_mark1_base = R_O_B
    t_gripper2base = []   # ^odom t_mark1_base = t_O_B
    R_target2cam = []     # ^camera R_tag = R_C_G
    t_target2cam = []     # ^camera t_tag = t_C_G

    for obs in tag_obs:
        R_gripper2base.append(obs.T_O_B[:3, :3])
        t_gripper2base.append(obs.T_O_B[:3, 3])
        R_target2cam.append(obs.T_C_G[:3, :3])
        t_target2cam.append(obs.T_C_G[:3, 3])

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

    print(f"[INFO] Initial T_B_C from tag {best_tag_id}, observations={n}")
    return T_B_C


# =============================================================================
# 5. Least-squares optimization
# =============================================================================

def pack_variables(T_B_C: np.ndarray, T_O_G_by_tag: Dict[int, np.ndarray], tag_ids: List[int]) -> np.ndarray:
    """
    Optimization vector:

        x = [ vec(T_B_C), vec(T_O_G_tag1), vec(T_O_G_tag2), ... ]
    """
    blocks = [T_to_vec6(T_B_C)]
    for tag_id in tag_ids:
        blocks.append(T_to_vec6(T_O_G_by_tag[tag_id]))
    return np.concatenate(blocks)


def unpack_variables(x: np.ndarray, tag_ids: List[int]) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    T_B_C = vec6_to_T(x[:6])
    T_O_G_by_tag: Dict[int, np.ndarray] = {}

    offset = 6
    for tag_id in tag_ids:
        T_O_G_by_tag[tag_id] = vec6_to_T(x[offset:offset + 6])
        offset += 6

    return T_B_C, T_O_G_by_tag


def build_initial_variables(observations: List[Observation]) -> Tuple[np.ndarray, List[int]]:
    """
    Initialize T_B_C and every tag pose T_O_G.

    Given an initial T_B_C, each observation predicts:

        T_O_G_pred = T_O_B @ T_B_C @ T_C_G

    For each tag, average all predicted T_O_G_pred to initialize T_O_G.
    """
    tag_ids = sorted(set(o.tag_id for o in observations))
    T_B_C_init = initial_T_B_C_from_repeated_tag(observations)

    T_O_G_init: Dict[int, np.ndarray] = {}
    for tag_id in tag_ids:
        predictions = [o.T_O_B @ T_B_C_init @ o.T_C_G for o in observations if o.tag_id == tag_id]
        T_O_G_init[tag_id] = mean_transform(predictions)

    x0 = pack_variables(T_B_C_init, T_O_G_init, tag_ids)
    return x0, tag_ids


def calibration_residuals(x: np.ndarray, observations: List[Observation], tag_ids: List[int]) -> np.ndarray:
    """
    Residual for the core equation:

        T_O_G_k = T_O_B_i @ T_B_C @ T_C_G_i

    For each observation, compute:

        T_error = inv(T_O_G_k) @ T_O_B_i @ T_B_C @ T_C_G_i

    Perfect calibration gives T_error = I.
    """
    T_B_C, T_O_G_by_tag = unpack_variables(x, tag_ids)

    residual_blocks = []
    for obs in observations:
        T_O_G_pred = obs.T_O_B @ T_B_C @ obs.T_C_G
        T_error = T_inv(T_O_G_by_tag[obs.tag_id]) @ T_O_G_pred
        residual_blocks.append(se3_residual(T_error))

    return np.concatenate(residual_blocks)


def estimate_T_B_C(observations: List[Observation]) -> Tuple[np.ndarray, Dict[int, np.ndarray], pd.DataFrame]:
    """Estimate T_B_C = ^mark1_base T_camera."""
    if len(observations) == 0:
        raise RuntimeError("No valid observations. Check image detection, dictionary, tag size, and odom matching.")

    x0, tag_ids = build_initial_variables(observations)

    result = least_squares(
        calibration_residuals,
        x0,
        args=(observations, tag_ids),
        max_nfev=MAX_OPTIMIZATION_EVALS,
    )
    if not result.success:
        print(f"[WARN] least_squares did not fully converge: {result.message}")

    T_B_C, T_O_G_by_tag = unpack_variables(result.x, tag_ids)
    residual_table = compute_residual_table(observations, T_B_C, T_O_G_by_tag)
    return T_B_C, T_O_G_by_tag, residual_table


def compute_residual_table(
    observations: Iterable[Observation],
    T_B_C: np.ndarray,
    T_O_G_by_tag: Dict[int, np.ndarray],
) -> pd.DataFrame:
    rows = []
    for obs in observations:
        T_O_G_pred = obs.T_O_B @ T_B_C @ obs.T_C_G
        T_error = T_inv(T_O_G_by_tag[obs.tag_id]) @ T_O_G_pred

        trans_error_m = float(np.linalg.norm(T_error[:3, 3]))
        rot_error_deg = float(np.linalg.norm(R.from_matrix(T_error[:3, :3]).as_rotvec()) * 180.0 / math.pi)

        rows.append(
            {
                "sample_idx": obs.sample_idx,
                "image": obs.image_name,
                "tag_id": obs.tag_id,
                "reproj_error_px": obs.reproj_error_px,
                "marker_area_px2": obs.marker_area_px2,
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
    print("       T_O_G = T_O_B @ T_B_C @ T_C_G")
    print("       T_B_A = T_B_C @ inv(T_A_C)")

    K, D = load_camera_intrinsics(CAMERA_NPY)
    print("\n[INFO] Camera intrinsics K:\n", K)
    print("[INFO] Distortion D:", D)

    observations, rejected = collect_observations(K, D)
    print(f"\n[INFO] valid observations: {len(observations)}")
    print(f"[INFO] valid samples: {len(set(o.sample_idx for o in observations))}")
    print(f"[INFO] tag counts: {Counter(o.tag_id for o in observations)}")

    with open(OUT_DIR / "rejected_observations.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(rejected))
        f.write("\n")

    T_B_C, T_O_G_by_tag, residual_table = estimate_T_B_C(observations)

    T_C_B = T_inv(T_B_C)
    T_B_A = compute_T_B_A(T_B_C)
    T_A_B = T_inv(T_B_A)

    save_transform(OUT_DIR, "T_mark1_camera", T_B_C)
    save_transform(OUT_DIR, "T_camera_mark1", T_C_B)
    save_transform(OUT_DIR, "T_mark1_armbase", T_B_A)
    save_transform(OUT_DIR, "T_armbase_mark1", T_A_B)

    residual_table.to_csv(OUT_DIR / "final_observation_residuals.csv", index=False)

    # Save fixed tag poses in odom only as one compact npz file, not many separate files.
    np.savez(
        OUT_DIR / "T_odom_tags.npz",
        **{f"tag_{tag_id}": T for tag_id, T in T_O_G_by_tag.items()},
    )

    #print_transform("T_mark1_camera  (^mark1_base T_camera)", T_B_C)
    #print_transform("T_mark1_armbase  (^mark1_base T_arm_base)", T_B_A)
    #print_transform("T_armbase_mark1  (^arm_base T_mark1_base)", T_A_B)
#
    #print("\n[INFO] residual summary:")
    #print(residual_table[["se3_trans_residual_m", "se3_rot_residual_deg", "reproj_error_px"]].describe())
    #print(f"\n[INFO] outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()

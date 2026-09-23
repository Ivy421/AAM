"""
fine_scan_viewpoint.py

Plan fine-scan viewpoints around defect center.

Core idea:
  1. Load the defect center.
  2. Use nearby points in the coarse target frame to estimate the local surface normal by PCA.
  3. Orient the normal toward the target camera.
  4. Build a local coordinate frame:
       local +X = outward surface normal
       local +Z = world-up projected onto the local tangent plane
       local +Y = local Z x local X
  5. Build the camera-facing upper quarter sphere around defect_center.
  6. Use a fixed fine-scan radius equal to the camera working distance dc = 0.30 m.
  7. Sample viewpoints on four elevation lines and place a 60 mm IK-search cube
     around every nominal viewpoint.
  8. Run the original joint-space optimization for every cube.

Input:
  construction/data/coarse_scan/defect_roi_result.json
  construction/data/coarse_scan/coarse_point_result.npz
  construction/data/coarse_scan/coarse_icp_result.json
  construction/data/coarse_scan/coarse_png_sequence.json
  config/calibration/right_camera/ecT.npy

Output:
  construction/data/fine_scan/fine_scanpose.json
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

OUTPUT_PATH = DATA_DIR / "fine_scan" / "fine_scanpose.json"

# Fine scan geometry
CAMERA_WORK_DISTANCE = 0.30   # dc [m], fixed fine-scan radius
VIEW_SPACING_M = 0.20         # target arc spacing on each latitude
ELEVATION_DEG = [5.0, 25.0, 45.0, 65.0]
YAW_MIN_DEG = -90.0
YAW_MAX_DEG = 90.0

# Local surface-normal estimation around defect center
NORMAL_NEIGHBOR_RADIUS_M = 0.03
NORMAL_MIN_POINTS = 20

# Keep the original fixed 60 mm IK-search cube
CUBE_SIZE_M = np.array([0.06, 0.06, 0.06])
CUBE_TOL_M = 0.03

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

# Base/world axes
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
        raise ValueError(f"Cannot normalize near-zero vector: {v}")
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
    return np.asarray(arr, dtype=float)


def load_points_and_poses(npz_path):
    """
    points_collection[i]:
        per-frame point cloud already expressed in robot base/world frame.
        It may be Nx3 or homogeneous Nx4; only XYZ is used here.

    bcT_collection[i]:
        T_base_cam of the same frame.
    """
    with np.load(npz_path, allow_pickle=True) as meta:
        points_collection = [
            np.asarray(p, dtype=float) for p in meta["points_collection"]
        ]
        bcT_collection = [
            np.asarray(T, dtype=float) for T in meta["bcT_collection"]
        ]
    return points_collection, bcT_collection


def points_to_xyz(points):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Invalid point cloud shape: {points.shape}")
    xyz = points[:, :3]
    return xyz[np.all(np.isfinite(xyz), axis=1)]


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
    rpy_deg = R.from_matrix(T_base_ee[:3, :3]).as_euler(
        EULER_ORDER, degrees=True
    )
    return [
        round(float(v), 2)
        for v in np.concatenate([xyz_mm, rpy_deg])
    ]


# =========================
# Local defect frame
# =========================
def estimate_defect_local_frame(
    defect_center,
    target_points_base,
    target_camera_position,
):
    """
    Estimate the local surface frame around defect_center.

    PCA:
      - smallest-eigenvalue eigenvector -> local surface normal
      - normal is flipped to point toward the target camera

    Local frame:
      +X = outward normal
      +Z = world +Z projected onto tangent plane
      +Y = Z x X
    """
    points = points_to_xyz(target_points_base)

    dist = np.linalg.norm(points - defect_center, axis=1)
    neighbors = points[dist <= NORMAL_NEIGHBOR_RADIUS_M]

    if len(neighbors) < NORMAL_MIN_POINTS:
        raise RuntimeError(
            f"Only {len(neighbors)} points found within "
            f"{NORMAL_NEIGHBOR_RADIUS_M * 1000:.1f} mm of defect center; "
            f"need at least {NORMAL_MIN_POINTS}."
        )

    centered = neighbors - np.mean(neighbors, axis=0)
    covariance = centered.T @ centered / len(neighbors)
    eigvals, eigvecs = np.linalg.eigh(covariance)

    # Smallest principal axis = local plane normal
    local_x = normalize(eigvecs[:, 0])

    # Orient normal toward the current/target camera
    camera_dir = normalize(target_camera_position - defect_center)
    if np.dot(local_x, camera_dir) < 0.0:
        local_x = -local_x

    # World-up projected onto local tangent plane
    local_z_raw = BASE_Z - np.dot(BASE_Z, local_x) * local_x

    # If the surface normal is almost parallel to world Z, use a PCA tangent axis.
    if np.linalg.norm(local_z_raw) < 1e-6:
        local_z_raw = eigvecs[:, 2]

    local_z = normalize(local_z_raw)
    if np.dot(local_z, BASE_Z) < 0.0:
        local_z = -local_z

    local_y = normalize(np.cross(local_z, local_x))
    local_z = normalize(np.cross(local_x, local_y))

    return {
        "x": local_x,
        "y": local_y,
        "z": local_z,
        "neighbor_count": int(len(neighbors)),
        "eigenvalues": eigvals,
    }


def local_spherical_direction(local_x, local_y, local_z, yaw_deg, elev_deg):
    """
    Direction on the defect-centered, camera-facing upper quarter sphere.

    yaw = 0:
        outward local surface normal (+X)

    yaw in [-90, +90]:
        sweeps from one local side to the other while remaining on the
        camera-facing half-space.

    elevation:
        raises the viewpoint toward local +Z.
    """
    yaw = np.deg2rad(yaw_deg)
    elev = np.deg2rad(elev_deg)

    horizontal = (
        np.cos(yaw) * local_x
        + np.sin(yaw) * local_y
    )

    direction = (
        np.cos(elev) * horizontal
        + np.sin(elev) * local_z
    )
    return normalize(direction)


def build_fine_cube_centers(defect_center, local_frame, radius):
    """
    Sample nominal viewpoints on the local front-upper quarter sphere.

    For each elevation line:
        front-half arc length = pi * radius * cos(elevation)

    The number of viewpoints is chosen so that mean spacing along the
    latitude arc is approximately VIEW_SPACING_M.

    Every nominal viewpoint is used as the center of the original 60 mm cube.
    """
    local_x = local_frame["x"]
    local_y = local_frame["y"]
    local_z = local_frame["z"]

    cube_records = []

    for elev_deg in ELEVATION_DEG:
        elev_rad = np.deg2rad(elev_deg)
        arc_length = np.pi * radius * np.cos(elev_rad)

        n_views = max(
            2,
            int(np.round(arc_length / VIEW_SPACING_M))
        )

        yaw_step = (YAW_MAX_DEG - YAW_MIN_DEG) / n_views
        yaw_values = (
            YAW_MIN_DEG
            + (np.arange(n_views, dtype=float) + 0.5) * yaw_step
        )

        actual_spacing = arc_length / n_views

        print(
            f"Elevation {elev_deg:>5.1f} deg: "
            f"arc={arc_length:.3f} m, "
            f"views={n_views}, "
            f"spacing~{actual_spacing:.3f} m"
        )

        for i, yaw_deg in enumerate(yaw_values):
            direction = local_spherical_direction(
                local_x,
                local_y,
                local_z,
                yaw_deg,
                elev_deg,
            )

            cube_center = defect_center + radius * direction

            name = (
                f"elev_{int(round(elev_deg)):02d}_"
                f"view_{i + 1:02d}_"
                f"yaw_{yaw_deg:+.1f}"
            )

            # All fine views look directly at defect_center.
            look_target = defect_center.copy()

            cube_records.append(
                (
                    name,
                    cube_center,
                    direction,
                    float(yaw_deg),
                    float(elev_deg),
                    float(radius),
                    look_target,
                )
            )

    return cube_records


# =========================
# Optimization helpers
# =========================
def cube_outside_residual(p, cube_center):
    half = CUBE_SIZE_M / 2.0
    outside = np.maximum(
        np.abs(p - cube_center) - half,
        0.0,
    )
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
    dot_value = float(
        np.clip(
            np.dot(z_cam, target_dir),
            -1.0,
            1.0,
        )
    )
    return float(np.rad2deg(np.arccos(dot_value)))


def is_inside_cube(p, cube_center):
    half = CUBE_SIZE_M / 2.0 + CUBE_TOL_M
    return bool(
        np.all(
            np.abs(p - cube_center) <= half
        )
    )


def q_to_cam_pose(model, data, frame_id, q, T_ee_cam):
    T_base_ee_pin = frame_pose(
        model,
        data,
        q,
        frame_id,
    )

    T_base_ee = np.eye(4)
    T_base_ee[:3, :3] = T_base_ee_pin.rotation
    T_base_ee[:3, 3] = T_base_ee_pin.translation

    T_base_cam = T_base_ee @ T_ee_cam
    return T_base_ee, T_base_cam


def joint_limit_margins_deg(q, lb, ub):
    margins_rad = np.minimum(
        q - lb,
        ub - q,
    )
    return np.rad2deg(margins_rad)


def pull_back_from_joint_limits(
    q,
    lb,
    ub,
    margin_deg=JOINT_LIMIT_FALLBACK_DEG,
):
    margin = np.deg2rad(margin_deg)
    lower = lb + margin
    upper = ub - margin

    too_narrow = lower > upper
    center = 0.5 * (lb + ub)

    lower = np.where(
        too_narrow,
        center,
        lower,
    )
    upper = np.where(
        too_narrow,
        center,
        upper,
    )

    q_safe = np.clip(
        q,
        lower,
        upper,
    )

    adjusted = bool(
        np.any(
            np.abs(q_safe - q) > 1e-10
        )
    )
    return q_safe, adjusted


def joint_limit_score_penalty(min_margin_deg):
    if min_margin_deg >= JOINT_LIMIT_WARN_DEG:
        return 0.0

    return (
        JOINT_LIMIT_WARN_DEG - min_margin_deg
    ) * JOINT_LIMIT_PENALTY_PER_DEG


def optimize_one_cube(
    cube_info,
    look_target,
    model,
    frame_id,
    T_ee_cam,
    lb,
    ub,
    seed_qs,
):
    (
        cube_name,
        cube_center,
        direction,
        yaw_deg,
        elev_deg,
        radius,
        _,
    ) = cube_info

    best = None

    # Keep the original fine-scan weighting.
    W_LOOK = 0.98
    W_CUBE = 0.02

    cube_scale = np.maximum(
        CUBE_SIZE_M / 2.0,
        1e-6,
    )

    look_scale = max(
        2.0
        * np.sin(
            np.deg2rad(LOOK_ANGLE_TOL_DEG) / 2.0
        ),
        1e-6,
    )

    for q0 in seed_qs:
        data = model.createData()

        def residual(q):
            _, T_base_cam = q_to_cam_pose(
                model,
                data,
                frame_id,
                q,
                T_ee_cam,
            )

            p_cam = T_base_cam[:3, 3]

            r_cube = (
                cube_outside_residual(
                    p_cam,
                    cube_center,
                )
                / cube_scale
            )

            r_look = (
                look_at_residual(
                    T_base_cam,
                    look_target,
                )
                / look_scale
            )

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

        raw_limit_margins_deg = joint_limit_margins_deg(
            q_raw,
            lb,
            ub,
        )

        q, joint_limit_adjusted = pull_back_from_joint_limits(
            q_raw,
            lb,
            ub,
        )

        limit_margins_deg = joint_limit_margins_deg(
            q,
            lb,
            ub,
        )

        min_limit_margin_deg = float(
            np.min(limit_margins_deg)
        )

        T_base_ee, T_base_cam = q_to_cam_pose(
            model,
            data,
            frame_id,
            q,
            T_ee_cam,
        )

        p_cam = T_base_cam[:3, 3]

        inside = is_inside_cube(
            p_cam,
            cube_center,
        )

        look_angle = camera_look_angle_deg(
            T_base_cam,
            look_target,
        )

        cube_err_m = float(
            np.linalg.norm(
                np.maximum(
                    np.abs(
                        p_cam - cube_center
                    )
                    - CUBE_SIZE_M / 2.0,
                    0.0,
                )
            )
        )

        feasible = (
            inside
            and (
                look_angle
                < LOOK_ANGLE_TOL_DEG
            )
        )

        score = 0.0
        score += 10000.0 if feasible else 0.0
        score -= 250.0 * look_angle
        score -= 600.0 * cube_err_m
        score -= 1.0 * float(
            np.linalg.norm(
                p_cam - cube_center
            )
        )
        score -= joint_limit_score_penalty(
            float(
                np.min(
                    raw_limit_margins_deg
                )
            )
        )

        rec = {
            "cube_name": cube_name,
            "success": bool(feasible),
            "least_squares_success": bool(res.success),
            "joint_limit_adjusted": joint_limit_adjusted,
            "min_joint_limit_margin_deg": round(
                min_limit_margin_deg,
                4,
            ),
            "raw_min_joint_limit_margin_deg": round(
                float(
                    np.min(
                        raw_limit_margins_deg
                    )
                ),
                4,
            ),
            "look_angle_deg": round(
                float(look_angle),
                4,
            ),
            "cube_outside_err_mm": round(
                cube_err_m * 1000.0,
                4,
            ),
            "camera_position_base_m": [
                round(float(v), 5)
                for v in p_cam
            ],
            "cube_center_base_m": [
                round(float(v), 5)
                for v in cube_center
            ],
            "look_target_base_m": [
                round(float(v), 5)
                for v in look_target
            ],
            "direction_from_defect_center": [
                round(float(v), 5)
                for v in direction
            ],
            "yaw_deg": round(
                float(yaw_deg),
                2,
            ),
            "elevation_deg": round(
                float(elev_deg),
                2,
            ),
            "radius_m": round(
                float(radius),
                4,
            ),
            "cube_size_m": [
                round(float(v), 5)
                for v in CUBE_SIZE_M
            ],
            "endpose": matrix_to_endpose_mm_deg(
                T_base_ee
            ),
            "joint_degrees": [
                round(float(v), 2)
                for v in np.rad2deg(q)
            ],
            "raw_joint_degrees": [
                round(float(v), 2)
                for v in np.rad2deg(q_raw)
            ],
            "jointctrl_args": [
                int(
                    round(
                        float(v) * 1000.0
                    )
                )
                for v in np.rad2deg(q)
            ],
            "score": float(score),
        }

        if (
            best is None
            or rec["score"] > best["score"]
        ):
            best = rec

        if feasible:
            break

    return best


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--coarse-scan-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--hand-eye",
        type=Path,
        default=None,
    )

    args = parser.parse_args()
    configure_paths(args)

    # 1. Defect center
    roi_result = load_json(
        DEFECT_ROI_JSON
    )
    defect_center = np.asarray(
        roi_result[
            "defect_roi_center_world_m"
        ],
        dtype=float,
    )

    # 2. Resolve coarse ICP target frame
    icp_result = load_json(
        COARSE_ICP_RESULT_FILE
    )
    sequence_names = load_json(
        PNG_SEQUENCE_FILE
    )

    target_index, target_name = resolve_target_index(
        icp_result,
        sequence_names,
    )

    points_collection, bcT_collection = load_points_and_poses(
        COARSE_POINT_FILE
    )

    target_points_base = points_collection[
        target_index
    ]

    T_base_cam_target = np.asarray(
        bcT_collection[target_index],
        dtype=float,
    )
    target_camera_position = T_base_cam_target[
        :3,
        3,
    ]

    # 3. Estimate local defect surface frame
    local_frame = estimate_defect_local_frame(
        defect_center=defect_center,
        target_points_base=target_points_base,
        target_camera_position=target_camera_position,
    )

    print(
        "\n========== Defect local frame =========="
    )
    print(
        f"Target frame: {target_name}"
    )
    print(
        f"Neighbor points: "
        f"{local_frame['neighbor_count']}"
    )
    print(
        f"Local X / outward normal: "
        f"{local_frame['x']}"
    )
    print(
        f"Local Y: {local_frame['y']}"
    )
    print(
        f"Local Z / tangent up: "
        f"{local_frame['z']}"
    )

    # 4. Fixed fine-scan radius = camera working distance
    fine_radius = CAMERA_WORK_DISTANCE

    # 5. Generate all fine-scan cubes
    print(
        "\n========== Fine viewpoint sampling =========="
    )

    fine_cubes = build_fine_cube_centers(
        defect_center=defect_center,
        local_frame=local_frame,
        radius=fine_radius,
    )

    print(
        f"Total candidate cubes: "
        f"{len(fine_cubes)}"
    )

    # 6. Robot / hand-eye
    T_ee_cam = load_matrix(
        HAND_EYE_PATH
    )

    model = load_arm_model()

    if not model.existFrame(
        DEFAULT_EE_FRAME
    ):
        raise RuntimeError(
            f"Cannot find EE frame: "
            f"{DEFAULT_EE_FRAME}"
        )

    frame_id = model.getFrameId(
        DEFAULT_EE_FRAME
    )

    lb, ub = get_safe_bounds(
        model
    )

    # 7. Joint seeds
    rng = np.random.default_rng(
        RANDOM_SEED
    )

    q_neutral = np.clip(
        pin.neutral(model),
        lb,
        ub,
    )

    q_zero = np.clip(
        np.zeros(model.nq),
        lb,
        ub,
    )

    seed_qs = [
        q_neutral,
        q_zero,
    ]

    for _ in range(
        N_RANDOM_SEEDS
    ):
        seed_qs.append(
            rng.uniform(
                lb,
                ub,
            )
        )

    # 8. Test every cube with the original IK optimization
    records = []

    for i, cube_info in enumerate(
        fine_cubes
    ):
        (
            name,
            cube_center,
            _,
            yaw,
            elev,
            radius,
            look_target,
        ) = cube_info

        print(
            f"\n{name}: "
            f"yaw={yaw:.2f}, "
            f"elev={elev:.2f}, "
            f"center={cube_center}"
        )

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

        rec[
            "defect_center_base_m"
        ] = [
            round(float(v), 6)
            for v in defect_center
        ]

        rec[
            "camera_work_distance_dc_m"
        ] = round(
            float(CAMERA_WORK_DISTANCE),
            6,
        )

        rec[
            "fine_radius_m"
        ] = round(
            float(fine_radius),
            6,
        )

        rec[
            "normal_neighbor_radius_m"
        ] = round(
            float(
                NORMAL_NEIGHBOR_RADIUS_M
            ),
            6,
        )

        rec[
            "normal_neighbor_count"
        ] = int(
            local_frame[
                "neighbor_count"
            ]
        )

        rec[
            "local_x_normal_base"
        ] = [
            round(float(v), 6)
            for v in local_frame["x"]
        ]

        rec[
            "local_y_base"
        ] = [
            round(float(v), 6)
            for v in local_frame["y"]
        ]

        rec[
            "local_z_tangent_up_base"
        ] = [
            round(float(v), 6)
            for v in local_frame["z"]
        ]

        rec[
            "target_frame_for_normal"
        ] = target_name

        records.append(
            rec
        )

        print(
            f"  success={rec['success']}, "
            f"look_angle="
            f"{rec['look_angle_deg']} deg, "
            f"cube_err="
            f"{rec['cube_outside_err_mm']} mm"
        )

    save_json(
        OUTPUT_PATH,
        records,
    )

    success_count = sum(
        int(rec["success"])
        for rec in records
    )

    print(
        "\n========== Fine scan result =========="
    )
    print(
        f"Defect center [m]: "
        f"{defect_center}"
    )
    print(
        f"Fine radius = dc [m]: "
        f"{fine_radius:.4f}"
    )
    print(
        f"View spacing target [m]: "
        f"{VIEW_SPACING_M:.3f}"
    )
    print(
        f"Fine cube size [m]: "
        f"{CUBE_SIZE_M}"
    )
    print(
        f"Candidate cubes: "
        f"{len(records)}"
    )
    print(
        f"Reachable cubes: "
        f"{success_count}/"
        f"{len(records)}"
    )
    print(
        f"Saved result to "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()

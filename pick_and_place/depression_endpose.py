"""Calculate reachable depression repair-block pick and fix endposes.

Latest data chain:
    completion/depression/*_motion
        -> already contains Mark1 motion
        -> fuse alignment delta_T_base
        -> calculate pick / fix / pre-pick / pre-fix endposes
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import pinocchio as pin

from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


# ============================================================
# Project
# ============================================================

PROJECT_ROOT = Path(
    os.getenv(
        "AAM_PROJECT_ROOT",
        Path(__file__).resolve().parents[1],
    )
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from Piper.endpose_reachability_safe import (
    DEFAULT_EE_FRAME,
    frame_pose,
    get_safe_bounds,
    load_arm_model,
    reachability_test,
)


# ============================================================
# Configuration
# ============================================================

ARM_GRIPPER_LENGTH_Z_MM = 147  # 142.5

PRINTER_CENTER_BASE_MM = np.array(
    [-266.425, 184.39, 60.8],
    dtype=float,
)

PRINT_YAW_DEG = 90.0

UNIT_SCALE = 1000.0

FIX_X_OFFSET = 0.0  # mm
FIX_Y_OFFSET = 0.0  # mm


# ============================================================
# Transform utilities
# ============================================================

def endpose_to_transform(endpose):
    endpose = np.asarray(
        endpose,
        dtype=float,
    ).reshape(6)

    transform = np.eye(
        4,
        dtype=float,
    )

    transform[:3, :3] = (
        Rotation
        .from_euler(
            "xyz",
            endpose[3:],
            degrees=True,
        )
        .as_matrix()
    )

    transform[:3, 3] = (
        endpose[:3]
    )

    return transform


def transform_to_endpose(transform):
    transform = np.asarray(
        transform,
        dtype=float,
    ).reshape(4, 4)

    return np.concatenate(
        [
            transform[:3, 3],

            Rotation
            .from_matrix(
                transform[:3, :3]
            )
            .as_euler(
                "xyz",
                degrees=True,
            ),
        ]
    )


def load_delta_T_base(alignment_path):
    alignment = json.loads(
        alignment_path.read_text(
            encoding="utf-8"
        )
    )

    return np.asarray(
        alignment[
            "delta_T_base"
        ],
        dtype=float,
    ).reshape(
        4,
        4,
    )


def transform_points_by_delta(
    points,
    delta_T_base,
):
    """
    Motion-frame points -> fuse-corrected Base frame.

        P' = R P + t
    """

    points = np.asarray(
        points,
        dtype=float,
    )

    rotation = (
        delta_T_base[:3, :3]
    )

    translation = (
        delta_T_base[:3, 3]
    )

    return (
        rotation @ points.T
    ).T + translation


def transform_vector_by_delta(
    vector,
    delta_T_base,
):
    """
    Direction vector:

        v' = R v
    """

    vector = np.asarray(
        vector,
        dtype=float,
    ).reshape(3)

    rotation = (
        delta_T_base[:3, :3]
    )

    vector = (
        rotation @ vector
    )

    norm = np.linalg.norm(
        vector
    )

    return (
        vector / norm
    )


# ============================================================
# Grab yaw search
# ============================================================

def generate_grab_yaw_candidates(
    grab_position_mm,
):
    """
    Return ordered, unique yaw candidates
    found by joint-space search.
    """

    grab_position_mm = np.asarray(
        grab_position_mm,
        dtype=float,
    ).reshape(3)

    target_position_m = (
        grab_position_mm
        / 1000.0
    )

    target_z = np.array(
        [0.0, 0.0, -1.0],
        dtype=float,
    )

    model = load_arm_model()

    frame_id = model.getFrameId(
        DEFAULT_EE_FRAME
    )

    lower, upper = (
        get_safe_bounds(
            model
        )
    )

    rng = np.random.default_rng(
        0
    )

    seeds = [
        np.clip(
            pin.neutral(model),
            lower,
            upper,
        ),

        np.clip(
            np.zeros(model.nq),
            lower,
            upper,
        ),
    ]

    seeds.extend(
        rng.uniform(
            lower,
            upper,
        )
        for _ in range(50)
    )

    candidates = []

    for seed in seeds:

        data = (
            model.createData()
        )

        def residual(q):

            pose = frame_pose(
                model,
                data,
                q,
                frame_id,
            )

            position_error = (
                pose.translation
                - target_position_m
            ) / 0.002

            ee_z = (
                pose.rotation[:, 2]
            )

            axis_error = (
                np.cross(
                    ee_z,
                    target_z,
                )
                / np.deg2rad(2.0)
            )

            direction_error = np.array(
                [
                    (
                        1.0
                        - np.dot(
                            ee_z,
                            target_z,
                        )
                    )
                    / 0.001
                ]
            )

            return np.concatenate(
                [
                    position_error,
                    axis_error,
                    direction_error,
                ]
            )

        solution = least_squares(
            residual,
            seed,
            bounds=(
                lower,
                upper,
            ),
            max_nfev=1500,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
        )

        q = (
            solution.x
        )

        pose = frame_pose(
            model,
            data,
            q,
            frame_id,
        )

        position_error_mm = float(
            np.linalg.norm(
                pose.translation
                - target_position_m
            )
            * 1000.0
        )

        tilt_error_deg = float(
            np.rad2deg(
                np.arccos(
                    np.clip(
                        np.dot(
                            pose.rotation[:, 2],
                            target_z,
                        ),
                        -1.0,
                        1.0,
                    )
                )
            )
        )

        if (
            position_error_mm > 2.0
            or tilt_error_deg > 2.0
        ):
            continue

        yaw_deg = float(
            np.degrees(
                np.arctan2(
                    pose.rotation[1, 0],
                    pose.rotation[0, 0],
                )
            )
            % 360.0
        )

        joint_margin = float(
            np.min(
                np.minimum(
                    q - lower,
                    upper - q,
                )
            )
        )

        candidates.append(
            {
                "joint_margin":
                    joint_margin,

                "position_error_mm":
                    position_error_mm,

                "tilt_error_deg":
                    tilt_error_deg,

                "yaw_deg":
                    yaw_deg,
            }
        )

    candidates.sort(
        key=lambda item: (
            -item[
                "joint_margin"
            ],
            item[
                "position_error_mm"
            ],
            item[
                "tilt_error_deg"
            ],
        )
    )

    unique_candidates = []
    seen_yaws = set()

    for candidate in candidates:

        yaw_key = round(
            candidate[
                "yaw_deg"
            ],
            1,
        )

        if yaw_key in seen_yaws:
            continue

        seen_yaws.add(
            yaw_key
        )

        unique_candidates.append(
            candidate
        )

    return unique_candidates


# ============================================================
# Pre-grab
# ============================================================

def find_pre_grab(grab_endpose):
    grab_endpose = np.asarray(grab_endpose, dtype=float).reshape(6)
    grab_result = reachability_test(grab_endpose)
    model = load_arm_model()
    frame_id = model.getFrameId(DEFAULT_EE_FRAME)
    lower, upper = get_safe_bounds(model)
    p0 = grab_endpose[:3] / 1000.0
    z_min, z_max = p0[2] + 0.008, p0[2] + 0.05
    xy_half = 0.02
    box_min, box_max = np.array([p0[0] - xy_half, p0[1] - xy_half, z_min]), np.array([p0[0] + xy_half, p0[1] + xy_half, z_max])
    target_z = Rotation.from_euler("xyz", grab_endpose[3:], degrees=True).as_matrix()[:, 2]
    q0 = np.deg2rad(np.asarray(grab_result["joint_degrees"], dtype=float))
    rng = np.random.default_rng(0)
    seeds = [np.clip(q0, lower, upper)]
    seeds += [np.clip(q0 + rng.normal(0, np.deg2rad(25), model.nq), lower, upper) for _ in range(60)]
    seeds += [rng.uniform(lower, upper) for _ in range(60)]
    best = None

    for seed in seeds:
        data = model.createData()

        def residual(q):
            pose = frame_pose(model, data, q, frame_id)
            target_p = np.clip(pose.translation, box_min, box_max)
            pos_err = pose.translation - target_p
            axis_err = np.cross(pose.rotation[:, 2], target_z)
            dir_err = np.array([1.0 - np.dot(pose.rotation[:, 2], target_z)])
            return np.r_[pos_err / 0.002, axis_err / np.deg2rad(2.0), dir_err / 0.001]

        solution = least_squares(residual, seed, bounds=(lower, upper), max_nfev=1500)
        q = solution.x
        pose = frame_pose(model, data, q, frame_id)
        in_box = np.all(pose.translation >= box_min) and np.all(pose.translation <= box_max)
        tilt_err_deg = np.rad2deg(np.arccos(np.clip(np.dot(pose.rotation[:, 2], target_z), -1.0, 1.0)))

        if in_box and tilt_err_deg < 5.0:
            margin = np.min(np.minimum(q - lower, upper - q))
            dz_mm = (pose.translation[2] - p0[2]) * 1000.0
            score = margin + dz_mm * 0.001 - tilt_err_deg * 0.01
            if best is None or score > best["score"]: best = {"score": score, "q": q.copy(), "pose": pose, "tilt_err_deg": tilt_err_deg}

    if best is not None:
        pre_grab_endpose = np.r_[best["pose"].translation * 1000.0, Rotation.from_matrix(best["pose"].rotation).as_euler("xyz", degrees=True)]
        joint_degrees = np.round(np.rad2deg(best["q"]), 3).tolist()
        print(f"Selected joint-space pre_grab in upper cube with free yaw: yaw={pre_grab_endpose[5] % 360.0:.2f} deg, +Z={(best['pose'].translation[2] - p0[2]) * 1000.0:.2f} mm, tilt error={best['tilt_err_deg']:.3f} deg")
        return pre_grab_endpose, joint_degrees

    print("Warning: pre_grab is not reachable; saving false.")
    return False, False

# ============================================================
# leave_clearance
# ============================================================

def find_leave_clearance(grab_endpose, grab_joint_degrees):
    current_pose = np.asarray(grab_endpose, dtype=float).reshape(6)
    current_q = np.deg2rad(np.asarray(grab_joint_degrees, dtype=float))
    p0 = current_pose[:3] / 1000.0
    box_min, box_max = np.array([p0[0] - 0.020, p0[1] - 0.020, p0[2] + 0.005]), np.array([p0[0] + 0.020, p0[1] + 0.020, p0[2] + 0.050])
    target_R = Rotation.from_euler("xyz", current_pose[3:], degrees=True).as_matrix()
    target_z = target_R[:, 2]

    def wrap_rad(a): return (a + np.pi) % (2.0 * np.pi) - np.pi

    def yaw_from_R(Rm):
        x = Rm[:, 0].copy()
        x[2] = 0.0
        return np.arctan2(x[1], x[0]) if np.linalg.norm(x) > 1e-8 else 0.0

    target_yaw = yaw_from_R(target_R)
    model = load_arm_model()
    frame_id = model.getFrameId(DEFAULT_EE_FRAME)
    lower, upper = get_safe_bounds(model)
    rng = np.random.default_rng(0)
    joint_std = np.deg2rad(np.array([15, 20, 25, 60, 30, 120], dtype=float))
    seeds = [np.clip(current_q, lower, upper)]
    seeds += [np.clip(current_q + rng.normal(0.0, joint_std, model.nq), lower, upper) for _ in range(120)]
    best = None

    for seed in seeds:
        data = model.createData()

        def residual(q):
            pose = frame_pose(model, data, q, frame_id)
            target_p = np.clip(pose.translation, box_min, box_max)
            pos_err = pose.translation - target_p
            axis_err = np.cross(pose.rotation[:, 2], target_z)
            dir_err = np.array([1.0 - np.dot(pose.rotation[:, 2], target_z)])
            yaw_err = np.array([wrap_rad(yaw_from_R(pose.rotation) - target_yaw)])
            return np.r_[pos_err / 0.002, axis_err / np.deg2rad(5.0), dir_err / 0.005, yaw_err / np.deg2rad(15.0)]

        solution = least_squares(residual, seed, bounds=(lower, upper), max_nfev=1500)
        q = solution.x
        pose = frame_pose(model, data, q, frame_id)
        in_box = np.all(pose.translation >= box_min) and np.all(pose.translation <= box_max)
        tilt_err_deg = np.rad2deg(np.arccos(np.clip(np.dot(pose.rotation[:, 2], target_z), -1.0, 1.0)))
        yaw_err_deg = abs(np.rad2deg(wrap_rad(yaw_from_R(pose.rotation) - target_yaw)))

        if in_box and tilt_err_deg < 15.0 and yaw_err_deg < 15.0:
            dz_mm = (pose.translation[2] - p0[2]) * 1000.0
            xy_err_mm = np.linalg.norm(pose.translation[:2] - p0[:2]) * 1000.0
            margin = np.min(np.minimum(q - lower, upper - q))
            score = margin + dz_mm * 0.001 - xy_err_mm * 0.001 - tilt_err_deg * 0.01 - yaw_err_deg * 0.01
            if best is None or score > best["score"]: best = {"score": score, "q": q.copy(), "pose": pose, "dz_mm": dz_mm, "xy_err_mm": xy_err_mm, "tilt_err_deg": tilt_err_deg, "yaw_err_deg": yaw_err_deg}

    if best is not None:
        leave_endpose = np.r_[best["pose"].translation * 1000.0, Rotation.from_matrix(best["pose"].rotation).as_euler("xyz", degrees=True)]
        leave_joint_degrees = np.round(np.rad2deg(best["q"]), 3).tolist()
        status = {"success": True, "reason": "pre_grab_failed_leave_clearance_found", "dz_mm": best["dz_mm"], "xy_err_mm": best["xy_err_mm"], "tilt_err_deg": best["tilt_err_deg"], "yaw_err_deg": best["yaw_err_deg"]}
        print(f"Selected leave_clearance: +Z {best['dz_mm']:.2f} mm, XY {best['xy_err_mm']:.2f} mm, tilt {best['tilt_err_deg']:.2f} deg, yaw error {best['yaw_err_deg']:.2f} deg")
        return leave_endpose, leave_joint_degrees, status

    status = {"success": False, "reason": "pre_grab_failed_and_leave_clearance_failed"}
    print("Warning: pre_grab failed and leave_clearance also failed; saving false, not interrupting calculation.")
    return False, False, status
    

# ============================================================
# Pre-fix
# ============================================================

def find_pre_fix(
    fix_endpose,
    segments_path,
    fix_points_path,
    delta_T_base,
):
    """
    Search joint space for an EE pose
    inside the continuous pre-fix box.

    Input geometry:
        fix_points_curve_motion.pcd
        adaptive segments generated from motion PCD

    Mark1 motion is already included.

    Only apply fuse alignment here.
    """

    # --------------------------------------------------------
    # Load adaptive segments
    # --------------------------------------------------------

    segment_data = json.loads(
        segments_path.read_text(
            encoding="utf-8"
        )
    )

    # --------------------------------------------------------
    # Average outward normal
    #
    # Segment normals are already after Mark1 motion.
    # Only rotate them by delta_T_base.
    # --------------------------------------------------------

    normal = np.asarray(
        [
            segment[
                "outward_normal_unit"
            ]
            for segment
            in segment_data[
                "segments"
            ]
        ],
        dtype=float,
    ).sum(
        axis=0
    )

    normal_norm = np.linalg.norm(
        normal
    )

    normal /= normal_norm

    normal = (
        transform_vector_by_delta(
            normal,
            delta_T_base,
        )
    )

    # --------------------------------------------------------
    # Load fix_points_curve_motion.pcd
    #
    # DO NOT apply Mark1 translation again.
    # --------------------------------------------------------

    points = np.asarray(
        o3d.io.read_point_cloud(
            str(
                fix_points_path
            )
        ).points,
        dtype=float,
    )

    # --------------------------------------------------------
    # Apply fuse alignment once
    # --------------------------------------------------------

    points = (
        transform_points_by_delta(
            points,
            delta_T_base,
        )
    )

    # --------------------------------------------------------
    # Determine pre-fix search corner
    # --------------------------------------------------------

    first_corner_m = (
        points[
            np.argmax(
                points @ normal
            )
        ]
        + 0.030
        * normal
    )

    fix_endpose = np.asarray(
        fix_endpose,
        dtype=float,
    ).reshape(6)

    box_min_xy = (
        first_corner_m[:2]
    )

    box_max_xy = (
        box_min_xy
        + np.array(
            [0.080, 0.080],
            dtype=float,
        )
    )

    fixed_z_m = (
        fix_endpose[2]
        / 1000.0
    )

    target_rotation = (
        Rotation
        .from_euler(
            "xyz",
            fix_endpose[3:],
            degrees=True,
        )
        .as_matrix()
    )

    # --------------------------------------------------------
    # IK setup
    # --------------------------------------------------------

    model = load_arm_model()

    frame_id = model.getFrameId(
        DEFAULT_EE_FRAME
    )

    lower, upper = (
        get_safe_bounds(
            model
        )
    )

    rng = np.random.default_rng(
        0
    )

    seeds = [
        np.clip(
            pin.neutral(model),
            lower,
            upper,
        ),

        np.clip(
            np.zeros(model.nq),
            lower,
            upper,
        ),
    ]

    seeds.extend(
        rng.uniform(
            lower,
            upper,
        )
        for _ in range(30)
    )

    position_scale_m = 0.002
    rotation_scale_rad = np.deg2rad(
        2.0
    )

    position_tolerance_m = 0.002
    rotation_tolerance_deg = 3.0

    best = None

    def closest_box_point(
        position,
    ):

        xy = np.clip(
            position[:2],
            box_min_xy,
            box_max_xy,
        )

        return np.array(
            [
                xy[0],
                xy[1],
                fixed_z_m,
            ],
            dtype=float,
        )

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    for seed in seeds:

        data = (
            model.createData()
        )

        def residual(q):

            pose = frame_pose(
                model,
                data,
                q,
                frame_id,
            )

            position_error = (
                pose.translation
                - closest_box_point(
                    pose.translation
                )
            )

            rotation_error = (
                Rotation
                .from_matrix(
                    target_rotation.T
                    @ pose.rotation
                )
                .as_rotvec()
            )

            return np.concatenate(
                [
                    position_error
                    / position_scale_m,

                    rotation_error
                    / rotation_scale_rad,
                ]
            )

        solution = least_squares(
            residual,
            seed,
            bounds=(
                lower,
                upper,
            ),
            max_nfev=1500,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
        )

        q = (
            solution.x
        )

        pose = frame_pose(
            model,
            data,
            q,
            frame_id,
        )

        target_position = (
            closest_box_point(
                pose.translation
            )
        )

        position_error_m = float(
            np.linalg.norm(
                pose.translation
                - target_position
            )
        )

        rotation_error_deg = float(
            np.rad2deg(
                np.linalg.norm(
                    Rotation
                    .from_matrix(
                        target_rotation.T
                        @ pose.rotation
                    )
                    .as_rotvec()
                )
            )
        )

        if (
            position_error_m
            > position_tolerance_m
            or
            rotation_error_deg
            > rotation_tolerance_deg
        ):
            continue

        joint_margin = float(
            np.min(
                np.minimum(
                    q - lower,
                    upper - q,
                )
            )
        )

        score = (
            joint_margin
            - 1000.0
            * position_error_m
            - np.deg2rad(
                rotation_error_deg
            )
        )

        if (
            best is None
            or score
            > best[
                "score"
            ]
        ):

            best = {
                "score":
                    score,

                "q":
                    q.copy(),

                "target_position":
                    target_position,

                "position_error_mm":
                    position_error_m
                    * 1000.0,

                "rotation_error_deg":
                    rotation_error_deg,
            }

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    if best is not None:

        pre_fix_endpose = (
            fix_endpose.copy()
        )

        pre_fix_endpose[:3] = (
            best[
                "target_position"
            ]
            * 1000.0
        )

        joint_degrees = (
            np.round(
                np.rad2deg(
                    best[
                        "q"
                    ]
                ),
                3,
            )
            .tolist()
        )

        print(
            "Selected continuous-box pre_fix: "
            f"position error="
            f"{best['position_error_mm']:.3f} mm, "
            f"rotation error="
            f"{best['rotation_error_deg']:.3f} deg"
        )

        return (
            pre_fix_endpose,
            joint_degrees,
        )

    print(
        "Warning: pre_fix is not "
        "reachable; saving false."
    )

    return (
        False,
        False,
    )


# ============================================================
# Arguments
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        help=(
            "Depression result folder "
            "for standalone debugging."
        ),
    )

    parser.add_argument(
        "--run-dir",
        type=Path,
        help=(
            "Pipeline run folder: "
            "read completion/depression "
            "and write pickplace."
        ),
    )

    parser.add_argument(
        "--depression-dir",
        type=Path,
    )

    parser.add_argument(
        "--pick-dir",
        type=Path,
    )

    parser.add_argument(
        "--output",
        type=Path,
    )

    parser.add_argument(
        "--fix-points",
        type=Path,
    )

    parser.add_argument(
        "--segments-json",
        type=Path,
    )

    parser.add_argument(
        "--orientation-meta",
        type=Path,
    )

    parser.add_argument(
    "--mark1-motion",
    type=Path,
    )

    parser.add_argument(
        "--alignment-json",
        type=Path,
    )

    return parser.parse_args()


# ============================================================
# Paths
# ============================================================

def resolve_paths(args):

    standalone = [
        path
        for path
        in (
            args.input_dir,
            args.depression_dir,
        )
        if path is not None
    ]

    # ========================================================
    # Pipeline mode
    # ========================================================

    if args.run_dir is not None:

        if standalone:
            raise ValueError(
                "--run-dir cannot be combined "
                "with input_dir or --depression-dir"
            )

        run_dir = (
            args.run_dir
            .expanduser()
            .resolve()
        )

        depression_path = (
            run_dir
            / "completion"
            / "depression"
        )

        pick_path = (
            args.pick_dir
            .expanduser()
            .resolve()
            if args.pick_dir
            else run_dir
            / "pickplace"
        )

        # ----------------------------------------------------
        # Motion-frame fix points
        # ----------------------------------------------------

        fix_points_path = (
            args.fix_points
            .expanduser()
            .resolve()
            if args.fix_points
            else depression_path
            / "fix_points_curve_motion.pcd"
        )

        # ----------------------------------------------------
        # Adaptive segments generated from
        # fix_points_curve_motion.pcd
        # ----------------------------------------------------

        segments_path = (
            args.segments_json
            .expanduser()
            .resolve()
            if args.segments_json
            else depression_path
            / "glue_brush_adaptive_segments.json"
        )

        # ----------------------------------------------------
        # Orientation metadata after Mark1 motion
        #
        # base_T_full inside this file must already be in
        # the new Base frame after Mark1 motion.
        # ----------------------------------------------------

        orientation_meta_path = (
            args.orientation_meta.expanduser().resolve()
            if args.orientation_meta
            else depression_path / "orientation_meta.npz"
        )

        mark1_motion_path = (
            args.mark1_motion.expanduser().resolve()
            if args.mark1_motion
            else depression_path / "mark1_motion.json"
        )

        # ----------------------------------------------------
        # Fuse alignment
        # ----------------------------------------------------

        alignment_path = (
            args.alignment_json
            .expanduser()
            .resolve()
            if args.alignment_json
            else run_dir
            / "pickplace"
            / "iterative_correction.json"
        )

    # ========================================================
    # Standalone mode
    # ========================================================

    else:

        if len(
            standalone
        ) != 1:

            raise ValueError(
                "Specify exactly one of "
                "--run-dir, input_dir, "
                "or --depression-dir"
            )

        depression_path = (
            standalone[0]
            .expanduser()
            .resolve()
        )

        if (
            args.pick_dir is None
            and args.output is None
        ):
            raise ValueError(
                "Standalone mode requires "
                "--pick-dir or --output"
            )

        pick_path = (
            args.pick_dir
            .expanduser()
            .resolve()
            if args.pick_dir
            else args.output
            .parent
            .resolve()
        )

        fix_points_path = (
            args.fix_points
            .expanduser()
            .resolve()
            if args.fix_points
            else depression_path
            / "fix_points_curve_motion.pcd"
        )

        segments_path = (
            args.segments_json
            .expanduser()
            .resolve()
            if args.segments_json
            else depression_path
            / "glue_brush_adaptive_segments.json"
        )

        orientation_meta_path = (
            args.orientation_meta
            .expanduser()
            .resolve()
            if args.orientation_meta
            else depression_path
            / "orientation_meta.npz"
        )
        mark1_motion_path = (
        args.mark1_motion.expanduser().resolve()
        if args.mark1_motion
        else depression_path / "mark1_motion.json" )

        if args.alignment_json is None:
            raise ValueError(
                "Standalone mode requires "
                "--alignment-json"
            )

        alignment_path = (
            args.alignment_json
            .expanduser()
            .resolve()
        )

    # ========================================================
    # Output
    # ========================================================

    output_path = (
        args.output
        .expanduser()
        .resolve()
        if args.output
        else pick_path
        / "pick_place_endpose.npz"
    )

    return (
        depression_path,
        orientation_meta_path,
        mark1_motion_path,
        fix_points_path,
        segments_path,
        alignment_path,
        output_path,
    )


# ============================================================
# Main calculation
# ============================================================

def calculate_pick_and_fix(
    depression_path,
    orientation_meta_path,
    mark1_motion_path,
    segments_path,
    fix_points_path,
    alignment_path,
):

    # ========================================================
    # Load metadata
    # ========================================================

    orientation_meta = np.load(
        orientation_meta_path,
        allow_pickle=True,
    )
    motion = json.loads( mark1_motion_path.read_text(  encoding="utf-8" ))

    mark1_delta_xy = np.asarray(
        motion["final_delta_base_m"],
        dtype=float,
    ).reshape(2)

    gripper_meta = np.load(
        depression_path
        / "gripper_meta.npz",
        allow_pickle=True,
    )

    # --------------------------------------------------------
    # Local geometry.
    #
    # These quantities describe the oriented printed object
    # itself and are independent of Base-frame alignment.
    # --------------------------------------------------------

    attach_center = np.asarray(
        orientation_meta[
            "attach_center_oriented"
        ],
        dtype=float,
    ) * UNIT_SCALE

    full_box_z_height = float(
        orientation_meta[
            "full_box_z_height"
        ]
    ) * UNIT_SCALE

    grip_height_total = float(
        gripper_meta[
            "grip_body_height"
        ]
        + gripper_meta[
            "base_height"
        ]
        + gripper_meta[
            "v_neck_height"
        ]
    ) * UNIT_SCALE

    # ========================================================
    # Printer frame
    # ========================================================

    theta = np.deg2rad(
        PRINT_YAW_DEG
    )

    base_T_printer = np.eye(
        4,
        dtype=float,
    )

    base_T_printer[:3, :3] = np.array(
        [
            [
                np.cos(theta),
                -np.sin(theta),
                0.0,
            ],
            [
                np.sin(theta),
                np.cos(theta),
                0.0,
            ],
            [
                0.0,
                0.0,
                1.0,
            ],
        ],
        dtype=float,
    )

    base_T_printer[:3, 3] = (
        PRINTER_CENTER_BASE_MM
    )

    printer_T_full = np.eye(
        4,
        dtype=float,
    )

    printer_T_full[
        2,
        3,
    ] = (
        full_box_z_height
        / 2.0
    )

    # ========================================================
    # Grab position on printer
    # ========================================================

    printer_P_grip = np.array(
        [
            attach_center[0],
            attach_center[1],

            full_box_z_height
            - grip_height_total
            / 2.0,

            1.0,
        ],
        dtype=float,
    )

    base_P_grip = (
        base_T_printer
        @ printer_P_grip
    )

    grab_position = (
        base_P_grip[:3]
        .copy()
    )

    grab_position[2] += (
        ARM_GRIPPER_LENGTH_Z_MM
    )

    # Original object pose before Mark1 motion
    base_T_object_fix_original = np.asarray(
        orientation_meta["base_T_full"],
        dtype=float,
    ).copy()

    T_mark1 = np.eye(
        4,
        dtype=float,
    )

    T_mark1[:3, 3] = np.array(
        [
            -mark1_delta_xy[0],
            -mark1_delta_xy[1],
            0.0,
        ],
        dtype=float,
    )

    base_T_object_fix_motion = (
        T_mark1
        @ base_T_object_fix_original
    )

    # --------------------------------------------------------
    # Then apply fuse alignment exactly once
    # --------------------------------------------------------

    delta_T_base = load_delta_T_base(
        alignment_path
    )

    base_T_object_fix = (
        delta_T_base
        @ base_T_object_fix_motion
    )

    # --------------------------------------------------------
    # m -> mm only after all Base-frame transforms
    # --------------------------------------------------------

    base_T_object_fix[
        :3,
        3,
    ] *= UNIT_SCALE

    base_T_object_fix[
        0,
        3,
    ] += FIX_X_OFFSET

    base_T_object_fix[
        1,
        3,
    ] += FIX_Y_OFFSET

    # ========================================================
    # Jointly reachable Grab + Fix
    # ========================================================

    selected = None

    for candidate in (
        generate_grab_yaw_candidates(
            grab_position
        )
    ):

        grab_endpose = np.array(
            [
                grab_position[0],
                grab_position[1],
                grab_position[2],

                180.0,
                0.0,

                candidate[
                    "yaw_deg"
                ],
            ],
            dtype=float,
        )

        grab_reachability = (
            reachability_test(
                grab_endpose
            )
        )

        if not grab_reachability[
            "reachable"
        ]:
            continue

        # ----------------------------------------------------
        # EE pose at printer grab position
        # ----------------------------------------------------

        base_T_end_grab = (
            endpose_to_transform(
                grab_endpose
            )
        )

        # ----------------------------------------------------
        # Fixed relation:
        #
        # end_grab_T_full
        #
        # represents object pose relative to EE
        # when the printed block is grabbed.
        # ----------------------------------------------------

        end_grab_T_full = (
            np.linalg.inv(
                base_T_end_grab
            )
            @ base_T_printer
            @ printer_T_full
        )

        # ----------------------------------------------------
        # Required EE pose at defect location
        # ----------------------------------------------------

        base_T_end_fix = (
            base_T_object_fix
            @ np.linalg.inv(
                end_grab_T_full
            )
        )

        fix_endpose = (
            transform_to_endpose(
                base_T_end_fix
            )
        )

        fix_reachability = (
            reachability_test(
                fix_endpose
            )
        )

        if not fix_reachability[
            "reachable"
        ]:
            continue

        selected = (
            grab_endpose,
            grab_reachability,
            fix_endpose,
            fix_reachability,
        )

        print(
            "Selected first jointly reachable yaw: "
            f"{candidate['yaw_deg']:.3f} deg"
        )

        break

    (
        grab_endpose,
        grab_reachability,
        fix_endpose,
        fix_reachability,
    ) = selected

    # ========================================================
    # Pre-grab
    # ========================================================

    #pre_grab_endpose, pre_grab_joint_degrees = find_pre_grab(grab_endpose)
    leave_clearance_endpose, leave_clearance_joint_degrees = False, False
    leave_clearance_status = {"success": None, "reason": "pre_grab_found_leave_clearance_not_used"}

    # if pre_grab_joint_degrees is False:
    #     print("pre_grab failed -> searching leave_clearance")
    #     leave_clearance_endpose, leave_clearance_joint_degrees, leave_clearance_status = find_leave_clearance(grab_endpose, grab_reachability["joint_degrees"])

    leave_clearance_endpose, leave_clearance_joint_degrees, leave_clearance_status = find_leave_clearance(grab_endpose, grab_reachability["joint_degrees"])

    (
        pre_fix_endpose,
        pre_fix_joint_degrees,
    ) = find_pre_fix(
        fix_endpose=
            fix_endpose,

        segments_path=
            segments_path,

        fix_points_path=
            fix_points_path,

        delta_T_base=
            delta_T_base,
    )

    return {
        "grab_endpose":
            grab_endpose,

        "grab_joint_degrees":
            grab_reachability[
                "joint_degrees"
            ],

        "fix_endpose":
            fix_endpose,

        "fix_joint_degrees":
            fix_reachability[
                "joint_degrees"
            ],

        #"pre_grab_endpose": pre_grab_endpose,
        #"pre_grab_joint_degrees": pre_grab_joint_degrees,
        "leave_clearance_endpose": leave_clearance_endpose,
        "leave_clearance_joint_degrees":leave_clearance_joint_degrees,
        "leave_clearance_status":leave_clearance_status,

        "pre_fix_endpose":
            pre_fix_endpose,

        "pre_fix_joint_degrees":
            pre_fix_joint_degrees,

        "delta_T_base":
            delta_T_base,

        "orientation_meta_path":
            str(
                orientation_meta_path
            ),

        "fix_points_path":
            str(
                fix_points_path
            ),

        "segments_path":
            str(
                segments_path
            ),

        "alignment_path":
            str(
                alignment_path
            ),
    }


# ============================================================
# Save
# ============================================================

def save_results(
    results,
    output_path,
):

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez(
        output_path,

        grab_endpose=
            results[
                "grab_endpose"
            ],

        fix_endpose=
            results[
                "fix_endpose"
            ],
    )

    def json_value(value):
        if value is False or value is None:
            return value
        if isinstance(value, dict):
            return {k: json_value(v) for k, v in value.items()}
        if isinstance(value, (str, int, float, bool)):
            return value
        return np.asarray(value).tolist()

    json_output_path = (
        output_path
        .with_suffix(
            ".json"
        )
    )

    with open(
        json_output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            {
                "grab_joint_degrees":
                    json_value(
                        results[
                            "grab_joint_degrees"
                        ]
                    ),

                "fix_joint_degrees":
                    json_value(
                        results[
                            "fix_joint_degrees"
                        ]
                    ),

                "grab_endpose":
                    json_value(
                        results[
                            "grab_endpose"
                        ]
                    ),

                "fix_endpose":
                    json_value(
                        results[
                            "fix_endpose"
                        ]
                    ),

               # "pre_grab_joint_degrees":
               #     json_value(
               #         results[
               #             "pre_grab_joint_degrees"
               #         ]
               #     ),
               # "pre_grab_endpose": json_value(
               #         results[
               #             "pre_grab_endpose"
               #         ]
               #     ),

                    "leave_clearance_endpose":
                    json_value(
                        results[
                            "leave_clearance_endpose"
                        ]
                    ),

                "leave_clearance_joint_degrees":
                    json_value(
                        results[
                            "leave_clearance_joint_degrees"
                        ]
                    ),
                "leave_clearance_status":
                    json_value(
                        results[
                            "leave_clearance_status"
                        ]
                    ),
                "pre_fix_joint_degrees":
                    json_value(
                        results[
                            "pre_fix_joint_degrees"
                        ]
                    ),

                "delta_T_base":
                    json_value(
                        results[
                            "delta_T_base"
                        ]
                    ),

                "data_chain": {
                    "orientation_meta":
                        results[
                            "orientation_meta_path"
                        ],

                    "fix_points":
                        results[
                            "fix_points_path"
                        ],

                    "segments":
                        results[
                            "segments_path"
                        ],

                    "alignment":
                        results[
                            "alignment_path"
                        ],
                },
            },

            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "saved:",
        output_path,
    )

    print(
        "saved:",
        json_output_path,
    )


# ============================================================
# Main
# ============================================================

def main():

    args = (
        parse_args()
    )

    (depression_path,
    orientation_meta_path,
    mark1_motion_path,
    fix_points_path,
    segments_path,
    alignment_path,
    output_path,) = resolve_paths(args)

    results = calculate_pick_and_fix(
        depression_path=depression_path,
        orientation_meta_path=orientation_meta_path,
        mark1_motion_path=mark1_motion_path,
        segments_path=segments_path,
        fix_points_path=fix_points_path,
        alignment_path=alignment_path,
    )

    save_results(
        results,
        output_path,
    )

    print(
        "grab joint degrees:\n",
        results[
            "grab_joint_degrees"
        ],
    )

    print(
        "fix joint degrees:\n",
        results[
            "fix_joint_degrees"
        ],
    )

    #print(
    #    "pre-grab joint degrees:\n",
    #    results[
    #        "pre_grab_joint_degrees"
    #    ],
    #)
    print(
    "leave-clearance joint degrees:\n",
    results[
        "leave_clearance_joint_degrees"
    ],
    )

    print(
        "pre-fix joint degrees:\n",
        results[
            "pre_fix_joint_degrees"
        ],
    )


if __name__ == "__main__":
    main()
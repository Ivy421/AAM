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

    if norm <= 1e-12:
        raise ValueError(
            "Invalid transformed vector."
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

    if not candidates:
        raise RuntimeError(
            "No reachable grab orientation "
            "found for yaw in [0, 360)."
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

def find_pre_grab(
    grab_endpose,
):
    """
    Return first reachable point
    15--70 mm above Grab.
    """

    for offset_mm in np.linspace(
        15.0,
        70.0,
        9,
    ):

        endpose = np.asarray(
            grab_endpose,
            dtype=float,
        ).copy()

        endpose[2] += (
            offset_mm
        )

        result = reachability_test(
            endpose
        )

        if result[
            "reachable"
        ]:

            return (
                endpose,
                result[
                    "joint_degrees"
                ],
            )

    print(
        "Warning: pre_grab is not "
        "reachable; saving false."
    )

    return (
        False,
        False,
    )


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

    if normal_norm <= 1e-12:
        raise ValueError(
            f"Invalid average outward normal "
            f"in {segments_path}"
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

    if len(points) == 0:
        raise RuntimeError(
            f"Empty point cloud: "
            f"{fix_points_path}"
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

    if selected is None:
        raise RuntimeError(
            "No yaw produces both reachable "
            "Grab and Fix endposes."
        )

    (
        grab_endpose,
        grab_reachability,
        fix_endpose,
        fix_reachability,
    ) = selected

    # ========================================================
    # Pre-grab
    # ========================================================

    (
        pre_grab_endpose,
        pre_grab_joint_degrees,
    ) = find_pre_grab(
        grab_endpose
    )

    # ========================================================
    # Pre-fix
    #
    # fix_points_curve_motion.pcd
    # + adaptive motion segments
    # + delta_T_base
    # ========================================================

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

        "pre_grab_endpose":
            pre_grab_endpose,

        "pre_grab_joint_degrees":
            pre_grab_joint_degrees,

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

        if value is False:
            return False

        return (
            np.asarray(
                value
            )
            .tolist()
        )

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

                "pre_grab_joint_degrees":
                    json_value(
                        results[
                            "pre_grab_joint_degrees"
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

    print(
        "\nInput data:"
    )

    print(
        "  orientation meta:",
        orientation_meta_path,
    )

    print(
        "  fix points:",
        fix_points_path,
    )

    print(
        "  segments:",
        segments_path,
    )

    print(
        "  fuse alignment:",
        alignment_path,
    )

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

    print(
        "pre-grab joint degrees:\n",
        results[
            "pre_grab_joint_degrees"
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
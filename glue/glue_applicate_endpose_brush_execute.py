"""Calculate and execute segmented sponge-brush glue application.

Pipeline:
    fix_points_curve_motion.pcd
        -> adaptive segmentation
        -> glue_brush_adaptive_segments.json
        -> fuse alignment delta_T_base
        -> contact / pre-app endpose + IK
        -> save glue_applicate_endpose_brush.json
        -> execute Piper motion
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


# ============================================================
# Project
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Piper import endpose_reachability_safe as ik
from Piper.piper_ctrl import connect_right


# ============================================================
# Robot model
# ============================================================

DEFAULT_URDF = Path(
    "/home/smmg/AAM/config/piper/piper_description.urdf"
)


# ============================================================
# Brush geometry
# ============================================================

TAG_P_BRUSH_CENTER_M = np.array(
    [0.0, 0.0, -0.086],
    dtype=float,
)

BRUSH_RADIUS_M = 0.02
PRESS_DEPTH_M = 0.0


# ============================================================
# Pre-app
# ============================================================

DEFAULT_PRE_OFFSET_M = 0.100
PRE_OFFSET_MIN_M = 0.040
PRE_OFFSET_MAX_M = 0.350


# ============================================================
# IK
# ============================================================

SWEEP_SAMPLE_STEP_M = 0.003

CONTINUOUS_SEARCH_STARTS = 24
CONTINUOUS_SEARCH_SEED = 0

WORLD_X = np.array(
    [1.0, 0.0, 0.0],
    dtype=float,
)

WORLD_Z = np.array(
    [0.0, 0.0, 1.0],
    dtype=float,
)


# ============================================================
# Piper execution
# ============================================================

TRAVEL_SPEED = 15
CONTACT_SPEED = 5

GRIPPER_HOLD_MM = -3.0
GRIPPER_OPEN_MM = 30.0
GRIPPER_FORCE = 1.5

PICK_ABOVE_Z_MM = 40.0

# Current safe-joint operation:
# Joint 1 -> 35 deg, other joints unchanged.
SAFE_JOINT_INDEX = 0
SAFE_JOINT_DEG = 35.0

MOTION_TIMEOUT = 30.0
JOINT_TOLERANCE_DEG = 2.0

ENDPOSE_POSITION_TOLERANCE_MM = 2.0
ENDPOSE_ANGLE_TOLERANCE_DEG = 3.0


# ============================================================
# Arguments
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--brush-pick-json",
        type=Path,
    )

    parser.add_argument(
        "--segments-json",
        type=Path,
    )

    parser.add_argument(
        "--alignment-json",
        type=Path,
    )

    parser.add_argument(
        "--output",
        type=Path,
    )

    parser.add_argument(
        "--urdf",
        type=Path,
        default=DEFAULT_URDF,
    )

    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Calculate and save endposes without moving Piper.",
    )

    return parser.parse_args()


def resolve_paths(args):

    run_dir = (
        args.run_dir
        .expanduser()
        .resolve()
    )

    completion_dir = (
        run_dir
        / "completion"
        / "depression"
    )

    pickplace_dir = (
        run_dir
        / "pickplace"
    )

    # --------------------------------------------------------
    # Brush pick result
    # --------------------------------------------------------

    args.brush_pick_json = (
        args.brush_pick_json
        or pickplace_dir
        / "glue_brush_pick_endpose.json"
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # This JSON must be generated from:
    #
    # fix_points_curve_motion.pcd
    #
    # Therefore Mark1 motion has already been applied.
    # --------------------------------------------------------

    args.segments_json = (
        args.segments_json
        or completion_dir
        / "glue_brush_adaptive_segments.json"
    )

    # --------------------------------------------------------
    # Fuse alignment after Mark1 motion
    # --------------------------------------------------------

    args.alignment_json = (
        args.alignment_json
        or pickplace_dir
        / "iterative_correction.json"
    )

    # --------------------------------------------------------
    # Calculated endpose result
    # --------------------------------------------------------

    args.output = (
        args.output
        or pickplace_dir
        / "glue_applicate_endpose_brush.json"
    )

    return args


# ============================================================
# Basic utilities
# ============================================================

def load_json(path):

    return json.loads(
        Path(path).read_text(
            encoding="utf-8"
        )
    )


def normalize(vector):

    vector = np.asarray(
        vector,
        dtype=float,
    ).reshape(3)

    norm = np.linalg.norm(vector)

    if norm <= 1e-12:
        raise ValueError(
            "Cannot normalize zero vector."
        )

    return vector / norm


# ============================================================
# Mark1-motion segment + fuse alignment
# ============================================================

def transform_segment_by_delta(
    segment,
    delta_T_base,
):
    """
    Input:
        segment already expressed in Base frame
        AFTER Mark1 motion.

    Apply:
        fuse alignment delta_T_base.

    Position:
        P' = R P + t

    Direction:
        v' = R v
    """

    transformed = dict(segment)

    rotation = (
        delta_T_base[:3, :3]
    )

    translation = (
        delta_T_base[:3, 3]
    )

    # --------------------------------------------------------
    # Center
    # --------------------------------------------------------

    center = np.asarray(
        segment[
            "center_point_base_m"
        ],
        dtype=float,
    )

    transformed[
        "center_point_base_m"
    ] = (
        rotation @ center
        + translation
    ).tolist()

    # --------------------------------------------------------
    # Normal
    # --------------------------------------------------------

    normal = np.asarray(
        segment[
            "outward_normal_unit"
        ],
        dtype=float,
    )

    transformed[
        "outward_normal_unit"
    ] = normalize(
        rotation @ normal
    ).tolist()

    # --------------------------------------------------------
    # Tangent
    # --------------------------------------------------------

    if segment.get(
        "tangent_unit"
    ) is not None:

        tangent = np.asarray(
            segment[
                "tangent_unit"
            ],
            dtype=float,
        )

        transformed[
            "tangent_unit"
        ] = normalize(
            rotation @ tangent
        ).tolist()

    return transformed


# ============================================================
# Transform / pose
# ============================================================

def endpose_to_transform(endpose):

    endpose = np.asarray(
        endpose,
        dtype=float,
    ).reshape(6)

    T = np.eye(
        4,
        dtype=float,
    )

    T[:3, :3] = (
        Rotation
        .from_euler(
            "xyz",
            endpose[3:],
            degrees=True,
        )
        .as_matrix()
    )

    T[:3, 3] = (
        endpose[:3]
        / 1000.0
    )

    return T


def transform_to_endpose(T):

    xyz_mm = (
        T[:3, 3]
        * 1000.0
    )

    rpy_deg = (
        Rotation
        .from_matrix(
            T[:3, :3]
        )
        .as_euler(
            "xyz",
            degrees=True,
        )
    )

    return [
        round(
            float(value),
            6,
        )
        for value
        in np.r_[
            xyz_mm,
            rpy_deg,
        ]
    ]


def compute_flange_T_tag(
    brush_pick,
):

    base_T_tag = np.asarray(
        brush_pick[
            "base_T_tag"
        ],
        dtype=float,
    )

    base_T_flange = (
        endpose_to_transform(
            brush_pick[
                "endpose"
            ]
        )
    )

    return (
        np.linalg.inv(
            base_T_flange
        )
        @ base_T_tag
    )


# ============================================================
# Brush orientation
# ============================================================

def orient_outward_normal(
    normal,
):

    normal = normalize(
        normal
    )

    if (
        float(
            np.dot(
                normal,
                WORLD_X,
            )
        )
        > 0.0
    ):
        normal = -normal

    return normal


def horizontal_normal(
    outward_normal,
):

    normal = orient_outward_normal(
        outward_normal
    )

    return normalize(
        np.array(
            [
                normal[0],
                normal[1],
                0.0,
            ],
            dtype=float,
        )
    )


def nominal_base_R_tag(
    outward_normal_xy,
):
    """
    Tag +X = outward normal
    Tag +Z = Base +Z
    Tag +Y = Z x X
    """

    tag_z = WORLD_Z

    tag_x = normalize(
        outward_normal_xy
    )

    tag_y = normalize(
        np.cross(
            tag_z,
            tag_x,
        )
    )

    return np.column_stack(
        (
            tag_x,
            tag_y,
            tag_z,
        )
    )


def rotate_about_tag_z(
    base_R_tag,
    angle_rad,
):

    Rz = (
        Rotation
        .from_rotvec(
            float(angle_rad)
            * np.array(
                [0.0, 0.0, 1.0]
            )
        )
        .as_matrix()
    )

    return (
        base_R_tag
        @ Rz
    )


def segment_tangent_xy(
    segment,
    base_R_tag,
):

    tangent = segment.get(
        "tangent_unit"
    )

    if tangent is None:

        tangent = (
            base_R_tag[:, 1]
        )

    else:

        tangent = np.asarray(
            tangent,
            dtype=float,
        ).reshape(3)

    tangent = np.array(
        [
            tangent[0],
            tangent[1],
            0.0,
        ],
        dtype=float,
    )

    if (
        np.linalg.norm(tangent)
        <= 1e-12
    ):

        tangent = np.array(
            [
                base_R_tag[0, 1],
                base_R_tag[1, 1],
                0.0,
            ],
            dtype=float,
        )

    tangent = normalize(
        tangent
    )

    if (
        float(
            np.dot(
                tangent,
                WORLD_X,
            )
        )
        < 0.0
    ):
        tangent = -tangent

    return tangent


# ============================================================
# Brush TCP -> flange
# ============================================================

def brush_center_to_flange_transform(
    brush_center_base,
    base_R_tag,
    flange_T_tag,
):

    tag_origin_base = (
        np.asarray(
            brush_center_base,
            dtype=float,
        )
        - base_R_tag
        @ TAG_P_BRUSH_CENTER_M
    )

    base_T_tag = np.eye(
        4,
        dtype=float,
    )

    base_T_tag[:3, :3] = (
        base_R_tag
    )

    base_T_tag[:3, 3] = (
        tag_origin_base
    )

    base_T_flange = (
        base_T_tag
        @ np.linalg.inv(
            flange_T_tag
        )
    )

    return (
        base_T_flange,
        base_T_tag,
    )


def build_pose_set(
    center,
    outward_normal,
    base_R_tag,
    flange_T_tag,
    pre_offset_m,
):

    # --------------------------------------------------------
    # Contact brush center
    # --------------------------------------------------------

    contact_center = (
        center
        + (
            BRUSH_RADIUS_M
            - PRESS_DEPTH_M
        )
        * outward_normal
    )

    # --------------------------------------------------------
    # Pre-app brush center
    # --------------------------------------------------------

    pre_center = (
        contact_center
        + pre_offset_m
        * outward_normal
    )

    # --------------------------------------------------------
    # Pre-app flange
    # --------------------------------------------------------

    pre_T_flange, _ = (
        brush_center_to_flange_transform(
            pre_center,
            base_R_tag,
            flange_T_tag,
        )
    )

    # --------------------------------------------------------
    # Contact flange
    # --------------------------------------------------------

    contact_T_flange, _ = (
        brush_center_to_flange_transform(
            contact_center,
            base_R_tag,
            flange_T_tag,
        )
    )

    _, base_T_tag = (
        brush_center_to_flange_transform(
            contact_center,
            base_R_tag,
            flange_T_tag,
        )
    )

    return {
        "pre_center":
            pre_center,

        "contact_center":
            contact_center,

        "pre_endpose":
            transform_to_endpose(
                pre_T_flange
            ),

        "contact_endpose":
            transform_to_endpose(
                contact_T_flange
            ),

        "base_T_tag":
            base_T_tag,
    }


def endpose_for_brush_center(
    brush_center,
    base_R_tag,
    flange_T_tag,
):

    base_T_flange, _ = (
        brush_center_to_flange_transform(
            brush_center,
            base_R_tag,
            flange_T_tag,
        )
    )

    return transform_to_endpose(
        base_T_flange
    )


# ============================================================
# Sweep geometry
# ============================================================

def add_sweep_poses(
    segment,
    poses,
    base_R_tag,
    flange_T_tag,
):

    tangent = segment_tangent_xy(
        segment,
        base_R_tag,
    )

    xoy_length = float(
        segment.get(
            "xoy_length",
            0.0,
        )
    )

    half_length = (
        max(
            0.0,
            xoy_length,
        )
        / 2.0
    )

    contact_center = np.asarray(
        poses[
            "contact_center"
        ],
        dtype=float,
    )

    start_center = (
        contact_center
        - tangent
        * half_length
    )

    end_center = (
        contact_center
        + tangent
        * half_length
    )

    poses[
        "sweep_tangent_unit"
    ] = tangent

    poses[
        "sweep_xoy_length_m"
    ] = xoy_length

    poses[
        "sweep_start_center"
    ] = start_center

    poses[
        "sweep_end_center"
    ] = end_center

    poses[
        "start_endpose"
    ] = endpose_for_brush_center(
        start_center,
        base_R_tag,
        flange_T_tag,
    )

    poses[
        "end_endpose"
    ] = endpose_for_brush_center(
        end_center,
        base_R_tag,
        flange_T_tag,
    )

    return poses


# ============================================================
# IK context
# ============================================================

def make_ik_context():

    model = ik.load_arm_model()

    if not model.existFrame(
        ik.DEFAULT_EE_FRAME
    ):

        raise RuntimeError(
            f"Cannot find EE frame: "
            f"{ik.DEFAULT_EE_FRAME}"
        )

    frame_id = model.getFrameId(
        ik.DEFAULT_EE_FRAME
    )

    data = model.createData()

    joint_lb, joint_ub = (
        ik.get_safe_bounds(
            model
        )
    )

    neutral = np.clip(
        pin.neutral(model),
        joint_lb,
        joint_ub,
    )

    return (
        model,
        data,
        frame_id,
        joint_lb,
        joint_ub,
        neutral,
    )


def pose_residual(
    model,
    data,
    frame_id,
    q,
    target,
):

    actual = ik.frame_pose(
        model,
        data,
        q,
        frame_id,
    )

    return np.concatenate(
        (
            (
                actual.translation
                - target[:3, 3]
            )
            / ik.POS_SCALE,

            pin.log3(
                target[:3, :3].T
                @ actual.rotation
            )
            / ik.ROT_SCALE,
        )
    )


def continuous_result(
    model,
    data,
    frame_id,
    q,
    target,
    target_endpose,
):

    actual = ik.frame_pose(
        model,
        data,
        q,
        frame_id,
    )

    pos_err_mm = (
        np.linalg.norm(
            actual.translation
            - target[:3, 3]
        )
        * 1000.0
    )

    rot_err_deg = np.rad2deg(
        np.linalg.norm(
            pin.log3(
                target[:3, :3].T
                @ actual.rotation
            )
        )
    )

    return {
        "reachable":
            bool(
                pos_err_mm
                < ik.POS_TOL_MM
                and
                rot_err_deg
                < ik.ROT_TOL_DEG
            ),

        "joint_degrees":
            np.rad2deg(
                q
            ).tolist(),

        "target_endpose":
            target_endpose,

        "pos_err_mm":
            float(
                pos_err_mm
            ),

        "rot_err_deg":
            float(
                rot_err_deg
            ),
    }


# ============================================================
# Contact search
# ============================================================

def search_reachable_contact(
    center,
    outward_normal,
    flange_T_tag,
):

    (
        model,
        data,
        frame_id,
        joint_lb,
        joint_ub,
        neutral,
    ) = make_ik_context()

    nominal_rotation = (
        nominal_base_R_tag(
            outward_normal
        )
    )

    nq = model.nq

    rng = np.random.default_rng(
        CONTINUOUS_SEARCH_SEED
    )

    def unpack(x):

        return (
            x[:nq],
            x[-1],
        )

    def target_pose(angle_rad):

        base_R_tag = (
            rotate_about_tag_z(
                nominal_rotation,
                angle_rad,
            )
        )

        poses = build_pose_set(
            center,
            outward_normal,
            base_R_tag,
            flange_T_tag,
            DEFAULT_PRE_OFFSET_M,
        )

        target = (
            endpose_to_transform(
                poses[
                    "contact_endpose"
                ]
            )
        )

        return (
            base_R_tag,
            poses,
            target,
        )

    def residual(x):

        q_contact, angle_rad = (
            unpack(x)
        )

        _, _, target = (
            target_pose(
                angle_rad
            )
        )

        return pose_residual(
            model,
            data,
            frame_id,
            q_contact,
            target,
        )

    lower = np.concatenate(
        (
            joint_lb,
            [0.0],
        )
    )

    upper = np.concatenate(
        (
            joint_ub,
            [2.0 * np.pi],
        )
    )

    starts = [
        np.concatenate(
            (
                neutral,
                [0.0],
            )
        )
    ]

    for _ in range(
        CONTINUOUS_SEARCH_STARTS
        - 1
    ):

        starts.append(
            np.concatenate(
                (
                    rng.uniform(
                        joint_lb,
                        joint_ub,
                    ),

                    [
                        rng.uniform(
                            0.0,
                            2.0 * np.pi,
                        )
                    ],
                )
            )
        )

    best = None

    for x0 in starts:

        solution = least_squares(
            residual,
            x0,
            bounds=(
                lower,
                upper,
            ),
            max_nfev=1500,
            xtol=1e-9,
            ftol=1e-9,
            gtol=1e-9,
        )

        (
            q_contact,
            angle_rad,
        ) = unpack(
            solution.x
        )

        (
            base_R_tag,
            poses,
            target,
        ) = target_pose(
            angle_rad
        )

        result = continuous_result(
            model,
            data,
            frame_id,
            q_contact,
            target,
            poses[
                "contact_endpose"
            ],
        )

        candidate = {
            "roll_deg":
                float(
                    np.rad2deg(
                        angle_rad
                    )
                    % 360.0
                ),

            "base_R_tag":
                base_R_tag,

            "poses":
                poses,

            "contact_result":
                result,

            "q_contact":
                q_contact,

            "score":
                (
                    result[
                        "pos_err_mm"
                    ]
                    + result[
                        "rot_err_deg"
                    ]
                ),
        }

        if (
            best is None
            or candidate["score"]
            < best["score"]
        ):
            best = candidate

        if result[
            "reachable"
        ]:
            return candidate

    return None


# ============================================================
# Pre-app search
# ============================================================

def search_reachable_pre(
    center,
    outward_normal,
    flange_T_tag,
    base_R_tag,
    q_contact,
):

    (
        model,
        data,
        frame_id,
        joint_lb,
        joint_ub,
        neutral,
    ) = make_ik_context()

    nq = model.nq

    rng = np.random.default_rng(
        CONTINUOUS_SEARCH_SEED
        + 1
    )

    joint_range = np.maximum(
        joint_ub
        - joint_lb,
        1e-6,
    )

    def unpack(x):

        return (
            x[:nq],
            x[-1],
        )

    def target_pose(
        pre_offset_m,
    ):

        poses = build_pose_set(
            center,
            outward_normal,
            base_R_tag,
            flange_T_tag,
            pre_offset_m,
        )

        target = (
            endpose_to_transform(
                poses[
                    "pre_endpose"
                ]
            )
        )

        return (
            poses,
            target,
        )

    def residual(x):

        q_pre, pre_offset_m = (
            unpack(x)
        )

        _, target = target_pose(
            pre_offset_m
        )

        pose_error = (
            pose_residual(
                model,
                data,
                frame_id,
                q_pre,
                target,
            )
        )

        offset_preference = (
            (
                pre_offset_m
                - DEFAULT_PRE_OFFSET_M
            )
            / (
                PRE_OFFSET_MAX_M
                - PRE_OFFSET_MIN_M
            )
        )

        continuity = (
            q_pre
            - q_contact
        ) / joint_range

        return np.concatenate(
            (
                pose_error,

                [
                    0.05
                    * offset_preference
                ],

                0.02
                * continuity,
            )
        )

    lower = np.concatenate(
        (
            joint_lb,
            [PRE_OFFSET_MIN_M],
        )
    )

    upper = np.concatenate(
        (
            joint_ub,
            [PRE_OFFSET_MAX_M],
        )
    )

    starts = [
        np.concatenate(
            (
                q_contact,
                [
                    DEFAULT_PRE_OFFSET_M
                ],
            )
        ),

        np.concatenate(
            (
                neutral,
                [
                    DEFAULT_PRE_OFFSET_M
                ],
            )
        ),
    ]

    for _ in range(
        CONTINUOUS_SEARCH_STARTS
        - len(starts)
    ):

        starts.append(
            np.concatenate(
                (
                    rng.uniform(
                        joint_lb,
                        joint_ub,
                    ),

                    [
                        rng.uniform(
                            PRE_OFFSET_MIN_M,
                            PRE_OFFSET_MAX_M,
                        )
                    ],
                )
            )
        )

    best = None

    for x0 in starts:

        solution = least_squares(
            residual,
            x0,
            bounds=(
                lower,
                upper,
            ),
            max_nfev=1500,
            xtol=1e-9,
            ftol=1e-9,
            gtol=1e-9,
        )

        (
            q_pre,
            pre_offset_m,
        ) = unpack(
            solution.x
        )

        poses, target = (
            target_pose(
                pre_offset_m
            )
        )

        result = continuous_result(
            model,
            data,
            frame_id,
            q_pre,
            target,
            poses[
                "pre_endpose"
            ],
        )

        candidate = {
            "pre_offset_m":
                float(
                    pre_offset_m
                ),

            "poses":
                poses,

            "pre_result":
                result,

            "score":
                (
                    result[
                        "pos_err_mm"
                    ]
                    + result[
                        "rot_err_deg"
                    ]
                ),
        }

        if (
            best is None
            or candidate["score"]
            < best["score"]
        ):

            best = candidate

        if result[
            "reachable"
        ]:

            return candidate

    return best


# ============================================================
# Seeded IK
# ============================================================

def solve_pose_with_seed(
    model,
    data,
    frame_id,
    joint_lb,
    joint_ub,
    seed_q,
    target_endpose,
):

    target = (
        endpose_to_transform(
            target_endpose
        )
    )

    def residual(q):

        return pose_residual(
            model,
            data,
            frame_id,
            q,
            target,
        )

    solution = least_squares(
        residual,
        np.clip(
            seed_q,
            joint_lb,
            joint_ub,
        ),
        bounds=(
            joint_lb,
            joint_ub,
        ),
        max_nfev=1200,
        xtol=1e-9,
        ftol=1e-9,
        gtol=1e-9,
    )

    q = solution.x

    return (
        q,
        continuous_result(
            model,
            data,
            frame_id,
            q,
            target,
            target_endpose,
        ),
    )


def check_pose_reachable(
    target_endpose,
    seed_q=None,
):

    (
        model,
        data,
        frame_id,
        joint_lb,
        joint_ub,
        neutral,
    ) = make_ik_context()

    if seed_q is None:
        seed_q = neutral

    return solve_pose_with_seed(
        model,
        data,
        frame_id,
        joint_lb,
        joint_ub,
        seed_q,
        target_endpose,
    )


# ============================================================
# Sweep reachability
# ============================================================

def find_reachable_sweep_endpoints(
    poses,
    base_R_tag,
    flange_T_tag,
    q_contact,
):

    start_center = np.asarray(
        poses[
            "sweep_start_center"
        ],
        dtype=float,
    )

    end_center = np.asarray(
        poses[
            "sweep_end_center"
        ],
        dtype=float,
    )

    contact_center = np.asarray(
        poses[
            "contact_center"
        ],
        dtype=float,
    )

    tangent = normalize(
        np.asarray(
            poses[
                "sweep_tangent_unit"
            ],
            dtype=float,
        )
    )

    start_q, start_result = (
        check_pose_reachable(
            poses[
                "start_endpose"
            ],
            q_contact,
        )
    )

    end_q, end_result = (
        check_pose_reachable(
            poses[
                "end_endpose"
            ],
            q_contact,
        )
    )

    def search_one_side(
        sign,
        full_center,
        seed_q,
    ):

        full_distance = float(
            np.linalg.norm(
                full_center
                - contact_center
            )
        )

        if full_distance <= 1e-9:

            endpose = (
                endpose_for_brush_center(
                    contact_center,
                    base_R_tag,
                    flange_T_tag,
                )
            )

            q, result = (
                check_pose_reachable(
                    endpose,
                    seed_q,
                )
            )

            return (
                contact_center,
                endpose,
                q,
                result,
                0.0,
            )

        sample_count = max(
            2,
            int(
                np.ceil(
                    full_distance
                    / SWEEP_SAMPLE_STEP_M
                )
            )
            + 1,
        )

        distances = np.linspace(
            full_distance,
            0.0,
            sample_count,
        )

        for distance in distances:

            center = (
                contact_center
                + sign
                * tangent
                * distance
            )

            endpose = (
                endpose_for_brush_center(
                    center,
                    base_R_tag,
                    flange_T_tag,
                )
            )

            q, result = (
                check_pose_reachable(
                    endpose,
                    seed_q,
                )
            )

            if result[
                "reachable"
            ]:

                return (
                    center,
                    endpose,
                    q,
                    result,
                    float(
                        distance
                    ),
                )

        endpose = (
            endpose_for_brush_center(
                contact_center,
                base_R_tag,
                flange_T_tag,
            )
        )

        q, result = (
            check_pose_reachable(
                endpose,
                seed_q,
            )
        )

        return (
            contact_center.copy(),
            endpose,
            q,
            result,
            0.0,
        )

    if start_result[
        "reachable"
    ]:

        reachable_start = (
            start_center,
            poses[
                "start_endpose"
            ],
            start_q,
            start_result,
            float(
                np.linalg.norm(
                    start_center
                    - contact_center
                )
            ),
        )

    else:

        reachable_start = (
            search_one_side(
                -1.0,
                start_center,
                q_contact,
            )
        )

    if end_result[
        "reachable"
    ]:

        reachable_end = (
            end_center,
            poses[
                "end_endpose"
            ],
            end_q,
            end_result,
            float(
                np.linalg.norm(
                    end_center
                    - contact_center
                )
            ),
        )

    else:

        reachable_end = (
            search_one_side(
                1.0,
                end_center,
                q_contact,
            )
        )

    return {
        "requested_start_result":
            start_result,

        "requested_end_result":
            end_result,

        "reachable_start":
            reachable_start,

        "reachable_end":
            reachable_end,
    }


def joint_degrees(
    result,
):

    return [
        round(
            float(value),
            3,
        )
        for value
        in result[
            "joint_degrees"
        ]
    ]


# ============================================================
# Record
# ============================================================

def build_record(
    segment,
    normal,
    roll_deg,
    pre_offset_m,
    poses,
    contact_result,
    pre_result,
):

    return {
        "segment_id":
            int(
                segment[
                    "id"
                ]
            ),

        "segment_center_base_m":
            segment[
                "center_point_base_m"
            ],

        "segment_tangent_unit":
            segment.get(
                "tangent_unit"
            ),

        "outward_normal_unit":
            np.round(
                normal,
                9,
            ).tolist(),

        "contact_reachable":
            bool(
                contact_result
                is not None
                and contact_result[
                    "reachable"
                ]
            ),

        "pre_app_reachable":
            bool(
                pre_result
                is not None
                and pre_result[
                    "reachable"
                ]
            ),

        "tag_z_search_angle_deg":
            round(
                float(
                    roll_deg
                ),
                6,
            ),

        "pre_app_offset_mm":
            round(
                float(
                    pre_offset_m
                )
                * 1000.0,
                3,
            ),

        "brush_center_pre_base_m":
            np.round(
                poses[
                    "pre_center"
                ],
                9,
            ).tolist(),

        "brush_center_contact_base_m":
            np.round(
                poses[
                    "contact_center"
                ],
                9,
            ).tolist(),

        "pre_app_endpose":
            poses[
                "pre_endpose"
            ],

        "contact_endpose":
            poses[
                "contact_endpose"
            ],

        "pre_app_joint_degrees":
            (
                None
                if (
                    pre_result
                    is None
                    or not pre_result[
                        "reachable"
                    ]
                )
                else joint_degrees(
                    pre_result
                )
            ),

        "contact_joint_degrees":
            (
                None
                if (
                    contact_result
                    is None
                    or not contact_result[
                        "reachable"
                    ]
                )
                else joint_degrees(
                    contact_result
                )
            ),

        "pre_app_ik_error":
            (
                None
                if pre_result
                is None
                else {
                    "pos_err_mm":
                        round(
                            float(
                                pre_result[
                                    "pos_err_mm"
                                ]
                            ),
                            6,
                        ),

                    "rot_err_deg":
                        round(
                            float(
                                pre_result[
                                    "rot_err_deg"
                                ]
                            ),
                            6,
                        ),
                }
            ),

        "contact_ik_error":
            {
                "pos_err_mm":
                    round(
                        float(
                            contact_result[
                                "pos_err_mm"
                            ]
                        ),
                        6,
                    ),

                "rot_err_deg":
                    round(
                        float(
                            contact_result[
                                "rot_err_deg"
                            ]
                        ),
                        6,
                    ),
            },

        "base_R_tag":
            np.round(
                poses[
                    "base_T_tag"
                ][:3, :3],
                9,
            ).tolist(),
    }


# ============================================================
# STEP 1
# Calculate glue endposes
# ============================================================

def calculate_glue_endposes(
    args,
):

    print(
        "\n"
        "========================================\n"
        "STEP 1: CALCULATE GLUE ENDPOSES\n"
        "========================================"
    )

    brush_pick = load_json(
        args.brush_pick_json
    )

    # --------------------------------------------------------
    # Segments generated from fix_points_curve_motion.pcd
    # --------------------------------------------------------

    segment_data = load_json(
        args.segments_json
    )

    # --------------------------------------------------------
    # Fuse alignment
    # --------------------------------------------------------

    alignment = load_json(
        args.alignment_json
    )

    delta_T_base = np.asarray(
        alignment[
            "delta_T_base"
        ],
        dtype=float,
    ).reshape(
        4,
        4,
    )

    print(
        "Segment input:",
        args.segments_json.resolve(),
    )

    print(
        "Fuse alignment:",
        args.alignment_json.resolve(),
    )

    # --------------------------------------------------------
    # Mark1 motion already contained in segments.
    #
    # Apply fuse alignment exactly once.
    # --------------------------------------------------------

    segments = [
        transform_segment_by_delta(
            segment,
            delta_T_base,
        )
        for segment
        in segment_data[
            "segments"
        ]
    ]

    flange_T_tag = (
        compute_flange_T_tag(
            brush_pick
        )
    )

    records = []
    skipped_segments = []
    pre_failed = []

    # --------------------------------------------------------
    # Segment -> endpose + IK
    # --------------------------------------------------------

    for segment in segments:

        segment_id = int(
            segment[
                "id"
            ]
        )

        center = np.asarray(
            segment[
                "center_point_base_m"
            ],
            dtype=float,
        )

        normal = horizontal_normal(
            segment[
                "outward_normal_unit"
            ]
        )

        # ----------------------------------------------------
        # Contact
        # ----------------------------------------------------

        contact_candidate = (
            search_reachable_contact(
                center,
                normal,
                flange_T_tag,
            )
        )

        if contact_candidate is None:

            skipped_segments.append(
                {
                    "segment_id":
                        segment_id,

                    "reason":
                        "contact fail",
                }
            )

            print(
                f"Segment {segment_id}: "
                "contact fail -> skipped"
            )

            continue

        # ----------------------------------------------------
        # Pre-app
        # ----------------------------------------------------

        pre_candidate = (
            search_reachable_pre(
                center=center,
                outward_normal=normal,
                flange_T_tag=
                    flange_T_tag,
                base_R_tag=
                    contact_candidate[
                        "base_R_tag"
                    ],
                q_contact=
                    contact_candidate[
                        "q_contact"
                    ],
            )
        )

        pre_reachable = bool(
            pre_candidate
            is not None
            and
            pre_candidate[
                "pre_result"
            ][
                "reachable"
            ]
        )

        if pre_candidate is None:

            pre_offset_m = (
                DEFAULT_PRE_OFFSET_M
            )

            poses = (
                contact_candidate[
                    "poses"
                ]
            )

            pre_result = None

        else:

            pre_offset_m = (
                pre_candidate[
                    "pre_offset_m"
                ]
            )

            poses = (
                pre_candidate[
                    "poses"
                ]
            )

            pre_result = (
                pre_candidate[
                    "pre_result"
                ]
            )

        if pre_reachable:

            print(
                f"Segment {segment_id}: "
                "contact success, "
                "pre-app success"
            )

        else:

            pre_failed.append(
                segment_id
            )

            print(
                f"Segment {segment_id}: "
                "pre-app fail"
            )
            continue

        records.append(
            build_record(
                segment=
                    segment,

                normal=
                    normal,

                roll_deg=
                    contact_candidate[
                        "roll_deg"
                    ],

                pre_offset_m=
                    pre_offset_m,

                poses=
                    poses,

                contact_result=
                    contact_candidate[
                        "contact_result"
                    ],

                pre_result=
                    pre_result,
            )
        )

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    output = {
        "coordinate_frame":
            "base",

        "data_chain":
            (
                "fix_points_curve_motion.pcd "
                "-> adaptive segments "
                "-> delta_T_base fuse alignment "
                "-> endpose / IK"
            ),

        "segment_source":
            str(
                args.segments_json.resolve()
            ),

        "alignment_source":
            str(
                args.alignment_json.resolve()
            ),

        "delta_T_base":
            np.round(
                delta_T_base,
                9,
            ).tolist(),

        "brush_radius_mm":
            BRUSH_RADIUS_M
            * 1000.0,

        "press_depth_mm":
            PRESS_DEPTH_M
            * 1000.0,

        "tag_p_brush_center_mm":
            (
                TAG_P_BRUSH_CENTER_M
                * 1000.0
            ).tolist(),

        "flange_T_tag":
            np.round(
                flange_T_tag,
                9,
            ).tolist(),

        "input_segment_count":
            len(
                segments
            ),

        "generated_segment_count":
            len(
                records
            ),

        "skipped_segments":
            skipped_segments,

        "pre_app_failed_segment_ids":
            pre_failed,

        "segments":
            records,
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        f"\nSaved endpose result: "
        f"{args.output.resolve()}"
    )

    return (
        brush_pick,
        output,
    )


# ============================================================
# Execution validation
# ============================================================

def validate_execution(
    output,
):

    if not output[
        "segments"
    ]:

        raise RuntimeError(
            "No valid glue segments."
        )

    invalid = []

    for segment in output[
        "segments"
    ]:

        if (
            segment[
                "pre_app_joint_degrees"
            ]
            is None
            or
            segment[
                "contact_joint_degrees"
            ]
            is None
        ):

            invalid.append(
                segment[
                    "segment_id"
                ]
            )

    if invalid:

        raise RuntimeError(
            "Execution aborted before robot motion. "
            "Missing pre-app/contact IK for segments: "
            f"{invalid}"
        )


# ============================================================
# Piper waiting
# ============================================================

def angle_error_deg(
    actual,
    target,
):

    return (
        np.asarray(
            actual
        )
        - np.asarray(
            target
        )
        + 180.0
    ) % 360.0 - 180.0


def wait_joint(
    piper,
    target,
    timeout=MOTION_TIMEOUT,
):

    target = np.asarray(
        target,
        dtype=float,
    )

    deadline = (
        time.monotonic()
        + timeout
    )

    while (
        time.monotonic()
        < deadline
    ):

        current = (
            piper.get_joint()
        )

        if current is not None:

            error = angle_error_deg(
                current,
                target,
            )

            if (
                np.max(
                    np.abs(
                        error
                    )
                )
                <= JOINT_TOLERANCE_DEG
            ):

                return

        time.sleep(
            0.05
        )

    raise TimeoutError(
        "Piper did not reach "
        "target joint angles."
    )


def wait_endpose(
    piper,
    target,
    timeout=MOTION_TIMEOUT,
):

    target = np.asarray(
        target,
        dtype=float,
    )

    deadline = (
        time.monotonic()
        + timeout
    )

    while (
        time.monotonic()
        < deadline
    ):

        current = (
            piper.get_endpose()
        )

        if current is not None:

            current = np.asarray(
                current,
                dtype=float,
            )

            pos_error = np.linalg.norm(
                current[:3]
                - target[:3]
            )

            rot_error = np.max(
                np.abs(
                    angle_error_deg(
                        current[3:],
                        target[3:],
                    )
                )
            )

            if (
                pos_error
                <= ENDPOSE_POSITION_TOLERANCE_MM
                and
                rot_error
                <= ENDPOSE_ANGLE_TOLERANCE_DEG
            ):

                return

        time.sleep(
            0.05
        )

    raise TimeoutError(
        "Piper did not reach "
        "target endpose."
    )


def move_joint_wait(
    piper,
    joints,
):

    piper.move_joint(
        *joints
    )

def move_endpose_wait(
    piper,
    endpose,
):

    piper.move_endpose(
        *endpose
    )



# ============================================================
# STEP 2A
# Initial motion after brush pick
# ============================================================

def prepare_after_pick(
    piper,
    brush_pick,
):

    print(
        "\nHold brush"
    )

    # --------------------------------------------------------
    # Re-establish active gripper force
    # --------------------------------------------------------

    piper.move_gripper(
        GRIPPER_HOLD_MM,
        force=
            GRIPPER_FORCE,
    )

    time.sleep(
        1.0
    )

    # --------------------------------------------------------
    # Travel speed
    # --------------------------------------------------------

    piper.set_speed(
        TRAVEL_SPEED
    )

    # --------------------------------------------------------
    # Pick endpose Z + 40 mm
    # --------------------------------------------------------

    pick_endpose = np.asarray(
        brush_pick[
            "endpose"
        ],
        dtype=float,
    )

    above_pick = (
        pick_endpose.copy()
    )

    above_pick[2] += (
        PICK_ABOVE_Z_MM
    )

    print(
        "Move to pick endpose "
        "Z + 40 mm"
    )

    move_endpose_wait(
        piper,
        above_pick.tolist(),
    )
    time.sleep(2)

    # --------------------------------------------------------
    # Read current joints.
    #
    # Current configuration:
    # Joint 1 -> 35 deg
    # J2-J6 unchanged.
    # --------------------------------------------------------

    current_joint = (
        piper.get_joint()
    )

    current_joint = list(
        current_joint
    )

    current_joint[
        SAFE_JOINT_INDEX
    ] = SAFE_JOINT_DEG

    print(
        f"Move Joint "
        f"{SAFE_JOINT_INDEX + 1} "
        f"to {SAFE_JOINT_DEG:.1f} deg"
    )

    move_joint_wait(
        piper,
        current_joint,
    )
    time.sleep(3)


# ============================================================
# STEP 2B
# Segment contact execution
# ============================================================

def execute_segments(
    piper,
    segments,
):

    print(
        "\n"
        "========================================\n"
        "STEP 2: EXECUTE GLUE CONTACTS\n"
        "========================================"
    )

    for index, segment in enumerate(
        segments
    ):

        segment_id = (
            segment[
                "segment_id"
            ]
        )

        pre_joint = (
            segment[
                "pre_app_joint_degrees"
            ]
        )

        contact_joint = (
            segment[
                "contact_joint_degrees"
            ]
        )

        print(
            f"\nSegment {segment_id} "
            f"({index + 1}/"
            f"{len(segments)})"
        )

        # ----------------------------------------------------
        # Travel -> pre-app
        # ----------------------------------------------------

        piper.set_speed(
            TRAVEL_SPEED
        )

        print(
            f"Segment {segment_id}: "
            "pre-app"
        )

        move_joint_wait(
            piper,
            pre_joint,
        )
        time.sleep(5)

        # ----------------------------------------------------
        # Slow contact
        # ----------------------------------------------------

        piper.set_speed(
            CONTACT_SPEED
        )

        print(
            f"Segment {segment_id}: "
            "contact"
        )

        move_joint_wait(
            piper,
            contact_joint,
        )
        time.sleep(5)

        # ----------------------------------------------------
        # Retreat to same pre-app
        # ----------------------------------------------------

        piper.set_speed(
            TRAVEL_SPEED
        )

        print(
            f"Segment {segment_id}: "
            "retreat"
        )

        move_joint_wait(
            piper,
            pre_joint,
        )
        time.sleep(5)


# ============================================================
# STEP 2C
# Return brush
# ============================================================

def return_brush(
    piper,
    brush_pick,
):

    piper.set_speed(
        TRAVEL_SPEED
    )

    # --------------------------------------------------------
    # Last pre-app -> pre-pick
    # --------------------------------------------------------

    print(
        "\nReturn to pre-pick"
    )

    move_joint_wait(
        piper,
        brush_pick[
            "prepick_joint_degrees"
        ],
    )
    time.sleep(8)

    # --------------------------------------------------------
    # Pre-pick -> pick endpose
    #
    # User requested move_endpose here.
    # --------------------------------------------------------

    print(
        "Return to pick"
    )

    move_endpose_wait(
        piper,
        brush_pick[
            "endpose"
        ],
    )
    time.sleep(3)

    # --------------------------------------------------------
    # Release
    # --------------------------------------------------------

    print(
        "Open gripper"
    )

    piper.move_gripper(
        GRIPPER_OPEN_MM,
        force=
            GRIPPER_FORCE,
    )

    time.sleep(
        1.0
    )


# ============================================================
# Main
# ============================================================

def main():

    args = resolve_paths(
        parse_args()
    )

    # --------------------------------------------------------
    # Configure IK model
    # --------------------------------------------------------

    ik.DEFAULT_URDF = str(
        args.urdf
        .expanduser()
        .resolve()
    )

    ik._MODEL_CACHE = None

    # ========================================================
    # STEP 1
    #
    # motion adaptive segments
    # -> fuse alignment
    # -> endpose
    # -> IK
    # -> save JSON
    # ========================================================

    brush_pick, output = (
        calculate_glue_endposes(
            args
        )
    )

    # --------------------------------------------------------
    # Check ALL required poses before touching the robot.
    # --------------------------------------------------------

    validate_execution(
        output
    )

    if args.plan_only:

        print(
            "\nPlan-only mode: "
            "robot execution skipped."
        )

        return

    # ========================================================
    # STEP 2
    #
    # Execute already calculated joint poses.
    # ========================================================

    piper = connect_right()

    try:

        piper.clear_error(
            clear_gripper=False
        )

        piper.enable()

        # ----------------------------------------------------
        # Re-grip
        # -> pick Z + 40
        # -> Joint1 = 35 deg
        # ----------------------------------------------------

        prepare_after_pick(
            piper,
            brush_pick,
        )

        # ----------------------------------------------------
        # pre1
        # -> contact1
        # -> pre1
        # -> pre2
        # -> contact2
        # -> pre2
        # ...
        # ----------------------------------------------------

        execute_segments(
            piper,
            output[
                "segments"
            ],
        )

        # ----------------------------------------------------
        # Last pre
        # -> prepick
        # -> pick
        # -> open gripper
        # ----------------------------------------------------

        return_brush(
            piper,
            brush_pick,
        )

        print(
            "\n"
            "========================================\n"
            "GLUE APPLICATION COMPLETE\n"
            "========================================"
        )

    finally:

        piper.disconnect()


if __name__ == "__main__":
    main()
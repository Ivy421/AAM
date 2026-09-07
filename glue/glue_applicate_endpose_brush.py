"""Generate pre-contact, contact and application endposes for the sponge brush."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Piper import endpose_reachability_safe as ik


DEFAULT_URDF = Path("/home/smmg/AAM/config/piper/piper_description.urdf")

TAG_P_BRUSH_CENTER_M = np.array([0.0, 0.0, -0.086], dtype=float)
BRUSH_RADIUS_M = 0.023
PRESS_DEPTH_M = 0.0

DEFAULT_PRE_OFFSET_M = 0.100
PRE_OFFSET_MIN_M = 0.040
PRE_OFFSET_MAX_M = 0.350
SWEEP_SAMPLE_STEP_M = 0.003

CONTINUOUS_SEARCH_STARTS = 24
CONTINUOUS_SEARCH_SEED = 0

WORLD_X = np.array([1.0, 0.0, 0.0], dtype=float)
WORLD_Z = np.array([0.0, 0.0, 1.0], dtype=float)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--brush-pick-json", type=Path)
    parser.add_argument("--segments-json", type=Path)
    parser.add_argument("--alignment-json", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--no-ik", action="store_true")
    return parser.parse_args()


def resolve_paths(args):
    if args.run_dir is not None:
        run_dir = args.run_dir.expanduser().resolve()
        args.brush_pick_json = (
            args.brush_pick_json
            or run_dir / "pickplace" / "glue_brush_pick_endpose.json"
        )
        if args.segments_json is None:
            pickplace_dir = run_dir / "pickplace"
            motion = json.loads(
                (pickplace_dir / "mark1_motion.json").read_text(
                    encoding="utf-8"
                )
            )
            segments_name = (
                "glue_brush_adaptive_segments_mark1.json"
                if motion["move"]
                else "glue_brush_adaptive_segments.json"
            )
            args.segments_json = pickplace_dir / segments_name
        args.output = (
            args.output
            or run_dir / "pickplace" / "glue_applicate_endpose_brush.json"
        )
        args.alignment_json = (
            args.alignment_json
            or run_dir / "pickplace" / "iterative_correction.json"
        )

    return args


def normalize(vector):
    vector = np.asarray(vector, dtype=float).reshape(3)
    return vector / np.linalg.norm(vector)


def transform_segment_by_delta(segment, delta_T_base):
    transformed = dict(segment)
    rotation = delta_T_base[:3, :3]
    translation = delta_T_base[:3, 3]

    center = np.asarray(segment["center_point_base_m"], dtype=float)
    transformed["center_point_base_m"] = (
        rotation @ center + translation
    ).tolist()

    normal = np.asarray(segment["outward_normal_unit"], dtype=float)
    transformed["outward_normal_unit"] = normalize(
        rotation @ normal
    ).tolist()

    if segment.get("tangent_unit") is not None:
        tangent = np.asarray(segment["tangent_unit"], dtype=float)
        transformed["tangent_unit"] = normalize(
            rotation @ tangent
        ).tolist()

    return transformed


def endpose_to_transform(endpose):
    endpose = np.asarray(endpose, dtype=float).reshape(6)
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = Rotation.from_euler(
        "xyz", endpose[3:], degrees=True
    ).as_matrix()
    transform[:3, 3] = endpose[:3] / 1000.0
    return transform


def transform_to_endpose(transform):
    xyz_mm = transform[:3, 3] * 1000.0
    rpy_deg = Rotation.from_matrix(transform[:3, :3]).as_euler(
        "xyz", degrees=True
    )
    return [
        round(float(value), 6)
        for value in np.concatenate((xyz_mm, rpy_deg))
    ]


def compute_flange_T_tag(brush_pick):
    base_T_tag = np.asarray(brush_pick["base_T_tag"], dtype=float)
    base_T_flange = endpose_to_transform(brush_pick["endpose"])
    return np.linalg.inv(base_T_flange) @ base_T_tag


def orient_outward_normal(normal):
    normal = normalize(normal)
    if float(np.dot(normal, WORLD_X)) > 0.0:
        normal = -normal
    return normal


def horizontal_normal(outward_normal):
    normal = orient_outward_normal(outward_normal)
    return normalize(np.array([normal[0], normal[1], 0.0]))


def nominal_base_R_tag(outward_normal_xy):
    """Keep the brush cylinder vertical; yaw is free because it is symmetric."""
    tag_z = WORLD_Z
    tag_x = normalize(outward_normal_xy)
    tag_y = normalize(np.cross(tag_z, tag_x))
    return np.column_stack((tag_x, tag_y, tag_z))


def segment_tangent_xy(segment, base_R_tag):
    tangent = segment.get("tangent_unit")
    if tangent is not None:
        tangent = np.asarray(tangent, dtype=float).reshape(3)
    else:
        tangent = base_R_tag[:, 1]
    tangent = np.array([tangent[0], tangent[1], 0.0], dtype=float)
    if np.linalg.norm(tangent) <= 1e-12:
        tangent = np.array([base_R_tag[0, 1], base_R_tag[1, 1], 0.0], dtype=float)
    tangent = normalize(tangent)
    if float(np.dot(tangent, WORLD_X)) < 0.0:
        tangent = -tangent
    return tangent


def rotate_about_tag_z(base_R_tag, angle_rad):
    local_rotation = Rotation.from_rotvec(
        float(angle_rad) * np.array([0.0, 0.0, 1.0])
    ).as_matrix()
    return base_R_tag @ local_rotation


def brush_center_to_flange_transform(
    brush_center_base,
    base_R_tag,
    flange_T_tag,
):
    tag_origin_base = (
        np.asarray(brush_center_base, dtype=float)
        - base_R_tag @ TAG_P_BRUSH_CENTER_M
    )

    base_T_tag = np.eye(4, dtype=float)
    base_T_tag[:3, :3] = base_R_tag
    base_T_tag[:3, 3] = tag_origin_base

    base_T_flange = base_T_tag @ np.linalg.inv(flange_T_tag)
    return base_T_flange, base_T_tag


def build_pose_set(center, outward_normal, base_R_tag, flange_T_tag, pre_offset_m):
    contact_center = (
        center + (BRUSH_RADIUS_M - PRESS_DEPTH_M) * outward_normal
    )
    pre_center = contact_center + pre_offset_m * outward_normal

    pre_T_flange, _ = brush_center_to_flange_transform(
        pre_center, base_R_tag, flange_T_tag
    )
    contact_T_flange, _ = brush_center_to_flange_transform(
        contact_center, base_R_tag, flange_T_tag
    )
    _, base_T_tag = brush_center_to_flange_transform(
        contact_center, base_R_tag, flange_T_tag
    )

    return {
        "pre_center": pre_center,
        "contact_center": contact_center,
        "pre_endpose": transform_to_endpose(pre_T_flange),
        "contact_endpose": transform_to_endpose(contact_T_flange),
        "base_T_tag": base_T_tag,
    }


def endpose_for_brush_center(brush_center, base_R_tag, flange_T_tag):
    base_T_flange, _ = brush_center_to_flange_transform(
        brush_center,
        base_R_tag,
        flange_T_tag,
    )
    return transform_to_endpose(base_T_flange)


def add_sweep_poses(segment, poses, base_R_tag, flange_T_tag):
    tangent = segment_tangent_xy(segment, base_R_tag)
    xoy_length = float(segment.get("xoy_length", 0.0))
    half_length = max(0.0, xoy_length) / 2.0
    contact_center = np.asarray(poses["contact_center"], dtype=float)
    start_center = contact_center - tangent * half_length
    end_center = contact_center + tangent * half_length

    poses["sweep_tangent_unit"] = tangent
    poses["sweep_xoy_length_m"] = xoy_length
    poses["sweep_start_center"] = start_center
    poses["sweep_end_center"] = end_center
    poses["start_endpose"] = endpose_for_brush_center(
        start_center,
        base_R_tag,
        flange_T_tag,
    )
    poses["end_endpose"] = endpose_for_brush_center(
        end_center,
        base_R_tag,
        flange_T_tag,
    )
    return poses


def make_ik_context():
    model = ik.load_arm_model()
    if not model.existFrame(ik.DEFAULT_EE_FRAME):
        raise RuntimeError(f"Cannot find EE frame: {ik.DEFAULT_EE_FRAME}")

    frame_id = model.getFrameId(ik.DEFAULT_EE_FRAME)
    data = model.createData()
    joint_lb, joint_ub = ik.get_safe_bounds(model)
    neutral = np.clip(pin.neutral(model), joint_lb, joint_ub)
    return model, data, frame_id, joint_lb, joint_ub, neutral


def pose_residual(model, data, frame_id, q, target):
    actual = ik.frame_pose(model, data, q, frame_id)
    return np.concatenate(
        (
            (actual.translation - target[:3, 3]) / ik.POS_SCALE,
            pin.log3(target[:3, :3].T @ actual.rotation) / ik.ROT_SCALE,
        )
    )


def search_reachable_contact(center, outward_normal, flange_T_tag):
    """Search contact first. Only contact failure skips the segment."""
    model, data, frame_id, joint_lb, joint_ub, neutral = make_ik_context()
    nominal_rotation = nominal_base_R_tag(outward_normal)
    nq = model.nq
    rng = np.random.default_rng(CONTINUOUS_SEARCH_SEED)

    def unpack(x):
        return x[:nq], x[-1]

    def target_pose(angle_rad):
        base_R_tag = rotate_about_tag_z(nominal_rotation, angle_rad)
        poses = build_pose_set(
            center,
            outward_normal,
            base_R_tag,
            flange_T_tag,
            DEFAULT_PRE_OFFSET_M,
        )
        target = endpose_to_transform(poses["contact_endpose"])
        return base_R_tag, poses, target

    def residual(x):
        q_contact, angle_rad = unpack(x)
        _, _, target = target_pose(angle_rad)
        return pose_residual(
            model, data, frame_id, q_contact, target
        )

    lower = np.concatenate((joint_lb, [0.0]))
    upper = np.concatenate((joint_ub, [2.0 * np.pi]))

    starts = [np.concatenate((neutral, [0.0]))]
    for _ in range(CONTINUOUS_SEARCH_STARTS - 1):
        starts.append(
            np.concatenate(
                (
                    rng.uniform(joint_lb, joint_ub),
                    [rng.uniform(0.0, 2.0 * np.pi)],
                )
            )
        )

    best = None
    for x0 in starts:
        solution = least_squares(
            residual,
            x0,
            bounds=(lower, upper),
            max_nfev=1500,
            xtol=1e-9,
            ftol=1e-9,
            gtol=1e-9,
        )
        q_contact, angle_rad = unpack(solution.x)
        base_R_tag, poses, target = target_pose(angle_rad)
        result = continuous_result(
            model,
            data,
            frame_id,
            q_contact,
            target,
            poses["contact_endpose"],
        )
        candidate = {
            "roll_deg": float(np.rad2deg(angle_rad) % 360.0),
            "base_R_tag": base_R_tag,
            "poses": poses,
            "contact_result": result,
            "q_contact": q_contact,
            "score": result["pos_err_mm"] + result["rot_err_deg"],
        }

        if best is None or candidate["score"] < best["score"]:
            best = candidate
        if result["reachable"]:
            return candidate

    return None


def search_reachable_pre(
    center,
    outward_normal,
    flange_T_tag,
    base_R_tag,
    q_contact,
):
    """Search pre-app after contact succeeds, using contact as the first seed."""
    model, data, frame_id, joint_lb, joint_ub, neutral = make_ik_context()
    nq = model.nq
    rng = np.random.default_rng(CONTINUOUS_SEARCH_SEED + 1)
    joint_range = np.maximum(joint_ub - joint_lb, 1e-6)

    def unpack(x):
        return x[:nq], x[-1]

    def target_pose(pre_offset_m):
        poses = build_pose_set(
            center,
            outward_normal,
            base_R_tag,
            flange_T_tag,
            pre_offset_m,
        )
        target = endpose_to_transform(poses["pre_endpose"])
        return poses, target

    def residual(x):
        q_pre, pre_offset_m = unpack(x)
        _, target = target_pose(pre_offset_m)
        pose_error = pose_residual(
            model, data, frame_id, q_pre, target
        )
        offset_preference = (
            (pre_offset_m - DEFAULT_PRE_OFFSET_M)
            / (PRE_OFFSET_MAX_M - PRE_OFFSET_MIN_M)
        )
        continuity = (q_pre - q_contact) / joint_range
        return np.concatenate(
            (
                pose_error,
                [0.05 * offset_preference],
                0.02 * continuity,
            )
        )

    lower = np.concatenate((joint_lb, [PRE_OFFSET_MIN_M]))
    upper = np.concatenate((joint_ub, [PRE_OFFSET_MAX_M]))

    starts = [
        np.concatenate((q_contact, [DEFAULT_PRE_OFFSET_M])),
        np.concatenate((neutral, [DEFAULT_PRE_OFFSET_M])),
    ]
    for _ in range(CONTINUOUS_SEARCH_STARTS - len(starts)):
        starts.append(
            np.concatenate(
                (
                    rng.uniform(joint_lb, joint_ub),
                    [rng.uniform(PRE_OFFSET_MIN_M, PRE_OFFSET_MAX_M)],
                )
            )
        )

    best = None
    for x0 in starts:
        solution = least_squares(
            residual,
            x0,
            bounds=(lower, upper),
            max_nfev=1500,
            xtol=1e-9,
            ftol=1e-9,
            gtol=1e-9,
        )
        q_pre, pre_offset_m = unpack(solution.x)
        poses, target = target_pose(pre_offset_m)
        result = continuous_result(
            model,
            data,
            frame_id,
            q_pre,
            target,
            poses["pre_endpose"],
        )
        candidate = {
            "pre_offset_m": float(pre_offset_m),
            "poses": poses,
            "pre_result": result,
            "score": result["pos_err_mm"] + result["rot_err_deg"],
        }

        if best is None or candidate["score"] < best["score"]:
            best = candidate
        if result["reachable"]:
            return candidate

    return best

def continuous_result(model, data, frame_id, q, target, target_endpose):
    actual = ik.frame_pose(model, data, q, frame_id)
    pos_err_mm = np.linalg.norm(actual.translation - target[:3, 3]) * 1000.0
    rot_err_deg = np.rad2deg(
        np.linalg.norm(pin.log3(target[:3, :3].T @ actual.rotation))
    )
    return {
        "reachable": bool(
            pos_err_mm < ik.POS_TOL_MM and rot_err_deg < ik.ROT_TOL_DEG
        ),
        "joint_degrees": np.rad2deg(q).tolist(),
        "target_endpose": target_endpose,
        "pos_err_mm": float(pos_err_mm),
        "rot_err_deg": float(rot_err_deg),
    }


def solve_pose_with_seed(model, data, frame_id, joint_lb, joint_ub, seed_q, target_endpose):
    target = endpose_to_transform(target_endpose)

    def residual(q):
        return pose_residual(model, data, frame_id, q, target)

    solution = least_squares(
        residual,
        np.clip(seed_q, joint_lb, joint_ub),
        bounds=(joint_lb, joint_ub),
        max_nfev=1200,
        xtol=1e-9,
        ftol=1e-9,
        gtol=1e-9,
    )
    q = solution.x
    return q, continuous_result(
        model,
        data,
        frame_id,
        q,
        target,
        target_endpose,
    )


def check_pose_reachable(target_endpose, seed_q=None):
    model, data, frame_id, joint_lb, joint_ub, neutral = make_ik_context()
    if seed_q is None:
        seed_q = neutral
    q, result = solve_pose_with_seed(
        model,
        data,
        frame_id,
        joint_lb,
        joint_ub,
        seed_q,
        target_endpose,
    )
    return q, result


def find_reachable_sweep_endpoints(poses, base_R_tag, flange_T_tag, q_contact):
    start_center = np.asarray(poses["sweep_start_center"], dtype=float)
    end_center = np.asarray(poses["sweep_end_center"], dtype=float)
    contact_center = np.asarray(poses["contact_center"], dtype=float)
    tangent = normalize(np.asarray(poses["sweep_tangent_unit"], dtype=float))

    start_q, start_result = check_pose_reachable(poses["start_endpose"], q_contact)
    end_q, end_result = check_pose_reachable(poses["end_endpose"], q_contact)

    def search_one_side(sign, full_center, seed_q):
        full_distance = float(np.linalg.norm(full_center - contact_center))
        if full_distance <= 1e-9:
            endpose = endpose_for_brush_center(
                contact_center,
                base_R_tag,
                flange_T_tag,
            )
            q, result = check_pose_reachable(endpose, seed_q)
            return contact_center, endpose, q, result, 0.0

        sample_count = max(2, int(np.ceil(full_distance / SWEEP_SAMPLE_STEP_M)) + 1)
        distances = np.linspace(full_distance, 0.0, sample_count)
        best = None
        for distance in distances:
            center = contact_center + sign * tangent * distance
            endpose = endpose_for_brush_center(
                center,
                base_R_tag,
                flange_T_tag,
            )
            q, result = check_pose_reachable(endpose, seed_q)
            if result["reachable"]:
                best = (center, endpose, q, result, float(distance))
                break
        if best is not None:
            return best

        center = contact_center.copy()
        endpose = endpose_for_brush_center(center, base_R_tag, flange_T_tag)
        q, result = check_pose_reachable(endpose, seed_q)
        return center, endpose, q, result, 0.0

    if start_result["reachable"]:
        reachable_start = (
            start_center,
            poses["start_endpose"],
            start_q,
            start_result,
            float(np.linalg.norm(start_center - contact_center)),
        )
    else:
        reachable_start = search_one_side(-1.0, start_center, q_contact)

    if end_result["reachable"]:
        reachable_end = (
            end_center,
            poses["end_endpose"],
            end_q,
            end_result,
            float(np.linalg.norm(end_center - contact_center)),
        )
    else:
        reachable_end = search_one_side(1.0, end_center, q_contact)

    return {
        "requested_start_result": start_result,
        "requested_end_result": end_result,
        "reachable_start": reachable_start,
        "reachable_end": reachable_end,
    }


def check_move_line_reachability(start_center, end_center, base_R_tag, flange_T_tag, q_seed):
    distance = float(np.linalg.norm(end_center - start_center))
    sample_count = max(2, int(np.ceil(distance / SWEEP_SAMPLE_STEP_M)) + 1)
    centers = np.linspace(start_center, end_center, sample_count)

    model, data, frame_id, joint_lb, joint_ub, neutral = make_ik_context()
    q_prev = np.asarray(q_seed, dtype=float)
    samples = []
    reachable = True

    for index, center in enumerate(centers):
        endpose = endpose_for_brush_center(center, base_R_tag, flange_T_tag)
        q, result = solve_pose_with_seed(
            model,
            data,
            frame_id,
            joint_lb,
            joint_ub,
            q_prev,
            endpose,
        )
        q_prev = q
        sample = {
            "index": int(index),
            "center_base_m": np.round(center, 9).tolist(),
            "endpose": endpose,
            "reachable": bool(result["reachable"]),
            "joint_degrees": joint_degrees(result),
            "pos_err_mm": round(float(result["pos_err_mm"]), 6),
            "rot_err_deg": round(float(result["rot_err_deg"]), 6),
        }
        samples.append(sample)
        if not result["reachable"]:
            reachable = False
            break

    return {
        "reachable": bool(reachable),
        "sample_step_mm": SWEEP_SAMPLE_STEP_M * 1000.0,
        "sample_count": len(samples),
        "line_length_mm": distance * 1000.0,
        "samples": samples,
    }


def joint_degrees(result):
    return [
        round(float(value), 3)
        for value in result["joint_degrees"]
    ]


def build_record(
    segment,
    normal,
    roll_deg,
    pre_offset_m,
    poses,
    contact_result=None,
    pre_result=None,
    sweep_result=None,
    move_line_result=None,
    ik_checked=True,
):
    contact_reachable = (
        None
        if not ik_checked
        else bool(contact_result is not None and contact_result["reachable"])
    )
    pre_reachable = (
        None
        if not ik_checked
        else bool(pre_result is not None and pre_result["reachable"])
    )

    return {
        "segment_id": int(segment["id"]),
        "segment_center_base_m": segment["center_point_base_m"],
        "segment_tangent_unit": segment.get("tangent_unit"),
        "sweep_tangent_unit": (
            None
            if "sweep_tangent_unit" not in poses
            else np.round(poses["sweep_tangent_unit"], 9).tolist()
        ),
        "sweep_xoy_length_mm": round(float(poses.get("sweep_xoy_length_m", 0.0)) * 1000.0, 3),
        "outward_normal_unit": np.round(normal, 9).tolist(),
        "ik_checked": bool(ik_checked),
        "contact_reachable": contact_reachable,
        "pre_app_reachable": pre_reachable,
        "tag_z_search_angle_deg": round(float(roll_deg), 6),
        "pre_app_offset_mm": round(float(pre_offset_m) * 1000.0, 3),
        "brush_center_pre_base_m": np.round(
            poses["pre_center"], 9
        ).tolist(),
        "brush_center_contact_base_m": np.round(
            poses["contact_center"], 9
        ).tolist(),
        "pre_app_endpose": poses["pre_endpose"],
        "contact_endpose": poses["contact_endpose"],
        "start_endpose": poses.get("start_endpose"),
        "end_endpose": poses.get("end_endpose"),
        "reachable_start_endpose": (
            None
            if sweep_result is None
            else sweep_result["reachable_start"][1]
        ),
        "reachable_end_endpose": (
            None
            if sweep_result is None
            else sweep_result["reachable_end"][1]
        ),
        "pre_app_joint_degrees": (
            None
            if pre_result is None or not pre_result["reachable"]
            else joint_degrees(pre_result)
        ),
        "contact_joint_degrees": (
            None
            if contact_result is None or not contact_result["reachable"]
            else joint_degrees(contact_result)
        ),
        "start_joint_degrees": (
            None
            if sweep_result is None
            else joint_degrees(sweep_result["reachable_start"][3])
        ),
        "end_joint_degrees": (
            None
            if sweep_result is None
            else joint_degrees(sweep_result["reachable_end"][3])
        ),
        "move_line_reachability": move_line_result,
        "pre_app_ik_error": (
            None
            if pre_result is None
            else {
                "pos_err_mm": round(float(pre_result["pos_err_mm"]), 6),
                "rot_err_deg": round(float(pre_result["rot_err_deg"]), 6),
            }
        ),
        "contact_ik_error": (
            None
            if contact_result is None
            else {
                "pos_err_mm": round(float(contact_result["pos_err_mm"]), 6),
                "rot_err_deg": round(float(contact_result["rot_err_deg"]), 6),
            }
        ),
        "base_R_tag": np.round(
            poses["base_T_tag"][:3, :3], 9
        ).tolist(),
    }

def main():
    args = resolve_paths(parse_args())

    brush_pick = json.loads(
        args.brush_pick_json.read_text(encoding="utf-8")
    )
    segment_data = json.loads(
        args.segments_json.read_text(encoding="utf-8")
    )
    alignment = json.loads(
        args.alignment_json.read_text(encoding="utf-8")
    )
    delta_T_base = np.asarray(
        alignment["delta_T_base"], dtype=float
    ).reshape(4, 4)
    segments = [
        transform_segment_by_delta(segment, delta_T_base)
        for segment in segment_data["segments"]
    ]

    flange_T_tag = compute_flange_T_tag(brush_pick)

    if not args.no_ik:
        ik.DEFAULT_URDF = str(args.urdf.resolve())
        ik._MODEL_CACHE = None

    records = []
    skipped_segments = []
    pre_app_failed_segment_ids = []

    for segment in segments:
        segment_id = int(segment["id"])
        center = np.asarray(segment["center_point_base_m"], dtype=float)
        normal = horizontal_normal(segment["outward_normal_unit"])

        if args.no_ik:
            base_R_tag = nominal_base_R_tag(normal)
            poses = build_pose_set(
                center,
                normal,
                base_R_tag,
                flange_T_tag,
                DEFAULT_PRE_OFFSET_M,
            )
            poses = add_sweep_poses(segment, poses, base_R_tag, flange_T_tag)
            records.append(
                build_record(
                    segment=segment,
                    normal=normal,
                    roll_deg=0.0,
                    pre_offset_m=DEFAULT_PRE_OFFSET_M,
                    poses=poses,
                    move_line_result={
                        "reachable": None,
                        "reason": "IK not checked",
                    },
                    ik_checked=False,
                )
            )
            print(f"Segment {segment_id}: IK not checked")
            continue

        contact_candidate = search_reachable_contact(
            center,
            normal,
            flange_T_tag,
        )

        if contact_candidate is None:
            skipped_segments.append(
                {
                    "segment_id": segment_id,
                    "reason": "contact fail",
                }
            )
            print(f"Segment {segment_id}: contact fail -> skipped")
            continue

        pre_candidate = search_reachable_pre(
            center=center,
            outward_normal=normal,
            flange_T_tag=flange_T_tag,
            base_R_tag=contact_candidate["base_R_tag"],
            q_contact=contact_candidate["q_contact"],
        )

        pre_reachable = bool(
            pre_candidate is not None
            and pre_candidate["pre_result"]["reachable"]
        )

        if pre_candidate is None:
            pre_offset_m = DEFAULT_PRE_OFFSET_M
            poses = contact_candidate["poses"]
            pre_result = None
        else:
            pre_offset_m = pre_candidate["pre_offset_m"]
            poses = pre_candidate["poses"]
            pre_result = pre_candidate["pre_result"]

        if pre_reachable:
            print(
                f"Segment {segment_id}: contact success, "
                "pre app success"
            )
        else:
            pre_app_failed_segment_ids.append(segment_id)
            print(
                f"Segment {segment_id}: pre app fail -> "
                "contact retained"
            )

        poses = add_sweep_poses(
            segment,
            poses,
            contact_candidate["base_R_tag"],
            flange_T_tag,
        )
        sweep_result = find_reachable_sweep_endpoints(
            poses,
            contact_candidate["base_R_tag"],
            flange_T_tag,
            contact_candidate["q_contact"],
        )
        start_center, _, q_start, start_result, _ = sweep_result["reachable_start"]
        end_center, _, _, end_result, _ = sweep_result["reachable_end"]
        if start_result["reachable"] and end_result["reachable"]:
            move_line_result = check_move_line_reachability(
                start_center,
                end_center,
                contact_candidate["base_R_tag"],
                flange_T_tag,
                q_start,
            )
        else:
            move_line_result = {
                "reachable": False,
                "reason": "reachable start or end point was not found",
                "requested_start_reachable": bool(
                    sweep_result["requested_start_result"]["reachable"]
                ),
                "requested_end_reachable": bool(
                    sweep_result["requested_end_result"]["reachable"]
                ),
            }

        records.append(
            build_record(
                segment=segment,
                normal=normal,
                roll_deg=contact_candidate["roll_deg"],
                pre_offset_m=pre_offset_m,
                poses=poses,
                contact_result=contact_candidate["contact_result"],
                pre_result=pre_result,
                sweep_result=sweep_result,
                move_line_result=move_line_result,
                ik_checked=True,
            )
        )

    skipped_segment_ids = [
        item["segment_id"] for item in skipped_segments
    ]

    output = {
        "coordinate_frame": "base",
        "brush_radius_mm": BRUSH_RADIUS_M * 1000.0,
        "press_depth_mm": PRESS_DEPTH_M * 1000.0,
        "default_pre_app_offset_mm": DEFAULT_PRE_OFFSET_M * 1000.0,
        "pre_app_offset_search_range_mm": [
            PRE_OFFSET_MIN_M * 1000.0,
            PRE_OFFSET_MAX_M * 1000.0,
        ],
        "tag_p_brush_center_mm": (
            TAG_P_BRUSH_CENTER_M * 1000.0
        ).tolist(),
        "brush_axis_definition": (
            "Tag +Z = brush cylinder axis = base +Z"
        ),
        "orientation_search": (
            "contact-first continuous yaw search about Tag +Z"
        ),
        "pre_app_search": (
            "after contact succeeds, search distance along the "
            "horizontal outward normal using contact joints as the first seed"
        ),
        "urdf": str(args.urdf),
        "flange_T_tag": np.round(flange_T_tag, 9).tolist(),
        "input_segment_count": len(segments),
        "generated_segment_count": len(records),
        "full_pose_set_count": sum(
            item.get("pre_app_reachable") is True
            for item in records
        ),
        "contact_only_segment_count": sum(
            item.get("contact_reachable") is True
            and item.get("pre_app_reachable") is False
            for item in records
        ),
        "skipped_segment_ids": skipped_segment_ids,
        "skipped_segments": skipped_segments,
        "pre_app_failed_segment_ids": pre_app_failed_segment_ids,
        "segments": records,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Input brush segments: {len(segments)}")
    print(f"Generated contact segments: {len(records)}")
    print(
        "Pre-app failed but contact retained: "
        f"{pre_app_failed_segment_ids}"
    )
    print(f"Skipped segments with reasons: {skipped_segments}")
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()

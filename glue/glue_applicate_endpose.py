"""Compute flange endposes for glue dispensing dots."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial.transform import Rotation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TAG_P_NEEDLE_M = np.array([0.118, -0.049, 0.038], dtype=float)
PRESS_DEPTH_M = 0.002
PRE_APPLICATE_OFFSET_M = 0.1
PRE_APPLICATE_MAX_OFFSET_M = 0.25
PRE_APPLICATE_SEARCH_STEP_M = 0.005
WORLD_X = np.array([1.0, 0.0, 0.0], dtype=float)
WORLD_Z = np.array([0.0, 0.0, 1.0], dtype=float)
ROLL_SEARCH_XATOL_DEG = 0.1
URDF = '/home/smmg/AAM/config/piper/piper_description.urdf'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Run directory; reads generated data and writes output into pickplace/.",
    )
    parser.add_argument("--servo-pick-json", type=Path)
    parser.add_argument("--dots-npz", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--urdf", type=Path, default = URDF, help="Piper URDF; required unless --no-ik is used.")
    parser.add_argument("--no-ik", action="store_true", help="Skip IK and save joint_degrees as null.")
    return parser.parse_args()


def resolve_data_paths(args):
    if args.run_dir is not None:
        run_dir = args.run_dir.expanduser().resolve()
        pickplace_dir = run_dir / "pickplace"
        args.servo_pick_json = (
            args.servo_pick_json or pickplace_dir / "glue_pick_endpose.json"
        )
        args.dots_npz = (
            args.dots_npz
            or run_dir / "completion" / "depression" / "glue_applicate_dots.npz"
        )
        args.output = args.output or pickplace_dir / "glue_applicate_endpose.json"

    missing = [
        name
        for name in ("servo_pick_json", "dots_npz", "output")
        if getattr(args, name) is None
    ]
    if missing:
        raise ValueError(
            "--run-dir or explicit data paths are required; missing: " + ", ".join(missing)
        )
    return args


def normalize(vector):
    vector = np.asarray(vector, dtype=float).reshape(3)
    norm = np.linalg.norm(vector)
    if norm <= 1e-12:
        raise ValueError("Cannot normalize a zero vector.")
    return vector / norm


def endpose_to_transform(endpose):
    endpose = np.asarray(endpose, dtype=float).reshape(6)
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = Rotation.from_euler("xyz", endpose[3:], degrees=True).as_matrix()
    transform[:3, 3] = endpose[:3] / 1000.0
    return transform


def transform_to_endpose(transform):
    position_mm = transform[:3, 3] * 1000.0
    rpy_deg = Rotation.from_matrix(transform[:3, :3]).as_euler("xyz", degrees=True)
    return np.concatenate((position_mm, rpy_deg))


def desired_base_R_tag(fix_normal):
    """Tag +X follows inward normal projected to base XOY; Tag +Z is world +Z."""
    normal = normalize(fix_normal)
    tag_x = np.array([normal[0], normal[1], 0.0], dtype=float)
    if np.linalg.norm(tag_x) <= 1e-12:
        raise ValueError("Fix-point normal has no projection in base XOY.")
    tag_x = normalize(tag_x)
    if float(np.dot(tag_x, WORLD_X)) < 0.0:
        tag_x = -tag_x
    if float(np.dot(tag_x, WORLD_X)) <= 0.0:
        raise ValueError("Target AprilTag +X must have dot(tag_x, world_x) > 0.")

    tag_z = WORLD_Z.copy()
    tag_y = normalize(np.cross(tag_z, tag_x))
    tag_x = normalize(np.cross(tag_y, tag_z))
    return np.column_stack((tag_x, tag_y, tag_z))


def compute_fixed_flange_T_tag(servo_pick):
    base_T_tag_pick = np.asarray(servo_pick["base_T_tag"], dtype=float)
    if base_T_tag_pick.shape != (4, 4):
        raise ValueError("base_T_tag in servo-pick JSON must be 4x4.")
    base_T_flange_pick = endpose_to_transform(servo_pick["endpose"])
    return np.linalg.inv(base_T_flange_pick) @ base_T_tag_pick


def compute_dot_flange_transform(dot_base, normal_base, flange_T_tag, roll_deg=0.0):
    normal_base = normalize(normal_base)
    nominal_base_R_tag = desired_base_R_tag(normal_base)
    # Post-multiplication rotates around the AprilTag local +X axis.  Its +X
    # direction remains fixed while +Y/+Z can rotate through the full circle.
    tag_R_roll = Rotation.from_rotvec(
        np.deg2rad(float(roll_deg)) * np.array([1.0, 0.0, 0.0])
    ).as_matrix()
    base_R_tag = nominal_base_R_tag @ tag_R_roll

    # Needle tip penetrates 2 mm along the inward local surface normal.
    needle_tip_base = np.asarray(dot_base, dtype=float).reshape(3) + PRESS_DEPTH_M * normal_base
    tag_origin_base = needle_tip_base - base_R_tag @ TAG_P_NEEDLE_M

    base_T_tag = np.eye(4, dtype=float)
    base_T_tag[:3, :3] = base_R_tag
    base_T_tag[:3, 3] = tag_origin_base
    base_T_flange = base_T_tag @ np.linalg.inv(flange_T_tag)

    return base_T_flange, base_T_tag, needle_tip_base


def pre_applicate_endpose(applicate_endpose, normal_base, offset_m=PRE_APPLICATE_OFFSET_M):
    pre_endpose = np.asarray(applicate_endpose, dtype=float).reshape(6).copy()
    normal_base = normalize(normal_base)
    pre_endpose[:3] -= normal_base * offset_m * 1000.0
    return [round(float(value), 6) for value in pre_endpose]


def find_reachable_pre_applicate(applicate_endpose, normal_base, ik):
    """Search a reachable pre-applicate point from 50 to 250 mm on the retreat ray."""
    best = None
    offsets = np.arange(
        PRE_APPLICATE_OFFSET_M,
        PRE_APPLICATE_MAX_OFFSET_M + 0.5 * PRE_APPLICATE_SEARCH_STEP_M,
        PRE_APPLICATE_SEARCH_STEP_M,
    )
    for offset_m in offsets:
        pre_endpose = pre_applicate_endpose(applicate_endpose, normal_base, offset_m)
        result = ik.reachability_test(pre_endpose)
        score = max(
            result["pos_err_mm"] / ik.POS_TOL_MM,
            result["rot_err_deg"] / ik.ROT_TOL_DEG,
        )
        candidate = {
            "endpose": pre_endpose,
            "result": result,
            "offset_m": float(offset_m),
            "score": float(score),
        }
        if best is None or score < best["score"]:
            best = candidate
        if result["reachable"]:
            return candidate
    return best


def solve_joint_degrees(endpose, urdf, pose_name="dispensing endpose"):
    if urdf is None:
        raise ValueError("--urdf is required unless --no-ik is used.")
    if not urdf.is_file():
        raise FileNotFoundError(urdf)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from Piper import endpose_reachability_safe as ik

    ik.DEFAULT_URDF = str(urdf.resolve())
    result = ik.reachability_test(endpose)
    if not result["reachable"]:
        raise RuntimeError(
            f"Unreachable {pose_name}: position error={result['pos_err_mm']:.3f} mm, "
            f"rotation error={result['rot_err_deg']:.3f} deg"
        )
    return [round(float(value), 3) for value in result["joint_degrees"]]


def search_reachable_roll(dot_base, normal_base, flange_T_tag, urdf, point_index):
    """Continuously search [0, 360) while keeping tip XYZ and Tag +X fixed."""
    if urdf is None:
        raise ValueError("--urdf is required unless --no-ik is used.")
    if not urdf.is_file():
        raise FileNotFoundError(urdf)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from Piper import endpose_reachability_safe as ik

    ik.DEFAULT_URDF = str(urdf.resolve())
    cache = {}
    best = None

    class ReachableRollFound(Exception):
        def __init__(self, candidate):
            self.candidate = candidate

    def evaluate(roll_deg):
        nonlocal best
        roll_deg = float(roll_deg) % 360.0
        cache_key = round(roll_deg, 8)
        if cache_key in cache:
            return cache[cache_key]

        base_T_flange, base_T_tag, needle_tip_base = compute_dot_flange_transform(
            dot_base, normal_base, flange_T_tag, roll_deg=roll_deg
        )
        endpose = [round(float(value), 6) for value in transform_to_endpose(base_T_flange)]
        applicate_result = ik.reachability_test(endpose)
        if applicate_result["reachable"]:
            pre_candidate = find_reachable_pre_applicate(endpose, normal_base, ik)
        else:
            pre_endpose = pre_applicate_endpose(endpose, normal_base)
            pre_result = ik.reachability_test(pre_endpose)
            pre_candidate = {
                "endpose": pre_endpose,
                "result": pre_result,
                "offset_m": PRE_APPLICATE_OFFSET_M,
                "score": max(
                    pre_result["pos_err_mm"] / ik.POS_TOL_MM,
                    pre_result["rot_err_deg"] / ik.ROT_TOL_DEG,
                ),
            }
        pre_endpose = pre_candidate["endpose"]
        pre_result = pre_candidate["result"]

        score = max(
            applicate_result["pos_err_mm"] / ik.POS_TOL_MM,
            applicate_result["rot_err_deg"] / ik.ROT_TOL_DEG,
            pre_candidate["score"],
        )
        candidate = {
            "roll_deg": roll_deg,
            "score": float(score),
            "base_T_tag": base_T_tag,
            "needle_tip_base": needle_tip_base,
            "endpose": endpose,
            "pre_endpose": pre_endpose,
            "pre_offset_m": pre_candidate["offset_m"],
            "joint_degrees": [round(float(v), 3) for v in applicate_result["joint_degrees"]],
            "pre_joint_degrees": [round(float(v), 3) for v in pre_result["joint_degrees"]],
            "applicate_result": applicate_result,
            "pre_result": pre_result,
        }
        cache[cache_key] = candidate
        if best is None or candidate["score"] < best["score"]:
            best = candidate
        if applicate_result["reachable"] and pre_result["reachable"]:
            raise ReachableRollFound(candidate)
        return candidate

    def objective(roll_deg):
        return evaluate(roll_deg)["score"]

    try:
        # Preserve the old overhead orientation when it is already reachable.
        evaluate(0.0)
        minimize_scalar(
            objective,
            bounds=(0.0, 360.0),
            method="bounded",
            options={"xatol": ROLL_SEARCH_XATOL_DEG},
        )
    except ReachableRollFound as found:
        return found.candidate

    app = best["applicate_result"]
    pre = best["pre_result"]
    print(
        f"Skip fix point {point_index}: no reachable roll in [0, 360); "
        f"best roll={best['roll_deg']:.3f} deg, "
        f"pre offset={best['pre_offset_m'] * 1000.0:.1f} mm, "
        f"applicate error=({app['pos_err_mm']:.3f} mm, {app['rot_err_deg']:.3f} deg), "
        f"pre error=({pre['pos_err_mm']:.3f} mm, {pre['rot_err_deg']:.3f} deg)"
    )
    return None


def main():
    args = resolve_data_paths(parse_args())
    for path in (args.servo_pick_json, args.dots_npz):
        if not path.is_file():
            raise FileNotFoundError(path)

    servo_pick = json.loads(args.servo_pick_json.read_text(encoding="utf-8"))
    dots_data = np.load(args.dots_npz, allow_pickle=True)
    points = np.asarray(dots_data["all_points"], dtype=float)
    normals = np.asarray(dots_data["all_normals"], dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"all_points must have shape (N, 3), got {points.shape}.")
    if normals.shape != points.shape:
        raise ValueError(f"all_normals shape {normals.shape} does not match all_points {points.shape}.")

    group_ids = (
        np.asarray(dots_data["dot_group_ids"], dtype=int)
        if "dot_group_ids" in dots_data.files
        else np.zeros(len(points), dtype=int)
    )
    flange_T_tag = compute_fixed_flange_T_tag(servo_pick)

    records = []
    skipped_unreachable = 0
    for index, (point, normal) in enumerate(zip(points, normals)):
        if args.no_ik:
            selected_roll_deg = 0.0
            selected_pre_offset_m = PRE_APPLICATE_OFFSET_M
            base_T_flange, base_T_tag, needle_tip_base = compute_dot_flange_transform(
                point, normal, flange_T_tag, roll_deg=selected_roll_deg
            )
            endpose = [round(float(value), 6) for value in transform_to_endpose(base_T_flange)]
            pre_endpose = pre_applicate_endpose(endpose, normal, selected_pre_offset_m)
            joint_degrees = None
            pre_joint_degrees = None
        else:
            candidate = search_reachable_roll(
                point, normal, flange_T_tag, args.urdf, point_index=index
            )
            if candidate is None:
                skipped_unreachable += 1
                continue
            selected_roll_deg = candidate["roll_deg"]
            selected_pre_offset_m = candidate["pre_offset_m"]
            base_T_tag = candidate["base_T_tag"]
            needle_tip_base = candidate["needle_tip_base"]
            endpose = candidate["endpose"]
            pre_endpose = candidate["pre_endpose"]
            joint_degrees = candidate["joint_degrees"]
            pre_joint_degrees = candidate["pre_joint_degrees"]
        records.append({
            "index": index,
            "group_id": int(group_ids[index]),
            "tag_x_roll_deg": round(float(selected_roll_deg), 6),
            "pre_applicate_offset_mm": round(selected_pre_offset_m * 1000.0, 3),
            "pre_applicate_endpose": pre_endpose,
            "pre_applicate_joint_degrees": pre_joint_degrees,
            "endpose": endpose,
            "joint_degrees": joint_degrees,
            "needle_tip_base_m": np.round(needle_tip_base, 9).tolist(),
            "base_R_tag": np.round(base_T_tag[:3, :3], 9).tolist(),
        })

    output = {
        "press_depth_mm": PRESS_DEPTH_M * 1000.0,
        "pre_applicate_min_offset_mm": PRE_APPLICATE_OFFSET_M * 1000.0,
        "pre_applicate_max_offset_mm": PRE_APPLICATE_MAX_OFFSET_M * 1000.0,
        "pre_applicate_search_step_mm": PRE_APPLICATE_SEARCH_STEP_M * 1000.0,
        "tag_p_needle_mm": (TAG_P_NEEDLE_M * 1000.0).tolist(),
        "flange_T_tag": np.round(flange_T_tag, 9).tolist(),
        "dot_endposes": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Generated dispensing endposes: {len(records)}")
    print(f"Skipped unreachable fix points: {skipped_unreachable}")
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()

"""Calculate two brush contact poses and reposition Mark1 when both initially fail."""

import argparse
import json
import sys
import threading
from pathlib import Path

import numpy as np
import pinocchio as pin
import rclpy
from rclpy.executors import MultiThreadedExecutor
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Piper import endpose_reachability_safe as ik
try:
    from Mark1.motion_ctrl import Mark1BaseController
except ImportError:
    from motion_ctrl import Mark1BaseController

DEFAULT_URDF = Path("/home/smmg/AAM/config/piper/piper_description.urdf")
TAG_P_BRUSH_CENTER_M = np.array([0.0, 0.0, -0.085])
BRUSH_RADIUS_M = 0.035
PRESS_DEPTH_M = 0.00
WORLD_X = np.array([1.0, 0.0, 0.0])
WORLD_Z = np.array([0.0, 0.0, 1.0])

ARM_STARTS = 16
RANDOM_SEED = 0

# n=1,2: x=5+5n cm -> 10,15 cm; y=5n cm -> 5,10 cm
MARK1_X_STARTS = [0.10, 0.15]
MARK1_Y_STARTS = [0.05, 0.10]
MARK1_X_BOUNDS = (0.05, 0.20)
MARK1_Y_BOUNDS = (0.00, 0.15)
MARK1_X_SPEED = 0.08
MARK1_Y_SPEED = 0.06


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--brush-pick-json", type=Path)
    parser.add_argument("--segments-json", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--execute-mark1", action="store_true")
    return parser.parse_args()


def resolve_paths(args):
    run_dir = args.run_dir.expanduser().resolve()
    args.brush_pick_json = args.brush_pick_json or run_dir / "pickplace" / "glue_brush_pick_endpose.json"
    args.segments_json = args.segments_json or run_dir / "completion" / "depression" / "glue_brush_adaptive_segments.json"
    args.output = args.output or run_dir / "pickplace" / "glue_applicate_endpose_brush_mark1.json"
    return args


def normalize(v):
    v = np.asarray(v, dtype=float).reshape(3)
    return v / np.linalg.norm(v)


def endpose_to_T(endpose):
    endpose = np.asarray(endpose, dtype=float).reshape(6)
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", endpose[3:], degrees=True).as_matrix()
    T[:3, 3] = endpose[:3] / 1000.0
    return T


def T_to_endpose(T):
    xyz = T[:3, 3] * 1000.0
    rpy = Rotation.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True)
    return [round(float(v), 6) for v in np.r_[xyz, rpy]]


def flange_T_tag_from_pick(brush_pick):
    base_T_tag = np.asarray(brush_pick["base_T_tag"], dtype=float)
    base_T_flange = endpose_to_T(brush_pick["endpose"])
    return np.linalg.inv(base_T_flange) @ base_T_tag


def horizontal_outward(normal):
    normal = normalize(normal)
    if np.dot(normal, WORLD_X) > 0.0:
        normal = -normal
    return normalize([normal[0], normal[1], 0.0])


def nominal_R_tag(normal):
    tag_x = normalize(normal)
    tag_y = normalize(np.cross(WORLD_Z, tag_x))
    return np.column_stack((tag_x, tag_y, WORLD_Z))


def rotate_tag_z(R_tag, angle):
    Rz = Rotation.from_rotvec(angle * np.array([0.0, 0.0, 1.0])).as_matrix()
    return R_tag @ Rz


def build_contact(center, normal, R_tag, flange_T_tag):
    brush_center = center + (BRUSH_RADIUS_M - PRESS_DEPTH_M) * normal
    tag_origin = brush_center - R_tag @ TAG_P_BRUSH_CENTER_M

    base_T_tag = np.eye(4)
    base_T_tag[:3, :3] = R_tag
    base_T_tag[:3, 3] = tag_origin
    base_T_flange = base_T_tag @ np.linalg.inv(flange_T_tag)

    return {
        "brush_center": brush_center,
        "base_T_tag": base_T_tag,
        "endpose": T_to_endpose(base_T_flange),
    }


def load_ik(urdf):
    ik.DEFAULT_URDF = str(urdf.resolve())
    ik._MODEL_CACHE = None
    model = ik.load_arm_model()
    frame_id = model.getFrameId(ik.DEFAULT_EE_FRAME)
    data = model.createData()
    lb, ub = ik.get_safe_bounds(model)
    neutral = np.clip(pin.neutral(model), lb, ub)
    return model, data, frame_id, lb, ub, neutral


def pose_error(model, data, frame_id, q, target):
    actual = ik.frame_pose(model, data, q, frame_id)
    return np.r_[
        (actual.translation - target[:3, 3]) / ik.POS_SCALE,
        pin.log3(target[:3, :3].T @ actual.rotation) / ik.ROT_SCALE,
    ]


def evaluate_contact(model, data, frame_id, q, target, pose):
    actual = ik.frame_pose(model, data, q, frame_id)
    pos_err = np.linalg.norm(actual.translation - target[:3, 3]) * 1000.0
    rot_err = np.rad2deg(np.linalg.norm(pin.log3(target[:3, :3].T @ actual.rotation)))
    return {
        "reachable": bool(pos_err < ik.POS_TOL_MM and rot_err < ik.ROT_TOL_DEG),
        "joint_degrees": np.rad2deg(q).tolist(),
        "contact_endpose": pose["endpose"],
        "contact_center_base_m": pose["brush_center"].tolist(),
        "base_R_tag": pose["base_T_tag"][:3, :3].tolist(),
        "pos_err_mm": float(pos_err),
        "rot_err_deg": float(rot_err),
    }


def transformed_segment(segment, base_delta):
    center = np.asarray(segment["center_point_base_m"], dtype=float)
    center = center - np.array([base_delta[0], base_delta[1], 0.0])
    normal = horizontal_outward(segment["outward_normal_unit"])
    return center, normal


def search_one_contact(center, normal, flange_T_tag, ctx, seed_q=None, seed_angle=0.0):
    model, data, frame_id, lb, ub, neutral = ctx
    nq = model.nq
    nominal = nominal_R_tag(normal)
    rng = np.random.default_rng(RANDOM_SEED)

    def target(angle):
        pose = build_contact(center, normal, rotate_tag_z(nominal, angle), flange_T_tag)
        return pose, endpose_to_T(pose["endpose"])

    def residual(x):
        q, angle = x[:nq], x[-1]
        _, target_T = target(angle)
        return pose_error(model, data, frame_id, q, target_T)

    q0 = neutral if seed_q is None else np.clip(seed_q, lb, ub)
    starts = [np.r_[q0, seed_angle % (2.0 * np.pi)]]
    starts += [np.r_[rng.uniform(lb, ub), rng.uniform(0.0, 2.0 * np.pi)] for _ in range(ARM_STARTS - 1)]
    bounds = (np.r_[lb, 0.0], np.r_[ub, 2.0 * np.pi])

    best = None
    for x0 in starts:
        solution = least_squares(residual, x0, bounds=bounds, max_nfev=1500, xtol=1e-9, ftol=1e-9, gtol=1e-9)
        q, angle = solution.x[:nq], solution.x[-1]
        pose, target_T = target(angle)
        result = evaluate_contact(model, data, frame_id, q, target_T, pose)
        candidate = {
            "q": q,
            "angle": float(angle),
            "result": result,
            "score": result["pos_err_mm"] + result["rot_err_deg"],
        }
        if best is None or candidate["score"] < best["score"]:
            best = candidate
        if result["reachable"]:
            return candidate
    return best


def search_mark1(segments, flange_T_tag, ctx, initial_contacts):
    model, data, frame_id, lb, ub, neutral = ctx
    nq = model.nq
    q_seed = [item["q"] if item is not None else neutral for item in initial_contacts]
    a_seed = [item["angle"] if item is not None else 0.0 for item in initial_contacts]

    def unpack(x):
        dx, dy = x[:2]
        q0 = x[2:2 + nq]
        a0 = x[2 + nq]
        q1_start = 3 + nq
        q1 = x[q1_start:q1_start + nq]
        a1 = x[-1]
        return dx, dy, (q0, q1), (a0, a1)

    def targets(dx, dy, angles):
        values = []
        for segment, angle in zip(segments, angles):
            center, normal = transformed_segment(segment, [dx, dy])
            pose = build_contact(center, normal, rotate_tag_z(nominal_R_tag(normal), angle), flange_T_tag)
            values.append((center, normal, pose, endpose_to_T(pose["endpose"])))
        return values

    def residual(x):
        dx, dy, qs, angles = unpack(x)
        values = targets(dx, dy, angles)
        arm_errors = [pose_error(model, data, frame_id, q, value[3]) for q, value in zip(qs, values)]
        move_penalty = [0.02 * dx / MARK1_X_BOUNDS[1], 0.02 * dy / MARK1_Y_BOUNDS[1]]
        return np.concatenate((*arm_errors, move_penalty))

    lower = np.r_[MARK1_X_BOUNDS[0], MARK1_Y_BOUNDS[0], lb, 0.0, lb, 0.0]
    upper = np.r_[MARK1_X_BOUNDS[1], MARK1_Y_BOUNDS[1], ub, 2.0 * np.pi, ub, 2.0 * np.pi]
    starts = [
        np.r_[dx, dy, q_seed[0], a_seed[0], q_seed[1], a_seed[1]]
        for dx in MARK1_X_STARTS for dy in MARK1_Y_STARTS
    ]

    best_one = None
    for index, x0 in enumerate(starts):
        solution = least_squares(residual, x0, bounds=(lower, upper), max_nfev=2200, xtol=1e-9, ftol=1e-9, gtol=1e-9)
        dx, dy, qs, angles = unpack(solution.x)
        values = targets(dx, dy, angles)
        contacts = []

        for q, angle, value in zip(qs, angles, values):
            result = evaluate_contact(model, data, frame_id, q, value[3], value[2])
            contacts.append({"q": q, "angle": float(angle), "result": result})

        reachable_count = sum(item["result"]["reachable"] for item in contacts)
        score = sum(item["result"]["pos_err_mm"] + item["result"]["rot_err_deg"] for item in contacts)
        candidate = {
            "start_index": index,
            "start_xy_m": [float(x0[0]), float(x0[1])],
            "dx_m": float(dx),
            "dy_m": float(dy),
            "reachable_count": int(reachable_count),
            "score": float(score),
            "move_norm_m": float(np.hypot(dx, dy)),
            "contacts": contacts,
        }

        print(f"Mark1 start {index}: dx={dx:.4f}, dy={dy:.4f}, reachable={reachable_count}/2")
        if reachable_count == 2:
            return candidate
        if reachable_count == 1:
            if best_one is None or (score, candidate["move_norm_m"]) < (best_one["score"], best_one["move_norm_m"]):
                best_one = candidate

    return best_one


def execute_mark1(dx, dy):
    rclpy.init()
    base = Mark1BaseController()
    executor = MultiThreadedExecutor()
    executor.add_node(base)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()

    base.wait_for_odom()
    start = np.array([base.x, base.y, base.yaw], dtype=float)
    if abs(dx) > 1e-4:
        base.move_x(dx, speed_mps=MARK1_X_SPEED)
    if abs(dy) > 1e-4:
        base.move_y(dy, speed_mps=MARK1_Y_SPEED)
    base.wait_for_odom()
    end = np.array([base.x, base.y, base.yaw], dtype=float)

    base.stop()
    executor.shutdown()
    thread.join(timeout=1.0)
    base.destroy_node()
    rclpy.shutdown()

    delta_odom = end[:2] - start[:2]
    c, s = np.cos(start[2]), np.sin(start[2])
    delta_base = np.array([[c, s], [-s, c]]) @ delta_odom
    return {
        "start_odom": start.tolist(),
        "end_odom": end.tolist(),
        "actual_delta_base_m": delta_base.tolist(),
    }


def contact_record(segment, center, contact):
    result = contact["result"]
    return {
        "segment_id": int(segment["id"]),
        "original_center_base_m": segment["center_point_base_m"],
        "transformed_center_base_m": np.round(center, 9).tolist(),
        "reachable": bool(result["reachable"]),
        "contact_endpose": result["contact_endpose"],
        "contact_joint_degrees": [round(float(v), 3) for v in result["joint_degrees"]],
        "tag_z_search_angle_deg": round(float(np.rad2deg(contact["angle"]) % 360.0), 6),
        "contact_center_base_m": np.round(result["contact_center_base_m"], 9).tolist(),
        "outward_normal_unit": np.round(horizontal_outward(segment["outward_normal_unit"]), 9).tolist(),
        "pos_err_mm": round(float(result["pos_err_mm"]), 6),
        "rot_err_deg": round(float(result["rot_err_deg"]), 6),
        "base_R_tag": np.round(result["base_R_tag"], 9).tolist(),
    }


def result_name(count):
    return "both_contact_reachable" if count == 2 else "one_contact_reachable" if count == 1 else "contact_fail"


def main():
    args = resolve_paths(parse_args())
    brush_pick = json.loads(args.brush_pick_json.read_text(encoding="utf-8"))
    segment_data = json.loads(args.segments_json.read_text(encoding="utf-8"))
    segments = segment_data["segments"][:2]

    flange_T_tag = flange_T_tag_from_pick(brush_pick)
    ctx = load_ik(args.urdf)

    initial_contacts = []
    for segment in segments:
        center, normal = transformed_segment(segment, [0.0, 0.0])
        initial_contacts.append(search_one_contact(center, normal, flange_T_tag, ctx))

    initial_count = sum(item["result"]["reachable"] for item in initial_contacts)
    search_triggered = initial_count == 0
    selected_contacts = initial_contacts
    planned_delta = np.zeros(2)
    search_result = None

    if search_triggered:
        print("Both contacts fail. Start Mark1 continuous search.")
        search_result = search_mark1(segments, flange_T_tag, ctx, initial_contacts)
        if search_result is not None:
            planned_delta = np.array([search_result["dx_m"], search_result["dy_m"]])
            selected_contacts = search_result["contacts"]

    execution = None
    final_delta = planned_delta.copy()

    if args.execute_mark1 and search_triggered and search_result is not None:
        execution = execute_mark1(*planned_delta)
        final_delta = np.asarray(execution["actual_delta_base_m"], dtype=float)
        selected_contacts = []
        for segment, seed in zip(segments, search_result["contacts"]):
            center, normal = transformed_segment(segment, final_delta)
            selected_contacts.append(
                search_one_contact(center, normal, flange_T_tag, ctx, seed_q=seed["q"], seed_angle=seed["angle"])
            )

    records = []
    for segment, contact in zip(segments, selected_contacts):
        center, _ = transformed_segment(segment, final_delta)
        records.append(contact_record(segment, center, contact))

    reachable_count = sum(item["reachable"] for item in records)
    old_base_T_new_base = np.eye(4)
    old_base_T_new_base[:2, 3] = final_delta

    output = {
        "result": result_name(reachable_count),
        "input_contact_count": len(segments),
        "reachable_contact_count": int(reachable_count),
        "mark1_search_triggered": bool(search_triggered),
        "mark1_executed": bool(execution is not None),
        "mark1_initial_guesses_m": [[x, y] for x in MARK1_X_STARTS for y in MARK1_Y_STARTS],
        "mark1_search_bounds_m": {"x": list(MARK1_X_BOUNDS), "y": list(MARK1_Y_BOUNDS)},
        "planned_mark1_delta_base_m": planned_delta.tolist(),
        "final_mark1_delta_base_m": final_delta.tolist(),
        "old_base_T_new_base": np.round(old_base_T_new_base, 9).tolist(),
        "mark1_execution": execution,
        "brush_radius_mm": BRUSH_RADIUS_M * 1000.0,
        "press_depth_mm": PRESS_DEPTH_M * 1000.0,
        "tag_p_brush_center_mm": (TAG_P_BRUSH_CENTER_M * 1000.0).tolist(),
        "flange_T_tag": np.round(flange_T_tag, 9).tolist(),
        "contacts": records,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Result: {output['result']}")
    print(f"Mark1 planned motion: x={planned_delta[0]:.4f} m, y={planned_delta[1]:.4f} m")
    if execution is not None:
        print(f"Mark1 actual motion: x={final_delta[0]:.4f} m, y={final_delta[1]:.4f} m")
    print(f"Reachable contacts: {reachable_count}/2")
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()

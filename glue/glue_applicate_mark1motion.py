"""Move Mark1 in 0.1 m steps when a brush contact X exceeds 0.4 m."""

import argparse
import json
import math
import shutil
import sys
import threading
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


TAG_P_BRUSH_CENTER_M = np.array([0.0, 0.0, -0.085])
BRUSH_RADIUS_M = 0.0275
PRESS_DEPTH_M = 0.0
PRE_OFFSET_M = 0.100
CONTACT_X_TARGET_M = 0.400
MARK1_X_STEP_M = 0.100
MARK1_X_SPEED = 0.10

WORLD_X = np.array([1.0, 0.0, 0.0])
WORLD_Z = np.array([0.0, 0.0, 1.0])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--brush-pick-json", type=Path)
    parser.add_argument("--segments-json", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def resolve_paths(args):
    run_dir = args.run_dir.expanduser().resolve()
    args.brush_pick_json = (
        args.brush_pick_json
        or run_dir / "pickplace" / "glue_brush_pick_endpose.json"
    )
    args.segments_json = (
        args.segments_json
        or run_dir
        / "completion"
        / "depression"
        / "glue_brush_adaptive_segments.json"
    )
    args.output = (
        args.output
        or run_dir / "pickplace" / "glue_brush_adaptive_segments_mark1.json"
    )
    return args


def normalize(vector):
    vector = np.asarray(vector, dtype=float).reshape(3)
    return vector / np.linalg.norm(vector)


def endpose_to_transform(endpose):
    endpose = np.asarray(endpose, dtype=float).reshape(6)
    transform = np.eye(4)
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
    return [round(float(value), 6) for value in np.r_[xyz_mm, rpy_deg]]


def flange_T_tag_from_pick(brush_pick):
    base_T_tag = np.asarray(brush_pick["base_T_tag"], dtype=float)
    base_T_flange = endpose_to_transform(brush_pick["endpose"])
    return np.linalg.inv(base_T_flange) @ base_T_tag


def horizontal_outward(normal):
    normal = normalize(normal)
    if np.dot(normal, WORLD_X) > 0.0:
        normal = -normal
    return normalize([normal[0], normal[1], 0.0])


def base_R_tag(normal):
    tag_x = normal
    tag_y = normalize(np.cross(WORLD_Z, tag_x))
    return np.column_stack((tag_x, tag_y, WORLD_Z))


def tangent_xy(segment, normal):
    tangent = segment.get("tangent_unit")
    if tangent is None:
        tangent = np.cross(WORLD_Z, normal)
    tangent = normalize([tangent[0], tangent[1], 0.0])
    return tangent if np.dot(tangent, WORLD_X) >= 0.0 else -tangent


def brush_center_endpose(brush_center, rotation, flange_T_tag):
    base_T_tag = np.eye(4)
    base_T_tag[:3, :3] = rotation
    base_T_tag[:3, 3] = brush_center - rotation @ TAG_P_BRUSH_CENTER_M
    base_T_flange = base_T_tag @ np.linalg.inv(flange_T_tag)
    return transform_to_endpose(base_T_flange)


def contact_center(segment, base_delta_xy=(0.0, 0.0)):
    center = np.asarray(segment["center_point_base_m"], dtype=float)
    center[:2] -= np.asarray(base_delta_xy, dtype=float)
    normal = horizontal_outward(segment["outward_normal_unit"])
    brush_center = center + (BRUSH_RADIUS_M - PRESS_DEPTH_M) * normal
    return center, normal, brush_center


def plan_segment(segment, base_delta_xy, flange_T_tag):
    center, normal, contact = contact_center(segment, base_delta_xy)
    rotation = base_R_tag(normal)
    tangent = tangent_xy(segment, normal)
    half_length = max(0.0, float(segment.get("xoy_length", 0.0))) / 2.0

    pre = contact + PRE_OFFSET_M * normal
    start = contact - half_length * tangent
    end = contact + half_length * tangent

    return {
        "segment_id": int(segment["id"]),
        "original_center_base_m": segment["center_point_base_m"],
        "transformed_center_base_m": np.round(center, 9).tolist(),
        "outward_normal_unit": np.round(normal, 9).tolist(),
        "sweep_tangent_unit": np.round(tangent, 9).tolist(),
        "sweep_xoy_length_mm": round(2.0 * half_length * 1000.0, 3),
        "brush_center_pre_base_m": np.round(pre, 9).tolist(),
        "brush_center_contact_base_m": np.round(contact, 9).tolist(),
        "brush_center_start_base_m": np.round(start, 9).tolist(),
        "brush_center_end_base_m": np.round(end, 9).tolist(),
        "pre_app_endpose": brush_center_endpose(pre, rotation, flange_T_tag),
        "contact_endpose": brush_center_endpose(contact, rotation, flange_T_tag),
        "start_endpose": brush_center_endpose(start, rotation, flange_T_tag),
        "end_endpose": brush_center_endpose(end, rotation, flange_T_tag),
        "base_R_tag": np.round(rotation, 9).tolist(),
        "ik_checked": False,
    }


def execute_mark1(dx):
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from Mark1.motion_ctrl import Mark1BaseController

    rclpy.init()
    base = Mark1BaseController()
    executor = MultiThreadedExecutor()
    executor.add_node(base)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()

    base.wait_for_odom()
    start = np.array([base.x, base.y, base.yaw], dtype=float)
    threading.Event().wait(1)
    if dx > 1e-4:
        base.move_x(dx, speed_mps=MARK1_X_SPEED)
    base.stop()
    threading.Event().wait(2)
    end = np.array([base.x, base.y, base.yaw], dtype=float)

    base.stop()
    executor.shutdown()
    thread.join(timeout=1.0)
    base.destroy_node()
    rclpy.shutdown()

    delta_odom = end[:2] - start[:2]
    c, s = np.cos(start[2]), np.sin(start[2])
    actual_delta = np.array([[c, s], [-s, c]]) @ delta_odom
    return actual_delta, {
        "odom_before": start.tolist(),
        "odom_after": end.tolist(),
    }


def main():
    args = resolve_paths(parse_args())
    segment_data = json.loads(args.segments_json.read_text(encoding="utf-8"))
    segments = segment_data["segments"]

    initial_contacts = [contact_center(segment)[2] for segment in segments]
    initial_max_x = max(point[0] for point in initial_contacts)
    if initial_max_x > CONTACT_X_TARGET_M:
        step_count = math.ceil(
            (initial_max_x - CONTACT_X_TARGET_M) / MARK1_X_STEP_M
        )
        planned_dx = step_count * MARK1_X_STEP_M
    else:
        planned_dx = 0.0

    copied_segments = None
    if planned_dx == 0.0:
        copied_segments = (
            args.run_dir.expanduser().resolve()
            / "pickplace"
            / "glue_brush_adaptive_segments.json"
        )
        copied_segments.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.segments_json, copied_segments)

    if planned_dx > 0.0:
        final_delta, execution = execute_mark1(planned_dx)
    else:
        final_delta = np.zeros(2, dtype=float)
        execution = {"odom_before": None, "odom_after": None}

    transformed_segments = []
    for segment in segments:
        transformed = dict(segment)
        center = np.asarray(segment["center_point_base_m"], dtype=float).copy()
        center[:2] -= final_delta
        transformed["center_point_base_m"] = np.round(center, 9).tolist()
        transformed_segments.append(transformed)

    final_max_x = max(
        contact_center(segment)[2][0] for segment in transformed_segments
    )
    output = {"segments": transformed_segments}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    motion_output = (
        args.run_dir.expanduser().resolve() / "pickplace" / "mark1_motion.json"
    )
    motion_output.parent.mkdir(parents=True, exist_ok=True)
    motion_output.write_text(
        json.dumps(
            {
                "move": planned_dx > 0.0,
                "planned_delta_base_m": [planned_dx, 0.0],
                "final_delta_base_m": final_delta.tolist(),
                "delta_base_m": final_delta.tolist(),
                "odom_before": execution["odom_before"],
                "odom_after": execution["odom_after"],
                "executed": planned_dx > 0.0,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    if execution["odom_before"] is not None:
        print(f"Odom before [x, y, yaw]: {execution['odom_before']}")
        print(f"Odom after  [x, y, yaw]: {execution['odom_after']}")
        print(f"Measured base delta [x, y]: {final_delta.tolist()}")
    print(f"Saved: {args.output.resolve()}")
    print(f"Saved: {motion_output.resolve()}")


if __name__ == "__main__":
    main()

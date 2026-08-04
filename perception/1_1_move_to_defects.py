import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pinocchio as pin
from scipy.spatial.transform import Rotation as R

PROJECT_ROOT = Path(os.getenv("AAM_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
sys.path.append(str(PROJECT_ROOT))


ROOT_DIR = PROJECT_ROOT / "perception" / "data" / "rough_screening"
OUTPUT_PATH = ROOT_DIR / "mark1_ctrl.json"
ECT_PATH = PROJECT_ROOT / "config" / "calibration" / "right_camera" / "ecT.npy"
URDF_PATH = PROJECT_ROOT / "config" / "piper" / "piper_description.urdf"
EE_FRAME = "link6"

# Desired arm pose for final visual alignment.
DESIRED_JOINT_DEG = np.array([ 0, 35, -30, 0, 30, 0], dtype=float)  # 0.0, 40.0, -25.0, 0.0, 10, 0.0

MAX_VX = 0.08
MAX_VY = 0.06
RATE_HZ = 20
MIN_DURATION = 0.5

# Depth saved from RealSense z16 is usually in millimeters.
DEPTH_SCALE = 0.001

# Chassis origin expressed in arm base_link frame.
# x forward, y left. The chassis origin is 23 cm behind and 21 cm left of base_link.
CHASSIS_ORIGIN_IN_ARM_BASE = np.array([-0.23, 0.21, 0.0], dtype=float)
CAMERA_OPTICAL_AXIS = np.array([0.0, 0.0, 1.0], dtype=float)


def latest_csv(root_dir):
    csv_files = sorted(root_dir.glob("*_RoughInspection.csv"), key=lambda p: p.stat().st_mtime)
    return csv_files[-1]


def load_camera_intrinsic(config_path):
    config = np.load(config_path, allow_pickle=True).item()
    return config["color_intrinsic"]


def transform_point(T, point):
    point_hom = np.array([point[0], point[1], point[2], 1.0], dtype=float)
    return (T @ point_hom)[:3]


def piper_endpose_json_to_T(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        pose = json.load(f)

    xyz_m = np.array([pose[0]["x"], pose[1]["y"], pose[2]["z"]], dtype=float) / 1_000_000.0
    rpy_deg = np.array([pose[3]["rx"], pose[4]["ry"], pose[5]["rz"]], dtype=float) / 1000.0

    T = np.eye(4)
    T[:3, :3] = R.from_euler("xyz", rpy_deg, degrees=True).as_matrix()
    T[:3, 3] = xyz_m
    return T


def reduce_to_arm_only(model):
    q_ref = pin.neutral(model)
    lock_ids = []
    for joint_name in ["joint7", "joint8"]:
        if model.existJointName(joint_name):
            lock_ids.append(model.getJointId(joint_name))
    if not lock_ids:
        return model
    return pin.buildReducedModel(model, lock_ids, q_ref)


def fixed_joint_ee_T():
    model = pin.buildModelFromUrdf(str(URDF_PATH))
    model = reduce_to_arm_only(model)
    data = model.createData()

    q = np.deg2rad(DESIRED_JOINT_DEG)
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    frame_id = model.getFrameId(EE_FRAME)
    ee_pose = data.oMf[frame_id]

    T = np.eye(4)
    T[:3, :3] = ee_pose.rotation
    T[:3, 3] = ee_pose.translation
    return T


def mask_center_pixel(mask):
    ys, xs = np.where(mask.astype(bool))
    if len(xs) == 0:
        return None
    return float(np.mean(xs)), float(np.mean(ys))


def robust_mask_depth(depth, mask):
    values = depth[mask.astype(bool)]
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) == 0:
        return None
    return float(np.median(values)) * DEPTH_SCALE


def pixel_to_camera(u, v, z, intr):
    x = (u - intr["ppx"]) / intr["fx"] * z
    y = (v - intr["ppy"]) / intr["fy"] * z
    return np.array([x, y, z], dtype=float)


def arm_base_to_chassis_origin(point_arm_base):
    return point_arm_base - CHASSIS_ORIGIN_IN_ARM_BASE


def target_base_translation_for_camera_axis(object_center_arm, desired_camera_T):
    camera_center = desired_camera_T[:3, 3]
    optical_axis = desired_camera_T[:3, :3] @ CAMERA_OPTICAL_AXIS
    optical_axis = optical_axis / np.linalg.norm(optical_axis)

    if abs(optical_axis[2]) < 1e-6:
        s = float(np.dot(object_center_arm - camera_center, optical_axis))
    else:
        s = float((object_center_arm[2] - camera_center[2]) / optical_axis[2])

    desired_object_on_axis = camera_center + s * optical_axis
    move_x = float(object_center_arm[0] - desired_object_on_axis[0])
    move_y = float(object_center_arm[1] - desired_object_on_axis[1])

    return np.array([move_x, move_y, 0.0], dtype=float), desired_object_on_axis, optical_axis


def make_move_command(move_xy):
    move_x = float(move_xy[0])
    move_y = float(move_xy[1])

    duration_x = abs(move_x) / MAX_VX if abs(move_x) > 1e-6 else 0.0
    duration_y = abs(move_y) / MAX_VY if abs(move_y) > 1e-6 else 0.0
    duration = max(duration_x, duration_y, MIN_DURATION)

    vx = move_x / duration
    vy = move_y / duration

    return {
        "vx": float(vx),
        "vy": float(vy),
        "wz": 0.0,
        "duration": float(duration),
        "rate_hz": RATE_HZ,
    }


def vector_to_xyz_dict(vector):
    return {
        "x": float(vector[0]),
        "y": float(vector[1]),
        "z": float(vector[2]),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--rough-csv", type=Path, default=None)
    parser.add_argument("--depth-path", type=Path, default=None)
    parser.add_argument("--camera-config", type=Path, default=None)
    parser.add_argument("--capture-pose", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def paths_from_run_dir(run_dir):
    rough_dir = run_dir / "perception" / "rough_screening"
    return {
        "rough_csv": rough_dir / f"{run_dir.name}_RoughInspection.csv",
        "depth_path": run_dir / "start.npy",
        "camera_config": run_dir / "camera_config.npy",
        "capture_pose": run_dir / "start.json",
        "output_json": rough_dir / "mark1_ctrl.json",
    }


def build_control_plan(args):
    run_paths = paths_from_run_dir(args.run_dir) if args.run_dir else {}

    csv_path = args.rough_csv or run_paths.get("rough_csv") or latest_csv(ROOT_DIR)
    timestamp = csv_path.name.replace("_RoughInspection.csv", "")
    depth_path = args.depth_path or run_paths.get("depth_path") or (csv_path.parent / f"{timestamp}.npy")
    config_path = args.camera_config or run_paths.get("camera_config") or (csv_path.parent / "camera_config.npy")
    capture_pose_path = args.capture_pose or run_paths.get("capture_pose") or (csv_path.parent / f"{timestamp}.json")
    output_json = args.output_json if args.output_json != OUTPUT_PATH else run_paths.get("output_json", args.output_json)

    df = pd.read_csv(csv_path)
    depth = np.load(depth_path)
    intr = load_camera_intrinsic(config_path)
    ecT = np.load(ECT_PATH)
    capture_ee_T = piper_endpose_json_to_T(capture_pose_path)
    capture_camera_T = capture_ee_T @ ecT
    desired_camera_T = fixed_joint_ee_T() @ ecT

    plans = []
    previous_target_translation = np.zeros(3, dtype=float)
    for i, row in df.iterrows():
        mask_path = row.get("mask_path", "")
        if not isinstance(mask_path, str) or not mask_path:
            continue

        mask = np.load(mask_path)
        center_pixel = mask_center_pixel(mask)
        center_depth = robust_mask_depth(depth, mask)
        if center_pixel is None or center_depth is None:
            continue

        u, v = center_pixel
        point_camera = pixel_to_camera(u, v, center_depth, intr)
        point_arm_base = transform_point(capture_camera_T, point_camera)
        point_chassis = arm_base_to_chassis_origin(point_arm_base)
        move_xy, desired_object_on_axis, optical_axis = target_base_translation_for_camera_axis(
            point_arm_base,
            desired_camera_T,
        )
        target_translation_from_initial = move_xy
        delta_from_previous_target = target_translation_from_initial - previous_target_translation
        command_from_initial = make_move_command(target_translation_from_initial)
        command_from_previous_target = make_move_command(delta_from_previous_target)
        previous_target_translation = target_translation_from_initial

        defect_id = int(row["id"]) if "id" in df.columns and not pd.isna(row["id"]) else int(i + 1)
        plans.append(
            {
                "id": defect_id,
                "object_center_coordinate": {
                    "arm_base_link": {
                        "x": float(point_arm_base[0]),
                        "y": float(point_arm_base[1]),
                        "z": float(point_arm_base[2]),
                        "distance": float(math.sqrt(point_arm_base[0] ** 2 + point_arm_base[1] ** 2)),
                    },
                    "mark1_chassis_origin": {
                        "x": float(point_chassis[0]),
                        "y": float(point_chassis[1]),
                        "z": float(point_chassis[2]),
                        "distance": float(math.sqrt(point_chassis[0] ** 2 + point_chassis[1] ** 2)),
                    },
                },
                "desired_camera_alignment": {
                    "joint_degrees": DESIRED_JOINT_DEG.tolist(),
                    "camera_center_in_arm_base": desired_camera_T[:3, 3].astype(float).tolist(),
                    "camera_optical_axis_in_arm_base": optical_axis.astype(float).tolist(),
                    "object_point_on_optical_axis_before_base_motion": desired_object_on_axis.astype(float).tolist(),
                    "required_base_translation": {
                        "x": float(move_xy[0]),
                        "y": float(move_xy[1]),
                        "z": 0.0,
                    },
                },
                "target_base_translation_from_initial": vector_to_xyz_dict(target_translation_from_initial),
                "delta_from_previous_target": vector_to_xyz_dict(delta_from_previous_target),
                "move_control_from_initial": command_from_initial,
                "move_control_from_previous_target": command_from_previous_target,
            }
        )

    output = {
        "source_csv": str(csv_path),
        "ecT_path": str(ECT_PATH),
        "urdf_path": str(URDF_PATH),
        "ee_frame": EE_FRAME,
        "chassis_origin_in_arm_base": {
            "x": float(CHASSIS_ORIGIN_IN_ARM_BASE[0]),
            "y": float(CHASSIS_ORIGIN_IN_ARM_BASE[1]),
            "z": float(CHASSIS_ORIGIN_IN_ARM_BASE[2]),
        },
        "multi_defect_execution_note": (
            "Use target_base_translation_from_initial as the absolute target offset from the initial base pose. "
            "For sequential inspection, move by delta_from_previous_target or recompute the delta from current odom."
        ),
        "commands": plans,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, ensure_ascii=False, indent=4), encoding="utf-8")
    return output


def main():
    args = parse_args()
    plan = build_control_plan(args)
    print(json.dumps(plan, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    main()

"""Calculate reachable depression repair-block pick and fix endposes."""

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


PROJECT_ROOT = Path(os.getenv("AAM_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Piper.endpose_reachability_safe import (
    DEFAULT_EE_FRAME,
    frame_pose,
    get_safe_bounds,
    load_arm_model,
    reachability_test,
)


ARM_GRIPPER_LENGTH_Z_MM = 142.5
PRINTER_CENTER_BASE_MM = np.array([-266.425, 184.39, 60.8])
PRINT_YAW_DEG = 90.0
UNIT_SCALE = 1000.0
FIX_X_OFFSET = 0
FIX_Y_OFFSET = 0


def endpose_to_transform(endpose):
    endpose = np.asarray(endpose, dtype=float).reshape(6)
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler(
        "xyz", endpose[3:], degrees=True
    ).as_matrix()
    transform[:3, 3] = endpose[:3]
    return transform


def transform_to_endpose(transform):
    transform = np.asarray(transform, dtype=float).reshape(4, 4)
    return np.concatenate([
        transform[:3, 3],
        Rotation.from_matrix(transform[:3, :3]).as_euler("xyz", degrees=True),
    ])


def generate_grab_yaw_candidates(grab_position_mm):
    """Return ordered, unique yaw candidates found by the joint-space search."""
    grab_position_mm = np.asarray(grab_position_mm, dtype=float).reshape(3)
    target_position_m = grab_position_mm / 1000.0
    target_z = np.array([0.0, 0.0, -1.0])

    model = load_arm_model()
    frame_id = model.getFrameId(DEFAULT_EE_FRAME)
    lower, upper = get_safe_bounds(model)
    rng = np.random.default_rng(0)
    seeds = [
        np.clip(pin.neutral(model), lower, upper),
        np.clip(np.zeros(model.nq), lower, upper),
    ]
    seeds.extend(rng.uniform(lower, upper) for _ in range(50))

    candidates = []
    for seed in seeds:
        data = model.createData()

        def residual(q):
            pose = frame_pose(model, data, q, frame_id)
            position_error = (pose.translation - target_position_m) / 0.002
            ee_z = pose.rotation[:, 2]
            axis_error = np.cross(ee_z, target_z) / np.deg2rad(2.0)
            direction_error = np.array([
                (1.0 - np.dot(ee_z, target_z)) / 0.001
            ])
            return np.concatenate([position_error, axis_error, direction_error])

        solution = least_squares(
            residual,
            seed,
            bounds=(lower, upper),
            max_nfev=1500,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
        )
        q = solution.x
        pose = frame_pose(model, data, q, frame_id)
        position_error_mm = float(
            np.linalg.norm(pose.translation - target_position_m) * 1000.0
        )
        tilt_error_deg = float(np.rad2deg(np.arccos(np.clip(
            np.dot(pose.rotation[:, 2], target_z), -1.0, 1.0
        ))))
        if position_error_mm > 2.0 or tilt_error_deg > 2.0:
            continue

        yaw_deg = float(np.degrees(np.arctan2(
            pose.rotation[1, 0], pose.rotation[0, 0]
        )) % 360.0)
        joint_margin = float(np.min(np.minimum(q - lower, upper - q)))
        candidates.append({
            "joint_margin": joint_margin,
            "position_error_mm": position_error_mm,
            "tilt_error_deg": tilt_error_deg,
            "yaw_deg": yaw_deg,
        })

    if not candidates:
        raise RuntimeError("No reachable grab orientation found for yaw in [0, 360).")

    candidates.sort(key=lambda item: (
        -item["joint_margin"],
        item["position_error_mm"],
        item["tilt_error_deg"],
    ))
    unique_candidates = []
    seen_yaws = set()
    for candidate in candidates:
        yaw_key = round(candidate["yaw_deg"], 1)
        if yaw_key not in seen_yaws:
            seen_yaws.add(yaw_key)
            unique_candidates.append(candidate)
    return unique_candidates


def find_pre_grab(grab_endpose):
    """Return the first reachable point 30--70 mm above Grab."""
    for offset_mm in np.linspace(15.0, 70.0, 9):
        endpose = np.asarray(grab_endpose, dtype=float).copy()
        endpose[2] += offset_mm
        result = reachability_test(endpose)
        if result["reachable"]:
            return endpose, result["joint_degrees"]
    print("Warning: pre_grab is not reachable; saving false.")
    return False, False


def find_pre_fix(
    fix_endpose,
    segments_path,
    fix_points_path,
    mark1_delta_xy,
):
    """Search joint space for an EE pose inside the continuous pre-fix box."""
    segment_data = json.loads(segments_path.read_text(encoding="utf-8"))
    normal = np.asarray(
        [segment["outward_normal_unit"] for segment in segment_data["segments"]],
        dtype=float,
    ).sum(axis=0)
    normal_norm = np.linalg.norm(normal)
    if normal_norm <= 1e-12:
        raise ValueError(f"Invalid average outward normal in {segments_path}")
    normal /= normal_norm

    points = np.asarray(o3d.io.read_point_cloud(str(fix_points_path)).points)
    points[:, :2] -= np.asarray(mark1_delta_xy, dtype=float).reshape(2)
    first_corner_m = points[np.argmax(points @ normal)] + 0.030 * normal

    fix_endpose = np.asarray(fix_endpose, dtype=float).reshape(6)
    box_min_xy = first_corner_m[:2]
    box_max_xy = box_min_xy + np.array([0.080, 0.080])
    fixed_z_m = fix_endpose[2] / 1000.0
    target_rotation = Rotation.from_euler(
        "xyz", fix_endpose[3:], degrees=True
    ).as_matrix()

    model = load_arm_model()
    frame_id = model.getFrameId(DEFAULT_EE_FRAME)
    lower, upper = get_safe_bounds(model)
    rng = np.random.default_rng(0)
    seeds = [
        np.clip(pin.neutral(model), lower, upper),
        np.clip(np.zeros(model.nq), lower, upper),
    ]
    seeds.extend(rng.uniform(lower, upper) for _ in range(30))

    position_scale_m = 0.002
    rotation_scale_rad = np.deg2rad(2.0)
    position_tolerance_m = 0.002
    rotation_tolerance_deg = 3.0
    best = None

    def closest_box_point(position):
        xy = np.clip(position[:2], box_min_xy, box_max_xy)
        return np.array([xy[0], xy[1], fixed_z_m])

    for seed in seeds:
        data = model.createData()

        def residual(q):
            pose = frame_pose(model, data, q, frame_id)
            position_error = pose.translation - closest_box_point(
                pose.translation
            )
            rotation_error = Rotation.from_matrix(
                target_rotation.T @ pose.rotation
            ).as_rotvec()
            return np.concatenate([
                position_error / position_scale_m,
                rotation_error / rotation_scale_rad,
            ])

        solution = least_squares(
            residual,
            seed,
            bounds=(lower, upper),
            max_nfev=1500,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
        )
        q = solution.x
        pose = frame_pose(model, data, q, frame_id)
        target_position = closest_box_point(pose.translation)
        position_error_m = float(
            np.linalg.norm(pose.translation - target_position)
        )
        rotation_error_deg = float(np.rad2deg(np.linalg.norm(
            Rotation.from_matrix(
                target_rotation.T @ pose.rotation
            ).as_rotvec()
        )))
        if (
            position_error_m > position_tolerance_m
            or rotation_error_deg > rotation_tolerance_deg
        ):
            continue

        joint_margin = float(np.min(np.minimum(q - lower, upper - q)))
        score = (
            joint_margin
            - 1000.0 * position_error_m
            - np.deg2rad(rotation_error_deg)
        )
        if best is None or score > best["score"]:
            best = {
                "score": score,
                "q": q.copy(),
                "target_position": target_position,
                "position_error_mm": position_error_m * 1000.0,
                "rotation_error_deg": rotation_error_deg,
            }

    if best is not None:
        pre_fix_endpose = fix_endpose.copy()
        pre_fix_endpose[:3] = best["target_position"] * 1000.0
        joint_degrees = np.round(np.rad2deg(best["q"]), 3).tolist()
        print(
            "Selected continuous-box pre_fix: "
            f"position error={best['position_error_mm']:.3f} mm, "
            f"rotation error={best['rotation_error_deg']:.3f} deg"
        )
        return pre_fix_endpose, joint_degrees

    print("Warning: pre_fix is not reachable; saving false.")
    return False, False


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        help="Depression result folder for standalone terminal debugging.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Pipeline run folder: read completion/depression and write pickplace.",
    )
    parser.add_argument("--depression-dir", type=Path)
    parser.add_argument("--pick-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fix-points", type=Path)
    parser.add_argument("--segments-json", type=Path)
    parser.add_argument("--mark1-motion", type=Path)
    return parser.parse_args()


def resolve_paths(args):
    standalone = [
        path for path in (args.input_dir, args.depression_dir) if path is not None
    ]
    if args.run_dir is not None:
        if standalone:
            raise ValueError(
                "--run-dir cannot be combined with input_dir or --depression-dir"
            )
        run_dir = args.run_dir.expanduser().resolve()
        depression_path = run_dir / "completion" / "depression"
        pick_path = (
            args.pick_dir.expanduser().resolve()
            if args.pick_dir else run_dir / "pickplace"
        )
        mark1_motion_path = (
            args.mark1_motion.expanduser().resolve()
            if args.mark1_motion
            else run_dir / "pickplace" / "mark1_motion.json"
        )
        if args.segments_json:
            segments_path = args.segments_json.expanduser().resolve()
        else:
            motion = json.loads(mark1_motion_path.read_text(encoding="utf-8"))
            segments_name = (
                "glue_brush_adaptive_segments_mark1.json"
                if motion["move"]
                else "glue_brush_adaptive_segments.json"
            )
            segments_path = run_dir / "pickplace" / segments_name
    else:
        if len(standalone) != 1:
            raise ValueError(
                "Specify exactly one of --run-dir, input_dir, or --depression-dir"
            )
        depression_path = standalone[0].expanduser().resolve()
        if args.pick_dir is None and args.output is None:
            raise ValueError("Standalone mode requires --pick-dir or --output")
        pick_path = (
            args.pick_dir.expanduser().resolve()
            if args.pick_dir else args.output.parent.resolve()
        )
        mark1_motion_path = (
            args.mark1_motion.expanduser().resolve()
            if args.mark1_motion else None
        )
        segments_path = (
            args.segments_json.expanduser().resolve()
            if args.segments_json
            else depression_path / "glue_brush_adaptive_segments.json"
        )

    output_path = (
        args.output.expanduser().resolve()
        if args.output else pick_path / "pick_place_endpose.npz"
    )
    return depression_path, output_path, mark1_motion_path, segments_path


def calculate_pick_and_fix(
    depression_path,
    segments_path,
    fix_points_path,
    mark1_motion_path=None,
):
    orientation_meta = np.load(
        depression_path / "orientation_meta.npz", allow_pickle=True
    )
    gripper_meta = np.load(
        depression_path / "gripper_meta.npz", allow_pickle=True
    )

    attach_center = np.asarray(
        orientation_meta["attach_center_oriented"], dtype=float
    ) * UNIT_SCALE
    full_box_z_height = float(
        orientation_meta["full_box_z_height"]
    ) * UNIT_SCALE
    grip_height_total = float(
        gripper_meta["grip_body_height"]
        + gripper_meta["base_height"]
        + gripper_meta["v_neck_height"]
    ) * UNIT_SCALE

    theta = np.deg2rad(PRINT_YAW_DEG)
    base_T_printer = np.eye(4)
    base_T_printer[:3, :3] = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta), np.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ])
    base_T_printer[:3, 3] = PRINTER_CENTER_BASE_MM

    printer_T_full = np.eye(4)
    printer_T_full[2,-1] = full_box_z_height/2

    printer_P_grip = np.array([
        attach_center[0],
        attach_center[1],
        full_box_z_height - grip_height_total / 2.0,
        1.0,
    ])
    base_P_grip = base_T_printer @ printer_P_grip
    grab_position = base_P_grip[:3].copy()
    grab_position[2] += ARM_GRIPPER_LENGTH_Z_MM

    base_T_object_fix = np.asarray(
        orientation_meta["base_T_full"], dtype=float
    ).copy()
    mark1_delta_xy = np.zeros(2)
    if mark1_motion_path is not None:
        motion = json.loads(mark1_motion_path.read_text(encoding="utf-8"))
        if motion["move"]:
            mark1_delta_xy = np.asarray(
                motion["final_delta_base_m"], dtype=float
            ).reshape(2)
            base_T_object_fix[:2, 3] -= mark1_delta_xy
    base_T_object_fix[:3,3] *= UNIT_SCALE
    base_T_object_fix[0, -1] += FIX_X_OFFSET
    base_T_object_fix[1, -1] += FIX_Y_OFFSET


    selected = None
    for candidate in generate_grab_yaw_candidates(grab_position):
        grab_endpose = np.array([
            grab_position[0], grab_position[1], grab_position[2],
            180.0, 0.0, candidate["yaw_deg"],
        ])
        grab_reachability = reachability_test(grab_endpose)
        if not grab_reachability["reachable"]:
            continue

        base_T_end_grab = endpose_to_transform(grab_endpose)
        end_grab_T_full = np.linalg.inv(base_T_end_grab) @ base_T_printer @ printer_T_full
        base_T_end_fix = base_T_object_fix @ np.linalg.inv(end_grab_T_full)
        fix_endpose = transform_to_endpose(base_T_end_fix)
        fix_reachability = reachability_test(fix_endpose)
        if not fix_reachability["reachable"]:
            continue

        selected = (
            grab_endpose, grab_reachability, fix_endpose, fix_reachability
        )
        print(f"Selected first jointly reachable yaw: {candidate['yaw_deg']:.3f} deg")
        break

    if selected is None:
        raise RuntimeError("No yaw produces both reachable Grab and Fix endposes.")

    grab_endpose, grab_reachability, fix_endpose, fix_reachability = selected

    # Pre points are deliberately calculated only after the Grab/Fix pair has
    # been accepted. Their failure does not invalidate that pair.
    pre_grab_endpose, pre_grab_joint_degrees = find_pre_grab(grab_endpose)
    pre_fix_endpose, pre_fix_joint_degrees = find_pre_fix(
        fix_endpose,
        segments_path,
        fix_points_path,
        mark1_delta_xy,
    )

    return {
        "grab_endpose": grab_endpose,
        "grab_joint_degrees": grab_reachability["joint_degrees"],
        "fix_endpose": fix_endpose,
        "fix_joint_degrees": fix_reachability["joint_degrees"],

        "pre_grab_endpose": pre_grab_endpose,
        "pre_grab_joint_degrees": pre_grab_joint_degrees,
        "pre_fix_endpose": pre_fix_endpose,
        "pre_fix_joint_degrees": pre_fix_joint_degrees,

    }


def save_results(results, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        #pre_grab_endpose=results["pre_grab_endpose"],
        grab_endpose=results["grab_endpose"],
        #pre_fix_endpose=results["pre_fix_endpose"],
        fix_endpose=results["fix_endpose"],
    )

    def json_value(value):
        return False if value is False else np.asarray(value).tolist()

    json_output_path = output_path.with_suffix(".json")
    with open(json_output_path, "w", encoding="utf-8") as file:
        json.dump({
            "grab_joint_degrees": json_value(results["grab_joint_degrees"]),
            "fix_joint_degrees": json_value(results["fix_joint_degrees"]),
            "grab_endpose": json_value(results["grab_endpose"]),
            "fix_endpose": json_value(results["fix_endpose"]),

            #"pre_grab_endpose": json_value(results["pre_grab_endpose"]),
            "pre_grab_joint_degrees": json_value( results["pre_grab_joint_degrees"]),
            #"pre_fix_endpose": json_value(results["pre_fix_endpose"]),
            "pre_fix_joint_degrees": json_value(results["pre_fix_joint_degrees"]),
        }, file, ensure_ascii=False, indent=2)

    print("saved:", output_path)
    print("saved:", json_output_path)


def main():
    args = parse_args()
    (
        depression_path,
        output_path,
        mark1_motion_path,
        segments_path,
    ) = resolve_paths(args)
    fix_points_path = (
        args.fix_points.expanduser().resolve()
        if args.fix_points else depression_path / "fix_points_curve.pcd"
    )
    results = calculate_pick_and_fix(
        depression_path,
        segments_path,
        fix_points_path,
        mark1_motion_path,
    )
    save_results(results, output_path)
    
    print("grab joint degrees:\n", results["grab_joint_degrees"])
    print("fix joint degrees:\n", results["fix_joint_degrees"])
    print("pre-grab joint degrees:\n", results["pre_grab_joint_degrees"])
    print("pre-fix joint degrees:\n", results["pre_fix_joint_degrees"])
    


if __name__ == "__main__":
    main()

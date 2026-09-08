"""Move Mark1 once if fix_points_curve.pcd is farther than 0.4 m in Base X.

Usage:
    python mark1_close_to_defect.py --run-dir /home/smmg/AAM/data/runs/xxxx

Default completion directory:
    <run_dir>/completion
"""

import argparse
import json
import sys
import threading
from pathlib import Path

import numpy as np
import open3d as o3d


AAM_ROOT = Path("/home/smmg/AAM")
if str(AAM_ROOT) not in sys.path:
    sys.path.insert(0, str(AAM_ROOT))


CONTACT_X_TARGET_M = 0.400
MARK1_X_STEP_M = 0.100
MARK1_X_SPEED = 0.10


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--completion-dir", type=Path, default=None)
    parser.add_argument("--fix-points-curve", type=Path, default=None)
    return parser.parse_args()


def resolve_paths(args):
    run_dir = args.run_dir.expanduser().resolve()
    completion_dir = (
        args.completion_dir.expanduser().resolve()
        if args.completion_dir is not None
        else run_dir / "completion"
    )
    fix_points_curve = (
        args.fix_points_curve.expanduser().resolve()
        if args.fix_points_curve is not None
        else completion_dir / "fix_points_curve.pcd"
    )
    motion_json = completion_dir / "mark1_motion.json"
    return completion_dir, fix_points_curve, motion_json


def load_points(pcd_path):
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    points = np.asarray(pcd.points, dtype=float)
    if len(points) == 0:
        raise RuntimeError(f"Empty point cloud: {pcd_path}")
    return points


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

    base.move_x(dx, speed_mps=MARK1_X_SPEED)
    base.stop()

    threading.Event().wait(0.5)
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


def transform_pcd(input_path, output_path, base_delta_xy):
    pcd = o3d.io.read_point_cloud(str(input_path))
    translation = np.array(
        [-base_delta_xy[0], -base_delta_xy[1], 0.0],
        dtype=float,
    )
    pcd.translate(translation)
    o3d.io.write_point_cloud(str(output_path), pcd)

def transform_meta(input_path, output_path, base_delta_xy):
    data = np.load(input_path, allow_pickle=True)

    meta = {
        key: data[key].copy()
        for key in data.files
    }

    delta = np.array(
        [base_delta_xy[0], base_delta_xy[1], 0.0],
        dtype=float,
    )

    # Base-frame position quantities
    if "top_plane_center" in meta:
        meta["top_plane_center"] = (
            np.asarray(meta["top_plane_center"], dtype=float) - delta
        )

    if "repair_point_center" in meta:
        meta["repair_point_center"] = (
            np.asarray(meta["repair_point_center"], dtype=float) - delta
        )

    # Plane equation:
    # original: n·p + d = 0
    # new frame: p_new = p_old - delta
    # therefore d_new = d_old + n·delta
    if "top_plane_model" in meta:
        plane = np.asarray(
            meta["top_plane_model"],
            dtype=float,
        ).copy()

        plane[3] += np.dot(plane[:3], delta)
        meta["top_plane_model"] = plane

    # Direction vectors such as:
    # u_axis, v_axis, n_axis, side_n_mark
    # are unchanged under pure translation.

    np.savez(output_path, **meta)

def transform_completion_pcds(completion_dir, base_delta_xy):
    outputs = []

    source_files = sorted(
        path
        for path in completion_dir.glob("*.pcd")
        if not path.stem.endswith("_motion")
    )

    for input_path in source_files:
        output_path = input_path.with_name(
            f"{input_path.stem}_motion{input_path.suffix}"
        )
        transform_pcd(input_path, output_path, base_delta_xy)
        outputs.append(output_path)

    return outputs


def main():
    args = parse_args()
    completion_dir, fix_points_curve, motion_json = resolve_paths(args)

    points = load_points(fix_points_curve)
    initial_max_x = float(points[:, 0].max())

    # Only one workspace decision is made.
    should_move = initial_max_x > CONTACT_X_TARGET_M

    if should_move:
        planned_dx = MARK1_X_STEP_M
        final_delta, execution = execute_mark1(planned_dx)
    else:
        planned_dx = 0.0
        final_delta = np.zeros(2, dtype=float)
        execution = {
            "odom_before": None,
            "odom_after": None,
        }

    transformed_pcds = transform_completion_pcds(completion_dir, final_delta )

    run_dir = args.run_dir.expanduser().resolve()

    fine_fuse_path = (
        run_dir
        / "construction"
        / "fine_scan"
        / "fine_fuse.pcd"
    )

    fine_fuse_motion_path = (
        completion_dir
        / "fine_fuse_motion.pcd"
    )

    transform_pcd(
        fine_fuse_path,
        fine_fuse_motion_path,
        final_delta,
    )

    transformed_pcds.append(fine_fuse_motion_path)
    meta_path = completion_dir / "meta.npz"
    meta_motion_path = completion_dir / "meta_motion.npz"
    transform_meta(meta_path, meta_motion_path, final_delta )
    final_max_x = initial_max_x - float(final_delta[0])

    result = {
        "move": bool(should_move),
        "contact_x_target_m": CONTACT_X_TARGET_M,
        "initial_fix_curve_max_x_m": initial_max_x,
        "planned_delta_base_m": [planned_dx, 0.0],
        "final_delta_base_m": final_delta.tolist(),
        "final_fix_curve_max_x_m": final_max_x,
        "odom_before": execution["odom_before"],
        "odom_after": execution["odom_after"],
        "executed": bool(should_move),
        "transformed_pcds": [str(path) for path in transformed_pcds],
    }

    motion_json.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Initial fix curve max X: {initial_max_x:.6f} m")
    print(f"Move Mark1: {should_move}")

    if execution["odom_before"] is not None:
        print(f"Odom before [x, y, yaw]: {execution['odom_before']}")
        print(f"Odom after  [x, y, yaw]: {execution['odom_after']}")
        print(f"Measured Base delta [x, y]: {final_delta.tolist()}")

    print(f"Final fix curve max X: {final_max_x:.6f} m")

    for path in transformed_pcds:
        print(f"Saved: {path.resolve()}")

    print(f"Saved: {motion_json.resolve()}")


if __name__ == "__main__":
    main()

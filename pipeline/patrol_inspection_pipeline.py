"""AAM patrol -> rough screening -> approaching -> three-view confirmation -> reconstruction.

Waypoints JSON: {"inspection_poses": [{"id": 1, "x": 0.0, "y": 0.0, "yaw": 0.0}, ...]}
Coordinates x/y are meters in ROS /odom; yaw is degrees.
The arm is held at --inspection-joints while the base patrols.
Use only in a cleared, supervised test area: this is odometry-based motion, not obstacle avoidance.
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(os.getenv('AAM_PROJECT_ROOT', Path(__file__).resolve().parents[1]))
sys.path.append(str(ROOT))
from Mark1.motion_ctrl import mark1_ctrl
from Piper.piper_ctrl import connect_right
import camera.camera_functions as camera_functions

PERCEPTION = ROOT / 'perception'
CONSTRUCTION = ROOT / 'construction'
RECONSTRUCTION = ROOT / 'pipeline' / 'perception_construction_pipeline.py'
SCRIPTS = {
    'rough': PERCEPTION / '1_rough_defects_segmentation.py',
    'approach': PERCEPTION / '1_1_move_to_defects.py',
    'confirm': PERCEPTION / '2_multiview_defect_confirmation.py',
    'coarse_center': CONSTRUCTION / 'coarse_scan_center.py',
    'coarse_plan': CONSTRUCTION / 'coarse_scan_9.py',
}


def call_script(path, *args, dry_run=False):
    cmd = [sys.executable, str(path), *map(str, args)]
    print('RUN:', ' '.join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=str(ROOT), check=True)


def read_odom(timeout=5.0):
    """Read one live odom message; yaw returned in radians."""
    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.qos import qos_profile_sensor_data
    rclpy.init()
    node = rclpy.create_node('aam_patrol_odom_reader')
    latest = []
    subscription = node.create_subscription(Odometry, '/odom', lambda msg: latest.append(msg), qos_profile_sensor_data)
    try:
        start = time.monotonic()
        while not latest and time.monotonic() - start < timeout:
            rclpy.spin_once(node, timeout_sec=0.2)
        if not latest:
            raise RuntimeError('No /odom received')
        p, q = latest[-1].pose.pose.position, latest[-1].pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        return float(p.x), float(p.y), yaw
    finally:
        node.destroy_subscription(subscription)
        node.destroy_node()
        rclpy.shutdown()


def angle_diff(a, b):
    return math.atan2(math.sin(a - b), math.cos(a - b))

def move_to_pose(pose, dry_run=False):
    if dry_run:
        print('DRY MOVE TO:', pose)
        return

    import rclpy
    import threading
    from rclpy.executors import MultiThreadedExecutor
    from Mark1.motion_ctrl import Mark1BaseController

    tx, ty = float(pose['x']), float(pose['y'])
    speed, tolerance, rate_hz = 0.06, 0.025, 20

    rclpy.init()
    base = Mark1BaseController()
    executor = MultiThreadedExecutor()
    executor.add_node(base)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()

    try:
        base.wait_for_odom()

        print(f'CURRENT pose: ({base.x:.4f}, {base.y:.4f}, {base.yaw:.4f})', flush=True)
        print(f'TARGET pose:  ({tx:.4f}, {ty:.4f}, {math.radians(float(pose["yaw"])):.4f})', flush=True)

        def move_axis(axis):
            dx, dy = tx - base.x, ty - base.y
            distance = abs(math.cos(base.yaw) * dx + math.sin(base.yaw) * dy) if axis == 'x' else abs(-math.sin(base.yaw) * dx + math.cos(base.yaw) * dy)
            deadline = time.monotonic() + distance / speed * 2 + 3.0

            try:
                while time.monotonic() < deadline:
                    dx, dy = tx - base.x, ty - base.y
                    ex = math.cos(base.yaw) * dx + math.sin(base.yaw) * dy
                    ey = -math.sin(base.yaw) * dx + math.cos(base.yaw) * dy
                    error = ex if axis == 'x' else ey

                    if abs(error) < tolerance:
                        return

                    velocity = math.copysign(speed, error)
                    base.send_cmd(vx=velocity if axis == 'x' else 0.0, vy=velocity if axis == 'y' else 0.0, wz=0.0)
                    time.sleep(1.0 / rate_hz)

                raise RuntimeError(f'Mark1 {axis.upper()} motion timeout')
            finally:
                base.stop()

        move_axis('x')
        move_axis('y')

        print(f'ARRIVED pose: ({base.x:.4f}, {base.y:.4f}, {base.yaw:.4f})', flush=True)

    finally:
        base.stop()
        executor.shutdown()
        thread.join(timeout=1.0)
        base.destroy_node()
        rclpy.shutdown()

def set_inspection_arm(joints, dry_run=False):
    if dry_run:
        return
    piper = connect_right(with_gripper=False)
    try:
        piper.enable()
        piper.set_speed(20)
        piper.move_joint(*map(float, joints))
        time.sleep(2)
    finally:
        piper.disconnect()


def create_inspection_dir(run_dir, name):
    trial = run_dir / f'inspection_{name}'
    for rel in ('perception/rough_screening', 'perception/confirmation', 'construction/coarse_scan', 'construction/fine_scan', 'completion', 'model2print', 'pickplace'):
        (trial / rel).mkdir(parents=True, exist_ok=True)
    return trial


def capture_frame(out_dir, stem, save_config, dry_run=False):
    if dry_run:
        print('DRY CAPTURE:', out_dir / f'{stem}.png')
        return
    camera_functions.camera_syn_endpose_path = str(out_dir / f'{stem}.json')
    camera_functions.json = json
    camera_functions.capture(img_save_path=str(out_dir) + '/', save_file_name=stem, AUTO_SAVE_INTERVAL=2.0, MAX_SAVE_FRAMES=1, SAVE_CONFIG=int(save_config), post_process=1, SAVE_ENDPOSE=True)
    for suffix in ('.png', '.npy', '.json'):
        if not (out_dir / f'{stem}{suffix}').exists():
            raise RuntimeError(f'Camera capture missing {stem}{suffix}')
    if save_config and not (out_dir / 'camera_config.npy').exists():
        raise RuntimeError('Camera config not saved')


def valid_rough_csv(csv_path):
    if not csv_path.exists():
        return False
    df = pd.read_csv(csv_path)
    if df.empty or 'mask_path' not in df.columns:
        return False
    return any(isinstance(row['mask_path'], str) and Path(row['mask_path']).exists() and ('score' not in df.columns or pd.notna(row['score'])) for _, row in df.iterrows())


def capture_first_three(coarse_dir, scanpose_path, dry_run=False):
    if dry_run:
        print('DRY CAPTURE coarse_scan_1..3')
        return
    records = json.loads(scanpose_path.read_text(encoding='utf-8'))
    records = [r for r in records if r.get('success', True)]
    if len(records) < 3:
        raise RuntimeError('Fewer than 3 successful coarse scan poses')
    piper = connect_right(with_gripper=False)
    try:
        piper.enable()
        piper.set_speed(10)
        for i, record in enumerate(records[:3], start=1):
            joints = record['joint_degrees']
            if len(joints) != 6:
                raise RuntimeError(f'Invalid coarse scan joint_degrees at frame {i}')
            piper.move_joint(*map(float, joints))
            time.sleep(12)
            capture_frame(coarse_dir, f'coarse_scan_{i}', False)
    finally:
        piper.disconnect()


def inspect_one(trial, dry_run=False):
    """Return (flag, reason). flag=True means three perception stages succeeded."""
    rough_dir = trial / 'perception' / 'rough_screening'
    conf_dir = trial / 'perception' / 'confirmation'
    coarse_dir = trial / 'construction' / 'coarse_scan'
    rough_csv = rough_dir / f'{trial.name}_RoughInspection.csv'
    plan_path = rough_dir / 'mark1_ctrl.json'
    confirmation_path = conf_dir / 'multiview_confirmation.json'
    try:
        capture_frame(trial, 'start', True, dry_run)
        call_script(SCRIPTS['rough'], '--image-path', trial / 'start.png', '--depth-path', trial / 'start.npy', '--output-csv', rough_csv, '--raw-output', rough_dir / 'screening_raw.json', '--mask-dir', rough_dir, dry_run=dry_run)
        if not dry_run and not valid_rough_csv(rough_csv):
            return False, 'no_suspect'
        call_script(SCRIPTS['approach'], '--rough-csv', rough_csv, '--depth-path', trial / 'start.npy', '--camera-config', trial / 'camera_config.npy', '--capture-pose', trial / 'start.json', '--output-json', plan_path, dry_run=dry_run)
        commands = [{'id': 1, 'move_control_from_previous_target': {'vx': 0, 'vy': 0, 'wz': 0, 'duration': 0.5, 'rate_hz': 20}}] if dry_run else json.loads(plan_path.read_text(encoding='utf-8')).get('commands', [])
        if not commands:
            return False, 'no_approach_command'
        # First version: inspect one suspect object per waypoint.
        command = commands[0]
        motion = command.get('move_control_from_initial') or command.get('move_control_from_previous_target')
        if not motion:
            return False, 'invalid_approach_command'
        if not dry_run:
            mark1_ctrl(vx=float(motion['vx']), vy=float(motion['vy']), wz=float(motion['wz']), duration=float(motion['duration']), rate_hz=int(motion['rate_hz']))
            time.sleep(5)
        capture_frame(rough_dir, 'front', False, dry_run)
        call_script(SCRIPTS['coarse_center'], '--run-dir', trial, dry_run=dry_run)
        call_script(SCRIPTS['coarse_plan'], '--run-dir', trial, dry_run=dry_run)
        capture_first_three(coarse_dir, coarse_dir / 'coarse_scanpose.json', dry_run)
        call_script(SCRIPTS['confirm'], '--defect-id', int(command.get('id', 1)), '--image-paths', *(coarse_dir / f'coarse_scan_{i}.png' for i in (1, 2, 3)), '--rough-csv', rough_csv, '--raw-output-json', conf_dir / 'multiview_confirmation_raw.json', '--output-json', confirmation_path, dry_run=dry_run)
        result = {'is_defect': 'yes'} if dry_run else json.loads(confirmation_path.read_text(encoding='utf-8'))
        if str(result.get('is_defect', '')).strip().lower() != 'yes':
            return False, 'not_confirmed'
        return True, 'confirmed'
    except Exception as e:
        return False, f'{type(e).__name__}: {e}'


def write_log(path, entries):
    path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding='utf-8')


def run_patrol(waypoints, run_dir, inspection_joints, run_fine=False, printing=False, dry_run=False):
    run_dir.mkdir(parents=True, exist_ok=True)
    poses = json.loads(waypoints.read_text(encoding='utf-8'))
    poses = poses['inspection_poses'] if isinstance(poses, dict) else poses
    entries = []
    log_path = run_dir / 'patrol_log.json'

    for index, pose in enumerate(poses, start=1):
        waypoint_id = str(pose.get('id', index))
        trial = run_dir / f'inspection_{index:03d}'
        record = {'id': waypoint_id, 'pose': pose, 'time': datetime.now().isoformat(timespec='seconds'), 'success': False}

        # Before confirmation, any task failure enters the same recovery/next-waypoint branch.
        try:
            set_inspection_arm(inspection_joints, dry_run)
            print('move to next waypoint')
            move_to_pose(pose, dry_run)
            create_inspection_dir(run_dir, f'{index:03d}')
            success, reason = inspect_one(trial, dry_run)
            record['success'], record['reason'] = success, reason
        except Exception as exc:
            success = False
            record['reason'] = f'{type(exc).__name__}: {exc}'

        if success:
            record['status'] = 'defect_confirmed'
            entries.append(record)
            write_log(log_path, entries)
            args = ['--run-dir', trial]
            if run_fine:
                args.append('--run-fine')
            if printing:
                args.append('--printing')
            try:
                call_script(RECONSTRUCTION, *args, dry_run=dry_run)
            except Exception as exc:
                record['status'] = 'reconstruction_failed'
                record['reason'] = f'{type(exc).__name__}: {exc}'
                write_log(log_path, entries)
                print(f'STOP: reconstruction failed; retained confirmed inspection data: {trial}', flush=True)
                return 'reconstruction_failed'
            record['status'] = 'reconstruction_preview' if dry_run else 'reconstruction_completed'
            write_log(log_path, entries)
            return record['status']

        # Perception failed: reset arm and return to the inspection waypoint before continuing.
        try:
            set_inspection_arm(inspection_joints, dry_run)
            #print('back to last waypoint')
            #move_to_pose(pose, dry_run)
        except Exception as exc:
            record['status'] = 'recovery_required'
            record['reason'] += f'; recovery_failed: {type(exc).__name__}: {exc}'
            entries.append(record)
            write_log(log_path, entries)
            print(f'STOP: waypoint {waypoint_id} cannot be safely recovered: {exc}', flush=True)
            return 'recovery_required'

        record['status'] = 'next_waypoint'
        #if trial.exists():
            #.rmtree(trial)
        entries.append(record)
        write_log(log_path, entries)

    return 'patrol_completed'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--waypoints', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--inspection-joints', type=float, nargs=6, required=True, metavar=('J1', 'J2', 'J3', 'J4', 'J5', 'J6'))
    parser.add_argument('--run-fine', action='store_true')
    parser.add_argument('--printing', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    status = run_patrol(args.waypoints, args.run_dir, args.inspection_joints, args.run_fine, args.printing, args.dry_run)
    print('PATROL RESULT:', status, flush=True)
    if status in ('reconstruction_failed', 'recovery_required'):
        sys.exit(1)


if __name__ == '__main__':
    main()

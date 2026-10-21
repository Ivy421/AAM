"""Standalone Mark1 base diagnostic (no AAM pipeline dependencies beyond Mark1.motion_ctrl).

Run from AAM root: python test_mark1_motion.py --axis y --speed 0.02 --duration 1.0
Execute one command per invocation; observe the robot and use emergency stop if needed.
"""
import argparse
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if not (ROOT / 'Mark1').exists():
    ROOT = Path('/home/smmg/AAM')
sys.path.insert(0, str(ROOT))

from Mark1.motion_ctrl import Mark1BaseController
import rclpy
from rclpy.executors import MultiThreadedExecutor
import threading


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--axis', choices=['x', 'y'], required=True)
    parser.add_argument('--speed', type=float, default=0.02)
    parser.add_argument('--duration', type=float, default=1.0)
    args = parser.parse_args()
    if not 0 < abs(args.speed) <= 0.2 or not 0 < args.duration <= 5.0:
        raise ValueError('Diagnostic limits: 0 < |speed| <= 0.03 m/s, 0 < duration <= 2 s')

    rclpy.init()
    base = Mark1BaseController()
    executor = MultiThreadedExecutor()
    executor.add_node(base)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    try:
        base.wait_for_odom(timeout_sec=5.0)
        before = (base.x, base.y, base.yaw)
        vx = args.speed if args.axis == 'x' else 0.0
        vy = args.speed if args.axis == 'y' else 0.0
        print(f'BEFORE odom: x={before[0]:+.4f} m, y={before[1]:+.4f} m, yaw={math.degrees(before[2]):+.2f} deg', flush=True)
        print(f'CMD: vx={vx:+.3f} m/s, vy={vy:+.3f} m/s, wz=0, duration={args.duration:.2f} s', flush=True)
        base.move_for_velocity(vx=vx, vy=vy, wz=0.0, duration=args.duration, rate_hz=20)
        base.stop()
        time.sleep(0.5)
        after = (base.x, base.y, base.yaw)
        dx, dy = after[0] - before[0], after[1] - before[1]
        c, s = math.cos(before[2]), math.sin(before[2])
        local_x, local_y = c * dx + s * dy, -s * dx + c * dy
        dyaw = math.atan2(math.sin(after[2] - before[2]), math.cos(after[2] - before[2]))
        print(f'AFTER  odom: x={after[0]:+.4f} m, y={after[1]:+.4f} m, yaw={math.degrees(after[2]):+.2f} deg')
        print(f'DELTA world: dx={dx:+.4f} m, dy={dy:+.4f} m, dyaw={math.degrees(dyaw):+.2f} deg')
        print(f'DELTA local (relative to initial yaw): dx={local_x:+.4f} m, dy={local_y:+.4f} m')
        print(f'IDEAL distance along {args.axis}: {args.speed * args.duration:+.4f} m')
    finally:
        base.stop()
        executor.shutdown()
        thread.join(timeout=1.0)
        base.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

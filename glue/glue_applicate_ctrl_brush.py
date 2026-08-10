"""Move Mark1, pick the glue brush, sweep every segment, and put it back."""

import argparse
import json
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Mark1.motion_ctrl import mark1_ctrl
from Piper.piper_ctrl import connect_left, connect_right


SAFE_JOINTS = [60, 40, -30, 0, 10, 0]
MARK1_VX_MPS = 0.02


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mark1-motion", type=Path)
    parser.add_argument("--pick-json", type=Path)
    parser.add_argument("--applicate-json", type=Path)
    parser.add_argument("--arm", choices=("right", "left"), default="right")
    parser.add_argument("--joint-speed", type=int, default=10)
    parser.add_argument("--contact-speed", type=int, default=5)
    parser.add_argument("--line-speed", type=int, default=5)
    parser.add_argument("--gripper-open-mm", type=float, default=40.0)
    parser.add_argument("--gripper-close-mm", type=float, default=0.0)
    parser.add_argument("--gripper-force", type=float, default=1.5)
    parser.add_argument("--motion-wait", type=float, default=5.0)
    parser.add_argument("--sweep-wait", type=float, default=3.0)
    return parser.parse_args()


def resolve_paths(args):
    pickplace = args.run_dir.expanduser().resolve() / "pickplace"
    args.mark1_motion = (
        args.mark1_motion or pickplace / "mark1_motion.json"
    )
    args.pick_json = (
        args.pick_json or pickplace / "glue_brush_pick_endpose.json"
    )
    args.applicate_json = (
        args.applicate_json
        or pickplace / "glue_applicate_endpose_brush.json"
    )
    return args


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def move_joint(piper, joints, speed, wait):
    piper.set_speed(speed)
    piper.move_joint(*joints)
    time.sleep(wait)


def move_line(piper, endpose, speed, wait):
    piper.set_speed(speed)
    piper.move_line(*endpose)
    time.sleep(wait)


def move_to_segment_safe_pose(piper, pre_app_joints, args):
    safe_pose_1 = list(pre_app_joints)
    safe_pose_1[2] = -30.0

    safe_pose_2 = safe_pose_1.copy()
    safe_pose_2[1] = 80.0

    move_joint(piper, safe_pose_1, args.joint_speed, 0.0)
    time.sleep(5.0)
    move_joint(piper, safe_pose_2, args.joint_speed, args.motion_wait)


def move_mark1(motion):
    if motion["move"] and not motion["executed"]:
        dx = float(motion["delta_base_m"][0])
        vx = MARK1_VX_MPS if dx >= 0.0 else -MARK1_VX_MPS
        duration = abs(dx) / MARK1_VX_MPS
        print(f"Move Mark1: vx={vx:.3f} m/s, duration={duration:.3f} s")
        mark1_ctrl(vx=vx, duration=duration)
        motion["executed"] = True


def pick_holder(piper, pick, args):
    print("Pick up glue brush")
    move_joint(
        piper, pick["prepick_joint_degrees"], args.joint_speed, args.motion_wait
    )
    piper.move_gripper(args.gripper_open_mm, force=args.gripper_force)
    time.sleep(5.0)
    move_joint(
        piper, pick["joint_degrees"], args.contact_speed, args.motion_wait
    )
    time.sleep(3.0)
    piper.move_gripper(args.gripper_close_mm, force=args.gripper_force)
    time.sleep(2.0)
    move_joint(
        piper, pick["prepick_joint_degrees"], args.contact_speed, args.motion_wait
    )
    time.sleep(3.0)
    move_joint(piper, SAFE_JOINTS, args.joint_speed, args.motion_wait)
    time.sleep(5.0)


def sweep_segments(piper, segments, args):
    for index, segment in enumerate(segments, start=1):
        print(f"移动到第 {index}个预定点")
        move_joint(
            piper,
            segment["pre_app_joint_degrees"],
            args.joint_speed,
            args.motion_wait,
        )
        time.sleep(10)
        print(f"移动到第 {index}个接触点")
        move_joint(
            piper,
            segment["contact_joint_degrees"],
            args.contact_speed,
            args.motion_wait,
        )
        time.sleep(6)
        print(f"刷过该接触面")
        move_line(
            piper,
            segment["reachable_start_endpose"],
            args.line_speed,
            args.sweep_wait,
        )
        time.sleep(5)
        move_line(
            piper,
            segment["reachable_end_endpose"],
            args.line_speed,
            args.sweep_wait,
        )
        time.sleep(5)
        print(f"返回预定点")
        move_joint(
            piper,
            segment["pre_app_joint_degrees"],
            args.joint_speed,
            args.motion_wait,
        )
        time.sleep(3)
        print(f"返回安全点")
        move_to_segment_safe_pose(
            piper, segment["pre_app_joint_degrees"], args
        )
        time.sleep(5)


def put_back_holder(piper, pick, args):
    print("Put back glue brush")
    move_joint(
        piper, pick["prepick_joint_degrees"], args.joint_speed, args.motion_wait
    )
    time.sleep(10)
    move_joint(
        piper, pick["joint_degrees"], args.contact_speed, args.motion_wait
    )
    time.sleep(5)
    piper.move_gripper(args.gripper_open_mm, force=args.gripper_force)
    time.sleep(2.0)
    move_joint(
        piper, pick["prepick_joint_degrees"], args.contact_speed, args.motion_wait
    )


def main():
    args = resolve_paths(parse_args())
    motion = load_json(args.mark1_motion)
    pick = load_json(args.pick_json)
    segments = load_json(args.applicate_json)["segments"]

    move_mark1(motion)
    args.mark1_motion.write_text(
        json.dumps(motion, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    connect = connect_right if args.arm == "right" else connect_left
    piper = connect()
    piper.enable()

    pick_holder(piper, pick, args)
    sweep_segments(piper, segments, args)
    put_back_holder(piper, pick, args)

    piper.disconnect()
    print(f"Completed {len(segments)} brush segments.")


if __name__ == "__main__":
    main()

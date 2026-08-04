"""Pick the glue holder, dispense at every feasible dot, then put it back."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from Piper.piper_ctrl import *

SAFE_POSE = [60, 40, -30, 0, 10, 0]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pick-json", type=Path, required=True)
    parser.add_argument("--applicate-json", type=Path, required=True)
    parser.add_argument(
        "--enable-arduino",
        action="store_true",
        help="Enable syringe actuation. Omit this flag for Piper-only motion debugging.",
    )
    parser.add_argument("--arduino-port", help="For example: /dev/ttyACM0")
    parser.add_argument("--arduino-baud", type=int, default=9600)
    parser.add_argument("--arm", choices=("right", "left"), default="right")
    parser.add_argument("--joint-speed", type=int, default=10)
    parser.add_argument("--joint-speed-low", type=int, default=5)
    parser.add_argument("--line-speed", type=int, default=5)
    parser.add_argument("--gripper-open-mm", type=float, default=40.0)
    parser.add_argument("--gripper-close-mm", type=float, default=0.0)
    parser.add_argument("--gripper-force", type=float, default=1.5)
    parser.add_argument("--motion-timeout", type=float, default=30.0)
    parser.add_argument("--arduino-timeout", type=float, default=5.0)
    return parser.parse_args()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def angle_error_deg(actual, target):
    return (np.asarray(actual) - np.asarray(target) + 180.0) % 360.0 - 180.0


def move_joint(piper, joints, speed, timeout):
    piper.set_speed(speed)
    piper.move_joint(*joints)


def move_line(piper, endpose, speed, timeout):
    piper.set_speed(speed)
    piper.move_line(*endpose)
    
def dispense(arduino, timeout):
    arduino.reset_input_buffer()
    arduino.write(b"s")
    arduino.flush()

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        line = arduino.readline().decode("utf-8", errors="ignore").strip()
        if line:
            print("Arduino:", line)
        if "转动完成" in line:
            return
    raise TimeoutError("Arduino dispensing feedback timed out.")


def pick_holder(piper, pick, args):
    print("Pick up glue holder")
    move_joint(piper, pick["prepick_joint_degrees"], args.joint_speed, args.motion_timeout)
    piper.move_gripper(args.gripper_open_mm, force=args.gripper_force)
    time.sleep(5.0)
    move_joint(piper, pick["joint_degrees"], args.joint_speed_low, args.motion_timeout)
    time.sleep(5.0)
    #move_line(piper, pick["endpose"], args.line_speed, args.motion_timeout)
    piper.move_gripper(args.gripper_close_mm, force=args.gripper_force)
    time.sleep(3.0)
    move_line(piper, pick["prepick_endpose"], args.line_speed, args.motion_timeout)
    time.sleep(5.0)
    move_joint(piper, SAFE_POSE, args.joint_speed, args.motion_timeout )
    time.sleep(5)



def put_back_holder(piper, pick, args):
    print("Put back glue holder")
    move_joint(piper, pick["prepick_joint_degrees"], args.joint_speed, args.motion_timeout)
    time.sleep(7)
    move_joint(piper, pick["joint_degrees"], args.joint_speed, args.motion_timeout)
    time.sleep(7)
    piper.move_gripper(args.gripper_open_mm, force=args.gripper_force)
    time.sleep(2)
    move_joint(piper, pick["prepick_joint_degrees"], args.line_speed, args.motion_timeout)

def main():
    args = parse_args()
    if args.enable_arduino and not args.arduino_port:
        raise ValueError("--arduino-port is required with --enable-arduino.")

    pick = load_json(args.pick_json)
    dots = load_json(args.applicate_json)["dot_endposes"]

    connect = connect_right if args.arm == "right" else connect_left
    piper = connect()
    arduino = None
    if args.enable_arduino:
        import serial

        arduino = serial.Serial(args.arduino_port, args.arduino_baud, timeout=0.2)
        time.sleep(3.0)

    try:
        piper.enable()
        pick_holder(piper, pick, args)

        if dots:
            print('移动到第0个预定点')
            move_joint(
                piper,
                dots[0]["pre_applicate_joint_degrees"],
                args.joint_speed,
                args.motion_timeout,
            )
            time.sleep(8)

        for index, dot in enumerate(dots):
            print(f"move joint to {index+1} fix point")
            move_joint(piper, dot['joint_degrees'], args.joint_speed_low, args.motion_timeout)
            time.sleep(10)

            if arduino is not None:
                dispense(arduino, args.arduino_timeout)
            else:
                print('>>>>>>>>>>dispensing>>>>>>>>')
            time.sleep(3)
            print('move joint back to pre-app pose::')
            move_joint(
                piper,
                dot["pre_applicate_joint_degrees"],
                args.joint_speed,
                args.motion_timeout,
            )
            time.sleep(8)
            print('move to safe pose::')
            move_joint(
                piper,
                SAFE_POSE,
                args.joint_speed,
                args.motion_timeout,
            )
            if index + 1 < len(dots):
                time.sleep(8)
                print('move joint to next pre app pose::')
                move_joint(
                    piper,
                    dots[index + 1]["pre_applicate_joint_degrees"],
                    args.joint_speed,
                    args.motion_timeout,
                )
                time.sleep(8)

        put_back_holder(piper, pick, args)
        print(f"Completed {len(dots)} dispensing dots.")
    finally:
        if arduino is not None:
            arduino.close()
        piper.disconnect()


if __name__ == "__main__":
    main()

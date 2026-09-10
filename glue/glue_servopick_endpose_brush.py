"""Estimate the AprilTag pose and Piper grasp pose for the glue brush."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from glue import glue_servopick_endpose as base
import camera.camera_functions as camera_functions
from Piper.piper_ctrl import connect_right


BRUSH_GRASP_POINT_TAG_MM = np.array([-50, 0, -20], dtype=float)
BRUSH_TAG_FAMILY = "tag36h11"
BRUSH_TAG_X_FROM_DETECTED_AXIS = "+y"

DEFAULT_OUTPUT_NAME = "glue_brush_pick_endpose.json"


_base_parse_args = base.parse_args


# ============================================================
# CLI
# ============================================================

def str2bool(value):
    if isinstance(value, bool):
        return value

    value = value.lower()

    if value in ("true", "1", "yes"):
        return True

    if value in ("false", "0", "no"):
        return False

    raise argparse.ArgumentTypeError(
        "Expected true or false."
    )


def parse_args():
    """
    Add --move-piper without modifying glue_servopick_endpose.py.
    """

    custom_parser = argparse.ArgumentParser(
        add_help=False
    )

    custom_parser.add_argument(
        "--move-piper",
        action="store_true",
        help="Execute grasp after pick endpose generation.",
    )

    custom_args, remaining_args = (
        custom_parser.parse_known_args()
    )

    # Let original glue_servopick_endpose parser
    # parse all its existing parameters.
    original_argv = sys.argv

    try:
        sys.argv = [
            sys.argv[0],
            *remaining_args,
        ]

        args = _base_parse_args()

    finally:
        sys.argv = original_argv

    args.move_piper = custom_args.move_piper

    args.tag_family = BRUSH_TAG_FAMILY
    args.needle_axis = BRUSH_TAG_X_FROM_DETECTED_AXIS

    if (
        args.run_dir is not None
        and args.output is None
    ):
        args.output = (
            args.run_dir.expanduser().resolve()
            / "pickplace"
            / DEFAULT_OUTPUT_NAME
        )

    return args

def capture_apriltag(args):

    if args.run_dir is None:
        raise ValueError(
            "--run-dir is required for AprilTag capture."
        )

    run_dir = args.run_dir.expanduser().resolve()
    pickplace_dir = run_dir / "pickplace"

    pickplace_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    pose_path = (pickplace_dir/ "apriltag.json")
    piper = connect_right()
    piper.enable()
    piper.set_speed(20)
    piper.move_joint(90, 20, -30, 0, 50, 0  )
    time.sleep(5)
    camera_functions.camera_syn_endpose_path = str( pose_path)
    camera_functions.json = json
    camera_functions.capture(
        img_save_path=(
            str(pickplace_dir)
            + os.sep
        ),
        save_file_name="apriltag",
        AUTO_SAVE_INTERVAL=2.0,
        MAX_SAVE_FRAMES=1,
        SAVE_CONFIG=1,
        post_process=1,
        SAVE_ENDPOSE=True,
    )

    piper.disconnect()


def execute_grasp(output_path):
    """
    Execute brush grasp using:
        prepick_joint_degrees
        joint_degrees
    from glue_brush_pick_endpose.json.
    """

    pick = json.loads(
        Path(output_path).read_text(
            encoding="utf-8"
        )
    )

    prepick_joint_degrees = pick["prepick_joint_degrees"]
    joint_degrees = pick["joint_degrees"]

    piper = connect_right()

    try:
        piper.clear_error()
        piper.enable()

        # 1. Open gripper to 40 mm
        print('move to pre-pick brush')
        piper.move_gripper(
            40,
            force=1.5,
        )

        # 2. Move to pre-pick pose
        piper.set_speed(20)

        piper.move_joint(
            *prepick_joint_degrees
        )

        time.sleep(7)

        # 3. Slowly move to grasp pose
        piper.set_speed(4)
        print('move to pick brush')
        piper.move_joint(
            *joint_degrees
        )

        time.sleep(3)

        # Close gripper
        piper.move_gripper(
            0,
            force=1.5,
        )
        time.sleep(1)
        
        print('pull up brush')
        piper.move_joint(*prepick_joint_degrees)

    finally:
        piper.disconnect()

# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    capture_apriltag(args)
    base.GRASP_POINT_TAG_MM = ( BRUSH_GRASP_POINT_TAG_MM)
    base.parse_args = lambda: args
    base.main()
    output_path = ( args.output.expanduser().resolve())
    if args.move_piper:
        execute_grasp(output_path)
    else:
        print(f"Pick endpose saved: {output_path}")



if __name__ == "__main__":
    main()
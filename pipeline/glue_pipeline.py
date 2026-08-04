"""Capture the glue holder, compute glue poses, and execute dispensing."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path('/home/smmg/AAM')
GLUE_DIR = PROJECT_ROOT / "glue"
sys.path.insert(0, str(PROJECT_ROOT))
from Piper.piper_ctrl import *

SCRIPTS = {
    "servo_pick": GLUE_DIR / "glue_servopick_endpose.py",
    "applicate_endpose": GLUE_DIR / "glue_applicate_endpose.py",
    "applicate_ctrl": GLUE_DIR / "glue_applicate_ctrl.py",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--hand-eye",
        type=Path,
        default=PROJECT_ROOT / "config" / "calibration" / "right_camera" / "ecT.npy",
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=PROJECT_ROOT / "config" / "piper" / "piper_description.urdf",
    )
    parser.add_argument("--skip-capture", action="store_true", help="Reuse existing apriltag files.")
    parser.add_argument("--enable-arduino", action="store_true")
    parser.add_argument("--arduino-port", help="For example: /dev/ttyACM0")
    parser.add_argument("--arduino-baud", type=int, default=9600)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_script(script, *args, dry_run=False):
    command = [sys.executable, str(script), *map(str, args)]
    print("\nRUN:", " ".join(command))
    if not dry_run:
        subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)


def capture_apriltag(pickplace_dir, dry_run=False):
    print(f"\nCAPTURE: {pickplace_dir / 'apriltag.png'}")
    if dry_run:
        return

    from camera import camera_functions

    camera_functions.capture(
        img_save_path=str(pickplace_dir) + os.sep,
        save_file_name="apriltag",
        AUTO_SAVE_INTERVAL=2.0,
        MAX_SAVE_FRAMES=1,
        SAVE_CONFIG=1,
        post_process=1,
        SAVE_ENDPOSE=True,
    )


def run_pipeline(args):
    run_dir = args.run_dir.expanduser().resolve()
    pickplace_dir = run_dir / "pickplace"
    pickplace_dir.mkdir(parents=True, exist_ok=True)

    r_piper = connect_right()
    r_piper.enable()
    r_piper.set_speed(10)
    r_piper.move_joint(90, 40, -40, 0, 65, 0)   
    r_piper.move_gripper(40,1)
    time.sleep(7)
    capture_apriltag(pickplace_dir, dry_run=args.dry_run)

    run_script(
        SCRIPTS["servo_pick"],
        "--run-dir", run_dir,
        "--camera-config", pickplace_dir / "camera_config.npy",
        "--hand-eye", args.hand_eye,
        "--urdf", args.urdf,
        dry_run=args.dry_run,
    )

    run_script(
        SCRIPTS["applicate_endpose"],
        "--run-dir", run_dir,
        "--urdf", args.urdf,
        dry_run=args.dry_run,
    )

    control_args = [
        "--pick-json", pickplace_dir / "glue_pick_endpose.json",
        "--applicate-json", pickplace_dir / "glue_applicate_endpose.json",
    ]
    if args.enable_arduino:
        if not args.arduino_port:
            raise ValueError("--arduino-port is required with --enable-arduino.")
        control_args.extend([
            "--enable-arduino",
            "--arduino-port", args.arduino_port,
            "--arduino-baud", args.arduino_baud,
        ])

    run_script(
        SCRIPTS["applicate_ctrl"],
        *control_args,
        dry_run=args.dry_run,
    )
    print("\nGLUE PIPELINE COMPLETE")


def main():
    run_pipeline(parse_args())


if __name__ == "__main__":
    main()

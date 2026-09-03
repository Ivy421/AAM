"""Move Mark1, capture the brush, compute brush paths, and execute them."""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GLUE_DIR = PROJECT_ROOT / "glue"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Piper.piper_ctrl import connect_right


CAPTURE_JOINTS = [90, 40, -40, 0, 65, 0]

SCRIPTS = {
    "mark1_motion": GLUE_DIR / "glue_applicate_mark1motion.py",
    "servo_pick": GLUE_DIR / "glue_servopick_endpose_brush.py",
    "applicate_endpose": GLUE_DIR / "glue_applicate_endpose_brush.py",
    "applicate_ctrl": GLUE_DIR / "glue_applicate_ctrl_brush.py",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--hand-eye",
        type=Path,
        default=(
            PROJECT_ROOT
            / "config"
            / "calibration"
            / "right_camera"
            / "ecT.npy"
        ),
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=PROJECT_ROOT / "config" / "piper" / "piper_description.urdf",
    )
    parser.add_argument("--skip-capture", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_script(script, *args, dry_run=False):
    command = [sys.executable, str(script), *map(str, args)]
    print("\nRUN:", " ".join(command))
    if not dry_run:
        subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)


def capture_brush(pickplace_dir, dry_run=False):
    print(f"\nCAPTURE: {pickplace_dir / 'apriltag.png'}")
    if dry_run:
        return

    piper = connect_right()
    piper.enable()
    piper.set_speed(10)
    piper.move_joint(*CAPTURE_JOINTS)
    piper.move_gripper(40, 1)
    time.sleep(7.0)

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
    piper.disconnect()


def run_pipeline(args):
    run_dir = args.run_dir.expanduser().resolve()
    pickplace_dir = run_dir / "pickplace"
    pickplace_dir.mkdir(parents=True, exist_ok=True)

    run_script(
        SCRIPTS["mark1_motion"],
        "--run-dir",
        run_dir,
        dry_run=args.dry_run,
    )

    if not args.skip_capture:
        capture_brush(pickplace_dir, dry_run=args.dry_run)

    run_script(
        SCRIPTS["servo_pick"],
        "--run-dir",
        run_dir,
        "--camera-config",
        pickplace_dir / "camera_config.npy",
        "--hand-eye",
        args.hand_eye,
        "--urdf",
        args.urdf,
        dry_run=args.dry_run,
    )

    run_script(
        SCRIPTS["applicate_endpose"],
        "--run-dir",
        run_dir,
        "--urdf",
        args.urdf,
        dry_run=args.dry_run,
    )

    run_script(
        SCRIPTS["applicate_ctrl"],
        "--run-dir",
        run_dir,
        dry_run=args.dry_run,
    )
    print("\nGLUE BRUSH PIPELINE COMPLETE")


def main():
    run_pipeline(parse_args())


if __name__ == "__main__":
    main()

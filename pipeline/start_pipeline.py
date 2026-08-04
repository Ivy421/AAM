import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(os.getenv("AAM_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
sys.path.append(str(PROJECT_ROOT))
import camera.camera_functions as camera_functions

RUNS_DIR = PROJECT_ROOT / "data" / "runs"
INITIAL_STEM = 'start'


def make_run_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_run_dirs(run_id):
    run_dir = RUNS_DIR / run_id
    dirs = {
        "run": run_dir,
        "perception": run_dir / "perception",
        "rough_screening": run_dir / "perception" / "rough_screening",
        "confirmation": run_dir / "perception" / "confirmation",
        "construction": run_dir / "construction",
        "coarse_scan": run_dir / "construction" / "coarse_scan",
        "fine_scan": run_dir / "construction" / "fine_scan",
        "completion": run_dir / "completion",
        "model2print": run_dir / "model2print",
        "pickplace": run_dir / "pickplace",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def capture_initial_frame(run_dir, dry_run=False):
    image_path = run_dir / f"{INITIAL_STEM}.png"
    depth_path = run_dir / f"{INITIAL_STEM}.npy"
    endpose_path = run_dir / f"{INITIAL_STEM}.json"
    config_path = run_dir / "camera_config.npy"

    print(f"\nCAPTURE initial scene: {image_path}")
    if dry_run:
        return image_path, depth_path, endpose_path, config_path

    camera_functions.camera_syn_endpose_path = str(endpose_path)
    camera_functions.json = json
    camera_functions.capture(
        img_save_path=str(run_dir) + "/",
        save_file_name=INITIAL_STEM,
        AUTO_SAVE_INTERVAL=2.0,
        MAX_SAVE_FRAMES=1,
        SAVE_CONFIG=1,
        post_process=1,
        SAVE_ENDPOSE=True,
    )
    return image_path, depth_path, endpose_path, config_path


def write_run_meta(run_dir, run_id, image_path, depth_path, endpose_path, config_path):
    meta = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "initial_frame": {
            "image_path": str(image_path),
            "depth_path": str(depth_path),
            "endpose_path": str(endpose_path),
            "camera_config_path": str(config_path),
        },
        "status": "initialized",
    }
    meta_path = run_dir / "run_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=4), encoding="utf-8")
    return meta_path


def run_main_pipeline(run_dir, run_fine=False, printing=False, dry_run=False):
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "pipeline" / "perception_construction_pipeline.py"),
        "--run-dir",
        str(run_dir),
    ]
    if run_fine:
        cmd.append("--run-fine")
    if printing:
        cmd.append("--printing")
    if dry_run:
        cmd.append("--dry-run")

    print("\nRUN:", " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None, help="Optional run id. Default: current timestamp.")
    parser.add_argument("--run-fine", action="store_true", help="Pass --run-fine to perception_construction_pipeline.py.")
    parser.add_argument("--printing", action="store_true", help="Pass --printing to completion_pipeline.py through perception_construction_pipeline.py.")
    parser.add_argument("--dry-run", action="store_true", help="Create no camera capture and only print commands.")
    args = parser.parse_args()

    run_id = args.run_id or make_run_id()
    dirs = create_run_dirs(run_id)

    image_path, depth_path, endpose_path, config_path = capture_initial_frame(
        dirs["run"],
        dry_run=args.dry_run,
    )
    meta_path = write_run_meta(
        dirs["run"],
        run_id,
        image_path,
        depth_path,
        endpose_path,
        config_path,
    )

    print(f"\nRUN META: {meta_path}")
    run_main_pipeline(dirs["run"], run_fine=args.run_fine, printing=args.printing, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

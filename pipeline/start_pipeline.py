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
PATROL_SCRIPT = PROJECT_ROOT / "pipeline" / "patrol_inspection_pipeline.py"
INITIAL_STEM = "start"


def make_run_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_run_dirs(run_id):
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def capture_initial_frame(run_dir, dry_run=False):
    """Capture one inspection frame; called from a waypoint, not during startup."""
    run_dir = Path(run_dir)
    image_path = run_dir / f"{INITIAL_STEM}.png"
    depth_path = run_dir / f"{INITIAL_STEM}.npy"
    endpose_path = run_dir / f"{INITIAL_STEM}.json"
    config_path = run_dir / "camera_config.npy"
    print(f"\nCAPTURE initial scene: {image_path}")
    if not dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)
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


def write_run_meta(run_dir, run_id, waypoints_path, inspection_joints):
    meta = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "waypoints_path": str(waypoints_path),
        "inspection_joint_degrees": list(inspection_joints),
        "inspection_data_layout": "inspection_xxx/start.png, start.npy, start.json, camera_config.npy",
        "patrol_log_path": str(run_dir / "patrol_log.json"),
        "status": "initialized",
    }
    meta_path = run_dir / "run_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=4), encoding="utf-8")
    return meta_path


def run_patrol_pipeline(run_dir, waypoints_path, inspection_joints, run_fine=False, printing=False, dry_run=False):
    cmd = [
        sys.executable,
        str(PATROL_SCRIPT),
        "--run-dir", str(run_dir),
        "--waypoints", str(waypoints_path),
        "--inspection-joints", *[str(v) for v in inspection_joints],
    ]
    if run_fine:
        cmd.append("--run-fine")
    if printing:
        cmd.append("--printing")
    if dry_run:
        cmd.append("--dry-run")
    print("\nRUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None, help="Patrol run ID; default is timestamp.")
    parser.add_argument("--waypoints", type=Path, default=PROJECT_ROOT / "config" / "patrol" / "waypoints.json")
    parser.add_argument("--inspection-joints", type=float, nargs=6, default= [ -90, 40, -30, 0, 20, 0 ])
    parser.add_argument("--run-fine", action="store_true", help="Enable fine scan after confirmed coarse scan.")
    parser.add_argument("--printing", action="store_true", help="Enable printing in completion pipeline.")
    parser.add_argument("--dry-run", action="store_true", help="Preview the patrol without robot/camera motion.")
    args = parser.parse_args()

    waypoints_path = args.waypoints.resolve()
    if not waypoints_path.is_file():
        parser.error(f"Waypoints file not found: {waypoints_path}")

    run_id = args.run_id or make_run_id()
    run_dir = create_run_dirs(run_id)
    meta_path = write_run_meta(run_dir, run_id, waypoints_path, args.inspection_joints)
    print(f"\nRUN_DIR: {run_dir}\nRUN META: {meta_path}")

    # Patrol owns arm positioning and RGB-D acquisition at EACH waypoint.
    # Its set_inspection_arm() restores the fixed pose before any base motion.
    try:
        run_patrol_pipeline(run_dir, waypoints_path, args.inspection_joints, args.run_fine, args.printing, args.dry_run)
    except Exception:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["status"] = "error"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=4), encoding="utf-8")
        raise
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["status"] = "patrol_finished"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=4), encoding="utf-8")


if __name__ == "__main__":
    main()

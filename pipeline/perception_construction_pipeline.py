import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(os.getenv("AAM_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
sys.path.append(str(PROJECT_ROOT))
import camera.camera_functions as camera_functions
from Piper.piper_ctrl import connect_right

DEFAULT_RUNS_DIR = PROJECT_ROOT / "data" / "runs"
CONSTRUCTION_DIR = PROJECT_ROOT / "construction"
SCRIPTS = {
    "coarse_point_extraction": CONSTRUCTION_DIR / "coarse_point_extraction.py",
    "coarse_icp": CONSTRUCTION_DIR / "coarse_icp_add.py",
    "coarse_icp_curvature": CONSTRUCTION_DIR / "coarse_icp_curvature.py",
    "corner_mode_mapping": CONSTRUCTION_DIR / "Depression_corner_mode_mapping.py",
    "fine_scan_center": CONSTRUCTION_DIR / "fine_scan_center.py",
    "fine_scan": CONSTRUCTION_DIR / "fine_scan.py",
    "fine_point_extraction": CONSTRUCTION_DIR / "fine_point_extraction.py",
    "fine_icp": CONSTRUCTION_DIR / "fine_icp_add.py",
    "fine_icp_curvature": CONSTRUCTION_DIR / "fine_icp_curvature.py",
    "completion": PROJECT_ROOT / "pipeline" / "completion_pipeline.py",
}

RUN_DIR = None
COARSE_SCAN_DIR = None
FINE_SCAN_DIR = None


def configure_run_paths(run_dir):
    global RUN_DIR, COARSE_SCAN_DIR, FINE_SCAN_DIR
    RUN_DIR = Path(run_dir).resolve()
    COARSE_SCAN_DIR = RUN_DIR / "construction" / "coarse_scan"
    FINE_SCAN_DIR = RUN_DIR / "construction" / "fine_scan"
    for rel in ("perception/rough_screening", "perception/confirmation", "construction/coarse_scan", "construction/fine_scan", "completion", "model2print", "pickplace"):
        (RUN_DIR / rel).mkdir(parents=True, exist_ok=True)


def run_script(script_path, *args, dry_run=False):
    cmd = [sys.executable, str(script_path), *map(str, args)]
    print("\nRUN:", " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def load_scanpose_records(scanpose_json):
    with open(scanpose_json, "r", encoding="utf-8") as f:
        records = json.load(f)
    return [record for record in records if record.get("success", True)]


def select_scanpose_records(records, view_mode):
    if view_mode == "coarse_remaining":
        return records[3:], 4
    if view_mode == "fine_all":
        return records, 1
    raise ValueError(f"Unknown reconstruction view_mode: {view_mode}")


def scan_output_dir_and_prefix(view_mode):
    if view_mode == "coarse_remaining":
        return COARSE_SCAN_DIR, "coarse_scan"
    if view_mode == "fine_all":
        return FINE_SCAN_DIR, "fine_scan"
    raise ValueError(f"Unknown reconstruction view_mode: {view_mode}")


def capture_scan_views(scanpose_json, view_mode, dry_run=False):
    if dry_run:
        print(f"\nDRY CAPTURE {view_mode}: {scanpose_json}")
        return

    records = load_scanpose_records(scanpose_json)
    if view_mode == "coarse_remaining" and len(records) < 3:
        raise RuntimeError("Coarse scan plan contains fewer than 3 successful viewpoints")
    selected_records, start_index = select_scanpose_records(records, view_mode)
    output_dir, file_prefix = scan_output_dir_and_prefix(view_mode)
    print(f"\nCAPTURE {view_mode}: {len(selected_records)} poses from {scanpose_json}", flush=True)
    if not selected_records:
        return

    piper = connect_right(with_gripper=False)
    try:
        piper.enable()
        piper.set_speed(10)
        camera_functions.json = json
        for offset, record in enumerate(selected_records):
            joint_degrees = record.get("joint_degrees")
            if not joint_degrees or len(joint_degrees) != 6:
                raise ValueError(f"Invalid scan pose: {record.get('cube_name', offset)}")

            frame_index = start_index + offset
            save_file_name = f"{file_prefix}_{frame_index}"
            print(f"\nPiper move_joint for {save_file_name}: {joint_degrees}", flush=True)
            piper.move_joint(*[float(v) for v in joint_degrees])
            time.sleep(12)

            camera_functions.camera_syn_endpose_path = str(output_dir / f"{save_file_name}.json")
            camera_functions.capture(
                img_save_path=str(output_dir) + "/",
                save_file_name=save_file_name,
                AUTO_SAVE_INTERVAL=2.0,
                MAX_SAVE_FRAMES=1,
                SAVE_CONFIG=0,
                post_process=1,
                SAVE_ENDPOSE=True,
            )
            for extension in (".png", ".npy", ".json"):
                if not (output_dir / f"{save_file_name}{extension}").exists():
                    raise RuntimeError(f"Missing captured frame: {save_file_name}{extension}")
    finally:
        piper.disconnect()


def validate_confirmed_input(dry_run=False):
    if dry_run:
        return
    confirmation_path = RUN_DIR / "perception" / "confirmation" / "multiview_confirmation.json"
    if not confirmation_path.exists():
        raise FileNotFoundError(f"Missing multiview confirmation: {confirmation_path}")
    with open(confirmation_path, "r", encoding="utf-8") as f:
        confirmation = json.load(f)
    if str(confirmation.get("is_defect", "")).strip().lower() != "yes":
        raise RuntimeError("Reconstruction requires successful multiview confirmation (is_defect=yes)")
    scanpose_path = COARSE_SCAN_DIR / "coarse_scanpose.json"
    records = load_scanpose_records(scanpose_path)
    if len(records) < 3:
        raise RuntimeError("Fewer than 3 valid coarse scan poses")
    for index in (1, 2, 3):
        for suffix in (".png", ".npy", ".json"):
            path = COARSE_SCAN_DIR / f"coarse_scan_{index}{suffix}"
            if not path.exists():
                raise FileNotFoundError(f"Missing patrol confirmation frame: {path}")


def run_coarse_reconstruction(dry_run=False):
    run_script(SCRIPTS["coarse_point_extraction"], "--run-dir", RUN_DIR, dry_run=dry_run)
    run_script(SCRIPTS["coarse_icp"], "--run-dir", RUN_DIR, dry_run=dry_run)
    run_script(SCRIPTS["coarse_icp_curvature"], "--run-dir", RUN_DIR, dry_run=dry_run)
    run_script(SCRIPTS["corner_mode_mapping"], "--run-dir", RUN_DIR, dry_run=dry_run)
    run_script(SCRIPTS["fine_scan_center"], "--run-dir", RUN_DIR, dry_run=dry_run)


def run_fine_reconstruction(dry_run=False):
    run_script(SCRIPTS["fine_scan"], "--run-dir", RUN_DIR, dry_run=dry_run)
    capture_scan_views(FINE_SCAN_DIR / "fine_scanpose.json", "fine_all", dry_run=dry_run)
    run_script(SCRIPTS["fine_point_extraction"], "--run-dir", RUN_DIR, dry_run=dry_run)
    run_script(SCRIPTS["fine_icp"], "--run-dir", RUN_DIR, dry_run=dry_run)
    run_script(SCRIPTS["fine_icp_curvature"], "--run-dir", RUN_DIR, dry_run=dry_run)


def run_pipeline(run_fine=False, printing=False, dry_run=False, run_dir=None):
    if run_dir is None:
        raise ValueError("Reconstruction requires --run-dir from patrol_inspection_pipeline.py")
    configure_run_paths(run_dir)
    print(f"\nRECONSTRUCTION RUN_DIR: {RUN_DIR}", flush=True)
    validate_confirmed_input(dry_run=dry_run)
    capture_scan_views(COARSE_SCAN_DIR / "coarse_scanpose.json", "coarse_remaining", dry_run=dry_run)
    run_coarse_reconstruction(dry_run=dry_run)
    if run_fine:
        run_fine_reconstruction(dry_run=dry_run)
        completion_args = ["--run-dir", RUN_DIR]
        if printing:
            completion_args.append("--printing")
        run_script(SCRIPTS["completion"], *completion_args, dry_run=dry_run)
    else:
        print("\nSKIP completion: run_fine is False, fine_fuse.pcd is not available.")
    print("\nRECONSTRUCTION PIPELINE COMPLETE")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True, help="Confirmed inspection directory created by patrol_inspection_pipeline.py")
    parser.add_argument("--run-fine", action="store_true")
    parser.add_argument("--printing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_pipeline(run_fine=args.run_fine, printing=args.printing, dry_run=args.dry_run, run_dir=args.run_dir)


if __name__ == "__main__":
    main()

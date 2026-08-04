import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(os.getenv("AAM_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
sys.path.append(str(PROJECT_ROOT))
import camera.camera_functions as camera_functions
from Piper.piper_ctrl import connect_right


DEFAULT_RUNS_DIR = PROJECT_ROOT / "data" / "runs"

PERCEPTION_DIR = PROJECT_ROOT / "perception"
CONSTRUCTION_DIR = PROJECT_ROOT / "construction"

RUN_DIR = None
PERCEPTION_DATA_DIR = None
ROUGH_SCREENING_DIR = None
COARSE_SCAN_DIR = None
CONFIRMATION_DIR = None
CONSTRUCTION_DATA_DIR = None
COMPLETION_DIR = None
FINE_SCAN_DIR = None
MARK1_PLAN_PATH = None
CONFIRMATION_RESULT_PATH = None

START_FRAME_STEM = "start"

SCRIPTS = {
    "rough_segmentation": PERCEPTION_DIR / "1_rough_defects_segmentation.py",
    "move_to_defects": PERCEPTION_DIR / "1_1_move_to_defects.py",
    "multiview_confirmation": PERCEPTION_DIR / "2_multiview_defect_confirmation.py",
    "coarse_scan_center": CONSTRUCTION_DIR / "coarse_scan_center.py",
    "coarse_scan": CONSTRUCTION_DIR / "coarse_scan_9.py",
    "coarse_point_extraction": CONSTRUCTION_DIR / "coarse_point_extraction.py",
    "coarse_icp": CONSTRUCTION_DIR / "coarse_icp_add.py",
    "corner_mode_mapping": CONSTRUCTION_DIR / "Depression_corner_mode_mapping.py",
    "fine_scan_center": CONSTRUCTION_DIR / "fine_scan_center.py",
    "fine_scan": CONSTRUCTION_DIR / "fine_scan.py",
    "fine_point_extraction": CONSTRUCTION_DIR / "fine_point_extraction.py",
    "fine_icp": CONSTRUCTION_DIR / "fine_icp_add.py",
    "completion": PROJECT_ROOT / "pipeline" / "completion_pipeline.py",
}


def ensure_run_dirs(run_dir):
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


def configure_run_paths(run_dir):
    global RUN_DIR
    global PERCEPTION_DATA_DIR, ROUGH_SCREENING_DIR, COARSE_SCAN_DIR, CONFIRMATION_DIR
    global CONSTRUCTION_DATA_DIR, COMPLETION_DIR, FINE_SCAN_DIR
    global MARK1_PLAN_PATH, CONFIRMATION_RESULT_PATH

    dirs = ensure_run_dirs(run_dir)

    RUN_DIR = dirs["run"]
    PERCEPTION_DATA_DIR = dirs["perception"]
    ROUGH_SCREENING_DIR = dirs["rough_screening"]
    CONSTRUCTION_DATA_DIR = dirs["construction"]
    COARSE_SCAN_DIR = dirs["coarse_scan"]
    CONFIRMATION_DIR = dirs["confirmation"]
    COMPLETION_DIR = dirs["completion"]
    FINE_SCAN_DIR = dirs["fine_scan"]

    MARK1_PLAN_PATH = ROUGH_SCREENING_DIR / "mark1_ctrl.json"
    CONFIRMATION_RESULT_PATH = CONFIRMATION_DIR / "multiview_confirmation.json"


def run_script(script_path, *args, dry_run=False):
    cmd = [sys.executable, str(script_path), *map(str, args)]
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def has_valid_rough_mask(csv_path):
    if csv_path is None or not csv_path.exists():
        return False

    df = pd.read_csv(csv_path)
    if len(df) == 0 or "mask_path" not in df.columns:
        return False

    for _, row in df.iterrows():
        mask_path = str(row.get("mask_path", "")).strip()
        score = row.get("score", None)
        if mask_path and Path(mask_path).exists() and not pd.isna(score):
            return True

    return False


def load_mark1_commands():
    with open(MARK1_PLAN_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("commands", [])


def run_mark1_motion(command, dry_run=False):
    motion = command.get("move_control_from_previous_target") or command.get("move_control_from_initial")
    if not motion:
        raise ValueError(f"No move control found for defect id={command.get('id')}")

    print(f"\nMOVE Mark1 to defect id={command.get('id')}: {motion}")
    if dry_run:
        return

    from Mark1.motion_ctrl import mark1_ctrl

    mark1_ctrl(
        vx=float(motion["vx"]),
        vy=float(motion["vy"]),
        wz=float(motion["wz"]),
        duration=float(motion["duration"]),
        rate_hz=int(motion["rate_hz"]),
    )


def capture_current_frame_for_construction(dry_run=False):
    print(f"\nCAPTURE front frame: {ROUGH_SCREENING_DIR / 'front.png'}")
    if dry_run:
        return

    camera_functions.camera_syn_endpose_path = str(ROUGH_SCREENING_DIR / "front.json")
    camera_functions.json = json
    camera_functions.capture(
        img_save_path=str(ROUGH_SCREENING_DIR) + "/",
        save_file_name="front",
        AUTO_SAVE_INTERVAL=2.0,
        MAX_SAVE_FRAMES=1,
        SAVE_CONFIG=0,
        post_process=1,
        SAVE_ENDPOSE=True,
    )


def load_scanpose_records(scanpose_json):
    with open(scanpose_json, "r", encoding="utf-8") as f:
        records = json.load(f)
    return [record for record in records if record.get("success", True)]


def select_scanpose_records(records, view_mode):
    if view_mode == "confirm_first_3":
        return records[:3], 1
    if view_mode == "coarse_remaining":
        return records[3:], 4
    if view_mode == "fine_all":
        return records, 1
    raise ValueError(f"Unknown view_mode: {view_mode}")


def scan_output_dir_and_prefix(view_mode):
    if view_mode in ("confirm_first_3", "coarse_remaining"):
        return COARSE_SCAN_DIR, "coarse_scan"
    if view_mode == "fine_all":
        return FINE_SCAN_DIR, "fine_scan"
    raise ValueError(f"Unknown view_mode: {view_mode}")


def capture_scan_views(scanpose_json, view_mode, dry_run=False):
    records = load_scanpose_records(scanpose_json) if not dry_run else [{"joint_degrees": [ 0, 30 , -30 , 0 , 35 , 0  ]}] * 3
    selected_records, start_index = select_scanpose_records(records, view_mode)
    output_dir, file_prefix = scan_output_dir_and_prefix(view_mode)

    print(f"\nCAPTURE {view_mode}: {len(selected_records)} poses from {scanpose_json}")
    if dry_run:
        return

    piper = connect_right(with_gripper=False)
    piper.enable()
    piper.set_speed(10)
    camera_functions.json = json

    try:
        for offset, record in enumerate(selected_records):
            joint_degrees = record.get("joint_degrees")
            if not joint_degrees or len(joint_degrees) != 6:
                print(f"Skip scan pose without valid joint_degrees: {record.get('cube_name', offset)}")
                continue

            frame_index = start_index + offset
            save_file_name = f"{file_prefix}_{frame_index}"
            print(f"\nPiper move_joint for {save_file_name}: {joint_degrees}")

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
    finally:
        piper.disconnect()


def load_confirmation_result():
    if not CONFIRMATION_RESULT_PATH.exists():
        return {}
    with open(CONFIRMATION_RESULT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def is_confirmed_defect(result):
    return str(result.get("is_defect", "")).strip().lower() == "yes"


def run_coarse_reconstruction(dry_run=False):
    run_script(SCRIPTS["coarse_point_extraction"], "--run-dir", RUN_DIR, dry_run=dry_run)
    run_script(SCRIPTS["coarse_icp"], "--run-dir", RUN_DIR, dry_run=dry_run)
    run_script(SCRIPTS["corner_mode_mapping"], "--run-dir", RUN_DIR, dry_run=dry_run)
    run_script(SCRIPTS["fine_scan_center"], "--run-dir", RUN_DIR, dry_run=dry_run)


def run_fine_reconstruction(dry_run=False):
    run_script(SCRIPTS["fine_scan"], "--run-dir", RUN_DIR, dry_run=dry_run)
    capture_scan_views(
        scanpose_json=FINE_SCAN_DIR / "fine_scanpose.json",
        view_mode="fine_all",
        dry_run=dry_run,
    )
    run_script(SCRIPTS["fine_point_extraction"], "--run-dir", RUN_DIR, dry_run=dry_run)
    run_script(SCRIPTS["fine_icp"], "--run-dir", RUN_DIR, dry_run=dry_run)


def run_pipeline(run_fine=False, printing=False, dry_run=False, run_dir=None):
    if run_dir is None:
        run_dir = DEFAULT_RUNS_DIR / time.strftime("%Y%m%d_%H%M%S")

    configure_run_paths(Path(run_dir))
    print(f"\nRUN_DIR: {RUN_DIR}")

    start_image_path = RUN_DIR / "start.png"
    start_depth_path = RUN_DIR / "start.npy"
    start_endpose_path = RUN_DIR / "start.json"
    camera_config_path = RUN_DIR / "camera_config.npy"
    rough_csv = ROUGH_SCREENING_DIR / f"{RUN_DIR.name}_RoughInspection.csv"
    screening_raw_path = ROUGH_SCREENING_DIR / "screening_raw.json"

    run_script(
        SCRIPTS["rough_segmentation"],
        "--image-path", start_image_path,
        "--depth-path", start_depth_path,
        "--output-csv", rough_csv,
        "--raw-output", screening_raw_path,
        "--mask-dir", ROUGH_SCREENING_DIR,
        dry_run=dry_run,
    )

    if not has_valid_rough_mask(rough_csv):
        print("\nEND: no valid rough mask.")
        return

    run_script(
        SCRIPTS["move_to_defects"],
        "--rough-csv", rough_csv,
        "--depth-path", start_depth_path,
        "--camera-config", camera_config_path,
        "--capture-pose", start_endpose_path,
        "--output-json", MARK1_PLAN_PATH,
        dry_run=dry_run,
    )

    commands = load_mark1_commands() if not dry_run else [{
        "id": 1,
        "move_control_from_previous_target": {
            "vx": 0.0,
            "vy": 0.0,
            "wz": 0.0,
            "duration": 0.5,
            "rate_hz": 20,
        },
    }]
    if not commands:
        print("\nEND: no Mark1 motion command.")
        return

    for command in commands:
        defect_id = int(command.get("id", 1))

        run_mark1_motion(command, dry_run=dry_run)
        time.sleep(5)
        capture_current_frame_for_construction(dry_run=dry_run)

        run_script(SCRIPTS["coarse_scan_center"], "--run-dir", RUN_DIR, dry_run=dry_run)
        run_script(SCRIPTS["coarse_scan"], "--run-dir", RUN_DIR, dry_run=dry_run)

        capture_scan_views(
            scanpose_json=COARSE_SCAN_DIR / "coarse_scanpose.json",
            view_mode="confirm_first_3",
            dry_run=dry_run,
        )

        confirmation_image_paths = [
            COARSE_SCAN_DIR / "coarse_scan_1.png",
            COARSE_SCAN_DIR / "coarse_scan_2.png",
            COARSE_SCAN_DIR / "coarse_scan_3.png",
        ]
        run_script(
            SCRIPTS["multiview_confirmation"],
            "--defect-id", defect_id,
            "--image-paths", *confirmation_image_paths,
            "--rough-csv", rough_csv,
            "--raw-output-json", CONFIRMATION_DIR / "multiview_confirmation_raw.json",
            "--output-json", CONFIRMATION_RESULT_PATH,
            dry_run=dry_run,
        )
        confirmation = load_confirmation_result() if not dry_run else {"is_defect": "yes"}

        if not is_confirmed_defect(confirmation):
            print(f"\nEND defect id={defect_id}: confirmation says no defect.")
            continue

        capture_scan_views(
            scanpose_json=COARSE_SCAN_DIR / "coarse_scanpose.json",
            view_mode="coarse_remaining",
            dry_run=dry_run,
        )

        run_coarse_reconstruction(dry_run=dry_run)

        if run_fine:
            run_fine_reconstruction(dry_run=dry_run)
            completion_args = ["--run-dir", RUN_DIR]
            if printing:
                completion_args.append("--printing")
            run_script(SCRIPTS["completion"], *completion_args, dry_run=dry_run)
        else:
            print("\nSKIP completion: run_fine is False, fine_fuse.pcd is not available.")

    print("\nPIPELINE COMPLETE")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-fine", action="store_true", help="Continue fine scan and fine ICP after coarse reconstruction.")
    parser.add_argument("--printing", action="store_true", help="Pass --printing to completion_pipeline.py after fine reconstruction.")
    parser.add_argument("--dry-run", action="store_true", help="Print pipeline commands without executing them.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Data run directory created by start_pipeline.py.")
    args = parser.parse_args()

    run_pipeline(run_fine=args.run_fine, printing=args.printing, dry_run=args.dry_run, run_dir=args.run_dir)


if __name__ == "__main__":
    main()

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(os.getenv("AAM_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
DEFAULT_RUNS_DIR = PROJECT_ROOT / "data" / "runs"
COMPLETION_CODE_DIR = PROJECT_ROOT / "completion"

SCRIPTS = {
    "deviation": COMPLETION_CODE_DIR / "depression_completion_deviation.py",
    "twoFit": COMPLETION_CODE_DIR / "depression_completion_twoFit.py",
    "glue_applicate_path": COMPLETION_CODE_DIR / "depression_glue_applicate_path_brush_adaptive.py", #depression_glue_applicate_path
    "ransacFit": COMPLETION_CODE_DIR / "depression_completion_ransacFit.py",
    "mesh_generation": COMPLETION_CODE_DIR / "mesh_generation.py",
    "depression_grip": COMPLETION_CODE_DIR / "Depression_grip.py",
    "depression_orient": COMPLETION_CODE_DIR / "Depression_model_orient.py",
    "printing": PROJECT_ROOT / "pipeline" / "printing_pipeline.py",
}


def run_script(script_path, *args, dry_run=False):
    cmd = [sys.executable, str(script_path), *map(str, args)]
    print("\nRUN:", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_defect_type(confirmation_json):
    data = load_json(confirmation_json)
    return str(data.get("defect_type", "")).strip().lower()


def run_depression_completion(run_dir, dry_run=False, printing=False):
    run_dir = Path(run_dir)
    completion_dir = run_dir / "completion" / "depression"
    completion_dir.mkdir(parents=True, exist_ok=True)

    fine_pcd = run_dir / "construction" / "fine_scan" / "fine_fuse.pcd"
    corner_json = run_dir / "construction" / "coarse_scan" / "corner_mapping_result.json"

    if not dry_run and not fine_pcd.exists():
        raise FileNotFoundError(f"Missing fine_fuse.pcd for completion: {fine_pcd}")
    if not dry_run and not corner_json.exists():
        raise FileNotFoundError(f"Missing corner_mapping_result.json for completion: {corner_json}")

    completion_args = [
        "--pcd", fine_pcd,
        "--corner-json", corner_json,
        "--output-dir", completion_dir,
        "--no-vis",
    ]

    # run_script(SCRIPTS["ransacFit"], *completion_args, dry_run=dry_run)
    # run_script(SCRIPTS["deviation"], *completion_args, dry_run=dry_run)
    run_script(SCRIPTS["twoFit"], *completion_args, dry_run=dry_run)

    # depression_completion_twoFit.py writes both files below into completion_dir.
    # Pass those concrete outputs to the glue-dot stage; it contains no run-specific paths.

    fix_points = completion_dir / "fix_points_curve.pcd"
    fix_mask = completion_dir / 'fix_mask.npz'
    print("\n \n !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n glue applicate path working \n !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  \n ")
    run_script(
        SCRIPTS["glue_applicate_path"],
        #'--fix-mask', fix_mask,
        "--fix-points", fix_points,
        "--out-dir", completion_dir,
        #"--fixpoint-choice", "fix_points",
        dry_run=dry_run,
    )
    print("\n \n !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n mesh_generation working \n !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  \n ")
    run_script(
        SCRIPTS["mesh_generation"],
        "--completion-dir", completion_dir,
        "--pcd", completion_dir / "model.pcd",
        "--raw-stl", completion_dir / "model.stl",
        "--processed-stl", completion_dir / "model_processed.stl",
        "--allow-invalid-volume",
        dry_run=dry_run,
    )

    run_depression_grip(completion_dir, dry_run=dry_run)
    run_depression_orientation(completion_dir, dry_run=dry_run)
    if printing:
        run_printing_pipeline(run_dir, dry_run=dry_run)
    else:
        print("\nSKIP printing pipeline: pass --printing to enable it.")
    #run_printing_pipeline(run_dir, dry_run=dry_run)


def run_depression_grip(completion_dir, dry_run=False):
    completion_dir = Path(completion_dir)
    print("\n \n !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n depression_grip generation working \n !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  \n ")
    run_script(
        SCRIPTS["depression_grip"],
        "--completion-dir", completion_dir,
        "--input-stl", completion_dir / "model_processed.stl",
        "--meta", completion_dir / "meta.npz",
        "--output-stl", completion_dir / "model_with_gripper.stl",
        "--gripper-only-stl", completion_dir / "gripper_only.stl",
        "--gripper-meta", completion_dir / "gripper_meta.npz",
        dry_run=dry_run,
    )


def run_depression_orientation(completion_dir, dry_run=False):
    completion_dir = Path(completion_dir)
    print("\n \n !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n depression_model orient working \n !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  \n ")
    run_script(
        SCRIPTS["depression_orient"],
        "--completion-dir", completion_dir,
        "--input-stl", completion_dir / "model_with_gripper.stl",
        "--output-stl", completion_dir / "model_oriented.stl",
        "--meta", completion_dir / "meta.npz",
        "--gripper-meta", completion_dir / "gripper_meta.npz",
        "--repair-only-geometry", completion_dir / "model.stl",
        "--repair-only-geometry-type", "stl",
        dry_run=dry_run,
    )


def run_printing_pipeline(run_dir, dry_run=False):
    print("\n \n !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n start printing \n !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  \n ")

    run_script(
        SCRIPTS["printing"],
        "--run-dir", Path(run_dir),
        dry_run=dry_run,
    )


def run_pipeline(run_dir, dry_run=False, printing=False):
    run_dir = Path(run_dir)
    confirmation_json = run_dir / "perception" / "confirmation" / "multiview_confirmation.json"

    if not confirmation_json.exists():
        raise FileNotFoundError(f"Missing confirmation json: {confirmation_json}")

    defect_type = get_defect_type(confirmation_json)
    print("defect_type:", defect_type)

    if defect_type == "depression":
        run_depression_completion(run_dir, dry_run=dry_run, printing=printing)
    elif defect_type == "hole":
        print("hole completion is not implemented yet")
        return
    else:
        print("wrong defect type")
        return

    print("\nCOMPLETION PIPELINE COMPLETE")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True, help="Data run directory.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--printing", action="store_true", help="Call printing_pipeline.py after depression completion.")
    args = parser.parse_args()

    run_pipeline(run_dir=args.run_dir, dry_run=args.dry_run, printing=args.printing)


if __name__ == "__main__":
    main()

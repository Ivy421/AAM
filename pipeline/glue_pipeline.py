"""Run the official glue-brush pipeline for one data run.

The pipeline stages are intentionally executed as separate processes because
the stage scripts own their camera, SAM, IK, and Piper lifecycles.
"""

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(
    os.getenv("AAM_PROJECT_ROOT", Path(__file__).resolve().parents[1])
).expanduser().resolve()
GLUE_DIR = PROJECT_ROOT / "glue"
PICK_AND_PLACE_DIR = PROJECT_ROOT / "pick_and_place"

HAND_EYE_PATH = (
    PROJECT_ROOT / "config" / "calibration" / "right_camera" / "ecT.npy"
)
URDF_PATH = PROJECT_ROOT / "config" / "piper" / "piper_description.urdf"

SCRIPTS = {
    "alignment": GLUE_DIR / "fuse_alignment.py",
    "brush_pick": GLUE_DIR / "glue_servopick_endpose_brush.py",
    "glue_execute": GLUE_DIR / "glue_applicate_endpose_brush_execute.py",
    "fix_model": PICK_AND_PLACE_DIR / "depression_endpose_ctrl.py",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Align the reconstructed surface, localize/pick the glue brush, "
            "and plan/execute glue application."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stage commands without running subprocesses or moving Piper.",
    )
    execution_mode = parser.add_mutually_exclusive_group()
    execution_mode.add_argument(
        "--plan-only",
        action="store_true",
        help=(
            "Capture alignment/brush images and calculate all poses, but do "
            "not grasp the brush or execute glue motion. Camera poses still "
            "move Piper."
        ),
    )
    execution_mode.add_argument(
        "--fix-model",
        action="store_true",
        help="Pick and install the repair model after glue application.",
    )
    return parser.parse_args()


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def command_text(command):
    return shlex.join([str(value) for value in command])


def run_script(script, *args, dry_run=False):
    command = [sys.executable, str(script), *map(str, args)]
    print("\nRUN:", command_text(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)


def require_file(path, description):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")


def load_json(path):
    path = Path(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {path}: {exc}") from exc


def require_finite_vector(value, length, name):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} must contain {length} values.")
    try:
        values = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values.") from exc
    if not all(math.isfinite(item) for item in values):
        raise ValueError(f"{name} contains a non-finite value.")
    return values


def require_finite_matrix4(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{name} must be a 4x4 matrix.")
    return [
        require_finite_vector(row, 4, f"{name}[{index}]")
        for index, row in enumerate(value)
    ]


def validate_alignment_output(path):
    data = load_json(path)
    require_finite_matrix4(data.get("delta_T_base"), "delta_T_base")

    final_iou = data.get("final_mask_iou")
    if not isinstance(final_iou, (int, float)) or not math.isfinite(final_iou):
        raise ValueError("final_mask_iou is missing or non-finite.")
    if not 0.0 <= float(final_iou) <= 1.0:
        raise ValueError(f"final_mask_iou is outside [0, 1]: {final_iou}")

    print(f"Validated alignment: final IoU={float(final_iou):.6f}")
    return data


def validate_brush_pick_output(path):
    data = load_json(path)
    require_finite_matrix4(data.get("base_T_tag"), "base_T_tag")
    for key in (
        "endpose",
        "joint_degrees",
        "prepick_endpose",
        "prepick_joint_degrees",
    ):
        require_finite_vector(data.get(key), 6, key)

    print("Validated brush pick pose and IK.")
    return data


def matrices_close(left, right, tolerance=1e-8):
    left_values = require_finite_matrix4(left, "alignment delta_T_base")
    right_values = require_finite_matrix4(right, "glue-plan delta_T_base")
    return all(
        abs(a - b) <= tolerance
        for left_row, right_row in zip(left_values, right_values)
        for a, b in zip(left_row, right_row)
    )


def validate_glue_output(path, alignment):
    data = load_json(path)
    segments = data.get("segments")
    generated_count = data.get("generated_segment_count")

    if not isinstance(segments, list) or not segments:
        raise ValueError("Glue plan contains no executable segments.")
    if generated_count != len(segments):
        raise ValueError(
            "generated_segment_count does not match the number of segments: "
            f"{generated_count} != {len(segments)}"
        )
    if not matrices_close(alignment.get("delta_T_base"), data.get("delta_T_base")):
        raise ValueError("Glue plan does not use the current alignment delta_T_base.")

    for index, segment in enumerate(segments):
        segment_id = segment.get("segment_id", index)
        require_finite_vector(
            segment.get("pre_app_joint_degrees"),
            6,
            f"segment {segment_id} pre_app_joint_degrees",
        )
        require_finite_vector(
            segment.get("contact_joint_degrees"),
            6,
            f"segment {segment_id} contact_joint_degrees",
        )

    skipped = data.get("skipped_segments", [])
    pre_failed = data.get("pre_app_failed_segment_ids", [])
    print(
        "Validated glue result: "
        f"generated={len(segments)}, skipped={len(skipped)}, "
        f"pre-app-failed={len(pre_failed)}"
    )
    return data


def write_status(path, status):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(status, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def set_stage_status(status_path, status, stage, stage_status, error=None):
    record = status["stages"].setdefault(stage, {})
    record["status"] = stage_status
    if stage_status == "running":
        record["started_at"] = now_iso()
    if stage_status in ("completed", "failed"):
        record["finished_at"] = now_iso()
    if error is not None:
        record["error"] = str(error)
    status["current_stage"] = stage
    status["status"] = "failed" if stage_status == "failed" else "running"
    write_status(status_path, status)


def run_pipeline(args):
    run_dir = args.run_dir.expanduser().resolve()
    completion_dir = run_dir / "completion" / "depression"
    pickplace_dir = run_dir / "pickplace"

    fine_fuse_motion = completion_dir / "fine_fuse_motion.pcd"
    segments_json = completion_dir / "glue_brush_adaptive_segments.json"
    camera_config = pickplace_dir / "camera_config.npy"
    alignment_json = pickplace_dir / "iterative_correction.json"
    brush_pick_json = pickplace_dir / "glue_brush_pick_endpose.json"
    glue_output_json = pickplace_dir / "glue_applicate_endpose_brush.json"
    repair_endpose_json = pickplace_dir / "pick_place_endpose.json"
    status_path = pickplace_dir / "glue_pipeline_status.json"

    print(f"\nRUN_DIR: {run_dir}")
    print("MODE:", "dry-run" if args.dry_run else "plan-only" if args.plan_only else "execute")

    if not args.dry_run:
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
        pickplace_dir.mkdir(parents=True, exist_ok=True)
        require_file(fine_fuse_motion, "Mark1-transformed fine fuse point cloud")
        require_file(segments_json, "adaptive glue-brush segments")
        require_file(HAND_EYE_PATH, "right-camera hand-eye calibration")
        require_file(URDF_PATH, "Piper URDF")
        required_stages = ["alignment", "brush_pick", "glue_execute"]
        if args.fix_model:
            required_stages.append("fix_model")
            require_file(repair_endpose_json, "repair-model pick/place endposes")
        for stage in required_stages:
            script = SCRIPTS[stage]
            require_file(script, f"{stage} script")

    status = {
        "run_dir": str(run_dir),
        "mode": "dry-run" if args.dry_run else "plan-only" if args.plan_only else "execute",
        "started_at": now_iso(),
        "status": "running",
        "current_stage": None,
        "stages": {},
    }
    if not args.dry_run:
        write_status(status_path, status)

    alignment = None
    current_stage = None

    try:
        current_stage = "alignment"
        if not args.dry_run:
            set_stage_status(status_path, status, current_stage, "running")
        run_script(
            SCRIPTS[current_stage],
            "--run-dir", run_dir,
            "--camera-config", camera_config,
            "--hand-eye", HAND_EYE_PATH,
            dry_run=args.dry_run,
        )
        if not args.dry_run:
            require_file(alignment_json, "alignment result")
            alignment = validate_alignment_output(alignment_json)
            set_stage_status(status_path, status, current_stage, "completed")

        current_stage = "brush_pick"
        if not args.dry_run:
            set_stage_status(status_path, status, current_stage, "running")
        brush_args = [
            "--run-dir", run_dir,
            "--camera-config", camera_config,
            "--hand-eye", HAND_EYE_PATH,
            "--urdf", URDF_PATH,
        ]
        if not args.plan_only:
            brush_args.append("--move-piper")
        run_script(SCRIPTS[current_stage], *brush_args, dry_run=args.dry_run)
        if not args.dry_run:
            require_file(brush_pick_json, "brush pick result")
            validate_brush_pick_output(brush_pick_json)
            set_stage_status(status_path, status, current_stage, "completed")

        current_stage = "glue_execute"
        if not args.dry_run:
            set_stage_status(status_path, status, current_stage, "running")
        glue_args = [
            "--run-dir", run_dir,
            "--brush-pick-json", brush_pick_json,
            "--segments-json", segments_json,
            "--alignment-json", alignment_json,
            "--output", glue_output_json,
            "--urdf", URDF_PATH,
        ]
        if args.plan_only:
            glue_args.append("--plan-only")
        run_script(SCRIPTS[current_stage], *glue_args, dry_run=args.dry_run)
        if not args.dry_run:
            require_file(glue_output_json, "glue planning/execution result")
            validate_glue_output(glue_output_json, alignment)
            set_stage_status(status_path, status, current_stage, "completed")

        if args.fix_model:
            current_stage = "fix_model"
            if not args.dry_run:
                set_stage_status(status_path, status, current_stage, "running")
            run_script(
                SCRIPTS[current_stage],
                "--run-dir", run_dir,
                dry_run=args.dry_run,
            )
            if not args.dry_run:
                set_stage_status(status_path, status, current_stage, "completed")

    except BaseException as exc:
        if not args.dry_run and current_stage is not None:
            set_stage_status(status_path, status, current_stage, "failed", error=exc)
        print(
            f"\nGLUE PIPELINE FAILED during stage '{current_stage}'. "
            "Do not automatically retry: inspect the Piper and brush state first.",
            file=sys.stderr,
        )
        raise

    if not args.dry_run:
        status["status"] = "completed"
        status["current_stage"] = None
        status["finished_at"] = now_iso()
        write_status(status_path, status)

    print("\nGLUE PIPELINE COMPLETE")
    if not args.dry_run:
        print("Status:", status_path)


def main():
    run_pipeline(parse_args())


if __name__ == "__main__":
    main()

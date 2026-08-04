import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path('/home/smmg/AAM')
MODEL2PRINT_DIR = PROJECT_ROOT / "model2print"
PRINTING_CONFIG_DIR = PROJECT_ROOT / "config" / "printing"
DEFAULT_PRINTER_SERIAL= "0300AA642100117" # 双臂系统专用"0300AA642100117"   公用“0309AA441000235“
DEFAULT_PRINTER_IP="10.41.3.121" # 双臂系统专用 "10.41.3.121"  共用 "10.41.3.35"
DEFAUL_ACCESS_CODE="454fe07a"

SCRIPTS = {
    "get_gcode": MODEL2PRINT_DIR / "get_gcode.py",
    "printer_control_monitor": MODEL2PRINT_DIR / "printer_control_monitor.py",
    "fix_pipeline": PROJECT_ROOT / "pipeline" / "fix_pipeline.py",
}


def run_script(script_path, *args, dry_run=False, check=True):
    cmd = [sys.executable, str(script_path), *map(str, args)]
    print("\nRUN:", " ".join(cmd))
    if dry_run:
        return 0
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=check)
    return result.returncode


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def resolve_paths(args):
    run_dir = Path(args.run_dir).resolve() if args.run_dir else None
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None

    if run_dir is not None:
        output_dir = output_dir or (run_dir / "model2print")
        stl_path = Path(args.stl_path).resolve() if args.stl_path else run_dir / "completion" / "depression" / "model_oriented.stl"
        orientation_meta = (
            Path(args.orientation_meta).resolve()
            if args.orientation_meta
            else run_dir / "completion" / "depression" / "orientation_meta.npz"
        )
    else:
        output_dir = output_dir or (PROJECT_ROOT / "data" / "temp" / "model2print")
        if not args.stl_path:
            raise ValueError("--stl-path is required when --run-dir is not provided.")
        stl_path = Path(args.stl_path).resolve()
        orientation_meta = Path(args.orientation_meta).resolve() if args.orientation_meta else None

    return {
        "run_dir": run_dir,
        "output_dir": output_dir,
        "stl_path": stl_path,
        "orientation_meta": orientation_meta,
        "print_stl": output_dir / "print_model.stl",
        "assemble_list": output_dir / "assemble_list.json",
        "output_3mf": output_dir / "print_sliced.3mf",
        "result_json": output_dir / "print_result.json",
    }


def prepare_print_inputs(paths, dry_run=False):
    output_dir = paths["output_dir"]
    print_stl = paths["print_stl"]
    stl_path = paths["stl_path"]
    assemble_list = paths["assemble_list"]

    print("\nPREPARE print inputs")
    print("source STL:", stl_path)
    print("print STL:", print_stl)
    print("assemble_list:", assemble_list)

    if dry_run:
        return

    if not stl_path.exists():
        raise FileNotFoundError(f"Missing oriented STL for printing: {stl_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(stl_path, print_stl)

    assemble_data = {
        "plates": [
            {
                "plate_index": 1,
                "plate_name": "plate_1",
                "need_arrange": False,
                "objects": [
                    {
                        "path": print_stl.name,
                        "count": 1,
                        "filaments": [1],
                        "pos_x": [0.0],
                        "pos_y": [0.0],
                        "pos_z": [0.0],
                    }
                ],
                "plate_params": {
                    "print_sequence": "by layer",
                },
            }
        ]
    }
    write_json(assemble_list, assemble_data)


def run_get_gcode(args, paths):
    cmd_args = [
        "--bambu", args.bambu,
        "--machine", args.machine,
        "--process", args.process,
        "--filament", args.filament,
        "--assemble-list", paths["assemble_list"],
        "--output", paths["output_3mf"],
        "--scale", args.scale,
    ]
    if paths["orientation_meta"] is not None:
        cmd_args.extend(["--orientation-meta", paths["orientation_meta"]])
    if args.strict_watertight:
        cmd_args.append("--strict-watertight")
    cmd_args.append("--use-xvfb" if args.use_xvfb else "--no-xvfb")

    run_script(SCRIPTS["get_gcode"], *cmd_args, dry_run=args.dry_run)


def run_printer_control_monitor(args, paths):
    cmd_args = [
        "--input-file", paths["output_3mf"],
        "--ip", args.printer_ip,
        "--serial", args.printer_serial,
        "--access-code", args.access_code,
        "--plate-number", args.plate_number,
        "--poll-interval", args.poll_interval,
        "--cooldown-bed-temp", args.cooldown_bed_temp,
        "--cooldown-timeout-sec", args.cooldown_timeout_sec,
        "--result-json", paths["result_json"],
        "--start-grace-sec", args.start_grace_sec,
        "--failed-confirm-count", args.failed_confirm_count,
    ]
    if args.pause_on_error:
        cmd_args.append("--pause-on-error")

    return_code = run_script(
        SCRIPTS["printer_control_monitor"],
        *cmd_args,
        dry_run=args.dry_run,
        check=False,
    )
    if return_code not in (0, None):
        print(f"printer_control_monitor exited with code {return_code}; reading result JSON for details.")


def should_continue_to_fix(print_result):
    return bool(
        print_result.get("success")
        and print_result.get("finished")
    )


def check_print_result_and_reserve_fix(paths, dry_run=False):
    result_json = paths["result_json"]

    if dry_run:
        print("\nDRY RUN: skip reading print_result.json and reserve fix_pipeline call.")
        return

    if not result_json.exists():
        raise FileNotFoundError(f"Missing print result JSON: {result_json}")

    result = load_json(result_json)
    print("\n========== Print Result ==========")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if should_continue_to_fix(result):
        print("\nPrint succeeded and cooldown condition reached.")
        print("Reserved: fix_pipeline.py would be called here, but it is not implemented yet.")
        # Future hook:
        # run_script(SCRIPTS["fix_pipeline"], "--run-dir", paths["run_dir"])
        return

    raise RuntimeError("Print did not finish successfully or cooldown_ready is false.")


def run_pipeline(args):
    paths = resolve_paths(args)
    prepare_print_inputs(paths, dry_run=args.dry_run)
    run_get_gcode(args, paths)
    run_printer_control_monitor(args, paths)
    check_print_result_and_reserve_fix(paths, dry_run=args.dry_run)
    print("\nPRINTING PIPELINE COMPLETE")


def parse_args():
    parser = argparse.ArgumentParser(description="Slice model, start Bambu printing, monitor print and cooldown.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Data run directory. Automation mode.")
    parser.add_argument("--stl-path", type=Path, default=None, help="Oriented STL path. Debug/manual mode.")
    parser.add_argument("--orientation-meta", type=Path, default=None, help="orientation_meta.npz path.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. Defaults to run-dir/model2print.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")

    parser.add_argument("--bambu", default="/home/smmg/BambuStudio_ubuntu-22.04.AppImage", help="Bambu Studio AppImage path.")
    parser.add_argument("--machine", default=str(PRINTING_CONFIG_DIR / "machine_FULL.json"), help="Machine full config JSON.")
    parser.add_argument("--process", default=str(PRINTING_CONFIG_DIR / "process_FULL.json"), help="Process full config JSON.")
    parser.add_argument("--filament", default=str(PRINTING_CONFIG_DIR / "filament_FULL.json"), help="Filament full config JSON.")
    parser.add_argument("--scale", type=float, default=1000.0, help="Bambu Studio scale argument.")
    parser.add_argument("--strict-watertight", action="store_true", help="Fail if STL is not watertight.")
    parser.add_argument("--use-xvfb", dest="use_xvfb", action="store_true", help="Use xvfb-run for Bambu Studio.")
    parser.add_argument("--no-xvfb", dest="use_xvfb", action="store_false", help="Run Bambu Studio without xvfb-run.")

    parser.add_argument("--printer-ip", default = DEFAULT_PRINTER_IP, help="Bambu printer IP.")
    parser.add_argument("--printer-serial", default = DEFAULT_PRINTER_SERIAL, help="Bambu printer serial number.")
    parser.add_argument("--access-code", default = DEFAUL_ACCESS_CODE, help="Bambu printer access code.")
    parser.add_argument("--plate-number", type=int, default=1, help="Plate number to print.")
    parser.add_argument("--poll-interval", type=int, default=5, help="Printer polling interval in seconds.")
    parser.add_argument("--cooldown-bed-temp", type=float, default=40.0, help="Cooldown bed temperature threshold.")
    parser.add_argument("--cooldown-timeout-sec", type=int, default=15 * 60, help="Cooldown timeout in seconds.")
    parser.add_argument("--start-grace-sec", type=int, default=90, help="Ignore stale FAILED state during print startup.")
    parser.add_argument("--failed-confirm-count", type=int, default=3, help="Failed-state confirmations before stopping.")
    parser.add_argument("--pause-on-error", action="store_true", help="Pause instead of stop on printer error.")

    parser.set_defaults(use_xvfb=True)
    return parser.parse_args()


def main():
    run_pipeline(parse_args())


if __name__ == "__main__":
    main()

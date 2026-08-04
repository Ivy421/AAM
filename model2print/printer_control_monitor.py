from io import BytesIO
import argparse
import json
import os
import time
import zipfile

# 建议用环境变量保存，避免把 access code 写死在代码里
IP = "10.41.3.121"
SERIAL = '0300AA642100117'
ACCESS_CODE = '36026318'

INPUT_FILE_PATH = "E:/HKUSTGZ/AAM_MASTER/AAM/data/print_sliced.3mf"
PLATE_NUMBER = 1
POLL_INTERVAL = 5  # seconds
STOP_ON_ERROR = False  # True: stop_print on error; False: pause_print on error.
COOLDOWN_BED_TEMP_C = 40.0
COOLDOWN_TIMEOUT_SEC = 15 * 60
RESULT_JSON_PATH = None
WAIT_SUFFER_SEC = 30.0
PRINT_START_GRACE_SEC = 90
FAILED_STATE_CONFIRM_COUNT = 3


def make_print_result(
    success: bool,
    finished: bool,
    stopped: bool,
    error=None,
    state=None,
    percentage=None,
    message: str = "",
):
    return {
        "success": bool(success),
        "finished": bool(finished),
        "stopped": bool(stopped),
        "error": None if error is None else str(error),
        "state": None if state is None else str(state),
        "percentage": percentage,
        "message": message,
    }


def gcode_files_in_3mf(zipfile_path: str) -> list[str]:
    with zipfile.ZipFile(zipfile_path) as zf:
        names = zf.namelist()
    return [
        n for n in names
        if n.endswith(".gcode") and n.startswith("Metadata/plate_")
    ]


def safe_call(obj, method_name, default=None):
    try:
        return getattr(obj, method_name)()
    except Exception as e:
        return default if default is not None else f"ERR: {e}"


def to_float(value, default=None):
    try:
        return float(value)
    except Exception:
        return default


def is_error_code(error_code) -> bool:
    if error_code is None:
        return False
    if isinstance(error_code, str):
        return error_code.strip() not in ("", "0", "None")
    return error_code != 0


def is_finished_state(state) -> bool:
    state_str = str(state).strip().lower()
    return state_str in {
        "finish",
        "finished",
        "complete",
        "completed",
        "success",
        "succeeded",
    }


def is_failed_state(state) -> bool:
    state_str = str(state).strip().lower()
    return state_str in {
        "failed",
        "failure",
        "error",
        "cancel",
        "cancelled",
        "canceled",
        "abort",
        "aborted",
    }


def monitor_print(printer):
    print("Start monitoring print status...")
    monitor_start = time.time()
    failed_state_count = 0

    while True:
        elapsed = time.time() - monitor_start
        state = safe_call(printer, "get_state", "unknown")
        percent = safe_call(printer, "get_percentage", 0)
        current_layer = safe_call(printer, "current_layer_num", "?")
        total_layer = safe_call(printer, "total_layer_num", "?")
        remain_time = safe_call(printer, "get_time", "?")
        error_code = safe_call(printer, "print_error_code", 0)

        print(
            f"state={state}, progress={percent}%, "
            f"layer={current_layer}/{total_layer}, "
            f"remain={remain_time} min, error={error_code}"
        )

        if is_error_code(error_code):
            message = f"Printer error detected: {error_code}"
            print(message)
            print('!!ERROR, keep task on printer!!')

            # Important: do not stop or pause here. The printer has already entered
            # its own error-waiting state. Returning only ends this monitoring script;
            # it does not cancel the print task stored on the printer.
            result = make_print_result(
                success=False,
                finished=False,
                stopped=False,
                error=error_code,
                state=state,
                percentage=percent,
                message=message,
            )
            result["task_kept"] = True
            return result

        if is_failed_state(state):
            percent_value = to_float(percent, default=0.0)
            in_start_grace = elapsed < PRINT_START_GRACE_SEC
            likely_stale_failed = (
                in_start_grace
                and not is_error_code(error_code)
                and percent_value == 0.0
            )

            if likely_stale_failed:
                failed_state_count = 0
                print(
                    "Ignoring transient FAILED state during print-start grace "
                    f"({elapsed:.1f}/{PRINT_START_GRACE_SEC}s), error={error_code}, progress={percent}%."
                )
                time.sleep(POLL_INTERVAL)
                continue

            failed_state_count += 1
            print(
                f"Failed-state confirmation {failed_state_count}/"
                f"{FAILED_STATE_CONFIRM_COUNT}: state={state}, error={error_code}, progress={percent}%"
            )
            if failed_state_count < FAILED_STATE_CONFIRM_COUNT:
                time.sleep(POLL_INTERVAL)
                continue

            message = f"Printer entered failed state: {state}"
            print(message)
            print("Stopping print...")
            printer.stop_print()
            return make_print_result(
                success=False,
                finished=False,
                stopped=True,
                error=state,
                state=state,
                percentage=percent,
                message=message,
            )
        else:
            failed_state_count = 0

        if is_finished_state(state):
            if elapsed < PRINT_START_GRACE_SEC:
                print(
                    "Ignoring cached FINISH state during print-start grace "
                    f"({elapsed:.1f}/{PRINT_START_GRACE_SEC}s), "
                    f"progress={percent}%, layer={current_layer}/{total_layer}."
                )
                time.sleep(POLL_INTERVAL)
                continue

            message = f"Print finished with state={state}."
            print(message)
            return make_print_result(
                success=True,
                finished=True,
                stopped=False,
                error=None,
                state=state,
                percentage=percent,
                message=message,
            )

        try:
            if float(percent) >= 100:
                if elapsed < PRINT_START_GRACE_SEC:
                    print(
                        "Ignoring cached 100% progress during print-start grace "
                        f"({elapsed:.1f}/{PRINT_START_GRACE_SEC}s), "
                        f"state={state}, layer={current_layer}/{total_layer}."
                    )
                    time.sleep(POLL_INTERVAL)
                    continue

                message = "Print progress reached 100%."
                print(message)
                return make_print_result(
                    success=True,
                    finished=True,
                    stopped=False,
                    error=None,
                    state=state,
                    percentage=percent,
                    message=message,
                )
        except Exception:
            pass

        time.sleep(POLL_INTERVAL)


def monitor_cooldown(
    printer,
    target_bed_temp: float = COOLDOWN_BED_TEMP_C,
    timeout_sec: int = COOLDOWN_TIMEOUT_SEC,
    poll_interval: int = POLL_INTERVAL,
):
    print("Start monitoring bed cooldown...")
    start_time = time.time()
    last_temp = None

    while True:
        elapsed = time.time() - start_time
        bed_temp_raw = safe_call(printer, "get_bed_temperature", None)
        bed_temp = to_float(bed_temp_raw, default=None)
        if bed_temp is not None:
            last_temp = bed_temp

        print(
            f"cooldown elapsed={elapsed:.1f}s, "
            f"bed_temperature={bed_temp_raw}, target={target_bed_temp}"
        )

        if bed_temp is not None and bed_temp <= target_bed_temp:
            message = f"Bed temperature reached target: {bed_temp:.2f} <= {target_bed_temp:.2f} C."
            print(message)
            return {
                "cooldown_ready": True,
                "bed_temperature": bed_temp,
                "cooldown_elapsed_sec": elapsed,
                "cooldown_reason": "target_temperature_reached",
                "message": message,
            }

        if elapsed >= timeout_sec:
            message = f"Cooldown timeout reached: {elapsed:.1f}s >= {timeout_sec}s."
            print(message)
            return {
                "cooldown_ready": True,
                "bed_temperature": last_temp,
                "cooldown_elapsed_sec": elapsed,
                "cooldown_reason": "timeout",
                "message": message,
            }

        time.sleep(poll_interval)


def write_result_json(result: dict, result_json_path):
    if not result_json_path:
        return
    result_path = os.path.abspath(result_json_path)
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Print result JSON saved: {result_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Upload sliced 3MF, start Bambu print, monitor print and cooldown.")
    parser.add_argument("--input-file", type=str, default=INPUT_FILE_PATH, help="Input sliced 3MF path.")
    parser.add_argument("--ip", type=str, default=IP, help="Printer IP.")
    parser.add_argument("--serial", type=str, default=SERIAL, help="Printer serial number.")
    parser.add_argument("--access-code", type=str, default=ACCESS_CODE, help="Printer access code.")
    parser.add_argument("--plate-number", type=int, default=PLATE_NUMBER, help="Plate number to print.")
    parser.add_argument("--poll-interval", type=int, default=POLL_INTERVAL, help="Polling interval in seconds.")
    parser.add_argument("--cooldown-bed-temp", type=float, default=COOLDOWN_BED_TEMP_C, help="Cooldown bed temperature threshold.")
    parser.add_argument("--cooldown-timeout-sec", type=int, default=COOLDOWN_TIMEOUT_SEC, help="Cooldown timeout in seconds.")
    parser.add_argument("--result-json", type=str, default=RESULT_JSON_PATH, help="Write final print result to JSON.")
    parser.add_argument("--wait-suffer", type=float, default=WAIT_SUFFER_SEC, help="Seconds to wait after start_print before reading state, progress, or error information.")
    parser.add_argument("--start-grace-sec", type=int, default=PRINT_START_GRACE_SEC, help="Seconds to ignore stale FAILED state after start_print when error=0 and progress=0.")
    parser.add_argument("--failed-confirm-count", type=int, default=FAILED_STATE_CONFIRM_COUNT, help="Consecutive failed-state reads required before stopping print.")
    parser.add_argument("--pause-on-error", action="store_true", help="Pause instead of stop when printer error is detected.")
    return parser.parse_args()


def configure_from_args(args):
    global INPUT_FILE_PATH, IP, SERIAL, ACCESS_CODE, PLATE_NUMBER
    global POLL_INTERVAL, STOP_ON_ERROR, COOLDOWN_BED_TEMP_C, COOLDOWN_TIMEOUT_SEC, RESULT_JSON_PATH
    global WAIT_SUFFER_SEC, PRINT_START_GRACE_SEC, FAILED_STATE_CONFIRM_COUNT

    INPUT_FILE_PATH = args.input_file
    IP = args.ip
    SERIAL = args.serial
    ACCESS_CODE = args.access_code
    PLATE_NUMBER = args.plate_number
    POLL_INTERVAL = args.poll_interval
    STOP_ON_ERROR = not args.pause_on_error
    COOLDOWN_BED_TEMP_C = args.cooldown_bed_temp
    COOLDOWN_TIMEOUT_SEC = args.cooldown_timeout_sec
    RESULT_JSON_PATH = args.result_json
    WAIT_SUFFER_SEC = max(0.0, args.wait_suffer)
    PRINT_START_GRACE_SEC = args.start_grace_sec
    FAILED_STATE_CONFIRM_COUNT = args.failed_confirm_count


def main():
    import bambulabs_api as bl

    if ACCESS_CODE == "PASTE_ACCESS_CODE_HERE":
        raise RuntimeError("Please set BAMBU_ACCESS_CODE or fill ACCESS_CODE in the script.")

    print("Connecting to Bambu Lab printer...")
    print(f"IP: {IP}")
    print(f"Serial: {SERIAL}")

    if not os.path.exists(INPUT_FILE_PATH):
        raise FileNotFoundError(INPUT_FILE_PATH)

    gcode_files = gcode_files_in_3mf(INPUT_FILE_PATH)
    if not gcode_files:
        raise RuntimeError("No Metadata/plate_*.gcode found in 3MF. This is not a sliced 3MF.")

    print(f"Found gcode files in 3MF: {gcode_files}")

    remote_filename = os.path.basename(INPUT_FILE_PATH)
    printer = bl.Printer(IP, ACCESS_CODE, SERIAL)

    try:
        printer.connect()
        time.sleep(5)

        with open(INPUT_FILE_PATH, "rb") as f:
            io_file = BytesIO(f.read())

        print(f"Uploading as remote file: {remote_filename}")
        result = printer.upload_file(io_file, remote_filename)
        print(f"Upload result: {result}")

        if result is None or "226" not in str(result):
            raise RuntimeError("Upload failed. Printer did not return FTP 226.")

        print("Starting print...")
        printer.start_print(
            remote_filename,
            PLATE_NUMBER,
            use_ams=False,
            flow_calibration=False,
        )

        if WAIT_SUFFER_SEC > 0:
            print(
                f"Print command sent. Waiting {WAIT_SUFFER_SEC:g}s for the printer "
                "to replace cached state and error information..."
            )
            time.sleep(WAIT_SUFFER_SEC)

        result = monitor_print(printer)
        if result.get("success"):
            cooldown_result = monitor_cooldown(
                printer,
                target_bed_temp=COOLDOWN_BED_TEMP_C,
                timeout_sec=COOLDOWN_TIMEOUT_SEC,
                poll_interval=POLL_INTERVAL,
            )
            result.update(cooldown_result)
        else:
            result.update({
                "cooldown_ready": False,
                "bed_temperature": None,
                "cooldown_elapsed_sec": 0.0,
                "cooldown_reason": "print_failed",
            })

        print(f"Print result: {result}")
        write_result_json(result, RESULT_JSON_PATH)
        return result

    finally:
        print("Disconnecting printer...")
        try:
            printer.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    configure_from_args(parse_args())
    print_result = main()
    if not print_result or not print_result.get("success"):
        raise SystemExit(1)

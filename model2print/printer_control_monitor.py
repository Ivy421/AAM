from io import BytesIO
import os
import time
import zipfile
import bambulabs_api as bl

# 建议用环境变量保存，避免把 access code 写死在代码里
IP = os.getenv("BAMBU_IP", "10.41.3.35")
SERIAL = os.getenv("BAMBU_SERIAL", "0309AA441000235")
ACCESS_CODE = os.getenv("BAMBU_ACCESS_CODE", "PASTE_ACCESS_CODE_HERE")

INPUT_FILE_PATH = "/home/smmg/AAM/model2print/data/whole_model_sliced.3mf"
PLATE_NUMBER = 1
POLL_INTERVAL = 5  # seconds
STOP_ON_ERROR = False  # True: 出错后 stop_print；False: 出错后 pause_print


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


def is_error_code(error_code) -> bool:
    if error_code is None:
        return False
    if isinstance(error_code, str):
        return error_code.strip() not in ("", "0", "None")
    return error_code != 0


def monitor_print(printer):
    print("Start monitoring print status...")

    while True:
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
            print(f"Printer error detected: {error_code}")
            if STOP_ON_ERROR:
                print("Stopping print...")
                printer.stop_print()
            else:
                print("Pausing print...")
                printer.pause_print()
            break

        state_str = str(state).lower()
        if state_str in ("finish", "finished", "complete", "completed", "idle"):
            print("Print finished or printer returned to idle.")
            break

        try:
            if float(percent) >= 100:
                print("Print progress reached 100%.")
                break
        except Exception:
            pass

        time.sleep(POLL_INTERVAL)


def main():
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

        monitor_print(printer)

    finally:
        print("Disconnecting printer...")
        try:
            printer.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()

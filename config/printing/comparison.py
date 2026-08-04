import json
import sys
from pathlib import Path


DEFAULT_FILE_A = "machine_FULL.json"
DEFAULT_FILE_B = "machine_FULL_old.json"


MISSING = object()


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_value(value, max_len=180):
    if value is MISSING:
        return "<MISSING>"
    text = json.dumps(value, ensure_ascii=False)
    if len(text) > max_len:
        text = text[:max_len] + " ..."
    return text


def compare_json(a, b, path="$", diffs=None):
    if diffs is None:
        diffs = []

    if type(a) is not type(b):
        diffs.append((path, "TYPE_DIFF", a, b))
        return diffs

    if isinstance(a, dict):
        keys = sorted(set(a.keys()) | set(b.keys()))
        for key in keys:
            next_path = f"{path}.{key}"
            va = a.get(key, MISSING)
            vb = b.get(key, MISSING)

            if va is MISSING:
                diffs.append((next_path, "ONLY_IN_B", va, vb))
            elif vb is MISSING:
                diffs.append((next_path, "ONLY_IN_A", va, vb))
            else:
                compare_json(va, vb, next_path, diffs)

    elif isinstance(a, list):
        if len(a) != len(b):
            diffs.append((path, "LIST_LEN_DIFF", len(a), len(b)))

        for i in range(min(len(a), len(b))):
            compare_json(a[i], b[i], f"{path}[{i}]", diffs)

        for i in range(min(len(a), len(b)), len(a)):
            diffs.append((f"{path}[{i}]", "ONLY_IN_A", a[i], MISSING))
        for i in range(min(len(a), len(b)), len(b)):
            diffs.append((f"{path}[{i}]", "ONLY_IN_B", MISSING, b[i]))

    else:
        if a != b:
            diffs.append((path, "VALUE_DIFF", a, b))

    return diffs


def main():
    if len(sys.argv) == 1:
        file_a = Path(DEFAULT_FILE_A)
        file_b = Path(DEFAULT_FILE_B)
    elif len(sys.argv) == 3:
        file_a = Path(sys.argv[1])
        file_b = Path(sys.argv[2])
    else:
        print("用法:")
        print("  python comparison.py")
        print("  python comparison.py machine_FULL.json machine_FULL_old.json")
        sys.exit(1)

    data_a = load_json(file_a)
    data_b = load_json(file_b)

    diffs = compare_json(data_a, data_b)

    print("========== JSON 对比结果 ==========")
    print(f"A: {file_a}")
    print(f"B: {file_b}")

    if not diffs:
        print("两份 JSON 所有字段和值完全一致。")
        return

    print(f"发现不同数量: {len(diffs)}\n")

    for idx, (path, diff_type, va, vb) in enumerate(diffs, start=1):
        print(f"[{idx}] {diff_type}: {path}")
        print(f"  A = {format_value(va)}")
        print(f"  B = {format_value(vb)}")
        print()


if __name__ == "__main__":
    main()

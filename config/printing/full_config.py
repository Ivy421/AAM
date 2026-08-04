import argparse
import json
from copy import deepcopy
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SEARCH_SUBDIRS = ("filament", "process")


def parse_args():
    parser = argparse.ArgumentParser(description="Build a full printing preset config.")
    parser.add_argument(
        "target_preset",
        nargs="?",
        type=Path,
        help="Target preset JSON file supplied externally (positional form).",
    )
    parser.add_argument("--target-preset", dest="target_preset_option", type=Path)
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=SCRIPT_DIR,
        help="Preset root containing filament/ and process/ subfolders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output full config JSON. Default: <target_dir>/<type>_FULL.json.",
    )
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_dict(base, override):
    merged = deepcopy(base)
    for key, value in override.items():
        merged[key] = deepcopy(value)
    return merged


def dependency_not_found(filename):
    print(f"{filename} not exist")
    raise FileNotFoundError(f"{filename} not exist")


def collect_profile_files(profile_dir, target_path):
    files = []
    seen = set()
    search_dirs = [target_path.parent, profile_dir]
    search_dirs.extend(profile_dir / subdir for subdir in SEARCH_SUBDIRS)

    for directory in search_dirs:
        if not directory.is_dir():
            continue
        for path in directory.glob("*.json"):
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                files.append(resolved)
    return files


def build_profile_index(profile_files):
    by_name = {}
    by_stem = {}
    by_filename = {}
    data_by_path = {}

    for path in profile_files:
        data = load_json(path)
        data_by_path[path] = data
        if data.get("name"):
            by_name.setdefault(str(data["name"]), []).append(path)
        by_stem.setdefault(path.stem, []).append(path)
        by_filename.setdefault(path.name, []).append(path)

    return {
        "by_name": by_name,
        "by_stem": by_stem,
        "by_filename": by_filename,
        "data_by_path": data_by_path,
    }


def choose_candidate(candidates, current_path, profile_dir):
    if not candidates:
        return None

    preferred_dirs = [current_path.parent]
    preferred_dirs.extend(profile_dir / subdir for subdir in SEARCH_SUBDIRS)
    preferred_dirs.append(profile_dir)
    for directory in preferred_dirs:
        directory = directory.resolve()
        for candidate in candidates:
            if candidate.parent == directory:
                return candidate
    return candidates[0]


def find_dependency(reference, current_path, profile_dir, index):
    reference = str(reference).strip()
    reference_path = Path(reference)

    direct_candidates = []
    if reference_path.is_absolute():
        direct_candidates.append(reference_path)
    else:
        direct_candidates.extend([
            current_path.parent / reference_path,
            profile_dir / reference_path,
        ])
        direct_candidates.extend(
            profile_dir / subdir / reference_path for subdir in SEARCH_SUBDIRS
        )

    for candidate in direct_candidates:
        if candidate.is_file():
            return candidate.resolve()
        if candidate.suffix.lower() != ".json":
            json_candidate = candidate.with_suffix(".json")
            if json_candidate.is_file():
                return json_candidate.resolve()

    lookup_keys = [reference, reference_path.name, reference_path.stem]
    candidates = []
    for key in lookup_keys:
        candidates.extend(index["by_name"].get(key, []))
        candidates.extend(index["by_filename"].get(key, []))
        candidates.extend(index["by_stem"].get(key, []))

    unique_candidates = list(dict.fromkeys(candidates))
    found = choose_candidate(unique_candidates, current_path, profile_dir)
    if found is None:
        dependency_not_found(reference)
    return found


def normalize_includes(include_value):
    if include_value is None:
        return []
    if isinstance(include_value, str):
        return [include_value]
    if isinstance(include_value, list):
        return include_value
    raise TypeError("include must be a string or a list of strings")


def resolve_profile(path, profile_dir, index, visiting=None):
    path = path.resolve()
    if visiting is None:
        visiting = set()
    if path in visiting:
        raise RuntimeError(f"circular preset dependency: {path.name}")

    visiting.add(path)
    current = index["data_by_path"].get(path)
    if current is None:
        current = load_json(path)
        index["data_by_path"][path] = current

    result = {}

    parent_name = current.get("inherits")
    if parent_name:
        parent_path = find_dependency(parent_name, path, profile_dir, index)
        result = resolve_profile(parent_path, profile_dir, index, visiting)

    for include_name in normalize_includes(current.get("include")):
        include_path = find_dependency(include_name, path, profile_dir, index)
        include_config = resolve_profile(include_path, profile_dir, index, visiting)
        result = merge_dict(result, include_config)

    result = merge_dict(result, current)
    visiting.remove(path)
    return result


def main():
    args = parse_args()
    targets = [path for path in (args.target_preset, args.target_preset_option) if path is not None]
    if len(targets) != 1:
        raise ValueError("Specify target_preset or --target-preset exactly once")
    target_path = targets[0].expanduser().resolve()
    profile_dir = args.profile_dir.expanduser().resolve()

    if not target_path.is_file():
        dependency_not_found(target_path.name)

    profile_files = collect_profile_files(profile_dir, target_path)
    index = build_profile_index(profile_files)
    full_config = resolve_profile(target_path, profile_dir, index)

    target = load_json(target_path)
    preset_type = str(target.get("type") or full_config.get("type") or "process")
    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else target_path.parent / f"{preset_type}_FULL.json"
    )

    full_config["inherits"] = ""
    full_config.pop("include", None)
    full_config["type"] = preset_type
    full_config["name"] = target.get("name", target_path.stem)
    full_config["instantiation"] = "true"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(full_config, f, indent=4, ensure_ascii=False)

    print("Saved:", output_path)
    print("Total keys:", len(full_config))


if __name__ == "__main__":
    main()

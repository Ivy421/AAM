# merge_bambu_machine_config.py
import json
from pathlib import Path
from copy import deepcopy

PROFILE_DIR = Path(r"E:/HKUSTGZ/AAM/config/printing/process")  # 改成你的 json 文件夹
TARGET_NAME = "0.16mm Optimal @BBL A1M"

OUT_FILE = PROFILE_DIR / "Bambu_Lab_A1M_0.16_optimal_FULL.json"


def load_all_profiles(profile_dir):
    profiles = {}
    for p in profile_dir.glob("*.json"):
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        name = data.get("name")
        if name:
            profiles[name] = data
    return profiles


def merge_dict(base, override):
    """override 覆盖 base"""
    merged = deepcopy(base)
    for k, v in override.items():
        merged[k] = deepcopy(v)
    return merged


def resolve_profile(name, profiles, visited=None):
    if visited is None:
        visited = set()

    if name in visited:
        raise RuntimeError(f"循环继承: {name}")

    if name not in profiles:
        raise FileNotFoundError(f"找不到 profile: {name}")

    visited.add(name)
    cur = profiles[name]

    result = {}

    # 1. 先合并父级 inherits
    parent = cur.get("inherits")
    if parent:
        result = resolve_profile(parent, profiles, visited)

    # 2. 再合并 include 模板，如果你有对应模板 json，会自动合并
    for inc in cur.get("include", []):
        if inc in profiles:
            result = merge_dict(result, resolve_profile(inc, profiles, visited))
        else:
            print(f"[WARN] include 模板未找到，跳过: {inc}")

    # 3. 最后当前 profile 覆盖
    result = merge_dict(result, cur)

    visited.remove(name)
    return result


profiles = load_all_profiles(PROFILE_DIR)
full_config = resolve_profile(TARGET_NAME, profiles)

# full config 不再依赖外部 inherits / include
full_config["inherits"] = ""
full_config.pop("include", None)

# 保持它是可实例化的 machine preset
full_config["type"] = "process"
full_config["name"] = TARGET_NAME
full_config["instantiation"] = "true"

with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(full_config, f, indent=4, ensure_ascii=False)

print("Saved:", OUT_FILE)
print("Total keys:", len(full_config))
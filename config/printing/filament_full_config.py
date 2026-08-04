import json
from pathlib import Path
from copy import deepcopy

# =========================
# 1. 文件路径
# =========================
ROOT = Path(r"E:/HKUSTGZ/AAM/config/printing/filament")  # 改成你的文件夹

template_file = ROOT / "filament_pla_template.json"
base_file = ROOT / "Bambu PLA Basic @base.json"
a1m_file = ROOT / "Bambu PLA Basic @BBL A1M.json"

out_file = ROOT / "filament_full.json"


# =========================
# 2. 工具函数
# =========================
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_dict(base, override):
    """后一份 json 覆盖前一份 json"""
    result = deepcopy(base)
    for k, v in override.items():
        result[k] = deepcopy(v)
    return result


# =========================
# 3. 读取并合并
# =========================
template = load_json(template_file)
base = load_json(base_file)
a1m = load_json(a1m_file)

full = merge_dict(template, base)
full = merge_dict(full, a1m)


# =========================
# 4. 整理为 full config
# =========================
full["type"] = "filament"
full["name"] = "Bambu PLA Basic @BBL A1M"
full["inherits"] = ""
full["from"] = "system"
full["instantiation"] = "true"

# PLA 温度范围：190–220 ℃
full["nozzle_temperature"] = ["220"]
full["nozzle_temperature_initial_layer"] = ["220"]
full["nozzle_temperature_range_low"] = ["190"]
full["nozzle_temperature_range_high"] = ["220"]

# 热床最高 60 ℃
full["hot_plate_temp"] = ["60"]
full["hot_plate_temp_initial_layer"] = ["60"]
full["textured_plate_temp"] = ["60"]
full["textured_plate_temp_initial_layer"] = ["60"]

# 风扇速度 100%
full["fan_max_speed"] = ["100"]
full["fan_min_speed"] = ["100"]
full["overhang_fan_speed"] = ["100"]

# 确保兼容 A1 mini
full["compatible_printers"] = [
    "Bambu Lab A1 mini 0.4 nozzle",
    "Bambu Lab A1 mini 0.6 nozzle",
    "Bambu Lab A1 mini 0.8 nozzle"
]


# =========================
# 5. 保存
# =========================
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(full, f, indent=4, ensure_ascii=False)

print("Saved:", out_file)
print("Total keys:", len(full))
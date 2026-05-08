import json
template = "E:/HKUSTGZ/AAM/config/printing/filament.json"
made = "E:/HKUSTGZ/AAM/config/printing/filament/filament_bambu_pla_basic_A1M_full.json"

with open(template, "r", encoding="utf-8") as f:
    d1 = json.load(f)

with open(made, "r", encoding="utf-8") as f:
    d2 = json.load(f)

keys1 = set(d1.keys())
keys2 = set(d2.keys())

print("最外层 key 是否一致：", keys1 == keys2)
print("只在 template:", sorted(keys1 - keys2))
print("只在 made:", sorted(keys2 - keys1))
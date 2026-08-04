import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AI_models.qwen import load_json, qwen3_inference


PROJECT_ROOT = Path(os.getenv("AAM_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
PROMPT_PATH = PROJECT_ROOT / "config" / "prompt" / "confirmation.json"
COARSE_SCAN_DIR = PROJECT_ROOT / "construction" / "data" / "coarse_scan"
ROUGH_CSV_PATH = PROJECT_ROOT / "perception" / "data" / "rough_screening" / "test_RoughInspection.csv"

ROUGH_DEFECT_ID = int(os.getenv("ROUGH_DEFECT_ID", "1"))
QWEN_MODEL_NAME = os.getenv("QWEN_CONFIRMATION_MODEL", "qwen3.7-plus")
QWEN_MAX_TOKENS = int(os.getenv("QWEN_CONFIRMATION_MAX_TOKENS", "512"))

CONFIRMATION_IMAGE_NAMES = [
    "coarse_scan_1.png",
    "coarse_scan_2.png",
    "coarse_scan_3.png",
]

RAW_OUTPUT_PATH = COARSE_SCAN_DIR / "multiview_confirmation_raw.json"
CONFIRMATION_RESULT_PATH = COARSE_SCAN_DIR / "multiview_confirmation.json"


def parse_json_object(text):
    text = str(text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("LLM output does not contain a JSON object.")
    return json.loads(text[start:end + 1])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--defect-id", type=int, default=ROUGH_DEFECT_ID)
    parser.add_argument("--image-paths", type=Path, nargs="+", default=None)
    parser.add_argument("--rough-csv", type=Path, default=ROUGH_CSV_PATH)
    parser.add_argument("--prompt-path", type=Path, default=PROMPT_PATH)
    parser.add_argument("--raw-output-json", type=Path, default=RAW_OUTPUT_PATH)
    parser.add_argument("--output-json", type=Path, default=CONFIRMATION_RESULT_PATH)
    parser.add_argument("--model-name", default=QWEN_MODEL_NAME)
    parser.add_argument("--max-tokens", type=int, default=QWEN_MAX_TOKENS)
    return parser.parse_args()


def paths_from_run_dir(run_dir):
    rough_dir = run_dir / "perception" / "rough_screening"
    confirmation_dir = run_dir / "perception" / "confirmation"
    coarse_dir = run_dir / "construction" / "coarse_scan"
    return {
        "image_paths": [coarse_dir / f"coarse_scan_{idx}.png" for idx in range(1, 4)],
        "rough_csv": rough_dir / f"{run_dir.name}_RoughInspection.csv",
        "raw_output_json": confirmation_dir / "multiview_confirmation_raw.json",
        "output_json": confirmation_dir / "multiview_confirmation.json",
    }


def load_rough_hint(rough_csv_path, defect_id):
    df = pd.read_csv(rough_csv_path)
    if "id" in df.columns:
        row = df[df["id"] == defect_id].iloc[0]
    else:
        row = df.iloc[defect_id - 1]

    return {
        "rough_object_name": str(row.get("object_name", "")),
        "rough_description": str(row.get("description", "")),
    }


def build_messages(args):
    messages = load_json(args.prompt_path)
    run_paths = paths_from_run_dir(args.run_dir) if args.run_dir else {}
    image_paths = args.image_paths or run_paths.get("image_paths") or [COARSE_SCAN_DIR / name for name in CONFIRMATION_IMAGE_NAMES]

    for idx, image_path in enumerate(image_paths):
        messages[1]["content"][idx]["image"] = str(image_path)

    rough_csv = args.rough_csv if args.rough_csv != ROUGH_CSV_PATH else run_paths.get("rough_csv", args.rough_csv)
    rough_hint = load_rough_hint(rough_csv, args.defect_id)
    hint_text = (
        "\n\nRough detection prior hint only, not final evidence. "
        "Use the three images as the final evidence and confirm or reject this hint:\n"
        f"rough_object_name: {rough_hint['rough_object_name']}\n"
        f"rough_description: {rough_hint['rough_description']}"
    )
    messages[2]["content"][0]["text"] += hint_text
    return messages


def main():
    args = parse_args()
    run_paths = paths_from_run_dir(args.run_dir) if args.run_dir else {}
    raw_output_json = args.raw_output_json if args.raw_output_json != RAW_OUTPUT_PATH else run_paths.get("raw_output_json", args.raw_output_json)
    output_json = args.output_json if args.output_json != CONFIRMATION_RESULT_PATH else run_paths.get("output_json", args.output_json)

    messages = build_messages(args)
    output = qwen3_inference(
        messages,
        model_name=args.model_name,
        max_tokens=args.max_tokens,
    )

    raw_output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    raw_output_json.write_text(
        json.dumps(output, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )

    result = parse_json_object(output[0]) if output else {}
    output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )

    print(json.dumps(result, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    main()

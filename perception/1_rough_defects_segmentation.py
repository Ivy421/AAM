import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

sys.path.append('/home/smmg/AAM')
from AI_models.qwen import load_json, qwen3_inference


PROJECT_ROOT = Path(os.getenv("AAM_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
ROOT_DIR = PROJECT_ROOT / "perception" / "result" / "1st_capturing"
PROMPT_PATH = PROJECT_ROOT / "config" / "prompt" / "screening.json"

QWEN_MODEL_NAME = os.getenv("QWEN_SCREENING_MODEL", "qwen3-vl-235b-a22b-thinking")
QWEN_MAX_TOKENS = int(os.getenv("QWEN_SCREENING_MAX_TOKENS", "512"))


def _to_numpy(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        if x.dtype in (torch.bfloat16, torch.float16):
            x = x.float()
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _squeeze_masks(masks):
    masks = _to_numpy(masks)
    if masks is not None and masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    return masks


def init_sam3(confidence_threshold=0.5):
    torch.cuda.empty_cache()
    try:
        model = build_sam3_image_model(enable_inst_interactivity=True)
    except TypeError:
        model = build_sam3_image_model()
    model.eval()
    processor = Sam3Processor(model, confidence_threshold=confidence_threshold)
    return model, processor


def set_sam3_image(processor, image_path):
    image = Image.open(image_path).convert("RGB")
    return processor.set_image(image)


def sam3_text_segment(processor, state, prompt):
    with torch.inference_mode():
        output = processor.set_text_prompt(state=state, prompt=prompt)

    masks = _squeeze_masks(output["masks"])
    boxes = _to_numpy(output["boxes"])
    scores = _to_numpy(output["scores"])
    return masks, boxes, scores


def mask_visualization(image_path, depth_path, mask, box, score):
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    depth = np.nan_to_num(np.load(depth_path), nan=0.0, posinf=0.0, neginf=0.0)
    valid = depth[depth > 0]
    if len(valid) == 0:
        valid = [0, 1]
    d_min, d_max = valid.min(), valid.max()
    d_range = d_max - d_min if d_max != d_min else 1.0

    depth_col = cv2.applyColorMap(
        np.clip((depth - d_min) / d_range * 255, 0, 255).astype(np.uint8),
        cv2.COLORMAP_VIRIDIS,
    )
    depth_col = cv2.cvtColor(depth_col, cv2.COLOR_BGR2RGB)

    mask_2d = np.squeeze(np.array(mask)) > 0
    highlight = np.zeros_like(img)
    highlight[mask_2d] = [255, 0, 0]

    rgb_out = cv2.addWeighted(img, 1.0, highlight, 1, 0)
    depth_out = cv2.addWeighted(depth_col, 1.0, highlight, 0.4, 0)

    x1, y1, x2, y2 = [int(v) for v in box[0]]
    for out in [rgb_out, depth_out]:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            out,
            f"{score[0]:.3f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(rgb_out)
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(depth_out)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def parse_json_array(text):
    text = str(text).strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []
    return json.loads(text[start:end + 1])


def to_confidence(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def pick_best_mask(masks, boxes, scores):
    if scores is None or len(scores) == 0:
        return None, None, None

    best_idx = int(np.argmax(scores))
    return masks[best_idx], boxes[best_idx], scores[best_idx]


def build_sam3_prompt_candidates(prompt_data, object_name="", description=""):
    if isinstance(prompt_data, dict):
        candidates = [
            prompt_data.get("long_defect_prompt", ""),
            prompt_data.get("balanced_prompt", ""),
            prompt_data.get("short_object_prompt", ""),
        ]
    else:
        candidates = [str(prompt_data or "")]

    fallback = f"{object_name} {description}".strip()
    if fallback:
        candidates.append(fallback)

    clean_candidates = []
    for prompt in candidates:
        prompt = str(prompt).strip()
        if prompt and prompt not in clean_candidates:
            clean_candidates.append(prompt)

    return clean_candidates[:3]


def segment_with_prompt_candidates(processor, state, prompt_candidates):
    for prompt in prompt_candidates:
        print(f"SAM3 segmentation prompt: {prompt}")
        masks, boxes, scores = sam3_text_segment(processor, state, prompt)
        mask, box, score = pick_best_mask(masks, boxes, scores)
        if mask is not None:
            return mask, box, score, prompt
        print(f"No valid mask detected for: {prompt}")

    return None, None, None, ""


def build_screening_message(image_path, prompt_path):
    message = load_json(prompt_path)
    message[1]["content"][0]["image"] = str(image_path)
    return message


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--image-path", type=Path, default=None)
    parser.add_argument("--depth-path", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--raw-output", type=Path, default=None)
    parser.add_argument("--mask-dir", type=Path, default=None)
    parser.add_argument("--prompt-path", type=Path, default=PROMPT_PATH)
    parser.add_argument("--model-name", default=QWEN_MODEL_NAME)
    parser.add_argument("--max-tokens", type=int, default=QWEN_MAX_TOKENS)
    return parser.parse_args()


def paths_from_run_dir(run_dir):
    rough_dir = run_dir / "perception" / "rough_screening"
    return {
        "image_path": run_dir / "start.png",
        "depth_path": run_dir / "start.npy",
        "csv_path": rough_dir / f"{run_dir.name}_RoughInspection.csv",
        "raw_output_path": rough_dir / "screening_raw.json",
        "mask_dir": rough_dir,
    }


def default_paths():
    timestamp = (ROOT_DIR / "capturing_timestamp.txt").read_text().strip()
    return {
        "timestamp": timestamp,
        "image_path": ROOT_DIR / f"{timestamp}.png",
        "depth_path": ROOT_DIR / f"{timestamp}.npy",
        "csv_path": ROOT_DIR / f"{timestamp}_RoughInspection.csv",
        "raw_output_path": ROOT_DIR / f"{timestamp}_screening_raw.json",
        "mask_dir": ROOT_DIR,
    }


def main():
    args = parse_args()
    defaults = paths_from_run_dir(args.run_dir) if args.run_dir else None
    if defaults is None and args.image_path is None:
        defaults = default_paths()

    image_path = args.image_path or defaults["image_path"]
    base_dir = image_path.parent
    timestamp = image_path.stem
    depth_path = args.depth_path or (defaults["depth_path"] if defaults else base_dir / f"{timestamp}.npy")
    csv_path = args.output_csv or (defaults["csv_path"] if defaults else base_dir / "rough_inspection.csv")
    raw_output_path = args.raw_output or (defaults["raw_output_path"] if defaults else base_dir / "screening_raw.json")
    mask_dir = args.mask_dir or (defaults["mask_dir"] if defaults else base_dir)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    message = build_screening_message(image_path, args.prompt_path)
    output = qwen3_inference(
        message,
        model_name=args.model_name,
        max_tokens=args.max_tokens,
    )

    raw_output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )

    for item in output:
        print(item)

    screening_results = parse_json_array(output[0]) if output else []
    columns = [
        "time",
        "scene_image",
        "object_name",
        "segmentation_text_prompt",
        "used_segmentation_prompt",
        "description",
        "confidence",
        "box",
        "mask_path",
        "score",
        "reference file",
    ]
    df = pd.DataFrame(columns=columns)

    for i, item in enumerate(screening_results):
        df.loc[i, "time"] = timestamp
        df.loc[i, "scene_image"] = str(image_path)
        df.loc[i, "object_name"] = item.get("object_name", "")
        df.loc[i, "segmentation_text_prompt"] = json.dumps(
            item.get("segmentation_text_prompt", {}),
            ensure_ascii=False,
        )
        df.loc[i, "description"] = item.get("description", "")
        df.loc[i, "confidence"] = to_confidence(item.get("confidence", 0))

    if len(df) == 0:
        df.to_csv(csv_path, index=False)
        print("No defect detected.")
        return

    df = df.sort_values(by="confidence", ascending=False).reset_index(drop=True)
    df = df.astype({"box": "object", "mask_path": "object"})

    sam3_model, sam3_processor = init_sam3()
    sam3_state = set_sam3_image(sam3_processor, image_path)

    for i in range(len(df)):
        prompt_data = json.loads(df.loc[i, "segmentation_text_prompt"])
        prompt_candidates = build_sam3_prompt_candidates(
            prompt_data,
            object_name=df.loc[i, "object_name"],
            description=df.loc[i, "description"],
        )
        mask, box, score, used_prompt = segment_with_prompt_candidates(
            sam3_processor,
            sam3_state,
            prompt_candidates,
        )

        if mask is None:
            print(f"No valid mask detected after trying prompts: {prompt_candidates}")
            continue

        mask_path = mask_dir / f"{timestamp}_mask_{i + 1}.npy"
        np.save(mask_path, mask)

        df.at[i, "box"] = np.asarray(box).flatten().tolist()
        df.at[i, "mask_path"] = str(mask_path)
        df.at[i, "score"] = float(score)
        df.at[i, "used_segmentation_prompt"] = used_prompt

        if depth_path.exists():
            mask_visualization(str(image_path), str(depth_path), [mask], [box], [score])

    df = df.dropna(subset=["score"]).reset_index(drop=True)
    df.to_csv(csv_path, index=False)
    print(df)


if __name__ == "__main__":
    main()

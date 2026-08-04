import gc

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


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
    if masks is None:
        return None

    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]

    return masks


def _as_2d_array(x, dim, name):
    arr = np.asarray(x, dtype=np.float32)

    if arr.ndim == 1:
        arr = arr[None, :]

    if arr.ndim != 2 or arr.shape[1] != dim:
        raise ValueError(f"{name} should have shape [{dim}] or [N, {dim}], got {arr.shape}")

    return arr


def _boxes_to_xyxy_and_norm_cxcywh(box, image_w, image_h, box_format="xyxy"):
    boxes = _as_2d_array(box, 4, "box")

    if box_format == "xyxy":
        x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    elif box_format == "xywh":
        x0, y0 = boxes[:, 0], boxes[:, 1]
        x1 = boxes[:, 0] + boxes[:, 2]
        y1 = boxes[:, 1] + boxes[:, 3]
    else:
        raise ValueError("box_format only supports 'xyxy' or 'xywh'")

    boxes_xyxy = np.stack([x0, y0, x1, y1], axis=1).astype(np.float32)

    cx = (x0 + x1) / 2.0 / image_w
    cy = (y0 + y1) / 2.0 / image_h
    bw = (x1 - x0) / image_w
    bh = (y1 - y0) / image_h

    boxes_norm_cxcywh = np.stack([cx, cy, bw, bh], axis=1).astype(np.float32)
    boxes_norm_cxcywh = np.clip(boxes_norm_cxcywh, 0.0, 1.0)

    return boxes_xyxy, boxes_norm_cxcywh


def _boxes_from_masks(masks):
    masks = _squeeze_masks(masks)

    if masks is None or len(masks) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    boxes = []
    for m in masks:
        m = m.astype(bool)
        ys, xs = np.where(m)

        if len(xs) == 0 or len(ys) == 0:
            boxes.append([0, 0, 0, 0])
        else:
            boxes.append([xs.min(), ys.min(), xs.max(), ys.max()])

    return np.asarray(boxes, dtype=np.float32)


def positioning(
    image_path,
    text=None,
    box=None,
    point=None,
    box_labels=None,
    point_labels=None,
    box_format="xyxy",
    confidence_threshold=0.5,
    multimask_output=False,
):
    """
    SAM3 image segmentation wrapper.

    Supported prompts:
        1. text: text prompt, e.g. "wood block"
        2. box: pixel box prompt, default format [x0, y0, x1, y1]
        3. point: pixel point prompt, e.g. [x, y] or [[x1, y1], [x2, y2]]
    """
    torch.cuda.empty_cache()
    gc.collect()

    try:
        model = build_sam3_image_model(enable_inst_interactivity=True)
    except TypeError:
        model = build_sam3_image_model()

    model.eval()
    processor = Sam3Processor(model, confidence_threshold=confidence_threshold)

    image = Image.open(image_path).convert("RGB")
    image_w, image_h = image.size

    inference_state = processor.set_image(image)

    has_text = text is not None and text != ""
    has_box = box is not None
    has_point = point is not None

    if not has_text and not has_box and not has_point:
        raise ValueError("At least one prompt should be provided: text, box, or point.")

    if has_point:
        if has_text:
            raise NotImplementedError(
                "Current wrapper does not support text + point together. "
                "Use text to get boxes first, then refine with box + point."
            )

        if not hasattr(model, "predict_inst"):
            raise RuntimeError(
                "Current SAM3 model does not have predict_inst. "
                "Please confirm build_sam3_image_model(enable_inst_interactivity=True) is available."
            )

        points = _as_2d_array(point, 2, "point")

        if point_labels is None:
            point_labels = np.ones((len(points),), dtype=np.int32)
        else:
            point_labels = np.asarray(point_labels, dtype=np.int32)

        if len(point_labels) != len(points):
            raise ValueError("point_labels length should match point number.")

        box_arg = None
        if has_box:
            boxes_xyxy, _ = _boxes_to_xyxy_and_norm_cxcywh(
                box, image_w, image_h, box_format=box_format
            )

            if len(boxes_xyxy) != 1:
                raise ValueError("point + box mode expects exactly one box.")

            box_arg = boxes_xyxy[None, 0, :]

        masks, scores, _ = model.predict_inst(
            inference_state,
            point_coords=points,
            point_labels=point_labels,
            box=box_arg,
            multimask_output=multimask_output,
        )

        masks = _squeeze_masks(masks)
        scores = _to_numpy(scores)
        boxes = _boxes_from_masks(masks)

        return masks, boxes, scores

    output = None

    if has_text:
        output = processor.set_text_prompt(
            state=inference_state,
            prompt=text,
        )

    if has_box:
        _, boxes_norm_cxcywh = _boxes_to_xyxy_and_norm_cxcywh(
            box, image_w, image_h, box_format=box_format
        )

        if box_labels is None:
            box_labels = [True] * len(boxes_norm_cxcywh)

        if len(box_labels) != len(boxes_norm_cxcywh):
            raise ValueError("box_labels length should match box number.")

        for one_box, one_label in zip(boxes_norm_cxcywh, box_labels):
            output = processor.add_geometric_prompt(
                state=inference_state,
                box=one_box.tolist(),
                label=bool(one_label),
            )

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

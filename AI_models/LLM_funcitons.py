import torch, gc, json, cv2
from glob import glob
import pandas as pd
from PIL import Image
import pandas as pd
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig


def qwen3_inference(messages):
    torch.cuda.empty_cache()
    gc.collect()
    qwen3_vl_path = '/home/smmg/.cache/modelscope/hub/models/Qwen/Qwen3-VL-8B-Instruct'
    model = AutoModelForImageTextToText.from_pretrained(
        qwen3_vl_path, 
        dtype="auto",
        #quantization_config=quant_config, 
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(qwen3_vl_path)
    with torch.no_grad():
        inputs = processor.apply_chat_template(
        messages,tokenize=True, add_generation_prompt=True,return_dict=True,return_tensors="pt")
        inputs = inputs.to(model.device)
        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=2560) #max_new_tokens=128
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return output

def positioning(image_path, text):
    torch.cuda.empty_cache()
    gc.collect()
    model = build_sam3_image_model()  # eval_mode = True
    processor = Sam3Processor(model)
    # Load image
    image = Image.open(image_path)
    inference_state = processor.set_image(image)
    # Prompt the model with text
    output = processor.set_text_prompt(state=inference_state, prompt=text)
    
    # Get the masks, bounding boxes, and scores
    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
    
    masks = masks.cpu().numpy()
    boxes =boxes.cpu().numpy()
    scores = scores.cpu().numpy()
    return masks, boxes, scores


import cv2
import numpy as np
import matplotlib.pyplot as plt

import cv2
import numpy as np
import matplotlib.pyplot as plt

def mask_visualization(image_path, depth_path, mask, box, score):
    # 1. 加载RGB
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    # 2. 加载深度图 & 转伪彩色
    depth = np.nan_to_num(np.load(depth_path), nan=0.0, posinf=0.0, neginf=0.0)
    valid = depth[depth > 0]
    if len(valid) == 0: valid = [0, 1]  # 防空数组报错保底
    d_min, d_max = valid.min(), valid.max()
    d_range = d_max - d_min if d_max != d_min else 1.0

    depth_col = cv2.applyColorMap(np.clip((depth - d_min) / d_range * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    depth_col = cv2.cvtColor(depth_col, cv2.COLOR_BGR2RGB)

    # 3. 创建Mask高亮层（仅mask区域有颜色，其余全0）
    mask_2d = np.squeeze(np.array(mask)) > 0
    highlight = np.zeros_like(img)
    highlight[mask_2d] = [255, 0, 0]  # 红色

    # 4. 透明叠加：addWeighted 会自动让非mask区域保持原图
    rgb_out = cv2.addWeighted(img, 1.0, highlight, 1, 0)
    depth_out = cv2.addWeighted(depth_col, 1.0, highlight, 0.4, 0)

    # 5. 绘制单框 + 分数
    x1, y1, x2, y2 = [int(v) for v in box[0]]
    for out in [rgb_out, depth_out]:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"{score[0]:.3f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 6. 显示
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1); plt.imshow(rgb_out); plt.axis('off')
    plt.subplot(1, 2, 2); plt.imshow(depth_out); plt.axis('off')
    plt.tight_layout(); plt.show()
    return




def save_json(output, target_path):
    with open(target_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=4)
    return
def load_json(text_path):
    with open(text_path, 'r', encoding='utf-8') as f:
        text = json.load(f)
    return text
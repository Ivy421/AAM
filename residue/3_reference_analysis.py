# %%
import transformers, os, torch, gc, json, cv2, difflib, ast
from glob import glob
import pandas as pd
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image,ImageDraw
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def save_json(output, target_path):
    with open(target_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=4)
    return
def load_json(text_path):
    with open(text_path, 'r', encoding='utf-8') as f:
        text = json.load(f)
    return text

def crop_image(image_path, box):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    padding = 20
    # 解包 box
    x_min, y_min, x_max, y_max = box

    # 扩展边界（注意：x 对应 width，y 对应 height）
    img_h, img_w = img.shape[:2]
    
    new_x_min = max(0, int(x_min - padding))
    new_y_min = max(0, int(y_min - padding))
    new_x_max = min(img_w, int(x_max + padding))
    new_y_max = min(img_h, int(y_max + padding))
    cropped = img[new_y_min:new_y_max, new_x_min:new_x_max]
    
    # show cropped region
    plt.figure(figsize=(8, 6))
    plt.imshow(cropped)
    plt.axis('off')  # 可选：隐藏坐标轴
    plt.show()
    
    return cropped

def qwen3_inference(messages):
    torch.cuda.empty_cache()
    gc.collect()
    model = AutoModelForImageTextToText.from_pretrained(
        "/public/home/rastus/test/downloaded_models/Qwen/Qwen3-VL-8B-Instruct", dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("/public/home/rastus/test/downloaded_models/Qwen/Qwen3-VL-8B-Instruct")
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

def ref_base_extract(dirname):
    base_dir = "/public/home/rastus/dataset/AAM_system/good sample/"
    ref_path = os.path.join(base_dir, dirname)
    files = list(ref_path.iterdir()) if isinstance(ref_path, Path) else os.listdir(ref_path)
    print("Files in reference dir:", [f.name if isinstance(f, Path) else f for f in files])
    
    # 4. 分离图片和 JSON 文件
    image_file = None
    json_file = None
    
    for f in ref_path.iterdir() if isinstance(ref_path, Path) else [Path(ref_path) / f for f in os.listdir(ref_path)]:
        if f.is_file():
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_file = f
            elif f.suffix.lower() == '.json':
                json_file = f
    
    # 5. 验证是否都找到
    if image_file is None or json_file is None:
        raise FileNotFoundError(f"Missing image or JSON in {ref_path}. Found: {image_file}, {json_file}")
    
    # 读取 JSON
    with open(json_file, 'r', encoding='utf-8') as f:
        ref_json = json.load(f)

    return str(image_file), ref_json

# %%
# load data
root_dir = '/home/smmg/AAM/perception/result/1st_capturing/'
csv_files = [f for f in os.listdir(root_dir) if f.endswith('.csv')]
df = pd.read_csv(root_dir + csv_files[0])

# 1. 获取reference base中所有文件名（不含子目录中的文件）
ref_base_path = "/public/home/rastus/dataset/AAM_system/good sample"

dirnames = [
    d for d in os.listdir(ref_base_path)
    if os.path.isdir(os.path.join(ref_base_path, d))
]

for i in range (len(df)):
    matches = difflib.get_close_matches(df.loc[i, 'object_name'], dirnames, n=1, cutoff=0.0) 
    confidence = difflib.SequenceMatcher(None, df.loc[i, 'object_name'], matches[0]).ratio()
    if confidence < 0.6:
        print('no reference good sample!')
    else:
        print('reference sample:', matches[0])
        df.loc[i, 'reference file'] = matches[0]
df.to_csv(root_dir +csv_files[0], index = False)

## crop image and analysis
message = load_json('/public/home/rastus/AAM_system/1_defect_analysis/prompt/reference_analysis.json')
drop_non_repairable = []
for i in range (len(df)): 
    if not pd.isna(df.loc[i, 'reference file']):  ## have reference
        cropped_image = crop_image(df.loc[i,'scene_image'], ast.literal_eval(df.loc[i,'box']))
        ref_image, ref_json = ref_base_extract(df.loc[i, 'reference file'])
        message[1]['content'][0]['image'] = ref_image
        message[1]['content'][1]['image'] = cropped_image
        message[1]['content'][2]['text'] = ref_json
        command = {"type":"text", "text": f"Does the industrial image show {df.loc[i, 'reference file']} ? ,DO ANALYSIS FOR IT."}
        message[1]['content'].append(command)
        output = qwen3_inference(message)
        for p in output:
            print(p)
        output = ast.literal_eval(output[0])
        if(output['3D printing repairable'].lower() == 'no'):
            drop_non_repairable.append(i)
        else:
            df.loc[i, 'component'] = output['defect component']
            
df = df.drop(drop_non_repairable).reset_index(drop=True)
df.to_csv(directory +'/'+ csv_files[0], index = False)

print(df)




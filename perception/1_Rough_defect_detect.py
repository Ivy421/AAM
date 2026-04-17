import os, torch, gc, json, sys
from glob import glob
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AI_models.LLM_funcitons import *

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # 解决内存碎片
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 防止异步内存分配问题

end = 0
#### concat prompt and inference
# load timestamp as filename
root_dir = '/home/smmg/AAM/perception/result/1st_capturing/'
timestamp_path =root_dir + 'capturing_timestamp.txt'
with open(timestamp_path, "r") as f:
        # 读取内容并去除首尾空白字符（如换行符）
        timestamp= f.read().strip()
    
img_path = root_dir + f"{timestamp}.png"
message1 = load_json('/home/smmg/AAM/config/prompt/screening.json')
message1[1]['content'][0]['image'] = img_path
output = qwen3_inference(message1)
for p in output:
    print(p)


output_json = json.loads(output[0])
if len(output_json) > 0 : ## not empty list
    # create an empty df with column name
    columns = ['time','scene_image','object_name', 'object_positioning', 'component', 'description','confidence' ,'box', 'mask', 'score', 'reference file']
    df = pd.DataFrame(columns=columns)
    for i in range (len(output_json)):
        df.loc[i,'time'] = timestamp
        df.loc[i,'scene_image'] = img_path
        df.loc[i,'object_name'] = output_json[i]['object name']
        df.loc[i,'object_positioning'] = output_json[i]['object positioning']
        df.loc[i,'component'] =  output_json[i]['component']
        df.loc[i, 'description'] = output_json[i]['description']
        df.loc[i, 'confidence'] = output_json[i]['confidence']
        
    df.to_csv(f"{root_dir}{timestamp}_RoughInspection.csv", index = False)
    print(df)
else: 
    end = 1
    print('No defect detected.')






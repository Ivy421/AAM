import os, sys
from glob import glob
import pandas as pd
import pandas as pd
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AI_models.LLM_funcitons import *


root_dir = '/home/smmg/AAM/perception/result/1st_capturing/'
csv_files = [f for f in os.listdir(root_dir) if f.endswith('.csv')]
df = pd.read_csv(root_dir + csv_files[0])
if len(df) > 1:
    df = df.sort_values(by='confidence', ascending=False)

# detect position of all rough defects
for i in range (len(df)):
    img_path = str(df.loc[i,'scene_image'])
    mask, box, score = positioning(img_path, str(df.loc[i,'object_positioning']) )
    if len(box) == 0:
        mask, box, score = positioning(img_path, str(df.loc[i,'object_name'] ))
    elif len(box) == 0:
        mask, box, score = positioning(img_path, str(df.loc[i,'description']))
    elif len(box) == 0: print('No defect detected.')
    elif len(score) > 1:  # 对于多个候选区域，选取最高score的区域作为 target
        print('sort the highest mask')
        max_score_idx = np.argmax(score)
        box = [box[max_score_idx]]
        mask = [mask[max_score_idx]]
        score = [score[max_score_idx]]
    elif len(score) == 1:
        mask_visualization(df.loc[i,'scene_image'], mask, box, score)
        df = df.astype({'box':'object'})
        df.at[i,'box'] = box.flatten().tolist()
        df.at[i,'mask'] = mask
        df.at[i,'score'] = score.tolist()[0]
        
# delete the non-detected mask objects
df = df.dropna(subset=['score'])
df.to_csv(root_dir + csv_files[0], index = False)

print(df)

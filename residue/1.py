import os, torch, gc, json, cv2
from glob import glob
import pandas as pd
from PIL import Image,ImageDraw
import pandas as pd
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import matplotlib.pyplot as plt
import numpy as np

torch.cuda.empty_cache()
gc.collect()
image_path = '/home/smmg/AAM/perception/result/1st_capturing/202603164021.png'
model = build_sam3_image_model()  # eval_mode = True
processor = Sam3Processor(model)
# Load image
image = Image.open(image_path)
inference_state = processor.set_image(image)
# Prompt the model with text
output = processor.set_text_prompt(state=inference_state, prompt='cup')

# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

masks = masks.cpu().numpy()
boxes =boxes.cpu().numpy()
scores = scores.cpu().numpy()

if len(scores) > 1:
    max_score_idx = np.argmax(scores)
    box = [boxs[max_score_idx]]
    mask = [masks[max_score_idx]]
    score = [scores[max_score_idx]]    

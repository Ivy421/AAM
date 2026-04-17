import time
from piper_sdk import *
import pyrealsense2 as rs
import numpy as np
import cv2, os, time, subprocess
from datetime import datetime
import pandas as pd

    # 配置管道 (Pipeline)
pipeline = rs.pipeline()
config = rs.config()

# 启用流：彩色 640x480 @ 30fps, 深度 640x480 @ 30fps
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth,640, 480, rs.format.z16, 30)

# 开始管道
pipeline.start(config)

# 创建对齐对象 (将深度图对齐到彩色图)
align = rs.align(rs.stream.color)
save_dir = '/home/smmg/AAM/config/calibration/right_camera/'

def save_frames(save_dir, color_image, image_seq):
    color_path = os.path.join(save_dir, f"{image_seq}.png")
    cv2.imwrite(color_path, color_image)
    return

## piper read and save position
piper = C_PiperInterface_V2('r_piper')
piper.ConnectPort()
count = 0
while True:
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    depth_data = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    if count == 30:
        result = piper.GetArmEndPoseMsgs()
        
        save_frames(save_dir , color_image, 9)
        print(piper.GetArmEndPoseMsgs())
        time.sleep(1)

    if count > 30:
        break

    else:
        time.sleep(0.1)
        count += 1

endpose = result.end_pose
data = {
    'X_axis': [endpose.X_axis],
    'Y_axis': [endpose.Y_axis],
    'Z_axis': [endpose.Z_axis],
    'RX_axis': [endpose.RX_axis],
    'RY_axis': [endpose.RY_axis],
    'RZ_axis': [endpose.RZ_axis]
}

df = pd.DataFrame(data)
df0 = pd.read_csv('endpose.csv')
df0 = pd.concat([df0, df], ignore_index=True)
df0.to_csv('endpose.csv', index=False)
    
    
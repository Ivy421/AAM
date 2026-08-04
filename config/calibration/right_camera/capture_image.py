# %%
import sys, time, json, os
sys.path.append('/home/smmg/AAM')
from camera.camera_functions import *
from piper_motion.piper_functions import *
from pathlib import Path
import pandas as pd


# %%
# piper = disable('r_piper')
#piper = enable('r_piper')


#%%
## 拍照
img_save_path = '/home/smmg/AAM/config/calibration/right_camera/image2/'
capture(img_save_path, save_file_name='20' ,AUTO_SAVE_INTERVAL=2.0, MAX_SAVE_FRAMES = 1, SAVE_CONFIG = 0)
endpose = get_endpose(piper)
data = [endpose.end_pose.X_axis,
        endpose.end_pose.Y_axis,
        endpose.end_pose.Z_axis,
        endpose.end_pose.RX_axis,
        endpose.end_pose.RY_axis,
        endpose.end_pose.RZ_axis]
df = pd.DataFrame([data])
# df0 = pd.read_csv('/home/smmg/AAM/config/calibration/right_camera/endpose1.csv')
df.to_csv('/home/smmg/AAM/config/calibration/right_camera/endpose2.csv', mode='a', header=False, index = False)
# %%

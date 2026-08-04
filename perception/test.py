
# %%
import sys, time, json, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from camera.camera_functions import *
from piper_motion.piper_functions import *
from pathlib import Path

#%%
config = np.load('/home/smmg/AAM/perception/result/1st_capturing/camera_config.npy',allow_pickle=True).item()
color_intr = config['color_intrinsic']
depth_intr = config['depth_intrinsic']
print(color_intr)
print(depth_intr)

# %%
#piper = disable('r_piper')
piper = enable('r_piper')
go_zero(piper)

#move_to_pos(piper,55,0,255,45,45,0)
endpose = get_endpose(piper)
print(endpose)

#%%
## 拍照
img_save_path = '/home/smmg/AAM/config/calibration/right_camera/images/'
capture(img_save_path, save_file_name='1' ,AUTO_SAVE_INTERVAL=2.0, MAX_SAVE_FRAMES = 1, SAVE_CONFIG = 1)

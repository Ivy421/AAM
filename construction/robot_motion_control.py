import sys, os, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from piper_motion.piper_functions import *
from camera.camera_functions import *
import time
from pathlib import Path

middle_scan_pose_path = str(Path(__file__).parent) +'/result/middle_scan_pose.json'
top_scan_pose_path = str(Path(__file__).parent) +'/result/top_scan_pose.json'
with open (middle_scan_pose_path,'r', encoding = 'utf-8') as f:
    middle_scan_pose = json.load(f)
with open (top_scan_pose_path,'r', encoding = 'utf-8') as f:
    top_scan_pose = json.load(f)

img_save_path = str(Path(__file__).parent)+'/result/'

piper = enable('r_piper')
piper = go_zero(piper)
time.sleep(1)
image_seq = 1
jump_pose = []
scale_to_mm = 1000 # 米转换成毫米
for idx, scan_pose in enumerate(middle_scan_pose):
    print(f"the {idx+1} scan point is working")
    x,y,z,rx,ry,rz = [val * 1000 for val in scan_pose]
    print(x,y,z,rx,ry,rz, '输入单位是mm')
    piper = move_to_pos(piper,x,y,z,rx,ry,rz)  #30,-80,300,0,85,-45  x,y,z,rx,ry,rz
    arm_status = get_arm_status(piper)
    if arm_status.arm_status != '0x00':
        print('unexpected arm status:', arm_status.arm_status)
        jump_pose.append([x,y,z,rx,ry,rz])
        break
    
    capture(img_save_path, save_file_name = str(image_seq), AUTO_SAVE_INTERVAL=2.0, MAX_SAVE_FRAMES = 1)
    image_seq +=1
    time.sleep(0.1)

#time.sleep(1)

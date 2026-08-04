import sys, time, json, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from camera.camera_functions import *
from piper_motion.piper_functions import *

piper = disable('r_piper')
piper = enable('r_piper')
piper = go_zero(piper)
time.sleep(1)
piper = move_to_pos(piper,110,-106,218,117,60,62) 
time.sleep(4)
img_save_path = '/home/smmg/AAM/perception/result/1st_capturing/'
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
capture(img_save_path, save_file_name=timestamp)
endpose_info = get_endpose(piper)
print(endpose_info)
data = {'x': endpose_info.end_pose.X_axis,
        'y':endpose_info.end_pose.Y_axis,
        'z':endpose_info.end_pose.Z_axis,
        'rx':endpose_info.end_pose.RX_axis,
        'ry':endpose_info.end_pose.RY_axis,
        'rz':endpose_info.end_pose.RZ_axis,
         'timestamp': timestamp }


with open(img_save_path+timestamp+'.json', 'w',encoding='utf-8') as f:
    json.dump(data,f,ensure_ascii=False,indent=4)
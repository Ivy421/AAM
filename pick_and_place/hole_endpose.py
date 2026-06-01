import json
import numpy as np
from scipy.spatial.transform import Rotation as R

grip_meta_path = 'E:\HKUSTGZ\AAM/construction/data/completion_result/grip_meta.npz'
grip_meta = np.load(grip_meta_path)
grip_dist_len = grip_meta['grip_dist_len']
grip_dist_width = grip_meta['grip_dist_width']

top_plane_middle_x = grip_dist_width
top_plane_middle_y = -(grip_dist_len + (grip_meta['handle_length'] + grip_meta['neck_length']) / 2)
top_plane_middle_z = grip_meta['handle_thickness']


## 在打印平台坐标系下的grip ROI四个角点的表示
p_top_lfet_back = np.array([])
p_top_right_back = np.array([])
p_top_left_front = np.array([])
p_top_right_front = np.array([])


#### p 代表打印机平台坐标系，b代表机械臂base坐标系 #######

theta = np.deg2rad(90)
Rz = np.array([
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta),  np.cos(theta), 0],
    [0,              0,             1]
])
bp_t = np.array([[-130],[200],[20]])  ## 后续要修改这里的绝对固定值
bpT = np.column_stack([Rz, bp_t])
bpT = np.vstack([bpT,np.array([0,0,0,1])])
p_M = np.array([top_plane_middle_x, top_plane_middle_y, top_plane_middle_z, 1])  ## mm 单位
#print(bpT, p_M)
b_M = bpT @ p_M

#### pre grasp point 输入到piper_sdk控制的值
b_E = np.array([b_M[0], b_M[1], b_M[2], 1])

b_E_pos  = [b_M[0], b_M[1], b_M[2]+200,180,0,0]
print(b_E_pos)

#### pre grasp point 输入到gazebo moveit控制的值
quat = R.from_euler('xyz', [180, 0, 0], degrees=True).as_quat()

target_orientation = quat.tolist()
print(target_orientation)
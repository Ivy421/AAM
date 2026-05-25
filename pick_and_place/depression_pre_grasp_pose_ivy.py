import json
import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d

depression_dir = 'E:\HKUSTGZ\AAM/construction/data/completion_result/depression'
pick_dir = 'E:\HKUSTGZ\AAM/pick_and_place/data/pick'
model_path = depression_dir + '/print_model.stl'
pcd_path = depression_dir + '/model_oriented.pcd'

#### p 代表打印机平台坐标系，b代表机械臂base坐标系 #######
theta = np.deg2rad(90)
Rz = np.array([
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta),  np.cos(theta), 0],
    [0,              0,             1]
])
bp_t = np.array([[-240],[240],[50]])  ## 后续要修改这里的绝对固定值
bpT = np.column_stack([Rz, bp_t])
bpT = np.vstack([bpT,np.array([0,0,0,1])])
pre_grasp_height = 200

######### -- 抓取位置定义为立侧面的垂直中线，最终下降到一半的z高度 #######
pcd = o3d.io.read_point_cloud(pcd_path)
points = np.asarray(pcd.points)

## pcd文件的点以m为单位，转换为mm
unit_scale = 1000
min_y = np.min(points[:, 1]) * unit_scale
max_x = np.max(points[:, 0]) * unit_scale
min_x = np.min(points[:, 0]) * unit_scale
max_z = np.max(points[:, 2]) * unit_scale
p_pick_pos = np.array([ (max_x+min_x)/2 , min_y, max_z/2, 1 ])
print(p_pick_pos)
b_pick_pos = bpT @ p_pick_pos
b_pick_pos  = [b_pick_pos[0], b_pick_pos[1], b_pick_pos[2]+pre_grasp_height,180,0,0]
with open (pick_dir+'pre_grasp_pose.json','w',encoding='utf-8') as f:
    json.dump(b_pick_pos,f)
print(b_pick_pos)
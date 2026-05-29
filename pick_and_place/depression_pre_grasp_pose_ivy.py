import json
import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d

def base_end_grab_trans(endpose):
    t = np.array([[endpose[0]],
                 [endpose[1]],
                 [endpose[2]]] )
    Rot = R.from_euler('xyz',[endpose[3], endpose[4], endpose[5]], degrees = True).as_matrix()
    T = np.column_stack([Rot,t])
    T = np.vstack([T,np.array([0,0,0,1])])
    return T

def trans_to_endpose(T):
    """
    4x4变换矩阵 -> 末端位姿 [x, y, z, rx, ry, rz]
    欧拉角顺序: xyz
    角度单位: degree
    """
    T = np.asarray(T, dtype=float)

    pos = T[:3, 3]
    rot_mat = T[:3, :3]

    euler = R.from_matrix(rot_mat).as_euler('xyz', degrees=True)

    endpose_fix = np.concatenate([pos, euler])
    return endpose_fix


arm_gripper_length_z = 142.5 # mm 末端Z轴朝前伸出方向
arm_gripper_width_y = 164 # mm  ## 夹爪开合方向
arm_gripper_thickness =75 # mm 
arm_flange = 10.5 # mm
 
depression_dir = 'E:\HKUSTGZ\AAM/construction/data/completion_result/depression'
pick_dir = 'E:\HKUSTGZ\AAM/pick_and_place/data/pick'

model_path = depression_dir + '/model_oriented.stl'
pcd_path = depression_dir + '/model_oriented.pcd'
meta = np.load(depression_dir + '/meta.npz', allow_pickle=True)
orient_meta = np.load(depression_dir + '/orientation_meta.npz', allow_pickle=True)
grip_meta = np.load(depression_dir + '/gripper_meta.npz', allow_pickle=True)
unit_scale = 1000

attach_center = np.asarray(orient_meta['attach_center_oriented'] ) * unit_scale
full_box_z_height = np.asarray(orient_meta['full_box_z_height'] ) * unit_scale
grip_height_total = (grip_meta['grip_body_height'] + grip_meta['base_height'] + grip_meta['v_neck_height']) * unit_scale

####
#### 打印物体抓取坐标系相对base坐标系的变换矩阵 #######
####

theta = np.deg2rad(90)
Rz = np.array([
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta),  np.cos(theta), 0],
    [0,              0,             1]
])
b_obj_grab_t = np.array([[-240],[240],[50 + full_box_z_height/2 ]])  ## 固定值代表打印平台原点在base坐标系下的表达
b_obj_grab_T = np.column_stack([Rz, b_obj_grab_t])
b_obj_grab_T = np.vstack([b_obj_grab_T,np.array([0,0,0,1])])


####
#### 定义抓取位姿: x, y follow attach_center, z 下降到GRIP的一半长度处
####

pre_grab_height = 200
# 在object_grab坐标系下的抓取位置(仅考虑gripper)

#p_grab_pos = np.array([attach_center[0], top_plane_center[1], top_plane_center[2]*2 + grip_height_total/2 ,1])
obj_grabx = attach_center[0]
obj_graby = attach_center[1] 
obj_grabz = (attach_center[2]  +grip_height_total/2 )
obj_grab_pos = np.array([obj_grabx, obj_graby , obj_grabz, 1 ])
# print('objecy grab pos:', obj_grab_pos)

#base坐标系下的抓取endpose
b_grab_pos = b_obj_grab_T @ obj_grab_pos
print('base grab pos:', b_grab_pos)
b_pre_end_grab  = np.array([b_grab_pos[0], b_grab_pos[1], b_grab_pos[2]+pre_grab_height,180,0,0])# mm,degree
b_end_grab =  np.array([b_grab_pos[0], b_grab_pos[1], b_grab_pos[2] + arm_gripper_length_z + arm_flange,180,0,0])
print('grab endpose:', b_end_grab)

#抓取末端姿态在base下的变换矩阵
base_end_grab_T = base_end_grab_trans(b_end_grab)

# object_grab 在 末端抓取时刻坐标系下的表示
end_grab_obj_grab_T = np.linalg.inv(base_end_grab_T) @ b_obj_grab_T 

#### 计算base_obj_fix_T
base_obj_fix_R = orient_meta['R']
base_obj_fix_t = orient_meta['t']
base_obj_fix_t = base_obj_fix_t * unit_scale
base_obj_fix_T = np.column_stack([base_obj_fix_R, base_obj_fix_t])
base_obj_fix_T = np.vstack([base_obj_fix_T, np.array([0,0,0,1])])

##########取逆
base_obj_fix_T = np.linalg.inv(base_obj_fix_T)

### 计算最终的安装旋转矩阵
base_end_fix_T =  base_obj_fix_T @ np.linalg.inv(end_grab_obj_grab_T)
endpose_fix = trans_to_endpose(base_end_fix_T)
print('fix endpose:', endpose_fix)

np.savez(
    pick_dir + '/pick_place_pose.npz',
    pre_pick_endpose = b_pre_end_grab,
    pick_endpose = b_end_grab,
    fix_endpose = endpose_fix
    )


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

depression_dir = 'E:\HKUSTGZ\AAM/construction/data/completion_result/depression'
pick_dir = 'E:\HKUSTGZ\AAM/pick_and_place/data/pick'

model_path = depression_dir + '/model_oriented.stl'
pcd_path = depression_dir + '/model_oriented.pcd'

## 抓取位置位于顶面中点
orient_mata = np.load(depression_dir + '/orientation_meta.npz',allow_pickle=True)
top_plane_center = orient_mata['top_plane_center_oriented']

## 抓取高度还要加上半个grip height
grip_meta = np.load(depression_dir + '/gripper_meta.npz',allow_pickle=True)
grip_height_total = (grip_meta['grip_body_height'] + grip_meta['base_height'] + grip_meta['v_neck_height']) * 1000

arm_gripper_length_z = 142.5 # mm 末端Z轴朝前伸出方向
arm_gripper_width_y = 164 # mm  ## 夹爪开合方向
arm_gripper_thickness =75 # mm 
#### p 代表打印机平台坐标系，b代表机械臂base坐标系 #######
theta = np.deg2rad(90)
Rz = np.array([
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta),  np.cos(theta), 0],
    [0,              0,             1]
])
b_obj_grab_t = np.array([[-240],[240],[50]])  ## 后续要修改这里的绝对固定值
b_obj_grab_T = np.column_stack([Rz, b_obj_grab_t])
b_obj_grab_T = np.vstack([b_obj_grab_T,np.array([0,0,0,1])])
pre_grab_height = 200

######### -- 抓取位置定义为立侧面的垂直中线，最终下降到一半的z高度 #######
pcd = o3d.io.read_point_cloud(pcd_path)
points = np.asarray(pcd.points)

## pcd文件的点以m为单位，转换为mm
unit_scale = 1000
#min_y = np.min(points[:, 1]) * unit_scale
#max_x = np.max(points[:, 0]) * unit_scale
#min_x = np.min(points[:, 0]) * unit_scale
#max_z = np.max(points[:, 2]) * unit_scale

# 在object_grab坐标系下的抓取位置(仅考虑gripper)
#p_grab_pos = np.array([ (max_x+min_x)/2 , min_y, max_z/2, 1 ])
p_grab_pos = np.array([top_plane_center[0], top_plane_center[1], top_plane_center[2] + grip_height_total/2 ,1])

#base坐标系下的抓取endpose
b_grab_pos = b_obj_grab_T @ p_grab_pos
b_pre_end_grab  = np.array([b_grab_pos[0], b_grab_pos[1], b_grab_pos[2]+pre_grab_height,180,0,0])# mm,degree
b_end_grab =  np.array([b_grab_pos[0], b_grab_pos[1], b_grab_pos[2] + arm_gripper_length_z,180,0,0])
print('grab endpose:', b_end_grab)
#抓取末端姿态在base下的变换矩阵
base_end_grab_T = base_end_grab_trans(b_end_grab)

# object_grab 在 末端抓取时刻坐标系下的表示
end_grab_obj_grab_T = np.linalg.inv(base_end_grab_T) @ b_obj_grab_T 

#### 计算base_obj_fix_T
orient_meta = np.load(depression_dir + '/orientation_meta.npz', allow_pickle=True)
base_obj_fix_R = orient_meta['R']
base_obj_fix_t = orient_meta['t']
base_obj_fix_t = base_obj_fix_t * 1000
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


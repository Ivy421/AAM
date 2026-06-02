import json
import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from reachability_test import *


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

def _is_reachable(result):
    if not isinstance(result, dict):
        raise TypeError("reachability_test(endpose) must return a dict")
    return bool(result.get("reachable", False))


def define_pregrab_endpose(endpose):
    """
    Generate pre-grasp candidate endposes by lifting the target grasp endpose
    along base Z. Keep x, y, rx, ry, rz unchanged.

    Candidate order, in mm:
        z + 50, z + 40, z + 60, z + 30, z + 25, z + 20

    The first reachable candidate is returned. If none is reachable, return None.
    """
    z_offsets = [50.0, 40.0, 60.0, 30.0, 25.0, 20.0]

    print("\n========== Pre-grab reachability search ==========")
    for dz in z_offsets:
        candidate = endpose.copy()
        candidate[2] += dz

        result = reachability_test(candidate)
        reachable = _is_reachable(result)

        pos_err = result.get("pos_err_mm", None) if isinstance(result, dict) else None
        rot_err = result.get("rot_err_deg", None) if isinstance(result, dict) else None
        if pos_err is not None and rot_err is not None:
            print(f"pregrab dz={dz:+.1f} mm -> reachable={reachable}, "
                  f"pos_err={pos_err:.3f} mm, rot_err={rot_err:.3f} deg")
        else:
            print(f"pregrab dz={dz:+.1f} mm -> reachable={reachable}")

        if reachable:
            print("Selected pregrab endpose:", candidate)
            return candidate

    print("Warning: no reachable pregrab endpose found.")
    return None


def define_prefix_endpose(endpose):
    """
    Generate pre-fix / pre-install candidate endposes in front of the final
    installation endpose. Keep z, rx, ry, rz unchanged.

    x offsets are tried in this order:
        -50, -40, -30, -60, -25, -20  mm

    y offset direction depends on the sign of input y:
        if y > 0:  -50, -40, -30, -60, -25, -20  mm
        if y <= 0: +50, +40, +30, +60, +25, +20  mm

    Search strategy:
        1. Try coupled offsets first: (dx_i, dy_i).
           This keeps the approach direction stable and usually gives a
           cleaner pre-install pose.
        2. If all coupled candidates fail, try the decoupled full grid
           dx_i × dy_j, preserving the same offset priority order.

    The first reachable candidate is returned. If none is reachable, return None.
    """

    x_offsets = [-50.0, -40.0, -30.0, -60.0, -25.0, -20.0]
    y_base = [-50.0, -40.0, -30.0, -60.0, -25.0, -20.0]
    if endpose[1] < 0:
        y_offsets = [-v for v in y_base]
    else:
        y_offsets = y_base

    # Coupled candidates first, then full grid candidates.
    candidates = []
    used = set()

    for dx, dy in zip(x_offsets, y_offsets):
        key = (dx, dy)
        candidates.append(key)
        used.add(key)

    for dx in x_offsets:
        for dy in y_offsets:
            key = (dx, dy)
            if key not in used:
                candidates.append(key)
                used.add(key)

    print("\n========== Pre-fix reachability search ==========")
    print("Search mode: coupled candidates first, then decoupled dx-dy grid")

    for dx, dy in candidates:
        candidate = endpose.copy()
        candidate[0] += dx
        candidate[1] += dy

        result = reachability_test(candidate)
        reachable = _is_reachable(result)

        pos_err = result.get("pos_err_mm", None) if isinstance(result, dict) else None
        rot_err = result.get("rot_err_deg", None) if isinstance(result, dict) else None
        if pos_err is not None and rot_err is not None:
            print(f"prefix dx={dx:+.1f} mm, dy={dy:+.1f} mm -> reachable={reachable}, "
                  f"pos_err={pos_err:.3f} mm, rot_err={rot_err:.3f} deg")
        else:
            print(f"prefix dx={dx:+.1f} mm, dy={dy:+.1f} mm -> reachable={reachable}")

        if reachable:
            print("Selected prefix endpose:", candidate)
            return candidate

    print("Warning: no reachable prefix endpose found.")
    return None

def round_endpose(endpose):
    """
    Keep each element of an endpose to 2 decimal places.
    endpose format: [x, y, z, rx, ry, rz]
    """
    if endpose is None:
        return None
    return np.round(np.asarray(endpose, dtype=float).reshape(-1), 2)


arm_gripper_length_z = 142.5 # mm 末端Z轴朝前伸出方向
arm_gripper_width_y = 164 # mm  ## 夹爪开合方向
arm_gripper_thickness =75 # mm 
arm_flange = 10.5+2 # mm
 
depression_dir = 'E:\HKUSTGZ\AAM/construction/data/completion_result/depression'
pick_dir = 'E:\HKUSTGZ\AAM/pick_and_place/data'

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
b_obj_grab_t = np.array([[-200],[200],[-20 + full_box_z_height/2 ]])  ## 固定值代表打印平台原点在base坐标系下的表达
b_obj_grab_T = np.column_stack([Rz, b_obj_grab_t])
b_obj_grab_T = np.vstack([b_obj_grab_T,np.array([0,0,0,1])])


####
#### 定义抓取位姿: x, y follow attach_center, z 下降到GRIP的一半长度处
####

# 在object_grab坐标系下的抓取位置(仅考虑gripper)

#p_grab_pos = np.array([attach_center[0], top_plane_center[1], top_plane_center[2]*2 + grip_height_total/2 ,1])
obj_grabx = attach_center[0]
obj_graby = attach_center[1] 
obj_grabz = (attach_center[2]  +grip_height_total/2 )
obj_grab_pos = np.array([obj_grabx, obj_graby , obj_grabz, 1 ])
# print('objecy grab pos:', obj_grab_pos)

#base坐标系下的抓取endpose
b_grab_pos = b_obj_grab_T @ obj_grab_pos
#print('base grab pos:', b_grab_pos)
#b_pre_end_grab  = np.array([b_grab_pos[0], b_grab_pos[1], b_grab_pos[2]+pre_grab_height,180,0,0])# mm,degree
grab_endpose =  np.array([b_grab_pos[0], b_grab_pos[1], b_grab_pos[2] + arm_gripper_length_z + arm_flange,180,0,0])
print('grab endpose:', grab_endpose)

#抓取末端姿态在base下的变换矩阵
base_end_grab_T = base_end_grab_trans(grab_endpose)

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
fix_endpose = trans_to_endpose(base_end_fix_T)
print('fix endpose:', fix_endpose)

pick_reachable = reachability_test(grab_endpose)
fix_reachable = reachability_test(fix_endpose)

######### 这里如果希望做的更robust，可以加入条件语句，若true，若false则调动底盘移动计算
############### 先默认都可达

if (pick_reachable['reachable'] == True) and (fix_reachable['reachable'] == True):
    print('reachable!')
    pregrab_endpose = round_endpose(define_pregrab_endpose(grab_endpose))
    prefix_endpose = round_endpose(define_prefix_endpose(fix_endpose))
    grab_endpose = round_endpose(grab_endpose)
    fix_endpose = round_endpose(fix_endpose)

    print('pregrab_endpose:',pregrab_endpose, 
          '\n prefix_endpose:' , prefix_endpose,
          '\n grab_endpose: ', grab_endpose,
          '\n fix_endpose: ', fix_endpose)

    #________________ save endpose data ______________#
    np.savez( pick_dir + '/pick_place_endpose.npz',
    pregrab_endpose = pregrab_endpose,
    prefix_endpose = prefix_endpose,
    grab_endpose = grab_endpose,
    fix_endpose = fix_endpose   )

elif pick_reachable['reachable'] == False:
    print('grab endpose 不可达')
elif fix_reachable['reachable'] == False:
    print('grab endpose 不可达')

import numpy as np
from scipy.spatial.transform import Rotation as R
from piper_sdk import *

def get_transformation_matrix(piper):
    # 1. 调用 GetFK 获取末端位姿
    fk_data = piper.GetFK(mode="feedback")  # 或 "control"
    
    # 2. 获取关节6相对于基座的位姿 (最后一行)
    end_effector_pose = fk_data[-1]  # [X, Y, Z, RX, RY, RZ]
    
    # 3. 单位转换
    x = end_effector_pose[0] * 0.001  # mm -> m
    y = end_effector_pose[1] * 0.001
    z = end_effector_pose[2] * 0.001
    
    rx = end_effector_pose[3] * 0.001  # degrees
    ry = end_effector_pose[4] * 0.001
    rz = end_effector_pose[5] * 0.001
    
    # 4. 将欧拉角转换为旋转矩阵
    rotation = R.from_euler('xyz', [rx, ry, rz], degrees=True)
    rotation_matrix = rotation.as_matrix()
    
    # 5. 构建 4x4 齐次变换矩阵
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix  # 旋转部分
    transformation_matrix[:3, 3] = [x, y, z]         # 平移部分
    
    return transformation_matrix

# 使用示例
piper = C_PiperInterface_V2('r_piper')
piper.ConnectPort()
T_base_to_end = get_transformation_matrix(piper)
print("变换矩阵:\n", T_base_to_end)
print(f"位置: {T_base_to_end[:3, 3]}")

data = np.load('/home/smmg/AAM/config/calibration/right_camera/R_camara_paras.npy', allow_pickle=True)
params = data.item()
K = params['intrinsic_mtx:']  # 确认键名是否正确
dist = params['dist']
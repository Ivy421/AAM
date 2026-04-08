# %%
import cv2
import numpy as np
import pandas as pd
import glob
import os
import os, json, cv2, sys
sys.path.append('/home/smmg/AAM')
from camera.camera_functions import *
from scipy.spatial.transform import Rotation as R

# ========================
# 1. 配置路径和参数
# ========================
# 文件夹路径
img_dir = r"/home/smmg/AAM/config/calibration/right_camera/images"
endpose_path = '/home/smmg/AAM/config/calibration/right_camera/endpose.csv'
image_pattern = os.path.join(img_dir, "*.png")      # 图片匹配模式 (根据你的图片格式修改，如 .png)

config = np.load(img_dir+'/camera_config.npy',allow_pickle=True).item()
color_intr = config['color_intrinsic']
depth_intr = config['depth_intrinsic']
# print(color_intr)

# 棋盘格参数
checkerboard_size = (9, 6)   # 内角点数量 (列, 行)
square_size = 0.023          # 棋盘格方格物理尺寸 (米) 2.31cm=0.0231m

## 内参矩阵
k = np.array([[color_intr['fx'], 0, color_intr['ppx']],
              [0, color_intr['fy'], color_intr['ppy']],  #color_intr['ppy']
              [0, 0, 1]], dtype=np.float64)
dist = np.array(color_intr['coeffs'])

#%%
# ========================
# 2. 计算base to end 的变换矩阵
# ========================
try:
    df = pd.read_csv(endpose_path)
    raw_data = df.values 
    img_name = (df.index+1).tolist()
    
    A_rvecs = []
    A_tvecs = []

    print(f"成功读取 {len(raw_data)} 组末端姿态数据。开始单位转换...")

    for i, row in enumerate(raw_data):
        # 解析原始数据
        x_raw, y_raw, z_raw = row[0], row[1], row[2]
        rx_raw, ry_raw, rz_raw = row[3], row[4], row[5]
        
        # 1. 位置转换: 0.001mm -> m
        # 1 mm = 0.001 m, 所以 0.001 mm = 0.000001 m (1e-6)
        t_vec = np.array([x_raw, y_raw, z_raw], dtype=np.float64) /1000000

        # 2. 旋转转换: 0.001 degrees

        #### 大写是绕定轴，小写绕动轴
        r_scipy = R.from_euler('XYZ', [rx_raw/1000, ry_raw/1000, rz_raw/1000], degrees=True)
    
        # 3. 获取旋转矩阵 (3x3)
        R_mat = r_scipy.as_matrix()

        A_rvecs.append(R_mat)
        A_tvecs.append(t_vec)

except Exception as e:
    print(f"❌ 读取 Excel 失败: {e}")
    print("请检查文件路径是否正确，以及是否安装了 openpyxl 库 (pip install openpyxl)")
    exit()

#%%
# ========================
# 3. 提取图像角点并计算 B 矩阵 (相机->标定板)
# ========================
object_points = np.zeros((np.prod(checkerboard_size), 3), np.float32)
object_points[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
object_points *= square_size
B_rvecs = []
B_tvecs = []
all_detected_corners = []
img_paths = []
for prex_name in img_name:
    img_paths.append(str(prex_name)+'.png')
valid_count = 0
for i, img_path in enumerate(img_paths):
    if i >= len(A_rvecs): break # 防止图片多于姿态数据
    
    img = cv2.imread(img_dir + '/'+img_path)
    if img is None: continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
    if ret:
        # 亚像素优化
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        #cv2.drawChessboardCorners(img, (9,6), corners, ret)
        #cv2.imshow('img', img)
        #cv2.waitKey(1000)
        # 求解 PnP (得到相机相对于标定板的位姿)
        ret, rvec, tvec = cv2.solvePnP(object_points, corners, k, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        
        if ret:
            R_mat, _ = cv2.Rodrigues(rvec)
            B_rvecs.append(R_mat)
            B_tvecs.append(tvec)
            all_detected_corners.append(corners)
            valid_count += 1
    else:
        print('failed image:', img_path)

print(f"✅ 成功处理 {valid_count} 组有效数据用于标定。")
A_rvecs_final = A_rvecs[:valid_count]
A_tvecs_final = A_tvecs[:valid_count]

# %%

# ========================
# 4. 执行手眼标定 (EYE-IN-HAND)
# ========================
# 方法选择: CALIB_HAND_EYE_TSAI 是最常用的方法
R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
    A_rvecs_final, A_tvecs_final,
    B_rvecs, B_tvecs,
    cv2.CALIB_HAND_EYE_TSAI
)

# ========================
# 5. 输出结果
# ========================
print("相机到末端执行器的变换 (Camera -> Gripper):")
print("\n旋转矩阵 R:")
print(R_cam2gripper)
print("\n平移向量 T (单位: 米):")
print(t_cam2gripper)

# 转换为更直观的 4x4 齐次变换矩阵
T_cam2gripper = np.eye(4)
T_cam2gripper[:3, :3] = R_cam2gripper
T_cam2gripper[:3, 3] = t_cam2gripper.flatten()

print("\n齐次变换矩阵:")
print(T_cam2gripper)

# 保存结果
# save_to_path = '/home/smmg/AAM/config/calibration/right_camera'
# np.save(os.path.join(data_dir, "hand_eye_result_R.npy"), R_cam2gripper)
# np.save(os.path.join(data_dir, "hand_eye_result_T.npy"), t_cam2gripper)
# np.save(os.path.join(data_dir, "T_cam2endeffector.npy"),T_cam2gripper)
# print(f"\n💾 结果已保存至: {data_dir}")

# %%

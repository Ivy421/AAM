# %%
from scipy.spatial.transform import Rotation as R
import os, torch, gc, json, cv2, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import open3d as o3d
from glob import glob
import pandas as pd
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import scipy
from scipy import stats
from AI_models.LLM_funcitons import positioning
# from camera.camera_functions import *

def load_camera_config(camera_config_path):
    config = np.load(camera_config_path, allow_pickle=True).item()
    depth_intrinsic = config['depth_intrinsic']
    color_intrinsic = config['color_intrinsic']
    depth_scale = config['depth_scale']
    depth_to_color_extrinsic = config['depth_to_color_extrinsic']

    return depth_intrinsic, color_intrinsic, depth_to_color_extrinsic, depth_scale

def mask_visualization(image_path, depth_path, mask, box, score):
    # 1. 加载RGB
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    # 2. 加载深度图 & 转伪彩色
    depth = np.nan_to_num(np.load(depth_path), nan=0.0, posinf=0.0, neginf=0.0)
    valid = depth[depth > 0]
    if len(valid) == 0: valid = [0, 1]  # 防空数组报错保底
    d_min, d_max = valid.min(), valid.max()
    d_range = d_max - d_min if d_max != d_min else 1.0

    depth_col = cv2.applyColorMap(np.clip((depth - d_min) / d_range * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    depth_col = cv2.cvtColor(depth_col, cv2.COLOR_BGR2RGB)

    # 3. 创建Mask高亮层（仅mask区域有颜色，其余全0）
    mask_2d = np.squeeze(np.array(mask)) > 0
    highlight = np.zeros_like(img)
    highlight[mask_2d] = [255, 0, 0]  # 红色

    # 4. 透明叠加：addWeighted 会自动让非mask区域保持原图
    rgb_out = cv2.addWeighted(img, 1.0, highlight, 1, 0)
    depth_out = cv2.addWeighted(depth_col, 1.0, highlight, 0.4, 0)

    # 5. 绘制单框 + 分数
    x1, y1, x2, y2 = [int(v) for v in box[0]]
    for out in [rgb_out, depth_out]:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"{score[0]:.3f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 6. 显示
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1); plt.imshow(rgb_out); plt.axis('off')
    plt.subplot(1, 2, 2); plt.imshow(depth_out); plt.axis('off')
    plt.tight_layout(); plt.show()
    return

def pose_to_matrix(x, y, z, rx, ry, rz):
    # 角度转弧度
    angles_deg = np.array([rx, ry, rz])
    angles_rad = np.deg2rad(angles_deg)

    # 先生成 end-effector 相对于 base 的旋转矩阵（base -> end）
    b_R_e = R.from_euler('xyz', angles_rad).as_matrix()  # (3,3)

    # 构造 4x4 齐次变换矩阵：[R | t]
    b_T_e = np.eye(4)
    b_T_e[:3, :3] = b_R_e
    b_T_e[:3, 3] = [x, y, z]  # 平移向量（注意：这是 base 坐标系下的末端位置）

    return b_T_e  # 返回 4x4 矩阵

def rotation_matrix_to_rpy(R):
    """
    将 3x3 旋转矩阵转换为 RPY 欧拉角 (X-Y-Z 顺序，即 Rx * Ry * Rz)
    注意：不同机械臂定义的欧拉角顺序可能不同，请根据实际调整
    """
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    
    singular = sy < 1e-6
    
    if not singular:
        rx = np.arctan2(R[2,1], R[2,2])
        ry = np.arctan2(-R[2,0], sy)
        rz = np.arctan2(R[1,0], R[0,0])
    else:
        rx = np.arctan2(-R[1,2], R[1,1])
        ry = np.arctan2(-R[2,0], sy)
        rz = 0
    rx = np.degrees(rx)
    ry = np.degrees(ry)
    rz = np.degrees(rz)
    
        
    return rx/1000, ry/1000, rz/1000

def define_scan_pose(scan_points, centroid_base, radius, height, rx=0, ry=0):
    scan_poses = []
    cx, cy, cz = centroid_base[:3]
    for i in range(scan_points):
        # 计算角度 (弧度制)
        # 注意：通常数学上 0 度在 X 轴，逆时针增加
        theta = i * (2 * np.pi / scan_points)
        
        # --- 计算位置 (Position) ---
        x_cam = centroid_base[0] + radius * np.cos(theta)
        y_cam = centroid_base[1] + radius * np.sin(theta)
        z_cam = height  # 固定高度
        Pc = np.array([x_cam, y_cam, z_cam])
        
        # --- 计算姿态 (Orientation) ---
        # 需求：末端平行于 X-O-Y 平面
        # 如果需要相机始终对着圆心，rz 需要随角度变化。
        # 获取相机光心到质心的单位向量
        vec_to_center = np.array([x_cam-cx, y_cam-cy, z_cam - cz])
        dist = np.linalg.norm(vec_to_center)
        if dist == 0: continue # 防止除零
        # 归一化，这就是我们想要的相机（末端） Z 轴方向 (在 Base 坐标系下)
        z_axis = vec_to_center / dist
        world_z = np.array([0, 0, 1])
        x_axis = np.cross(world_z, z_axis)
        # print('camera x_axis:',x_axis)
        # 如果 z_axis 和世界 Z 轴平行（即相机垂直向下或向上），叉乘结果为 0
        if np.linalg.norm(x_axis) < 1e-6:
            # 这种情况通常发生在扫描点在圆心正上方或正下方，此时任意指定一个 X 轴
            x_axis = np.array([1, 0, 0])
        else:
            x_axis = x_axis / np.linalg.norm(x_axis)
                # 重新计算 Y 轴以确保正交：Y = Z_cam x X
        
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        # --- 3. 构建旋转矩阵 R_base_to_end ---
        # 注意：这里假设我们要控制的是末端法兰盘 (End Effector)
        # 如果传入的是相机位置，还需要考虑 hand-eye 变换，
        # 但通常我们先算出 End 的姿态，再反解 RPY。
        
        # 矩阵的列向量分别为 X, Y, Z 轴
        b_R_c = np.column_stack((x_axis, y_axis, z_axis))
        # 检查行列式，确保是右手坐标系
        if np.linalg.det(b_R_c) < 0:
            b_R_c[:, 0] *= -1 # 翻转 X 轴修正
        
        b_T_c = np.eye(4)
        b_T_c[:3, :3] = b_R_c
        b_T_c[:3, 3] = Pc

        b_T_e = b_T_c @ np.linalg.inv(e_T_c)
        Pe = b_T_e[:3, 3]
        b_R_e = b_T_e[:3,:3]

        # --- 4. 转换为 RPY (如果需要) ---
        # 欧拉角，旋转矩阵转换为rx,ry,rz
        rx, ry, rz = rotation_matrix_to_rpy(b_R_e)
        
        pose = [Pe[0],Pe[1],Pe[2],rx,ry,rz]
        scan_poses.append(pose)
        
    return scan_poses


depth_scale = 1000.0

torch.cuda.empty_cache()
gc.collect()
image_path = str(Path(__file__).parent.parent)+ '/perception/result/1st_capturing/test.png'
depth_path = str(Path(__file__).parent.parent)+ '/perception/result/1st_capturing/test.npy'
config_path = str(Path(__file__).parent.parent)+ '/perception/result/1st_capturing/camera_config.npy'
# image_visualization(image_path , depth_path)
# 加载数据
depth = np.load(depth_path)
img = cv2.imread(image_path)
depth_intrinsic,_,_,_ = load_camera_config(config_path)

hand_eye_R = np.load(str(Path(__file__).parent.parent)+'/config/calibration/right_camera/hand_eye_result_R.npy', allow_pickle = True)
hand_eye_T = np.load(str(Path(__file__).parent.parent)+'/config/calibration/right_camera/hand_eye_result_T.npy', allow_pickle = True)
e_T_c = np.concatenate((hand_eye_R,hand_eye_T),axis = 1)
e_T_c =np.concatenate((e_T_c,np.array([[0,0,0,1]])),axis = 0)

fx = depth_intrinsic['fx']
fy = depth_intrinsic['fy']
cx = depth_intrinsic['ppx']
cy = depth_intrinsic['ppy']

mask, box, score = positioning(image_path,'cup')
if len(score) > 1:
        max_score_idx = np.argmax(score)
        box = [box[max_score_idx]]
        mask = [mask[max_score_idx]]
        score = [score[max_score_idx]]  

# mask_visualization(image_path, depth_path, mask, box, score)

mask = mask[0][0]
# 确保深度图是 float32 并转换为米
if depth.dtype == np.uint16:
    depth = depth.astype(np.float32) / depth_scale
elif depth.dtype == np.uint8:
    depth = depth.astype(np.float32) * 0.01  # 假设是 0-255 -> 0-2.55m


# 创建网格坐标 (u, v)
H, W = depth.shape
v_coords, u_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

# 提取有效点的像素坐标和深度值
min_depth, max_depth = 0.1, 0.8  
depth_valid = (depth > min_depth) & (depth < max_depth)
valid_mask = mask & depth_valid
u_valid = u_coords[valid_mask]
v_valid = v_coords[valid_mask]
d_valid = depth[valid_mask]
print('最近点深度：',np.min(d_valid))

# 反投影到相机坐标系 (X, Y, Z)
# 公式: X = (u - cx) * Z / fx, Y = (v - cy) * Z / fy, Z = d
# fx, fy 是相机内参
Z_cam = d_valid
X_cam = (u_valid - cx) * Z_cam / fx
Y_cam = (v_valid - cy) * Z_cam / fy

# 构建点云矩阵 N x 3
points = np.stack([X_cam, Y_cam, Z_cam], axis=1)  # shape: (N, 3)

print(f"原始点云数量: {points.shape[0]}")

# %%
#点云过滤
# 创建点云对象
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

if 'img' in locals() and 'v_valid' in locals():
    colors = img[v_valid, u_valid] / 255.0
    pcd.colors = o3d.utility.Vector3dVector(colors)

# 统计滤波（杯子场景优化参数）
cl, ind = pcd.remove_statistical_outlier(
    nb_neighbors=50,      # 检查 20 个邻居
    std_ratio=1.0,        # 2 倍标准差阈值
    print_progress=True
)

# 获取过滤结果
inlier_cloud = pcd.select_by_index(ind)

cl, ind = inlier_cloud.remove_radius_outlier(
        nb_points=50,
        radius=0.015
)

points = np.asarray(cl.points)
print('过滤后点云数量:',points.shape[0])

# %%

# ==================== PCA 分析 ====================
# 1. 计算质心
centroid = np.mean(points, axis=0)
print("质心:", centroid)
# 2. 去中心化
points_centered = points - centroid
# 3. 计算协方差矩阵
cov_matrix = np.cov(points_centered, rowvar=False)  # 3x3
# 4. 特征值分解
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)  # eigh 用于对称矩阵，更稳定
# 按特征值降序排列
sorted_idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_idx]
eigenvectors = eigenvectors[:, sorted_idx]  # 每列是一个特征向量
# 5. 构建旋转矩阵（确保右手系）
R_pca = eigenvectors.copy()
if np.linalg.det(R_pca) < 0:
    R_pca[:, -1] *= -1  # 翻转最小特征值对应的轴，保证行列式为+1

#print("特征值（方差大小）:", eigenvalues)
#print("主轴方向X,Y,Z（列向量）:\n", R_pca)

# 6. 定义物体局部坐标系轴（在相机坐标系下）
object_x_axis = R_pca[:, 0]  # 最长轴 → 通常作为物体的“前向”或“长边”
object_y_axis = R_pca[:, 1]  # 次长轴
object_z_axis = R_pca[:, 2]  # 最短轴 → 通常垂直于物体平面

## mask sure object z axis is upward 
object_z_base = hand_eye_R @ object_z_axis

# 若object Z axis相对世界朝下，反转坐标系
if object_z_base[2] < 0:
    object_z_axis = -object_z_axis
    object_x_axis = -object_x_axis
    print('reverse axises')

# %%

#构建base-end的旋转矩阵
image_endpose_path = str(Path(__file__).parent.parent) +'/perception/result/1st_capturing/'
json_files = [f.name for f in Path(image_endpose_path).glob("*.json")]
# print(image_endpose_path+json_files[0]+'.json')
with open(image_endpose_path+json_files[0],'r',encoding='utf-8') as f:
    data = json.load(f)

##转换为米单位
x = data['x'] /1000000
y = data['y']/1000000
z = data['z']/1000000
rx = data['rx']/1000000
ry = data['ry']/1000000
rz = data['rz']/1000000

b_T_e = pose_to_matrix(x,y,z,rx*1000,ry*1000,rz*1000) # 输入单位ie： mm, degree

# %%
# 构建点云其次坐标
points_hom = np.hstack([points, np.ones((points.shape[0], 1))]) 
centroid_hom = np.hstack([centroid, [1.0]]) 

points_base = b_T_e @ e_T_c @ points_hom.T
centroid_base = b_T_e @ e_T_c @ centroid_hom.T
# 转换为 Nx4 矩阵
points_base = points_base.T

centroid_base_z = centroid_base[2]
object_min_z = min(points_base[:,2])
object_max_z = max(points_base[:,2])
circle1_h = centroid_base_z
circle2_h = centroid_base_z +object_max_z
# decide a scanning scircle
radius = np.abs(np.max(points_base[:,2])-np.min(points_base[:,2]))*1.3
print('扫描半径：', radius)

## 生成扫描点
## centroid圈
scan_point_num = 8
middle_scan_poses = define_scan_pose(scan_point_num, centroid_base, radius, circle1_h, rx=0, ry=90)
# print('质心圈扫描点：', middle_scan_poses)
## 俯视圈
scan_point_num = 8
top_scan_poses = define_scan_pose(scan_point_num, centroid_base, radius, circle2_h, rx=0, ry=135)

with open (str(Path(__file__).parent) +'/result/middle_scan_pose.json', 'w', encoding ='utf-8') as f:
    json.dump(middle_scan_poses,f,ensure_ascii=False,indent =4)

with open (str(Path(__file__).parent) +'/result/top_scan_pose.json', 'w', encoding = 'utf-8') as f:
    json.dump(top_scan_poses,f,ensure_ascii=False,indent =4)


# %%
# 可视化得到的points原始点云
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# 如果有对应的颜色（从img中提取）
# 假设你有 u_valid, v_valid 这些像素坐标
colors = img[v_valid, u_valid] / 255.0  # 归一化到 0-1
pcd.colors = o3d.utility.Vector3dVector(colors)

o3d.visualization.draw_geometries([pcd])


# %%
centroid_base

# %%

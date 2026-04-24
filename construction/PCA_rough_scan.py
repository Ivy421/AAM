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

    return color_intrinsic, depth_intrinsic, depth_to_color_extrinsic, depth_scale

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
    beR = R.from_euler('XYZ', angles_rad).as_matrix()  # (3,3)

    # 构造 4x4 齐次变换矩阵：[R | t]
    beT = np.eye(4)
    beT[:3, :3] = beR
    beT[:3, 3] = [x, y, z]  # 平移向量（注意：这是 base 坐标系下的末端位置）

    return beT  # 返回 4x4 矩阵

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

def define_scan_pose(centroid_base, endpose, radius):
    scan_poses = []
    xe,ye,ze, rxe, rye, rze = endpose
    xo,yo,zo = centroid_base[:3]
    bPo = np.array([[xo],
                   [yo],
                   [zo],
                   [1]])
    boT = np.array(
        [
            [1,0,0,xo],
            [0,1,0,yo],
            [0,0,1,zo],
            [0,0,0,1]
        ]
    )
    #末端在物体坐标系下的位置
    oPe = np.linalg.inv(boT) @ np.array([[xe],[ye],[ze],[1]])
    oxe, oye, oze, _ = oPe
    # 计算当前方向角 phi 和俯仰角 theta
    phi = np.arctan(oye/oxe)
    phi = np.degrees(phi)[0]
    theta = np.arctan(oze/(np.sqrt(oxe * oxe +oye * oye)))
    theta = np.degrees(theta)[0]

    z_b = np.array([0,0,1])

    for phi_can in [phi-40,phi-30,phi-10,phi+10,phi+20,phi+30,phi+40]:
        for theta_can in [theta-10,theta+10,theta+20,theta+30,theta+40,theta+50,theta+60]:

            phi_can = np.radians(phi_can)
            theta_can = np.radians(theta_can)

            oPe_can = np.array([radius * np.cos(phi_can) * np.cos(theta_can),
                               radius * np.sin(phi_can) * np.cos(theta_can) ,
                               radius * np.sin(theta_can),
                               1]).reshape(4,1)
            
            # 计算候选点的世界坐标系下的轴方向
            bPe_can = boT @ oPe_can
            z_e_axis = (bPo - bPe_can)[:3, 0]
            z_e_axis = z_e_axis / np.linalg.norm(z_e_axis)
            # print('z_e_axis:',z_e_axis)
            y_e_axis = np.cross(z_b, z_e_axis)
            x_e_axis = np.cross(y_e_axis, z_e_axis)

            # 通过轴方向获得beR旋转矩阵
            rotation_matrix = np.column_stack((x_e_axis, y_e_axis, z_e_axis))
            r = R.from_matrix(rotation_matrix)
            euler_angles = r.as_euler('xyz', degrees=True)
            rx_can, ry_can, rz_can = euler_angles
            pose_can = [bPe_can[0][0], bPe_can[1][0], bPe_can[2][0], rx_can, ry_can, rz_can]
            
            scan_poses.append(pose_can)
        
    return scan_poses


depth_scale = 1000.0

torch.cuda.empty_cache()
gc.collect()
image_path = os.getcwd() + '/image.png'
depth_path = os.getcwd() + '/image.npy'
config_path = os.getcwd() + '/perception/result/1st_capturing/camera_config.npy'
# image_visualization(image_path , depth_path)
# 加载数据
depth = np.load(depth_path)
img = cv2.imread(image_path)
color_intrinsic,_,_,_ = load_camera_config(config_path)




ecT = np.array([[0.2432,  0.9077,   0.342,      -0.0737+ 0.044*np.sin(15*np.pi/180)],
                [-0.9659, 0.2588,   0,          0.044*np.cos(15*np.pi/180)],
                [-0.0885,-0.3304,   0.9397,     0.0493],
                [0,0,0,1]], dtype=np.float64)


fx = color_intrinsic['fx']
fy = color_intrinsic['fy']
cx = color_intrinsic['ppx']
cy = color_intrinsic['ppy']

# %%

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
    nb_neighbors=50,      # 检查 个邻居
    std_ratio=1.0,        #  倍标准差阈值
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

# %%

#构建base-end的旋转矩阵
# image_endpose_path = os.getcwd() +'/perception/result/1st_capturing/'
# json_files = [f.name for f in Path(image_endpose_path).glob("*.json")]
# # print(image_endpose_path+json_files[0]+'.json')
# with open(image_endpose_path+json_files[0],'r',encoding='utf-8') as f:
#     data = json.load(f)

##转换为米单位
x = 27027   /1000000
y = 2984   /1000000
z = 265785   /1000000
rx = 143630 /1000000
ry = 72557 /1000000
rz = 138711 /1000000

beT = pose_to_matrix(x,y,z,rx*1000,ry*1000,rz*1000) # 输入单位ie： mm, degree, end 在base下的表示

# %%
# 构建点云其次坐标
points_hom = np.hstack([points, np.ones((points.shape[0], 1))]) 
centroid_hom = np.hstack([centroid, [1.0]]) 

points_base = beT @ ecT @ points_hom.T
centroid_base = beT @ ecT @ centroid_hom.T
# 转换为 Nx4 矩阵
points_base = points_base.T

centroid_base_z = centroid_base[2]
object_min_z = min(points_base[:,2])
object_max_z = max(points_base[:,2])
circle1_h = centroid_base_z
circle2_h = centroid_base_z +object_max_z
# decide a scanning scircle
# radius = np.abs(np.max(points_base[:,2])-np.min(points_base[:,2]))*1.3
radius = np.abs(centroid[2])
print('质心（base坐标系下）：' , centroid_base)
print('扫描半径：', radius)

## 生成扫描点
## centroid圈
# middle_scan_poses = define_scan_pose(centroid_base,[x,y,z,rx,ry,rz], radius)


## 俯视圈
scan_poses = define_scan_pose(centroid_base, [x,y,z,rx,ry,rz], radius)
print(scan_poses[23:27])

# with open (os.getcwd() +'/result/middle_scan_pose.json', 'w', encoding ='utf-8') as f:
#     json.dump(middle_scan_poses,f,ensure_ascii=False,indent =4)
# 
# with open (os.getcwd() +'/result/top_scan_pose.json', 'w', encoding = 'utf-8') as f:
#     json.dump(top_scan_poses,f,ensure_ascii=False,indent =4)


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

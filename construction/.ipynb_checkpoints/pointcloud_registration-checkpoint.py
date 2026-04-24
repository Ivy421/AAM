from scipy.spatial.transform import Rotation as R
import os, torch, gc, json, cv2, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import open3d as o3d
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from AI_models.LLM_funcitons import positioning

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

def end_to_base_transformationMatrix(endpose_path):
    with open(endpose_path, 'r', encoding='utf-8') as f:
        endpose = json.load(f)
    x = endpose['x']
    y = endpose['y']
    z = endpose['y']
    rx = endpose['rx']
    ry = endpose['ry']
    rz = endpose['rz']
    rot = R.from_euler('xyz', [rx, ry, rz], degrees=False)
    beT_rot = rot.as_matrix()  # 3x3 旋转矩阵

    # 构造 4x4 齐次变换矩阵 [R | t; 0 0 0 1]
    beT = np.eye(4)
    beT[:3, :3] = beT_rot
    beT[:3, 3] = [x, y, z]
    return beT

def frame_pointcloud_generation(image_path,depth_path,config_path):
    depth = np.load(depth_path)
    img = cv2.imread(image_path)
    color_intrinsic,_,_,_ = load_camera_config(config_path)
    ecT = np.array([[0,0.9397,0.342,-0.0737],
       [-1,0,0,0.044],
       [0,-0.342,0.9397,0.0493],
       [0,0,0,1]], dtype=np.float64)
    fx = color_intrinsic['fx']
    fy = color_intrinsic['fy']
    cx = color_intrinsic['ppx']
    cy = color_intrinsic['ppy']

    depth_scale = 1000.0
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
    points_cam = np.stack([X_cam, Y_cam, Z_cam], axis=1)  # shape: (N, 3)

    print(f"单帧原始点云数量: {points_cam.shape[0]}")

    # 点坐标转换到base坐标系下表示
    points_base = beT @ ecT @ points_cam

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
    print('单帧过滤后点云数量:',points.shape[0])
    return points



torch.cuda.empty_cache()
gc.collect()
folder_path = Path(r"/home/smmg/AAM/construction/result")
png_files = list(folder_path.glob('*.png'))
png_names = [p.name for p in png_files] # p.name 仅获取文件名，不含路径
points_collection = []
for name in png_names:
    image_path = str(folder_path) + '/'+ name + '.png'
    depth_path = str(folder_path) + '/'+ name + '.npy'
    endpose_path = str(folder_path) + '/'+ name + '.json'
    config_path = os.getcwd() + '/perception/result/1st_capturing/camera_config.npy'
    points = frame_pointcloud_generation(image_path,depth_path,endpose_path, config_path)
    points_collection.append(points)




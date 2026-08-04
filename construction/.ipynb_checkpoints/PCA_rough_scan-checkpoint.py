import os, torch, gc, json, cv2
from glob import glob
import pandas as pd
from PIL import Image
import pandas as pd
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import matplotlib.pyplot as plt
import numpy as np

def draw_fig(depth, points, centroid):
    fig = plt.figure(figsize=(10, 5))

    # 子图1：原始深度图
    ax1 = fig.add_subplot(121)
    ax1.imshow(depth, cmap='viridis')
    ax1.set_title("Depth Map")
    ax1.axis('off')

    # 子图2：点云 + PCA轴
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, alpha=0.5, label='Point Cloud')

    # 画质心
    ax2.scatter(*centroid, color='yellow', s=30, label='Centroid')

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Point Cloud with PCA Axes')
    ax2.legend()
    ax2.view_init(elev=0, azim=20)

    plt.tight_layout()
    plt.show()

    return 0

def get_mask(image_path, text):
    model = build_sam3_image_model()  # eval_mode = True
    processor = Sam3Processor(model)
    # Load image
    image = Image.open(image_path)
    inference_state = processor.set_image(image)
    # Prompt the model with text
    output = processor.set_text_prompt(state=inference_state, prompt=text)

    # Get the masks, bounding boxes, and scores
    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

    mask = masks.cpu().numpy()
    box =boxes.cpu().numpy()
    score = scores.cpu().numpy()

    if len(scores) > 1:
        max_score_idx = np.argmax(scores)
        box = [box[max_score_idx]]
        mask = [mask[max_score_idx]]
        score = [score[max_score_idx]]    

    return box, mask, score


depth_scale = 1000.0

torch.cuda.empty_cache()
gc.collect()
image_path = '/home/smmg/AAM/perception/result/1st_capturing/202603164021.png'
depth_path = '/home/smmg/AAM/perception/result/1st_capturing/202603164021.npy'

# 加载数据
depth = np.load(depth_path)
img = cv2.imread(image_path)
intrinsic_mtx = np.load('/home/smmg/AAM/config/calibration/right_camera/R_camera_paras.npy', allow_pickle = True)
hand_eye_R = np.load('/home/smmg/AAM/config/calibration/right_camera/hand_eye_result_R.npy', allow_pickle = True)
hand_eye_T = np.load('/home/smmg/AAM/config/calibration/right_camera/hand_eye_result_T.npy', allow_pickle = True)
e_T_c = np.concatenate((hand_eye_R,hand_eye_T),axis = 1)
e_T_c =np.concatenate((e_T_c,np.array([[0,0,0,1]])),axis = 0)

intrinsic_mtx = intrinsic_mtx.item()
K = intrinsic_mtx['intrinsic_mtx:']
dist = intrinsic_mtx['dist']
fx = K[0,0]
fy = K[1,1]
cx = K[0,-1]
cy = K[1,-1]

box, mask, score = get_mask(image_path, 'cup')

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
u_valid = u_coords[mask]
v_valid = v_coords[mask]
d_valid = depth[mask]

# 反投影到相机坐标系 (X, Y, Z)
# 公式: X = (u - cx) * Z / fx, Y = (v - cy) * Z / fy, Z = d
# fx, fy 是相机内参
Z_cam = d_valid
X_cam = (u_valid - cx) * Z_cam / fx
Y_cam = (v_valid - cy) * Z_cam / fy

# 构建点云矩阵 N x 3
points = np.stack([X_cam, Y_cam, Z_cam], axis=1)  # shape: (N, 3)

print(f"点云数量: {points.shape[0]}")

# ==================== PCA 分析 ====================
# 1. 计算质心
centroid = np.mean(points, axis=0)
print("质心 (相机坐标系):", centroid)
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

print("特征值（方差大小）:", eigenvalues)
print("主轴方向X,Y,Z（列向量）:\n", R_pca)

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


## decide a circular scanning path
    # all point cloud tranform to arm_base coordinate system

    #read machine endpose

image_endpose_path = '/home/smmg/AAM/perception/result/1st_capturing/'

with json.open(image_endpose_path,'r',encoding='utf-8') as f:
    data = json.load(f)
x = data['x']
y = data['y']
z = data['z']
rx = data['rx']
ry = data['ry']
rz = data['rz']

from scipy.spatial.transform import Rotation as R
r = R.from_euler('xyz', [rx, ry, rz])
R_matrix = r.as_matrix()

# 3. 构建 4x4 齐次变换矩阵
T_end_to_base = np.eye(4) # 创建单位矩阵
T_end_to_base[:3, :3] = R_matrix # 填充旋转部分
T_end_to_base[:3, 3] = [x, y, z] # 填充位置部分

points_base = Tbe*e_T_c*points
centroid_base = Tbe*e_T_c*centroid
centroid_base_z = centroid_base[2]
object_min_z = min(points_base[:,2])
object_max_z = max(points_base[:,2])
circle1_h = centroid_base_z
circle2_h = centroid_base_z +object_max_z
    # set the initial start point






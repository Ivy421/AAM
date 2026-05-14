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

def normalize(v, eps=1e-9):
    v = np.asarray(v, dtype=float).reshape(3)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("zero-length vector cannot be normalized")
    return v / n


def matrix_to_pose_xyzrpy_deg(T, euler_order='XYZ'):
    """
    4x4 齐次矩阵 -> [x, y, z, rx, ry, rz]
    rx/ry/rz 单位是 degree，和你原始 define_scan_pose 的输出保持一致。
    """
    T = np.asarray(T, dtype=float)
    xyz = T[:3, 3].tolist()
    rpy = R.from_matrix(T[:3, :3]).as_euler(euler_order, degrees=True).tolist()
    return xyz + rpy


def look_at_camera_pose(camera_pos_base, target_pos_base, up_base=np.array([0.0, 0.0, 1.0])):
    """
    构造 T_base_camera。
    假设 RGB-D 相机坐标系为：+Z 为光轴朝前，+X 向右，+Y 向下。
    因此让 camera 的 +Z 指向 target，同时让 camera 的 +Y 尽量朝向 base 的 -Z。
    """
    camera_pos_base = np.asarray(camera_pos_base, dtype=float).reshape(3)
    target_pos_base = np.asarray(target_pos_base, dtype=float).reshape(3)
    up_base = normalize(up_base)

    z_cam = normalize(target_pos_base - camera_pos_base)  # camera optical axis, points to object

    # 将世界 up 投影到与 z_cam 垂直的平面，再取负号作为 camera y-down
    up_proj = up_base - np.dot(up_base, z_cam) * z_cam
    if np.linalg.norm(up_proj) < 1e-6:
        # 接近正俯视时，up 和光轴几乎平行，换一个参考轴避免叉乘退化
        alt_up = np.array([0.0, 1.0, 0.0])
        up_proj = alt_up - np.dot(alt_up, z_cam) * z_cam

    y_cam = normalize(-up_proj)              # image y roughly points downward
    x_cam = normalize(np.cross(y_cam, z_cam))
    y_cam = normalize(np.cross(z_cam, x_cam))  # re-orthogonalize

    T = np.eye(4)
    T[:3, :3] = np.column_stack([x_cam, y_cam, z_cam])
    T[:3, 3] = camera_pos_base
    return T


def define_local_nbv_scan_poses(
    centroid_base,
    current_beT,
    T_ee_camera,
    radius=None,
    yaw_range_deg=(-90, 90),
    pitch_range_deg=(0, 90),
    yaw_step_deg=15,
    pitch_step_deg=15,
    euler_order='XYZ',
):
    """
    生成局部 NBV 候选视角，不再生成完整 360° 扫描圈。

    参数：
    centroid_base: 目标/缺陷中心点在 base 坐标系下的位置，shape 可以是 (3,), (4,), (4,1)
    current_beT: 当前 T_base_end，4x4
    T_ee_camera: 手眼标定 T_end_camera，4x4。你的代码里 ecT 就是这个。
    radius: 扫描半径。None 时自动使用当前 camera 到 centroid 的距离。
    yaw_range_deg: 以当前视角方位角为中心的左右扫描范围，默认 -90° 到 +90°
    pitch_range_deg: 绝对俯仰角范围，默认 0° 到 90°，即从水平视角到俯视。
    yaw_step_deg / pitch_step_deg: 候选点角度间隔。

    返回：
    records: list[dict]
        每个元素包含：
        - pose_ee_xyzrpy_deg: MoveIt/机械臂末端目标位姿 [x,y,z,rx,ry,rz]
        - T_base_ee: 4x4 末端位姿矩阵
        - T_base_camera: 4x4 相机候选位姿矩阵
        - yaw_offset_deg / pitch_deg: 候选点角度参数
    """
    target = np.asarray(centroid_base, dtype=float).reshape(-1)[:3]
    current_beT = np.asarray(current_beT, dtype=float)
    T_ee_camera = np.asarray(T_ee_camera, dtype=float)

    current_bcT = current_beT @ T_ee_camera
    current_cam_pos = current_bcT[:3, 3]

    vec_target_to_cam = current_cam_pos - target
    if radius is None:
        radius = float(np.linalg.norm(vec_target_to_cam))
    if radius <= 0:
        raise ValueError("radius must be positive")

    # 当前视角对应的方位角，作为 yaw=0 的中心方向
    phi0 = np.degrees(np.arctan2(vec_target_to_cam[1], vec_target_to_cam[0]))

    yaw_offsets = np.arange(yaw_range_deg[0], yaw_range_deg[1] + 1e-6, yaw_step_deg)
    pitch_list = np.arange(pitch_range_deg[0], pitch_range_deg[1] + 1e-6, pitch_step_deg)

    records = []
    used_keys = set()

    for yaw_offset in yaw_offsets:
        phi = np.radians(phi0 + yaw_offset)

        for pitch_deg in pitch_list:
            theta = np.radians(pitch_deg)

            # 球坐标：以 target 为球心，当前视角方位角为中心，只扫前方 ±90° 半空间
            cam_pos = target + np.array([
                radius * np.cos(theta) * np.cos(phi),
                radius * np.cos(theta) * np.sin(phi),
                radius * np.sin(theta),
            ])

            # pitch=90° 时不同 yaw 会生成同一个顶视点，做去重
            key = tuple(np.round(cam_pos, 5))
            if key in used_keys:
                continue
            used_keys.add(key)

            T_base_camera = look_at_camera_pose(cam_pos, target)
            T_base_ee = T_base_camera @ np.linalg.inv(T_ee_camera)
            pose_ee = matrix_to_pose_xyzrpy_deg(T_base_ee, euler_order=euler_order)

            records.append({
                "yaw_offset_deg": float(yaw_offset),
                "pitch_deg": float(pitch_deg),
                "radius": float(radius),
                "pose_ee_xyzrpy_deg": [float(v) for v in pose_ee],
                "T_base_ee": T_base_ee.tolist(),
                "T_base_camera": T_base_camera.tolist(),
            })

    return records


def save_scan_pose_records(records, json_path):
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=4)
    print(f"已保存候选 NBV 位姿: {json_path}")


def filter_reachable_with_moveit1(
    records,
    group_name="manipulator",
    ee_link=None,
    planning_time=2.0,
    num_planning_attempts=5,
    execute_best=False,
):
    """
    MoveIt1 / ROS Noetic 版本：只 plan，不 execute，用于候选位姿可达性过滤。

    使用前提：
    1. 已启动 roscore
    2. 已启动 robot move_group
    3. 已加载 URDF/SRDF 和 controllers
    4. 如果要检查桌面/物体碰撞，需要先在 PlanningScene 里添加障碍物

    返回：
    reachable_records: list[dict]
        只保留 MoveIt 规划成功的候选点，并附带 plan_index。
    """
    try:
        import rospy
        import moveit_commander
        from geometry_msgs.msg import Pose
    except ImportError as e:
        raise ImportError(
            "当前 Python 环境没有 ROS/MoveIt1。请在 ROS 环境中运行，例如："
            "source /opt/ros/noetic/setup.bash && source ~/catkin_ws/devel/setup.bash"
        ) from e

    moveit_commander.roscpp_initialize(sys.argv)
    if not rospy.core.is_initialized():
        rospy.init_node("nbv_reachability_filter", anonymous=True)

    group = moveit_commander.MoveGroupCommander(group_name)
    group.set_planning_time(planning_time)
    group.set_num_planning_attempts(num_planning_attempts)

    reachable_records = []
    plans = []

    for i, rec in enumerate(records):
        x, y, z, rx, ry, rz = rec["pose_ee_xyzrpy_deg"]
        quat = R.from_euler('XYZ', [rx, ry, rz], degrees=True).as_quat()  # x,y,z,w

        goal = Pose()
        goal.position.x = x
        goal.position.y = y
        goal.position.z = z
        goal.orientation.x = quat[0]
        goal.orientation.y = quat[1]
        goal.orientation.z = quat[2]
        goal.orientation.w = quat[3]

        group.clear_pose_targets()
        if ee_link is None:
            group.set_pose_target(goal)
        else:
            group.set_pose_target(goal, end_effector_link=ee_link)

        plan_result = group.plan()

        # MoveIt 不同版本返回格式略不同：可能是 tuple，也可能直接是 RobotTrajectory
        if isinstance(plan_result, tuple):
            success = bool(plan_result[0])
            plan = plan_result[1]
        else:
            plan = plan_result
            success = len(plan.joint_trajectory.points) > 0

        if success and len(plan.joint_trajectory.points) > 0:
            rec = dict(rec)
            rec["moveit_reachable"] = True
            rec["plan_index"] = len(plans)
            rec["trajectory_points"] = len(plan.joint_trajectory.points)
            reachable_records.append(rec)
            plans.append(plan)
            print(f"[OK] candidate {i:03d}: yaw={rec['yaw_offset_deg']:.1f}, pitch={rec['pitch_deg']:.1f}")
        else:
            print(f"[FAIL] candidate {i:03d}: yaw={rec['yaw_offset_deg']:.1f}, pitch={rec['pitch_deg']:.1f}")

    group.clear_pose_targets()

    if execute_best and len(reachable_records) > 0:
        # 这里默认执行第一个可达点。实际 NBV 中建议按 score 排序后再执行。
        best_plan = plans[reachable_records[0]["plan_index"]]
        group.execute(best_plan, wait=True)

    return reachable_records, plans


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

ecT = np.load('E:/HKUSTGZ/AAM/config/calibration/right_camera/ecT.npy')
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

## 生成局部 NBV 候选扫描点
## 目标：以当前视角为中心，方位角 yaw_offset = -90° 到 +90°，俯仰角 pitch = 0° 到 90°
## 注意：这里生成的是相机候选位姿，再通过 ecT = T_end_camera 转换为机械臂末端位姿。
scan_pose_records = define_local_nbv_scan_poses(
    centroid_base=centroid_base,
    current_beT=beT,
    T_ee_camera=ecT,
    radius=radius,              # 也可以改成固定值，例如 0.35
    yaw_range_deg=(-90, 90),
    pitch_range_deg=(0, 90),
    yaw_step_deg=15,
    pitch_step_deg=15,
    euler_order='XYZ',
)

scan_poses = [rec["pose_ee_xyzrpy_deg"] for rec in scan_pose_records]
print(f"候选 NBV 点数量: {len(scan_poses)}")
print("前 5 个末端候选位姿 [x,y,z,rx,ry,rz(deg)]:")
for p in scan_poses[:5]:
    print(p)

save_scan_pose_records(
    scan_pose_records,
    os.path.join(os.getcwd(), 'result', 'local_nbv_scan_pose_records.json')
)

## ===== 可选：在 ROS + MoveIt1 环境中做可达性过滤，只 plan，不执行 =====
## 如果你当前不是在 ROS 环境里运行，保持 False。
USE_MOVEIT_REACHABILITY_CHECK = False

if USE_MOVEIT_REACHABILITY_CHECK:
    reachable_records, plans = filter_reachable_with_moveit1(
        scan_pose_records,
        group_name="manipulator",  # 按你的 MoveIt planning group 修改
        ee_link=None,                # 如果有固定末端 link 名称可填，例如 "tool0"
        planning_time=2.0,
        num_planning_attempts=5,
        execute_best=False,          # 这里只验证，不真实执行
    )
    save_scan_pose_records(
        reachable_records,
        os.path.join(os.getcwd(), 'result', 'local_nbv_reachable_pose_records.json')
    )
    print(f"MoveIt 可达候选点数量: {len(reachable_records)} / {len(scan_pose_records)}")


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

import os, torch, gc, json, cv2, sys
sys.path.append('/home/smmg/AAM')
import numpy as np
import ikpy.chain
from scipy.spatial.transform import Rotation as R
from ikpy.inverse_kinematics import inverse_kinematic_optimization


# ================= 配置区域 =================
URDF_PATH = "/home/smmg/AAM/piper_motion/piper_link.urdf"  # 👈 替换为你的URDF实际路径
BASE_LINK = "base_link"
# active_links_mask: [base_link(fixed), link1, link2, link3, link4, link5, link6]
ACTIVE_LINKS_MASK = [False, True, True, True, True, True, True]
# ===========================================

# 加载机械臂链 (use_collision=False 避免 package:// mesh 路径问题)
chain = ikpy.chain.Chain.from_urdf_file(
    URDF_PATH,
    base_elements=[BASE_LINK] ,
    active_links_mask=ACTIVE_LINKS_MASK,
    name = 'piper'
)

# Piper 关节限位 (从URDF提取，用于双重验证)
PIPER_JOINT_LIMITS = np.array([
    [-2.618, 2.168],    # joint1
    [0.0, 3.14],        # joint2  
    [-2.967, 0.0],      # joint3
    [-1.745, 1.745],    # joint4
    [-1.22, 1.22],      # joint5
    [-2.0944, 2.0944]   # joint6
])

def validate_and_solve_ik(x, y, z, rx, ry, rz, current_joints=None, 
                         pos_tol=2e-2, rot_tol=3e-2, euler_order='xyz'):
    """
    Piper机械臂逆运动学可达性验证
    
    参数:
    - x,y,z: 末端位置 (米, 相对于base_link坐标系)
    - rx,ry,rz: 末端姿态欧拉角 (弧度), 顺序由euler_order指定
    - current_joints: 当前关节角 (6维), 用于保证解的连续性
    - pos_tol: 位置容差 (米), 默认1mm
    - rot_tol: 姿态容差 (弧度), 默认~0.57°
    - euler_order: 欧拉角顺序, 默认'xyz'(即RPY: roll-pitch-yaw intrinsic)
    
    返回:
    - dict: 包含success, joint_angles, 误差等信息
    """
    # 1. 构建目标齐次变换矩阵
    target_pos = np.array([x, y, z])
    target_rot = R.from_euler(euler_order, [rx, ry, rz]).as_matrix()
    target_pose = np.eye(4)
    target_pose[:3, :3] = target_rot
    target_pose[:3, 3] = target_pos
    
    if current_joints is None:
        # 使用中间位置（7个元素，包括 base_link）
        initial_position = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # 设置活动关节的初始值
        for i in range(1, 7):
            lower, upper = PIPER_JOINT_LIMITS[i]
            initial_position[i] = (lower + upper) / 2
    else:
        # current_joints 是6维数组，需要扩展为7维
        initial_position = np.zeros(7)
        initial_position[1:7] = current_joints  # 跳过 base_link

    # 2. 逆运动学求解 (ikpy自动应用URDF中的joint limits)
    ik_result = inverse_kinematic_optimization(
        chain=chain,
        target_frame=target_pose,           # ✅ 4x4 齐次变换矩阵
        starting_nodes_angles=initial_position,
        orientation_mode='all',              # ✅ 控制全部三个轴（完整姿态）
        max_iter=100,
        regularization_parameter=0.1,        # 正则化，防止奇异
        optimizer='least_squares'            # 使用最小二乘法
    )
    print('inverse kinematics result: ', ik_result)

    # 3. 提取6个活动关节角 (ik_result[0]是base_link占位符)
    active_joints = ik_result[1:7]

    # 4. 可达性验证
    # 4.1 关节限位检查 (双重验证)
    in_limits = np.all((active_joints >= PIPER_JOINT_LIMITS[:, 0] - 1e-6) & 
                      (active_joints <= PIPER_JOINT_LIMITS[:, 1] + 1e-6))

    # 4.2 正向运动学验证位姿误差
    fk_result = chain.forward_kinematics(ik_result)
    pos_error = np.linalg.norm(fk_result[:3, 3] - target_pos) ## 计算位置误差
    
    # 旋转误差计算 (旋转矩阵夹角)
    fk_rot = fk_result[:3, :3]
    cos_angle = np.clip((np.trace(fk_rot.T @ target_rot) - 1) / 2, -1.0, 1.0)
    rot_error = np.arccos(cos_angle)

    # 综合判断
    reachable = (pos_error < pos_tol) and (rot_error < rot_tol) and in_limits

    return {
        "success": reachable,
        "joint_angles": active_joints,  # 6维数组 (rad)
        "position_error": pos_error,
        "orientation_error": rot_error,
        "within_limits": in_limits,
        "fk_pose": fk_result  # 实际达到的位姿
    }



if __name__ == "__main__":
    print("🔧 Piper IK 验证模块加载完成")
    
    # 示例1: 验证工作空间内的位姿
    print("\n=== 测试可达位姿 ===")
    result = validate_and_solve_ik(
        x=0.336, y=0.22, z=0.217,
        rx=0.0, ry=np.pi/2, rz=0.0,  # xyz欧拉角顺序
        current_joints=np.zeros(6)
    )
    
    if result["success"]:
        print("✅ 可达!")
        print(f"关节角 (rad): {result['joint_angles']}")
        print(f"关节角 (deg): {np.degrees(result['joint_angles'])}")
        print(f"位置误差: {result['position_error']*1000:.3f} mm")
        print(f"姿态误差: {np.degrees(result['orientation_error']):.3f} °")
    else:
        print("❌ 不可达")
        print(f"  位置误差: {result['position_error']*1000:.3f} mm (阈值: 1mm)")
        print(f"  姿态误差: {np.degrees(result['orientation_error']):.3f} ° (阈值: ~0.57°)")
        print(f"  关节限位: {'✅' if result['within_limits'] else '❌ 超限'}")
    
    # # 示例2: 验证超出工作空间的位姿
    # print("\n=== 测试不可达位姿 (太远) ===")
    # result2 = validate_and_solve_ik(
    #     x=0.36, y=0.22, z=0.25,  # Piper臂长约0.7m, 此点大概率不可达
    #     rx=0.0, ry=0.0, rz=0.0,
    #     current_joints=np.zeros(6)
    # )
    # print(f"可达: {result2['success']}, 位置误差: {result2['position_error']*1000:.2f} mm")
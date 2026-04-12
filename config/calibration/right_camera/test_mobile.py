import cv2
import numpy as np
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R_transform

class D435FixedPixelToMove:
    def __init__(self):
        """
        初始化D435相机和相关参数
        """
        # 1. 配置并启动Realsense D435管道
        self.pipeline = rs.pipeline()
        config = rs.config()
        
        # 启用深度流和彩色流
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # 启动管道并获取配置文件
        profile = self.pipeline.start(config)

        # 获取深度传感器的深度比例因子
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        # 获取彩色流的内参 (用于像素到空间的转换)
        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        self.color_intrinsics = color_profile.get_intrinsics()

        # 对齐深度图到彩色图
        self.align_to_color = rs.align(rs.stream.color)

        # 2. 定义手眼标定得到的变换矩阵 cam2gripper (4x4 齐次矩阵)
        # 请务必用您实际标定的结果替换此矩阵
        # 示例矩阵 endpose.csv标定结果
        self.cam2gripper_matrix = np.array([
            [0.436, -0.2421, -0.8667, 0.0272],
            [-0.7247, 0.4765, -0.4976, 0.01617],
            [0.5335, 0.8452, 0.0322, -0.0508],
            [0, 0, 0, 1]
        ], dtype=np.float64)

        '''
        np.array([[0,0.9397,0.342,-0.069],
       [-1,0,0,0.044],
       [0,-0.342,0.9397,0.0476],
       [0,0,0,1]], dtype=np.float64)
        '''


        # 3. 定义固定的目标像素坐标
        self.target_pixel = (250, 250) # (u, v)
        print(f"目标像素坐标已设定为: {self.target_pixel}")

    def calculate_pose(self, depth_frame):
        """
        核心计算函数：给定深度帧，计算目标像素点对应的机械臂末端位姿
        """
        u, v = self.target_pixel
        
        # 检查像素坐标是否在图像范围内
        height, width = depth_frame.profile.height(), depth_frame.profile.width()
        if not (0 <= u < width and 0 <= v < height):
            print(f"错误: 像素坐标 ({u}, {v}) 超出图像范围 ({width}x{height})")
            return None

        # 获取点击点的深度值 (米)
        depth_value = depth_frame.get_distance(u, v)
        
        if depth_value == 0:
            print(f"警告: 像素点 ({u}, {v}) 处无法获取有效深度信息，可能是遮挡或超出测量范围。")
            return None

        print(f"在像素点 ({u}, {v}) 处测得深度: {depth_value:.3f} 米")

        # 3. 将像素坐标和深度值转换为相机坐标系下的3D点
        # 使用相机内参
        x_cam = (u - self.color_intrinsics.ppx) * depth_value / self.color_intrinsics.fx
        y_cam = (v - self.color_intrinsics.ppy) * depth_value / self.color_intrinsics.fy
        z_cam = depth_value
        
        point_cam = np.array([x_cam, y_cam, z_cam, 1.0]) # 齐次坐标
        print(f"相机坐标系下的3D点: [{x_cam:.3f}, {y_cam:.3f}, {z_cam:.3f}]")

        # 4. 使用手眼标定矩阵将相机坐标系下的点转换到机械臂末端坐标系
        point_gripper = self.cam2gripper_matrix @ point_cam # 使用@符号进行矩阵乘法

        target_x = point_gripper[0]
        target_y = point_gripper[1]
        target_z = point_gripper[2]

        # 5. 提取姿态 (rx, ry, rz) - 从变换矩阵的旋转部分
        # 获取旋转矩阵 (3x3)
        R_cam2gripper = self.cam2gripper_matrix[:3, :3]

        try:
            r = R_transform.from_matrix(R_cam2gripper)

            euler_angles_degrees = r.as_euler('XYZ', degrees=True) 
            
            rx, ry, rz = euler_angles_degrees # 直接获取度为单位的角度

        except Exception as e:
            print(f"使用 scipy 计算欧拉角时出错: {e}")
            return None


        return [val*1000000 for val in [target_x, target_y, target_z,rx,ry,rz]]


    def run_once(self):
        """
        执行一次计算流程：获取一帧数据，处理并打印结果。
        """
        print("正在获取一帧图像和深度数据...")
        # 等待新的一帧
        frames = self.pipeline.wait_for_frames()
        
        # 对齐深度帧到彩色帧
        aligned_frames = self.align_to_color.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            print("错误: 无法获取有效的深度或彩色帧。")
            return

        # 将帧转换为numpy数组以便可视化
        color_image = np.asanyarray(color_frame.get_data())

        # 在图像上标记目标像素点
        cv2.circle(color_image, self.target_pixel, radius=5, color=(0, 255, 0), thickness=-1)
        cv2.imshow('Target Pixel Preview', color_image)
        print(f"已在预览图像上标记目标像素点 {self.target_pixel} (绿色圆圈)")
        
        # 执行核心计算
        result = self.calculate_pose(depth_frame)
        
        if result:
            print('target end pose:' , result)
        else:
            print("未能计算出有效位姿。")

        cv2.waitKey(1000) # 显示图像1秒后关闭


    def cleanup(self):
        """
        清理资源
        """
        self.pipeline.stop()


if __name__ == "__main__":
    app = D435FixedPixelToMove()
    
    try:
        # --- 选择运行模式 ---
        
        # 模式1: 只运行一次，获取当前时刻的位姿
        print("--- 运行一次计算 ---")
        app.run_once()

    finally:
        app.cleanup()
        print("程序已退出，资源已释放。")

    # enable('r_piper')
    # move_to_pose()
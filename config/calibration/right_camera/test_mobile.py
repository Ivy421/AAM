import cv2, sys, os
sys.path.append('/home/smmg/AAM')
import numpy as np
from scipy.spatial.transform import Rotation as R
from camera.camera_functions import *

class transformation:
    def __init__(self):
        ## ecT变换矩阵，单位：米
        self.cam2gripper_matrix = np.array([[0,0.9397,0.342,-0.069],
       [-1,0,0,0.044],
       [0,-0.342,0.9397,0.0476],
       [0,0,0,1]], dtype=np.float64)

        '''
        1. 手动计算结果
        np.array([[0,0.9397,0.342,-0.069],
       [-1,0,0,0.044],
       [0,-0.342,0.9397,0.0476],
       [0,0,0,1]], dtype=np.float64)
        '''

        '''
        2. endpose.csv 结果
        np.array([
            [0.436, -0.2421, -0.8667, 0.0272],
            [-0.7247, 0.4765, -0.4976, 0.01617],
            [0.5335, 0.8452, 0.0322, -0.0508],
            [0, 0, 0, 1]
        ], dtype=np.float64)
        '''

        '''
        3. endpose2.csv 结果
        np.array([
            [0.3139, 0.0912, -0.9451, 0.0289],
            [-0.5513, 0.828, -0.1032, -0.0324],
            [0.773, 0.5534, 0.3101, -0.0555],
            [0, 0, 0, 1]
        ], dtype=np.float64)
        '''
        # 3. 定义固定的目标像素坐标
        self.target_pixel = (340, 220) # (u, v)
        print(f"目标像素坐标已设定为: {self.target_pixel}")

    def calculate_end_coord(self):
        """
        核心计算函数：给定深度帧，计算目标像素点对应的机械臂末端位姿
        """
        u, v = self.target_pixel
        depth = np.load('/home/smmg/AAM/image.npy',allow_pickle=True)
        # 确保深度图是 float32 并转换为米
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) / 1000
        elif depth.dtype == np.uint8:
            depth = depth.astype(np.float32) * 0.01  # 假设是 0-255 -> 0-2.55m
    
        # 获取点击点的深度值 (米)
        depth_value = depth[u,v]
        
        if depth_value == 0:
            print(f"警告: 像素点 ({u}, {v}) 处无法获取有效深度信息，可能是遮挡或超出测量范围。")
            return None

        print(f"在像素点 ({u}, {v}) 处测得深度: {depth_value:.3f} 米")

        # 3. 将像素坐标和深度值转换为相机坐标系下的3D点
        # 使用相机内参
        x_cam = (u - 320) * depth_value / 606.812  
        y_cam = (v - 256) * depth_value / 606.648
        z_cam = depth_value
        
        point_cam = np.array([x_cam, y_cam, z_cam, 1.0]) # 齐次坐标
        print(f"相机坐标系下的3D点: [{x_cam:.3f}, {y_cam:.3f}, {z_cam:.3f}]")

        # 4. 使用手眼标定矩阵将相机坐标系下的点转换到机械臂末端坐标系
        point_gripper = self.cam2gripper_matrix @ point_cam # 使用@符号进行矩阵乘法
        ePx = point_gripper[0]
        ePy = point_gripper[1]
        ePz = point_gripper[2]

        return [ePx, ePy, ePz]  # 单位：m
    
    def calculate_base_coord(self, eP, current_endpose):
        # 小写是绕定轴，大写是动轴
        beR = R.from_euler('xyz', [current_endpose[3],current_endpose[4],current_endpose[5]], degrees = True ).as_matrix()  # (3,3)

        # 构造 4x4 齐次变换矩阵：[R | t]
        beT = np.eye(4)
        beT[:3, :3] = beR
        beT[:3, 3] = [current_endpose[0], current_endpose[1], current_endpose[2]]  # 平移向量（注意：这是 base 坐标系下的末端位置）
        bP = beT @ np.array([eP[0], eP[1], eP[2], 1.0]) 
        return bP
    
    def draw_point(self ):
        img = cv2.imread('/home/smmg/AAM/image.png')

        # 2. 定义点的坐标 (x, y) = (222, 333)
        point = self.target_pixel
        color = (0, 0, 255)      # BGR格式：红色
        radius = 5               # 点的半径（像素）
        thickness = -1           # -1 表示实心填充

        # 方法一：使用 circle 画点（最常用）
        cv2.circle(img, point, radius, color, thickness)

        # 方法二：使用 drawMarker 画标记点（OpenCV 3.0+ 推荐，自带十字/圆形等样式）
        # cv2.drawMarker(img, point, color, markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

        # 3. 可视化图像
        cv2.imshow('Draw Point at (222, 333)', img)
        cv2.waitKey(0)           # 等待按键，0表示无限等待
        cv2.destroyAllWindows()  # 关闭窗口

        return 


    def run_once(self):
        """
        执行一次计算流程：获取一帧数据，处理并打印结果。
        """
        capture('/home/smmg/AAM/','image',2,1,0,0)
        x= 240477 /1000000
        y=-36096  /1000000
        z=340573  /1000000
        rx=163164 /1000
        ry=46507  /1000
        rz=163732  /1000
        current_endpose = [x,y,z,rx,ry,rz]
        
        # 执行核心计算
        eP = self.calculate_end_coord()
        bP = self.calculate_base_coord(eP, current_endpose)
        print('末端坐标系下的点:' ,eP )
        print('基坐标系下的点:' ,bP )


if __name__ == "__main__":
    app = transformation()

    app.run_once()
    app.draw_point()


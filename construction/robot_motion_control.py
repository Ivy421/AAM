from scipy.spatial.transform import Rotation as R
import os, torch, gc, json, cv2, sys
sys.path.append('/home/smmg/AAM')
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

def define_scan_pose_endDirect(scan_points, centroid_base, radius, height_base):
    scan_poses = []
    cenx, ceny, cenz = centroid_base[:3]
    for i in range(scan_points):
        # 计算角度 (弧度制)
        theta = i * (2 * np.pi / scan_points)
        
        # 相机光心的理论位置，在base坐标系下的表示
        x_end = centroid_base[0] + radius * np.cos(theta)
        y_end = centroid_base[1] + radius * np.sin(theta)
        z_end = height_base  # 固定高度
        Pe = np.array([x_end, y_end, z_end])
        
        # --- 计算相机坐标系在base下的表示 ---
        # 相机坐标系Z轴始终指向质心
        # 获取相机光心到质心的单位向量， 在base坐标系下
        Zend_to_cen = np.array([cenx - x_end, ceny - y_end, cenz - z_end])
        dist = np.linalg.norm(Zend_to_cen)
        if dist == 0: continue # 防止除零
        # 归一化，末端 Z 轴方向 (在 Base 坐标系下)
        z_axis = Zend_to_cen / dist
        base_z = np.array([0, 0, 1])
        y_axis = np.cross(base_z, z_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        x_axis = np.cross( y_axis , z_axis )
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        # --- 3. 构建旋转矩阵 beT ---
        # 矩阵的列向量分别为 X, Y, Z 轴
        beR = np.column_stack((x_axis, y_axis, z_axis))
        
        # 检查行列式，确保是右手坐标系
        if np.linalg.det(beR) < 0:
            print('beR 不是右手系，需要校正！')
            beR[:, 0] *= -1 # 翻转 X 轴修正
        
        beT = np.eye(4)
        beT[:3, :3] = beR
        beT[:3, 3] = Pe
 
        rot = R.from_matrix(beR)
        r_degrees = rot.as_euler('xyz', degrees=False)   # 角度
        rx, ry, rz = r_degrees

        pose = [ Pe[0], Pe[1], Pe[2], rx , ry , rz ]
        scan_poses.append(pose)
        if i == 2:
            print('beR: ',beR)
            print('z_axis under base: ', z_axis)
            print('endpose:', pose)


        
    return scan_poses

define_scan_pose_endDirect(4, np.array([0.43452015, 0.0328951 , 0.16984754 ,1 ])  , 0.328, 0.217077)
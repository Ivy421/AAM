import os, json, sys
sys.path.append('E:/HKUSTGZ/AAM')
import open3d as o3d
from glob import glob
import numpy as np
pcd_raw = o3d.io.read_point_cloud("E:/HKUSTGZ/AAM/construction/data/completion_result/fused2.pcd")
defect_pcd1 = o3d.io.read_point_cloud(
    "E:/HKUSTGZ/AAM/construction/data/completion_result/defect_pcd1.pcd"
)
defect_pcd2 = o3d.io.read_point_cloud(
    "E:/HKUSTGZ/AAM/construction/data/completion_result/defect_pcd2.pcd"
)
defect_pcd1.paint_uniform_color([0.55, 0.2 , 0.8 ]) 
defect_pcd2.paint_uniform_color([0.55, 0.2 , 0.8 ]) 
pcd_raw.paint_uniform_color([0,0,1 ]) 
o3d.visualization.draw_geometries([pcd_raw, defect_pcd1, defect_pcd2]) # [defect_pcd1, defect_pcd2]

import sys
sys.path.append('/home/smmg/AAM')
import open3d as o3d
import numpy as np
import pyransac3d as pyrsc

# load pointcloud
pcd = o3d.io.read_point_cloud("/home/smmg/AAM/construction/result/fused.pcd")
o3d.visualization.draw_geometries([pcd])
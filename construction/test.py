import argparse
import json
import os
from pathlib import Path

import numpy as np
import open3d as o3d

fine_scan_path = '/home/smmg/AAM/data/runs/20260914_172825/construction/coarse_scan/coarse_point_result.npz'

coarse_icp_path ='/home/smmg/AAM/data/runs/20260914_172825/construction/coarse_scan/coarse_fuse.pcd'
coarse_icp_curv_path = '/home/smmg/AAM/data/runs/20260914_172825/construction/coarse_scan/coarse_fuse_curvature.pcd'
fine_icp_path ='/home/smmg/AAM/data/runs/20260914_172825/construction/fine_scan/fine_fuse.pcd'
fine_icp_curv_path = '/home/smmg/AAM/data/runs/20260914_172825/construction/fine_scan/fine_fuse_curvature.pcd'

meta = np.load(fine_scan_path, allow_pickle=True)
points = np.asarray(meta['points_collection'][3])
points = points[:,:3]

coarse_pcd = o3d.io.read_point_cloud(coarse_icp_path)
coarse_curv_pcd = o3d.io.read_point_cloud(coarse_icp_curv_path)
coarse_pcd.paint_uniform_color([1,0,0])
coarse_curv_pcd.paint_uniform_color([0,1,0])

fine_pcd = o3d.io.read_point_cloud(fine_icp_path)
fine_curv_pcd = o3d.io.read_point_cloud(fine_icp_curv_path)
fine_pcd.paint_uniform_color([1,0,0])
fine_curv_pcd.paint_uniform_color([0,1,0])


pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.paint_uniform_color([1,0,0])

o3d.visualization.draw_geometries([ fine_pcd  , fine_curv_pcd])

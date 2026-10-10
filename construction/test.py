import argparse
import json
import os
from pathlib import Path

import numpy as np
import open3d as o3d

fine_scan_path = '/home/smmg/AAM/data/runs/20261010_144417/construction/coarse_scan/coarse_point_result.npz'

coarse_icp_path ='/home/smmg/AAM/data/runs/20261010_144417/construction/coarse_scan/coarse_fuse.pcd'
fine_icp_curv_path = '/home/smmg/AAM/data/runs/20261010_144417/construction/fine_scan/fine_fuse.pcd'
model_raw_path = '/home/smmg/AAM/data/runs/20261010_144417/completion/depression/model.pcd'

meta = np.load(fine_scan_path, allow_pickle=True)
points = np.asarray(meta['points_collection'][3])
points = points[:,:3]

coarse_pcd = o3d.io.read_point_cloud(coarse_icp_path)
coarse_pcd.paint_uniform_color([1,0,0])

fine_curv_pcd = o3d.io.read_point_cloud(fine_icp_curv_path)
fine_curv_pcd.paint_uniform_color([0,1,0])
model_pcd = o3d.io.read_point_cloud(model_raw_path)
fine_curv_pcd.paint_uniform_color([0, 0, 1])

o3d.visualization.draw_geometries([   model_pcd, fine_curv_pcd ])

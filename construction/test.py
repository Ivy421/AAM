import argparse
import json
import os
from pathlib import Path

import numpy as np
import open3d as o3d

fine_scan_path = '/home/smmg/AAM/data/runs/20260812_104525/construction/fine_scan/fine_point_result.npz'
meta = np.load(fine_scan_path, allow_pickle=True)
points = np.asarray(meta['points_collection'][1])
points = points[:,:3]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.paint_uniform_color([1,0,0])

o3d.visualization.draw_geometries([pcd])

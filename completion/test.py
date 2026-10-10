import open3d as o3d
import numpy as np

model_path = '/home/smmg/AAM/data/runs/20261010_144417/completion/depression/model.pcd'
fine_fuse_path = '/home/smmg/AAM/data/runs/20261010_144417/construction/fine_scan/fine_fuse.pcd'
model_pcd = o3d.io.read_point_cloud(model_path)
finefuse_pcd = o3d.io.read_point_cloud(fine_fuse_path)
o3d.visualization.draw_geometries([ model_pcd,finefuse_pcd  ])
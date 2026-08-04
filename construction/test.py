import argparse
import json
import os
from pathlib import Path

import numpy as np
import open3d as o3d


# =========================
# Paths / parameters
# =========================
PROJECT_ROOT = Path('/home/smmg/AAM')
DATA_DIR = PROJECT_ROOT / "data" / "runs"
COARSE_DIR = DATA_DIR / "coarse_scan"
FINE_DIR = DATA_DIR / "20260727_173247_constructionfail" / "construction" /"coarse_scan"

FINE_POINT_FILE = FINE_DIR / "coarse_point_result.npz"
fine_fuse_path = FINE_DIR / 'coarse_fuse.pcd' 


def points_to_xyz(points):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2:
        points = points.reshape(-1, points.shape[-1])
    points = points[:, :3]
    return points[np.all(np.isfinite(points), axis=1)]


def load_points_from_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)

    # 优先读 packed 格式；没有则读 points_collection object array
    if "all_points" in data and "offsets" in data:
        all_points = np.asarray(data["all_points"], dtype=float)
        offsets = np.asarray(data["offsets"], dtype=int)
        return [points_to_xyz(all_points[offsets[i]:offsets[i + 1]]) for i in range(len(offsets) - 1)]

    pc = data["points_collection"].tolist()
    return [points_to_xyz(p) for p in pc]

meta = load_points_from_npz(FINE_POINT_FILE)#np.load(FINE_POINT_FILE,allow_pickle=True)
print(meta[0])
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(meta[0])
o3d.visualization.draw_geometries([pcd])

fine_fuse_pcd = o3d.io.read_point_cloud(fine_fuse_path)
o3d.visualization.draw_geometries([fine_fuse_pcd])

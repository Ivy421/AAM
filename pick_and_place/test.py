import numpy as np
import open3d as o3d
gripper_meta = np.load('/home/smmg/AAM/data/runs/20260810_113122/completion/depression/gripper_meta.npz',allow_pickle=True)
top_plane_pcd = o3d.io.read_point_cloud('/home/smmg/AAM/data/runs/20260810_113122/completion/depression/top_plane.pcd')
top_points = np.asarray(top_plane_pcd.points)
grip_height_total = float(
        gripper_meta["grip_body_height"]
        + gripper_meta["base_height"]
        + gripper_meta["v_neck_height"])
print(min(top_points[:, -1]) )
o3d.visualization.draw_geometries([top_plane_pcd])
import open3d as o3d
import numpy as np


completion_out_dir = "E:/HKUSTGZ/AAM/construction/data/completion_result/depression"
bf_pcd  = o3d.io.read_point_cloud(completion_out_dir + '/model.pcd')
aft_pcd  = o3d.io.read_point_cloud(completion_out_dir + '/model_oriented.pcd')
bf_points = np.asarray(bf_pcd.points)
aft_points = np.asarray(aft_pcd.points)

# 自动设置坐标轴尺寸
all_points = np.vstack([bf_points, aft_points])
bbox_min = all_points.min(axis=0)
bbox_max = all_points.max(axis=0)
scale = np.linalg.norm(bbox_max - bbox_min)
axis_size = scale * 0.15

# 世界坐标系：放在原点
world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=axis_size,
    origin=[0, 0, 0]
)

# before 点云坐标系：放在 before 点云中心
bf_center = bf_pcd.get_center()
bf_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=axis_size,
    origin=bf_center
)

# after 点云坐标系：放在 after 点云中心
aft_center = aft_pcd.get_center()
aft_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=axis_size,
    origin=aft_center
)

# 可选：显示包围盒
bf_bbox = bf_pcd.get_axis_aligned_bounding_box()
bf_bbox.color = [1.0, 0.0, 0.0]

aft_bbox = aft_pcd.get_axis_aligned_bounding_box()
aft_bbox.color = [0.0, 0.4, 1.0]

bf_pcd.paint_uniform_color([0.68, 0.45 ,0.55])
aft_pcd.paint_uniform_color([0.5,0.22,0.8])
o3d.visualization.draw_geometries([
    bf_pcd,
    aft_pcd,
    world_frame,
    #bf_frame,
    #aft_frame,
    #bf_bbox,
    #aft_bbox
])
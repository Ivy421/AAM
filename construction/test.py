import numpy as np
import open3d as o3d

pcd_raw = o3d.io.read_point_cloud(
    "E:/HKUSTGZ/AAM/construction/data/completion_result/fused2.pcd"
)
repair_block = o3d.io.read_point_cloud(
    "E:/HKUSTGZ/AAM/construction/data/completion_result/repair_block_points.pcd"
)
pcd_raw.paint_uniform_color([1, 0.5 , 0.5 ])
repair_block.paint_uniform_color([0,0.5,1 ])

o3d.visualization.draw_geometries([pcd_raw, repair_block])
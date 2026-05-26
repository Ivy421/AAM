import open3d as o3d
import numpy as np

depression_dir = 'E:\HKUSTGZ\AAM/construction/data/completion_result/depression'
model_pcd = o3d.io.read_point_cloud(depression_dir+'/model.pcd')
model_p = np.asarray(model_pcd.points)
print(model_p[5000:5010]* 1000)

frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.04)
o3d.visualization.draw_geometries([model_pcd, frame])
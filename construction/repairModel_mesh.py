import numpy as np
import open3d as o3d
import pymeshlab
from model_orientation import orient_stl
import json

repair_block_pcd = o3d.io.read_point_cloud('E:/HKUSTGZ/AAM/construction/data/completion_result/repair_model_pcd.pcd')
#o3d.visualization.draw_geometries([repair_block_pcd])
print("\n开始从 repair_block_pcd 生成网格...")

# 先做一次轻量下采样，避免点太密
#repair_block_pcd = repair_block_pcd.voxel_down_sample(voxel_size=0.0005)

# 估计法向
repair_block_pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.002,
        max_nn=30
    )
)

print("用于重建的点数:", len(repair_block_pcd.points))

############
########## 1， Alpha 重建 Shape 参数：越小越贴点云，越大越容易补过头
alpha = 0.005
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(repair_block_pcd, alpha)
############
############

if len(mesh.triangles) == 0:
    raise RuntimeError("Alpha Shape 重建失败，mesh 为空，请尝试调大 alpha")

# 清理网格
mesh.remove_duplicated_vertices()
mesh.remove_duplicated_triangles()
mesh.remove_degenerate_triangles()
mesh.remove_unreferenced_vertices()
mesh.remove_non_manifold_edges()
mesh.compute_vertex_normals()

print("初始网格顶点数:", len(mesh.vertices))
print("初始网格三角面数:", len(mesh.triangles))


# =========================================================
# 9. 可选：轻微平滑
#    如果你想尽量保棱边，可以先把这段注释掉
# =========================================================
mesh = mesh.filter_smooth_taubin(number_of_iterations=3)
mesh.compute_vertex_normals()

print("平滑后网格顶点数:", len(mesh.vertices))
print("平滑后网格三角面数:", len(mesh.triangles))

v = np.asarray(mesh.vertices)
f = np.asarray(mesh.triangles)

ms = pymeshlab.MeshSet()
ms.add_mesh(pymeshlab.Mesh(vertex_matrix=v, face_matrix=f), "repair_block")

# maxholesize 越大，允许补的孔越大
# 先从小值开始试，比如 50 / 100
ms.meshing_close_holes(maxholesize=200)

# 可选：再清理一下
ms.meshing_remove_connected_component_by_diameter(mincomponentdiag=pymeshlab.PercentageValue(1.0))

m = ms.current_mesh()
v2 = m.vertex_matrix()
f2 = m.face_matrix()

mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(v2)
mesh.triangles = o3d.utility.Vector3iVector(f2)
mesh.compute_vertex_normals()
# =========================================================
# 可视化
# =========================================================
mesh_show = o3d.geometry.TriangleMesh(mesh)
mesh_show.paint_uniform_color([0.7, 0.7, 0.7])
#o3d.visualization.draw_geometries([mesh_show])

# =========================================================
# 10. 保存为可打印格式
# =========================================================
save_dir = "E:/HKUSTGZ/AAM/construction/data/completion_result/"
o3d.io.write_point_cloud(save_dir + "repair_model.pcd", repair_block_pcd)
stl_path = save_dir + "hole_repair_model.stl"
ok1 = o3d.io.write_triangle_mesh(stl_path, mesh)


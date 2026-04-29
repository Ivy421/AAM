import numpy as np
import open3d as o3d
import pymeshlab


# =========================================================
# 1. 读取数据
# =========================================================
defect_pcd1 = o3d.io.read_point_cloud(
    "E:/HKUSTGZ/AAM/construction/data/completion_result/defect_pcd1.pcd"
)
defect_pcd2 = o3d.io.read_point_cloud(
    "E:/HKUSTGZ/AAM/construction/data/completion_result/defect_pcd2.pcd"
)
plane1_pcd = o3d.io.read_point_cloud(
    "E:/HKUSTGZ/AAM/construction/data/completion_result/plane1_pcd.pcd"
)
plane2_pcd = o3d.io.read_point_cloud(
    "E:/HKUSTGZ/AAM/construction/data/completion_result/plane2_pcd.pcd"
)

meta = np.load(
    "E:/HKUSTGZ/AAM/construction/data/completion_result/planes_meta.npz"
)

plane1_model = meta["plane1_model"]
plane2_model = meta["plane2_model"]
n1 = meta["n1"]
n2 = meta["n2"]
object_center = meta["object_center"]


# =========================================================
# 2. 参数
#    如果你的点云单位是“米”，4 mm = 0.004
#    如果你的点云单位是“毫米”，改成 thickness = 4.0
# =========================================================
thickness = 0.004     # 4 mm
step = 0.0003         # 每层间距 0.5 mm


# =========================================================
# 3. 工具函数
# =========================================================
def normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("法向量长度接近 0，无法归一化")
    return v / n


def get_centroid(pcd):
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        raise ValueError("点云为空，无法计算中心")
    return pts.mean(axis=0)


def orient_normal_inward(normal, plane_pcd, object_center):
    """
    让法向量指向物体内部：
    如果法向量朝向 object_center，就保留；
    否则翻转。
    """
    n = normalize(normal)
    c_plane = get_centroid(plane_pcd)

    # 从平面中心指向物体中心
    to_center = np.asarray(object_center, dtype=float) - c_plane

    if np.dot(n, to_center) < 0:
        n = -n

    return n


def extrude_point_cloud_along_normal(defect_pcd, normal, thickness, step):
    """
    将 defect 点云沿 normal 方向拉伸成有厚度的点云
    """
    pts = np.asarray(defect_pcd.points)
    if len(pts) == 0:
        return o3d.geometry.PointCloud()

    n = normalize(normal)

    ts = np.arange(0.0, thickness + 1e-12, step)
    all_layers = [pts + t * n for t in ts]
    pts_extruded = np.vstack(all_layers)

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(pts_extruded)
    return out


def paint_pcd(pcd, color):
    out = o3d.geometry.PointCloud(pcd)
    pts = np.asarray(out.points)
    if len(pts) > 0:
        colors = np.tile(np.array(color).reshape(1, 3), (len(pts), 1))
        out.colors = o3d.utility.Vector3dVector(colors)
    return out


# =========================================================
# 4. 确定两个平面的“向内法向”
# =========================================================
n1_in = orient_normal_inward(n1, plane1_pcd, object_center)
n2_in = orient_normal_inward(n2, plane2_pcd, object_center)

print("n1_in =", n1_in)
print("n2_in =", n2_in)


# =========================================================
# 5. 分别拉伸 4 mm
# =========================================================
extrude_pcd1 = extrude_point_cloud_along_normal(
    defect_pcd1, n1_in, thickness=thickness, step=step
)
extrude_pcd2 = extrude_point_cloud_along_normal(
    defect_pcd2, n2_in, thickness=thickness, step=step
)

pts1 = np.asarray(extrude_pcd1.points)
pts2 = np.asarray(extrude_pcd2.points)

repair_block_pcd = o3d.geometry.PointCloud()
if len(pts1) + len(pts2) > 0:
    repair_block_pcd.points = o3d.utility.Vector3dVector(
        np.vstack([pts1, pts2])
    )

print("extrude_pcd1 点数:", len(pts1))
print("extrude_pcd2 点数:", len(pts2))
print("repair_block_pcd 总点数:", len(np.asarray(repair_block_pcd.points)))


# =========================================================
# 6. 保存结果
# =========================================================
save_dir = "E:/HKUSTGZ/AAM/construction/data/completion_result/"
o3d.io.write_point_cloud(save_dir + "repair_model.pcd", repair_block_pcd)



# =========================================================
# 7. 可视化
# =========================================================
defect_show1 = paint_pcd(defect_pcd1, [1.0, 0.5, 0.5])     # 红
defect_show2 = paint_pcd(defect_pcd2, [0.5, 1.0, 0.5])     # 绿
extrude_show = paint_pcd(repair_block_pcd, [0, 0 ,1 ])  # 蓝

o3d.visualization.draw_geometries([
    defect_show1,
    defect_show2,
    extrude_show
])

# =========================================================
# 8. 从拉伸后的补块点云生成可打印网格
# =========================================================
print("\n开始从 repair_block_pcd 生成网格...")

# 先做一次轻量下采样，避免点太密
mesh_pcd = repair_block_pcd.voxel_down_sample(voxel_size=0.0005)

# 估计法向
mesh_pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.002,
        max_nn=30
    )
)

print("用于重建的点数:", len(mesh_pcd.points))

# Alpha Shape 参数：越小越贴点云，越大越容易补过头
alpha = 0.003

mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(mesh_pcd, alpha)

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
ms.meshing_close_holes(maxholesize=100)

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
# 10. 保存为可打印格式
# =========================================================
stl_path = save_dir + "repair_model.stl"
ply_mesh_path = save_dir + "repair_model.ply"
obj_path = save_dir + "repair_model.obj"

ok1 = o3d.io.write_triangle_mesh(stl_path, mesh)
ok2 = o3d.io.write_triangle_mesh(ply_mesh_path, mesh)
ok3 = o3d.io.write_triangle_mesh(obj_path, mesh)

print("保存结果:")
print("STL:", stl_path, ok1)
print("PLY mesh:", ply_mesh_path, ok2)
print("OBJ:", obj_path, ok3)


# =========================================================
# 11. 可视化最终网格
# =========================================================
mesh_show = o3d.geometry.TriangleMesh(mesh)
mesh_show.paint_uniform_color([0.7, 0.7, 0.7])

o3d.visualization.draw_geometries([mesh_show])
import numpy as np
import open3d as o3d
import json

def estimate_plane_normal(pcd):
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.01,
            max_nn=30
        )
    )
    normals = np.asarray(pcd.normals)
    normal = normals.mean(axis=0)
    norm = np.linalg.norm(normal)
    if norm < 1e-8:
        raise RuntimeError("法向估计失败")
    return normal / norm

def rotation_matrix_from_vectors(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    v = np.cross(a, b)
    c = np.dot(a, b)

    if np.isclose(c, 1.0):
        return np.eye(3)

    if np.isclose(c, -1.0):
        axis = np.array([1, 0, 0])
        if abs(a[0]) > 0.9:
            axis = np.array([0, 1, 0])
        v = np.cross(a, axis)
        v /= np.linalg.norm(v)
        return o3d.geometry.get_rotation_matrix_from_axis_angle(v * np.pi)

    s = np.linalg.norm(v)

    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))
    return R

def orient_model(mesh, n_top, n_side,side_defect_pcd, grip_center):
    """
    输入:
        mesh: open3d TriangleMesh
        top_defect_pcd: 顶面缺陷点云
        side_defect_pcd: 侧面缺陷点云

    输出:
        已对齐的 mesh
    """

    mesh = o3d.geometry.TriangleMesh(mesh)  # copy

    # ---------- Step 1: 顶面对齐到 XOY ----------
    target_z = np.array([0.0, 0.0, 1.0])
    R1 = rotation_matrix_from_vectors(n_top, target_z)
    mesh.rotate(R1, center=(0, 0, 0))
    side_defect_pcd.rotate(R1, center=(0, 0, 0))
    n_top = R1 @ n_top
    n_side = R1 @ n_side
    grip_center = R1 @ grip_center

    # ---------- Step 2: 侧面对齐到 X 轴 ----------
    # 投影到 XOY
    
    n_side_xy = np.array([n_side[0], n_side[1], 0.0])
    norm_xy = np.linalg.norm(n_side_xy)

    if norm_xy < 1e-8:
        raise RuntimeError("side 法向投影失败（可能接近 Z 方向）")

    n_side_xy /= norm_xy

    target_y = np.array([0.0, 1.0, 0.0])

    # 只绕 Z 轴旋转
    cross_val = np.cross(n_side_xy, target_y)
    dot_val = np.dot(n_side_xy, target_y)

    angle = np.arctan2(cross_val[2], dot_val)
    print(angle)
    R2 = o3d.geometry.get_rotation_matrix_from_axis_angle(
        np.array([0.0, 0.0, angle])
    )

    mesh.rotate(R2, center=(0, 0, 0))
    grip_center = R2 @ grip_center

    # ---------- Step 3: 平移到 z = 0 ----------
    bbox = mesh.get_axis_aligned_bounding_box()
    min_bound = bbox.min_bound

    mesh.translate((0.0, 0.0, -min_bound[2]))
    grip_center[2] -= min_bound[2]

    ########### step 4: 模型居中摆放 ##############
    bbox = mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()

    mesh.translate((-center[0], -center[1], 0.0))
    grip_center[0] -= center[0]
    grip_center[1] -= center[1]
    grip_ROI_dist = np.linalg.norm(grip_center[:2] )

    return mesh, grip_ROI_dist

def orient_stl(
    input_stl_path,
    output_stl_path, n_top, n_side,side_pcd , grip_center):
    """
    一步完成：
    STL读取 → 对齐 → 落地 → 保存
    """
    mesh = o3d.io.read_triangle_mesh(input_stl_path)
    if mesh.is_empty():
        raise RuntimeError("输入 STL 为空")

    mesh,grip_ROI_dist = orient_model(mesh, n_top,n_side ,side_pcd,grip_center )
    mesh.compute_vertex_normals()
    ok = o3d.io.write_triangle_mesh(output_stl_path, mesh)
    if not ok:
        raise RuntimeError("写出 STL 失败")

    print(f"[OK] Oriented STL saved to: {output_stl_path}")


save_dir = "E:/HKUSTGZ/AAM/construction/data/completion_result/"
stl_path = save_dir + "hole_whole_model.stl"
mark_path = save_dir + 'mark.json'
with open (mark_path, 'r', encoding='utf-8') as f:
    mark = json.load(f)
meta = np.load(save_dir + 'planes_meta.npz')
if mark['top'] == 'defect_pcd1':
    n_top = meta['n1']
    n_side = meta['n2']
    side_pcd = o3d.io.read_point_cloud(save_dir + 'defect_pcd2.pcd')
    grip_center = meta['defect2_center']
elif mark['top'] == 'defect_pcd2':
    n_top = meta['n2']
    n_side = meta['n1']
    side_pcd = o3d.io.read_point_cloud(save_dir + 'defect_pcd1.pcd')
    grip_center = meta['defect1_center']

orient_stl(
    input_stl_path=stl_path,
    output_stl_path=save_dir + "hole_repair_model_orient.stl",
    n_top = n_top,n_side = n_side,side_pcd = side_pcd , grip_center = grip_center)
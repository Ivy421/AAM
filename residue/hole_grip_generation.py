"""
grip_generation_parellel.py

目标：
1. grip 仍然在 side defect plane 的 fixed_center 生长；
2. grip 生长方向仍然沿 side outward normal；
3. 通过 top defect plane 自动计算 handle_thickness；
4. 使 handle 的一个大面与 top defect plane 对齐，减少打印支撑。

注意：
- 默认 planes_meta.npz / pcd 中坐标单位是 m；
- STL 打印模型按 mm 处理；
- 所以 center 和 plane_model 的 d 项都会乘 UNIT_SCALE=1000。
"""

import json
from pathlib import Path
import open3d as o3d
import numpy as np
import trimesh

BASE_DIR = Path(r"E:\HKUSTGZ\AAM\construction\data\completion_result")

REPAIR_STL = BASE_DIR / "hole_repair_model.stl"
OUTPUT_STL = BASE_DIR / "hole_whole_model.stl"
PLANE_META_PATH = BASE_DIR / "planes_meta.npz"
MARK_PATH = BASE_DIR / "mark.json"

# 原始点云/平面参数通常是 m，打印 STL 用 mm
UNIT_SCALE = 1000.0

# =========================
# 基础工具
# =========================
def normalize(v, eps=1e-12):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("zero vector")
    return v / n

def normalize_plane(plane_model):
    """
    plane_model = [a, b, c, d]
    ax + by + cz + d = 0
    返回单位法向形式。
    """
    p = np.asarray(plane_model, dtype=float).reshape(4)
    n = p[:3]
    d = p[3]
    s = np.linalg.norm(n)
    if s < 1e-12:
        raise ValueError("invalid plane normal")
    return np.r_[n / s, d / s]

def scale_plane_d(plane_model, scale):
    """
    坐标从 m 转为 mm 时，平面方程中法向不变，d 需要乘 scale。
    原式：n·x_m + d_m = 0
    x_mm = scale * x_m
    新式：n·x_mm + d_mm = 0，其中 d_mm = scale * d_m
    """
    p = normalize_plane(plane_model)
    return np.r_[p[:3], p[3] * scale]

def orient_plane_to_normal(plane_model, target_normal):
    """
    调整 plane_model 的法向方向，使其和 target_normal 同向。
    """
    p = normalize_plane(plane_model)
    target_normal = normalize(target_normal)
    if np.dot(p[:3], target_normal) < 0:
        p = -p
    return p

def build_frame_from_side_and_top(origin, side_outward_normal, top_outward_normal):
    """
    local x: grip 生长方向 = side_outward_normal
    local z: handle 厚度方向，尽量等于 top_outward_normal
    local y: handle 宽度方向

    这样可以控制长方体绕 x_axis 的 roll angle，避免 handle 旋转歪。
    """
    x_axis = normalize(side_outward_normal)
    top_outward_normal = normalize(top_outward_normal)

    # 若 top normal 与 x 不完全垂直，则投影到垂直于 x 的平面内
    z_axis = top_outward_normal - np.dot(top_outward_normal, x_axis) * x_axis

    if np.linalg.norm(z_axis) < 1e-8:
        ref = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(ref, x_axis)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        z_axis = ref - np.dot(ref, x_axis) * x_axis

    z_axis = normalize(z_axis)
    y_axis = normalize(np.cross(z_axis, x_axis))
    z_axis = normalize(np.cross(x_axis, y_axis))

    T = np.eye(4)
    T[:3, 0] = x_axis
    T[:3, 1] = y_axis
    T[:3, 2] = z_axis
    T[:3, 3] = np.asarray(origin, dtype=float)

    return T, x_axis, y_axis, z_axis

def compute_handle_thickness_to_top_plane(
    attach_center_mm,
    z_axis,
    top_plane_model_mm,
    top_outward_normal,
    margin=0.0,
    min_thickness=3.0,
    max_thickness=40.0,
):
    """
    计算 handle_thickness，使 handle 的一个大面与 top defect plane 对齐。

    本代码中 handle 截面以 attach_center 为中心：
        face_plus  = attach_center + 0.5 * thickness * z_axis
        face_minus = attach_center - 0.5 * thickness * z_axis

    这里自动判断 top plane 在 +z_axis 侧还是 -z_axis 侧，选择能得到正厚度的那一侧。
    """
    attach_center_mm = np.asarray(attach_center_mm, dtype=float)
    z_axis = normalize(z_axis)

    top_plane = orient_plane_to_normal(top_plane_model_mm, top_outward_normal)
    n_top = top_plane[:3]
    d_top = top_plane[3]

    s0 = float(np.dot(n_top, attach_center_mm) + d_top)
    denom = float(np.dot(n_top, z_axis))

    if abs(denom) < 1e-8:
        raise ValueError(
            "z_axis is nearly parallel to top plane; cannot compute handle thickness."
        )

    # 令 attach_center + sign * 0.5 * t * z_axis 落到 top plane 上
    # n·(C + sign * 0.5*t*z) + d = 0
    candidates = []
    for sign in [+1.0, -1.0]:
        t = -2.0 * s0 / (sign * denom)
        if t > 0:
            candidates.append((t, sign))

    if len(candidates) == 0:
        raise ValueError(
            f"Cannot find positive handle_thickness. s0={s0}, denom={denom}"
        )

    # 选较小正值，避免 handle 过厚
    raw_thickness, aligned_face_sign = min(candidates, key=lambda x: x[0])
    handle_thickness = raw_thickness + margin
    handle_thickness = float(np.clip(handle_thickness, min_thickness, max_thickness))

    aligned_face_center = attach_center_mm + aligned_face_sign * 0.5 * handle_thickness * z_axis
    residual_dist = float(np.dot(n_top, aligned_face_center) + d_top)

    info = {
        "s0_center_to_top_plane_signed_mm": s0,
        "n_top_dot_z_axis": denom,
        "raw_thickness_mm": raw_thickness,
        "margin_mm": margin,
        "handle_thickness_mm": handle_thickness,
        "aligned_face_sign": aligned_face_sign,
        "aligned_face_residual_signed_dist_mm": residual_dist,
    }

    return handle_thickness, info

def create_rectangular_loft(sections):
    """
    sections = [(x, width_y, thickness_z), ...]
    local x: 生长方向
    local y: 宽度方向
    local z: 厚度方向
    """
    vertices = []
    faces = []

    for x, w, t in sections:
        vertices.extend([
            [x, -w / 2, -t / 2],
            [x,  w / 2, -t / 2],
            [x,  w / 2,  t / 2],
            [x, -w / 2,  t / 2],
        ])

    for i in range(len(sections) - 1):
        a = 4 * i
        b = 4 * (i + 1)
        quads = [
            [a + 0, a + 1, b + 1, b + 0],
            [a + 1, a + 2, b + 2, b + 1],
            [a + 2, a + 3, b + 3, b + 2],
            [a + 3, a + 0, b + 0, b + 3],
        ]
        for q in quads:
            faces.append([q[0], q[1], q[2]])
            faces.append([q[0], q[2], q[3]])

    # 起始端盖
    faces.append([0, 2, 1])
    faces.append([0, 3, 2])

    # 末端端盖
    last = 4 * (len(sections) - 1)
    faces.append([last + 0, last + 1, last + 2])
    faces.append([last + 0, last + 2, last + 3])

    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=float),
        faces=np.asarray(faces, dtype=int),
        process=True,
    )
    mesh.fix_normals()
    return mesh

def create_grip_local(
    handle_length=30.0,
    handle_width=3.0,
    handle_thickness=10.0,
    neck_length=1.0,
    neck_width=2.0,
    neck_thickness=1.2,
    fracture_distance=0.45,
    notch_length=0.6,
    notch_ratio=0.8,
    transition_length=1.0,
    embed_depth=1.0,
):
    """
    无底座：模型外表面 -> 短窄颈 -> 预断槽 -> 过渡段 -> handle。
    注意：neck 不跟随 handle_thickness 变厚，避免连接残留变大。
    """
    if fracture_distance <= 0.25:
        raise ValueError("fracture_distance too small")
    if fracture_distance >= neck_length:
        raise ValueError("fracture_distance must be smaller than neck_length")

    x0 = -embed_depth
    x1 = 0.0
    x2 = max(fracture_distance - notch_length / 2.0, 0.05)
    x3 = fracture_distance
    x4 = min(fracture_distance + notch_length / 2.0, neck_length - 0.05)
    x5 = neck_length
    x6 = neck_length + transition_length
    x7 = neck_length + transition_length + handle_length

    waist_width = neck_width
    waist_thickness = neck_thickness * notch_ratio

    sections = [
        (x0, neck_width, neck_thickness),
        (x1, neck_width, neck_thickness),
        (x2, neck_width, neck_thickness),
        (x3, waist_width, waist_thickness),
        (x4, neck_width, neck_thickness),
        (x5, neck_width, neck_thickness),
        (x6, handle_width, handle_thickness),
        (x7, handle_width, handle_thickness),
    ]
    return create_rectangular_loft(sections)

def load_repair_mesh_as_mm(repair_stl, unit_scale=1000.0):
    repair = trimesh.load(str(repair_stl), force="mesh")
    if repair.extents.max() < 1.0:
        print("[INFO] repair STL seems in meters, scaling to mm")
        repair.apply_scale(unit_scale)
    repair.remove_unreferenced_vertices()
    repair.fix_normals()
    return repair

def export_combined(repair, grip_world, output_stl, show=False):
    combined = trimesh.util.concatenate([repair, grip_world])
    combined.remove_unreferenced_vertices()
    combined.fix_normals()
    combined.export(str(output_stl))

    print("[OK] exported combined STL:", output_stl)
    print("[INFO] repair faces:", len(repair.faces))
    print("[INFO] grip faces:", len(grip_world.faces))
    print("[INFO] combined faces:", len(combined.faces))

    if show:
        combined.show()

    return combined

def add_parallel_grip_structure(
    repair_stl,
    output_stl,
    attach_center_m,
    side_outward_normal,
    top_outward_normal,
    top_plane_model_m,
    handle_length=50.0,
    handle_width=3.0,
    neck_length=1.0,
    neck_width=3.0,
    neck_thickness=1.2,
    fracture_distance=0.45,
    notch_length=0.6,
    notch_ratio=0.8,
    transition_length=1.0,
    embed_depth=1.0,
    thickness_margin=0.0,
    min_handle_thickness=3.0,
    max_handle_thickness=40.0,
    export_grip_only=None,
    show=False,
):
    repair = load_repair_mesh_as_mm(repair_stl, UNIT_SCALE)

    attach_center_mm = np.asarray(attach_center_m, dtype=float) * UNIT_SCALE
    side_outward_normal = normalize(side_outward_normal)
    top_outward_normal = normalize(top_outward_normal)
    top_plane_model_mm = scale_plane_d(top_plane_model_m, UNIT_SCALE)

    T, x_axis, y_axis, z_axis = build_frame_from_side_and_top(
        origin=attach_center_mm,
        side_outward_normal=side_outward_normal,
        top_outward_normal=top_outward_normal,
    )

    handle_thickness, thickness_info = compute_handle_thickness_to_top_plane(
        attach_center_mm=attach_center_mm,
        z_axis=z_axis,
        top_plane_model_mm=top_plane_model_mm,
        top_outward_normal=top_outward_normal,
        margin=thickness_margin,
        min_thickness=min_handle_thickness,
        max_thickness=max_handle_thickness,
    )

    print("========== AUTO HANDLE THICKNESS ==========")
    for k, v in thickness_info.items():
        print(f"{k}: {v}")

    grip_local = create_grip_local(
        handle_length=handle_length,
        handle_width=handle_width,
        handle_thickness=handle_thickness,
        neck_length=neck_length,
        neck_width=neck_width,
        neck_thickness=neck_thickness,
        fracture_distance=fracture_distance,
        notch_length=notch_length,
        notch_ratio=notch_ratio,
        transition_length=0,  #transition_length
        embed_depth=embed_depth,
    )

    grip_world = grip_local.copy()
    grip_world.apply_transform(T)

    if export_grip_only is not None:
        grip_world.export(str(export_grip_only))
        print("[OK] exported grip only:", export_grip_only)

    combined = export_combined(repair, grip_world, output_stl, show=show)

    print("========== GRIP FRAME ==========")
    print("attach_center_mm:", attach_center_mm)
    print("x_axis / side outward:", x_axis)
    print("y_axis / handle width:", y_axis)
    print("z_axis / handle thickness:", z_axis)
    print("handle_thickness_mm:", handle_thickness)

    return combined, grip_world, handle_thickness, thickness_info

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
    grip_dist_len = grip_center[1]     # grip起始点的打印平台坐标y向偏移量
    grip_dist_width = grip_center[0]   # grip起始点的打印平台坐标x向偏移量

    return mesh, grip_dist_len,grip_dist_width

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
    
    grip_center = np.asarray(grip_center, dtype=float) * UNIT_SCALE
    mesh,grip_dist_len, grip_dist_width = orient_model(mesh, n_top,n_side ,side_pcd,grip_center )
    mesh.compute_vertex_normals()
    ok = o3d.io.write_triangle_mesh(output_stl_path, mesh)
    if not ok:
        raise RuntimeError("写出 STL 失败")

    print(f"[OK] Oriented STL saved to: {output_stl_path}")
    return grip_dist_len, grip_dist_width

if __name__ == "__main__":
    plane_meta = np.load(str(PLANE_META_PATH))
    with open(str(MARK_PATH), "r", encoding="utf-8") as f:
        mark = json.load(f)

    side_mark = mark["side"]
    print("side plane:", side_mark)

    if side_mark == "defect_pcd1":
        side_id = 1
        top_id = 2
    elif side_mark == "defect_pcd2":
        side_id = 2
        top_id = 1
    else:
        raise ValueError(f"Unknown side mark: {side_mark}")

    attach_center_m = plane_meta[f"defect{side_id}_center"]

    # 你已确认 n1/n2 是朝内的，所以 outward = -inward
    side_outward_normal = -plane_meta[f"n{side_id}"]
    top_outward_normal = -plane_meta[f"n{top_id}"]
    n_top = plane_meta[f"n{top_id}"]
    n_side = plane_meta[f"n{side_id}"]
    top_plane_model_m = plane_meta[f"plane{top_id}_model"]

    print("attach_center_m:", attach_center_m)
    print("side_outward_normal:", side_outward_normal)
    print("top_outward_normal:", top_outward_normal)

    combined, grip_world, handle_thickness, thickness_info = add_parallel_grip_structure(
        repair_stl=REPAIR_STL,
        output_stl=OUTPUT_STL,
        attach_center_m=attach_center_m,
        side_outward_normal=side_outward_normal,
        top_outward_normal=top_outward_normal,
        top_plane_model_m=top_plane_model_m,

        # grip 主体尺寸
        handle_length=50.0,
        handle_width=3.0,

        # 无底座，直接窄颈连接到 side 外表面
        neck_length=1.0,
        neck_width=3.0,
        neck_thickness=1.2,
        fracture_distance=0.45,
        notch_length=0.6,
        notch_ratio=0.8,
        transition_length=1.0,
        embed_depth=1.0,

        # 自动厚度控制
        thickness_margin=0.0,
        min_handle_thickness=3.0,
        max_handle_thickness=40.0,

        export_grip_only=None,
        show=False,
    )

    save_dir = "E:/HKUSTGZ/AAM/construction/data/completion_result/"
    stl_path = save_dir + "hole_whole_model.stl"
    mark_path = save_dir + 'mark.json'
    with open (mark_path, 'r', encoding='utf-8') as f:
        mark = json.load(f)
    meta = np.load(save_dir + 'planes_meta.npz')
    if mark['top'] == 'defect_pcd1':
        #n_top = meta['n1']
        #n_side = meta['n2']
        side_pcd = o3d.io.read_point_cloud(save_dir + 'defect_pcd2.pcd')
        #grip_center = meta['defect2_center']
    elif mark['top'] == 'defect_pcd2':
        #n_top = meta['n2']
        #n_side = meta['n1']
        side_pcd = o3d.io.read_point_cloud(save_dir + 'defect_pcd1.pcd')
        #grip_center = meta['defect1_center']

    grip_dist_len, grip_dist_width = orient_stl(
        input_stl_path=stl_path,
        output_stl_path=save_dir + "hole_repair_model_orient.stl",
        n_top = n_top,n_side = n_side,side_pcd = side_pcd , grip_center = attach_center_m)
    
    np.savez(
        save_dir + 'grip_meta.npz',
        grip_dist_len = grip_dist_len,
        grip_dist_width = grip_dist_width,
        handle_length=50.0,
        handle_width=3.0,
        neck_length=1.0,
        neck_width=3.0,
        neck_thickness=1.2,
        fracture_distance=0.45,
        notch_length=0.6,
        notch_ratio=0.8,
        transition_length=1.0,
        embed_depth=1.0,
        thickness_margin=0.0,
        min_handle_thickness=3.0,
        max_handle_thickness=40.0,
        handle_thickness = handle_thickness

    )
    print(grip_dist_len, grip_dist_width, handle_thickness)
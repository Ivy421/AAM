import numpy as np
import open3d as o3d
import trimesh


def normalize(v, eps=1e-12):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("zero vector")
    return v / n


def build_frame_from_normal(origin, outward_normal, preferred_up=np.array([0, 0, 1.0])):
    """
    局部坐标：
    x轴 = 夹持结构向外生长方向
    y轴 = 夹持杆宽度方向
    z轴 = 夹持杆厚度方向
    """
    x_axis = normalize(outward_normal)

    preferred_up = normalize(preferred_up)
    z_axis = preferred_up - np.dot(preferred_up, x_axis) * x_axis

    if np.linalg.norm(z_axis) < 1e-6:
        preferred_up = np.array([0, 1.0, 0])
        z_axis = preferred_up - np.dot(preferred_up, x_axis) * x_axis

    z_axis = normalize(z_axis)
    y_axis = normalize(np.cross(z_axis, x_axis))
    z_axis = normalize(np.cross(x_axis, y_axis))

    T = np.eye(4)
    T[:3, 0] = x_axis
    T[:3, 1] = y_axis
    T[:3, 2] = z_axis
    T[:3, 3] = origin

    return T


def create_rectangular_loft(sections):
    """
    sections:
    [
        (x, width_y, thickness_z),
        ...
    ]
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
        vertices=np.asarray(vertices),
        faces=np.asarray(faces),
        process=True
    )
    mesh.fix_normals()
    return mesh


def create_min_residue_grip_local(
    handle_length=45.0,
    handle_width=10.0,
    handle_thickness=4.0,

    neck_length=3.0,
    neck_width=1.0,
    neck_thickness=0.8,

    fracture_distance=0.55,
    notch_length=0.7,
    notch_ratio=0.38,

    transition_length=4.0,
    embed_depth=0.25,
):
    """
    无底座夹持结构：
    模型外表面 -> 极短残留窄颈 -> V形预断槽 -> 窄颈 -> 过渡段 -> 长方体夹持杆

    fracture_distance 越小，折断后残留越少。
    建议不要小于 0.4 mm，否则容易撕坏模型外表面。
    """

    if fracture_distance <= 0.3:
        raise ValueError("fracture_distance too small, risk damaging repair surface.")

    if fracture_distance >= neck_length:
        raise ValueError("fracture_distance must be smaller than neck_length.")

    x0 = -embed_depth
    x1 = 0.0

    x2 = max(fracture_distance - notch_length / 2, 0.05)
    x3 = fracture_distance
    x4 = min(fracture_distance + notch_length / 2, neck_length - 0.05)

    x5 = neck_length
    x6 = neck_length + transition_length
    x7 = neck_length + transition_length + handle_length

    waist_width = neck_width * notch_ratio
    waist_thickness = neck_thickness * notch_ratio

    sections = [
        # 轻微嵌入修补块内部，保证 Boolean Union 能融合
        (x0, neck_width, neck_thickness),

        # 模型外表面位置，无底座，直接窄颈连接
        (x1, neck_width, neck_thickness),

        # V槽前
        (x2, neck_width, neck_thickness),

        # 最细处，优先从这里折断
        (x3, waist_width, waist_thickness),

        # V槽后
        (x4, neck_width, neck_thickness),

        # 窄颈结束
        (x5, neck_width, neck_thickness),

        # 过渡到加宽夹持杆
        (x6, handle_width, handle_thickness),

        # 长方体夹持杆末端
        (x7, handle_width, handle_thickness),
    ]

    return create_rectangular_loft(sections)

def export_combined(repair, grip_world, output_stl):
    """
    不做 Boolean，直接把 repair + grip 合成一个 STL。
    只要 grip 和 repair 有 0.2~0.5 mm 的实体重叠，Bambu Studio 通常可以正常切片。
    """
    combined = trimesh.util.concatenate([repair, grip_world])
    combined.show()
    combined.remove_unreferenced_vertices()
    combined.fix_normals()
    

    combined.export(output_stl)

    print("[OK] exported combined STL:", output_stl)
    print("[INFO] repair vertices:", len(repair.vertices), "faces:", len(repair.faces))
    print("[INFO] grip vertices:", len(grip_world.vertices), "faces:", len(grip_world.faces))
    print("[INFO] combined vertices:", len(combined.vertices), "faces:", len(combined.faces))

    return combined

def add_grip_structure(
    repair_stl,
    output_stl,
    attach_center,
    outward_normal,

    handle_length=45.0,
    handle_width=10.0,
    handle_thickness=4.0,

    neck_length=3.0,
    neck_width=1.0,
    neck_thickness=0.8,

    fracture_distance=0.55,
    notch_length=0.7,
    notch_ratio=0.38,

    transition_length=4.0,
    embed_depth=0.25,

    export_grip_only="grip_only.stl",
):
    repair = trimesh.load(repair_stl, force="mesh")
    repair.apply_scale(1000.0)
    repair.remove_unreferenced_vertices()
    repair.fix_normals()

    attach_center = np.asarray(attach_center, dtype=float)
    outward_normal = normalize(outward_normal)

    grip_local = create_min_residue_grip_local(
        handle_length=handle_length,
        handle_width=handle_width,
        handle_thickness=handle_thickness,
        neck_length=neck_length,
        neck_width=neck_width,
        neck_thickness=neck_thickness,
        fracture_distance=fracture_distance,
        notch_length=notch_length,
        notch_ratio=notch_ratio,
        transition_length=transition_length,
        embed_depth=embed_depth,
    )

    T = build_frame_from_normal(
        origin=attach_center,
        outward_normal=outward_normal,
    )

    grip_world = grip_local.copy()
    grip_world.apply_transform(T)

    if export_grip_only is not None:
        grip_world.export(export_grip_only)

    combined = export_combined(repair=repair, grip_world=grip_world,  output_stl=output_stl )

    print("[OK] exported:", output_stl)
    print("[INFO] attach_center:", attach_center)
    print("[INFO] outward_normal:", outward_normal)
    print("[INFO] estimated remaining residue length:", fracture_distance, "mm")

    return combined, grip_world


if __name__ == "__main__":

    # ===============================
    # 你需要改这里
    # ===============================

    repair_stl = "E:\HKUSTGZ\AAM\construction\data\completion_result/repair_model.stl"
    output_stl = "E:\HKUSTGZ\AAM\construction\data\completion_result/grip.stl"

    plane_meta = np.load('E:\HKUSTGZ\AAM\construction\data\completion_result/planes_meta.npz')

    #确认安装中心为plane1 center
    fixed_center = plane_meta['defect1_center']
    fixed_center = fixed_center * 1000
    # fixed_center= np.array( [400.83036001, -28.31026938 ,143.04948514], float)
    n1 = -plane_meta['n1']  # 令法向量朝外

    whole_model, grip = add_grip_structure(
        repair_stl=repair_stl,
        output_stl=output_stl,
        attach_center=fixed_center,
        outward_normal=n1,

        # 夹爪夹持长方体
        handle_length=45.0,
        handle_width=10.0,
        handle_thickness=4.0,

        # 无底座，直接窄颈连接到外表面
        neck_length=3.0,
        neck_width=1.0,
        neck_thickness=0.8,

        # 越小残留越少；建议 0.45–0.8 mm
        fracture_distance=0.55,

        # V形预断槽
        notch_length=0.7,
        notch_ratio=0.38,

        # 从窄颈过渡到宽夹持杆
        transition_length=4.0,

        # 轻微嵌入模型内部，保证融合
        embed_depth=0.25,

        export_grip_only = None,
    )

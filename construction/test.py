import open3d as o3d
import numpy as np
import json,trimesh

def make_arrow(start, direction, length=20, color=[1, 0, 0]):
    direction = np.asarray(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)

    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.0005,
        cone_radius=0.0012,
        cylinder_height=length * 0.8,
        cone_height=length * 0.2,
    )

    # 默认 arrow 沿 +Z，需要旋转到 direction
    z = np.array([0, 0, 1.0])
    R = o3d.geometry.get_rotation_matrix_from_axis_angle(
        np.cross(z, direction) / (np.linalg.norm(np.cross(z, direction)) + 1e-12)
        * np.arccos(np.clip(np.dot(z, direction), -1, 1))
    )

    arrow.rotate(R, center=np.zeros(3))
    arrow.translate(start)
    arrow.paint_uniform_color(color)
    return arrow

repair_stl = "E:\HKUSTGZ\AAM\construction\data\completion_result/repair_model.stl"
plane_meta = np.load('E:\HKUSTGZ\AAM\construction\data\completion_result/planes_meta.npz')
with open ('E:\HKUSTGZ\AAM\construction\data\completion_result/mark.json','r',encoding='utf-8') as f:
    mark = json.load(f)
repair_model_pcd = o3d.io.read_point_cloud("E:/HKUSTGZ/AAM/construction/data/completion_result/repair_model_pcd.pcd")

print('side plane:',mark['side'])
#确认安装中心为侧面的平面
if mark['side'] == 'defect_pcd1':
    fixed_center = plane_meta['defect1_center']
    n = -plane_meta['n1']  # 令法向量朝外
else:
    fixed_center = plane_meta['defect2_center']
    n = -plane_meta['n2']

# 改成你的 center
outward_normal = n   # 改成你的 outward_normal

arrow = make_arrow(
    start=fixed_center,
    direction=outward_normal,
    length=0.02,
    color=[1, 0, 0]
)

o3d.visualization.draw_geometries(
    [repair_model_pcd, arrow],
    mesh_show_back_face=True
)
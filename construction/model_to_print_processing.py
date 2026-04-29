import os
import sys
import traceback
from pathlib import Path
import open3d as o3d
import numpy as np
import trimesh

# =========================================================
# 配置区：按需修改
# =========================================================
INPUT_STL = r"E:/HKUSTGZ/AAM/construction/data/completion_result/repair_block_4mm.stl"
OUTPUT_STL = r"E:/HKUSTGZ/AAM/construction/data/completion_result/repair_block_4mm_fixed.stl"

# 如果你前面的点云/网格坐标单位是“米”，而你希望导出给切片软件时用“毫米”，设为 True
# 若你的 STL 已经是毫米单位，就设为 False
SCALE_TO_MM = True

# 是否尝试补小孔
FILL_HOLES = True

# 是否做轻微网格简化（通常先关掉）
SIMPLIFY = False
TARGET_FACE_COUNT = 50000

# =========================================================
# 工具函数
# =========================================================
def print_header(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

def mesh_summary(mesh, name="mesh"):
    print(f"[{name}] 顶点数: {len(mesh.vertices)}")
    print(f"[{name}] 三角面数: {len(mesh.faces)}")
    print(f"[{name}] watertight: {mesh.is_watertight}")
    print(f"[{name}] winding_consistent: {mesh.is_winding_consistent}")
    print(f"[{name}] euler_number: {mesh.euler_number}")
    try:
        print(f"[{name}] 体积: {mesh.volume}")
    except Exception:
        pass
    bounds = mesh.bounds
    size = bounds[1] - bounds[0]
    print(f"[{name}] 包围盒尺寸: {size}")

def save_mesh(mesh, path: str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out_path))
    print(f"已保存: {out_path}")

# =========================================================
# 主流程
# =========================================================
def main():
    print_header("开始读取 STL")

    in_path = Path(INPUT_STL)
    if not in_path.exists():
        raise FileNotFoundError(f"找不到输入文件: {in_path}")

    mesh = trimesh.load_mesh(str(in_path), process=False)
    if mesh is None:
        raise RuntimeError("读取 STL 失败")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            [g for g in mesh.geometry.values()]
        )

    mesh_summary(mesh, "原始网格")

    print_header("基础修复")
    # 去掉无穷/NaN 顶点
    if not np.isfinite(mesh.vertices).all():
        valid_vertex = np.isfinite(mesh.vertices).all(axis=1)
        mesh.update_vertices(valid_vertex)
        print("已移除 NaN/Inf 顶点")

    # 删除重复面
    mesh.update_faces(mesh.unique_faces())

    # 删除退化面
    mesh.update_faces(mesh.nondegenerate_faces())

    # 删除未引用顶点
    mesh.remove_unreferenced_vertices()

    # 修复法向和朝向
    mesh.fix_normals()
    try:
        trimesh.repair.fix_winding(mesh)
    except Exception:
        pass

    # 尝试补小孔
    if FILL_HOLES:
        try:
            before = mesh.is_watertight
            trimesh.repair.fill_holes(mesh)
            after = mesh.is_watertight
            print(f"fill_holes 已执行，watertight: {before} -> {after}")
        except Exception as e:
            print(f"fill_holes 执行失败: {e}")

    # 再清理一次
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()

    mesh_summary(mesh, "基础修复后")

    print_header("单位检查与缩放")
    bounds = mesh.bounds
    size = bounds[1] - bounds[0]
    print(f"当前包围盒尺寸: {size}")
    if SCALE_TO_MM:
        mesh.apply_scale(1000.0)
        print("已按 1000 倍缩放：米 -> 毫米")
    else:
        print("未做单位缩放")

    mesh_summary(mesh, "缩放后")

    print_header("可选简化")
    if SIMPLIFY:
        try:
            mesh = mesh.simplify_quadric_decimation(TARGET_FACE_COUNT)
            print(f"已简化到约 {TARGET_FACE_COUNT} 个三角面")
        except Exception as e:
            print(f"简化失败，跳过: {e}")
    else:
        print("未启用简化")

    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()

    print_header("最终检查")
    mesh_summary(mesh, "最终网格")

    # 若仍非 watertight，给出提醒，但仍然导出，便于在切片软件里再检查
    if not mesh.is_watertight:
        print("警告: 网格仍然不是 watertight。建议再用 MeshLab / Blender / 切片软件检查。")

    print_header("导出")
    save_mesh(mesh, OUTPUT_STL)

    # 额外导出一份 OBJ，便于检查
    extra_obj = str(Path(OUTPUT_STL).with_suffix(".obj"))
    save_mesh(mesh, extra_obj)

    ######### 可视化最终mesh
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
    mesh_o3d.compute_vertex_normals()
    mesh_o3d.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([mesh_o3d],window_name="Final Mesh")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n程序报错：")
        print(e)
        print("\n详细回溯：")
        traceback.print_exc()
        sys.exit(1)

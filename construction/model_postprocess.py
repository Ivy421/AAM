import os
import traceback

depression_dir = "E:/HKUSTGZ/AAM/construction/data/completion_result/depression"
INPUT_MESH = depression_dir + "/model_oriented.stl"
OUTPUT_MESH = depression_dir + "/print_model.stl"


# =========================================================
# 方案1：优先用 pymeshlab 做较完整的 mesh repair
# =========================================================
def repair_with_pymeshlab(input_path, output_path):
    import pymeshlab

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(input_path)
    m = ms.current_mesh()

    print("=== PyMeshLab repair ===")
    print(f"[Before] vertices = {m.vertex_number()}, faces = {m.face_number()}")

    def safe_filter(name, **kwargs):
        try:
            ms.apply_filter(name, **kwargs)
            print(f"[OK] {name}")
        except Exception as e:
            print(f"[SKIP] {name} -> {e}")

    # ---------- 基础清理 ----------
    safe_filter("meshing_remove_duplicate_vertices")
    safe_filter("meshing_remove_duplicate_faces")
    safe_filter("meshing_remove_null_faces")
    safe_filter("meshing_remove_unreferenced_vertices")

    # ---------- 非流形修复 ----------
    safe_filter("meshing_repair_non_manifold_edges")
    safe_filter("meshing_repair_non_manifold_vertices")

    # ---------- 小孔修补（可按需要调大/调小） ----------
    safe_filter("meshing_close_holes", maxholesize=1000)

    # ---------- 再清理一次 ----------
    safe_filter("meshing_remove_duplicate_vertices")
    safe_filter("meshing_remove_duplicate_faces")
    safe_filter("meshing_remove_null_faces")
    safe_filter("meshing_remove_unreferenced_vertices")

    # ---------- 法向/面朝向整理 ----------
    safe_filter("meshing_re_orient_faces_coherently")

    m2 = ms.current_mesh()
    print(f"[After ] vertices = {m2.vertex_number()}, faces = {m2.face_number()}")

    ms.save_current_mesh(output_path)
    print(f"Saved: {output_path}")
    return True


# =========================================================
# 方案2：如果没有 pymeshlab，就用 open3d 做基础修复
# =========================================================
def repair_with_open3d(input_path, output_path):
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(input_path)
    if mesh.is_empty():
        raise RuntimeError("读取 STL 失败或网格为空")

    print("=== Open3D repair (fallback) ===")
    print(f"[Before] vertices = {len(mesh.vertices)}, faces = {len(mesh.triangles)}")

    # 基础清理
    mesh = mesh.remove_duplicated_vertices()
    mesh = mesh.remove_duplicated_triangles()
    mesh = mesh.remove_degenerate_triangles()
    mesh = mesh.remove_non_manifold_edges()
    mesh = mesh.remove_unreferenced_vertices()

    # 方向整理（有些版本可能不支持，包一层 try）
    try:
        if mesh.is_orientable():
            mesh = mesh.orient_triangles()
    except Exception:
        pass

    mesh.compute_vertex_normals()

    print(f"[After ] vertices = {len(mesh.vertices)}, faces = {len(mesh.triangles)}")

    ok = o3d.io.write_triangle_mesh(output_path, mesh, write_ascii=False)
    if not ok:
        raise RuntimeError("写出 STL 失败")
    print(f"Saved: {output_path}")
    return True

# =========================================================
# 可视化 mesh
# =========================================================
def visualize_mesh(mesh_path, window_name="Mesh Viewer"):
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if mesh.is_empty():
        raise RuntimeError(f"读取失败或网格为空: {mesh_path}")

    mesh.compute_vertex_normals()

    print(f"[Visualize] {mesh_path}")
    print(f"vertices = {len(mesh.vertices)}, faces = {len(mesh.triangles)}")

    o3d.visualization.draw_geometries(
        [mesh],
        window_name=window_name,
        mesh_show_back_face=True
    )
# =========================================================
# 主程序
# =========================================================
if __name__ == "__main__":
    if not os.path.exists(INPUT_MESH):
        raise FileNotFoundError(f"找不到输入文件: {INPUT_MESH}")

    try:
        repair_with_pymeshlab(INPUT_MESH, OUTPUT_MESH)
    except Exception as e:
        print("\n[PyMeshLab failed] 开始尝试 Open3D fallback...")
        print(e)
        traceback.print_exc()
        repair_with_open3d(INPUT_MESH, OUTPUT_MESH)
    visualize_mesh(OUTPUT_MESH, window_name="Processed STL")
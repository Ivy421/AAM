import os
import glob
import numpy as np
import open3d as o3d

"""
STL坐标系变换定义：
XOY 平面：
    打印平台平面 / 模型最低点所在平面

Z=0：
    模型最低点接触打印平台

XY 原点：
    repair_point_center 投影到 XOY 平面的位置

-Z 方向：
    被选中的 side 面 outward normal 方向

-Y 方向：
    另一个 side 面 outward normal 方向，也就是你说的“朝外”方向
"""

# =========================================================
# User config
# =========================================================
COMPLETION_DIR = r"E:\HKUSTGZ\AAM\construction\data\completion_result\depression"

# defect_world_y -> which side family should face down on XOY plane
# 如果你后续发现左右逻辑相反，只需要交换这里的 u / v。
DEFECT_Y_TO_DOWN_SIDE = {
    "left": "u",
    "right": "v",
}
UNIT_SCALE = 1000

TARGET_DOWN_NORMAL = np.array([0.0, 0.0, -1.0])   # side outward normal points to -Z, face touches XOY bed
TARGET_OTHER_NORMAL = np.array([0.0, -1.0, 0.0])  # other side outward normal points to -Y, face parallel to XOZ


# =========================================================
# Basic geometry utils
# =========================================================
def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError(f"zero-length vector: {v}")
    return v / n


def unwrap_np_value(x):
    """Convert np scalar/object array to normal Python value when possible."""
    if isinstance(x, np.ndarray):
        if x.shape == ():
            return x.item()
        return x
    return x


def load_meta(meta_path: str | None = None, completion_dir: str = COMPLETION_DIR):
    """
    Load meta.npz.

    当前 hole_completion 代码中可能存在两种路径：
        1) .../depression/meta.npz
        2) .../depressionmeta.npz      # 如果保存时少写了 '/'
    本函数会自动兼容。
    """
    candidates = []

    if meta_path is not None:
        candidates.append(meta_path)

    candidates.extend([
        os.path.join(completion_dir, "meta.npz"),
        completion_dir + "meta.npz",
        os.path.join(os.path.dirname(completion_dir), "depressionmeta.npz"),
    ])

    meta_path_found = None
    for p in candidates:
        if p and os.path.exists(p):
            meta_path_found = p
            break

    if meta_path_found is None:
        raise FileNotFoundError(
            "Cannot find meta.npz. Tried:\n" + "\n".join(candidates)
        )

    data = np.load(meta_path_found, allow_pickle=True)

    side_n_mark = unwrap_np_value(data["side_n_mark"])
    if not isinstance(side_n_mark, dict):
        raise TypeError(f"side_n_mark should be dict, got {type(side_n_mark)}")

    side_n_mark = {
        str(k): normalize(np.asarray(v, dtype=float))
        for k, v in side_n_mark.items()
    }

    repair_point_center = np.asarray(data["repair_point_center"], dtype=float).reshape(3)
    n_axis = normalize(np.asarray(data["n_axis"], dtype=float).reshape(3))
    defect_world_y = str(unwrap_np_value(data["defect_world_y"])).lower()

    meta = {
        "meta_path": meta_path_found,
        "side_n_mark": side_n_mark,
        "repair_point_center": repair_point_center,
        "n_axis": n_axis,
        "defect_world_y": defect_world_y,
    }

    if "top_plane_model" in data:
        meta["top_plane_model"] = np.asarray(data["top_plane_model"], dtype=float)

    return meta


def select_side_keys(side_n_mark: dict, defect_world_y: str):
    """
    Choose which side should be down and which side should face -Y.

    Example side_n_mark keys:
        max_u_outward, min_v_outward, max_v_outward, ...
    """
    if defect_world_y not in DEFECT_Y_TO_DOWN_SIDE:
        raise ValueError(
            f"defect_world_y must be one of {list(DEFECT_Y_TO_DOWN_SIDE.keys())}, "
            f"got {defect_world_y}"
        )

    down_family = DEFECT_Y_TO_DOWN_SIDE[defect_world_y]  # 'u' or 'v'
    other_family = "v" if down_family == "u" else "u"

    def find_key_by_family(family: str):
        for key in side_n_mark.keys():
            parts = key.split("_")
            # expected: max_u_outward / min_v_outward
            if len(parts) >= 3 and parts[1] == family and parts[-1] == "outward":
                return key
        # fallback
        for key in side_n_mark.keys():
            if f"_{family}_" in key:
                return key
        return None

    down_key = find_key_by_family(down_family)
    other_key = find_key_by_family(other_family)

    if down_key is None or other_key is None:
        raise KeyError(
            f"Cannot select side keys from side_n_mark={list(side_n_mark.keys())}. "
            f"Need one {down_family}-side and one {other_family}-side."
        )

    return down_key, other_key, down_family, other_family


def rotation_from_two_normals(
    down_normal_src: np.ndarray,
    other_normal_src: np.ndarray,
    down_normal_dst: np.ndarray = TARGET_DOWN_NORMAL,
    other_normal_dst: np.ndarray = TARGET_OTHER_NORMAL,
):
    """
    Build rotation R so that:
        R @ down_normal_src  ~= down_normal_dst  (-Z)
        R @ other_normal_src ~= other_normal_dst (-Y)

    This constrains:
        selected side face touches XOY plane;
        the other side face is parallel to XOZ plane and its outward normal points opposite STL Y.
    """
    a = normalize(down_normal_src)
    b_raw = normalize(other_normal_src)

    # make b orthogonal to a for a stable source basis
    b = b_raw - np.dot(b_raw, a) * a
    if np.linalg.norm(b) < 1e-8:
        raise ValueError("down_normal_src and other_normal_src are nearly parallel")
    b = normalize(b)
    c = normalize(np.cross(a, b))

    A = normalize(down_normal_dst)
    B_raw = normalize(other_normal_dst)
    B = B_raw - np.dot(B_raw, A) * A
    if np.linalg.norm(B) < 1e-8:
        raise ValueError("down_normal_dst and other_normal_dst are nearly parallel")
    B = normalize(B)
    C = normalize(np.cross(A, B))

    S = np.column_stack([a, b, c])
    T = np.column_stack([A, B, C])

    R = T @ S.T

    # numerical cleanup
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    return R


def transform_points(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    return (R @ points.T).T + t.reshape(1, 3)

# =========================================================
# Main API
# =========================================================
def orient_stl(
    input_stl_path: str | None = None,
    output_stl_path: str | None = None,
    meta_path: str | None = None,
    completion_dir: str = COMPLETION_DIR,
    visualize: bool = False,
):
    """
    Orient repair model STL for printing.

    Rules:
        1. Read defect_world_y from meta.npz.
        2. defect_world_y decides u-side or v-side faces down on XOY plane.
        3. The other side is parallel to XOZ plane, and its outward normal is aligned to -Y.
        4. repair_point_center is placed at STL XY center: (0, 0).
        5. The lowest mesh point is shifted to Z=0.

    Returns:
        output_stl_path, orientation_info
    """
    meta = load_meta(meta_path=meta_path, completion_dir=completion_dir)

    side_n_mark = meta["side_n_mark"]
    repair_point_center = meta["repair_point_center"]  # * UNIT_SCALE
    defect_world_y = meta["defect_world_y"]

    down_key, other_key, down_family, other_family = select_side_keys(
        side_n_mark,
        defect_world_y,
    )

    down_normal_src = side_n_mark[down_key]
    other_normal_src = side_n_mark[other_key]

    R = rotation_from_two_normals(
        down_normal_src=down_normal_src,
        other_normal_src=other_normal_src,
        down_normal_dst=TARGET_DOWN_NORMAL,
        other_normal_dst=TARGET_OTHER_NORMAL,
    )

    if output_stl_path is None:
        output_stl_path = os.path.join(completion_dir, "model_oriented.stl")

    mesh = o3d.io.read_triangle_mesh(input_stl_path)
    
    if len(mesh.vertices) == 0:
        raise RuntimeError(f"Empty STL mesh: {input_stl_path}")

    vertices = np.asarray(mesh.vertices, dtype=float)
    #vertices = np.asarray(mesh.vertices, dtype=float) * UNIT_SCALE

    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    vertices_rot = (R @ vertices.T).T

    center_rot = R @ repair_point_center

    # XY center: repair_point_center -> (0,0)
    # Z placement: lowest model point touches XOY plane, z=0
    t = np.array([
        -center_rot[0],
        -center_rot[1],
        -vertices_rot[:, 2].min(),
    ])

    vertices_out = vertices_rot + t.reshape(1, 3)
    mesh.vertices = o3d.utility.Vector3dVector(vertices_out)
    mesh.compute_vertex_normals()

    os.makedirs(os.path.dirname(output_stl_path), exist_ok=True)
    ok = o3d.io.write_triangle_mesh(output_stl_path, mesh)
    if not ok:
        raise RuntimeError(f"Failed to write STL: {output_stl_path}")

    # Save oriented point cloud if model.pcd exists
    input_pcd_path = os.path.join(completion_dir, "model.pcd")
    output_pcd_path = os.path.join(completion_dir, "model_oriented.pcd")
    if os.path.exists(input_pcd_path):
        pcd = o3d.io.read_point_cloud(input_pcd_path)
        pts = np.asarray(pcd.points)
        if len(pts) > 0:
            pts_out = transform_points(pts, R, t)
            pcd.points = o3d.utility.Vector3dVector(pts_out)
            o3d.io.write_point_cloud(output_pcd_path, pcd)

    # Optional: orient side pcds for visual check
    for name in ["u_side_plane", "v_side_plane", "top_plane", "top_defect_margin"]:
        in_path = os.path.join(completion_dir, f"{name}.pcd")
        out_path = os.path.join(completion_dir, f"{name}_oriented.pcd")
        if os.path.exists(in_path):
            pcd = o3d.io.read_point_cloud(in_path)
            pts = np.asarray(pcd.points)
            if len(pts) > 0:
                pts_out = transform_points(pts, R, t)
                pcd.points = o3d.utility.Vector3dVector(pts_out)
                o3d.io.write_point_cloud(out_path, pcd)

    # Save transform metadata
    orientation_meta_path = os.path.join(completion_dir, "orientation_meta.npz")
    np.savez(
        orientation_meta_path,
        R=R,
        t=t,
        input_stl_path=input_stl_path,
        output_stl_path=output_stl_path,
        meta_path=meta["meta_path"],
        defect_world_y=defect_world_y,
        down_family=down_family,
        other_family=other_family,
        down_side_key=down_key,
        other_side_key=other_key,
        down_normal_src=down_normal_src,
        other_normal_src=other_normal_src,
        down_normal_after=R @ down_normal_src,
        other_normal_after=R @ other_normal_src,
        repair_point_center_src=repair_point_center,
        repair_point_center_after=R @ repair_point_center + t,
    )

    print("\n========== STL orientation done ==========")
    print("input STL:", input_stl_path)
    print("output STL:", output_stl_path)
    print("meta:", meta["meta_path"])
    print("defect_world_y:", defect_world_y)
    print("down side:", down_key, "-> target -Z / XOY bed")
    print("other side:", other_key, "-> target -Y / parallel to XOZ")
    print("R:\n", R)
    print("t:", t)
    print("down normal after:", R @ down_normal_src)
    print("other normal after:", R @ other_normal_src)
    print("repair center after:", R @ repair_point_center + t)
    print("min z after:", np.asarray(mesh.vertices)[:, 2].min())
    print("orientation meta:", orientation_meta_path)

    if visualize:
        mesh_show = o3d.geometry.TriangleMesh(mesh)
        mesh_show.paint_uniform_color([0.7, 0.7, 0.7])
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.06)
        o3d.visualization.draw_geometries([mesh_show,frame])

    orientation_info = {
        "R": R,
        "t": t,
        "defect_world_y": defect_world_y,
        "down_side_key": down_key,
        "other_side_key": other_key,
        "output_stl_path": output_stl_path,
        "orientation_meta_path": orientation_meta_path,
    }

    return output_stl_path, orientation_info


if __name__ == "__main__":
    model_dir = "E:/HKUSTGZ/AAM/construction/data/completion_result/depression/"
    input_stl_path = model_dir + '/model.stl'
    output_stl_path = model_dir + '/model_oriented.stl'

    orient_stl(
        input_stl_path=input_stl_path,
        output_stl_path=output_stl_path,
        meta_path=None,
        completion_dir=COMPLETION_DIR,
        visualize=True,
    )


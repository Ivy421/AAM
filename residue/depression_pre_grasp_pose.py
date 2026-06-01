import os
import json
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R


# ============================================================
# User config
# ============================================================

COMPLETION_DIR = r"E:\HKUSTGZ\AAM\construction\data\completion_result\depression"

# 已经完成定向后的 STL / PCD
ORIENTED_STL_PATH = os.path.join(COMPLETION_DIR, "model_oriented.stl")
ORIENTED_MODEL_PCD_PATH = os.path.join(COMPLETION_DIR, "model_oriented.pcd")
ORIENTATION_META_PATH = os.path.join(COMPLETION_DIR, "orientation_meta.npz")

# 侧面点云：model_orientation.py 如果同步变换了 side pcd，会生成这两个文件
U_SIDE_ORIENTED_PCD = os.path.join(COMPLETION_DIR, "u_side_plane_oriented.pcd")
V_SIDE_ORIENTED_PCD = os.path.join(COMPLETION_DIR, "v_side_plane_oriented.pcd")

# 输出
OUT_NPZ_PATH = os.path.join(COMPLETION_DIR, "depression_pre_grasp_pose.npz")
OUT_JSON_PATH = os.path.join(COMPLETION_DIR, "depression_pre_grasp_pose.json")

# ------------------------------------------------------------
# 坐标单位
# ------------------------------------------------------------
# 你的点云/mesh 代码里多数坐标是 m，比如 grid_res=0.0008 表示 0.8 mm。
# 但机械臂/打印平台位姿这里按 mm 计算。
# 如果你的 oriented STL 已经是 mm 单位，把 MODEL_UNIT_TO_MM 改成 1.0。
MODEL_UNIT_TO_MM = 1.0

# ------------------------------------------------------------
# 模型在打印平台上的摆放方式
# ------------------------------------------------------------
# "bbox_center_to_plate_center":
#     Bambu Studio 中把模型包围盒中心放到平台中心。
#
# "stl_origin_to_plate_center": STL坐标系原点位于打印平台中心
PLACEMENT_MODE =  "stl_origin_to_plate_center" #"bbox_center_to_plate_center"  

# 打印平台坐标系下的平台中心，单位 mm
PLATE_CENTER_MM = np.array([0.0, 0.0, 0.0], dtype=float)

# 预抓取高度：末端原点位于目标棱边正上方多少 mm
PRE_GRASP_HEIGHT_MM = 200.0

# ------------------------------------------------------------
# 打印平台坐标系 p 到机械臂 base 坐标系 b 的固定变换
# 参考 hole_pre_grasp_pose.py：
#     theta = 90 deg
#     bp_t = [-130, 200, 20] mm
# ------------------------------------------------------------
BP_RZ_DEG = 90.0
BP_T_MM = np.array([-130.0, 200.0, 20.0], dtype=float)

# 从上往下垂直抓取的末端姿态
# 参考 hole_pre_grasp_pose.py: [180, 0, 0]
EE_EULER_XYZ_DEG = np.array([180.0, 0.0, 0.0], dtype=float)

# 用侧面点云计算两侧面交线时的距离阈值
EDGE_TOL_MM = 2.0


# ============================================================
# Basic utils
# ============================================================

def normalize(v):
    v = np.asarray(v, dtype=float).reshape(3)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError(f"zero-length vector: {v}")
    return v / n


def make_T(R_mat, t_vec):
    T = np.eye(4)
    T[:3, :3] = np.asarray(R_mat, dtype=float).reshape(3, 3)
    T[:3, 3] = np.asarray(t_vec, dtype=float).reshape(3)
    return T


def transform_point(T, p):
    p = np.asarray(p, dtype=float).reshape(3)
    ph = np.r_[p, 1.0]
    return (T @ ph)[:3]


def build_bpT():
    """
    p: 打印平台坐标系
    b: 机械臂 base 坐标系

    返回:
        bpT: 4x4, p -> b, 单位 mm
    """
    theta = np.deg2rad(BP_RZ_DEG)

    Rz = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta),  np.cos(theta), 0.0],
        [0.0,            0.0,           1.0],
    ])

    return make_T(Rz, BP_T_MM)


def npz_scalar_to_str(x):
    if isinstance(x, np.ndarray):
        if x.shape == ():
            return str(x.item())
        return str(x.tolist())
    return str(x)


def load_orientation_meta(meta_path=ORIENTATION_META_PATH):
    if not os.path.exists(meta_path):
        print(f"[WARN] orientation_meta not found: {meta_path}")
        return {}

    data = np.load(meta_path, allow_pickle=True)
    meta = {}

    for k in data.files:
        meta[k] = data[k]

    return meta


def read_mesh_vertices(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find oriented STL: {path}")

    mesh = o3d.io.read_triangle_mesh(path)
    vertices = np.asarray(mesh.vertices, dtype=float)

    if len(vertices) == 0:
        raise RuntimeError(f"Empty mesh vertices: {path}")

    return mesh, vertices


def read_pcd_points(path):
    if not os.path.exists(path):
        return None

    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points, dtype=float)

    if len(pts) == 0:
        return None

    return pts


def fit_plane_pca(points):
    """
    PCA 拟合平面:
        normal · x + d = 0
    """
    points = np.asarray(points, dtype=float)

    if len(points) < 3:
        raise ValueError("Not enough points to fit plane")

    centroid = points.mean(axis=0)
    pts = points - centroid
    cov = pts.T @ pts

    eigvals, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, np.argmin(eigvals)]
    normal = normalize(normal)
    d = -np.dot(normal, centroid)

    return centroid, normal, d


def compute_edge_midpoint_from_side_pcds(
    down_family,
    other_family,
    edge_tol_model_unit,
    u_side_path=U_SIDE_ORIENTED_PCD,
    v_side_path=V_SIDE_ORIENTED_PCD,
):
    """
    从 oriented side pcds 计算：
        down side 与 other side 的交线棱边中点

    返回：
        edge_mid_model: 模型坐标/原 STL 坐标单位
        edge_points_model: 棱边附近点
    """
    pcd_path = {
        "u": u_side_path,
        "v": v_side_path,
    }

    down_pts = read_pcd_points(pcd_path[down_family])
    other_pts = read_pcd_points(pcd_path[other_family])

    if down_pts is None or other_pts is None:
        return None, None

    _, other_normal, other_d = fit_plane_pca(other_pts)

    # 在 down side 中，找距离 other side 平面最近的一圈点
    dist = np.abs(down_pts @ other_normal + other_d)
    edge_mask = dist <= edge_tol_model_unit
    edge_points = down_pts[edge_mask]

    if len(edge_points) < 5:
        # fallback: 取最近 5% 点
        k = max(5, int(0.05 * len(down_pts)))
        idx = np.argsort(dist)[:k]
        edge_points = down_pts[idx]

    edge_mid = edge_points.mean(axis=0)

    return edge_mid, edge_points


def get_down_other_family_from_meta(meta):
    """
    从 orientation_meta.npz 中读取：
        down_family, other_family

    如果没有，则使用默认假设：
        down_family = 'u'
        other_family = 'v'
    """
    if "down_family" in meta:
        down_family = npz_scalar_to_str(meta["down_family"]).lower()
    else:
        down_family = "u"

    if "other_family" in meta:
        other_family = npz_scalar_to_str(meta["other_family"]).lower()
    else:
        other_family = "v" if down_family == "u" else "u"

    if down_family not in ["u", "v"]:
        raise ValueError(f"down_family must be u/v, got {down_family}")
    if other_family not in ["u", "v"]:
        raise ValueError(f"other_family must be u/v, got {other_family}")

    return down_family, other_family


def get_target_edge_midpoint_model(mesh_vertices, meta, down_family, other_family):
    """
    优先级：
    1. orientation_meta.npz 中已有 edge_mid_after / edge_mid_after_xy_shift 等字段；
    2. 由 oriented u/v side pcds 计算两侧面交线中点；
    3. fallback: 使用 STL 坐标系原点在最低 Z 处，即 [0,0,min_z]。
    """
    candidate_keys = [
        "edge_mid_after",
        "edge_mid_final",
        "edge_mid_before_xy_shift",
    ]

    for k in candidate_keys:
        if k in meta:
            edge_mid = np.asarray(meta[k], dtype=float).reshape(3)
            print(f"[INFO] edge midpoint loaded from orientation_meta: {k} = {edge_mid}")
            return edge_mid, None, f"orientation_meta:{k}"

    edge_tol_model_unit = EDGE_TOL_MM / MODEL_UNIT_TO_MM
    edge_mid, edge_points = compute_edge_midpoint_from_side_pcds(
        down_family=down_family,
        other_family=other_family,
        edge_tol_model_unit=edge_tol_model_unit,
    )

    if edge_mid is not None:
        print(f"[INFO] edge midpoint computed from side pcds: {edge_mid}")
        return edge_mid, edge_points, "side_pcd_intersection"

    min_z = mesh_vertices[:, 2].min()
    edge_mid = np.array([0.0, 0.0, min_z], dtype=float)

    print(
        "[WARN] Cannot compute edge midpoint from meta or side pcds. "
        f"Fallback to STL origin with min z: {edge_mid}"
    )

    return edge_mid, None, "fallback_stl_origin_min_z"


def model_point_to_plate_mm(
    point_model,
    mesh_vertices,
    placement_mode=PLACEMENT_MODE,
    plate_center_mm=PLATE_CENTER_MM,
):
    """
    把 oriented STL 坐标中的目标点转换到打印平台坐标系 p，单位 mm。

    注意：
        这里只处理 Bambu Studio 摆放平移，不处理旋转；
        模型旋转已经在 model_orientation.py 里完成。
    """
    point_model = np.asarray(point_model, dtype=float).reshape(3)
    vertices_mm = mesh_vertices * MODEL_UNIT_TO_MM
    point_mm = point_model * MODEL_UNIT_TO_MM

    bbox_min = vertices_mm.min(axis=0)
    bbox_max = vertices_mm.max(axis=0)
    bbox_center = 0.5 * (bbox_min + bbox_max)
    min_z = bbox_min[2]

    if placement_mode == "bbox_center_to_plate_center":
        # XY: 模型包围盒中心放到平台中心
        # Z : 模型最低点落在平台 z=0
        p = np.array([
            plate_center_mm[0] + (point_mm[0] - bbox_center[0]),
            plate_center_mm[1] + (point_mm[1] - bbox_center[1]),
            plate_center_mm[2] + (point_mm[2] - min_z),
        ], dtype=float)

    elif placement_mode == "stl_origin_to_plate_center":
        # XY: STL 原点对齐平台中心
        # Z : 模型最低点落在平台 z=0
        p = np.array([
            plate_center_mm[0] + point_mm[0],
            plate_center_mm[1] + point_mm[1],
            plate_center_mm[2] + (point_mm[2] - min_z),
        ], dtype=float)

    else:
        raise ValueError(
            "PLACEMENT_MODE must be 'bbox_center_to_plate_center' "
            "or 'stl_origin_to_plate_center'"
        )

    return p, {
        "bbox_min_mm": bbox_min,
        "bbox_max_mm": bbox_max,
        "bbox_center_mm": bbox_center,
        "model_point_mm": point_mm,
    }


def make_jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {str(k): make_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [make_jsonable(v) for v in x]
    if isinstance(x, (np.float32, np.float64)):
        return float(x)
    if isinstance(x, (np.int32, np.int64)):
        return int(x)
    return x


# ============================================================
# Main
# ============================================================

def compute_depression_pre_grasp_pose(
    oriented_stl_path=ORIENTED_STL_PATH,
    orientation_meta_path=ORIENTATION_META_PATH,
    pre_grasp_height_mm=PRE_GRASP_HEIGHT_MM,
    placement_mode=PLACEMENT_MODE,
    save=True,
):
    """
    计算 depression repair block 的 pre-grasp pose。

    目标：
        1. 末端坐标系原点 XY 对准立起来的棱边中点；
        2. Z 高度 = 棱边点 Z + pre_grasp_height_mm；
        3. 姿态为从上往下垂直抓取，沿用 [180, 0, 0] 欧拉角。
    """

    meta = load_orientation_meta(orientation_meta_path)
    down_family, other_family = get_down_other_family_from_meta(meta)

    mesh, vertices = read_mesh_vertices(oriented_stl_path)

    print("\n========== Loaded oriented model ==========")
    print("oriented STL:", oriented_stl_path)
    print("MODEL_UNIT_TO_MM:", MODEL_UNIT_TO_MM)
    print("PLACEMENT_MODE:", placement_mode)
    print("down_family:", down_family)
    print("other_family:", other_family)
    print("mesh bbox size in model unit:", vertices.max(axis=0) - vertices.min(axis=0))
    print("mesh bbox size in mm:", (vertices.max(axis=0) - vertices.min(axis=0)) * MODEL_UNIT_TO_MM)

    edge_mid_model, edge_points_model, edge_source = get_target_edge_midpoint_model(
        mesh_vertices=vertices,
        meta=meta,
        down_family=down_family,
        other_family=other_family,
    )

    p_edge_mm, placement_info = model_point_to_plate_mm(
        point_model=edge_mid_model,
        mesh_vertices=vertices,
        placement_mode=placement_mode,
        plate_center_mm=PLATE_CENTER_MM,
    )

    p_pre_grasp_mm = p_edge_mm.copy()
    p_pre_grasp_mm[2] += pre_grasp_height_mm

    bpT = build_bpT()

    b_edge_mm = transform_point(bpT, p_edge_mm)
    b_pre_grasp_mm = transform_point(bpT, p_pre_grasp_mm)

    quat_xyzw = R.from_euler("xyz", EE_EULER_XYZ_DEG, degrees=True).as_quat()

    # piper_sdk 常用输入格式: [x, y, z, rx, ry, rz]，单位 mm + deg
    piper_pre_grasp_pose = [
        float(b_pre_grasp_mm[0]),
        float(b_pre_grasp_mm[1]),
        float(b_pre_grasp_mm[2]),
        float(EE_EULER_XYZ_DEG[0]),
        float(EE_EULER_XYZ_DEG[1]),
        float(EE_EULER_XYZ_DEG[2]),
    ]

    # MoveIt 通常使用 m + quaternion
    moveit_pre_grasp_pose = {
        "position_m": (b_pre_grasp_mm / 1000.0),
        "orientation_xyzw": quat_xyzw,
    }

    result = {
        "edge_source": edge_source,
        "placement_mode": placement_mode,
        "down_family": down_family,
        "other_family": other_family,
        "edge_mid_model": edge_mid_model,
        "edge_mid_plate_mm": p_edge_mm,
        "pre_grasp_plate_mm": p_pre_grasp_mm,
        "edge_mid_base_mm": b_edge_mm,
        "pre_grasp_base_mm": b_pre_grasp_mm,
        "piper_pre_grasp_pose": piper_pre_grasp_pose,
        "moveit_position_m": moveit_pre_grasp_pose["position_m"],
        "moveit_orientation_xyzw": moveit_pre_grasp_pose["orientation_xyzw"],
        "bpT": bpT,
        "EE_EULER_XYZ_DEG": EE_EULER_XYZ_DEG,
        "pre_grasp_height_mm": pre_grasp_height_mm,
        "placement_info": placement_info,
    }

    print("\n========== Depression pre-grasp pose ==========")
    print("edge source:", edge_source)
    print("edge midpoint in oriented STL/model unit:", edge_mid_model)
    print("edge midpoint in printer plate frame p, mm:", p_edge_mm)
    print("pre-grasp point in printer plate frame p, mm:", p_pre_grasp_mm)
    print("edge midpoint in robot base frame b, mm:", b_edge_mm)
    print("pre-grasp point in robot base frame b, mm:", b_pre_grasp_mm)
    print("piper pre-grasp pose [x,y,z,rx,ry,rz]:")
    print(piper_pre_grasp_pose)
    print("moveit position m:", moveit_pre_grasp_pose["position_m"])
    print("moveit orientation xyzw:", moveit_pre_grasp_pose["orientation_xyzw"])

    if save:
        os.makedirs(COMPLETION_DIR, exist_ok=True)

        np.savez(
            OUT_NPZ_PATH,
            edge_mid_model=edge_mid_model,
            edge_mid_plate_mm=p_edge_mm,
            pre_grasp_plate_mm=p_pre_grasp_mm,
            edge_mid_base_mm=b_edge_mm,
            pre_grasp_base_mm=b_pre_grasp_mm,
            piper_pre_grasp_pose=np.asarray(piper_pre_grasp_pose, dtype=float),
            moveit_position_m=moveit_pre_grasp_pose["position_m"],
            moveit_orientation_xyzw=moveit_pre_grasp_pose["orientation_xyzw"],
            bpT=bpT,
            down_family=down_family,
            other_family=other_family,
            edge_source=edge_source,
            placement_mode=placement_mode,
            MODEL_UNIT_TO_MM=MODEL_UNIT_TO_MM,
            PRE_GRASP_HEIGHT_MM=pre_grasp_height_mm,
        )

        with open(OUT_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(make_jsonable(result), f, indent=2, ensure_ascii=False)

        print("\nSaved:")
        print(OUT_NPZ_PATH)
        print(OUT_JSON_PATH)

    return result


if __name__ == "__main__":
    compute_depression_pre_grasp_pose(
        oriented_stl_path=ORIENTED_STL_PATH,
        orientation_meta_path=ORIENTATION_META_PATH,
        pre_grasp_height_mm=PRE_GRASP_HEIGHT_MM,
        placement_mode=PLACEMENT_MODE,
        save=False,
    )

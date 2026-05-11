import os, json, sys
sys.path.append('E:/HKUSTGZ/AAM')
import open3d as o3d
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

def find_plane(points_np, voxel_size=0.001, distance_threshold=0.001):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np[:, :3])

    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = pcd.select_by_index(ind)

    plane1_model, inliers1 = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=3,
        num_iterations=3000
    )
    plane1_pcd = pcd.select_by_index(inliers1)
    rest_pcd = pcd.select_by_index(inliers1, invert=True)

    plane2_model, inliers2 = rest_pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=3,
        num_iterations=3000
    )
    plane2_pcd = rest_pcd.select_by_index(inliers2)
    rest_pcd = rest_pcd.select_by_index(inliers2, invert=True)

    return plane1_model, plane1_pcd, plane2_model, plane2_pcd, rest_pcd


def plane_from_pcd(plane_pcd):
    pts = np.asarray(plane_pcd.points)
    c = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - c, full_matrices=False)
    n = vh[-1]
    n = n / np.linalg.norm(n)
    return c, n

def plane_basis_from_pcd(plane_pcd):
    c, n = plane_from_pcd(plane_pcd)

    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    bu = np.cross(n, ref)
    bu = bu / np.linalg.norm(bu)
    bv = np.cross(n, bu)
    bv = bv / np.linalg.norm(bv)

    return c, bu, bv, n

def plane_intersection_line(plane1_pcd, plane2_pcd):
    c1, n1 = plane_from_pcd(plane1_pcd)
    c2, n2 = plane_from_pcd(plane2_pcd)

    d = np.cross(n1, n2)
    d_norm = np.linalg.norm(d)
    if d_norm < 1e-10:
        raise ValueError("两个平面几乎平行，无法稳定求交线")
    d = d / d_norm

    # 解一个交线上点 p0，同时满足：
    # n1·p0 = n1·c1
    # n2·p0 = n2·c2
    # d ·p0 = 0
    A = np.vstack([n1, n2, d])
    b = np.array([n1 @ c1, n2 @ c2, 0.0])
    p0 = np.linalg.solve(A, b)

    return p0, d

def split_points_by_two_planes(pcd, plane1_pcd, plane2_pcd, dist_thresh=0.0005, margin=0.0003):
    pts = np.asarray(pcd.points)

    c1, n1 = plane_from_pcd(plane1_pcd)
    c2, n2 = plane_from_pcd(plane2_pcd)

    d1 = np.abs((pts - c1) @ n1)
    d2 = np.abs((pts - c2) @ n2)

    mask1 = (d1 < dist_thresh) & (d1 + margin < d2)
    mask2 = (d2 < dist_thresh) & (d2 + margin < d1)
    mask_edge = ~(mask1 | mask2)   # 棱边附近/歧义点

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pts[mask1])

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pts[mask2])

    pcd_edge = o3d.geometry.PointCloud()
    pcd_edge.points = o3d.utility.Vector3dVector(pts[mask_edge])

    return pcd1, pcd2, pcd_edge, (c1, n1), (c2, n2)

def project_points_to_plane(pcd_part, plane_pcd):
    pts = np.asarray(pcd_part.points)
    c, u, v, n = plane_basis_from_pcd(plane_pcd)

    rel = pts - c
    du = rel @ u
    dv = rel @ v
    proj_3d = c + np.outer(du, u) + np.outer(dv, v)
    return du, dv, proj_3d
def largest_component(mask, min_pixels=30):
    lab, num = ndimage.label(mask)
    if num == 0:
        return np.zeros_like(mask, dtype=bool)
    areas = ndimage.sum(mask, lab, range(1, num + 1))
    idx = np.argmax(areas) + 1
    out = (lab == idx)
    return out if out.sum() >= min_pixels else np.zeros_like(mask, dtype=bool)

def uv_to_defect_mask(u, v, plane_pcd, other_plane_pcd, grid_res=0.0005, pad=0.002):
    uv = np.stack([u, v], axis=1)

    # -------------------------
    # 1) 建立二维栅格范围
    # -------------------------
    umin = uv[:, 0].min() - pad
    umax = uv[:, 0].max() + pad
    vmin = uv[:, 1].min() - pad
    vmax = uv[:, 1].max() + pad

    nx = int(np.ceil((umax - umin) / grid_res)) + 1
    ny = int(np.ceil((vmax - vmin) / grid_res)) + 1

    # -------------------------
    # 2) 实际占据图
    # -------------------------
    actual_mask = np.zeros((ny, nx), dtype=bool)

    ix = np.floor((u - umin) / grid_res).astype(int)
    iy = np.floor((v - vmin) / grid_res).astype(int)

    ix = np.clip(ix, 0, nx - 1)
    iy = np.clip(iy, 0, ny - 1)

    actual_mask[iy, ix] = True
    #actual_mask = ndimage.binary_dilation(actual_mask, iterations=1)
    actual_mask = ndimage.binary_closing(actual_mask, iterations=1)

    # -------------------------
    # 3) 原始矩形 ideal_mask
    # -------------------------
    u0, u1 = np.percentile(u, [1, 99])
    v0, v1 = np.percentile(v, [1, 99])

    ideal_mask = np.zeros((ny, nx), dtype=bool)

    ix0 = int(np.floor((u0 - umin) / grid_res))
    ix1 = int(np.ceil((u1 - umin) / grid_res))
    iy0 = int(np.floor((v0 - vmin) / grid_res))
    iy1 = int(np.ceil((v1 - vmin) / grid_res))

    ix0 = np.clip(ix0, 0, nx - 1)
    ix1 = np.clip(ix1, 0, nx - 1)
    iy0 = np.clip(iy0, 0, ny - 1)
    iy1 = np.clip(iy1, 0, ny - 1)

    ideal_mask[iy0:iy1 + 1, ix0:ix1 + 1] = True

    # -------------------------
    # 4) 用两平面交线裁剪 ideal_mask
    # -------------------------
    c, bu, bv, n = plane_basis_from_pcd(plane_pcd)
    p0_3d, d_3d = plane_intersection_line(plane_pcd, other_plane_pcd)

    # 交线投影到当前平面的 uv 坐标
    p0_uv = np.array([(p0_3d - c) @ bu, (p0_3d - c) @ bv])
    p1_uv = np.array([(p0_3d + d_3d - c) @ bu, (p0_3d + d_3d - c) @ bv])

    du_line = p1_uv[0] - p0_uv[0]
    dv_line = p1_uv[1] - p0_uv[1]

    # 判断“实际点云”在哪一侧，就保留哪一侧
    # side > 0 / side < 0 表示在线的两边
    side_pts = du_line * (v - p0_uv[1]) - dv_line * (u - p0_uv[0])
    keep_positive = np.sum(side_pts >= 0) >= np.sum(side_pts <= 0)

    # 对整个栅格求半平面mask
    uu = umin + (np.arange(nx) + 0.5) * grid_res
    vv = vmin + (np.arange(ny) + 0.5) * grid_res
    UU, VV = np.meshgrid(uu, vv)

    side_grid = du_line * (VV - p0_uv[1]) - dv_line * (UU - p0_uv[0])

    if keep_positive:
        half_mask = side_grid >= -grid_res
    else:
        half_mask = side_grid <= grid_res

    # 用交线裁掉多余那半边
    ideal_mask = ideal_mask & half_mask

    # -------------------------
    # 5) 缺陷区域 = 理想 - 实际
    # -------------------------
    defect_mask = ideal_mask & (~actual_mask)
    defect_mask = largest_component(defect_mask, min_pixels=30)
    # defect_mask = ndimage.binary_closing(defect_mask, iterations=1)


    info = {
        "umin": umin,
        "vmin": vmin,
        "grid_res": grid_res,
        "actual_mask": actual_mask,
        "ideal_mask": ideal_mask,
        "defect_mask": defect_mask,
        "line_p0_uv": p0_uv,
        "line_p1_uv": p1_uv,
        "half_mask": half_mask,
    }
    return defect_mask, info


def defect_mask_to_3d(defect_mask, info, plane_pcd):
    c, bu, bv, n = plane_basis_from_pcd(plane_pcd)

    ys, xs = np.where(defect_mask)
    if len(xs) == 0:
        return o3d.geometry.PointCloud()

    u = info["umin"] + (xs + 0.5) * info["grid_res"]
    v = info["vmin"] + (ys + 0.5) * info["grid_res"]

    pts3d = c + np.outer(u, bu) + np.outer(v, bv)

    pcd_defect = o3d.geometry.PointCloud()
    pcd_defect.points = o3d.utility.Vector3dVector(pts3d)
    return pcd_defect


##############模型拉伸建设立体 ###############
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

def find_defect_plane(points_np, distance_threshold=1):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np[:, :3])
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=3,
        num_iterations=3000
    )
    plane_pcd = pcd.select_by_index(inliers)
    rest_pcd = pcd.select_by_index(inliers, invert=True)

    return plane_model, plane_pcd, rest_pcd

def n_direction(n, n_ref, d):
    if np.dot(n, n_ref) < 0:
        print('change side of n')
        n = -n
        d = -d
    n - np.asarray(n)
    return n, d
def point_smoothing(pcd,n, d, n_ref, n_otherplane, d_otherplane):
    points = np.asarray(pcd.points)
    filtered_points = [p for p in points if np.dot(n_otherplane, p) + d_otherplane >= 0]
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(np.array(filtered_points))
    pcd = filtered_pcd

    return pcd


####### run code
pcd_raw = o3d.io.read_point_cloud("E:/HKUSTGZ/AAM/construction/data/completion_result/fused2.pcd")
# o3d.visualization.draw_geometries([pcd_raw])
pcd_raw = pcd_raw.voxel_down_sample(voxel_size=0.001)
points_raw= np.asarray(pcd_raw.points)
#draw_points_mat([pcd_raw, pcd_raw],elev = 45, azim = 120)

pcd = get_largest_cluster(pcd_raw)
# draw_points_mat(np.asarray(largest_pcd.points))
plane1_model, plane1_pcd, plane2_model, plane2_pcd, rest_pcd = find_plane(
    np.asarray(pcd.points),
    voxel_size=0.003,
    distance_threshold=0.0015
)

print("Plane 1:", plane1_model)   # ax + by + cz + d = 0
print("Plane 2:", plane2_model)
pcd1, pcd2, pcd_edge, plane1, plane2 = split_points_by_two_planes(
    pcd, plane1_pcd, plane2_pcd,
    dist_thresh=0.005, # 0.005
    margin=0.0005
)

# 点在两个拟合平面的坐标表示
u1, v1, proj1_3d = project_points_to_plane(pcd1, plane1_pcd)
u2, v2, proj2_3d = project_points_to_plane(pcd2, plane2_pcd)

# 返回两个平面的缺陷mask
defect_mask1, info1 = uv_to_defect_mask(
    u1, v1,
    plane1_pcd, plane2_pcd,
    grid_res=0.0006, pad=0.0005)

defect_mask2, info2 = uv_to_defect_mask(
    u2, v2,
    plane2_pcd, plane1_pcd,
    grid_res=0.0006, pad=0.0005)

#把缺陷mask重投影回3D空间
defect_pcd1 = defect_mask_to_3d(defect_mask1, info1, plane1_pcd)
defect_pcd2 = defect_mask_to_3d(defect_mask2, info2, plane2_pcd)
defect_all = o3d.geometry.PointCloud()
defect_all.points = o3d.utility.Vector3dVector(np.vstack([np.asarray(defect_pcd1.points), np.asarray(defect_pcd2.points)]))
#defect_all.paint_uniform_color([0.55, 0.2 , 0.8 ])  # 红色
#pcd_raw.paint_uniform_color([0, 0, 1 ])  # 蓝色
#o3d.visualization.draw_geometries([defect_all, pcd_raw])

##################### 生成参数 #######################
plane1_model = np.asarray(plane1_model, dtype=float)
plane2_model = np.asarray(plane2_model, dtype=float)
plane1_model[3] = plane1_model[3] 
plane2_model[3] = plane2_model[3] 
n1 = plane1_model[:3]
n1 = n1 / np.linalg.norm(n1)
n2 = plane2_model[:3]
n2 = n2 / np.linalg.norm(n2)

### 确定两个平面的法线朝向，必须向内 ###
object_center = ((np.asarray(pcd.points).mean(axis=0))) 
plane1_center = np.asarray(plane1_pcd.points).mean(axis=0)
plane2_center = np.asarray(plane2_pcd.points).mean(axis=0)
defect1_center = np.asarray(defect_pcd1.points).mean(axis=0)
defect2_center = np.asarray(defect_pcd2.points).mean(axis=0)
if np.dot(n1, ( object_center - plane1_center )) < 0:
    print('plane1 法向量改为朝内 ')
    n1 = -n1
if np.dot(n2, ( object_center - plane2_center )) < 0:
    print('plane2 法向量改为朝内 ')
    n2 = -n2


################ 缺陷表面平滑 ####################
defect_plane1_model,_,_ = find_defect_plane(np.asarray(defect_pcd1.points), distance_threshold=1)
defect_plane2_model,_,_ = find_defect_plane(np.asarray(defect_pcd2.points),  distance_threshold=1)
n1_d = defect_plane1_model[:3]
n2_d = defect_plane2_model[:3]
d1_d = defect_plane1_model[-1]
d2_d = defect_plane2_model[-1]

## adjust n direction
n1_d, d1_d = n_direction(n1_d, n1, d1_d)
n2_d, d2_d = n_direction(n2_d, n2, d2_d)
### defect_pcd1 ccacel points
defect_pcd1 = point_smoothing(defect_pcd1,n1_d, d1_d, n1, n2_d, d2_d)
defect_pcd2 = point_smoothing(defect_pcd2, n2_d, d2_d, n2, n1_d, d1_d)

################ 面点云拉伸为体点云 #################
thickness = 0.004     # 4 mm
step = 0.0003         # 每层间距 0.3 mm
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

repair_model_pcd = o3d.geometry.PointCloud()
if len(pts1) + len(pts2) > 0:
    repair_model_pcd.points = o3d.utility.Vector3dVector(
        np.vstack([pts1, pts2])
    )

print("extrude_pcd1 点数:", len(pts1))
print("extrude_pcd2 点数:", len(pts2))
print("repair_block_pcd 总点数:", len(np.asarray(repair_model_pcd.points)))

pcd_raw.paint_uniform_color([0, 0, 1 ])
repair_model_pcd.paint_uniform_color([0.2,0.8,0.33])
o3d.visualization.draw_geometries([repair_model_pcd])
################ 保存参数到本地 #################
o3d.io.write_point_cloud("E:/HKUSTGZ/AAM/construction/data/completion_result/defect_pcd1.pcd", defect_pcd1)
o3d.io.write_point_cloud("E:/HKUSTGZ/AAM/construction/data/completion_result/defect_pcd2.pcd", defect_pcd2)
o3d.io.write_point_cloud("E:/HKUSTGZ/AAM/construction/data/completion_result/plane1_pcd.pcd", plane1_pcd)
o3d.io.write_point_cloud("E:/HKUSTGZ/AAM/construction/data/completion_result/plane2_pcd.pcd", plane2_pcd)
o3d.io.write_point_cloud("E:/HKUSTGZ/AAM/construction/data/completion_result/repair_model_pcd.pcd", repair_model_pcd)

np.savez(
    "E:/HKUSTGZ/AAM/construction/data/completion_result/planes_meta.npz",
    plane1_model=plane1_model,
    plane2_model=plane2_model,
    n1=n1,
    n2=n2,
    plane1_center = plane1_center,
    plane2_center = plane2_center,

    object_center=object_center,
    
    defect1_center = defect1_center,
    defect2_center = defect2_center

)
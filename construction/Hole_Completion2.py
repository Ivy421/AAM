import os, json, sys
sys.path.append('E:/HKUSTGZ/AAM')
import open3d as o3d
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

def get_largest_cluster(pcd, eps=0.02, min_points=10):
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    max_label = labels.max()
    
    if max_label < 0:
        print("未找到有效的连通域（全是噪声）")
        return None

    largest_cluster_id = -1
    max_count = 0

    for i in range(max_label + 1):
        count = np.sum(labels == i)
        if count > max_count:
            max_count = count
            largest_cluster_id = i
            
    print(f"找到 {max_label + 1} 个簇，最大簇包含 {max_count} 个点")

    indices = np.where(labels == largest_cluster_id)[0]
    return pcd.select_by_index(indices)

def find_plane(points_np, voxel_size=0.001, distance_threshold=0.001):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np[:, :3])

    #pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

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
    world_x = np.array([1,0,0])
    world_y = np.array([0,1,0])
    c, n = plane_from_pcd(plane_pcd)
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u_temp = np.cross(n, ref)
    u_temp = u_temp / np.linalg.norm(u_temp)

    # 改变u_axis方向使u_xais指向机械臂
    if np.abs(np.dot(u_temp, world_x)) < np.abs(np.dot(u_temp, world_y)):
        u_axis = np.cross(n, u_temp)
        u_axis = u_axis / np.linalg.norm(u_axis)
    else:
        u_axis = u_temp

    if np.dot(u_axis, world_x) >0:
        u_axis = -u_axis
    v_axis = np.cross(n, u_axis)
    v_axis = v_axis / np.linalg.norm(v_axis)

    return c, u_axis, v_axis, n

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
def uv_to_defect_mask(
    u, v,
    plane_pcd, other_plane_pcd,
    grid_res=0.0005,
    pad=0.002,
    depth_percentile=95,          # 当前面“远离交线”的理论边界，鲁棒分位数
    tangent_percentiles=(1, 99),  # 沿交线方向的范围
    mask_pad_pix=1.5              # 给 domain_mask 留一点余量
):
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
    # 2) 实际占据图 actual_mask
    # -------------------------
    actual_mask = np.zeros((ny, nx), dtype=bool)

    ix = np.floor((u - umin) / grid_res).astype(int)
    iy = np.floor((v - vmin) / grid_res).astype(int)

    ix = np.clip(ix, 0, nx - 1)
    iy = np.clip(iy, 0, ny - 1)

    actual_mask[iy, ix] = True
    actual_mask = ndimage.binary_closing(actual_mask, iterations=1)

    # -------------------------
    # 3) 计算两平面理论交线，并投影到当前 UV 平面
    # -------------------------
    c, bu, bv, n = plane_basis_from_pcd(plane_pcd)
    p0_3d, d_3d = plane_intersection_line(plane_pcd, other_plane_pcd)

    p0_uv = np.array([(p0_3d - c) @ bu, (p0_3d - c) @ bv])
    p1_uv = np.array([(p0_3d + d_3d - c) @ bu, (p0_3d + d_3d - c) @ bv])

    t = p1_uv - p0_uv
    t_norm = np.linalg.norm(t)
    if t_norm < 1e-12:
        raise ValueError("交线方向长度过小")
    t = t / t_norm                     # 交线方向（面内）
    q = np.array([-t[1], t[0]])        # 面内、垂直于交线的方向

    # -------------------------
    # 4) 用交线构造 intersection_half_mask
    #    不再用矩形 ideal_mask
    # -------------------------
    signed_depth = (uv - p0_uv) @ q

    # 当前面的点主要落在哪一侧，就保留哪一侧
    keep_positive = np.sum(signed_depth >= 0) >= np.sum(signed_depth <= 0)
    inward_sign = 1.0 if keep_positive else -1.0

    # 统一成：当前面内部方向的“深度”
    inward_depth = inward_sign * signed_depth

    # -------------------------
    # 5) 用鲁棒分位数生成 boundary_half_mask
    #    表示“这个面离交线多远为止”
    # -------------------------
    valid_depth = inward_depth[inward_depth >= 0]
    if len(valid_depth) == 0:
        depth_max = 0.0
    else:
        depth_max = np.percentile(valid_depth, depth_percentile)

    # -------------------------
    # 6) 沿交线方向做有限裁剪 tangent_mask
    #    防止 domain_mask 无限延伸
    # -------------------------
    tang_coord = (uv - p0_uv) @ t
    t0, t1 = np.percentile(tang_coord, tangent_percentiles)

    # -------------------------
    # 7) 在整张栅格上计算三个 mask
    # -------------------------
    uu = umin + (np.arange(nx) + 0.5) * grid_res
    vv = vmin + (np.arange(ny) + 0.5) * grid_res
    UU, VV = np.meshgrid(uu, vv)

    dU = UU - p0_uv[0]
    dV = VV - p0_uv[1]

    inward_grid = inward_sign * (dU * q[0] + dV * q[1])
    tang_grid   = dU * t[0] + dV * t[1]

    intersection_half_mask = inward_grid >= -mask_pad_pix * grid_res
    boundary_half_mask     = inward_grid <= depth_max + mask_pad_pix * grid_res
    tangent_mask           = (
        (tang_grid >= t0 - mask_pad_pix * grid_res) &
        (tang_grid <= t1 + mask_pad_pix * grid_res)
    )

    # 这就是新的“理论完整区域”
    domain_mask = intersection_half_mask & boundary_half_mask & tangent_mask

    # -------------------------
    # 8) 缺陷区域 = 理论完整区域 - 实际占据区域
    # -------------------------
    defect_mask_before_lcc = domain_mask & (~actual_mask)
    defect_mask = largest_component(defect_mask_before_lcc, min_pixels=30)

    info = {
        "umin": umin,
        "vmin": vmin,
        "grid_res": grid_res,

        "actual_mask": actual_mask,

        # 新规则
        "domain_mask": domain_mask,
        "intersection_half_mask": intersection_half_mask,
        "boundary_half_mask": boundary_half_mask,
        "tangent_mask": tangent_mask,

        # 为兼容你原来调试代码，保留这个键
        "ideal_mask": domain_mask,

        "defect_mask_before_lcc": defect_mask_before_lcc,
        "defect_mask": defect_mask,

        "line_p0_uv": p0_uv,
        "line_p1_uv": p1_uv,
        "keep_positive": keep_positive,
        "depth_max": depth_max,
        "t0": t0,
        "t1": t1,
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
#o3d.visualization.draw_geometries([pcd_raw])
pcd_raw = pcd_raw.voxel_down_sample(voxel_size=0.001)
points_raw= np.asarray(pcd_raw.points)
#draw_points_mat([pcd_raw, pcd_raw],elev = 45, azim = 120)

pcd = get_largest_cluster(pcd_raw)
# draw_points_mat(np.asarray(largest_pcd.points))
plane1_model, plane1_pcd, plane2_model, plane2_pcd, rest_pcd = find_plane(
    np.asarray(pcd.points),
    voxel_size=0.003,
    distance_threshold=0.003
)

plane1_pcd.paint_uniform_color([0,0,1])
plane2_pcd.paint_uniform_color([0,1,0])
o3d.visualization.draw_geometries([plane1_pcd,plane2_pcd])

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
    pcd1, pcd2,
    grid_res=0.0006, pad=0.0005)

defect_mask2, info2 = uv_to_defect_mask(
    u2, v2,
    pcd1, pcd2,  # plane1_pcd, plan2_pcd
    grid_res=0.0006, pad=0.0005)
print(defect_mask1)

uv1 = np.stack([u1, v1], axis=1)

print(uv1.shape)
print(defect_mask1.shape)

#for defect_mask in [defect_mask1,defect_mask2]:
#    
#    y_false1, x_false1 = np.where(~defect_mask)
#    y_true1,  x_true1  = np.where(defect_mask)
#    plt.figure(figsize=(6, 6))
#    plt.scatter(x_false1, y_false1, s=0.5, c='gray', label='False')
#    plt.scatter(x_true1,  y_true1,  s=0.5, c='red',  label='True')
#    #for p in uv1:
#    #    plt.scatter(p[0],p[1], s = 0.5,c = 'yellow')
#    plt.gca().invert_yaxis()
#    plt.axis('equal')
#    plt.xlabel('x / column')
#    plt.ylabel('y / row')
#    plt.legend()
#    plt.tight_layout()
#    plt.show()


#把缺陷mask重投影回3D空间
defect_pcd1 = defect_mask_to_3d(defect_mask1, info1, plane1_pcd)
defect_pcd2 = defect_mask_to_3d(defect_mask2, info2, plane2_pcd)
defect_all = o3d.geometry.PointCloud()
defect_all.points = o3d.utility.Vector3dVector(np.vstack([np.asarray(defect_pcd1.points), np.asarray(defect_pcd2.points)]))
defect_all.paint_uniform_color([0.55, 0.2 , 0.8 ])  
pcd_raw.paint_uniform_color([0, 0, 1 ])  
#o3d.visualization.draw_geometries([defect_all, pcd_raw])
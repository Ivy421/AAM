import os, json, sys
sys.path.append('E:/HKUSTGZ/AAM')
import open3d as o3d
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
def draw_points_mat(points_list, title = 'point cloud visualize', elev = 45, azim = 120 ):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    color = ['red','orange','yellow','green','blue','black','purple']
    for i in range (len(points_list)):
        ax.scatter(points_list[i][:, 0], points_list[i][:, 1], points_list[i][:, 2], s=1, c = color[i])  # s=1 for small points
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title(title)
        ax.view_init(elev, azim)
    plt.show()    

    return

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

def find_plane(points_np, voxel_size=0.001, distance_threshold=0.0015):
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

def split_points_by_two_planes(pcd, plane1_pcd, plane2_pcd, dist_thresh=0.002, margin=0.0005):
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


def uv_to_defect_mask(u, v, plane_pcd, other_plane_pcd, grid_res=0.001, pad=0.002):
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
    dist_thresh=0.002,
    margin=0.0005
)

# 点在两个拟合平面的坐标表示
u1, v1, proj1_3d = project_points_to_plane(pcd1, plane1_pcd)
u2, v2, proj2_3d = project_points_to_plane(pcd2, plane2_pcd)

# 返回两个平面的缺陷mask
defect_mask1, info1 = uv_to_defect_mask(
    u1, v1,
    plane1_pcd, plane2_pcd,
    grid_res=0.001, pad=0.001)

defect_mask2, info2 = uv_to_defect_mask(
    u2, v2,
    plane2_pcd, plane1_pcd,
    grid_res=0.001, pad=0.001)

#把缺陷mask重投影回3D空间
defect_pcd1 = defect_mask_to_3d(defect_mask1, info1, plane1_pcd)
defect_pcd2 = defect_mask_to_3d(defect_mask2, info2, plane2_pcd)

# 合并两个平面的补全点云
pts1 = np.asarray(defect_pcd1.points)
pts2 = np.asarray(defect_pcd2.points)

defect_all = o3d.geometry.PointCloud()
if len(pts1) + len(pts2) > 0:
    defect_all.points = o3d.utility.Vector3dVector(np.vstack([pts1, pts2]))

print("plane1 缺陷点数:", len(pts1))
print("plane2 缺陷点数:", len(pts2))
print("总缺陷点数:", len(np.asarray(defect_all.points)))

#visualize
# defect_all.paint_uniform_color([1, 0.5 , 0.5 ])  # 红色
# pcd_raw.paint_uniform_color([0, 0, 1 ])  # 蓝色
# o3d.visualization.draw_geometries([defect_all, pcd_raw])

################ 保存参数到本地 #################
plane1_model = np.asarray(plane1_model, dtype=float)
plane2_model = np.asarray(plane2_model, dtype=float)
n1 = plane1_model[:3]
n1 = n1 / np.linalg.norm(n1)
n2 = plane2_model[:3]
n2 = n2 / np.linalg.norm(n2)
object_center = np.asarray(pcd.points).mean(axis=0)
o3d.io.write_point_cloud("E:/HKUSTGZ/AAM/construction/data/completion_result/defect_pcd1.pcd", defect_pcd1)
o3d.io.write_point_cloud("E:/HKUSTGZ/AAM/construction/data/completion_result/defect_pcd2.pcd", defect_pcd2)
o3d.io.write_point_cloud("E:/HKUSTGZ/AAM/construction/data/completion_result/plane1_pcd.pcd", plane1_pcd)
o3d.io.write_point_cloud("E:/HKUSTGZ/AAM/construction/data/completion_result/plane2_pcd.pcd", plane2_pcd)
np.savez(
    "E:/HKUSTGZ/AAM/construction/data/completion_result/planes_meta.npz",
    plane1_model=plane1_model,
    plane2_model=plane2_model,
    n1=n1,
    n2=n2,
    object_center=object_center
)



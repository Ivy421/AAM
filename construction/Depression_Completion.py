import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy import ndimage
# ============================================================
# 用边界直线拟合代替 uv.min/max 定义矩形边界
# ============================================================

def extract_side_points(uv, side, bin_size=0.003):
    """
    从二维点云中提取某一侧的边界候选点。

    side:
        "u_min": 每个 v 分箱中取最小 u
        "u_max": 每个 v 分箱中取最大 u
        "v_min": 每个 u 分箱中取最小 v
        "v_max": 每个 u 分箱中取最大 v
    """
    pts = []

    if side in ["u_min", "u_max"]:
        coord = uv[:, 1]  # 按 v 分箱
        bins = np.arange(coord.min(), coord.max(), bin_size)

        for i in range(len(bins) - 1):
            mask = (coord >= bins[i]) & (coord < bins[i + 1])
            bin_pts = uv[mask]

            if len(bin_pts) < 5:
                continue

            if side == "u_min":
                pts.append(bin_pts[np.argmin(bin_pts[:, 0])])
            else:
                pts.append(bin_pts[np.argmax(bin_pts[:, 0])])

    elif side in ["v_min", "v_max"]:
        coord = uv[:, 0]  # 按 u 分箱
        bins = np.arange(coord.min(), coord.max(), bin_size)

        for i in range(len(bins) - 1):
            mask = (coord >= bins[i]) & (coord < bins[i + 1])
            bin_pts = uv[mask]

            if len(bin_pts) < 5:
                continue

            if side == "v_min":
                pts.append(bin_pts[np.argmin(bin_pts[:, 1])])
            else:
                pts.append(bin_pts[np.argmax(bin_pts[:, 1])])

    return np.asarray(pts)


def fit_line_svd(points):
    """
    最小二乘拟合二维直线。
    输出直线形式：
        a*u + b*v + c = 0
    """
    center = points.mean(axis=0)
    pts = points - center

    _, _, vh = np.linalg.svd(pts)
    direction = vh[0]

    # 直线法向量
    normal = np.array([-direction[1], direction[0]])
    normal = normal / (np.linalg.norm(normal) + 1e-12)

    c = -np.dot(normal, center)

    return np.array([normal[0], normal[1], c])


def line_to_boundary_value(line, side, uv, sample_num=200):
    """
    将拟合直线转换成一个代表性的边界坐标值。

    对 u_min/u_max：
        在 v 范围内采样，计算对应 u，取中位数。

    对 v_min/v_max：
        在 u 范围内采样，计算对应 v，取中位数。
    """
    a, b, c = line

    u0, v0 = np.percentile(uv, 5, axis=0)
    u1, v1 = np.percentile(uv, 95, axis=0)

    if side in ["u_min", "u_max"]:
        vs = np.linspace(v0, v1, sample_num)

        # a*u + b*v + c = 0  =>  u = -(b*v+c)/a
        us = -(b * vs + c) / (a + 1e-12)

        return np.median(us)

    elif side in ["v_min", "v_max"]:
        us = np.linspace(u0, u1, sample_num)

        # a*u + b*v + c = 0  =>  v = -(a*u+c)/b
        vs = -(a * us + c) / (b + 1e-12)

        return np.median(vs)
    

def find_plane(pcd, voxel_size=0.001, distance_threshold=0.005):
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    plane1_model, inliers1 = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=3,
        num_iterations=3000
    )
    plane1_pcd = pcd.select_by_index(inliers1)
    rest_pcd = pcd.select_by_index(inliers1, invert=True)

    return plane1_model, plane1_pcd, rest_pcd

def normalize(v):
    return v / (np.linalg.norm(v) + 1e-12)

## u 是长边，v是短边
def build_plane_basis(points, normal):
    """
    基于顶面点云建立局部坐标系：
    origin: 顶面中心
    u_axis, v_axis: 顶面内两个正交方向
    n_axis: 顶面法向
    """
    n_axis = normalize(normal)
    origin = points.mean(axis=0)

    # 去掉法向分量，只保留平面内分量
    pts = points - origin
    pts_plane = pts - np.outer(pts @ n_axis, n_axis)

    # PCA 找顶面内主方向
    cov = np.cov(pts_plane.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    
    u_axis = eigvecs[:, np.argmax(eigvals)]
    u_axis = normalize(u_axis - np.dot(u_axis, n_axis) * n_axis)

    v_axis = normalize(np.cross(n_axis, u_axis))

    return origin, u_axis, v_axis, n_axis


def project_to_uv(points, origin, u_axis, v_axis):
    """
    3D 点投影到顶面二维坐标 uv
    """
    vec = points - origin
    u = vec @ u_axis
    v = vec @ v_axis
    return np.column_stack([u, v])


def uv_to_3d(uv, origin, u_axis, v_axis):
    """
    uv 反投影回顶面 3D 点
    """
    return origin + uv[:, 0:1] * u_axis + uv[:, 1:2] * v_axis


def build_corner_roi(H, W, mode, frac_u=0.35, frac_v=0.35):
    """
    只保留某个角附近的缺陷区域
    """
    roi = np.zeros((H, W), dtype=bool)

    h = int(H * frac_v)
    w = int(W * frac_u)

    if mode == "min_u_min_v":
        roi[:h, :w] = True

    elif mode == "max_u_min_v":
        roi[:h, W - w:] = True

    elif mode == "min_u_max_v":
        roi[H - h:, :w] = True

    elif mode == "max_u_max_v":
        roi[H - h:, W - w:] = True

    else:
        raise ValueError("CORNER_MODE 设置错误")

    return roi

# ============================================================
# flood fill 填充
# ============================================================

def project_to_uvz(points, origin, u_axis, v_axis, z_axis):
    """
    把 3D 点转换到局部坐标：
    u, v: 顶面二维坐标
    z: 沿厚度方向的距离
    """
    vec = points - origin
    u = vec @ u_axis
    v = vec @ v_axis
    z = vec @ z_axis
    return np.column_stack([u, v, z])


def uvz_to_3d(uv, z, origin, u_axis, v_axis, z_axis):
    """
    把局部 uvz 点反变换回 3D
    """
    uv = np.asarray(uv)
    z = np.asarray(z).reshape(-1, 1)

    pts = (
        origin
        + uv[:, 0:1] * u_axis
        + uv[:, 1:2] * v_axis
        + z * z_axis
    )
    return pts


def uv_to_grid_index(uv, u_min, v_min, grid_res, W, H):
    """
    uv 坐标转栅格索引
    """
    xs = ((uv[:, 0] - u_min) / grid_res).astype(int)
    ys = ((uv[:, 1] - v_min) / grid_res).astype(int)

    valid = (
        (xs >= 0) & (xs < W) &
        (ys >= 0) & (ys < H)
    )

    return xs, ys, valid


def get_seed_cell(corner_mode, H, W, domain_mask):
    """
    根据缺陷角位置，选择 flood fill 的起始种子点。
    如果角点本身不可用，就找最近的可用点。
    """
    if corner_mode == "min_u_min_v":
        target = np.array([0, 0])          # row, col
    elif corner_mode == "max_u_min_v":
        target = np.array([0, W - 1])
    elif corner_mode == "min_u_max_v":
        target = np.array([H - 1, 0])
    elif corner_mode == "max_u_max_v":
        target = np.array([H - 1, W - 1])
    else:
        raise ValueError("CORNER_MODE 设置错误")

    if domain_mask[target[0], target[1]]:
        return tuple(target)

    ys, xs = np.where(domain_mask)
    coords = np.column_stack([ys, xs])

    if len(coords) == 0:
        raise RuntimeError("domain_mask 为空，无法选择 seed")

    d = np.linalg.norm(coords - target[None, :], axis=1)
    seed = coords[np.argmin(d)]
    return tuple(seed)


def flood_fill_from_corner(barrier_mask, domain_mask, seed):
    """
    从理想缺角位置开始 flood fill。
    barrier_mask 是缺陷凹陷面的截面阻挡边界。
    domain_mask 是允许搜索的区域。
    """
    allowed = domain_mask & (~barrier_mask)

    if not allowed[seed]:
        # 如果 seed 被 barrier 挡住，找最近的 allowed 点
        ys, xs = np.where(allowed)
        if len(xs) == 0:
            return np.zeros_like(domain_mask, dtype=bool)

        coords = np.column_stack([ys, xs])
        d = np.linalg.norm(coords - np.array(seed)[None, :], axis=1)
        seed = tuple(coords[np.argmin(d)])

    seed_mask = np.zeros_like(domain_mask, dtype=bool)
    seed_mask[seed] = True

    # binary_propagation 就是带 mask 约束的区域生长
    filled = ndimage.binary_propagation(seed_mask, mask=allowed)

    return filled


def mask_to_uv_points(mask, u_min, v_min, grid_res):
    """
    mask 中为 True 的格点转成 uv 坐标。
    使用栅格中心点。
    """
    ys, xs = np.where(mask)

    u = u_min + (xs + 0.5) * grid_res
    v = v_min + (ys + 0.5) * grid_res

    return np.column_stack([u, v])


# 点云单位如果是 m，则 0.001 = 1 mm
grid_res = 0.0008        # 2D 栅格分辨率
pad = 0            # 理想矩形外扩边界

# 只补缺陷角附近，避免把普通扫描空洞也补上
# 可选："min_u_min_v", "max_u_min_v", "min_u_max_v", "max_u_max_v"
CORNER_MODE = "max_u_min_v"

# 缺陷角 ROI 大小比例，越大补得范围越大
roi_frac_u = 0.15
roi_frac_v = 0.35

# 对已有顶面占据图做膨胀，避免点云稀疏导致误判为空洞
occ_dilate_iter = 2

# 对 defect mask 做平滑
defect_close_iter = 2
pcd_raw = o3d.io.read_point_cloud("E:/HKUSTGZ/AAM/construction/data/frame_result/target.pcd")
points_raw = np.asarray(pcd_raw.points)
plane1, plane1_pcd,  rest_pcd = find_plane(pcd_raw, voxel_size=0.001, distance_threshold=0.001)
n = plane1[:3]
inward_vec = np.array([0,0,1])
if np.dot(n, inward_vec) >0:
    print('change side of n')
    n = -n
top_points = np.asarray(plane1_pcd.points)


origin, u_axis, v_axis, n_axis = build_plane_basis(top_points, n)
uv = project_to_uv(top_points, origin, u_axis, v_axis)

edge_bin_size = 0.0005  # 可根据点云密度调整

pts_u_min = extract_side_points(uv, "u_min", bin_size=edge_bin_size)
pts_u_max = extract_side_points(uv, "u_max", bin_size=edge_bin_size)
pts_v_min = extract_side_points(uv, "v_min", bin_size=edge_bin_size)
pts_v_max = extract_side_points(uv, "v_max", bin_size=edge_bin_size)

for p in pts_u_max:
    plt.scatter(p[0], p[1], color = 'b',s = 0.5)
for t in pts_v_min:
    plt.scatter(t[0], t[1], color = 'r',s = 0.5)
#plt.show()

u_min, v_min = uv.min(axis=0) - pad
u_max, v_max = uv.max(axis=0) + pad

u_grid = np.arange(u_min, u_max + grid_res, grid_res)
v_grid = np.arange(v_min, v_max + grid_res, grid_res)

W = len(u_grid)
H = len(v_grid)

# 理想矩形区域：整个二维 bounding box
ideal_mask = np.ones((H, W), dtype=bool)

# ============================================================
# 4. 生成当前顶面占据 mask
# ============================================================

# uv 坐标转成栅格 index
u_idx = ((uv[:, 0] - u_min) / grid_res).astype(int)
v_idx = ((uv[:, 1] - v_min) / grid_res).astype(int)

valid = (
    (u_idx >= 0) & (u_idx < W) &
    (v_idx >= 0) & (v_idx < H)
)

u_idx = u_idx[valid]
v_idx = v_idx[valid]

top_occ_mask = np.zeros((H, W), dtype=bool)
top_occ_mask[v_idx, u_idx] = True

# 膨胀已有点云占据区域，避免稀疏点云产生大量假空洞
top_occ_mask = ndimage.binary_dilation(
    top_occ_mask,
    iterations=occ_dilate_iter
)

# 可选：闭运算，让已有顶面区域更连续
top_occ_mask = ndimage.binary_closing(
    top_occ_mask,
    iterations=1
)

corner_roi = build_corner_roi(
    H, W,
    CORNER_MODE,
    frac_u=roi_frac_u,
    frac_v=roi_frac_v
)

# 理想矩形中没有被当前顶面点云占据的位置，就是候选缺陷区域
defect_mask = ideal_mask & (~top_occ_mask)

# 只保留缺陷角附近
defect_mask = defect_mask & corner_roi

# 平滑 defect mask
defect_mask = ndimage.binary_closing(
    defect_mask,
    iterations=defect_close_iter
)

# 只保留最大连通区域，去掉零散噪点
label_mask, num = ndimage.label(defect_mask)

if num > 0:
    areas = ndimage.sum(defect_mask, label_mask, index=np.arange(1, num + 1))
    largest_label = np.argmax(areas) + 1
    defect_mask = label_mask == largest_label
else:
    print("警告：没有检测到 defect mask，请检查 CORNER_MODE 或 roi_frac")


ys, xs = np.where(defect_mask)

# 栅格中心点对应的 uv 坐标
repair_u = u_min + (xs+0.5) * grid_res
repair_v = v_min + (ys+0.5) * grid_res

repair_uv = np.column_stack([repair_u, repair_v])
repair_top_points = uv_to_3d(repair_uv, origin, u_axis, v_axis)

repair_top_pcd = o3d.geometry.PointCloud()
repair_top_pcd.points = o3d.utility.Vector3dVector(repair_top_points)
repair_top_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # 红色：补全顶面点

plane1_pcd.paint_uniform_color([0.6, 0.6, 0.6])  # 灰色：已有顶面



out_dir = "E:/HKUSTGZ/AAM/construction/data/frame_result"

# ----------------------------
# 参数
# ----------------------------
layer_step = 0.001          # 每层厚度间隔，单位 m，3 mm
band_width = 0.0007          # 每层取点厚度范围，建议略小于/接近 layer_step
thres_points_num = 50       # 每层缺陷点少于该值，则认为该层点不足
max_bad_layers = 3          # 连续 3 层失败则停止

barrier_dilate_iter = 4     # 把缺陷截面点膨胀成阻挡边界
barrier_close_iter = 2      # 连接断裂的 barrier

max_area_ratio = 0.85       # flood fill 面积过大，认为 barrier 无效
min_area_pixels = 20        # repair mask 太小也认为无效

max_search_depth = 0.08     # 最大搜索深度，保险限制，单位 m；不知道厚度时防止死循环


# ============================================================
# 自动判断厚度方向
# ============================================================

rest_points = np.asarray(rest_pcd.points)

# 先用当前 n_axis 试投影
rest_uv_tmp = project_to_uv(rest_points, origin, u_axis, v_axis)
rest_z_tmp = (rest_points - origin) @ n_axis

xs_tmp, ys_tmp, valid_tmp = uv_to_grid_index(
    rest_uv_tmp, u_min, v_min, grid_res, W, H
)

roi_valid = valid_tmp & corner_roi[ys_tmp.clip(0, H - 1), xs_tmp.clip(0, W - 1)]

# 希望缺陷凹陷面在厚度方向 z > 0
# 如果 ROI 内大部分 rest 点的 z 是负数，就翻转厚度方向
if roi_valid.sum() > 20:
    med_z = np.median(rest_z_tmp[roi_valid])
    if med_z < 0:
        thickness_axis = -n_axis
    else:
        thickness_axis = n_axis
else:
    print("警告：ROI 内 rest 点较少，暂时使用 n_axis 作为厚度方向")
    thickness_axis = n_axis

print("thickness_axis:", thickness_axis)


# ============================================================
# 3. 提取缺陷凹陷面候选点
# ============================================================

rest_local = project_to_uvz(
    rest_points,
    origin,
    u_axis,
    v_axis,
    thickness_axis
)

rest_uv = rest_local[:, :2]
rest_z = rest_local[:, 2]

xs, ys, valid_grid = uv_to_grid_index(
    rest_uv, u_min, v_min, grid_res, W, H
)

# 只取：
# 1. 在二维理想矩形范围内
# 2. 在缺陷角 ROI 内
# 3. 位于顶面以下，也就是 z >= 0 附近
valid_roi = np.zeros(len(rest_points), dtype=bool)
valid_roi[valid_grid] = corner_roi[ys[valid_grid], xs[valid_grid]]

defect_surface_mask = (
    valid_grid &
    valid_roi &
    (rest_z > -0.002)    # 允许少量噪声在顶面上方
)

defect_surface_points = rest_points[defect_surface_mask]
defect_surface_uv = rest_uv[defect_surface_mask]
defect_surface_z = rest_z[defect_surface_mask]

print("缺陷凹陷面候选点数量:", len(defect_surface_points))

defect_surface_pcd = o3d.geometry.PointCloud()
defect_surface_pcd.points = o3d.utility.Vector3dVector(defect_surface_points)
defect_surface_pcd.paint_uniform_color([0.0, 0.0, 1.0])  # 蓝色：缺陷凹陷面候选点


# ============================================================
# 4. 按厚度方向切片，逐层生成 repair mask
# ============================================================

domain_mask = ideal_mask & corner_roi
domain_area = domain_mask.sum()

seed = get_seed_cell(CORNER_MODE, H, W, domain_mask)

# 第 0 层直接用你前面得到的顶面 defect_mask
repair_layers = []
repair_layers.append({
    "z": 0.0,
    "mask": defect_mask.copy(),
    "type": "top_defect_mask",
    "point_num": -1,
    "area_ratio": defect_mask.sum() / (domain_area + 1e-12),
})

bad_barrier_count = 0
low_points_count = 0

debug_records = []

z = layer_step

while z <= max_search_depth:

    # ----------------------------
    # 4.1 当前层取缺陷凹陷面截面点
    # ----------------------------
    slice_mask = np.abs(defect_surface_z - z) <= band_width
    slice_uv = defect_surface_uv[slice_mask]
    slice_num = len(slice_uv)

    # 判断当前层缺陷点是否过少
    low_points = slice_num < thres_points_num

    if low_points:
        low_points_count += 1
    else:
        low_points_count = 0

    # ----------------------------
    # 4.2 把当前层截面点转成 barrier mask
    # ----------------------------
    valid_barrier = False
    repair_mask_z = None
    area_ratio = np.nan

    if not low_points:
        sx, sy, svalid = uv_to_grid_index(
            slice_uv, u_min, v_min, grid_res, W, H
        )

        barrier_mask = np.zeros((H, W), dtype=bool)
        barrier_mask[sy[svalid], sx[svalid]] = True

        # 只保留 ROI 内的 barrier
        barrier_mask = barrier_mask & domain_mask

        # 把稀疏截面点膨胀/闭运算，形成连续阻挡边界
        barrier_mask = ndimage.binary_dilation(
            barrier_mask,
            iterations=barrier_dilate_iter
        )

        barrier_mask = ndimage.binary_closing(
            barrier_mask,
            iterations=barrier_close_iter
        )

        barrier_mask = barrier_mask & domain_mask

        # ----------------------------
        # 4.3 从缺陷角开始 flood fill
        # ----------------------------
        repair_mask_z = flood_fill_from_corner(
            barrier_mask,
            domain_mask,
            seed
        )

        repair_area = repair_mask_z.sum()
        area_ratio = repair_area / (domain_area + 1e-12)

        # ----------------------------
        # 4.4 判断 barrier 是否有效
        # ----------------------------
        valid_barrier = (
            (repair_area >= min_area_pixels) and
            (area_ratio < max_area_ratio)
        )

    # ----------------------------
    # 4.5 根据有效性更新连续失败计数
    # ----------------------------
    if valid_barrier:
        bad_barrier_count = 0

        repair_layers.append({
            "z": z,
            "mask": repair_mask_z.copy(),
            "type": "slice_floodfill",
            "point_num": slice_num,
            "area_ratio": area_ratio,
        })

        print(f"[有效] z={z:.4f} m, slice points={slice_num}, area ratio={area_ratio:.3f}")

    else:
        bad_barrier_count += 1

        print(f"[无效] z={z:.4f} m, slice points={slice_num}, "
              f"low_points={low_points}, area ratio={area_ratio}")

    debug_records.append({
        "z": z,
        "slice_num": slice_num,
        "low_points": low_points,
        "valid_barrier": valid_barrier,
        "area_ratio": area_ratio,
        "bad_barrier_count": bad_barrier_count,
        "low_points_count": low_points_count,
    })

    # ----------------------------
    # 4.6 停止条件
    # ----------------------------
    if bad_barrier_count >= max_bad_layers:
        print("\n停止：连续 3 层无法形成有效阻挡")
        break

    if low_points_count >= max_bad_layers:
        print("\n停止：连续 3 层缺陷点数量少于 thres_points_num")
        break

    z += layer_step


# ============================================================
# 5. 把所有有效层的 repair mask 转成 3D 点云
# ============================================================

all_layer_points = []

for layer in repair_layers:
    z_layer = layer["z"]
    mask_layer = layer["mask"]

    uv_layer = mask_to_uv_points(mask_layer, u_min, v_min, grid_res)

    if len(uv_layer) == 0:
        continue

    z_arr = np.full(len(uv_layer), z_layer)

    pts_3d = uvz_to_3d(
        uv_layer,
        z_arr,
        origin,
        u_axis,
        v_axis,
        thickness_axis
    )

    all_layer_points.append(pts_3d)

repair_volume_points = np.vstack(all_layer_points)

repair_model_pcd = o3d.geometry.PointCloud()
repair_model_pcd.points = o3d.utility.Vector3dVector(repair_volume_points)
repair_model_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # 红色：分层补全体点云

print("\n有效层数:", len(repair_layers))
print("补全体点云数量:", len(repair_volume_points))
print("最大有效深度:", max([layer["z"] for layer in repair_layers]), "m")

# ============================================================
# 7. 可视化
# ============================================================

pcd_raw.paint_uniform_color([0.6, 0.6, 0.6])        # 灰色：顶面
#defect_surface_pcd.paint_uniform_color([0.0, 0.0, 1.0]) # 蓝色：缺陷凹陷面
repair_model_pcd.paint_uniform_color([1.0, 0.0, 0.0]) # 红色：分层补全点云

o3d.visualization.draw_geometries([
    #pcd_raw,
    #defect_surface_pcd,
    repair_model_pcd
])

o3d.io.write_point_cloud("E:/HKUSTGZ/AAM/construction/data/completion_result/depression_repair_model.pcd", repair_model_pcd)

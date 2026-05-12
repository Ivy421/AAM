import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.cluster import DBSCAN
from scipy.signal import savgol_filter
from scipy.signal import find_peaks, savgol_filter
def find_plane(pcd, voxel_size=0.001, distance_threshold=0.003):
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

def filter_side_points_by_local_median(points, side, window=5, th=0.003):
    """
    边界点连续性滤波：
    用当前点与前 window 个已保留点的局部中位数比较。
    
    对 u_min / u_max 边：
        边界主要沿 v 方向延伸，所以检查 u 方向跳变。
    
    对 v_min / v_max 边：
        边界主要沿 u 方向延伸，所以检查 v 方向跳变。
    
    points: (N, 2)
    side: "u_min", "u_max", "v_min", "v_max"
    window: 使用前几个已保留点做局部中位数
    th: 垂直方向跳变阈值，单位 m
    """
    #points = np.asarray(points)

    if len(points) <= window:
        return points

    # 确保边界点顺序正确
    if side in ["u_min", "u_max"]:
        # u 边界沿 v 方向走
        #points = points[np.argsort(points[:, 1])]
        check_dim = 0   # 检查 u 方向跳变
    elif side in ["v_min", "v_max"]:
        # v 边界沿 u 方向走
        #points = points[np.argsort(points[:, 0])]
        check_dim = 1   # 检查 v 方向跳变
    else:
        raise ValueError("side 必须是 u_min/u_max/v_min/v_max")

    kept = []

    for p in points:
        if len(kept) < window:
            kept.append(p)
            continue

        ref = np.median(np.asarray(kept[-window:])[:, check_dim])
        dist = abs(p[check_dim] - ref)

        # 垂直跳变太大，认为是抖动点，删除
        if dist <= th:
            kept.append(p)

    return np.asarray(kept)

def smooth_line(points, window=15, polyorder=2):
    """
    对排序后的点进行 Savitzky-Golay 平滑
    """
    n = len(points)

    if n < 5:
        return points

    # window 必须是奇数，并且不能超过点数
    window = min(window, n if n % 2 == 1 else n - 1)

    if window <= polyorder:
        window = polyorder + 3
        if window % 2 == 0:
            window += 1

    window = min(window, n if n % 2 == 1 else n - 1)

    u_smooth = savgol_filter(points[:, 0], window, polyorder)
    v_smooth = savgol_filter(points[:, 1], window, polyorder)

    return np.column_stack([u_smooth, v_smooth])

def local_pca_tangent(points, win=21):
    """
    步骤4：滑动窗口 PCA 拟合局部切线方向
    points: (N, 2)，已经排序和平滑后的曲线点
    win: 局部窗口大小，必须为奇数
    """
    if win % 2 == 0:
        win += 1

    n = len(points)
    half = win // 2
    tangents = np.zeros_like(points)

    for i in range(n):
        i0 = max(0, i - half)
        i1 = min(n, i + half + 1)

        local_pts = points[i0:i1]
        local_pts = local_pts - local_pts.mean(axis=0)

        cov = local_pts.T @ local_pts
        eigvals, eigvecs = np.linalg.eigh(cov)

        # 最大特征值对应主方向
        t = eigvecs[:, np.argmax(eigvals)]

        # 保证切线方向连续，避免 180° 翻转
        if i > 0 and np.dot(t, tangents[i - 1]) < 0:
            t = -t

        tangents[i] = t

    theta = np.arctan2(tangents[:, 1], tangents[:, 0])
    theta = np.unwrap(theta)

    return tangents, theta

def compute_angle_change(points, theta, step=15, smooth_win=21):
    """
    步骤5：计算切线角变化
    用前后 step 个点的方向差，避免局部小波动干扰
    """
    n = len(points)
    angle_change = np.full(n, np.nan)

    for i in range(step, n - step):
        dtheta = theta[i + step] - theta[i - step]
        ds = np.linalg.norm(points[i + step] - points[i - step])

        if ds > 1e-12:
            angle_change[i] = abs(dtheta) / ds

    score = np.nan_to_num(angle_change, nan=0.0)

    # 再平滑一次曲率分数
    if smooth_win % 2 == 0:
        smooth_win += 1

    if smooth_win < len(score):
        score = savgol_filter(score, smooth_win, polyorder=2)

    return score

def find_turn_points(points, score,
                     distance=25,
                     prominence_ratio=0.25,
                     exclude_ratio=0.05):
    """
    步骤6：根据曲率/角度变化峰值找转折点
    """
    n = len(points)

    # 去掉两端，避免端点被误判为转折
    start = int(n * exclude_ratio)
    end = int(n * (1 - exclude_ratio))

    score_valid = score.copy()
    score_valid[:start] = 0
    score_valid[end:] = 0

    prominence = prominence_ratio * np.max(score_valid)

    peaks, props = find_peaks(
        score_valid,
        distance=distance,
        prominence=prominence
    )

    if len(peaks) == 0:
        return None, peaks

    # 默认选曲率分数最大的峰
    best_idx = peaks[np.argmax(score_valid[peaks])]

    return best_idx, peaks

def detect_turn(points,
                pca_win=31,
                step=20,
                score_smooth_win=31,
                distance=30,
                prominence_ratio=0.25):
    tangents, theta = local_pca_tangent(points, win=pca_win)

    score = compute_angle_change(
        points,
        theta,
        step=step,
        smooth_win=score_smooth_win
    )

    best_idx, peaks = find_turn_points(
        points,
        score,
        distance=distance,
        prominence_ratio=prominence_ratio
    )

    return best_idx, peaks, score, theta, tangents

def keep_main_cluster(points, eps=0.003, min_samples=20):
    """
    用 DBSCAN 去掉离群点，只保留最大主簇
    points: (N, 2)
    """
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)

    valid_labels = labels[labels != -1]
    if len(valid_labels) == 0:
        return points

    main_label = np.bincount(valid_labels).argmax()
    return points[labels == main_label]


def compute_corner_roi_frac_from_turns(
    uv,
    u_line,
    v_line,
    turn_idx_u,
    turn_idx_v,
    corner_mode,
    margin_ratio=1.10,
    min_frac=0.03,
    max_frac=0.80
):
    """
    根据两个转折点自动计算 corner_roi 的 frac_u 和 frac_v。

    注意：
    u_line 是 u 方向边界线，主要沿 v 方向变化，所以它决定 frac_v
    v_line 是 v 方向边界线，主要沿 u 方向变化，所以它决定 frac_u
    """

    u_min_all, v_min_all = uv.min(axis=0)
    u_max_all, v_max_all = uv.max(axis=0)

    total_u = u_max_all - u_min_all
    total_v = v_max_all - v_min_all

    # 转折点坐标
    turn_u_on_v_line = v_line[turn_idx_v, 0]  # 用来算 u 方向缺陷长度
    turn_v_on_u_line = u_line[turn_idx_u, 1]  # 用来算 v 方向缺陷长度

    if corner_mode == "max_u_min_v":
        len_u_defect = u_max_all - turn_u_on_v_line
        len_v_defect = turn_v_on_u_line - v_min_all

    elif corner_mode == "min_u_min_v":
        len_u_defect = turn_u_on_v_line - u_min_all
        len_v_defect = turn_v_on_u_line - v_min_all

    elif corner_mode == "max_u_max_v":
        len_u_defect = u_max_all - turn_u_on_v_line
        len_v_defect = v_max_all - turn_v_on_u_line

    elif corner_mode == "min_u_max_v":
        len_u_defect = turn_u_on_v_line - u_min_all
        len_v_defect = v_max_all - turn_v_on_u_line

    else:
        raise ValueError("CORNER_MODE 设置错误")

    frac_u = len_u_defect / total_u
    frac_v = len_v_defect / total_v

    # 加一点余量，避免 ROI 正好卡在转折点
    frac_u *= margin_ratio
    frac_v *= margin_ratio

    # 防止异常值太小或太大
    frac_u = np.clip(frac_u, min_frac, max_frac)
    frac_v = np.clip(frac_v, min_frac, max_frac)

    return frac_u, frac_v

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
    
    #物体坐标系U轴（x轴）朝前指向机械臂
    world_x = np.array([1,0,0])
    if np.dot(world_x,u_axis) > 0:
        print('change side of u')
        u_axis = - u_axis
    v_axis = normalize(np.cross(n_axis, u_axis))

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

#################################################################################
#################################################################################

# 点云单位如果是 m，则 0.001 = 1 mm
grid_res = 0.0008        # 2D 栅格分辨率
pad = 0            # 理想矩形外扩边界

# 只补缺陷角附近，避免把普通扫描空洞也补上
# 可选："min_u_min_v", "max_u_min_v", "min_u_max_v", "max_u_max_v"
CORNER_MODE = "max_u_min_v"
# 默认的ROI占据边长的比例
ROI_restriction = 0.15  # 物理约束 缺陷长宽不超过10cm
roi_frac_u = 0.5
roi_frac_v = 0.5

# 对已有顶面占据图做膨胀，避免点云稀疏导致误判为空洞
occ_dilate_iter = 2

# 对 defect mask 做平滑
defect_close_iter = 2
pcd_raw = o3d.io.read_point_cloud("E:/HKUSTGZ/AAM/construction/data/frame_result/depression_target.pcd")
points_raw = np.asarray(pcd_raw.points)
plane1, plane1_pcd,  _ = find_plane(pcd_raw, voxel_size=0.001, distance_threshold=0.003)
_, _,  rest_pcd = find_plane(pcd_raw, voxel_size=0.001, distance_threshold=0.001)
n = plane1[:3]
inward_vec = np.array([0,0,1])
if np.dot(n, inward_vec) >0:
    print('change side of n')
    n = -n
top_points = np.asarray(plane1_pcd.points)

o3d.visualization.draw_geometries([plane1_pcd])

origin, u_axis, v_axis, n_axis = build_plane_basis(top_points, n)
uv = project_to_uv(top_points, origin, u_axis, v_axis)

edge_bin_size = 0.0005  # 可根据点云密度调整

pts_u_min = extract_side_points(uv, "u_min", bin_size=edge_bin_size)
pts_u_max = extract_side_points(uv, "u_max", bin_size=edge_bin_size)
pts_v_min = extract_side_points(uv, "v_min", bin_size=edge_bin_size)
pts_v_max = extract_side_points(uv, "v_max", bin_size=edge_bin_size)

# 边界点处理
pts_u_max = pts_u_max[10:-10]
pts_u_max = pts_u_max[::-1]
pts_u_min = pts_u_min[10:-10]
pts_u_min = pts_u_min[::-1]
pts_v_min = pts_v_min [10:-10]
pts_v_max = pts_v_max [10:-10]

boundary_jump_th = 0.006   # 3 mm，可调 0.002~0.004
median_window = 20

pts_u_min = filter_side_points_by_local_median(pts_u_min, "u_min", window=median_window, th=boundary_jump_th)
pts_u_max = filter_side_points_by_local_median(pts_u_max, "u_max", window=median_window, th=boundary_jump_th)
pts_v_min = filter_side_points_by_local_median(pts_v_min, "v_min", window=median_window, th=boundary_jump_th)
pts_v_max = filter_side_points_by_local_median(pts_v_max, "v_max", window=median_window, th=boundary_jump_th)

plt.figure()
for p in pts_v_min:
    plt.scatter(p[0],p[1],s = 0.5, c = 'yellow')
for p in pts_v_max:
    plt.scatter(p[0],p[1],s = 0.5, c = 'grey')   
for p in pts_u_min:
    plt.scatter(p[0],p[1],s = 0.5, c = 'blue')
for p in pts_u_max:
    plt.scatter(p[0],p[1],s = 0.5, c = 'red') 
plt.show()

if CORNER_MODE == 'max_u_min_v':
    len_v_phy = max(pts_v_min[:,0]) - min(pts_v_min[:,0] )
    len_u_phy = max(pts_u_max[:,1]) - min(pts_u_max[:,1] )
    fac_v = min(round(ROI_restriction / len_v_phy,2),1)
    fac_u = min (round( ROI_restriction / len_u_phy,2),1)
    len_v = pts_v_min.shape[0]
    len_u = pts_u_max.shape[0]
    frac_v = int(len_v * fac_v)
    frac_u = int(len_u * fac_u)

    u_line = pts_u_max[:frac_u]
    v_line = pts_v_min[-frac_v:]

    u_line_clean = keep_main_cluster(u_line,eps=0.002,min_samples=15)
    u_line = smooth_line(u_line_clean,window=41,polyorder=2)
    v_line_clean = keep_main_cluster(v_line,eps=0.002,min_samples=15)
    v_line = smooth_line(v_line_clean,window=41,polyorder=2)
    # u line
    turn_idx_u, peaks_u, score_u, theta_u, tangents_u = detect_turn(
        u_line,pca_win=31,step=21,score_smooth_win=31,distance=30,prominence_ratio=0.1)

    # v line
    turn_idx_v, peaks_v, score_v, theta_v, tangents_v = detect_turn(
        v_line,pca_win=21,step=15,score_smooth_win=31,distance=30,prominence_ratio=0.15)

    defect_frac_v = v_line[turn_idx_v]
    defect_frac_u = u_line[turn_idx_u]

    plt.figure()
    for p in u_line:
        plt.scatter(p[0], p[1],s = 0.5,c = 'r')
    plt.scatter(defect_frac_u[0], defect_frac_u[1],s = 20, c = 'r')

    for p in v_line:
        plt.scatter(p[0], p[1],s = 0.5,c = 'b')
    plt.scatter(defect_frac_v[0],defect_frac_v[1],s = 20, c = 'b')
    plt.show()

    roi_frac_u, roi_frac_v = compute_corner_roi_frac_from_turns(
        uv=uv,
        u_line=u_line,
        v_line=v_line,
        turn_idx_u=turn_idx_u,
        turn_idx_v=turn_idx_v,
        corner_mode=CORNER_MODE,
        margin_ratio=1.10
    )
    print(f'缺陷边界转折比例// 短边，长边：{roi_frac_u, roi_frac_v}')

u_min, v_min = uv.min(axis=0) - pad
u_max, v_max = uv.max(axis=0) + pad

u_max_boundary_mean = u_line[turn_idx_u:, 0].mean()
v_min_boundary_mean = v_line[:turn_idx_v, 1].mean()

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
max_bad_layers = 5          # 连续 3 层失败则停止

barrier_dilate_iter = 4     # 把缺陷截面点膨胀成阻挡边界
barrier_close_iter = 2      # 连接断裂的 barrier

max_area_ratio = 0.6       # flood fill 面积过大，认为 barrier 无效
min_area_pixels = 20        # repair mask 太小也认为无效

max_search_depth = 0.08     # 最大搜索深度，保险限制，单位 m；不知道厚度时防止死循环

# ============================================================
# 2. 提取缺陷凹陷面候选点
# ============================================================
rest_points = np.asarray(rest_pcd.points)
thickness_axis = n_axis
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
    (rest_z > -0.003)    # 允许少量噪声在顶面上方
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
    pcd_raw,
    #defect_surface_pcd,
    repair_model_pcd
])

#o3d.io.write_point_cloud("E:/HKUSTGZ/AAM/construction/data/completion_result/depression_repair_model.pcd", repair_model_pcd)


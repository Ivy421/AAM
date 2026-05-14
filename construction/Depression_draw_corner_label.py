import os, sys
sys.path.append('E:/HKUSTGZ/AAM')
import json
import numpy as np
import open3d as o3d
import cv2
from AI_models.LLM_funcitons  import *

# ============================================================
# 路径配置
# ============================================================

ROOT = os.getcwd()
FRAME_DIR = os.path.join(ROOT, "construction/data/frame_result")

# 你要标注的 RGB 图
# 建议先手动指定为和点云对应的那一帧 RGB
IMG_SEQ = os.path.join(FRAME_DIR, "png_sequence.json")

# 融合后的目标点云
PCD_PATH = os.path.join(FRAME_DIR, "depression_target.pcd")

CAMERA_CONFIG_PATH  = ('E:/HKUSTGZ/AAM/config/calibration/right_camera/camera_config.npy')

TRANSFORMATION_PATH = os.path.join(FRAME_DIR, "frame_point_result.npy")

# 输出标注图
OUT_PATH = os.path.join(FRAME_DIR, "rgb_with_uv_corner_labels.png")


# ============================================================
# 基础函数：和你当前代码保持同名/同逻辑
# ============================================================

def normalize(v):
    return v / (np.linalg.norm(v) + 1e-12)


def find_plane(pcd, voxel_size=0.001, distance_threshold=0.003):
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    plane_model, inliers = pcd_down.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=3,
        num_iterations=3000
    )
    plane_pcd = pcd_down.select_by_index(inliers)
    rest_pcd = pcd_down.select_by_index(inliers, invert=True)
    return plane_model, plane_pcd, rest_pcd


def build_plane_basis(points, normal):
    """
    基于顶面点云建立局部坐标系：
    origin: 顶面中心
    u_axis, v_axis: 顶面内两个正交方向
    n_axis: 顶面法向
    """
    n_axis = normalize(normal)
    origin = points.mean(axis=0)

    pts = points - origin
    pts_plane = pts - np.outer(pts @ n_axis, n_axis)

    cov = np.cov(pts_plane.T)
    eigvals, eigvecs = np.linalg.eigh(cov)

    u_axis = eigvecs[:, np.argmax(eigvals)]
    u_axis = normalize(u_axis - np.dot(u_axis, n_axis) * n_axis)

    # 和你原代码保持一致：控制 u_axis 方向
    world_x = np.array([1, 0, 0])
    if np.dot(world_x, u_axis) > 0:
        print("change side of u")
        u_axis = -u_axis

    v_axis = normalize(np.cross(n_axis, u_axis))
    return origin, u_axis, v_axis, n_axis


def project_to_uv(points, origin, u_axis, v_axis):
    vec = points - origin
    u = vec @ u_axis
    v = vec @ v_axis
    return np.column_stack([u, v])


def uv_to_3d(uv, origin, u_axis, v_axis):
    uv = np.asarray(uv)
    return origin + uv[:, 0:1] * u_axis + uv[:, 1:2] * v_axis


# ============================================================
# 3D 点投影到 RGB 图像
# ============================================================

def transform_points(points, T):
    """
    points: (N, 3)
    T: 4x4
    """
    points = np.asarray(points)
    points_h = np.column_stack([points, np.ones(len(points))])
    points_t = (T @ points_h.T).T
    return points_t[:, :3]


def project_points_to_image(points_3d, K, cbT=None):
    """
    points_3d:
        如果 cbT=None，则认为点云已经在 RGB 相机坐标系下。
        如果点云在世界/机械臂坐标系，则需要传入 cbT
    """
    pts = np.asarray(points_3d)

    if cbT is not None:
        pts_cam = transform_points(pts, cbT)
    else:
        pts_cam = pts

    z = pts_cam[:, 2]
    valid = z > 1e-6

    pixels = np.full((len(pts_cam), 2), np.nan)

    x = pts_cam[valid, 0]
    y = pts_cam[valid, 1]
    z_valid = pts_cam[valid, 2]

    fx = K['fx']
    fy = K['fy']
    cx = K['ppx']
    cy = K['ppy']

    pixels[valid, 0] = fx * x / z_valid + cx
    pixels[valid, 1] = fy * y / z_valid + cy

    return pixels, valid


# ============================================================
# 绘图辅助
# ============================================================

def draw_label(img, text, pt, color=(0, 255, 255)):
    x, y = int(pt[0]), int(pt[1])

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 1

    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)

    cv2.rectangle(
        img,
        (x - 4, y - th - 8),
        (x + tw + 4, y + 4),
        (255, 255, 255),
        -1
    )

    cv2.putText(
        img,
        text,
        (x, y),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )

def draw_arrow(img, p0, p1, color, text):
    p0 = tuple(np.round(p0).astype(int))
    p1 = tuple(np.round(p1).astype(int))

    cv2.arrowedLine(
        img,
        p0,
        p1,
        color,
        thickness=2,
        tipLength=0.25
    )

    cv2.putText(
        img,
        text,
        (p1[0] + 5, p1[1] - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA
    )

# ============================================================
# AI推理corner mode
# ============================================================

def build_defect_corner_prompt(
    source_image_path: str,
    labeled_image_path: str,
    object_name: str,
):
    text_prompt = (
        f"These two images show the same defective item, '{object_name}', "
        "taken from the same position. "
        "Figure 1 is the source image without manually added label. "
        "Figure 2 labels the four corners with text codes. "
        "Identify the text code corresponding to the defective corner in Figure 2. "
        "Answer with the text code only."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": source_image_path,
                },
                {
                    "type": "image",
                    "image": labeled_image_path,
                },
                {
                    "type": "text",
                    "text": text_prompt,
                },
            ],
        }
    ]

    return messages
# ============================================================
# 主流程
# ============================================================

def main():
    with open(IMG_SEQ,'r', encoding='utf-8') as f:
        png_seq = json.load(f)
    png = png_seq[0]   
    RGB_PATH = FRAME_DIR+ '/depression_defect/'+png +'.png'

    img = cv2.imread(RGB_PATH)
    if img is None:
        raise FileNotFoundError(f"无法读取 RGB 图像: {RGB_PATH}")

    # 为了让文字更清晰，先放大图像
    VIS_SCALE = 1.2
    img = cv2.resize(img, None, fx=VIS_SCALE, fy=VIS_SCALE, interpolation=cv2.INTER_CUBIC)

    # 为了防止边缘文字被截断，给图像四周加白边
    BORDER = 120
    img = cv2.copyMakeBorder(
        img,
        BORDER, BORDER, BORDER, BORDER,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255)
    )

    H_img, W_img = img.shape[:2]

    pcd_raw = o3d.io.read_point_cloud(PCD_PATH)
    points_raw = np.asarray(pcd_raw.points)

    if len(points_raw) == 0:
        raise RuntimeError("点云为空")

    cam_data = np.load(CAMERA_CONFIG_PATH, allow_pickle=True).item()
    K = cam_data['color_intrinsic']
    frame_data = np.load(TRANSFORMATION_PATH,allow_pickle = True).item()
    bcT = frame_data['bcT_collection'][0]
    cbT = np.linalg.inv(bcT)

    plane1, plane1_pcd, _ = find_plane(
        pcd_raw,
        voxel_size=0.001,
        distance_threshold=0.003
    )

    n = plane1[:3].astype(float)

    # 和你现有代码保持一致：让 n 朝木板内部
    inward_vec = np.array([0, 0, 1])
    if np.dot(n, inward_vec) > 0:
        print("change side of n")
        n = -n

    top_points = np.asarray(plane1_pcd.points)

    origin, u_axis, v_axis, n_axis = build_plane_basis(top_points, n)
    uv = project_to_uv(top_points, origin, u_axis, v_axis)

    u_min, v_min = uv.min(axis=0)
    u_max, v_max = uv.max(axis=0)

    print("u range:", u_min, u_max)
    print("v range:", v_min, v_max)

    corner_uv = {
        "min_u_min_v": np.array([u_min, v_min]),
        "max_u_min_v": np.array([u_max, v_min]),
        "min_u_max_v": np.array([u_min, v_max]),
        "max_u_max_v": np.array([u_max, v_max]),
    }

    corner_names = list(corner_uv.keys())
    corner_uv_arr = np.array([corner_uv[k] for k in corner_names])

    corner_3d = uv_to_3d(corner_uv_arr,origin, u_axis,v_axis)

    corner_pixels, corner_valid = project_points_to_image( corner_3d,K,cbT=cbT)
    corner_pixels = corner_pixels * VIS_SCALE + BORDER
    
    uv_center = np.array([(u_min + u_max) / 2,(v_min + v_max) / 2])

    du = u_max - u_min
    dv = v_max - v_min
    arrow_len = 0.18 * min(du, dv)

    arrow_uv = np.array([
        uv_center,
        uv_center + np.array([arrow_len, 0.0]),
        uv_center + np.array([0.0, arrow_len])
    ])

    arrow_3d = uv_to_3d(arrow_uv,origin,u_axis,v_axis)

    arrow_pixels, arrow_valid = project_points_to_image(arrow_3d,K,cbT=cbT)
    arrow_pixels = arrow_pixels * VIS_SCALE + BORDER

    # ----------------------------
    # 6. 在 RGB 图上绘制四角轮廓
    # ----------------------------
    # 角点顺序：左下 -> 右下 -> 右上 -> 左上
    polygon_order = [
        "min_u_min_v",
        "max_u_min_v",
        "max_u_max_v",
        "min_u_max_v"
    ]

    polygon_pts = []
    for name in polygon_order:
        idx = corner_names.index(name)
        if corner_valid[idx]:
            p = corner_pixels[idx]
            polygon_pts.append([int(round(p[0])), int(round(p[1]))])

    if len(polygon_pts) == 4:
        polygon_pts_np = np.array(polygon_pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img,[polygon_pts_np],isClosed=True,color=(0, 255, 255),thickness=2)
    else:
        print("警告：部分角点投影无效，未绘制完整四边形")

    # ----------------------------
    # 7. 标注四个角
    # ----------------------------
    label_colors = {
        "min_u_min_v": (255, 0, 0),
        "max_u_min_v": (0, 0, 255),
        "min_u_max_v": (0, 180, 255),
        "max_u_max_v": (0, 100, 0),
    }

    for i, name in enumerate(corner_names):
        if not corner_valid[i]:
            print(f"角点 {name} 投影无效")
            continue

        p = corner_pixels[i]

        if not (0 <= p[0] < W_img and 0 <= p[1] < H_img):
            print(f"角点 {name} 投影在图像外: {p}")

        draw_label(img,name,p,color=label_colors[name])

        cv2.circle(img,tuple(np.round(p).astype(int)),5,label_colors[name],-1)

    # ----------------------------
    # 8. 画 +u / +v 箭头
    # ----------------------------
    if np.all(arrow_valid):
        draw_arrow(
            img,
            arrow_pixels[0],
            arrow_pixels[1],
            color=(0, 0, 255),   # 红色 +u
            text="+u"
        )

        draw_arrow(
            img,
            arrow_pixels[0],
            arrow_pixels[2],
            color=(255, 0, 0),   # 蓝色 +v
            text="+v"
        )
    else:
        print("警告：u/v 箭头投影无效")

    # ----------------------------
    # 9. 保存结果
    # ----------------------------
    cv2.imwrite(OUT_PATH, img)
    print("Saved annotated RGB:", OUT_PATH)

    cv2.imshow("annotated_rgb", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    ######################### inference ######################### 
    messages = build_defect_corner_prompt(RGB_PATH,OUT_PATH,'a black object' )
    corner_mode = qwen3_inference(messages)
    print(corner_mode, type(corner_mode))


if __name__ == "__main__":
    main()
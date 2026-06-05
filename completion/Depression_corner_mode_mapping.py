import os, sys
sys.path.append('/public/home/rastus/AAM')
import json
import re
import numpy as np
import open3d as o3d
import cv2
from AI_models.LLM_funcitons  import *

# ============================================================
# 路径配置
# ============================================================

ROOT = os.getcwd()
FRAME_DIR = os.path.join(ROOT, "data/frame_result")

# 你要标注的 RGB 图
# 建议先手动指定为和点云对应的那一帧 RGB
IMG_SEQ = os.path.join(FRAME_DIR, "png_sequence.json")

# 融合后的目标点云
PCD_PATH = os.path.join(FRAME_DIR, "depression_target.pcd")

CAMERA_CONFIG_PATH  = ('/public/home/rastus/AAM/config/calibration/right_camera/camera_config.npy')

TRANSFORMATION_PATH = os.path.join(FRAME_DIR, "frame_point_result.npy")

# 输出标注图
OUT_PATH = os.path.join(FRAME_DIR, "rgb_with_uv_corner_labels.png")
MAPPING_PATH = os.path.join(FRAME_DIR, "corner_label_mapping.json")
INFERENCE_RESULT_PATH = os.path.join(FRAME_DIR, "corner_inference_result.json")

# 固定用短标签给 VLM 看，代码再映射回真实 corner_mode
# 按你之前约定：A/B/C/D -> 四个 uv corner mode
LETTER_TO_CORNER = {
    "A": "min_u_min_v",
    "B": "min_u_max_v",
    "C": "max_u_min_v",
    "D": "max_u_max_v",
}
CORNER_TO_LETTER = {v: k for k, v in LETTER_TO_CORNER.items()}


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

def clamp_int(v, lo, hi):
    return int(max(lo, min(hi, v)))


def draw_label(img, text, pt, color=(0, 0, 0), bg_color=(255, 255, 255)):
    """
    大号白底标签。保留这个函数名，避免影响其他代码调用。
    """
    x, y = int(pt[0]), int(pt[1])

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.4
    thickness = 3
    pad = 10

    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    H, W = img.shape[:2]

    x1 = clamp_int(x - pad, 0, W - 1)
    y1 = clamp_int(y - th - pad - baseline, 0, H - 1)
    x2 = clamp_int(x + tw + pad, 0, W - 1)
    y2 = clamp_int(y + pad, 0, H - 1)

    cv2.rectangle(img, (x1, y1), (x2, y2), bg_color, -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_corner_letter_label(img, letter, corner_pt, object_center_pt, color):
    """
    在角点附近画大号 [A]/[B]/[C]/[D]，并用箭头指向真实角点。
    这样比直接写 min_u_min_v 这类长字符串更容易被 VLM/OCR 识别。
    """
    H, W = img.shape[:2]
    corner_pt = np.asarray(corner_pt, dtype=float)
    object_center_pt = np.asarray(object_center_pt, dtype=float)

    direction = corner_pt - object_center_pt
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        direction = np.array([1.0, -1.0])
    else:
        direction = direction / norm

    # 标签放在角点外侧，避免盖住角点；已有 BORDER，所以一般不会出界
    label_center = corner_pt + direction * 70.0

    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"{letter}"
    scale = 1.8
    thickness = 4
    pad = 14

    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

    # 将 label_center 转为文本左下角坐标
    tx = int(round(label_center[0] - tw / 2))
    ty = int(round(label_center[1] + th / 2))

    # 防止文字被裁剪
    tx = clamp_int(tx, pad, W - tw - pad)
    ty = clamp_int(ty, th + pad, H - baseline - pad)

    x1 = clamp_int(tx - pad, 0, W - 1)
    y1 = clamp_int(ty - th - pad, 0, H - 1)
    x2 = clamp_int(tx + tw + pad, 0, W - 1)
    y2 = clamp_int(ty + baseline + pad, 0, H - 1)

    # 标签中心，用于画箭头
    box_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=float)
    p_corner = tuple(np.round(corner_pt).astype(int))
    p_box = tuple(np.round(box_center).astype(int))

    # 角点圆圈 + 箭头 + 白底大号标签
    cv2.circle(img, p_corner, 9, color, -1)
    cv2.circle(img, p_corner, 13, (255, 255, 255), 2)
    #cv2.arrowedLine(img, p_box, p_corner, color, thickness=3, tipLength=0.22)

    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
    cv2.putText(img, text, (tx, ty), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)

    return {
        "letter": letter,
        "corner_pixel": [float(corner_pt[0]), float(corner_pt[1])],
        "label_box": [int(x1), int(y1), int(x2), int(y2)],
    }


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
        f"These two images show the same defective item, '{object_name}', taken from the same camera view.\n"
        "Image 1 is the original RGB image.\n"
        "Image 2 is the same image with four corner labels: A, B, C， and D.\n"
        "The labels are attached to the four physical object corners.\n\n"
        "Task: identify which label in Image 2 corresponds to the defective/damaged corner visible in Image 1.\n\n"
        "Rules:\n"
        "- Choose exactly one from A, B, C, D.\n"
        "- Do not output any other words."
        "- Do not explain.\n"
        "- Return only JSON in this format: {\"label\": \"A\"}\n"
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


def parse_corner_letter(model_output: str) -> str:
    """
    从模型输出中解析 A/B/C/D。
    兼容：{"label":"A"}、A、[A]、The answer is A 等输出。
    """
    if model_output is None:
        raise ValueError("模型输出为空，无法解析 corner label")

    s = str(model_output).strip()
    s_upper = s.upper()

    # 优先解析 JSON
    try:
        data = json.loads(s)
        if isinstance(data, dict):
            label = str(data.get("label", "")).strip().upper()
            if label in LETTER_TO_CORNER:
                return label
    except Exception:
        pass

    # 解析 "label": "A" 这种不完全标准 JSON
    m = re.search(r'["\']?label["\']?\s*[:=]\s*["\']?([ABCD])["\']?', s, flags=re.I)
    if m:
        return m.group(1).upper()

    # 解析单独的 A/B/C/D，避免把 Answer 里的 A 当成标签
    candidates = re.findall(r'(?<![A-Z0-9])([ABCD])(?![A-Z0-9])', s_upper)
    if candidates:
        return candidates[-1]

    raise ValueError(f"无法从模型输出中解析 A/B/C/D: {model_output}")


def save_corner_mapping(mapping_path, corner_names, corner_uv, corner_pixels, corner_valid, label_draw_infos=None):
    """
    保存 A/B/C/D 到真实 corner_mode 的映射，后续模型只需要识别字母。
    """
    label_draw_infos = label_draw_infos or {}
    mapping = {}
    for i, corner_mode in enumerate(corner_names):
        letter = CORNER_TO_LETTER[corner_mode]
        p = corner_pixels[i]
        uv = corner_uv[corner_mode]
        mapping[letter] = {
            "corner_mode": corner_mode,
            "uv": [float(uv[0]), float(uv[1])],
            "pixel": [float(p[0]), float(p[1])],
            "valid": bool(corner_valid[i]),
        }
        if letter in label_draw_infos:
            mapping[letter].update(label_draw_infos[letter])

    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    print("Saved corner label mapping:", mapping_path)
    return mapping


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
        "A": (255, 0, 0),       # blue in BGR
        "B": (0, 180, 255),     # orange/yellow
        "C": (0, 0, 255),       # red
        "D": (0, 130, 0),       # green
    }

    # 用投影后的四角中心作为文字外偏移的参考中心
    valid_corner_pixels = corner_pixels[corner_valid]
    if len(valid_corner_pixels) > 0:
        object_center_pixel = np.nanmean(valid_corner_pixels, axis=0)
    else:
        object_center_pixel = np.array([W_img / 2, H_img / 2], dtype=float)

    label_draw_infos = {}

    for i, name in enumerate(corner_names):
        if not corner_valid[i]:
            print(f"角点 {name} 投影无效")
            continue

        p = corner_pixels[i]

        if not (0 <= p[0] < W_img and 0 <= p[1] < H_img):
            print(f"角点 {name} 投影在图像外: {p}")

        letter = CORNER_TO_LETTER[name]
        label_draw_infos[letter] = draw_corner_letter_label(
            img,
            letter=letter,
            corner_pt=p,
            object_center_pt=object_center_pixel,
            color=label_colors[letter]
        )

    # 保存 A/B/C/D -> corner_mode 的映射，模型只判断字母，代码再转回真实 corner_mode
    corner_mapping = save_corner_mapping(
        MAPPING_PATH,
        corner_names=corner_names,
        corner_uv=corner_uv,
        corner_pixels=corner_pixels,
        corner_valid=corner_valid,
        label_draw_infos=label_draw_infos,
    )

    # ----------------------------
    # 8. 画 +u / +v 箭头
    # ----------------------------
    #if np.all(arrow_valid):
    #    draw_arrow(
    #        img,
    #        arrow_pixels[0],
    #        arrow_pixels[1],
    #        color=(0, 0, 255),   # 红色 +u
    #        text="+u"
    #    )
#
    #    draw_arrow(
    #        img,
    #        arrow_pixels[0],
    #        arrow_pixels[2],
    #        color=(255, 0, 0),   # 蓝色 +v
    #        text="+v"
    #    )
    #else:
    #    print("警告：u/v 箭头投影无效")

    # ----------------------------
    # 9. 保存结果
    # ----------------------------
    cv2.imwrite(OUT_PATH, img)
    print("Saved annotated RGB:", OUT_PATH)

    # 注意：SSH/Jupyter/超算节点通常没有 GUI，不要用 cv2.imshow，容易导致 kernel died。
    # 如需查看结果，直接打开 OUT_PATH 保存的图片。

    ######################### inference ######################### 
    messages = build_defect_corner_prompt(RGB_PATH, OUT_PATH, 'a black object')
    raw_response = qwen3_inference(messages)
    corner_letter = parse_corner_letter(raw_response)
    corner_mode = LETTER_TO_CORNER[corner_letter]

    #result = {
    #    "raw_response": str(raw_response),
    #    "corner_letter": corner_letter,
    #    "corner_mode": corner_mode,
    #    "mapping_path": MAPPING_PATH,
    #    "labeled_image_path": OUT_PATH,
    #}
    #with open(INFERENCE_RESULT_PATH, "w", encoding="utf-8") as f:
    #    json.dump(result, f, indent=2, ensure_ascii=False)
#
    print("raw_response:", raw_response)
    print("corner_letter:", corner_letter)
    print("corner_mode:", corner_mode)
    #print("Saved inference result:", INFERENCE_RESULT_PATH)


if __name__ == "__main__":
    main()
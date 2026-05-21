import cv2
import numpy as np
import math


def segment_white_grip(
    image_path,
    roi=None,
    min_area=500,
    min_aspect=2.5,
    hsv_s_max=80,
    hsv_v_min=140,
    save_path="grip_seg_result.png"
):
    """
    分割白色 grip，并用 PCA 拟合中心和主轴。

    Args:
        image_path: 输入 RGB 图像路径
        roi: (x, y, w, h)，理论 grip ROI。None 表示全图，但不推荐。
        min_area: 最小连通域面积
        min_aspect: 最小长宽比
        hsv_s_max: 白色阈值，饱和度上限
        hsv_v_min: 白色阈值，亮度下限
        save_path: 可视化结果保存路径

    Returns:
        result dict:
            center: grip 中心像素坐标 (u, v)
            angle_deg: grip 主轴角度，图像坐标系下
            axis: PCA 主轴方向向量
            mask: 分割 mask
    """

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    H, W = img.shape[:2]

    # -----------------------------
    # 1. 裁剪理论 ROI
    # -----------------------------
    if roi is None:
        x, y, w, h = 0, 0, W, H
    else:
        x, y, w, h = roi
        x = max(0, x)
        y = max(0, y)
        w = min(w, W - x)
        h = min(h, H - y)

    crop = img[y:y+h, x:x+w].copy()

    # -----------------------------
    # 2. HSV 白色阈值分割
    # -----------------------------
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, hsv_v_min])
    upper_white = np.array([179, hsv_s_max, 255])

    mask = cv2.inRange(hsv, lower_white, upper_white)

    # -----------------------------
    # 3. 形态学去噪
    # -----------------------------
    kernel_open = np.ones((3, 3), np.uint8)
    kernel_close = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    # -----------------------------
    # 4. 连通域筛选
    # -----------------------------
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)

    best_label = None
    best_score = -1

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        bx = stats[i, cv2.CC_STAT_LEFT]
        by = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]

        if area < min_area:
            continue

        aspect = max(bw, bh) / max(1, min(bw, bh))

        if aspect < min_aspect:
            continue

        # grip 是细长结构，优先选择面积大且长宽比大的区域
        score = area * aspect

        if score > best_score:
            best_score = score
            best_label = i

    if best_label is None:
        print("未找到符合条件的 grip 区域")
        return None

    grip_mask_roi = np.zeros_like(mask)
    grip_mask_roi[labels == best_label] = 255

    # -----------------------------
    # 5. PCA 拟合 grip 中心和主轴
    # -----------------------------
    ys, xs = np.where(grip_mask_roi > 0)
    pts = np.stack([xs, ys], axis=1).astype(np.float32)

    mean, eigenvectors, eigenvalues = cv2.PCACompute2(pts, mean=None)

    center_roi = mean[0]                 # ROI 内坐标
    axis = eigenvectors[0]               # 主轴方向
    angle_deg = math.degrees(math.atan2(axis[1], axis[0]))

    center_full = np.array([center_roi[0] + x, center_roi[1] + y])

    # -----------------------------
    # 6. 可视化
    # -----------------------------
    vis = img.copy()

    # 画 ROI
    cv2.rectangle(vis, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # 画 grip mask 轮廓
    contours, _ = cv2.findContours(grip_mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        c_full = c + np.array([[[x, y]]])
        cv2.drawContours(vis, [c_full], -1, (0, 255, 0), 2)

    # 画 PCA 中心
    c = tuple(center_full.astype(int))
    cv2.circle(vis, c, 5, (0, 0, 255), -1)

    # 画 PCA 主轴
    length = 120
    p1 = (
        int(center_full[0] - length * axis[0]),
        int(center_full[1] - length * axis[1])
    )
    p2 = (
        int(center_full[0] + length * axis[0]),
        int(center_full[1] + length * axis[1])
    )
    cv2.line(vis, p1, p2, (0, 0, 255), 3)

    cv2.putText(
        vis,
        f"center=({center_full[0]:.1f},{center_full[1]:.1f}), angle={angle_deg:.1f}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    cv2.imwrite(save_path, vis)

    print("grip center pixel:", center_full)
    print("grip axis:", axis)
    print("grip angle deg:", angle_deg)
    print("result saved to:", save_path)

    return {
        "center": center_full,
        "axis": axis,
        "angle_deg": angle_deg,
        "mask": grip_mask_roi,
        "roi": (x, y, w, h)
    }


if __name__ == "__main__":
    image_path = "D:/downloads/example_down160mm_Color.png"

    # 先手动给 ROI。
    # 这个 ROI 是针对你当前样例图的大概区域，后面应替换成 CAD/刚性计算投影得到的 grip ROI。
    roi = (350, 100, 90, 150)  # x, y, w, h

    result = segment_white_grip(
        image_path=image_path,
        roi=roi,
        min_area=500,
        min_aspect=2.5,
        hsv_s_max=80,
        hsv_v_min=140,
        save_path="grip_seg_result.png"
    )
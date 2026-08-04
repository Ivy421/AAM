"""
手眼标定结果可视化
用硬件计算的ECT和手眼标定ECT绘制对比
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy as np


def check_rotation(R, name="R"):
    R = np.asarray(R, dtype=float)
    print(f"{name} det =", np.linalg.det(R))
    print(f"{name} orthogonal error =", np.linalg.norm(R.T @ R - np.eye(3)))


def draw_frame(ax, R, origin=np.zeros(3), name="A", length=1.0):
    """
    R: 该坐标系相对于世界/A坐标系的旋转矩阵
       R[:,0], R[:,1], R[:,2] 分别是该坐标系 x,y,z 轴方向
    """
    colors = ["r", "g", "b"]
    labels = ["x", "y", "z"]

    for i in range(3):
        axis = R[:, i]
        ax.quiver(
            origin[0], origin[1], origin[2],
            axis[0], axis[1], axis[2],
            length=length,
            color=colors[i],
            arrow_length_ratio=0.12,
            linewidth=2
        )
        end = origin + length * axis
        ax.text(end[0], end[1], end[2], f"{name}_{labels[i]}", color=colors[i])

    ax.text(origin[0], origin[1], origin[2], name, fontsize=12)


def set_axes_equal(ax, lim=1.2):
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("X_A")
    ax.set_ylabel("Y_A")
    ax.set_zlabel("Z_A")

ecT_old = np.load('/home/smmg/AAM/config/calibration/right_camera/ecT_0706.npy')
# =========================
# 你的两个旋转矩阵
# =========================

ecT = np.load('/home/smmg/AAM/config/alignment/test_data/ecT_20260727.npy')

R_AB = ecT[:3,:3]
R_AC = ecT_old[:3,:3]

# 检查是否为合法旋转矩阵
check_rotation(R_AB, "R_AB")
check_rotation(R_AC, "R_AC")

# =========================
# 可视化
# =========================

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")

# A坐标系
#draw_frame(ax, np.eye(3), name="A", length=1.0)

# B坐标系
draw_frame(ax, R_AB, name="B", length=0.8)

# C坐标系
draw_frame(ax, R_AC, name="C", length=0.8)

set_axes_equal(ax, lim=1.2)
ax.set_title("Rotation Matrix Visualization")

plt.show()
print(ecT_old[:3,3], ecT[:3,3])
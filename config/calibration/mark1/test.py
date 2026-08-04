"""
手眼标定结果可视化
用硬件计算的ECT和手眼标定ECT绘制对比
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

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

mark_base_T = np.load('E:/HKUSTGZ/AAM/config/calibration/mark1/data1/temp_chess1/calibration_result/T_mark1_armbase.npy')
mark_base_T1 = np.load('E:/HKUSTGZ/AAM/config/calibration/mark1/data1/calibration_result/T_mark1_armbase.npy')


ecT = np.load('E:/HKUSTGZ/AAM/config/calibration/right_camera/ecT.npy')
print(ecT)
r = R.from_euler('xyz', [0,85,0], degrees=True).as_matrix()
beT = np.column_stack([r,
                       np.array([[0.056],[0],[0.213]])])
beT = np.vstack([beT,np.array([0,0,0,1])])
#print(beT)
bcT = beT @ ecT
print('bcT:        /n ', bcT)
np.save("E:\HKUSTGZ\AAM\config\calibration\mark1/bcT.npy", bcT)


fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")

# A坐标系
draw_frame(ax, np.eye(3), name="mark", length=0.8)
# B坐标系
draw_frame(ax, mark_base_T[:3, :3], origin =np.array([-0.623, -0.09, 0]  ), name="t0", length=1.2)

draw_frame(ax, mark_base_T1[:3, :3], origin =np.array([-0.58, -0.26, 0]), name="t1", length=1.2)

set_axes_equal(ax, lim=1.2)
plt.show()

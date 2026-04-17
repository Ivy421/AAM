import sys
sys.path.append('/home/smmg/AAM')
import open3d as o3d
import numpy as np
import pyransac3d as pyrsc

points = np.load('/home/smmg/AAM/construction/result/image.npy')

print("\n--- 使用 pyransac3d 进行智能圆柱拟合 ---")

pts_3d = points.astype(np.float32)

# 初始化圆柱 RANSAC 拟合器

cylinder = pyrsc.Cylinder()
center, axis, radius, inliers = cylinder.fit(np.array(points), thresh=0.01, maxIteration=200) 

print(f"✅ 拟合成功！")
print(f"📍 底面中心: {center}")
print(f"📏 半径: {radius:.4f}")
print(f"🧭 轴线方向: {axis}")
print(f"📊 内点占比: {len(inliers)/len(pts_3d)*100:.1f}%")

# 估算杯子高度（沿轴线投影的最大跨度）
if len(inliers) > 0:
    inlier_pts = pts_3d[inliers]
    proj = np.dot(inlier_pts - center, axis)
    height = proj.max() - proj.min()
    print(f"📐 估算高度: {height:.4f}")


############可视化

proj = np.dot(points - center, axis)
h_min, h_max = proj.min(), proj.max()

# 3. 构建垂直于 axis 的局部坐标系 (u, v)
if abs(axis[0]) < 0.9:
    temp = np.array([1.0, 0.0, 0.0])
else:
    temp = np.array([0.0, 1.0, 0.0])
u = np.cross(axis, temp)
u = u / np.linalg.norm(u)
v = np.cross(axis, u)  # v 自动单位化

# 4. 参数化生成圆柱面点云
n_theta = 100  # 圆周分辨率
n_h = 30       # 高度分辨率
thetas = np.linspace(0, 2*np.pi, n_theta)
hs = np.linspace(h_min, h_max, n_h)

cylinder_pts = []
for h in hs:
    for theta in thetas:
        # 圆柱参数方程: P = center + h*axis + r*(cosθ*u + sinθ*v)
        pt = center + h * axis + radius * (np.cos(theta)*u + np.sin(theta)*v)
        cylinder_pts.append(pt)
cylinder_pts = np.array(cylinder_pts)


pcd = o3d.geometry.PointCloud()
pcd_cylinder = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd_cylinder.points = o3d.utility.Vector3dVector(cylinder_pts)

# 如果有对应的颜色
# pcd.colors = o3d.utility.Vector3dVector(np.ndarray([0.5,0,0.9]))

o3d.visualization.draw_geometries([pcd, pcd_cylinder])

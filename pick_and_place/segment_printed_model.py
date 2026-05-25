import numpy as np
import matplotlib.pyplot as plt

data_dir= "E:/HKUSTGZ/AAM/config/alignment/test_data/depression_defect/pick_data"
# 1. load depth npy
depth = np.load(data_dir + "/1.npy")   # 修改为你的 .npy 路径

# 2. squeeze if shape is HxWx1
depth = np.squeeze(depth)

# 3. mask invalid values if needed
depth_vis = depth.astype(float)
depth_vis[depth_vis <= 0] = np.nan   # 深度为0通常表示无效，可按需删除

# 4. visualize
plt.figure(figsize=(8, 6))
plt.imshow(depth_vis, cmap="viridis")
plt.colorbar(label="Depth")
plt.title("Depth Map")
plt.axis("off")
plt.show()


##################### 白色物体阈值分割

import cv2
import numpy as np

img_path = data_dir + '/1.png'
img = cv2.imread(img_path)

# HSV 白色阈值分割
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 白色：低饱和度 + 高亮度
lower_white = np.array([0, 0, 160])
upper_white = np.array([180, 60, 255])

mask = cv2.inRange(hsv, lower_white, upper_white)

# 形态学去噪
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# 提取白色区域
result = cv2.bitwise_and(img, img, mask=mask)

# 可视化
cv2.imshow("image", img)
cv2.imshow("white mask", mask)
cv2.imshow("white result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
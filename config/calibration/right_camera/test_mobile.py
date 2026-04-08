import cv2, os
import numpy as np


img1 = cv2.imread('/home/smmg/AAM/config/calibration/right_camera/3_Color.png')
img2 = cv2.imread('/home/smmg/AAM/config/calibration/right_camera/4_Color.png')

# 转换为灰度图用于比较
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

results = {}

# 1. 均方误差 (MSE) - 值越小表示图像越相似
mse = np.mean((gray1.astype(float) - gray2.astype(float)) ** 2)
results['MSE'] = mse

# 2. 峰值信噪比 (PSNR) - 值越大表示图像质量越好
if mse == 0:
    psnr = float('inf')
else:
    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
results['PSNR'] = psnr

# 3. 结构相似性指数 (SSIM) - 越接近1.0表示越相似
# ssim_score, ssim_diff = ssim(gray1, gray2, full=True)
# results['SSIM'] = ssim_score

# 4. 绝对差值均值
abs_diff = cv2.absdiff(gray1, gray2)
mean_abs_diff = np.mean(abs_diff)
results['平均绝对差值'] = mean_abs_diff

# 5. 归一化互相关系数
corr = np.corrcoef(gray1.flatten(), gray2.flatten())[0, 1]
results['相关系数'] = corr

shift, error = cv2.phaseCorrelate(np.float32(gray1), np.float32(gray2))
results['X方向位移(像素)'] = shift[0]
results['Y方向位移(像素)'] = shift[1]
results['总位移(像素)'] = np.sqrt(shift[0]**2 + shift[1]**2)

print(results)
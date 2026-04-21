import os, json, cv2, sys
sys.path.append('/home/smmg/AAM')
#from camera.camera_functions import *
from piper_motion.piper_functions import *

def draw_point( ):
        img = cv2.imread('/home/smmg/AAM/image.png')

        # 2. 定义点的坐标 (x, y) = (222, 333)
        point = (320,256)
        color = (0, 0, 255)      # BGR格式：红色
        radius = 5               # 点的半径（像素）
        thickness = -1           # -1 表示实心填充

        # 方法一：使用 circle 画点（最常用）
        cv2.circle(img, point, radius, color, thickness)

        # 方法二：使用 drawMarker 画标记点（OpenCV 3.0+ 推荐，自带十字/圆形等样式）
        # cv2.drawMarker(img, point, color, markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

        # 3. 可视化图像
        cv2.imshow('Draw Point at cx,cy', img)
        cv2.waitKey(0)           # 等待按键，0表示无限等待
        cv2.destroyAllWindows()  # 关闭窗口

        return 

#disable('r_piper')
piper = enable('r_piper')
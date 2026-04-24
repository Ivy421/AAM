import pyrealsense2 as rs
import numpy as np
import cv2, os, time, subprocess
from datetime import datetime
import matplotlib.pyplot as plt
from piper_sdk import *
import sys, os,json


# 保存帧的函数
def save_frames(save_dir, save_file_name, color_image, depth_data, frame_count):
    """
    保存彩色帧和深度帧到本地文件
    """
    np.save(save_dir + save_file_name +'.npy', depth_data)
    
    # 保存彩色图像 (PNG 格式)
    color_path = os.path.join(save_dir, f"{save_file_name}.png")
    cv2.imwrite(color_path, color_image)
    
    print(f"✓ 已保存帧 #{frame_count}:")
    print(f"  - 彩色图像：{color_path}")

    
    return 0

def save_config(camera_depth_intrinsics, camera_color_intrinsics,camera_depth_to_color_extrinsics, depth_scale,save_to_path):
    config_data = {
        'depth_intrinsic': {
            'width': camera_depth_intrinsics.width,
            'height': camera_depth_intrinsics.height,
            'ppx': camera_depth_intrinsics.ppx,
            'ppy': camera_depth_intrinsics.ppy,
            'fx': camera_depth_intrinsics.fx,
            'fy': camera_depth_intrinsics.fy,
            'coeffs': list(camera_depth_intrinsics.coeffs),
            'model': int(camera_depth_intrinsics.model)  # 转为 int 避免枚举类型序列化问题
        },
        'color_intrinsic': {
            'width': camera_color_intrinsics.width,
            'height': camera_color_intrinsics.height,
            'ppx': camera_color_intrinsics.ppx,
            'ppy': camera_color_intrinsics.ppy,
            'fx': camera_color_intrinsics.fx,
            'fy': camera_color_intrinsics.fy,
            'coeffs': list(camera_color_intrinsics.coeffs),
            'model': int(camera_color_intrinsics.model)
        },
        'depth_to_color_extrinsic': {
            'rotation': list(camera_depth_to_color_extrinsics.rotation),
            'translation': list(camera_depth_to_color_extrinsics.translation)
        },
        'depth_scale': float(depth_scale)
    }
    np.save(save_to_path+'/camera_config.npy', config_data)
    return 

def capture(img_save_path, save_file_name='image' ,AUTO_SAVE_INTERVAL=2.0, MAX_SAVE_FRAMES = 1, SAVE_CONFIG = 1, post_process = 1,SAVE_ENDPOSE=True):
    # 配置管道 (Pipeline)
    pipeline = rs.pipeline()
    config = rs.config()

    # 启用流：彩色 640x480 @ 30fps, 深度 640x480 @ 30fps
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth,640, 480, rs.format.z16, 30)

    # 开始管道
    cfg = pipeline.start(config)

    ## 过滤深度信息
    if post_process == 1:
        # 声明滤波器（按照推荐顺序）
        spatial = rs.spatial_filter()
        temporal = rs.temporal_filter()
        # 配置滤波器参数
        spatial.set_option(rs.option.filter_magnitude, 2)     # 2次迭代
        spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
        spatial.set_option(rs.option.filter_smooth_delta, 20)
        # spatial.set_option(rs.option.holes_fill, 0)  # 不填充空洞
        temporal.set_option(rs.option.filter_smooth_alpha, 0.3)
        temporal.set_option(rs.option.filter_smooth_delta, 20)
        temporal.set_option(rs.option.holes_fill, 2)  # Valid in 2/last 4

            
    ## 保存相机内置参数
    if SAVE_CONFIG ==1:
        # get scale
        depth_sensor = cfg.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        # get intrinsics
        camera_depth_profile = cfg.get_stream(rs.stream.depth)                                      # fetch depth depth stream profile
        camera_depth_intrinsics = camera_depth_profile.as_video_stream_profile().get_intrinsics()   # downcast to video_stream_profile and fetch intrinsics
        
        camera_color_profile = cfg.get_stream(rs.stream.color)                                      # fetch color stream profile
        camera_color_intrinsics = camera_color_profile.as_video_stream_profile().get_intrinsics()   # downcast to video_stream_profile and fetch intrinsics
        
        camera_depth_to_color_extrinsics = camera_depth_profile.get_extrinsics_to(camera_color_profile)
        save_config(camera_depth_intrinsics, camera_color_intrinsics,camera_depth_to_color_extrinsics, depth_scale, img_save_path)

    
    # 深度图对齐到彩色图 后续都采用RGB相机的内参
    align = rs.align(rs.stream.color)

    print("Intel RealSense D435 - 自动捕获程序")

    try:
        save_count = 0
        last_save_time = time.time()
        
        while True:
            current_time = time.time()
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
    
            # 获取对齐后的深度帧和彩色帧
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if post_process == 1:     # 1. 降采样
                depth_frame = spatial.process(depth_frame)         # 2. 空间滤波
                depth_frame = temporal.process(depth_frame)        # 3. 时间滤波
            
            if not depth_frame or not color_frame:
                continue
            depth_data = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # 检查是否达到自动保存时间
            if current_time - last_save_time >= AUTO_SAVE_INTERVAL:
                save_count += 1
                save_frames(img_save_path, save_file_name, color_image, depth_data, save_count)
                if SAVE_ENDPOSE == True:
                    endpose_info = synchron_piper('r_piper',img_save_path+save_file_name+'.json' )
                    print(endpose_info)

                last_save_time = current_time
                
                #⭕检查是否达到最大保存帧数，达到则自动退出
                if save_count >= MAX_SAVE_FRAMES:
                    print(f"\n capturing complete")

                    break
    finally:
        # 停止管道
        pipeline.stop()
        cv2.destroyAllWindows()
        print(f"\n程序结束，共保存 {save_count} 帧")

    return 


def image_visualization(image_path, depth_path):
    # 1. 加载RGB图像
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 2. 加载深度图 (.npy)
    depth = np.load(depth_path)
    if depth.ndim != 2:
        raise ValueError("深度图应为2维数组 (H, W)，当前维度: {}".format(depth.ndim))
    
    # 3. 预处理深度图（处理无效值与极端离群点）
    depth_vis = depth.copy().astype(np.float32)
    # 标记无效点（NaN, Inf, <=0 通常为传感器未测得或过滤残留）
    invalid_mask = np.isnan(depth_vis) | np.isinf(depth_vis) | (depth_vis <= 0)
    depth_vis[invalid_mask] = np.nan
    
    # 动态范围压缩：使用百分位数截断，避免残留极值导致主体深度对比度不足
    valid_vals = depth_vis[~np.isnan(depth_vis)]
    if len(valid_vals) > 0:
        vmin = np.percentile(valid_vals, 2)
        vmax = np.percentile(valid_vals, 98)
    else:
        vmin, vmax = 0, 1
        
    # 4. 创建画布并可视化
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # 左侧：RGB原图
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original RGB Image", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # 右侧：深度图
    # 使用 'viridis' 感知均匀色图，NaN 区域会自动显示为透明/底色
    cax = axes[1].imshow(depth, cmap='viridis', vmin=vmin, vmax=vmax, interpolation='nearest')
    axes[1].set_title("Processed Depth Map", fontsize=14, fontweight='bold')
    axes[1].axis('off')
    fig.colorbar(cax, ax=axes[1], label="Depth Value", shrink=0.8)
    
    plt.tight_layout()
    plt.show()

def synchron_piper(arm_name,camera_syn_endpose_path):
    piper = C_PiperInterface_V2('r_piper')
    piper.ConnectPort()
    time.sleep(0.1)
    while( not piper.EnablePiper()):
        time.sleep(0.01)
    print("使能成功!!!!")
    endpose_info = piper.GetArmEndPoseMsgs()
    data = [{'x':endpose_info.end_pose.X_axis},
            {"y":endpose_info.end_pose.Y_axis},
            {"z":endpose_info.end_pose.Z_axis},
            {"rx":endpose_info.end_pose.RX_axis},
            {"ry":endpose_info.end_pose.RY_axis},
            {"rz":endpose_info.end_pose.RZ_axis}]
    with open(camera_syn_endpose_path, 'w', encoding = 'utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return endpose_info


capture('/home/smmg/AAM/config/alignment/test_data/','5',1,1,0,1,True)
import pyrealsense2 as rs
import numpy as np
import cv2, os, time, json
import matplotlib.pyplot as plt

COLOR_WIDTH, COLOR_HEIGHT = 640, 480
DEPTH_WIDTH, DEPTH_HEIGHT = 640, 480
FPS = 30
WARMUP_FRAMES = 30

def save_frames(save_dir, save_file_name, color_image, depth_data):
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f"{save_file_name}.npy"), depth_data)
    cv2.imwrite(os.path.join(save_dir, f"{save_file_name}.png"), color_image)

def save_config(camera_depth_intrinsics, camera_color_intrinsics, camera_depth_to_color_extrinsics, depth_scale, save_to_path):
    config_data = {
        "depth_intrinsic": {"width": camera_depth_intrinsics.width, "height": camera_depth_intrinsics.height, "ppx": camera_depth_intrinsics.ppx, "ppy": camera_depth_intrinsics.ppy, "fx": camera_depth_intrinsics.fx, "fy": camera_depth_intrinsics.fy, "coeffs": list(camera_depth_intrinsics.coeffs), "model": int(camera_depth_intrinsics.model)},
        "color_intrinsic": {"width": camera_color_intrinsics.width, "height": camera_color_intrinsics.height, "ppx": camera_color_intrinsics.ppx, "ppy": camera_color_intrinsics.ppy, "fx": camera_color_intrinsics.fx, "fy": camera_color_intrinsics.fy, "coeffs": list(camera_color_intrinsics.coeffs), "model": int(camera_color_intrinsics.model)},
        "depth_to_color_extrinsic": {"rotation": list(camera_depth_to_color_extrinsics.rotation), "translation": list(camera_depth_to_color_extrinsics.translation)},
        "depth_scale": float(depth_scale),
    }
    os.makedirs(save_to_path, exist_ok=True)
    np.save(os.path.join(save_to_path, "camera_config.npy"), config_data)

def capture(img_save_path, save_file_name="image", AUTO_SAVE_INTERVAL=2.0, MAX_SAVE_FRAMES=1, SAVE_CONFIG=1, post_process=0, SAVE_ENDPOSE=True):
    os.makedirs(img_save_path, exist_ok=True)
    pipeline, config = rs.pipeline(), rs.config()
    config.enable_stream(rs.stream.color, COLOR_WIDTH, COLOR_HEIGHT, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, DEPTH_WIDTH, DEPTH_HEIGHT, rs.format.z16, FPS)
    cfg = None

    try:
        cfg = pipeline.start(config)
        depth_sensor = cfg.get_device().first_depth_sensor()

        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 1)

        if depth_sensor.supports(rs.option.laser_power):
            laser_range = depth_sensor.get_option_range(rs.option.laser_power)
            laser_power = laser_range.min + 0.8 * (laser_range.max - laser_range.min)
            depth_sensor.set_option(rs.option.laser_power, laser_power)
            print(f"Laser power: {laser_power:.1f}")

        spatial, temporal = None, None
        if post_process == 1:
            spatial, temporal = rs.spatial_filter(), rs.temporal_filter()
            spatial.set_option(rs.option.filter_magnitude, 2)
            spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
            spatial.set_option(rs.option.filter_smooth_delta, 20)
            temporal.set_option(rs.option.filter_smooth_alpha, 0.3)
            temporal.set_option(rs.option.filter_smooth_delta, 20)

        if SAVE_CONFIG == 1:
            depth_scale = depth_sensor.get_depth_scale()
            camera_depth_profile = cfg.get_stream(rs.stream.depth)
            camera_color_profile = cfg.get_stream(rs.stream.color)
            camera_depth_intrinsics = camera_depth_profile.as_video_stream_profile().get_intrinsics()
            camera_color_intrinsics = camera_color_profile.as_video_stream_profile().get_intrinsics()
            camera_depth_to_color_extrinsics = camera_depth_profile.get_extrinsics_to(camera_color_profile)
            save_config(camera_depth_intrinsics, camera_color_intrinsics, camera_depth_to_color_extrinsics, depth_scale, img_save_path)

        align = rs.align(rs.stream.color)
        print("开始相机 warmup ...")

        for _ in range(WARMUP_FRAMES):
            aligned_frames = align.process(pipeline.wait_for_frames())
            if post_process == 1:
                depth_frame = aligned_frames.get_depth_frame()
                if depth_frame:
                    depth_frame = temporal.process(spatial.process(depth_frame))

        print(f"Warmup 完成：已丢弃 {WARMUP_FRAMES} 帧")
        print("开始正式拍摄")

        aligned_frames = align.process(pipeline.wait_for_frames())
        depth_frame, color_frame = aligned_frames.get_depth_frame(), aligned_frames.get_color_frame()
        if not depth_frame:
            raise RuntimeError("failed to obtain aligned depth frame")
        if not color_frame:
            raise RuntimeError("failed to obtain color frame")

        if post_process == 1:
            depth_frame = temporal.process(spatial.process(depth_frame))

        if SAVE_ENDPOSE:
            endpose_path = os.path.join(img_save_path, save_file_name + ".json")
            endpose_info = synchron_piper("r_piper", endpose_path)
            print("Endpose:", endpose_info)

        depth_data = np.asanyarray(depth_frame.get_data()).copy()
        color_image = np.asanyarray(color_frame.get_data()).copy()
        save_frames(img_save_path, save_file_name, color_image, depth_data)

        valid_ratio = float(np.count_nonzero(depth_data > 0) / depth_data.size)
        print(f"保存完成: {save_file_name}.png / {save_file_name}.npy")
        print(f"Valid depth ratio: {valid_ratio:.4f}")

    finally:
        if cfg is not None:
            pipeline.stop()
        cv2.destroyAllWindows()
        print("\n拍摄完成！")

def image_visualization(image_path, depth_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    depth = np.load(depth_path)
    if depth.ndim != 2:
        raise ValueError(f"深度图应为2维数组 (H, W)，当前维度: {depth.ndim}")

    depth_vis = depth.astype(np.float32, copy=True)
    invalid_mask = np.isnan(depth_vis) | np.isinf(depth_vis) | (depth_vis <= 0)
    depth_vis[invalid_mask] = np.nan
    valid_vals = depth_vis[~np.isnan(depth_vis)]
    vmin, vmax = (np.percentile(valid_vals, 2), np.percentile(valid_vals, 98)) if len(valid_vals) > 0 else (0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original RGB Image", fontsize=14, fontweight="bold")
    axes[0].axis("off")
    cax = axes[1].imshow(depth_vis, cmap="jet", vmin=vmin, vmax=vmax, interpolation="nearest")
    axes[1].set_title("Processed Depth Map", fontsize=14, fontweight="bold")
    axes[1].axis("off")
    fig.colorbar(cax, ax=axes[1], label="Depth Value", shrink=0.8)
    plt.tight_layout()
    plt.show()

def synchron_piper(arm_name="r_piper", save_path=None):
    from Piper.piper_ctrl import connect_piper

    piper = connect_piper(arm_name, with_gripper=False)
    try:
        endpose = None
        for _ in range(10):
            endpose = piper.get_endpose()
            if endpose is not None:
                break
            time.sleep(0.1)

        if endpose is None:
            raise RuntimeError(f"failed to read {arm_name} endpose")

        x, y, z, rx, ry, rz = endpose
        data = [{"x": x * 1000.0}, {"y": y * 1000.0}, {"z": z * 1000.0}, {"rx": rx * 1000.0}, {"ry": ry * 1000.0}, {"rz": rz * 1000.0}]

        if save_path is not None:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        return data
    finally:
        piper.disconnect()

if __name__ == "__main__":
    save_path = "E:/HKUSTGZ/temp/"
    save_name = "test_fx"
    capture(save_path, save_name, 2, 1, 1, 0, False)

import json
import pyrealsense2 as rs


def get_camera_config():
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)

    device = profile.get_device()
    depth_sensor = device.first_depth_sensor()

    depth_profile = profile.get_stream(
        rs.stream.depth
    ).as_video_stream_profile()

    color_profile = profile.get_stream(
        rs.stream.color
    ).as_video_stream_profile()

    depth_intrinsics = depth_profile.get_intrinsics()
    color_intrinsics = color_profile.get_intrinsics()
    depth_to_color = depth_profile.get_extrinsics_to(color_profile)

    camera_config = {
        "camera_name": device.get_info(rs.camera_info.name),
        "serial_number": device.get_info(rs.camera_info.serial_number),
        "depth_scale": depth_sensor.get_depth_scale(),
        "depth": {
            "width": depth_profile.width(),
            "height": depth_profile.height(),
            "fps": depth_profile.fps(),
            "format": str(depth_profile.format()),
            "intrinsics": {
                "fx": depth_intrinsics.fx,
                "fy": depth_intrinsics.fy,
                "cx": depth_intrinsics.ppx,
                "cy": depth_intrinsics.ppy,
                "coeffs": list(depth_intrinsics.coeffs),
            },
        },
        "color": {
            "width": color_profile.width(),
            "height": color_profile.height(),
            "fps": color_profile.fps(),
            "format": str(color_profile.format()),
            "intrinsics": {
                "fx": color_intrinsics.fx,
                "fy": color_intrinsics.fy,
                "cx": color_intrinsics.ppx,
                "cy": color_intrinsics.ppy,
                "coeffs": list(color_intrinsics.coeffs),
            },
        },
        "depth_to_color": {
            "rotation": list(depth_to_color.rotation),
            "translation": list(depth_to_color.translation),
        },
    }

    pipeline.stop()
    return camera_config


if __name__ == "__main__":
    print(json.dumps(get_camera_config(), indent=4))

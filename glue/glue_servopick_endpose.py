"""Estimate the glue-station AprilTag pose and compute the Piper grasp pose.

--needle-axis=-y,apriltag粘贴时候要保证正向朝针头轴向

"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from Piper import endpose_reachability_safe as ik
CAMERA_CONFIG = '/home/smmg/AAM/config/calibration/right_camera/camera_config.npy'
EYE_HAND = '/home/smmg/AAM/config/calibration/right_camera/ecT.npy'
URDF = '/home/smmg/AAM/config/piper/piper_description.urdf'

# Grasp point expressed in the custom station/tag frame, unit: mm.
GRASP_POINT_TAG_MM = np.array([35.4, -63.17, 73.899], dtype=float)
arm_gripper_length_z = 142.5 + 2 # mm 末端Z轴朝前伸出方向
arm_gripper_width_y = 164 # mm  ## 夹爪开合方向
arm_gripper_thickness =75 # mm
arm_flange = 10.5+2 # mm
PREPICK_MIN_OFFSET_M = 0.030


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Run directory; uses its pickplace/apriltag.* files and writes output there.",
    )
    parser.add_argument("--image", type=Path, help="RGB image containing the AprilTag.")
    parser.add_argument("--depth", type=Path, help="Aligned depth image saved as .npy.")
    parser.add_argument(
        "--robot-pose",
        type=Path,
        help="Robot end-pose JSON captured with the image.",
    )
    parser.add_argument(
        "--camera-config",
        type=Path,
        default = CAMERA_CONFIG,
        help="RealSense camera_config.npy.",
    )
    parser.add_argument(
        "--hand-eye",
        type=Path,
        default=EYE_HAND,
        help="Eye-in-hand calibration matrix ee_T_camera saved as .npy.",
    )
    parser.add_argument("--output", type=Path, help="Output JSON path.")
    parser.add_argument(
        "--urdf",
        type=Path,
        default = URDF,
        help="Piper URDF path. Required unless --no-ik is used.",
    )
    parser.add_argument("--no-ik", action="store_true", help="Skip IK and output joint_degrees as null.")
    parser.add_argument("--tag-id", type=int, help="Target tag ID; default selects the largest detected tag.")
    parser.add_argument(
        "--tag-family",
        default="tag25h9",
        choices=("tag16h5", "tag25h9", "tag36h10", "tag36h11"),
    )
    parser.add_argument(
        "--tag-size",
        type=float,
        default=0.029,
        help="Printed AprilTag side length in metres.",
    )
    parser.add_argument(
        "--needle-axis",
        default="-y",
        choices=("+x", "-x", "+y", "-y"),
        help="Detected AprilTag in-plane axis pointing toward the needle. Current mounting: +y.",
    )
    parser.add_argument(
        "--depth-window",
        type=int,
        default=7,
        help="Odd median-depth window centred on the AprilTag centre.",
    )
    return parser.parse_args()


def resolve_data_paths(args):
    if args.run_dir is not None:
        pickplace_dir = args.run_dir.expanduser().resolve() / "pickplace"
        args.image = args.image or pickplace_dir / "apriltag.png"
        args.depth = args.depth or pickplace_dir / "apriltag.npy"
        args.robot_pose = args.robot_pose or pickplace_dir / "apriltag.json"
        args.output = args.output or pickplace_dir / "glue_pick_endpose.json"

    missing = [
        name
        for name in ("image", "depth", "robot_pose", "output")
        if getattr(args, name) is None
    ]
    if missing:
        raise ValueError(
            "--run-dir or explicit data paths are required; missing: " + ", ".join(missing)
        )
    return args


def load_npy(path):
    value = np.load(path, allow_pickle=True)
    if isinstance(value, np.ndarray) and value.shape == () and value.dtype == object:
        value = value.item()
    return value


def normalize(vector):
    vector = np.asarray(vector, dtype=float).reshape(3)
    norm = np.linalg.norm(vector)
    if norm <= 1e-12:
        raise ValueError("Cannot normalize a zero vector.")
    return vector / norm


def load_endpose_transform(path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        pose = {}
        for item in raw:
            pose.update(item)
    else:
        pose = raw

    xyz_m = np.array([pose[key] for key in ("x", "y", "z")], dtype=float) / 1_000_000.0
    rpy_deg = np.array([pose[key] for key in ("rx", "ry", "rz")], dtype=float) / 1000.0

    transform = np.eye(4, dtype=float)
    transform[:3, :3] = Rotation.from_euler("xyz", rpy_deg, degrees=True).as_matrix()
    transform[:3, 3] = xyz_m
    return transform


def detect_tag(image, family, wanted_id):
    dictionaries = {
        "tag16h5": cv2.aruco.DICT_APRILTAG_16h5,
        "tag25h9": cv2.aruco.DICT_APRILTAG_25h9,
        "tag36h10": cv2.aruco.DICT_APRILTAG_36h10,
        "tag36h11": cv2.aruco.DICT_APRILTAG_36h11,
    }
    dictionary = cv2.aruco.getPredefinedDictionary(dictionaries[family])
    parameters = cv2.aruco.DetectorParameters()

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners, ids, _ = detector.detectMarkers(image)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(
            image,
            dictionary,
            parameters=parameters,
        )

    if ids is None:
        raise RuntimeError(f"No {family} AprilTag detected.")

    ids = ids.reshape(-1)
    candidates = [
        index
        for index, tag_id in enumerate(ids)
        if wanted_id is None or int(tag_id) == wanted_id
    ]
    if not candidates:
        raise RuntimeError(
            f"AprilTag ID {wanted_id} not detected; detected IDs: {ids.tolist()}"
        )

    index = max(
        candidates,
        key=lambda i: abs(cv2.contourArea(corners[i].reshape(4, 2))),
    )
    tag_corners = corners[index].reshape(4, 2).astype(np.float64)
    center_uv = tag_corners.mean(axis=0)
    return tag_corners, center_uv, int(ids[index])


def camera_matrix_from_intrinsics(intrinsics):
    return np.array(
        [
            [float(intrinsics["fx"]), 0.0, float(intrinsics["ppx"])],
            [0.0, float(intrinsics["fy"]), float(intrinsics["ppy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def distortion_from_intrinsics(intrinsics):
    coeffs = intrinsics.get(
        "coeffs",
        intrinsics.get("dist_coeffs", [0, 0, 0, 0, 0]),
    )
    return np.asarray(coeffs, dtype=np.float64).reshape(-1, 1)


def estimate_camera_R_detected_tag(tag_corners, intrinsics, tag_size_m):
    half = tag_size_m / 2.0

    # OpenCV ArUco corner order:
    # top-left, top-right, bottom-right, bottom-left.
    object_points = np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )

    success, rvec, _ = cv2.solvePnP(
        objectPoints=object_points,
        imagePoints=np.asarray(tag_corners, dtype=np.float64),
        cameraMatrix=camera_matrix_from_intrinsics(intrinsics),
        distCoeffs=distortion_from_intrinsics(intrinsics),
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not success:
        raise RuntimeError("Failed to estimate AprilTag rotation with solvePnP.")

    camera_R_detected_tag, _ = cv2.Rodrigues(rvec)
    return camera_R_detected_tag


def deproject_center(center_uv, depth_raw, intrinsics, depth_scale, window):
    if window < 1 or window % 2 == 0:
        raise ValueError("--depth-window must be a positive odd number.")

    u, v = np.rint(center_uv).astype(int)
    height, width = depth_raw.shape
    radius = window // 2
    patch = depth_raw[
        max(0, v - radius):min(height, v + radius + 1),
        max(0, u - radius):min(width, u + radius + 1),
    ]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if valid.size == 0:
        raise RuntimeError(f"No valid depth around AprilTag centre ({u}, {v}).")

    raw_depth = float(np.median(valid))
    if np.issubdtype(depth_raw.dtype, np.integer):
        z = raw_depth * depth_scale
    else:
        z = raw_depth
        if z > 10.0:
            z *= depth_scale

    x = (float(center_uv[0]) - float(intrinsics["ppx"])) * z / float(intrinsics["fx"])
    y = (float(center_uv[1]) - float(intrinsics["ppy"])) * z / float(intrinsics["fy"])
    return np.array([x, y, z], dtype=float)


def desired_tag_rotation_in_base(base_R_detected_tag, needle_axis):
    axis_map = {
        "+x": np.array([1.0, 0.0, 0.0]),
        "-x": np.array([-1.0, 0.0, 0.0]),
        "+y": np.array([0.0, 1.0, 0.0]),
        "-y": np.array([0.0, -1.0, 0.0]),
    }

    needle_direction_base = base_R_detected_tag @ axis_map[needle_axis]

    # Custom frame:
    # +X points toward the needle and is constrained to base XOY.
    # +Z is base/world +Z.
    x_axis = np.array(
        [needle_direction_base[0], needle_direction_base[1], 0.0],
        dtype=float,
    )
    if np.linalg.norm(x_axis) <= 1e-9:
        raise RuntimeError("Detected needle direction is nearly parallel to base Z.")

    x_axis = normalize(x_axis)
    z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    y_axis = normalize(np.cross(z_axis, x_axis))
    x_axis = normalize(np.cross(y_axis, z_axis))

    return np.column_stack((x_axis, y_axis, z_axis))


def vertical_pick_tool_rotation(base_R_tag):
    tag_x_base = normalize(base_R_tag[:, 0])

    # Gripper/tool +Z points vertically downward.
    tool_z_base = np.array([0.0, 0.0, -1.0], dtype=float)

    # Gripper/tool +X points opposite to station/tag +X.
    tool_x_base = -tag_x_base
    tool_x_base -= np.dot(tool_x_base, tool_z_base) * tool_z_base
    tool_x_base = normalize(tool_x_base)

    tool_y_base = normalize(np.cross(tool_z_base, tool_x_base))
    return np.column_stack((tool_x_base, tool_y_base, tool_z_base))


def transform_to_endpose_mm(transform):
    xyz_mm = transform[:3, 3] * 1000.0
    rpy_deg = Rotation.from_matrix(transform[:3, :3]).as_euler(
        "xyz",
        degrees=True,
    )
    return np.concatenate((xyz_mm, rpy_deg))


def solve_joint_degrees(endpose, urdf):
    if urdf is None:
        raise ValueError("--urdf is required unless --no-ik is used.")
    if not urdf.is_file():
        raise FileNotFoundError(urdf)



    ik.DEFAULT_URDF = str(urdf.resolve())
    ik._MODEL_CACHE = None

    result = ik.reachability_test(endpose)
    if not result["reachable"]:
        raise RuntimeError(
            "Final grasp pose is unreachable: "
            f"position error={result['pos_err_mm']:.3f} mm, "
            f"rotation error={result['rot_err_deg']:.3f} deg"
        )

    return [round(float(value), 3) for value in result["joint_degrees"]]


def define_prepick_endpose(endpose, base_T_tag, urdf):
    """Search joint angles on the ray from endpose along AprilTag +Z."""
    if urdf is None:
        raise ValueError("--urdf is required to search prepick_endpose.")
    if not urdf.is_file():
        raise FileNotFoundError(urdf)

    ik.DEFAULT_URDF = str(urdf.resolve())
    ik._MODEL_CACHE = None
    model = ik.load_arm_model()
    if not model.existFrame(ik.DEFAULT_EE_FRAME):
        raise RuntimeError(f"Cannot find EE frame '{ik.DEFAULT_EE_FRAME}'.")
    frame_id = model.getFrameId(ik.DEFAULT_EE_FRAME)
    lb, ub = ik.get_safe_bounds(model)

    endpose = np.asarray(endpose, dtype=float).reshape(6)
    ray_origin = endpose[:3] / 1000.0
    ray_direction = normalize(np.asarray(base_T_tag, dtype=float)[:3, 2])
    target_rotation = Rotation.from_euler(
        "xyz", endpose[3:], degrees=True
    ).as_matrix()

    rng = np.random.default_rng(ik.RANDOM_SEED)
    seed_qs = [
        np.clip(pin.neutral(model), lb, ub),
        np.clip(np.zeros(model.nq), lb, ub),
    ]
    for _ in range(ik.N_RANDOM_INIT):
        seed_qs.append(rng.uniform(lb, ub))

    best = None
    for q0 in seed_qs:
        data = model.createData()

        def residual(q):
            pose = ik.frame_pose(model, data, q, frame_id)
            distance = float(np.dot(pose.translation - ray_origin, ray_direction))
            distance = max(distance, PREPICK_MIN_OFFSET_M)
            closest_point = ray_origin + distance * ray_direction
            position_error = pose.translation - closest_point
            rotation_error = Rotation.from_matrix(
                target_rotation.T @ pose.rotation
            ).as_rotvec()
            return np.concatenate([
                position_error / ik.POS_SCALE,
                rotation_error / ik.ROT_SCALE,
            ])

        result = least_squares(
            residual,
            q0,
            bounds=(lb, ub),
            max_nfev=1000,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
        )

        q = result.x
        pose = ik.frame_pose(model, data, q, frame_id)
        ray_distance = float(np.dot(pose.translation - ray_origin, ray_direction))
        selected_distance = max(ray_distance, PREPICK_MIN_OFFSET_M)
        selected_position = ray_origin + selected_distance * ray_direction
        position_error_mm = float(np.linalg.norm(pose.translation - selected_position) * 1000.0)
        rotation_error_deg = float(np.rad2deg(np.linalg.norm(
            Rotation.from_matrix(target_rotation.T @ pose.rotation).as_rotvec()
        )))
        feasible = (
            ray_distance >= PREPICK_MIN_OFFSET_M - 1e-6
            and position_error_mm < ik.POS_TOL_MM
            and rotation_error_deg < ik.ROT_TOL_DEG
        )
        joint_margin = float(np.min(np.minimum(q - lb, ub - q)))
        score = (
            (10000.0 if feasible else 0.0)
            - position_error_mm
            - rotation_error_deg
            - selected_distance
            + joint_margin
        )
        if best is None or score > best["score"]:
            best = {
                "score": score,
                "feasible": feasible,
                "position": selected_position,
            }

    if best is None or not best["feasible"]:
        raise RuntimeError("No reachable prepick endpose found on the AprilTag +Z ray.")

    prepick_endpose = np.concatenate([
        best["position"] * 1000.0,
        endpose[3:],
    ])
    prepick_endpose = [round(float(value), 3) for value in prepick_endpose]
    prepick_joint_degrees = solve_joint_degrees(prepick_endpose, urdf)
    return prepick_endpose, prepick_joint_degrees


def validate_input_files(args):
    paths = [
        args.image,
        args.depth,
        args.robot_pose,
        args.camera_config,
        args.hand_eye,
    ]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)


def main():
    args = resolve_data_paths(parse_args())
    validate_input_files(args)

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read image: {args.image}")

    depth = np.asarray(load_npy(args.depth))
    if depth.ndim != 2:
        raise ValueError(f"Depth array must be HxW, got {depth.shape}.")

    camera_config = load_npy(args.camera_config)
    intrinsics = camera_config["color_intrinsic"]

    tag_corners, center_uv, tag_id = detect_tag(
        image=image,
        family=args.tag_family,
        wanted_id=args.tag_id,
    )

    point_camera = deproject_center(
        center_uv=center_uv,
        depth_raw=depth,
        intrinsics=intrinsics,
        depth_scale=float(camera_config["depth_scale"]),
        window=args.depth_window,
    )

    camera_R_detected_tag = estimate_camera_R_detected_tag(
        tag_corners=tag_corners,
        intrinsics=intrinsics,
        tag_size_m=args.tag_size,
    )

    base_T_ee = load_endpose_transform(args.robot_pose)
    ee_T_camera = np.asarray(load_npy(args.hand_eye), dtype=float)
    if ee_T_camera.shape != (4, 4):
        raise ValueError(f"Hand-eye transform must be 4x4, got {ee_T_camera.shape}.")

    base_T_camera = base_T_ee @ ee_T_camera
    tag_origin_base = (base_T_camera @ np.r_[point_camera, 1.0])[:3]
    base_R_detected_tag = base_T_camera[:3, :3] @ camera_R_detected_tag

    base_T_tag = np.eye(4, dtype=float)
    base_T_tag[:3, :3] = desired_tag_rotation_in_base(
        base_R_detected_tag=base_R_detected_tag,
        needle_axis=args.needle_axis,
    )
    base_T_tag[:3, 3] = tag_origin_base

    base_T_tool = np.eye(4, dtype=float)
    base_T_tool[:3, :3] = vertical_pick_tool_rotation(base_T_tag[:3, :3])
    base_T_tool[:3, 3] = (
        base_T_tag[:3, 3]
        + base_T_tag[:3, :3] @ (GRASP_POINT_TAG_MM / 1000.0)
    )

    endpose = [
        round(float(value), 3)
        for value in transform_to_endpose_mm(base_T_tool)
    ]
    endpose[2] +=  arm_gripper_length_z

    joint_degrees = None
    prepick_endpose = None
    prepick_joint_degrees = None
    if not args.no_ik:
        joint_degrees = solve_joint_degrees(endpose, args.urdf)
        prepick_endpose, prepick_joint_degrees = define_prepick_endpose(
            endpose=endpose,
            base_T_tag=base_T_tag,
            urdf=args.urdf,
        )

    output = {
        "tag_id": int(tag_id),
        "needle_axis_in_detected_tag": args.needle_axis,
        "base_T_tag": np.round(base_T_tag, 9).tolist(),
        "base_R_tag": np.round(base_T_tag[:3, :3], 9).tolist(),
        "tag_origin_base_m": np.round(base_T_tag[:3, 3], 9).tolist(),
        "grasp_point_tag_mm": GRASP_POINT_TAG_MM.tolist(),
        "prepick_endpose": prepick_endpose,
        "prepick_joint_degrees": prepick_joint_degrees,
        "endpose": endpose,
        "joint_degrees": joint_degrees,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    tag_pose = transform_to_endpose_mm(base_T_tag)
    print(f"AprilTag {tag_id} base pose [mm, deg]: {np.round(tag_pose, 3).tolist()}")
    print("base_R_tag:")
    print(np.round(base_T_tag[:3, :3], 6))
    print(f"Detected tag axis pointing to needle: {args.needle_axis}")
    print(f"Pre-pick endpose [mm, deg]: {prepick_endpose}")
    print(f"Pre-pick joint degrees: {prepick_joint_degrees}")
    print(f"Grasp endpose [mm, deg]: {endpose}")
    print(f"Joint degrees: {joint_degrees}")
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()

"""Calibrate printer plate pose with RGB-D AprilTag positioning.

Tag origin is obtained from aligned depth. AprilTag PnP is used only for the
in-plane +X direction. The tag frame is constrained to the horizontal printer
plate: tag +Z = arm-base +Z.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from Piper import endpose_reachability_safe as ik


DEFAULT_IMAGE = REPO_ROOT / "data/plate_calibration.png"
DEFAULT_DEPTH = REPO_ROOT / "data/plate_calibration.npy"
DEFAULT_CAPTURE_POSE = REPO_ROOT / "data/plate_calibration.json"
DEFAULT_CAMERA_CONFIG = REPO_ROOT / "config/calibration/right_camera/camera_config.npy"
DEFAULT_ECT = REPO_ROOT / "config/calibration/right_camera/ecT.npy"
DEFAULT_URDF = REPO_ROOT / "config/piper/piper_description.urdf"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/plate_calibration"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--depth", type=Path, default=DEFAULT_DEPTH)
    parser.add_argument("--capture-pose", type=Path, default=DEFAULT_CAPTURE_POSE)
    parser.add_argument("--camera-config", type=Path, default=DEFAULT_CAMERA_CONFIG)
    parser.add_argument("--ect", type=Path, default=DEFAULT_ECT)
    parser.add_argument("--tag-size-mm", type=float, default=29.0)
    parser.add_argument("--tag-id", type=int, default=0)
    parser.add_argument(
        "--plate-center-mm",
        type=float,
        nargs=3,
        default=[78.0, 70.0, 0.0],
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument("--validation-height-mm", type=float, default=180.0)
    parser.add_argument("--depth-window", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--no-ik", action="store_true")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def load_camera(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    config = np.load(path, allow_pickle=True).item()
    intrinsic = config["color_intrinsic"]
    camera_matrix = np.array(
        [
            [intrinsic["fx"], 0.0, intrinsic["ppx"]],
            [0.0, intrinsic["fy"], intrinsic["ppy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.asarray(
        intrinsic.get("coeffs", [0, 0, 0, 0, 0]), dtype=np.float64
    ).reshape(-1)
    return camera_matrix, dist_coeffs, config


def load_end_to_base(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    values = {next(iter(item)): float(next(iter(item.values()))) for item in raw}
    endpose = np.array(
        [
            values["x"] / 1000.0,
            values["y"] / 1000.0,
            values["z"] / 1000.0,
            values["rx"] / 1000.0,
            values["ry"] / 1000.0,
            values["rz"] / 1000.0,
        ]
    )
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler(
        "xyz", endpose[3:], degrees=True
    ).as_matrix()
    transform[:3, 3] = endpose[:3] / 1000.0
    return transform, endpose


def detect_tag(image: np.ndarray, tag_id: int) -> tuple[np.ndarray, np.ndarray]:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_25h9)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.cornerRefinementWinSize = 5
    corners, ids, _ = cv2.aruco.ArucoDetector(
        dictionary, parameters
    ).detectMarkers(image)
    index = int(np.flatnonzero(ids.reshape(-1) == tag_id)[0])
    tag_corners = corners[index].reshape(4, 2).astype(np.float64)
    return tag_corners, tag_corners.mean(axis=0)


def tag_object_corners(tag_size_m: float) -> np.ndarray:
    half = tag_size_m / 2.0
    return np.array(
        [
            [-half, +half, 0.0],
            [+half, +half, 0.0],
            [+half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )


def estimate_camera_r_tag(
    image_corners: np.ndarray,
    tag_size_m: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    object_corners = tag_object_corners(tag_size_m)
    _, rvec, tvec = cv2.solvePnP(
        object_corners,
        image_corners,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    projected, _ = cv2.projectPoints(
        object_corners, rvec, tvec, camera_matrix, dist_coeffs
    )
    projected = projected.reshape(4, 2)
    rmse = float(
        np.sqrt(np.mean(np.sum((projected - image_corners) ** 2, axis=1)))
    )
    return cv2.Rodrigues(rvec)[0], projected, rmse


def deproject_tag_center(
    center_uv: np.ndarray,
    depth: np.ndarray,
    camera_matrix: np.ndarray,
    depth_scale: float,
    window: int,
) -> np.ndarray:
    u, v = np.rint(center_uv).astype(int)
    radius = window // 2
    patch = depth[v - radius:v + radius + 1, u - radius:u + radius + 1]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    z = float(np.median(valid))
    if np.issubdtype(depth.dtype, np.integer) or z > 10.0:
        z *= depth_scale

    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    return np.array(
        [
            (center_uv[0] - cx) * z / fx,
            (center_uv[1] - cy) * z / fy,
            z,
        ]
    )


def normalize(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def constrained_tag_rotation(base_r_detected_tag: np.ndarray) -> np.ndarray:
    x_axis = base_r_detected_tag[:, 0].copy()
    x_axis[2] = 0.0
    x_axis = normalize(x_axis)
    z_axis = np.array([0.0, 0.0, 1.0])
    y_axis = normalize(np.cross(z_axis, x_axis))
    x_axis = normalize(np.cross(y_axis, z_axis))
    return np.column_stack((x_axis, y_axis, z_axis))


def downward_rotation(reference_rotation: np.ndarray) -> np.ndarray:
    z_axis = np.array([0.0, 0.0, -1.0])
    x_axis = reference_rotation[:, 0].copy()
    x_axis[2] = 0.0
    x_axis = normalize(x_axis)
    y_axis = normalize(np.cross(z_axis, x_axis))
    x_axis = normalize(np.cross(y_axis, z_axis))
    return np.column_stack((x_axis, y_axis, z_axis))


def transform_to_endpose(transform: np.ndarray) -> np.ndarray:
    return np.r_[
        transform[:3, 3] * 1000.0,
        Rotation.from_matrix(transform[:3, :3]).as_euler("xyz", degrees=True),
    ]


def solve_validation_ik(
    position_m: np.ndarray,
    nominal_rotation: np.ndarray,
    urdf: Path,
) -> dict[str, Any]:
    ik.DEFAULT_URDF = str(urdf.resolve())
    ik._MODEL_CACHE = None

    best = None
    for yaw_deg in np.arange(-180.0, 180.0, 15.0):
        yaw = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()
        transform = np.eye(4)
        transform[:3, :3] = nominal_rotation @ yaw
        transform[:3, 3] = position_m
        endpose = transform_to_endpose(transform)
        result = ik.reachability_test(endpose)
        score = result["pos_err_mm"] + result["rot_err_deg"]
        if best is None or score < best["score"]:
            best = {
                "score": score,
                "yaw_offset_deg": float(yaw_deg),
                "endpose": endpose,
                "result": result,
            }

    return {
        "reachable": bool(best["result"]["reachable"]),
        "yaw_offset_deg": best["yaw_offset_deg"],
        "target_endpose": best["endpose"],
        "joint_degrees": best["result"].get("joint_degrees"),
        "pos_err_mm": best["result"]["pos_err_mm"],
        "rot_err_deg": best["result"]["rot_err_deg"],
    }


def project_point(
    point_tag: np.ndarray,
    camera_r_tag: np.ndarray,
    camera_p_tag: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    pixel, _ = cv2.projectPoints(
        np.asarray(point_tag).reshape(1, 3),
        cv2.Rodrigues(camera_r_tag)[0],
        camera_p_tag,
        camera_matrix,
        dist_coeffs,
    )
    return pixel.reshape(2)


def make_debug_image(
    image: np.ndarray,
    image_corners: np.ndarray,
    projected_corners: np.ndarray,
    camera_r_tag: np.ndarray,
    camera_p_tag: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    center_tag: np.ndarray,
    validation_tag: np.ndarray,
    tag_size_m: float,
    tag_id: int,
    rmse: float,
) -> np.ndarray:
    output = image.copy()
    cv2.polylines(
        output,
        [np.rint(image_corners).astype(np.int32)],
        True,
        (0, 255, 0),
        2,
    )
    for point in projected_corners:
        cv2.drawMarker(
            output,
            tuple(np.rint(point).astype(int)),
            (255, 255, 255),
            cv2.MARKER_CROSS,
            9,
            1,
        )

    rvec = cv2.Rodrigues(camera_r_tag)[0]
    cv2.drawFrameAxes(
        output,
        camera_matrix,
        dist_coeffs,
        rvec,
        camera_p_tag,
        max(0.01, tag_size_m),
        2,
    )

    center_pixel = project_point(
        center_tag, camera_r_tag, camera_p_tag, camera_matrix, dist_coeffs
    )
    validation_pixel = project_point(
        validation_tag, camera_r_tag, camera_p_tag, camera_matrix, dist_coeffs
    )
    cv2.drawMarker(
        output,
        tuple(np.rint(center_pixel).astype(int)),
        (255, 0, 255),
        cv2.MARKER_CROSS,
        18,
        2,
    )
    cv2.drawMarker(
        output,
        tuple(np.rint(validation_pixel).astype(int)),
        (255, 255, 0),
        cv2.MARKER_TILTED_CROSS,
        18,
        2,
    )
    cv2.putText(
        output,
        f"tag25h9 id={tag_id}, RMSE={rmse:.3f}px",
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )
    return output


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    depth = np.load(args.depth)
    camera_matrix, dist_coeffs, camera_config = load_camera(args.camera_config)
    base_t_end_capture, capture_endpose = load_end_to_base(args.capture_pose)
    end_t_camera = np.asarray(np.load(args.ect), dtype=float)

    image_corners, center_uv = detect_tag(image, args.tag_id)
    camera_r_detected_tag, projected_corners, reprojection_rmse = (
        estimate_camera_r_tag(
            image_corners,
            args.tag_size_mm / 1000.0,
            camera_matrix,
            dist_coeffs,
        )
    )
    camera_p_tag = deproject_tag_center(
        center_uv,
        depth,
        camera_matrix,
        float(camera_config["depth_scale"]),
        args.depth_window,
    )

    base_t_camera = base_t_end_capture @ end_t_camera
    tag_origin_base = (base_t_camera @ np.r_[camera_p_tag, 1.0])[:3]
    base_r_detected_tag = base_t_camera[:3, :3] @ camera_r_detected_tag
    base_r_tag = constrained_tag_rotation(base_r_detected_tag)

    base_t_tag = np.eye(4)
    base_t_tag[:3, :3] = base_r_tag
    base_t_tag[:3, 3] = tag_origin_base

    center_tag = np.asarray(args.plate_center_mm, dtype=float) / 1000.0
    validation_tag = center_tag + np.array(
        [0.0, 0.0, args.validation_height_mm / 1000.0]
    )
    center_base = (base_t_tag @ np.r_[center_tag, 1.0])[:3]
    validation_base = center_base + np.array(
        [0.0, 0.0, args.validation_height_mm / 1000.0]
    )

    validation_rotation = downward_rotation(base_t_end_capture[:3, :3])
    validation_transform = np.eye(4)
    validation_transform[:3, :3] = validation_rotation
    validation_transform[:3, 3] = validation_base
    validation_endpose = transform_to_endpose(validation_transform)

    ik_result = None
    if not args.no_ik:
        ik_result = solve_validation_ik(
            validation_base,
            validation_rotation,
            args.urdf,
        )
        validation_endpose = np.asarray(ik_result["target_endpose"], dtype=float)
        validation_transform[:3, :3] = Rotation.from_euler(
            "xyz", validation_endpose[3:], degrees=True
        ).as_matrix()

    camera_r_tag = base_t_camera[:3, :3].T @ base_r_tag
    debug = make_debug_image(
        image,
        image_corners,
        projected_corners,
        camera_r_tag,
        camera_p_tag,
        camera_matrix,
        dist_coeffs,
        center_tag,
        validation_tag,
        args.tag_size_mm / 1000.0,
        args.tag_id,
        reprojection_rmse,
    )
    cv2.imwrite(str(output_dir / "plate_apriltag_axes_rgbd.png"), debug)

    joint_degrees = None if ik_result is None else ik_result["joint_degrees"]

    np.save(output_dir / "camera_P_tag_rgbd.npy", camera_p_tag)
    np.save(output_dir / "arm_base_P_tag_center.npy", tag_origin_base)
    np.save(output_dir / "arm_base_T_tag.npy", base_t_tag)
    np.save(output_dir / "arm_base_P_plate_center.npy", center_base)
    np.save(output_dir / "arm_base_T_end_validation.npy", validation_transform)
    np.save(output_dir / "validation_endpose.npy", validation_endpose)
    if joint_degrees is not None:
        np.save(output_dir / "validation_joint_degrees.npy", joint_degrees)

    result = {
        "method": "RGB-D tag origin + constrained horizontal tag frame",
        "tag_family": "tag25h9",
        "tag_id": args.tag_id,
        "tag_size_mm": args.tag_size_mm,
        "image_corners_px": image_corners,
        "tag_center_pixel": center_uv,
        "reprojection_rmse_px": reprojection_rmse,
        "camera_P_tag_rgbd_m": camera_p_tag,
        "camera_R_detected_tag": camera_r_detected_tag,
        "capture_endpose_mm_deg": capture_endpose,
        "end_T_camera": end_t_camera,
        "arm_base_T_camera": base_t_camera,
        "arm_base_P_tag_center_m": tag_origin_base,
        "arm_base_T_tag": base_t_tag,
        "tag_frame_constraint": {
            "x_axis": "detected tag +X projected onto arm-base XOY",
            "y_axis": "cross(arm-base +Z, constrained tag +X)",
            "z_axis": "arm-base +Z",
        },
        "tag_P_plate_center_m": center_tag,
        "arm_base_P_plate_center_m": center_base,
        "arm_base_P_validation_m": validation_base,
        "validation_endpose_mm_deg": validation_endpose,
        "validation_joint_degrees": joint_degrees,
        "ik": ik_result,
    }
    (output_dir / "plate_apriltag_calibration_rgbd.json").write_text(
        json.dumps(json_ready(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Corner reprojection RMSE: {reprojection_rmse:.3f} px")
    print(f"Tag center in camera [m]: {camera_p_tag}")
    print(f"Tag center in arm base [m]: {tag_origin_base}")
    print(f"Plate center in arm base [m]: {center_base}")
    print(f"Validation point in arm base [m]: {validation_base}")
    print(f"Validation endpose [mm, deg]: {np.round(validation_endpose, 4)}")
    print(f"Validation joint degrees: {joint_degrees}")
    print(f"Results written to {output_dir}")

    if args.show:
        cv2.imshow("Plate calibration RGB-D", debug)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

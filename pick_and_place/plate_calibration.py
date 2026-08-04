"""Calibrate the printer plate from a tag25h9 AprilTag.

Tag frame convention (right handed):
    origin: centre of the printed AprilTag's outer black square
    +X: canonical marker right
    +Y: canonical marker up
    +Z: outward from the plate

The marker must be installed so the plate centre is at
tag_P_plate_center = [90, 90, 0] mm.  The validation pose moves its origin to
tag_P_plate_center + [0, 0, 200] mm and constrains EE +Z to arm-base -Z.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from Piper import endpose_reachability_safe as ik

DEFAULT_IMAGE = REPO_ROOT / "data/plate_calibration_7.png"
DEFAULT_DEPTH = REPO_ROOT / "data/plate_calibration.npy"
DEFAULT_CAPTURE_POSE = REPO_ROOT / "data/plate_calibration.json"
DEFAULT_CAMERA_CONFIG = REPO_ROOT / "config/calibration/right_camera/camera_config.npy"
DEFAULT_ECT = REPO_ROOT / "config/calibration/right_camera/ecT.npy"
DEFAULT_URDF = REPO_ROOT / "config/piper/piper_description.urdf"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/plate_calibration"
STRICT_POS_TOL_MM = 0.05
STRICT_ROT_TOL_DEG = 0.05
STRICT_POS_SCALE_M = STRICT_POS_TOL_MM / 1000.0
STRICT_ROT_SCALE_RAD = np.deg2rad(STRICT_ROT_TOL_DEG)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--depth", type=Path, default=DEFAULT_DEPTH)
    parser.add_argument("--capture-pose", type=Path, default=DEFAULT_CAPTURE_POSE)
    parser.add_argument("--camera-config", type=Path, default=DEFAULT_CAMERA_CONFIG)
    parser.add_argument(
        "--ect",
        type=Path,
        default=DEFAULT_ECT,
        help="4x4 end_T_camera hand-eye transform.",
    )
    parser.add_argument(
        "--tag-size-mm",
        type=float,
        default=29,
        help="Measured outer black-square side length; tag25h9 is only a family name.",
    )
    parser.add_argument("--tag-id", type=int, default=0)
    parser.add_argument(
        "--plate-center-mm",
        type=float,
        nargs=3,
        default=[79.0, 71.0, 0.0],
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument("--validation-height-mm", type=float, default=190.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument(
        "--current-joints-deg",
        type=float,
        nargs=6,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help=(
            "Captured arm joint angles used as the primary IK seed. If omitted, "
            "they are approximated by solving IK for the captured endpose."
        ),
    )
    parser.add_argument(
        "--no-ik",
        action="store_true",
        help="Compute poses but skip Pinocchio IK and joint degrees.",
    )
    parser.add_argument(
        "--max-reprojection-error-px",
        type=float,
        default=2.0,
        help="Reject the tag pose if corner reprojection RMSE exceeds this value.",
    )
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
    coefficients = np.asarray(intrinsic.get("coeffs", []), dtype=np.float64).reshape(-1)
    if len(coefficients) not in (4, 5, 8, 12, 14):
        coefficients = np.zeros(5, dtype=np.float64)
    return camera_matrix, coefficients, config


def load_end_to_base(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8") as file:
        raw = json.load(file)
    values = {next(iter(item)): float(next(iter(item.values()))) for item in raw}
    current_endpose = np.array(
        [
            values["x"] / 1000.0,
            values["y"] / 1000.0,
            values["z"] / 1000.0,
            values["rx"] / 1000.0,
            values["ry"] / 1000.0,
            values["rz"] / 1000.0,
        ],
        dtype=float,
    )
    base_t_end = np.eye(4, dtype=float)
    base_t_end[:3, :3] = Rotation.from_euler(
        "xyz", current_endpose[3:], degrees=True
    ).as_matrix()
    base_t_end[:3, 3] = current_endpose[:3] / 1000.0  # 转换成m
    return base_t_end, current_endpose


def detect_tag(
    image: np.ndarray, requested_id: int
) -> tuple[np.ndarray, int, list[np.ndarray]]:
    if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "DICT_APRILTAG_25h9"):
        raise RuntimeError("OpenCV with the aruco module and DICT_APRILTAG_25h9 is required.")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_25h9)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.cornerRefinementWinSize = 5
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    corners, ids, rejected = detector.detectMarkers(image)
    if ids is None:
        raise RuntimeError("No tag25h9 AprilTag detected.")
    ids_flat = ids.reshape(-1)
    matches = np.flatnonzero(ids_flat == requested_id)
    if len(matches) == 0:
        raise RuntimeError(
            f"Requested tag ID {requested_id} was not found; detected IDs: {ids_flat.tolist()}"
        )
    index = int(matches[0])
    return corners[index].reshape(4, 2).astype(np.float64), int(ids_flat[index]), rejected


def tag_object_corners(tag_size_m: float) -> np.ndarray:
    """IPPE_SQUARE order: top-left, top-right, bottom-right, bottom-left.

    This coordinate choice makes +X canonical-right, +Y canonical-up and +Z
    outward from the visible marker face.
    """
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


def estimate_camera_t_tag(
    image_corners: np.ndarray,
    tag_size_m: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    object_corners = tag_object_corners(tag_size_m)

    result = cv2.solvePnPGeneric(
        object_corners,
        image_corners,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    success, rvecs, tvecs = result[:3]

    candidates = []
    for index, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
        rotation = cv2.Rodrigues(rvec)[0]
        translation = np.asarray(tvec, dtype=float).reshape(3)

        projected, _ = cv2.projectPoints(
            object_corners, rvec, tvec, camera_matrix, dist_coeffs
        )
        projected = projected.reshape(4, 2)

        rmse = float(
            np.sqrt(np.mean(np.sum((projected - image_corners) ** 2, axis=1)))
        )

        transform = np.eye(4, dtype=float)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation

        candidates.append(
            {
                "index": index,
                "camera_T_tag": transform,
                "projected_corners": projected,
                "rmse_px": rmse,
                "depth_m": float(translation[2]),
                "euler_xyz_deg": Rotation.from_matrix(rotation).as_euler(
                    "xyz", degrees=True
                ),
            }
        )

    valid = [item for item in candidates if item["depth_m"] > 0]

    valid.sort(key=lambda item: item["rmse_px"])
    best = valid[0]

    pnp_check = {
        "selected_index": best["index"],
        "candidate_count": len(candidates),
        "candidates": [
            {
                "index": item["index"],
                "rmse_px": item["rmse_px"],
                "depth_m": item["depth_m"],
                "euler_xyz_deg": item["euler_xyz_deg"],
                "camera_T_tag": item["camera_T_tag"],
            }
            for item in candidates
        ],
    }

    if len(valid) >= 2:
        first, second = valid[:2]
        z1 = first["camera_T_tag"][:3, 2]
        z2 = second["camera_T_tag"][:3, 2]

        pnp_check["rmse_gap_px"] = second["rmse_px"] - first["rmse_px"]
        pnp_check["normal_angle_deg"] = float(
            np.degrees(
                np.arccos(np.clip(np.dot(z1, z2), -1.0, 1.0))
            )
        )

    return (
        best["camera_T_tag"],
        best["projected_corners"],
        best["rmse_px"],
        pnp_check,
    )

def transform_to_endpose(transform: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (
            transform[:3, 3] * 1000.0,
            Rotation.from_matrix(transform[:3, :3]).as_euler("xyz", degrees=True),
        )
    )


def downward_validation_rotation(current_rotation: np.ndarray) -> np.ndarray:
    """Make EE +Z equal base -Z while retaining the current horizontal yaw."""
    ee_z = np.array([0.0, 0.0, -1.0], dtype=float)
    ee_x = np.asarray(current_rotation, dtype=float)[:, 0].copy()
    ee_x[2] = 0.0
    if np.linalg.norm(ee_x) <= 1e-9:
        # Degenerate only when the captured EE +X is vertical.
        ee_x = np.asarray(current_rotation, dtype=float)[:, 1].copy()
        ee_x[2] = 0.0
    if np.linalg.norm(ee_x) <= 1e-9:
        ee_x = np.array([1.0, 0.0, 0.0], dtype=float)
    ee_x /= np.linalg.norm(ee_x)
    ee_y = np.cross(ee_z, ee_x)
    ee_y /= np.linalg.norm(ee_y)
    ee_x = np.cross(ee_y, ee_z)
    return np.column_stack((ee_x, ee_y, ee_z))


def project_point(
    point_tag: np.ndarray,
    camera_t_tag: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    rvec = cv2.Rodrigues(camera_t_tag[:3, :3])[0]
    point, _ = cv2.projectPoints(
        np.asarray(point_tag, dtype=float).reshape(1, 3),
        rvec,
        camera_t_tag[:3, 3],
        camera_matrix,
        dist_coeffs,
    )
    return point.reshape(2)


def sample_depth_m(depth: np.ndarray, pixel: np.ndarray, depth_scale: float) -> float | None:
    """Median depth around a pixel, used only as an independent PnP sanity check."""
    u, v = np.rint(pixel).astype(int)
    y0, y1 = max(0, v - 2), min(depth.shape[0], v + 3)
    x0, x1 = max(0, u - 2), min(depth.shape[1], u + 3)
    patch = np.asarray(depth[y0:y1, x0:x1], dtype=float)
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if len(valid) == 0:
        return None
    value = float(np.median(valid))
    return value * depth_scale if np.issubdtype(depth.dtype, np.integer) or value > 20 else value


def make_debug_image(
    image: np.ndarray,
    image_corners: np.ndarray,
    projected_corners: np.ndarray,
    camera_t_tag: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    axis_length_m: float,
    center_tag: np.ndarray,
    validation_tag: np.ndarray,
    tag_id: int,
    rmse: float,
) -> np.ndarray:
    output = image.copy()
    cv2.polylines(
        output, [np.rint(image_corners).astype(np.int32)], True, (0, 255, 0), 2
    )
    for index, point in enumerate(image_corners):
        pixel = tuple(np.rint(point).astype(int))
        cv2.circle(output, pixel, 4, (0, 255, 255), -1)
        cv2.putText(
            output, str(index), (pixel[0] + 5, pixel[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1,
        )
    for point in projected_corners:
        cv2.drawMarker(
            output, tuple(np.rint(point).astype(int)), (255, 255, 255),
            cv2.MARKER_CROSS, 9, 1,
        )

    rotation_vector = cv2.Rodrigues(camera_t_tag[:3, :3])[0]
    cv2.drawFrameAxes(
        output,
        camera_matrix,
        dist_coeffs,
        rotation_vector,
        camera_t_tag[:3, 3],
        axis_length_m,
        2,
    )
    center_pixel = project_point(
        center_tag, camera_t_tag, camera_matrix, dist_coeffs
    )
    validation_pixel = project_point(
        validation_tag, camera_t_tag, camera_matrix, dist_coeffs
    )
    cv2.drawMarker(
        output, tuple(np.rint(center_pixel).astype(int)), (255, 0, 255),
        cv2.MARKER_CROSS, 18, 2,
    )
    cv2.putText(
        output, "plate center", tuple(np.rint(center_pixel + [7, -7]).astype(int)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2,
    )
    cv2.drawMarker(
        output, tuple(np.rint(validation_pixel).astype(int)), (255, 255, 0),
        cv2.MARKER_TILTED_CROSS, 18, 2,
    )
    cv2.putText(
        output, f"tag25h9 id={tag_id}, RMSE={rmse:.2f}px", (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
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


def _solve_one_target(
    model: Any,
    target: Any,
    initial_guesses: list[np.ndarray],
    max_nfev: int,
) -> dict[str, Any]:
    data = model.createData()
    frame_id = model.getFrameId(ik.DEFAULT_EE_FRAME)
    lower, upper = ik.get_safe_bounds(model)

    def residual(q: np.ndarray) -> np.ndarray:
        actual = ik.frame_pose(model, data, q, frame_id)
        position_error = actual.translation - target.translation
        rotation_error = ik.pin.log3(target.rotation.T @ actual.rotation)
        return np.concatenate(
            (
                position_error / STRICT_POS_SCALE_M,
                rotation_error / STRICT_ROT_SCALE_RAD,
            )
        )

    best = None
    for initial in initial_guesses:
        initial = np.clip(np.asarray(initial, dtype=float), lower, upper)
        solution = least_squares(
            residual,
            initial,
            bounds=(lower, upper),
            max_nfev=max_nfev,
            xtol=1e-13,
            ftol=1e-13,
            gtol=1e-13,
        )
        q = solution.x
        actual = ik.frame_pose(model, data, q, frame_id)
        position_error_m = float(
            np.linalg.norm(actual.translation - target.translation)
        )
        rotation_error_rad = float(
            np.linalg.norm(ik.pin.log3(target.rotation.T @ actual.rotation))
        )
        score = max(
            position_error_m / STRICT_POS_SCALE_M,
            rotation_error_rad / STRICT_ROT_SCALE_RAD,
        )
        candidate = {
            "q": q,
            "actual": actual.copy(),
            "pos_err_mm": position_error_m * 1000.0,
            "rot_err_deg": float(np.rad2deg(rotation_error_rad)),
            "normalized_max_error": float(score),
            "least_squares_success": bool(solution.success),
            "least_squares_message": str(solution.message),
        }
        if best is None or candidate["normalized_max_error"] < best["normalized_max_error"]:
            best = candidate
    if best is None:
        raise RuntimeError("Strict IK did not produce any candidate.")
    return best


def solve_strict_yaw_ik(
    target_position_m: np.ndarray,
    nominal_rotation: np.ndarray,
    capture_endpose: np.ndarray,
    current_joints_deg: list[float] | None,
    urdf: Path,
) -> dict[str, Any]:
    """Search downward-facing yaw and accept only a near-exact IK solution."""
    ik.DEFAULT_URDF = str(urdf.expanduser().resolve())
    ik._MODEL_CACHE = None
    model = ik.load_arm_model()
    if not model.existFrame(ik.DEFAULT_EE_FRAME):
        raise RuntimeError(f"URDF has no end frame {ik.DEFAULT_EE_FRAME!r}.")
    lower, upper = ik.get_safe_bounds(model)

    if current_joints_deg is not None:
        q_current = np.deg2rad(np.asarray(current_joints_deg, dtype=float))
        if q_current.shape != (model.nq,):
            raise ValueError(
                f"--current-joints-deg has 6 values but the arm model has nq={model.nq}."
            )
        q_current = np.clip(q_current, lower, upper)
        seed_source = "provided_current_joints"
    else:
        capture_target = ik.endpose_to_se3(capture_endpose)
        capture_solution = ik.solve_ik(model, capture_target)
        q_current = np.asarray(capture_solution["q"], dtype=float)
        seed_source = "captured_endpose_ik_approximation"

    neutral = np.clip(ik.pin.neutral(model), lower, upper)
    zero = np.clip(np.zeros(model.nq), lower, upper)
    rng = np.random.default_rng(0)
    random_seeds = [rng.uniform(lower, upper) for _ in range(4)]
    base_seeds = [q_current, neutral, zero, *random_seeds]

    def target_at_yaw(yaw_deg: float) -> Any:
        yaw_local = Rotation.from_rotvec(
            np.deg2rad(yaw_deg) * np.array([0.0, 0.0, 1.0])
        ).as_matrix()
        return ik.pin.SE3(
            nominal_rotation @ yaw_local,
            np.asarray(target_position_m, dtype=float),
        )

    best = None
    # Coarse full-circle search. Post-rotation about local Z preserves EE +Z=-Base Z.
    for yaw_deg in np.arange(-180.0, 180.0, 15.0):
        candidate = _solve_one_target(
            model, target_at_yaw(float(yaw_deg)), base_seeds, max_nfev=1500
        )
        candidate["yaw_offset_deg"] = float(yaw_deg)
        if best is None or candidate["normalized_max_error"] < best["normalized_max_error"]:
            best = candidate

    assert best is not None
    # Fine yaw search around the best coarse orientation, seeded by both the
    # current configuration and the best coarse joint solution.
    fine_best = best
    fine_angles = np.arange(
        best["yaw_offset_deg"] - 15.0,
        best["yaw_offset_deg"] + 15.0 + 0.5,
        1.0,
    )
    for yaw_deg in fine_angles:
        wrapped_yaw = float((yaw_deg + 180.0) % 360.0 - 180.0)
        candidate = _solve_one_target(
            model,
            target_at_yaw(wrapped_yaw),
            [fine_best["q"], q_current, neutral],
            max_nfev=2500,
        )
        candidate["yaw_offset_deg"] = wrapped_yaw
        if candidate["normalized_max_error"] < fine_best["normalized_max_error"]:
            fine_best = candidate

    # Final high-precision solve at the selected yaw.
    selected_target = target_at_yaw(fine_best["yaw_offset_deg"])
    final = _solve_one_target(
        model,
        selected_target,
        [fine_best["q"], q_current, neutral, zero],
        max_nfev=5000,
    )
    final["yaw_offset_deg"] = fine_best["yaw_offset_deg"]
    final["seed_source"] = seed_source
    final["target"] = selected_target
    accurate = (
        final["pos_err_mm"] < STRICT_POS_TOL_MM
        and final["rot_err_deg"] < STRICT_ROT_TOL_DEG
    )
    path_ok = ik.check_joint_path(model, q_current, final["q"])
    final["reachable"] = bool(accurate)
    final["path_ok"] = bool(path_ok)
    final["target_endpose"] = ik.se3_to_endpose(selected_target)
    final["achieved_endpose"] = ik.se3_to_endpose(final["actual"])
    final["q_solution_rad"] = final["q"]
    final["q_solution_deg"] = np.rad2deg(final["q"])
    # Never expose an executable joint target when strict accuracy or the
    # joint-limit path check fails.
    final["joint_degrees"] = (
        np.rad2deg(final["q"]).tolist() if accurate and path_ok else None
    )
    final["position_tolerance_mm"] = STRICT_POS_TOL_MM
    final["rotation_tolerance_deg"] = STRICT_ROT_TOL_DEG
    final.pop("q")
    final.pop("actual")
    final.pop("target")
    return final


def main() -> None:
    args = parse_args()
    for path in (
        args.image,
        args.depth,
        args.capture_pose,
        args.camera_config,
        args.ect,
    ):
        if not path.expanduser().is_file():
            raise FileNotFoundError(path)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"Failed to read {args.image}")
    depth = np.load(args.depth)
    camera_matrix, dist_coeffs, camera_config = load_camera(args.camera_config)
    base_t_end_capture, capture_endpose = load_end_to_base(args.capture_pose)
    end_t_camera = np.asarray(np.load(args.ect), dtype=float)

    image_corners, tag_id, _ = detect_tag(image, args.tag_id)
    camera_t_tag, projected_corners, reprojection_rmse, pnp_check = (
        estimate_camera_t_tag(
            image_corners,
            args.tag_size_mm / 1000.0,
            camera_matrix,
            dist_coeffs,
        )
    )
    if reprojection_rmse > args.max_reprojection_error_px:
        raise RuntimeError(
            f"Tag reprojection RMSE {reprojection_rmse:.3f}px exceeds "
            f"{args.max_reprojection_error_px:.3f}px."
        )

    base_t_camera = base_t_end_capture @ end_t_camera
    base_t_tag = base_t_camera @ camera_t_tag
    center_tag = np.asarray(args.plate_center_mm, dtype=float) / 1000.0
    validation_tag = center_tag + np.array(
        [0.0, 0.0, args.validation_height_mm / 1000.0]
    )
    center_tag_h = np.append(center_tag, 1.0)
    validation_tag_h = np.append(validation_tag, 1.0)
    center_base = (base_t_tag @ center_tag_h)[:3]
    #validation_base = (base_t_tag @ validation_tag_h)[:3]
    validation_base = center_base + np.array([0, 0, args.validation_height_mm / 1000.0])

    # Point the end-frame +Z vertically downward (base -Z). Preserve only the
    # captured horizontal yaw to avoid introducing an arbitrary wrist angle.
    base_t_end_validation = np.eye(4, dtype=float)
    base_t_end_validation[:3, :3] = downward_validation_rotation(
        base_t_end_capture[:3, :3]
    )
    base_t_end_validation[:3, 3] = validation_base
    validation_endpose = transform_to_endpose(base_t_end_validation)

    ik_result = (
        None
        if args.no_ik
        else solve_strict_yaw_ik(
            validation_base,
            base_t_end_validation[:3, :3],
            capture_endpose,
            args.current_joints_deg,
            args.urdf,
        )
    )
    if ik_result is not None:
        # The strict solver may change only yaw. Save its selected target pose,
        # not the nominal pre-search orientation.
        validation_endpose = np.asarray(ik_result["target_endpose"], dtype=float)
        base_t_end_validation[:3, :3] = Rotation.from_euler(
            "xyz", validation_endpose[3:], degrees=True
        ).as_matrix()
    joint_degrees = None if ik_result is None else ik_result["joint_degrees"]
    depth_scale = float(camera_config.get("depth_scale", 0.001))
    tag_depth_measured = sample_depth_m(
        depth, image_corners.mean(axis=0), depth_scale
    )
    tag_depth_pnp = float(camera_t_tag[2, 3])
    depth_difference = (
        None
        if tag_depth_measured is None
        else float(tag_depth_pnp - tag_depth_measured)
    )

    debug = make_debug_image(
        image,
        image_corners,
        projected_corners,
        camera_t_tag,
        camera_matrix,
        dist_coeffs,
        max(0.01, args.tag_size_mm / 1000.0),
        center_tag,
        validation_tag,
        tag_id,
        reprojection_rmse,
    )
    debug_path = output_dir / "plate_apriltag_axes.png"
    if not cv2.imwrite(str(debug_path), debug):
        raise OSError(f"Failed to write {debug_path}")

    np.save(output_dir / "camera_T_tag.npy", camera_t_tag)
    np.save(output_dir / "arm_base_T_tag.npy", base_t_tag)
    np.save(output_dir / "arm_base_P_plate_center.npy", center_base)
    np.save(output_dir / "arm_base_T_end_validation.npy", base_t_end_validation)
    np.save(output_dir / "validation_endpose.npy", validation_endpose)
    if joint_degrees is not None:
        np.save(output_dir / "validation_joint_degrees.npy", joint_degrees)

    result = {
        "tag_family": "tag25h9",
        "tag_id": tag_id,
        "tag_size_mm": args.tag_size_mm,
        "tag_frame": {
            "origin": "centre of the outer black square",
            "x_axis": "canonical marker right",
            "y_axis": "canonical marker up",
            "z_axis": "outward from plate",
            "corner_order": [
                "canonical top-left",
                "canonical top-right",
                "canonical bottom-right",
                "canonical bottom-left",
            ],
        },
        "pnp_ippe_check": pnp_check,
        "image_corners_px": image_corners,
        "reprojection_rmse_px": reprojection_rmse,
        "pnp_tag_depth_m": tag_depth_pnp,
        "measured_tag_depth_m": tag_depth_measured,
        "pnp_minus_measured_depth_m": depth_difference,
        "capture_endpose_mm_deg": capture_endpose,
        "end_T_camera": end_t_camera,
        "camera_T_tag": camera_t_tag,
        "arm_base_T_camera": base_t_camera,
        "arm_base_T_tag": base_t_tag,
        "tag_P_plate_center_m": center_tag,
        "arm_base_P_plate_center_m": center_base,
        "tag_P_validation_m": validation_tag,
        "arm_base_P_validation_m": validation_base,
        "validation_orientation": (
            "EE +Z equals arm-base -Z; yaw is searched for strict IK accuracy"
        ),
        "strict_ik_tolerance_mm_deg": [
            STRICT_POS_TOL_MM,
            STRICT_ROT_TOL_DEG,
        ],
        "validation_endpose_mm_deg": validation_endpose,
        "validation_joint_degrees": joint_degrees,
        "ik": ik_result,
        "warning": (
            "Execute only after checking the debug axes image, IK reachable/path_ok, "
            "joint limits and collision-free clearance. path_ok checks joint limits only."
        ),
    }
    result_path = output_dir / "plate_apriltag_calibration.json"
    with result_path.open("w", encoding="utf-8") as file:
        json.dump(json_ready(result), file, ensure_ascii=False, indent=2)

    print(f"Detected tag25h9 ID: {tag_id}")
    print(f"Corner reprojection RMSE: {reprojection_rmse:.3f} px")
    print('tag center in base frame:',base_t_tag[:3,3])
    print(f"Plate center in arm base [m]: {center_base}")
    print("\nIPPE candidate poses:")
    for candidate in pnp_check["candidates"]:
        print(
            f"  candidate {candidate['index']}: "
            f"RMSE={candidate['rmse_px']:.4f}px, "
            f"depth={candidate['depth_m']:.4f}m, "
            f"euler={np.round(candidate['euler_xyz_deg'], 3)}deg"
        )

    if "rmse_gap_px" in pnp_check:
        print(f"IPPE RMSE gap: {pnp_check['rmse_gap_px']:.4f} px")
        print(
            f"IPPE normal difference: "
            f"{pnp_check['normal_angle_deg']:.3f} deg \n \n"
        )

    if ik_result is not None:
        print(f"Validation point in arm base [m]: {validation_base}")
        print(f"Validation endpose [mm, deg]: {np.round(validation_endpose, 4)}")
        print(f"Validation joint degrees: {joint_degrees}")
        print(
            f"IK reachable={ik_result['reachable']}, path_ok={ik_result['path_ok']}, "
            f"position_error={ik_result['pos_err_mm']:.6f} mm, "
            f"rotation_error={ik_result['rot_err_deg']:.6f} deg, "
            f"yaw_offset={ik_result['yaw_offset_deg']:.3f} deg"
        )
        if joint_degrees is None:
            print(
                "No executable joint degrees were generated because strict IK "
                "accuracy or path validation failed."
            )
    if depth_difference is not None:
        print(f"PnP-depth difference: {depth_difference * 1000.0:.2f} mm")
    print(f"Results written to {output_dir}")

    if args.show:
        cv2.imshow("AprilTag plate calibration", debug)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

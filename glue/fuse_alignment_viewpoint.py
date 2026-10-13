"""
fuse_alignment_viewpoint.py

1. Load completion Part-1 top/rest point clouds.
2. Continuously optimize viewpoint direction with J_area + J_obs.
3. Build a 10 x 10 x 10 mm cube around the optimal camera position.
4. Search Piper joint space for a reachable camera pose inside the cube.
5. If no reachable solution exists, use fixed fallback joint degrees.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import pinocchio as pin
from scipy import ndimage
from scipy.optimize import differential_evolution, least_squares
from scipy.spatial.transform import Rotation as R

PROJECT_ROOT = Path(os.getenv("AAM_PROJECT_ROOT", "/home/smmg/AAM"))
sys.path.append(str(PROJECT_ROOT))
from Piper.endpose_reachability_safe import load_arm_model, get_safe_bounds, frame_pose, DEFAULT_EE_FRAME

CAMERA_CONFIG_PATH = PROJECT_ROOT / "config" / "calibration" / "right_camera" / "camera_config.npy"
HAND_EYE_PATH = PROJECT_ROOT / "config" / "calibration" / "right_camera" / "ecT.npy"
OUTPUT_NAME = "fuse_alignment_viewpoint.json"

CAMERA_WORK_DISTANCE_M = 0.25
RADIUS_MIN_M = 0.25
RADIUS_MAX_M = 0.45
ELEVATION_MIN_DEG = 10.0
ELEVATION_MAX_DEG = 70.0
FOV_MARGIN_PX = 20
MIN_DEPTH_M = 0.05

W_TOP = 0.45
W_SIDE = 0.55
LAMBDA_AREA = 0.40
LAMBDA_OBS = 0.60
W_MASK = 0.30
W_CONTOUR = 0.40
W_DEPTH = 0.30
CONTOUR_NORM_PX = 10.0
DEPTH_NORM_M = 0.010
PERTURB_TRANS_M = 0.001
PERTURB_YAW_DEG = 1.0
MASK_DILATE_ITER = 2

VIEW_OPT_MAXITER = 20
VIEW_OPT_POPSIZE = 8
VIEW_OPT_SEED = 1

CUBE_SIZE_M = np.array([0.010, 0.010, 0.010], dtype=float)
LOOK_ANGLE_TOL_DEG = 10.0
IK_RANDOM_SEEDS = 30
IK_MAX_NFEV = 500
IK_SEED = 1
FALLBACK_JOINT_DEGREES = [0.0, 20.0, -30.0, 0.0, 45.0, 20.0]

BASE_Z = np.array([0.0, 0.0, 1.0], dtype=float)
BASE_Y = np.array([0.0, 1.0, 0.0], dtype=float)
EULER_ORDER = "xyz"


def normalize(v):
    return np.asarray(v, dtype=float) / (np.linalg.norm(v) + 1e-12)


def load_matrix(path):
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.shape == () and arr.dtype == object:
        arr = arr.item()
    return np.asarray(arr, dtype=float)


def load_camera_intrinsic(path):
    cfg = np.load(path, allow_pickle=True)
    if isinstance(cfg, np.ndarray) and cfg.shape == ():
        cfg = cfg.item()
    intr = cfg["color_intrinsic"]
    return {"fx": float(intr["fx"]), "fy": float(intr["fy"]), "cx": float(intr["ppx"]), "cy": float(intr["ppy"]), "width": int(intr.get("width", 640)), "height": int(intr.get("height", 480))}


def load_points(path):
    return np.asarray(o3d.io.read_point_cloud(str(path)).points, dtype=float)


def radial_direction(azimuth_deg, elevation_deg):
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    return normalize(np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], dtype=float))


def look_at_camera_pose(camera_position, target):
    z_cam = normalize(target - camera_position)
    up = BASE_Z if abs(float(np.dot(z_cam, BASE_Z))) < 0.95 else BASE_Y
    x_cam = normalize(np.cross(z_cam, up))
    y_cam = normalize(np.cross(z_cam, x_cam))
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x_cam, y_cam, z_cam])
    T[:3, 3] = camera_position
    return T


def render_points(top_points, side_points, T_base_cam, intr, check_fov=False):
    points = np.vstack([top_points, side_points])
    labels = np.concatenate([np.ones(len(top_points), dtype=np.uint8), np.full(len(side_points), 2, dtype=np.uint8)])
    Rcb = T_base_cam[:3, :3].T
    tbc = T_base_cam[:3, 3]
    cam = (Rcb @ (points - tbc).T).T
    z = cam[:, 2]
    u = intr["fx"] * cam[:, 0] / np.maximum(z, 1e-9) + intr["cx"]
    v = intr["fy"] * cam[:, 1] / np.maximum(z, 1e-9) + intr["cy"]
    if check_fov:
        inside_margin = (z > MIN_DEPTH_M) & (u >= FOV_MARGIN_PX) & (u < intr["width"] - FOV_MARGIN_PX) & (v >= FOV_MARGIN_PX) & (v < intr["height"] - FOV_MARGIN_PX)
        if not np.all(inside_margin):
            return None
    valid = (z > MIN_DEPTH_M) & (u >= 0) & (u < intr["width"]) & (v >= 0) & (v < intr["height"])
    if not np.any(valid):
        return None
    ui = np.floor(u[valid]).astype(int)
    vi = np.floor(v[valid]).astype(int)
    zv = z[valid]
    lv = labels[valid]
    pixel_idx = vi * intr["width"] + ui
    order = np.lexsort((zv, pixel_idx))
    pixel_sorted = pixel_idx[order]
    first = np.r_[True, pixel_sorted[1:] != pixel_sorted[:-1]]
    keep = order[first]
    ui = ui[keep]
    vi = vi[keep]
    zv = zv[keep]
    lv = lv[keep]
    label_img = np.zeros((intr["height"], intr["width"]), dtype=np.uint8)
    depth_img = np.zeros((intr["height"], intr["width"]), dtype=np.float32)
    label_img[vi, ui] = lv
    depth_img[vi, ui] = zv
    mask = label_img > 0
    dense_mask = ndimage.binary_dilation(mask, iterations=MASK_DILATE_ITER)
    dense_mask = ndimage.binary_closing(dense_mask, iterations=1)
    top_area = int(np.count_nonzero(label_img == 1))
    side_area = int(np.count_nonzero(label_img == 2))
    depth_median = float(np.median(zv))
    return {"mask": dense_mask, "depth_median": depth_median, "top_area": top_area, "side_area": side_area}


def mask_iou(a, b):
    inter = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    return float(inter / max(union, 1))


def contour_distance(a, b):
    edge_a = a & (~ndimage.binary_erosion(a))
    edge_b = b & (~ndimage.binary_erosion(b))
    if not np.any(edge_a) or not np.any(edge_b):
        return CONTOUR_NORM_PX
    dist_a = ndimage.distance_transform_edt(~edge_a)
    dist_b = ndimage.distance_transform_edt(~edge_b)
    return 0.5 * (float(np.mean(dist_a[edge_b])) + float(np.mean(dist_b[edge_a])))


def perturb_points(points, center, mode):
    if mode == "x":
        return points + np.array([PERTURB_TRANS_M, 0.0, 0.0])
    if mode == "y":
        return points + np.array([0.0, PERTURB_TRANS_M, 0.0])
    if mode == "z":
        return points + np.array([0.0, 0.0, PERTURB_TRANS_M])
    Rz = R.from_euler("z", PERTURB_YAW_DEG, degrees=True).as_matrix()
    return (Rz @ (points - center).T).T + center


def observability_score(top_points, side_points, center, T_base_cam, intr, nominal):
    values = []
    for mode in ("x", "y", "z", "yaw"):
        top_p = perturb_points(top_points, center, mode)
        side_p = perturb_points(side_points, center, mode)
        perturbed = render_points(top_p, side_p, T_base_cam, intr, check_fov=False)
        if perturbed is None:
            values.append(1e-6)
            continue
        d_mask = 1.0 - mask_iou(nominal["mask"], perturbed["mask"])
        d_contour = min(contour_distance(nominal["mask"], perturbed["mask"]) / CONTOUR_NORM_PX, 1.0)
        d_depth = min(abs(perturbed["depth_median"] - nominal["depth_median"]) / DEPTH_NORM_M, 1.0)
        values.append(max(W_MASK * d_mask + W_CONTOUR * d_contour + W_DEPTH * d_depth, 1e-6))
    return float(np.prod(values) ** 0.25)


def viewpoint_score(angles, top_points, side_points, center, radius, intr):
    radial = radial_direction(angles[0], angles[1])
    camera_position = center + radius * radial
    T_base_cam = look_at_camera_pose(camera_position, center)
    nominal = render_points(top_points, side_points, T_base_cam, intr, check_fov=True)
    if nominal is None:
        return -1e6
    top_area = nominal["top_area"] / max(len(top_points), 1)
    side_area = nominal["side_area"] / max(len(side_points), 1)
    j_area = W_TOP * top_area + W_SIDE * side_area
    j_obs = observability_score(top_points, side_points, center, T_base_cam, intr, nominal)
    return float(LAMBDA_AREA * j_area + LAMBDA_OBS * j_obs)


def optimize_viewpoint(top_points, side_points, center, radius, intr):
    objective = lambda x: -viewpoint_score(x, top_points, side_points, center, radius, intr)
    result = differential_evolution(objective, [(-180.0, 180.0), (ELEVATION_MIN_DEG, ELEVATION_MAX_DEG)], maxiter=VIEW_OPT_MAXITER, popsize=VIEW_OPT_POPSIZE, seed=VIEW_OPT_SEED, polish=True, workers=1)
    azimuth_deg = float(result.x[0])
    elevation_deg = float(result.x[1])
    radial = radial_direction(azimuth_deg, elevation_deg)
    camera_position = center + radius * radial
    T_base_cam = look_at_camera_pose(camera_position, center)
    return T_base_cam


def matrix_to_endpose_mm_deg(T_base_ee):
    xyz_mm = T_base_ee[:3, 3] * 1000.0
    rpy_deg = R.from_matrix(T_base_ee[:3, :3]).as_euler(EULER_ORDER, degrees=True)
    return [round(float(v), 2) for v in np.concatenate([xyz_mm, rpy_deg])]


def q_to_camera_pose(model, data, frame_id, q, T_ee_cam):
    pose = frame_pose(model, data, q, frame_id)
    T_base_ee = np.eye(4)
    T_base_ee[:3, :3] = pose.rotation
    T_base_ee[:3, 3] = pose.translation
    return T_base_ee, T_base_ee @ T_ee_cam


def cube_outside_vector(position, cube_center):
    return np.maximum(np.abs(position - cube_center) - CUBE_SIZE_M / 2.0, 0.0)


def camera_look_angle_deg(T_base_cam, target):
    z_cam = normalize(T_base_cam[:3, 2])
    target_dir = normalize(target - T_base_cam[:3, 3])
    return float(np.rad2deg(np.arccos(np.clip(np.dot(z_cam, target_dir), -1.0, 1.0))))


def search_reachable_pose(best_T_base_cam, target, T_ee_cam):
    model = load_arm_model()
    frame_id = model.getFrameId(DEFAULT_EE_FRAME)
    lb, ub = get_safe_bounds(model)
    cube_center = best_T_base_cam[:3, 3]
    rng = np.random.default_rng(IK_SEED)
    fallback_q = np.deg2rad(np.asarray(FALLBACK_JOINT_DEGREES, dtype=float))
    seeds = [np.clip(pin.neutral(model), lb, ub), np.clip(fallback_q, lb, ub)]
    for _ in range(IK_RANDOM_SEEDS):
        seeds.append(rng.uniform(lb, ub))
    best = None
    for q0 in seeds:
        data = model.createData()
        def residual(q):
            _, T_base_cam = q_to_camera_pose(model, data, frame_id, q, T_ee_cam)
            p_cam = T_base_cam[:3, 3]
            target_dir = normalize(target - p_cam)
            r_cube = cube_outside_vector(p_cam, cube_center) / 0.005
            r_look = (T_base_cam[:3, 2] - target_dir) / 0.05
            return np.concatenate([r_cube, r_look])
        res = least_squares(residual, q0, bounds=(lb, ub), max_nfev=IK_MAX_NFEV)
        q = res.x
        T_base_ee, T_base_cam = q_to_camera_pose(model, data, frame_id, q, T_ee_cam)
        cube_err = float(np.linalg.norm(cube_outside_vector(T_base_cam[:3, 3], cube_center)))
        look_angle = camera_look_angle_deg(T_base_cam, target)
        feasible = cube_err <= 1e-6 and look_angle <= LOOK_ANGLE_TOL_DEG
        score = look_angle + 1000.0 * cube_err + float(np.linalg.norm(T_base_cam[:3, 3] - cube_center))
        if feasible and (best is None or score < best["score"]):
            best = {"q": q.copy(), "T_base_ee": T_base_ee.copy(), "score": score}
    if best is not None:
        return True, best["T_base_ee"], np.rad2deg(best["q"]).tolist()
    data = model.createData()
    fallback_T_base_ee, _ = q_to_camera_pose(model, data, frame_id, fallback_q, T_ee_cam)
    return False, fallback_T_base_ee, FALLBACK_JOINT_DEGREES


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--camera-config", type=Path, default=CAMERA_CONFIG_PATH)
    parser.add_argument("--hand-eye", type=Path, default=HAND_EYE_PATH)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    completion_dir = run_dir / "completion" / "depression"
    output_dir = run_dir / "pickplace"
    plane_path = completion_dir / "_part1_plane.pcd"
    rest_path = completion_dir / "_part1_rest.pcd"
    output_path = args.output_json or output_dir / OUTPUT_NAME
    top_points = load_points(plane_path)
    side_points = load_points(rest_path)
    all_points = np.vstack([top_points, side_points])
    center = all_points.mean(axis=0)
    bbox_diag = float(np.linalg.norm(all_points.max(axis=0) - all_points.min(axis=0)))
    radius = float(np.clip(CAMERA_WORK_DISTANCE_M + 0.5 * bbox_diag, RADIUS_MIN_M, RADIUS_MAX_M))
    intr = load_camera_intrinsic(args.camera_config)
    T_ee_cam = load_matrix(args.hand_eye)
    best_T_base_cam = optimize_viewpoint(top_points, side_points, center, radius, intr)
    found, T_base_ee, joint_degrees = search_reachable_pose(best_T_base_cam, center, T_ee_cam)
    result = {"best viewpoint exist": "yes" if found else "no", "endpose": matrix_to_endpose_mm_deg(T_base_ee), "joint degrees": [round(float(v), 2) for v in joint_degrees]}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

import json, sys
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import numpy as np

sys.path.append('E:/HKUSTGZ/AAM')
from Piper.endpose_reachability import *


# =========================
# Configurable parameters
# =========================
DATA_DIR = Path(r"E:\HKUSTGZ\AAM\construction\data")
STEM = "2"

DEPTH_PATH = DATA_DIR / f"{STEM}.npy"
ENDPOSE_PATH = DATA_DIR / f"{STEM}.json"
OUTPUT_PATH = DATA_DIR / "reachable_scanpose.json"

# Calibration files.
CAMERA_CONFIG_PATH = Path(r"E:\HKUSTGZ\AAM\config\calibration\right_camera\camera_config.npy")
HAND_EYE_PATH = Path(r"E:\HKUSTGZ\AAM\config\calibration\right_camera\ecT.npy")

# Optional: SAM / SAM3 object mask. Keep None if not available.
# Strongly recommended to provide object mask, otherwise the whole depth image is used.
OBJECT_MASK_PATH = None  # e.g. DATA_DIR / f"{STEM}_mask.npy"

# Unit conversion for the current data format.
DEPTH_SCALE = 1e-3       # depth raw unit -> meter
POSITION_SCALE = 1e-6    # endpose raw position -> meter
ANGLE_SCALE = 1e-3       # endpose raw angle -> degree if ANGLE_UNIT == "deg"
ANGLE_UNIT = "deg"      # "deg" or "rad"

# Robot endpose output convention.
OUTPUT_EULER_ORDER = "xyz"

# Target estimation.
TARGET_PERCENTILE = 2.0

# Radius planning.
# radius_baseline = bbox diagonal length in meter.
# Then radius_center = RADIUS_BASELINE_SCALE * radius_baseline.
# Finally generate NUM_RADIUS radii around radius_center.
RADIUS_BASELINE_SCALE = 2.0
RADIUS_FLOAT_M = 0.12
RADIUS_MIN = 0.25
RADIUS_MAX = 0.70
NUM_RADIUS = 4

# Candidate angle grid in object coordinate.
# yaw/pitch rotate the first-view direction around the object coordinate axes.
# roll rotates camera around its optical axis while still looking at object_center.
YAW_DEG_LIST = [ -55, -45, -30, 0, 30, 42, 50 ] #7
PITCH_DEG_LIST = [ -20, -10, 0, 15, 30, 45, 60 ] #7
ROLL_DEG_LIST = [ -70, -55, -30, 0, 30, 55, 70 ] #3

# If True, stop once enough reachable poses are found.
# If False, test all candidates and save every reachable pose.
STOP_AFTER_MIN_REACHABLE = False
MIN_REACHABLE_TO_SAVE = 30

# =========================
# Basic geometry utilities
# =========================
def normalize(v, eps=1e-9):
    v = np.asarray(v, dtype=float).reshape(3)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError(f"Cannot normalize near-zero vector: {v}")
    return v / n


def unpack_npy_object(arr):
    if isinstance(arr, np.ndarray) and arr.shape == () and arr.dtype == object:
        return arr.item()
    if isinstance(arr, np.ndarray) and arr.dtype == object and arr.size == 1:
        return arr.reshape(-1)[0]
    return arr


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# =========================
# Data loading
# =========================
def parse_endpose_json(path, position_scale=1e-6, angle_scale=1e-3, angle_unit="deg"):
    """
    Parse robot endpose json into T_base_ee.
    Input json may be a dict or a list of single-key dicts.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    pose = {}
    if isinstance(raw, list):
        for item in raw:
            pose.update(item)
    elif isinstance(raw, dict):
        pose = raw
    else:
        raise ValueError("Endpose JSON must be a dict or a list of single-key dicts.")

    x = float(pose["x"]) * position_scale
    y = float(pose["y"]) * position_scale
    z = float(pose["z"]) * position_scale

    rx = float(pose["rx"]) * angle_scale
    ry = float(pose["ry"]) * angle_scale
    rz = float(pose["rz"]) * angle_scale

    t_base_ee = np.eye(4)
    t_base_ee[:3, :3] = R.from_euler(
        OUTPUT_EULER_ORDER,
        [rx, ry, rz],
        degrees=(angle_unit == "deg")
    ).as_matrix()
    t_base_ee[:3, 3] = [x, y, z]
    return t_base_ee


def load_camera_config(camera_config_path):
    config = np.load(camera_config_path, allow_pickle=True).item()
    depth_intrinsic = config['depth_intrinsic']
    color_intrinsic = config['color_intrinsic']
    depth_scale = config['depth_scale']
    depth_to_color_extrinsic = config['depth_to_color_extrinsic']
    return color_intrinsic, depth_intrinsic, depth_to_color_extrinsic, depth_scale


def load_optional_matrix(path):
    if path is None:
        return np.eye(4)
    arr = np.load(path, allow_pickle=True)
    arr = unpack_npy_object(arr)
    arr = np.asarray(arr, dtype=float)
    if arr.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 matrix from {path}, got {arr.shape}")
    return arr


# =========================
# Point cloud and object frame
# =========================
def depth_to_points(depth, fx, fy, cx, cy, depth_scale=1e-3, mask=None):
    """Back-project depth image into camera-coordinate point cloud."""
    z = depth.astype(float) * depth_scale
    valid = np.isfinite(z) & (z > 0)
    if mask is not None:
        valid &= mask.astype(bool)

    v, u = np.nonzero(valid)
    if len(u) == 0:
        raise ValueError("No valid depth points found.")

    zz = z[v, u]
    xx = (u - cx) * zz / fx
    yy = (v - cy) * zz / fy
    return np.column_stack([xx, yy, zz])


def transform_points(t, points):
    points_h = np.column_stack([points, np.ones(len(points))])
    return (t @ points_h.T).T[:, :3]


def estimate_target(points_base, percentile=2.0):
    """
    Robust object center and bbox using percentiles instead of raw min/max.
    """
    low = np.percentile(points_base, percentile, axis=0)
    high = np.percentile(points_base, 100.0 - percentile, axis=0)
    center = 0.5 * (low + high)
    size = high - low
    return center, size, low, high


def build_object_coordinate(points_base, object_center, world_up=np.array([0.0, 0.0, 1.0])):
    """
    Build object coordinate from first-frame object point cloud.

    Origin: object_center.
    Axes: PCA axes, with object z-axis chosen as the PCA axis closest to world_up.
    Return: T_base_object, whose columns are object x/y/z axes in base frame.
    """
    pts = np.asarray(points_base, dtype=float) - object_center.reshape(1, 3)
    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    axes = [normalize(eigvecs[:, i]) for i in order]

    world_up = normalize(world_up)
    z_idx = int(np.argmax([abs(np.dot(a, world_up)) for a in axes]))
    z_axis = axes[z_idx]
    if np.dot(z_axis, world_up) < 0:
        z_axis = -z_axis

    remain = [i for i in range(3) if i != z_idx]
    # Use the remaining axis with larger variance as object x-axis.
    x_axis = axes[remain[0]]
    x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
    if np.linalg.norm(x_axis) < 1e-9:
        x_axis = np.cross(np.array([1.0, 0.0, 0.0]), z_axis)
        if np.linalg.norm(x_axis) < 1e-9:
            x_axis = np.cross(np.array([0.0, 1.0, 0.0]), z_axis)
    x_axis = normalize(x_axis)

    y_axis = normalize(np.cross(z_axis, x_axis))
    x_axis = normalize(np.cross(y_axis, z_axis))

    t_base_object = np.eye(4)
    t_base_object[:3, :3] = np.column_stack([x_axis, y_axis, z_axis])
    t_base_object[:3, 3] = object_center
    return t_base_object


def build_radius_list(bbox_diag):
    """
    Generate at least NUM_RADIUS distinct radii from bbox diagonal baseline.
    All radii are in meters.
    """
    radius_center = float(np.clip(RADIUS_BASELINE_SCALE * bbox_diag, RADIUS_MIN, RADIUS_MAX))
    low = max(RADIUS_MIN, radius_center - RADIUS_FLOAT_M)
    high = min(RADIUS_MAX, radius_center + RADIUS_FLOAT_M)

    # If clipping makes the range too narrow, use the whole allowed range.
    if high - low < 0.03:
        low, high = RADIUS_MIN, RADIUS_MAX

    radii = np.linspace(low, high, NUM_RADIUS)
    radii = np.unique(np.round(radii, 6))

    if len(radii) < NUM_RADIUS:
        radii = np.linspace(RADIUS_MIN, RADIUS_MAX, NUM_RADIUS)

    return [float(r) for r in radii]


# =========================
# Candidate generation
# =========================
def rotate_first_view_direction(v0_base, t_base_object, yaw_deg, pitch_deg):
    """
    Rotate first-view direction in object coordinate.

    v0_base: vector from object_center to first-frame camera position, expressed in base.
    yaw:   rotation around object z-axis.
    pitch: rotation around object y-axis after yaw/pitch composition.
    """
    r_base_object = t_base_object[:3, :3]
    v0_object = r_base_object.T @ normalize(v0_base)

    # Object-frame rotation: yaw around object Z, pitch around object Y.
    rot_object = R.from_euler('zy', [yaw_deg, pitch_deg], degrees=True).as_matrix()
    direction_object = normalize(rot_object @ v0_object)
    direction_base = normalize(r_base_object @ direction_object)
    return direction_base


def choose_safe_up_hint(z_cam, preferred_up):
    """
    Avoid look-at singularity when up_hint is parallel to optical axis.
    """
    preferred_up = normalize(preferred_up)
    if abs(float(np.dot(preferred_up, normalize(z_cam)))) < 0.95:
        return preferred_up

    fallback_list = [
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
    ]
    for cand in fallback_list:
        if abs(float(np.dot(normalize(cand), normalize(z_cam)))) < 0.95:
            return normalize(cand)
    raise ValueError("Failed to choose a valid up hint.")


def look_at_camera_pose_with_roll(p_cam, target, up_hint, roll_deg):
    """
    Build camera pose that looks at target, then apply roll around camera optical axis.
    Assumption: camera +Z axis is the optical/depth direction.
    """
    z_cam = normalize(target - p_cam)
    up_hint = choose_safe_up_hint(z_cam, up_hint)
    x_cam = normalize(np.cross(up_hint, z_cam))
    y_cam = normalize(np.cross(z_cam, x_cam))

    r_base_cam = np.column_stack([x_cam, y_cam, z_cam])
    r_roll_cam = R.from_euler('z', roll_deg, degrees=True).as_matrix()
    r_base_cam = r_base_cam @ r_roll_cam

    t_base_cam = np.eye(4)
    t_base_cam[:3, :3] = r_base_cam
    t_base_cam[:3, 3] = p_cam
    return t_base_cam


def matrix_to_endpose_mm_deg(t_base_ee, euler_order=OUTPUT_EULER_ORDER):
    """Convert T_base_ee into [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]."""
    xyz_mm = t_base_ee[:3, 3] * 1000.0
    rpy_deg = R.from_matrix(t_base_ee[:3, :3]).as_euler(euler_order, degrees=True)
    return [float(v) for v in np.concatenate([xyz_mm, rpy_deg])]


def generate_candidate_endposes(object_center, t_base_object, p_cam0, t_ee_cam, radius_list):
    """
    Generate endpose candidates:
      - position: object-centered sphere with multiple radii
      - direction: first-frame direction rotated by yaw/pitch in object coordinate
      - orientation: camera +Z always looks at object_center
      - roll: rotation around camera optical axis
    """
    v0_base = normalize(p_cam0 - object_center)
    object_z_axis = t_base_object[:3, 2]

    records = []
    candidate_idx = 0
    inv_t_ee_cam = np.linalg.inv(t_ee_cam)

    for radius in radius_list:
        for yaw in YAW_DEG_LIST:
            for pitch in PITCH_DEG_LIST:
                direction_base = rotate_first_view_direction(v0_base, t_base_object, yaw, pitch)
                p_cam = object_center + radius * direction_base

                for roll in ROLL_DEG_LIST:
                    t_base_cam = look_at_camera_pose_with_roll(
                        p_cam=p_cam,
                        target=object_center,
                        up_hint=object_z_axis,
                        roll_deg=roll,
                    )
                    t_base_ee = t_base_cam @ inv_t_ee_cam
                    endpose = matrix_to_endpose_mm_deg(t_base_ee)

                    records.append({
                        "candidate_idx": candidate_idx,
                        'radius':float(radius),
                        "yaw": float(yaw),
                        "pitch": float(pitch),
                        "roll": float(roll),
                        "endpose": endpose,
                    })
                    candidate_idx += 1

    return records


def filter_reachable_candidates(candidate_records):
    reachable_records = []

    for cand in candidate_records:
        endpose = cand["endpose"]
        result = reachability_test(endpose)
        if result['reachable'] == True:
            out_record = {
                "idx": len(reachable_records),
                'radius':cand['radius'],
                "yaw": cand["yaw"],
                "pitch": cand["pitch"],
                "roll": cand["roll"],
                "endpose": cand["endpose"],
                'joint_degrees':result['joint_degrees']
            }
            reachable_records.append(out_record)
            print("reachable angles:", out_record['yaw'], out_record['pitch'], out_record['roll'] )

            if STOP_AFTER_MIN_REACHABLE and len(reachable_records) >= MIN_REACHABLE_TO_SAVE:
                break

    return reachable_records


# =========================
# Main pipeline
# =========================
def main():
    depth = np.load(DEPTH_PATH)
    color_intrinsic, _, _, _ = load_camera_config(CAMERA_CONFIG_PATH)
    fx = color_intrinsic['fx']
    fy = color_intrinsic['fy']
    cx = color_intrinsic['ppx']
    cy = color_intrinsic['ppy']

    mask = np.load(OBJECT_MASK_PATH) if OBJECT_MASK_PATH else None
    if mask is None:
        print("WARNING: OBJECT_MASK_PATH is None. The whole valid depth image will be used as object points.")

    points_cam = depth_to_points(depth, fx, fy, cx, cy, DEPTH_SCALE, mask)

    t_base_ee0 = parse_endpose_json(
        ENDPOSE_PATH,
        position_scale=POSITION_SCALE,
        angle_scale=ANGLE_SCALE,
        angle_unit=ANGLE_UNIT,
    )
    t_ee_cam = load_optional_matrix(HAND_EYE_PATH)
    t_base_cam0 = t_base_ee0 @ t_ee_cam

    points_base = transform_points(t_base_cam0, points_cam)
    object_center, bbox_size, bbox_min, bbox_max = estimate_target(points_base, TARGET_PERCENTILE)
    bbox_diag = float(np.linalg.norm(bbox_size))
    radius_list = build_radius_list(bbox_diag)

    t_base_object = build_object_coordinate(points_base, object_center)
    p_cam0 = t_base_cam0[:3, 3]

    candidate_records = generate_candidate_endposes(
        object_center=object_center,
        t_base_object=t_base_object,
        p_cam0=p_cam0,
        t_ee_cam=t_ee_cam,
        radius_list=radius_list,
    )

    print(f"Object center base [m]: {object_center}")
    print(f"BBox size [m]: {bbox_size}")
    print(f"BBox diagonal baseline [m]: {bbox_diag:.6f}")
    print(f"Radius list [m]: {radius_list}")
    print(f"Generated {len(candidate_records)} candidate endposes. Start reachability_test...")
    reachable_records = filter_reachable_candidates(candidate_records)

    if len(reachable_records) < MIN_REACHABLE_TO_SAVE:
        print(
            f"WARNING: Only {len(reachable_records)} reachable poses found, "
            f"less than MIN_REACHABLE_TO_SAVE={MIN_REACHABLE_TO_SAVE}. "
            "The output still only contains tested reachable poses.")

    save_json(OUTPUT_PATH, reachable_records)
    print(f"Saved {len(reachable_records)} reachable scan poses to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from PIL import Image
from scipy.spatial import Delaunay
from scipy.optimize import minimize

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import camera.camera_functions as camera_functions
from Piper.piper_ctrl import connect_right


# =========================
# Parameters
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="Capture and align fine_fuse.pcd.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--fuse-pcd", type=Path)
    parser.add_argument("--camera-config", type=Path)
    parser.add_argument("--hand-eye", type=Path)
    return parser.parse_args()


ARGS = parse_args()
RUN_DIR = ARGS.run_dir.expanduser().resolve()
DATA_DIR = RUN_DIR / "pickplace"
DATA_DIR.mkdir(parents=True, exist_ok=True)

FUSE_PCD_PATH = ARGS.fuse_pcd or DATA_DIR / "fine_fuse_motion.pcd"
IMAGE_PATH = DATA_DIR / "fuse_align.png"
POSE_PATH = DATA_DIR / "fuse_align.json"

CAMERA_CONFIG_PATH = ARGS.camera_config or DATA_DIR / "camera_config.npy"
ECT_PATH = ARGS.hand_eye or PROJECT_ROOT / "config" / "calibration" / "right_camera" / "ecT.npy"

CORRECTED_REPROJECT_PATH = DATA_DIR /'after_alignment.png'

MREAL_PATH = DATA_DIR / "Mreal.png"
MFUSE_PATH = DATA_DIR / "Mfuse.png"

CLUSTER_EPS = 0.01
CLUSTER_MIN_POINTS = 50

# Alpha Shape:
# larger -> easier to connect sparse points
# smaller -> tighter boundary
ALPHA_RADIUS_PX = 8.0

# SAM3
SAM3_CONFIDENCE_THRESHOLD = 0.30

# =========================
# DeepIM-inspired refinement
# =========================

OUTER_REFINEMENT_ITERS = 5

# optimization uses deterministic subsampling
# only for faster loss evaluation
OPT_POINT_STEP = 8

# Each refinement predicts only a SMALL delta pose
VX_BOUND_PX = 8.0
VY_BOUND_PX = 8.0
VZ_BOUND = 0.04
YAW_BOUND_DEG = 3.0

# Loss
W_IOU = 1.0
W_EDGE = 0.5
W_DEPTH = 0.5
W_REG = 0.01

EDGE_NORM_PX = 10.0
DEPTH_NORM_M = 0.01

POWELL_MAXITER = 30

DEPTH_PATH = DATA_DIR / "fuse_align.npy"

CORRECTED_PCD_PATH = DATA_DIR / "fine_fuse_corrected.pcd"
CORRECTION_PATH = DATA_DIR / "iterative_correction.json"
FINAL_MFUSE_PATH = DATA_DIR / "Mfuse_final.png"

CORRECTED_REPROJECT_PATH = DATA_DIR / "after_alignment.png"


def capture_alignment_frame():
    piper = connect_right()
    piper.enable()
    piper.set_speed(10)
    piper.move_joint( 0, 20, -30, 0,45, 0   )
    time.sleep(7.0)

    camera_functions.camera_syn_endpose_path = str(POSE_PATH)
    camera_functions.json = json
    camera_functions.capture(
        img_save_path=str(DATA_DIR) + os.sep,
        save_file_name="fuse_align",
        AUTO_SAVE_INTERVAL=2.0,
        MAX_SAVE_FRAMES=1,
        SAVE_CONFIG=1,
        post_process=1,
        SAVE_ENDPOSE=True,
    )
    piper.disconnect()


capture_alignment_frame()


def projected_points_to_alpha_mask(u, v, h, w, alpha_radius_px):
    """
    Generate Mfuse directly from projected pixels using 2D Alpha Shape.

    No dilation.
    No morphological closing.
    Boundary is determined by actual projected points.
    """

    points = np.column_stack([u, v]).astype(np.float64)

    # Remove duplicate projected pixels
    points = np.unique(points, axis=0)

    tri = Delaunay(points)

    triangles = points[tri.simplices]

    p0 = triangles[:, 0]
    p1 = triangles[:, 1]
    p2 = triangles[:, 2]

    # Triangle edge lengths
    a = np.linalg.norm(p1 - p2, axis=1)
    b = np.linalg.norm(p0 - p2, axis=1)
    c = np.linalg.norm(p0 - p1, axis=1)

    # Heron's formula
    s = (a + b + c) / 2.0

    area = np.sqrt(
        np.maximum(
            s * (s - a) * (s - b) * (s - c),
            0.0
        )
    )

    # Circumradius
    circum_radius = (
        a * b * c
        / (4.0 * area + 1e-12)
    )

    # Alpha Shape rule
    keep = circum_radius <= alpha_radius_px

    mask = np.zeros(
        (h, w),
        dtype=np.uint8
    )

    # Fill accepted Delaunay triangles
    for triangle in triangles[keep]:

        polygon = np.rint(
            triangle
        ).astype(np.int32)

        cv2.fillConvexPoly(
            mask,
            polygon,
            255
        )

    return mask


# ============================================================
# DeepIM-inspired helper functions
# ============================================================

def transform_points(points, T):
    points_h = np.column_stack([
        points,
        np.ones(len(points))
    ])

    return (
        T @ points_h.T
    ).T[:, :3]


def render_fuse(points_base):
    """
    base point cloud
        -> camera
        -> image projection
        -> Alpha Shape mask + Z-buffer depth
    """

    points_cam = transform_points(
        points_base,
        camera_T_base
    )

    x = points_cam[:, 0]
    y = points_cam[:, 1]
    z = points_cam[:, 2]

    valid = z > 0

    x = x[valid]
    y = y[valid]
    z = z[valid]

    u = fx * x / z + cx
    v = fy * y / z + cy

    u = np.rint(u).astype(np.int32)
    v = np.rint(v).astype(np.int32)

    inside = (
        (u >= 0) & (u < w) &
        (v >= 0) & (v < h)
    )

    u = u[inside]
    v = v[inside]
    z = z[inside]

    # ----------------------------------------
    # Mask
    # ----------------------------------------
    mask = projected_points_to_alpha_mask(
        u,
        v,
        h,
        w,
        ALPHA_RADIUS_PX
    )

    # ----------------------------------------
    # Z-buffer
    # ----------------------------------------
    depth_map = np.full(
        h * w,
        np.inf,
        dtype=np.float64
    )

    pixel_idx = v * w + u

    np.minimum.at(
        depth_map,
        pixel_idx,
        z
    )

    depth_map = depth_map.reshape(h, w)

    depth_map[
        ~np.isfinite(depth_map)
    ] = 0.0

    return mask, depth_map, u, v


def mask_edge(mask):
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    edge = np.zeros_like(mask)

    cv2.drawContours(
        edge,
        contours,
        -1,
        255,
        1
    )

    return edge


def edge_distance_map(edge):
    """
    Each pixel stores distance to nearest contour pixel.
    """

    img_dist = np.ones_like(
        edge,
        dtype=np.uint8
    )

    img_dist[edge > 0] = 0

    return cv2.distanceTransform(
        img_dist,
        cv2.DIST_L2,
        3
    )


def calculate_pose_loss(
    Mfuse_candidate,
    Dfuse_candidate,
    theta,
):
    """
    Loss =
        IoU
        + contour
        + RGB-D depth
        + weak regularization
    """

    # ========================================================
    # 1. IoU loss
    # ========================================================
    real = Mreal > 0
    fuse = Mfuse_candidate > 0

    intersection = np.count_nonzero(
        real & fuse
    )

    union = np.count_nonzero(
        real | fuse
    )

    L_iou = 1.0 - intersection / union


    # ========================================================
    # 2. Bidirectional contour distance
    # ========================================================
    fuse_edge = mask_edge(
        Mfuse_candidate
    )

    fuse_dist = edge_distance_map(
        fuse_edge
    )

    d_fuse_to_real = np.mean(
        real_dist[fuse_edge > 0]
    )

    d_real_to_fuse = np.mean(
        fuse_dist[real_edge > 0]
    )

    L_edge = (
        0.5
        * (
            d_fuse_to_real
            + d_real_to_fuse
        )
        / EDGE_NORM_PX
    )


    # ========================================================
    # 3. Depth loss
    # ========================================================
    overlap_depth = (
        (Mreal > 0)
        &
        (depth_real > 0)
        &
        (Dfuse_candidate > 0)
    )

    if np.any(overlap_depth):

        L_depth = np.median(
            np.abs(
                depth_real[overlap_depth]
                -
                Dfuse_candidate[overlap_depth]
            )
        )

        L_depth /= DEPTH_NORM_M

    else:
        L_depth = 10.0


    # ========================================================
    # 4. Small-pose regularization
    # ========================================================
    vx, vy, vz, yaw = theta

    L_reg = (
        (vx / VX_BOUND_PX) ** 2
        +
        (vy / VY_BOUND_PX) ** 2
        +
        (vz / VZ_BOUND) ** 2
        +
        (yaw / YAW_BOUND_DEG) ** 2
    )


    # ========================================================
    # Final loss
    # ========================================================
    L = (
        W_IOU * L_iou
        +
        W_EDGE * L_edge
        +
        W_DEPTH * L_depth
        +
        W_REG * L_reg
    )

    return L


def deepim_delta_to_base_T(
    theta,
    center_base_current
):
    """
    theta = [vx, vy, vz, yaw_base_z]

    DeepIM-inspired translation parameterization:

        z_t = z_s * exp(-vz)

        x_t = z_t * (x_s/z_s + vx/fx)

        y_t = z_t * (y_s/z_s + vy/fy)

    Rotation:
        only around BASE Z

    Rotation center:
        current fuse center
    """

    vx, vy, vz, yaw_deg = theta

    # --------------------------------------------------------
    # Current object center:
    # base -> camera
    # --------------------------------------------------------
    center_base_h = np.r_[
        center_base_current,
        1.0
    ]

    center_cam = (
        camera_T_base
        @ center_base_h
    )[:3]

    xs, ys, zs = center_cam


    # ========================================================
    # DeepIM translation parameterization
    # ========================================================
    zt = zs * np.exp(-vz)

    xt = zt * (
        xs / zs
        +
        vx / fx
    )

    yt = zt * (
        ys / zs
        +
        vy / fy
    )

    center_target_cam = np.array([
        xt,
        yt,
        zt
    ])

    delta_center_cam = (
        center_target_cam
        -
        center_cam
    )


    # --------------------------------------------------------
    # camera translation vector -> base vector
    # --------------------------------------------------------
    delta_center_base = (
        base_T_camera[:3, :3]
        @ delta_center_cam
    )


    # ========================================================
    # Rotation only around BASE Z
    # ========================================================
    Rz = R.from_euler(
        "z",
        yaw_deg,
        degrees=True
    ).as_matrix()


    # ========================================================
    # Rotate around current object center + translation
    #
    # p_new =
    # Rz @ (p - center)
    # + center
    # + delta_center_base
    # ========================================================
    delta_T_base = np.eye(4)

    delta_T_base[:3, :3] = Rz

    delta_T_base[:3, 3] = (
        center_base_current
        +
        delta_center_base
        -
        Rz @ center_base_current
    )

    return (
        delta_T_base,
        delta_center_base
    )

# =========================
# Load image / camera / ecT
# =========================
img = cv2.imread(str(IMAGE_PATH))
h, w = img.shape[:2]

config = np.load(
    CAMERA_CONFIG_PATH,
    allow_pickle=True
).item()

intr = config["color_intrinsic"]

fx = float(intr["fx"])
fy = float(intr["fy"])
cx = float(intr["ppx"])
cy = float(intr["ppy"])

K = np.array([
    [fx, 0.0, cx],
    [0.0, fy, cy],
    [0.0, 0.0, 1.0],
], dtype=np.float64)

ecT = np.load(ECT_PATH)    # end_T_camera


# =========================
# Load endpose
# =========================
with open(POSE_PATH, "r", encoding="utf-8") as f:
    raw = json.load(f)

pose = {}

if isinstance(raw, list):
    for item in raw:
        pose.update(item)
else:
    pose = raw

x, y, z = pose["x"], pose["y"], pose["z"]
rx, ry, rz = pose["rx"], pose["ry"], pose["rz"]

xyz_m = np.array(
    [x, y, z],
    dtype=float
)

rpy_deg = np.array(
    [rx, ry, rz],
    dtype=float
)

# Position unit conversion
if np.max(np.abs(xyz_m)) > 10000:
    xyz_m = xyz_m / 1_000_000.0
else:
    xyz_m = xyz_m / 1000.0

# Rotation unit conversion
if np.max(np.abs(rpy_deg)) > 360:
    rpy_deg = rpy_deg / 1000.0


# =========================
# base_T_end
# =========================
base_T_end = np.eye(4)

base_T_end[:3, :3] = R.from_euler(
    "xyz",
    rpy_deg,
    degrees=True
).as_matrix()

base_T_end[:3, 3] = xyz_m


# =========================
# base_T_camera
# =========================
base_T_camera = base_T_end @ ecT

camera_T_base = np.linalg.inv(
    base_T_camera
)


# =========================
# Load fused point cloud
# base frame
# =========================
pcd = o3d.io.read_point_cloud(
    str(FUSE_PCD_PATH)
)

if len(pcd.points) == 0:
    raise RuntimeError(
        f"Empty point cloud: {FUSE_PCD_PATH}"
    )


# =========================
# Keep largest cluster
# =========================
labels = np.asarray(
    pcd.cluster_dbscan(
        eps=CLUSTER_EPS,
        min_points=CLUSTER_MIN_POINTS,
        print_progress=False,
    )
)

if labels.size and labels.max() >= 0:

    valid_labels = labels[labels >= 0]

    largest = np.bincount(
        valid_labels
    ).argmax()

    keep_idx = np.flatnonzero(
        labels == largest
    )

    pcd = pcd.select_by_index(
        keep_idx
    )


points_base = np.asarray(
    pcd.points
)

print("fused points:", len(points_base))


# ============================================================
# 1. Calculate fine_fuse.pcd center
# ============================================================
# Open3D get_center() = point-cloud centroid
center_base = np.asarray(
    pcd.get_center(),
    dtype=np.float64
)

print("\n========== Fuse center ==========")
print("center_base:")
print(center_base)


# =========================
# center: base -> camera
# =========================
center_base_h = np.array([
    center_base[0],
    center_base[1],
    center_base[2],
    1.0
])

center_cam = (
    camera_T_base
    @ center_base_h
)[:3]

print("\ncenter_cam:")
print(center_cam)

xc, yc, zc = center_cam

if zc <= 0:
    raise RuntimeError(
        f"Fuse center is behind camera: z={zc}"
    )


# =========================
# center: camera -> pixel
# =========================
prompt_u = fx * xc / zc + cx
prompt_v = fy * yc / zc + cy

prompt_u = float(prompt_u)
prompt_v = float(prompt_v)

print("\n========== SAM3 point prompt ==========")
print(
    f"point = ({prompt_u:.2f}, "
    f"{prompt_v:.2f})"
)

if not (
    0 <= prompt_u < w
    and
    0 <= prompt_v < h
):
    raise RuntimeError(
        "Projected fuse center is outside image: "
        f"({prompt_u:.2f}, {prompt_v:.2f})"
    )


# ============================================================
# 2. SAM3 point-prompt segmentation -> Mreal
# ============================================================
print("\n========== Load SAM3 ==========")

try:
    sam3_model = build_sam3_image_model(
        enable_inst_interactivity=True
    )
except TypeError as e:
    raise RuntimeError(
        "Current SAM3 version does not support "
        "enable_inst_interactivity=True. "
        "Point-prompt segmentation requires "
        "SAM3 interactive predictor."
    ) from e

sam3_model.eval()

if (
    not hasattr(
        sam3_model,
        "inst_interactive_predictor"
    )
    or
    sam3_model.inst_interactive_predictor is None
):
    raise RuntimeError(
        "SAM3 interactive predictor was not created."
    )

processor = Sam3Processor(
    sam3_model,
    confidence_threshold=SAM3_CONFIDENCE_THRESHOLD,
)


# =========================
# SAM3 expects RGB/PIL image
# =========================
image_pil = Image.open(
    IMAGE_PATH
).convert("RGB")

inference_state = processor.set_image(
    image_pil
)


# =========================
# One positive point prompt
#
# shape:
# point_coords = Nx2
# point_labels = N
#
# 1 = foreground
# =========================
point_coords = np.array(
    [[prompt_u, prompt_v]],
    dtype=np.float32
)

point_labels = np.array(
    [1],
    dtype=np.int32
)


# =========================
# SAM3 interactive prediction
# =========================
masks, scores, logits = sam3_model.predict_inst(
    inference_state,
    point_coords=point_coords,
    point_labels=point_labels,
    multimask_output=True,
)

masks = np.asarray(masks)
scores = np.asarray(scores).reshape(-1)

print("\nSAM3 masks shape:", masks.shape)
print("SAM3 scores:", scores)


# =========================
# Select highest-score mask
# =========================
best_idx = int(
    np.argmax(scores)
)

Mreal = masks[best_idx]

# Remove singleton dimensions if present
Mreal = np.squeeze(Mreal)

Mreal = (
    Mreal > 0
).astype(np.uint8) * 255

cv2.imwrite(str(MREAL_PATH), Mreal)

if Mreal.shape != (h, w):
    raise RuntimeError(
        f"Mreal shape {Mreal.shape} "
        f"!= image shape {(h, w)}"
    )

print(
    "Selected SAM3 mask:",
    best_idx,
    "score:",
    scores[best_idx]
)

print(
    "Mreal pixels:",
    np.count_nonzero(Mreal)
)


# ============================================================
# 3. fine_fuse: base -> camera
# ============================================================
points_base_h = np.column_stack([
    points_base,
    np.ones(len(points_base))
])

points_cam = (
    camera_T_base
    @ points_base_h.T
).T[:, :3]


# ============================================================
# 4. Project fine_fuse onto image -> Mfuse
# ============================================================
x = points_cam[:, 0]
y = points_cam[:, 1]
z = points_cam[:, 2]

valid = (
    np.isfinite(points_cam).all(axis=1)
    &
    (z > 1e-6)
)

x = x[valid]
y = y[valid]
z = z[valid]

u = fx * x / z + cx
v = fy * y / z + cy

u = np.rint(u).astype(np.int32)
v = np.rint(v).astype(np.int32)

inside = (
    (u >= 0)
    &
    (u < w)
    &
    (v >= 0)
    &
    (v < h)
)

u = u[inside]
v = v[inside]

print("\n========== Fuse projection ==========")
print("projected points inside image:", len(u))


# ============================================================
# 5. Projected points -> Alpha Shape -> Mfuse
#
# IMPORTANT:
# No dilation / no closing.
# Therefore the mask boundary will NOT expand outward
# artificially.
# ============================================================

Mfuse = projected_points_to_alpha_mask(
    u,
    v,
    h,
    w,
    ALPHA_RADIUS_PX
)

cv2.imwrite(str(MFUSE_PATH), Mfuse)

print(
    "Mfuse pixels:",
    np.count_nonzero(Mfuse)
)


print(
    "Mfuse pixels:",
    np.count_nonzero(Mfuse)
)




# ============================================================
# 9. Display Mreal / Mfuse
# ============================================================
# Draw point prompt only for display
Mreal_vis = cv2.cvtColor(
    Mreal,
    cv2.COLOR_GRAY2BGR
)

cv2.circle(Mreal_vis,
    (int(round(prompt_u)),int(round(prompt_v))),6,
    (0, 0, 255),-1)


# ============================================================
# >>> START: DEEPIM-INSPIRED ITERATIVE 4-DOF REFINEMENT <<<
#
# Optimization variables:
#
# theta = [vx, vy, vz, yaw_base_z]
#
# Each iteration:
#
# current fuse
#      ↓
# render Mfuse
#      ↓
# compare with Mreal
#      ↓
# optimize SMALL delta pose
#      ↓
# update fuse
#      ↓
# next iteration starts from UPDATED fuse
#
# NOT RANDOM SEARCH
# ============================================================


# ============================================================
# STEP 1. Load real RGB-D depth
# mm -> m
# ============================================================
depth_real = (
    np.load(DEPTH_PATH)
    .astype(np.float64)
    / 1000.0
)

Z_real = np.median(
    depth_real[
        (Mreal > 0)
        &
        (depth_real > 0)
    ]
)

print("\n========== Real depth ==========")
print(f"Z_real = {Z_real:.6f} m")


# ============================================================
# STEP 2. Pre-compute Mreal contour distance
# ============================================================
real_edge = mask_edge(
    Mreal
)

real_dist = edge_distance_map(
    real_edge
)


# ============================================================
# STEP 3. Initialize iterative refinement
# ============================================================

# Full-resolution point cloud continuously updated
points_current_base = points_base.copy()

# Total correction:
#
# P_final =
# T_total @ P_original
#
T_total_base = np.eye(4)

iteration_records = []


# ============================================================
# STEP 4. DeepIM-style iterative pose refinement
# ============================================================

for iteration in range(
    OUTER_REFINEMENT_ITERS
):

    print(
        f"\n"
        f"========================================\n"
        f" Refinement iteration {iteration + 1}"
        f"/{OUTER_REFINEMENT_ITERS}\n"
        f"========================================"
    )


    # --------------------------------------------------------
    # Current object center in BASE
    # --------------------------------------------------------
    center_base_current = np.mean(
        points_current_base,
        axis=0
    )


    # --------------------------------------------------------
    # Deterministic point subset
    # used ONLY for optimizer evaluation
    # --------------------------------------------------------
    points_opt = (
        points_current_base[
            ::OPT_POINT_STEP
        ]
    )


    # ========================================================
    # Render current pose
    # ========================================================
    M_current, D_current, _, _ = (
        render_fuse(
            points_opt
        )
    )


    # ========================================================
    # DeepIM-inspired initial delta
    #
    # vx, vy:
    # current rendered center -> real mask center
    #
    # vz:
    # current depth -> real depth
    # ========================================================
    m_real = cv2.moments(
        Mreal,
        binaryImage=True
    )

    u_real = (
        m_real["m10"]
        /
        m_real["m00"]
    )

    v_real = (
        m_real["m01"]
        /
        m_real["m00"]
    )


    m_current = cv2.moments(
        M_current,
        binaryImage=True
    )

    u_current = (
        m_current["m10"]
        /
        m_current["m00"]
    )

    v_current = (
        m_current["m01"]
        /
        m_current["m00"]
    )


    vx0 = (
        u_real
        -
        u_current
    )

    vy0 = (
        v_real
        -
        v_current
    )


    z_current_values = (
        D_current[
            D_current > 0
        ]
    )

    Z_current = np.median(
        z_current_values
    )

    # DeepIM:
    # vz = log(z_source / z_target)
    vz0 = np.log(
        Z_current
        /
        Z_real
    )


    # --------------------------------------------------------
    # Every iteration only searches a SMALL relative pose
    # --------------------------------------------------------
    theta0 = np.array([
        np.clip(
            vx0,
            -VX_BOUND_PX,
            VX_BOUND_PX
        ),

        np.clip(
            vy0,
            -VY_BOUND_PX,
            VY_BOUND_PX
        ),

        np.clip(
            vz0,
            -VZ_BOUND,
            VZ_BOUND
        ),

        0.0
    ])


    print(
        "initial theta "
        "[vx px, vy px, vz, yaw deg]:"
    )
    print(theta0)


    # ========================================================
    # Objective:
    # theta -> delta_T -> render -> loss
    # ========================================================
    def objective(theta):

        delta_T, _ = (
            deepim_delta_to_base_T(
                theta,
                center_base_current
            )
        )

        candidate_points = (
            transform_points(
                points_opt,
                delta_T
            )
        )

        M_candidate, D_candidate, _, _ = (
            render_fuse(
                candidate_points
            )
        )

        return calculate_pose_loss(
            M_candidate,
            D_candidate,
            theta
        )


    # ========================================================
    # Current loss
    # ========================================================
    initial_loss = objective(
        np.zeros(4)
    )

    print(
        f"loss before iteration = "
        f"{initial_loss:.6f}"
    )


    # ========================================================
    # STEP 4.x Powell optimization
    #
    # NO gradient
    # NO random initialization
    #
    # Search around current pose
    # ========================================================
    optimization = minimize(

        objective,

        theta0,

        method="Powell",

        bounds=[
            (
                -VX_BOUND_PX,
                VX_BOUND_PX
            ),
            (
                -VY_BOUND_PX,
                VY_BOUND_PX
            ),
            (
                -VZ_BOUND,
                VZ_BOUND
            ),
            (
                -YAW_BOUND_DEG,
                YAW_BOUND_DEG
            )
        ],

        options={
            "maxiter": POWELL_MAXITER,
            "xtol": 1e-3,
            "ftol": 1e-4
        }
    )


    theta_best = optimization.x


    # ========================================================
    # Convert this iteration's result -> BASE delta T
    # ========================================================
    delta_T_j, delta_center_base = (
        deepim_delta_to_base_T(
            theta_best,
            center_base_current
        )
    )


    # ========================================================
    # IMPORTANT:
    #
    # iteration j starts from result of iteration j-1
    #
    # P_j =
    # delta_T_j @ P_(j-1)
    # ========================================================
    points_current_base = (
        transform_points(
            points_current_base,
            delta_T_j
        )
    )


    # ========================================================
    # Accumulate total correction
    #
    # T_total =
    # delta_T_j @ ... @ delta_T_1
    # ========================================================
    T_total_base = (
        delta_T_j
        @ T_total_base
    )


    print(
        "best theta "
        "[vx px, vy px, vz, yaw deg]:"
    )
    print(theta_best)

    print(
        "center translation this iteration "
        "[mm]:"
    )
    print(
        delta_center_base
        * 1000.0
    )

    print(
        f"loss after iteration = "
        f"{optimization.fun:.6f}"
    )


    iteration_records.append({

        "iteration":
            iteration + 1,

        "theta_vx_vy_vz_yaw":
            theta_best.tolist(),

        "center_translation_base_mm":
            (
                delta_center_base
                * 1000.0
            ).tolist(),

        "loss_before":
            float(initial_loss),

        "loss_after":
            float(
                optimization.fun
            ),

        "delta_T_base":
            delta_T_j.tolist()
    })


# ============================================================
# STEP 5. Final correction
# ============================================================

points_base_corrected = (
    points_current_base
)


# ============================================================
# Final yaw around BASE Z
# ============================================================
R_total = (
    T_total_base[:3, :3]
)

total_yaw_deg = np.degrees(
    np.arctan2(
        R_total[1, 0],
        R_total[0, 0]
    )
)


# ============================================================
# Actual object-center displacement
# ============================================================
center_original_base = np.mean(
    points_base,
    axis=0
)

center_corrected_base = np.mean(
    points_base_corrected,
    axis=0
)

center_shift_base = (
    center_corrected_base
    -
    center_original_base
)


print(
    "\n========== FINAL CORRECTION =========="
)

print(
    "center shift base [mm]:"
)
print(
    center_shift_base * 1000.0
)

print(
    "base Z yaw correction [deg]:"
)
print(
    total_yaw_deg
)

print(
    "total delta_T_base:"
)
print(
    T_total_base
)


# ============================================================
# STEP 6. Save corrected point cloud
# ============================================================

pcd_corrected = (
    o3d.geometry.PointCloud()
)

pcd_corrected.points = (
    o3d.utility.Vector3dVector(
        points_base_corrected
    )
)

if pcd.has_colors():
    pcd_corrected.colors = (
        pcd.colors
    )

o3d.io.write_point_cloud(
    str(CORRECTED_PCD_PATH),
    pcd_corrected
)


# ============================================================
# STEP 7. Final full-resolution reprojection
# ============================================================

Mfuse_final, Dfuse_final, u_final, v_final = (
    render_fuse(
        points_base_corrected
    )
)

cv2.imwrite(
    str(FINAL_MFUSE_PATH),
    Mfuse_final
)


# ============================================================
# STEP 8. Final mask metrics
# ============================================================

intersection = np.count_nonzero(
    (Mreal > 0)
    &
    (Mfuse_final > 0)
)

union = np.count_nonzero(
    (Mreal > 0)
    |
    (Mfuse_final > 0)
)

final_iou = (
    intersection
    /
    union
)

print(
    f"final mask IoU = "
    f"{final_iou:.6f}"
)


# ============================================================
# STEP 9. Save complete correction
#
# IMPORTANT:
# Since rotation now exists, use FULL delta_T_base.
#
# Do NOT only add delta_t_base_mm.
# ============================================================

result = {

    "outer_iterations":
        OUTER_REFINEMENT_ITERS,

    "iterations":
        iteration_records,

    "center_shift_base_mm":
        (
            center_shift_base
            * 1000.0
        ).tolist(),

    "delta_yaw_base_z_deg":
        float(
            total_yaw_deg
        ),

    "delta_T_base":
        T_total_base.tolist(),

    "final_mask_iou":
        float(
            final_iou
        )
}


with open(
    CORRECTION_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        result,
        f,
        indent=4
    )


# ============================================================
# STEP 10. Visualize final reprojection
#
# RED:
# corrected fine_fuse points
#
# GREEN:
# Mreal contour
# ============================================================

vis_corrected = img.copy()
overlay_corrected = img.copy()

for u_i, v_i in zip(
    u_final,
    v_final
):

    cv2.circle(
        overlay_corrected,
        (
            int(u_i),
            int(v_i)
        ),
        1,
        (0, 0, 255),
        -1
    )


vis_corrected = cv2.addWeighted(
    overlay_corrected,
    0.5,
    vis_corrected,
    0.5,
    0
)


real_contours, _ = (
    cv2.findContours(
        Mreal,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
)

cv2.drawContours(
    vis_corrected,
    real_contours,
    -1,
    (0, 255, 0),
    2
)


cv2.imwrite(
    str(CORRECTED_REPROJECT_PATH),
    vis_corrected
)


print(
    "\n========== Saved =========="
)

print(
    "Corrected PCD:",
    CORRECTED_PCD_PATH
)

print(
    "Final Mfuse:",
    FINAL_MFUSE_PATH
)

print(
    "Final reprojection:",
    CORRECTED_REPROJECT_PATH
)

print(
    "Correction:",
    CORRECTION_PATH
)

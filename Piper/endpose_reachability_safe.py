# reachability_test_function.py
# Callable Pinocchio-based reachability test for Piper arm.
# Usage in another file:
#     from reachability_test_function import reachability_test
#     result = reachability_test([x, y, z, rx, ry, rz])  # mm, degree
#     print(result["reachable"], result["q_solution_deg"])

import os
import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

# ============================================================
# Fixed config: modify here if your URDF or EE frame changes.
# ============================================================
DEFAULT_URDF = os.path.expanduser(r"/home/smmg/AAM/config/piper/piper_description.urdf")
DEFAULT_URDF = os.path.expanduser(r"E:/HKUSTGZ/AAM/config/piper/piper_description.urdf")

DEFAULT_EE_FRAME = "link6"   # If you use TCP/gripper frame later, change this.

# IK judgement thresholds
POS_TOL_MM = 2.0
ROT_TOL_DEG = 3.0

# IK search config
N_RANDOM_INIT = 100
RANDOM_SEED = 0
POS_SCALE = 0.005             # m, residual normalization
ROT_SCALE = np.deg2rad(3.0)   # rad, residual normalization

# Simple joint interpolation path check
PATH_CHECK_STEPS = 50

# Internal cache: avoids rebuilding URDF model every call.
_MODEL_CACHE = None


# ============================================================
# Basic conversion utilities
# ============================================================
def endpose_to_se3(endpose):
    """
    Convert endpose [x, y, z, rx, ry, rz] to pin.SE3.

    x, y, z: mm
    rx, ry, rz: degree
    Euler order: scipy 'xyz', same as your original code.
    """
    endpose = np.asarray(endpose, dtype=float).reshape(6)
    x, y, z, rx, ry, rz = endpose

    t = np.array([x, y, z], dtype=float) / 1000.0
    rot = R.from_euler("xyz", [rx, ry, rz], degrees=True).as_matrix()

    return pin.SE3(rot, t)


def se3_to_endpose(M):
    """
    Convert pin.SE3 to [x, y, z, rx, ry, rz].

    x, y, z: mm
    rx, ry, rz: degree
    Euler order: scipy 'xyz'.
    """
    t_mm = M.translation * 1000.0
    rpy_deg = R.from_matrix(M.rotation).as_euler("xyz", degrees=True)
    return np.concatenate([t_mm, rpy_deg])


# ============================================================
# Model utilities
# ============================================================
def reduce_to_arm_only(model):
    """
    Lock gripper joints and keep only the 6-DOF arm.

    For Piper URDF:
        joint1~joint6: arm joints
        joint7/joint8: gripper prismatic joints
    """
    q_ref = pin.neutral(model)
    lock_names = ["joint7", "joint8"]
    lock_ids = []

    for name in lock_names:
        if model.existJointName(name):
            lock_ids.append(model.getJointId(name))

    if len(lock_ids) == 0:
        return model

    return pin.buildReducedModel(model, lock_ids, q_ref)


def load_arm_model():
    """
    Load and cache reduced Piper arm model.
    """
    global _MODEL_CACHE

    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    urdf_path = os.path.expanduser(DEFAULT_URDF)
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"Cannot find Piper URDF: {urdf_path}")

    model = pin.buildModelFromUrdf(urdf_path)
    model = reduce_to_arm_only(model)
    _MODEL_CACHE = model
    return model


def print_model_info():
    """
    Optional helper: print joint and frame names.
    Call this only when you need to check DEFAULT_EE_FRAME.
    """
    model = load_arm_model()

    print("\n========== Joints ==========")
    for i, name in enumerate(model.names):
        print(f"{i:2d}: {name}")

    print("\n========== Frames ==========")
    for i, frame in enumerate(model.frames):
        print(f"{i:3d}: {frame.name}")

SAFE_MARGIN_DEG = 2.0

def get_safe_bounds(model, margin_deg=SAFE_MARGIN_DEG):
    lb, ub = get_clean_bounds(model)

    margin = np.deg2rad(margin_deg)
    lb_safe = lb + margin
    ub_safe = ub - margin

    # 防止某些关节范围太小导致上下限反转
    invalid = lb_safe >= ub_safe
    lb_safe[invalid] = lb[invalid]
    ub_safe[invalid] = ub[invalid]

    return lb_safe, ub_safe

def get_clean_bounds(model):
    """
    Clean joint lower/upper bounds.
    Pinocchio may store huge/unbounded values; replace them with [-pi, pi].
    """
    lb = model.lowerPositionLimit.copy()
    ub = model.upperPositionLimit.copy()

    for i in range(model.nq):
        if not np.isfinite(lb[i]) or not np.isfinite(ub[i]):
            lb[i] = -np.pi
            ub[i] = np.pi

        if ub[i] - lb[i] > 100:
            lb[i] = -np.pi
            ub[i] = np.pi

        if ub[i] - lb[i] < 1e-9:
            lb[i] -= 1e-6
            ub[i] += 1e-6

    return lb, ub


def frame_pose(model, data, q, frame_id):
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    return data.oMf[frame_id]


# ============================================================
# IK and simple path check
# ============================================================
def solve_ik(model, target_M):
    """
    Solve IK for DEFAULT_EE_FRAME.
    Returns best result dict.
    """
    data = model.createData()

    if not model.existFrame(DEFAULT_EE_FRAME):
        raise RuntimeError(
            f"Cannot find EE frame '{DEFAULT_EE_FRAME}'. "
            f"Call print_model_info() to check frame names."
        )

    frame_id = model.getFrameId(DEFAULT_EE_FRAME)
    lb, ub = get_safe_bounds(model)

    def residual(q):
        M = frame_pose(model, data, q, frame_id)
        pos_err = M.translation - target_M.translation
        rot_err = pin.log3(target_M.rotation.T @ M.rotation)
        return np.concatenate([
            pos_err / POS_SCALE,
            rot_err / ROT_SCALE,
        ])

    rng = np.random.default_rng(RANDOM_SEED)
    q_neutral = np.clip(pin.neutral(model), lb, ub)
    q_zero = np.clip(np.zeros(model.nq), lb, ub)

    init_list = [q_neutral, q_zero]
    for _ in range(N_RANDOM_INIT):
        init_list.append(rng.uniform(lb, ub))

    best = None

    for q0 in init_list:
        res = least_squares(
            residual,
            q0,
            bounds=(lb, ub),
            max_nfev=1000,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
        )

        q = res.x
        M = frame_pose(model, data, q, frame_id)

        pos_err = np.linalg.norm(M.translation - target_M.translation)
        rot_err = np.linalg.norm(pin.log3(target_M.rotation.T @ M.rotation))
        score = pos_err + 0.05 * rot_err

        if best is None or score < best["score"]:
            best = {
                "q": q,
                "M": M.copy(),
                "pos_err_m": float(pos_err),
                "rot_err_rad": float(rot_err),
                "score": float(score),
                "least_squares_success": bool(res.success),
                "least_squares_message": str(res.message),
            }

    return best


def check_joint_path(model, q_start, q_goal, steps=PATH_CHECK_STEPS):
    """
    Check a simple joint-space interpolation path from q_start to q_goal.
    This only checks joint limits.
    It does NOT check self-collision or environment collision.
    """
    lb, ub = get_clean_bounds(model)

    for s in np.linspace(0.0, 1.0, steps):
        q = pin.interpolate(model, q_start, q_goal, s)
        if np.any(q < lb - 1e-6) or np.any(q > ub + 1e-6):
            return False

    return True


# ============================================================
# Public API
# ============================================================
def reachability_test(endpose):

    target_endpose = np.asarray(endpose, dtype=float).reshape(6)

    model = load_arm_model()
    target_M = endpose_to_se3(target_endpose)

    ik_result = solve_ik(model, target_M)

    q_sol = ik_result["q"]
    M_sol = ik_result["M"]

    joint_degrees = [float(v) for v in (np.rad2deg(q_sol)) ]
    joint_degrees = [round(v, 3) for v in joint_degrees]

    pos_err_mm = ik_result["pos_err_m"] * 1000.0
    rot_err_deg = np.rad2deg(ik_result["rot_err_rad"])

    reachable = (pos_err_mm < POS_TOL_MM) and (rot_err_deg < ROT_TOL_DEG)

    lb, ub = get_clean_bounds(model)
    q_home = np.clip(pin.neutral(model), lb, ub)
    path_ok = check_joint_path(model, q_home, q_sol, steps=PATH_CHECK_STEPS)

    result = {
        "reachable": bool(reachable),
        "path_ok": bool(path_ok),
        "q_solution_rad": q_sol,
        "q_solution_deg": np.rad2deg(q_sol),
        'joint_degrees': joint_degrees,
        "achieved_endpose": se3_to_endpose(M_sol),
        "target_endpose": target_endpose,
        "pos_err_mm": float(pos_err_mm),
        "rot_err_deg": float(rot_err_deg),
        "ee_frame": DEFAULT_EE_FRAME,
        "urdf_path": os.path.expanduser(DEFAULT_URDF),
        "least_squares_success": ik_result["least_squares_success"],
        "least_squares_message": ik_result["least_squares_message"],
    }



    return result

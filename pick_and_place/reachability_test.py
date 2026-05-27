# check_piper_reachability_pin.py
import os
import argparse
import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R


DEFAULT_URDF = os.path.expanduser(
    "E:/HKUSTGZ/AAM/config/piper/piper_description.urdf"
)

DEFAULT_TARGET = [363.78098485,  95.62617437, 234.60791516 ,175.21392495,  -1.80263506,130.35]
# 对你的 URDF，gripper_base 和 link6 位姿相同
DEFAULT_EE_FRAME = "link6" #gripper_base


def endpose_to_se3(endpose):
    """
    [x, y, z, rx, ry, rz]
    x y z: mm
    rx ry rz: degree

    注意：这里保持你之前代码的 scipy 'xyz' 欧拉角定义。
    """
    x, y, z, rx, ry, rz = endpose

    t = np.array([x, y, z], dtype=float) / 1000.0

    Rot = R.from_euler(
        "xyz",
        [rx, ry, rz],
        degrees=True
    ).as_matrix()

    return pin.SE3(Rot, t)


def se3_to_endpose(M):
    t_mm = M.translation * 1000.0
    rpy_deg = R.from_matrix(M.rotation).as_euler("xyz", degrees=True)
    return np.concatenate([t_mm, rpy_deg])


def reduce_to_arm_only(model):
    """
    根据你的 URDF：
    joint1~joint6 是机械臂
    joint7/joint8 是夹爪 prismatic joint
    所以锁住 joint7 和 joint8
    """
    q_ref = pin.neutral(model)

    lock_names = ["joint7", "joint8"]
    lock_ids = []

    for name in lock_names:
        if model.existJointName(name):
            jid = model.getJointId(name)
            lock_ids.append(jid)
            print(f"Lock gripper joint: {name}, id = {jid}")

    if len(lock_ids) == 0:
        print("No gripper joints found. Use original model.")
        return model

    reduced_model = pin.buildReducedModel(
        model,
        lock_ids,
        q_ref
    )

    return reduced_model


def print_model_info(model):
    print("\n========== Joints ==========")
    for i, name in enumerate(model.names):
        print(f"{i:2d}: {name}")

    print("\n========== Frames ==========")
    for i, frame in enumerate(model.frames):
        print(f"{i:3d}: {frame.name}")


def get_clean_bounds(model):
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


def solve_ik(
    model,
    ee_frame,
    target_M,
    n_random=100,
    pos_scale=0.005,
    rot_scale=np.deg2rad(3.0),
    seed=0
):
    data = model.createData()

    if not model.existFrame(ee_frame):
        print(f"\n找不到末端 frame: {ee_frame}")
        print_model_info(model)
        raise RuntimeError("请检查 ee_frame 名字，比如 gripper_base 或 link6")

    frame_id = model.getFrameId(ee_frame)

    lb, ub = get_clean_bounds(model)

    def residual(q):
        M = frame_pose(model, data, q, frame_id)

        pos_err = M.translation - target_M.translation

        # 当前姿态到目标姿态的误差
        rot_err = pin.log3(target_M.rotation.T @ M.rotation)

        return np.concatenate([
            pos_err / pos_scale,
            rot_err / rot_scale
        ])

    rng = np.random.default_rng(seed)

    q_neutral = np.clip(pin.neutral(model), lb, ub)
    q_zero = np.clip(np.zeros(model.nq), lb, ub)

    init_list = [q_neutral, q_zero]

    for _ in range(n_random):
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
            gtol=1e-10
        )

        q = res.x
        M = frame_pose(model, data, q, frame_id)

        pos_err = np.linalg.norm(M.translation - target_M.translation)
        rot_err = np.linalg.norm(
            pin.log3(target_M.rotation.T @ M.rotation)
        )

        score = pos_err + 0.05 * rot_err

        if best is None or score < best["score"]:
            best = {
                "q": q,
                "M": M.copy(),
                "pos_err": pos_err,
                "rot_err": rot_err,
                "score": score,
                "success": res.success
            }

    return best


def check_joint_path(model, q_start, q_goal, steps=50):
    """
    这里只检查关节插值是否越限。
    不检查碰撞、不检查自碰撞。
    """
    lb, ub = get_clean_bounds(model)

    for s in np.linspace(0.0, 1.0, steps):
        q = pin.interpolate(model, q_start, q_goal, s)

        if np.any(q < lb - 1e-6) or np.any(q > ub + 1e-6):
            return False

    return True


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--urdf", type=str, default=DEFAULT_URDF)
    parser.add_argument("--ee", type=str, default=DEFAULT_EE_FRAME)
    parser.add_argument("--target", type=float, nargs=6, default=DEFAULT_TARGET)
    parser.add_argument("--random", type=int, default=100)
    parser.add_argument("--list", action="store_true")

    args = parser.parse_args()

    urdf_path = os.path.expanduser(args.urdf)

    print("URDF:", urdf_path)

    model = pin.buildModelFromUrdf(urdf_path)

    print("\nOriginal model:")
    print("nq:", model.nq)
    print("nv:", model.nv)

    model = reduce_to_arm_only(model)

    print("\nReduced arm model:")
    print("nq:", model.nq)
    print("nv:", model.nv)

    if args.list:
        print_model_info(model)
        return

    target_M = endpose_to_se3(args.target)

    result = solve_ik(
        model=model,
        ee_frame=args.ee,
        target_M=target_M,
        n_random=args.random
    )

    q_sol = result["q"]
    M_sol = result["M"]

    pos_err_mm = result["pos_err"] * 1000.0
    rot_err_deg = np.rad2deg(result["rot_err"])

    reachable = (pos_err_mm < 2.0) and (rot_err_deg < 3.0)

    print("\n========== IK Result ==========")
    print("Reachable:", reachable)
    print(f"Position error: {pos_err_mm:.4f} mm")
    print(f"Rotation error: {rot_err_deg:.4f} deg")

    print("\nq solution [rad]:")
    print(q_sol)

    print("\nq solution [deg]:")
    print(np.rad2deg(q_sol))

    print("\nAchieved end pose [mm, deg]:")
    print(se3_to_endpose(M_sol))

    print("\nTarget end pose [mm, deg]:")
    print(np.array(args.target))

    lb, ub = get_clean_bounds(model)
    q_home = np.clip(pin.neutral(model), lb, ub)

    path_ok = check_joint_path(
        model,
        q_home,
        q_sol,
        steps=50
    )

    print("\nSimple joint-space path from neutral:")
    print("Path OK:", path_ok)

    if not reachable:
        print("\n可能原因：")
        print("1. 目标位姿确实不可达")
        print("2. ee frame 选错，试试 --ee link6")
        print("3. 末端欧拉角顺序和控制器定义不一致")
        print("4. 目标位姿其实是 TCP 位姿，但 URDF 里没有额外 TCP frame")


if __name__ == "__main__":
    main()
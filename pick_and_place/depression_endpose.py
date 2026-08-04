import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import open3d as o3d
import pinocchio as pin
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
PROJECT_ROOT = Path(os.getenv("AAM_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from Piper.endpose_reachability_safe import (
    DEFAULT_EE_FRAME,
    frame_pose,
    get_safe_bounds,
    load_arm_model,
    reachability_test,
    se3_to_endpose,
)


def base_end_grab_trans(endpose):
    t = np.array([[endpose[0]],
                 [endpose[1]],
                 [endpose[2]]] )
    Rot = R.from_euler('xyz',[endpose[3], endpose[4], endpose[5]], degrees = True).as_matrix()
    T = np.column_stack([Rot,t])
    T = np.vstack([T,np.array([0,0,0,1])])
    return T

def trans_to_endpose(T):
    """
    4x4变换矩阵 -> 末端位姿 [x, y, z, rx, ry, rz]
    欧拉角顺序: xyz
    角度单位: degree
    """
    T = np.asarray(T, dtype=float)

    pos = T[:3, 3]
    rot_mat = T[:3, :3]

    euler = R.from_matrix(rot_mat).as_euler('xyz', degrees=True)

    endpose_fix = np.concatenate([pos, euler])
    return endpose_fix


def select_reachable_grab_yaw(grab_position_mm):
    """Keep roll=180/pitch=0 and search the full yaw circle in joint space."""
    grab_position_mm = np.asarray(grab_position_mm, dtype=float).reshape(3)
    target_position_m = grab_position_mm / 1000.0
    target_z = np.array([0.0, 0.0, -1.0])

    model = load_arm_model()
    frame_id = model.getFrameId(DEFAULT_EE_FRAME)
    lb, ub = get_safe_bounds(model)
    rng = np.random.default_rng(0)
    seed_qs = [
        np.clip(pin.neutral(model), lb, ub),
        np.clip(np.zeros(model.nq), lb, ub),
    ]
    seed_qs.extend(rng.uniform(lb, ub) for _ in range(50))

    candidates = []
    for q0 in seed_qs:
        data = model.createData()

        def residual(q):
            pose = frame_pose(model, data, q, frame_id)
            position_residual = (pose.translation - target_position_m) / 0.002
            ee_z = pose.rotation[:, 2]
            axis_residual = np.cross(ee_z, target_z) / np.deg2rad(2.0)
            direction_residual = np.array([(1.0 - np.dot(ee_z, target_z)) / 0.001])
            return np.concatenate([position_residual, axis_residual, direction_residual])

        result = least_squares(
            residual,
            q0,
            bounds=(lb, ub),
            max_nfev=1500,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
        )
        q = result.x
        pose = frame_pose(model, data, q, frame_id)
        position_error_mm = float(
            np.linalg.norm(pose.translation - target_position_m) * 1000.0
        )
        tilt_error_deg = float(np.rad2deg(np.arccos(np.clip(
            np.dot(pose.rotation[:, 2], target_z), -1.0, 1.0
        ))))
        if position_error_mm > 2.0 or tilt_error_deg > 2.0:
            continue

        yaw_deg = float(np.degrees(np.arctan2(
            pose.rotation[1, 0], pose.rotation[0, 0]
        )) % 360.0)
        joint_margin = float(np.min(np.minimum(q - lb, ub - q)))
        candidates.append((joint_margin, position_error_mm, tilt_error_deg, yaw_deg, q))

    if not candidates:
        raise RuntimeError("No reachable grab orientation found for yaw in [0, 360).")

    # Prefer the solution farthest from joint limits, then the smaller FK errors.
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    for _, _, _, yaw_deg, q in candidates:
        grab_endpose = np.array([
            grab_position_mm[0], grab_position_mm[1], grab_position_mm[2],
            180.0, 0.0, yaw_deg,
        ])
        reachable = reachability_test(grab_endpose)
        if reachable["reachable"]:
            return grab_endpose, reachable["joint_degrees"]

    raise RuntimeError("Yaw candidates were found, but none passed the final reachability test.")


def define_pregrab_endpose(endpose):
    """
    Search joint angles whose FK pose lies on the continuous base-Z line
    above grab_endpose: p(s) = p_grab + [0, 0, s], s in [30, 70] mm.
    Keep the grasp orientation unchanged.
    """
    endpose = np.asarray(endpose, dtype=float).reshape(6)
    target_xy_m = endpose[:2] / 1000.0
    z_min_m = (endpose[2] + 30.0) / 1000.0
    z_max_m = (endpose[2] + 70.0) / 1000.0
    target_R = R.from_euler('xyz', endpose[3:], degrees=True).as_matrix()

    position_scale_m = 0.002
    rotation_scale_rad = np.deg2rad(3.0)
    position_tol_m = 0.002
    rotation_tol_deg = 3.0
    random_seed_count = 30

    model = load_arm_model()
    frame_id = model.getFrameId(DEFAULT_EE_FRAME)
    lb, ub = get_safe_bounds(model)

    rng = np.random.default_rng(0)
    seed_qs = [
        np.clip(pin.neutral(model), lb, ub),
        np.clip(np.zeros(model.nq), lb, ub),
    ]
    for _ in range(random_seed_count):
        seed_qs.append(rng.uniform(lb, ub))

    best = None
    print("\n========== Pre-grab continuous line search ==========")
    print("Search line: grab position + base-Z [30, 70] mm")

    for q0 in seed_qs:
        data = model.createData()

        def residual(q):
            pose = frame_pose(model, data, q, frame_id)
            position = pose.translation
            z_outside = position[2] - np.clip(position[2], z_min_m, z_max_m)
            position_residual = np.array([
                position[0] - target_xy_m[0],
                position[1] - target_xy_m[1],
                z_outside,
            ]) / position_scale_m
            rotation_residual = R.from_matrix(target_R.T @ pose.rotation).as_rotvec()
            return np.concatenate([
                position_residual,
                rotation_residual / rotation_scale_rad,
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
        pose = frame_pose(model, data, q, frame_id)
        xy_error_m = float(np.linalg.norm(pose.translation[:2] - target_xy_m))
        z_error_m = float(max(z_min_m - pose.translation[2], 0.0, pose.translation[2] - z_max_m))
        rotation_error_deg = float(np.rad2deg(
            np.linalg.norm(R.from_matrix(target_R.T @ pose.rotation).as_rotvec())
        ))
        feasible = (
            xy_error_m <= position_tol_m
            and z_error_m <= position_tol_m
            and rotation_error_deg <= rotation_tol_deg
        )
        joint_margin = float(np.min(np.minimum(q - lb, ub - q)))
        score = (
            10000.0 if feasible else 0.0
        ) - 1000.0 * xy_error_m - 1000.0 * z_error_m - rotation_error_deg + joint_margin

        if best is None or score > best["score"]:
            best = {
                "score": score,
                "feasible": feasible,
                "pose": pose.copy(),
                "q": q.copy(),
                "xy_error_mm": xy_error_m * 1000.0,
                "z_error_mm": z_error_m * 1000.0,
                "rotation_error_deg": rotation_error_deg,
            }

    if best is None or not best["feasible"]:
        print("Warning: no reachable pregrab pose.")
        return None

    pregrab_endpose = se3_to_endpose(best["pose"])
    print(
        f"Selected pregrab: xy_err={best['xy_error_mm']:.3f} mm, "
        f"z_err={best['z_error_mm']:.3f} mm, "
        f"rot_err={best['rotation_error_deg']:.3f} deg"
    )
    print("Selected pregrab joint degrees:", np.round(np.rad2deg(best["q"]), 3))
    return pregrab_endpose


def define_prefix_endpose(endpose, fix_normal_path, fix_points_path):
    """
    Search joint angles on a world-XY square outside the fix surface.

    The anchor is the fix_points point farthest along n_fix_plane. Move it
    30 mm along n_fix_plane, then use its XY projection as the first corner
    of an 80 mm x 80 mm square extending along world +X and +Y. The input fix
    endpose z and orientation are kept fixed.
    """
    endpose = np.asarray(endpose, dtype=float).reshape(6)

    with open(fix_normal_path, "r", encoding="utf-8") as f:
        normal_data = json.load(f)
    normal = np.asarray(normal_data["n_fix_plane"], dtype=float).reshape(3)
    normal_norm = np.linalg.norm(normal)
    if normal_norm <= 1e-12:
        raise ValueError(f"Invalid zero n_fix_plane in {fix_normal_path}")
    normal = normal / normal_norm

    fix_pcd = o3d.io.read_point_cloud(str(fix_points_path))
    fix_points = np.asarray(fix_pcd.points, dtype=float)
    if len(fix_points) == 0:
        raise RuntimeError(f"No points found in {fix_points_path}")
    anchor = fix_points[np.argmax(fix_points @ normal)]

    first_corner = anchor + 0.030 * normal
    square_min_xy = first_corner[:2]
    square_max_xy = square_min_xy + np.array([0.080, 0.080])
    fixed_z_m = endpose[2] / 1000.0
    target_R = R.from_euler('xyz', endpose[3:], degrees=True).as_matrix()

    position_scale_m = 1e-5
    rotation_scale_rad = np.deg2rad(1e-3)
    numerical_position_tol_m = 1e-6
    numerical_rotation_tol_deg = 1e-4
    random_seed_count = 30

    model = load_arm_model()
    frame_id = model.getFrameId(DEFAULT_EE_FRAME)
    lb, ub = get_safe_bounds(model)

    rng = np.random.default_rng(0)
    seed_qs = [
        np.clip(pin.neutral(model), lb, ub),
        np.clip(np.zeros(model.nq), lb, ub),
    ]
    for _ in range(random_seed_count):
        seed_qs.append(rng.uniform(lb, ub))

    def closest_square_point(position):
        closest_xy = np.clip(position[:2], square_min_xy, square_max_xy)
        return np.array([closest_xy[0], closest_xy[1], fixed_z_m])

    best = None
    for q0 in seed_qs:
        data = model.createData()

        def residual(q):
            pose = frame_pose(model, data, q, frame_id)
            closest_point = closest_square_point(pose.translation)
            position_residual = (pose.translation - closest_point) / position_scale_m
            rotation_residual = R.from_matrix(target_R.T @ pose.rotation).as_rotvec()
            return np.concatenate([
                position_residual,
                rotation_residual / rotation_scale_rad,
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
        pose = frame_pose(model, data, q, frame_id)
        closest_point = closest_square_point(pose.translation)
        position_error_m = float(np.linalg.norm(pose.translation - closest_point))
        rotation_error_deg = float(np.rad2deg(
            np.linalg.norm(R.from_matrix(target_R.T @ pose.rotation).as_rotvec())
        ))
        feasible = (
            position_error_m <= numerical_position_tol_m
            and rotation_error_deg <= numerical_rotation_tol_deg
        )
        joint_margin = float(np.min(np.minimum(q - lb, ub - q)))
        score = (
            10000.0 if feasible else 0.0
        ) - 1000.0 * position_error_m - rotation_error_deg + joint_margin

        if best is None or score > best["score"]:
            best = {
                "score": score,
                "feasible": feasible,
                "pose": pose.copy(),
                "q": q.copy(),
                "position_error_mm": position_error_m * 1000.0,
                "rotation_error_deg": rotation_error_deg,
            }

    if best is None or not best["feasible"]:
        print("Warning: no reachable prefix pose.")
        return None

    selected_xy = np.clip(best["pose"].translation[:2], square_min_xy, square_max_xy) * 1000.0
    prefix_endpose = np.array([
        selected_xy[0],
        selected_xy[1],
        endpose[2],
        endpose[3],
        endpose[4],
        endpose[5],
    ])
    print(
        f"Selected prefix: pos_err={best['position_error_mm']:.3f} mm, "
        f"rot_err={best['rotation_error_deg']:.3f} deg"
    )
    return prefix_endpose

def round_endpose(endpose):
    """
    Keep each element of an endpose to 2 decimal places.
    endpose format: [x, y, z, rx, ry, rz]
    """
    return np.round(np.asarray(endpose, dtype=float).reshape(-1), 2)


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate depression pick/place endposes.")
    parser.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        help="Depression result folder for standalone terminal debugging.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Pipeline run folder: read completion/depression and write pickplace.",
    )
    parser.add_argument("--depression-dir", type=Path, help="Explicit depression result folder.")
    parser.add_argument("--pick-dir", type=Path, help="Explicit pick/place output folder.")
    parser.add_argument("--output", type=Path, help="Explicit output .npz file.")
    parser.add_argument("--fix-points", type=Path, help="Override fix_points.pcd path.")
    parser.add_argument("--fix-normal", type=Path, help="Override n_fix_plane.json path.")
    return parser.parse_args()


def resolve_paths(args):
    standalone_inputs = [path for path in (args.input_dir, args.depression_dir) if path is not None]
    if args.run_dir is not None:
        if standalone_inputs:
            raise ValueError("--run-dir cannot be combined with input_dir or --depression-dir")
        run_dir = args.run_dir.expanduser().resolve()
        depression_path = run_dir / "completion" / "depression"
        pick_path = args.pick_dir.expanduser().resolve() if args.pick_dir else run_dir / "pickplace"
    else:
        if len(standalone_inputs) != 1:
            raise ValueError("Specify exactly one of --run-dir, input_dir, or --depression-dir")
        depression_path = standalone_inputs[0].expanduser().resolve()
        if args.pick_dir is None and args.output is None:
            raise ValueError("Standalone mode requires --pick-dir or --output")
        pick_path = args.pick_dir.expanduser().resolve() if args.pick_dir else args.output.parent.resolve()

    output_path = args.output.expanduser().resolve() if args.output else pick_path / "pick_place_endpose.npz"
    return depression_path, pick_path, output_path


args = parse_args()
depression_path, pick_path, output_path = resolve_paths(args)
fix_points_path = args.fix_points.expanduser().resolve() if args.fix_points else depression_path / "fix_points.pcd"
fix_normal_path = args.fix_normal.expanduser().resolve() if args.fix_normal else depression_path / "n_fix_plane.json"


arm_gripper_length_z = 148 # mm 末端Z轴朝前伸出方向  142.5
arm_gripper_width_y = 164 # mm  ## 夹爪开合方向
arm_gripper_thickness =75 # mm 
arm_flange = 10.5+2 # mm

printerx = -266.425 #mm
printery = 184.39
printerz = 60.8
 
depression_dir = str(depression_path)
pick_dir = str(pick_path)

model_path = depression_dir + '/model_oriented.stl'
pcd_path = depression_dir + '/model_oriented.pcd'
meta = np.load(depression_dir + '/meta.npz', allow_pickle=True)
orient_meta = np.load(depression_dir + '/orientation_meta.npz', allow_pickle=True)
grip_meta = np.load(depression_dir + '/gripper_meta.npz', allow_pickle=True)
unit_scale = 1000

attach_center = np.asarray(orient_meta['attach_center_oriented'] ) * unit_scale
full_box_z_height = np.asarray(orient_meta['full_box_z_height'] ) * unit_scale
grip_height_total = (grip_meta['grip_body_height'] + grip_meta['base_height'] + grip_meta['v_neck_height']) * unit_scale

####
#### 打印物体抓取坐标系相对base坐标系的变换矩阵 #######
####

theta = np.deg2rad(90)
Rz = np.array([
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta),  np.cos(theta), 0],
    [0,              0,             1]
])
b_obj_grab_t = np.array([[printerx],[printery],[printerz ]])  ## 固定值代表打印盘中心相对base的位置
b_obj_grab_T = np.column_stack([Rz, b_obj_grab_t])
b_obj_grab_T = np.vstack([b_obj_grab_T,np.array([0,0,0,1])])


####
#### 定义抓取位姿: x, y follow attach_center, z 下降到GRIP的一半长度处
####

# 在object_grab坐标系下的抓取位置(仅考虑gripper)

#p_grab_pos = np.array([attach_center[0], top_plane_center[1], top_plane_center[2]*2 + grip_height_total/2 ,1])
obj_grabx = attach_center[0]
obj_graby = attach_center[1] 
obj_grabz = (full_box_z_height - grip_height_total/2 )
obj_grab_pos = np.array([obj_grabx, obj_graby , obj_grabz, 1 ])

#base坐标系下的抓取endpose
b_grab_pos = b_obj_grab_T @ obj_grab_pos
#b_pre_end_grab  = np.array([b_grab_pos[0], b_grab_pos[1], b_grab_pos[2]+pre_grab_height,180,0,0])# mm,degree
grab_position = np.array([
    b_grab_pos[0],
    b_grab_pos[1],
    b_grab_pos[2] + arm_gripper_length_z,
])
grab_endpose, grab_joint_degrees = select_reachable_grab_yaw(grab_position)

#抓取末端姿态在base下的变换矩阵
base_end_grab_T = base_end_grab_trans(grab_endpose)

# object_grab 在 末端抓取时刻坐标系下的表示
end_grab_obj_grab_T = np.linalg.inv(base_end_grab_T) @ b_obj_grab_T 

base_obj_fix_T = orient_meta['base_T_full']
### 计算最终的安装旋转矩阵
base_end_fix_T =  base_obj_fix_T @ np.linalg.inv(end_grab_obj_grab_T)
fix_endpose = trans_to_endpose(base_end_fix_T)
#fix_endpose[0] += 0.003
#fix_endpose[1] -= 0.003

fix_reachable = reachability_test(fix_endpose)
if fix_reachable["reachable"]:
    print('😍  fix enpose reachable! 😍 ')
    fix_joint_degrees = fix_reachable["joint_degrees"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path,
    #pregrab_endpose = pregrab_endpose,
    #prefix_endpose = prefix_endpose,
    grab_endpose = grab_endpose,
    fix_endpose = fix_endpose   )

    json_output_path = output_path.with_suffix('.json')
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'garb_joint_degrees':grab_joint_degrees,
            #"pre_fix_endpose": prefix_endpose.tolist(),
            #"pre_fix_joint_degrees": pre_fix_joint_degrees,
            "fix_endpose": fix_endpose.tolist(),
            "fix_joint_degrees": fix_joint_degrees,
        }, f, ensure_ascii=False, indent=2)
    print('saved:', json_output_path)
    print('pick joint degrees: \n ', grab_joint_degrees , '\n \n fix joint degrees: \n ', fix_joint_degrees)


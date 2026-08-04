import argparse
import json
import os
from pathlib import Path

import numpy as np
import open3d as o3d


# =========================
# Paths / parameters
# =========================
PROJECT_ROOT = Path('/home/smmg/AAM')
DATA_DIR = PROJECT_ROOT / "construction" / "data"
COARSE_DIR = DATA_DIR / "coarse_scan"
FINE_DIR = DATA_DIR / "fine_scan"

COARSE_POINT_FILE = COARSE_DIR / "coarse_point_result.npz"
COARSE_SEQ_FILE = COARSE_DIR / "coarse_png_sequence.json"
COARSE_SCANPOSE_FILE = COARSE_DIR / "coarse_scanpose.json"
FINE_POINT_FILE = FINE_DIR / "fine_point_result.npz"
FINE_SEQ_FILE = FINE_DIR / "fine_png_sequence.json"

OUTPUT_FUSED_PCD = FINE_DIR / "fine_fuse.pcd"
OUTPUT_RESULT_JSON = FINE_DIR / "fine_icp_result.json"
OUTPUT_TRANSFORM_NPZ = FINE_DIR / "fine_icp_transforms.npz"

# preprocess
VOXEL_SIZE = 0.002
STAT_NB_NEIGHBORS = 30
STAT_STD_RATIO = 2.5
NORMAL_RADIUS = 0.015
NORMAL_MAX_NN = 20

# ICP
ICP_DISTANCE_THRESHOLD = 0.015
ICP_MAX_ITERATION = 100
MIN_FITNESS = 0.70
MAX_RMSE = 0.005
MAX_TRANSLATION = 0.08
MAX_ROTATION_DEG = 15.0
MIN_OVERLAP_RATIO = 0.50
OVERLAP_DISTANCE_THRESHOLD = 0.01   # 10 mm，配准后 source 点到 target 最近点距离阈值

FUSED_VOXEL_SIZE = 0.001
VISUALIZE = False
TARGET_CUBE_NAME = "front_high"
FALLBACK_TARGET_CUBE_NAME = "left_high"
TARGET_MIN_POINTS = 500
TARGET_MAX_POINTS = 80000


def configure_paths(args):
    global DATA_DIR, COARSE_DIR, FINE_DIR, COARSE_POINT_FILE, COARSE_SEQ_FILE
    global COARSE_SCANPOSE_FILE, FINE_POINT_FILE, FINE_SEQ_FILE, OUTPUT_FUSED_PCD, OUTPUT_RESULT_JSON
    global OUTPUT_TRANSFORM_NPZ, VISUALIZE

    if args.run_dir:
        DATA_DIR = Path(args.run_dir) / "construction"
        COARSE_DIR = DATA_DIR / "coarse_scan"
        FINE_DIR = DATA_DIR / "fine_scan"

    if args.coarse_scan_dir:
        COARSE_DIR = Path(args.coarse_scan_dir)
    if args.fine_scan_dir:
        FINE_DIR = Path(args.fine_scan_dir)

    COARSE_POINT_FILE = Path(args.coarse_point_file) if args.coarse_point_file else COARSE_DIR / "coarse_point_result.npz"
    COARSE_SEQ_FILE = Path(args.coarse_seq_file) if args.coarse_seq_file else COARSE_DIR / "coarse_png_sequence.json"
    COARSE_SCANPOSE_FILE = COARSE_DIR / "coarse_scanpose.json"
    FINE_POINT_FILE = Path(args.fine_point_file) if args.fine_point_file else FINE_DIR / "fine_point_result.npz"
    FINE_SEQ_FILE = Path(args.fine_seq_file) if args.fine_seq_file else FINE_DIR / "fine_png_sequence.json"
    OUTPUT_FUSED_PCD = Path(args.output_pcd) if args.output_pcd else FINE_DIR / "fine_fuse.pcd"
    OUTPUT_RESULT_JSON = Path(args.output_json) if args.output_json else FINE_DIR / "fine_icp_result.json"
    OUTPUT_TRANSFORM_NPZ = Path(args.output_transforms) if args.output_transforms else FINE_DIR / "fine_icp_transforms.npz"
    VISUALIZE = bool(args.visualize)

    FINE_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# IO
# =========================
def load_json(path):
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def points_to_xyz(points):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2:
        points = points.reshape(-1, points.shape[-1])
    points = points[:, :3]
    return points[np.all(np.isfinite(points), axis=1)]


def load_points_from_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)

    # 优先读 packed 格式；没有则读 points_collection object array
    if "all_points" in data and "offsets" in data:
        all_points = np.asarray(data["all_points"], dtype=float)
        offsets = np.asarray(data["offsets"], dtype=int)
        return [points_to_xyz(all_points[offsets[i]:offsets[i + 1]]) for i in range(len(offsets) - 1)]

    pc = data["points_collection"].tolist()
    return [points_to_xyz(p) for p in pc]


def load_dataset(point_file, seq_file, prefix):
    points_list = load_points_from_npz(point_file)
    seq = load_json(seq_file)
    if len(seq) != len(points_list):
        seq = [f"{prefix}_{i}" for i in range(len(points_list))]

    items = []
    for i, pts in enumerate(points_list):
        items.append({
            "dataset": prefix,
            "name": seq[i],
            "global_name": f"{prefix}:{seq[i]}",
            "points": pts,
        })
    return items


def load_cube_name_to_frame_name(scanpose_file):
    records = load_json(scanpose_file)
    mapping = {}
    for record in records:
        cube_name = record.get("cube_name")
        idx = record.get("idx")
        if cube_name is None or idx is None:
            continue
        mapping[str(cube_name)] = f"coarse_scan_{int(idx) + 1}"
    return mapping


def is_target_points_normal(raw_count, processed_count):
    return TARGET_MIN_POINTS <= raw_count <= TARGET_MAX_POINTS and processed_count > 0


def select_target_idx(items, raw_counts, proc_counts):
    cube_to_frame = load_cube_name_to_frame_name(COARSE_SCANPOSE_FILE)
    coarse_name_to_idx = {
        item["name"]: i
        for i, item in enumerate(items)
        if item["dataset"] == "coarse"
    }

    front_frame = cube_to_frame.get(TARGET_CUBE_NAME)
    left_frame = cube_to_frame.get(FALLBACK_TARGET_CUBE_NAME)

    front_idx = coarse_name_to_idx.get(front_frame)
    if front_idx is not None and is_target_points_normal(raw_counts[front_idx], proc_counts[front_idx]):
        return front_idx, {
            "rule": "front_high preferred",
            "selected_cube_name": TARGET_CUBE_NAME,
            "selected_frame": front_frame,
            "front_high_raw_points": int(raw_counts[front_idx]),
            "front_high_processed_points": int(proc_counts[front_idx]),
        }

    left_idx = coarse_name_to_idx.get(left_frame)
    if left_idx is not None and proc_counts[left_idx] > 0:
        return left_idx, {
            "rule": "front_high abnormal, fallback to left_high",
            "selected_cube_name": FALLBACK_TARGET_CUBE_NAME,
            "selected_frame": left_frame,
            "front_high_frame": front_frame,
            "front_high_raw_points": int(raw_counts[front_idx]) if front_idx is not None else None,
            "front_high_processed_points": int(proc_counts[front_idx]) if front_idx is not None else None,
            "target_min_points": TARGET_MIN_POINTS,
            "target_max_points": TARGET_MAX_POINTS,
        }

    valid_idx = [i for i, c in enumerate(raw_counts) if c > 0 and proc_counts[i] > 0]
    target_idx = max(valid_idx, key=lambda i: raw_counts[i])
    return target_idx, {
        "rule": "front_high/left_high missing or empty, fallback to max raw points",
        "selected_cube_name": None,
        "selected_frame": items[target_idx]["name"],
        "target_min_points": TARGET_MIN_POINTS,
        "target_max_points": TARGET_MAX_POINTS,
    }


# =========================
# Point cloud utils
# =========================
def numpy_to_pcd(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float))
    return pcd


def estimate_normals(pcd):
    if len(pcd.points) == 0:
        return pcd
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=NORMAL_RADIUS,
            max_nn=NORMAL_MAX_NN,
        )
    )
    pcd.normalize_normals()
    return pcd


def preprocess(points):
    pcd = numpy_to_pcd(points_to_xyz(points))
    if len(pcd.points) == 0:
        return pcd

    #if len(pcd.points) > STAT_NB_NEIGHBORS:
    #    _, ind = pcd.remove_statistical_outlier(
    #        nb_neighbors=STAT_NB_NEIGHBORS,
    #        std_ratio=STAT_STD_RATIO,
    #        print_progress=False,
    #    )
    #    pcd = pcd.select_by_index(ind)

    #pcd = pcd.voxel_down_sample(VOXEL_SIZE)
    estimate_normals(pcd)
    return pcd


def copy_pcd(pcd):
    new = o3d.geometry.PointCloud()
    new.points = o3d.utility.Vector3dVector(np.asarray(pcd.points).copy())
    if pcd.has_normals():
        new.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals).copy())
    if pcd.has_colors():
        new.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors).copy())
    return new


def run_icp(source, target):
    estimate_normals(source)
    estimate_normals(target)
    return o3d.pipelines.registration.registration_icp(
        source,
        target,
        ICP_DISTANCE_THRESHOLD,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=ICP_MAX_ITERATION),
    )

def rotation_angle_deg(T):
    R = np.asarray(T[:3, :3], dtype=float)
    value = (np.trace(R) - 1.0) / 2.0
    value = np.clip(value, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(value)))


def compute_overlap_ratio(source, target, T, threshold=OVERLAP_DISTANCE_THRESHOLD):
    if len(source.points) == 0 or len(target.points) == 0:
        return 0.0

    src = copy_pcd(source)
    src.transform(T)

    tree = o3d.geometry.KDTreeFlann(target)
    src_pts = np.asarray(src.points)

    matched = 0
    threshold2 = threshold * threshold

    for p in src_pts:
        k, idx, dist2 = tree.search_knn_vector_3d(p, 1)
        if k > 0 and dist2[0] < threshold2:
            matched += 1

    return float(matched / len(src_pts))

# =========================
# Main ICP fusion
# =========================
def main():
    FINE_DIR.mkdir(parents=True, exist_ok=True)

    items = []
    items += load_dataset(COARSE_POINT_FILE, COARSE_SEQ_FILE, "coarse")
    items += load_dataset(FINE_POINT_FILE, FINE_SEQ_FILE, "fine")

    processed = []
    raw_counts = []
    proc_counts = []

    for item in items:
        raw_counts.append(len(item["points"]))
        pcd = preprocess(item["points"])
        processed.append(pcd)
        proc_counts.append(len(pcd.points))

    target_idx, target_selection = select_target_idx(items, raw_counts, proc_counts)
    target = processed[target_idx]
    estimate_normals(target)

    print("========== Fine ICP target ==========")
    print("target:", items[target_idx]["global_name"])
    print("raw points:", raw_counts[target_idx])
    print("processed points:", proc_counts[target_idx])
    print("target selection:", target_selection)

    fused_list = [copy_pcd(target)]
    transforms = []
    records = []

    for i, pcd in enumerate(processed):
        T = np.eye(4)
        record = {
            "index": i,
            "dataset": items[i]["dataset"],
            "name": items[i]["name"],
            "global_name": items[i]["global_name"],
            "raw_points": raw_counts[i],
            "processed_points": proc_counts[i],
            "is_target": i == target_idx,
            "kept": False,
            "fitness": None,
            "inlier_rmse": None,
            "translation_norm_m": 0.0,
            "rotation_angle_deg": 0.0,
            "overlap_ratio": 1.0,
            "transformation": T.tolist(),
            "reason": "",
        }

        if i == target_idx:
            record["kept"] = True
            record["reason"] = "target frame"
            transforms.append(T)
            records.append(record)
            continue

        if len(pcd.points) == 0:
            record["reason"] = "empty cloud"
            transforms.append(T)
            records.append(record)
            continue

        print(f"\nICP source: {items[i]['global_name']}")
        reg = run_icp(pcd, target)
        T = np.asarray(reg.transformation, dtype=float)
        fitness = float(reg.fitness)
        rmse = float(reg.inlier_rmse)
        trans_norm = float(np.linalg.norm(T[:3, 3]))
        rot_deg = rotation_angle_deg(T)
        overlap_ratio = compute_overlap_ratio(pcd, target, T)

        keep = True
        reason = "kept"

        # 过滤条件：有效匹配点少 / rmse 过大
        if fitness < MIN_FITNESS or rmse > MAX_RMSE:
            keep = False
            reason = "bad quality: fitness too low or rmse too large"

        # 平移过多，认为配准滑动
        if keep and trans_norm > MAX_TRANSLATION:
            keep = False
            reason = "translation too large"

        # 旋转过大，认为配准异常
        if keep and rot_deg > MAX_ROTATION_DEG:
            keep = False
            reason = "rotation too large"

        # overlap 太低，说明 source 和 target 重叠区域不足
        if keep and overlap_ratio < MIN_OVERLAP_RATIO:
            keep = False
            reason = "overlap ratio too low"    

        print(f"fitness: {fitness:.4f}, rmse: {rmse:.6f}, trans: {trans_norm:.5f}, keep: {keep}")
        print(reason)

        record.update({
            "kept": bool(keep),
            "fitness": fitness,
            "inlier_rmse": rmse,
            "translation_norm_m": trans_norm,
            "transformation": T.tolist(),
            "reason": reason,
        })
        transforms.append(T)
        records.append(record)

        if keep:
            aligned = copy_pcd(pcd)
            aligned.transform(T)
            fused_list.append(aligned)

    fused = o3d.geometry.PointCloud()
    for pcd in fused_list:
        fused += pcd

    fused = fused.voxel_down_sample(FUSED_VOXEL_SIZE)
    #if len(fused.points) > STAT_NB_NEIGHBORS:
    #    _, ind = fused.remove_statistical_outlier(
    #        nb_neighbors=STAT_NB_NEIGHBORS,
    #        std_ratio=STAT_STD_RATIO,
    #        print_progress=False,
    #    )
    #    fused = fused.select_by_index(ind)
    estimate_normals(fused)

    summary = {
        "coarse_point_file": str(COARSE_POINT_FILE),
        "fine_point_file": str(FINE_POINT_FILE),
        "target_index": int(target_idx),
        "target_dataset": items[target_idx]["dataset"],
        "target_name": items[target_idx]["name"],
        "target_global_name": items[target_idx]["global_name"],
        "target_selection": target_selection,
        "filter_rule": {
            "min_fitness": MIN_FITNESS,
            "max_inlier_rmse": MAX_RMSE,
            "max_translation_m": MAX_TRANSLATION,
        },
        "voxel_size": VOXEL_SIZE,
        "icp_distance_threshold": ICP_DISTANCE_THRESHOLD,
        "fused_points": int(len(fused.points)),
        "records": records,
    }

    o3d.io.write_point_cloud(str(OUTPUT_FUSED_PCD), fused)
    np.savez(
        OUTPUT_TRANSFORM_NPZ,
        transforms=np.asarray(transforms, dtype=float),
        names=np.asarray([item["global_name"] for item in items], dtype=object),
    )
    save_json(OUTPUT_RESULT_JSON, summary)

    print("\n========== Fine ICP result ==========")
    print("saved fused pcd:", OUTPUT_FUSED_PCD)
    print("saved result json:", OUTPUT_RESULT_JSON)
    print("saved transforms:", OUTPUT_TRANSFORM_NPZ)
    print("fused points:", len(fused.points))

    if VISUALIZE:
        vis = copy_pcd(fused)
        vis.paint_uniform_color([0.2, 0.7, 1.0])
        o3d.visualization.draw_geometries([vis], window_name="Fine ICP fused point cloud")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--coarse-scan-dir", type=Path, default=None)
    parser.add_argument("--fine-scan-dir", type=Path, default=None)
    parser.add_argument("--coarse-point-file", type=Path, default=None)
    parser.add_argument("--coarse-seq-file", type=Path, default=None)
    parser.add_argument("--fine-point-file", type=Path, default=None)
    parser.add_argument("--fine-seq-file", type=Path, default=None)
    parser.add_argument("--output-pcd", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-transforms", type=Path, default=None)
    parser.add_argument("--visualize", action="store_true")
    configure_paths(parser.parse_args())
    main()

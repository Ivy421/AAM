import argparse
import json
import os
from pathlib import Path

import numpy as np
import open3d as o3d

from coarse_icp import (
    FUSED_VOXEL_SIZE,
    MAX_RMSE,
    MAX_ROTATION_DEG,
    MAX_TRANSLATION,
    MIN_FITNESS,
    MIN_OVERLAP_RATIO,
    STAT_NB_NEIGHBORS,
    STAT_STD_RATIO,
    copy_pcd,
    compute_overlap_ratio,
    estimate_normals,
    load_dataset,
    preprocess,
    rotation_angle_deg,
    run_icp,
)


PROJECT_ROOT = Path(os.getenv("AAM_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
DATA_DIR = PROJECT_ROOT / "construction" / "data"
COARSE_DIR = DATA_DIR / "coarse_scan"

COARSE_POINT_FILE = COARSE_DIR / "coarse_point_result.npz"
COARSE_SEQ_FILE = COARSE_DIR / "coarse_png_sequence.json"
COARSE_SCANPOSE_FILE = COARSE_DIR / "coarse_scanpose.json"
OUTPUT_FUSED_PCD = COARSE_DIR / "coarse_fuse.pcd"
OUTPUT_RESULT_JSON = COARSE_DIR / "coarse_icp_result.json"

TARGET_CUBE_NAME = "front_high"
FALLBACK_TARGET_CUBE_NAME = "left_high"
TARGET_MIN_POINTS = 500
TARGET_MAX_POINTS = 80000


def configure_paths(args):
    global DATA_DIR, COARSE_DIR, COARSE_POINT_FILE, COARSE_SEQ_FILE, COARSE_SCANPOSE_FILE
    global OUTPUT_FUSED_PCD, OUTPUT_RESULT_JSON

    if args.run_dir:
        DATA_DIR = Path(args.run_dir) / "construction"
        COARSE_DIR = DATA_DIR / "coarse_scan"

    if args.coarse_scan_dir:
        COARSE_DIR = Path(args.coarse_scan_dir)

    COARSE_POINT_FILE = Path(args.coarse_point_file) if args.coarse_point_file else COARSE_DIR / "coarse_point_result.npz"
    COARSE_SEQ_FILE = Path(args.coarse_seq_file) if args.coarse_seq_file else COARSE_DIR / "coarse_png_sequence.json"
    COARSE_SCANPOSE_FILE = COARSE_DIR / "coarse_scanpose.json"
    OUTPUT_FUSED_PCD = Path(args.output_pcd) if args.output_pcd else COARSE_DIR / "coarse_fuse.pcd"
    OUTPUT_RESULT_JSON = Path(args.output_json) if args.output_json else COARSE_DIR / "coarse_icp_result.json"

    COARSE_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_cube_name(name):
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def load_cube_name_to_frame_name(scanpose_file):
    records = load_json(scanpose_file)
    mapping = {}
    for record in records:
        cube_name = record.get("cube_name")
        idx = record.get("idx")
        if cube_name is None or idx is None:
            continue
        mapping[normalize_cube_name(cube_name)] = f"coarse_scan_{int(idx) + 1}"
    return mapping


def is_target_points_normal(raw_count, processed_count):
    return TARGET_MIN_POINTS <= raw_count <= TARGET_MAX_POINTS and processed_count > 0


def select_target_idx(items, raw_counts, proc_counts):
    cube_to_frame = load_cube_name_to_frame_name(COARSE_SCANPOSE_FILE)
    name_to_idx = {item["name"]: i for i, item in enumerate(items)}

    front_frame = cube_to_frame.get(normalize_cube_name(TARGET_CUBE_NAME))
    left_frame = cube_to_frame.get(normalize_cube_name(FALLBACK_TARGET_CUBE_NAME))

    front_idx = name_to_idx.get(front_frame)
    if front_idx is not None and is_target_points_normal(raw_counts[front_idx], proc_counts[front_idx]):
        target_idx = front_idx
        return front_idx, {
            "rule": "front_high preferred",
            "selected_cube_name": TARGET_CUBE_NAME,
            "selected_frame": front_frame,
            "front_high_raw_points": int(raw_counts[front_idx]),
            "front_high_processed_points": int(proc_counts[front_idx]),
        }


    left_idx = name_to_idx.get(left_frame)
    if left_idx is not None:
        target_idx = left_idx
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
    #target_idx = max(valid_idx, key=lambda i: raw_counts[i])
    
    return target_idx, {
        "rule": "front_high/left_high missing, fallback to max raw points",
        "selected_cube_name": None,
        "selected_frame": items[target_idx]["name"],
        "target_min_points": TARGET_MIN_POINTS,
        "target_max_points": TARGET_MAX_POINTS,
    }


def pass_filter(fitness, rmse, trans_norm, rot_deg, overlap_ratio):
    if fitness < MIN_FITNESS or rmse > MAX_RMSE:
        return False, "bad quality: fitness too low or rmse too large"
    if trans_norm > MAX_TRANSLATION:
        return False, "translation too large"
    if rot_deg > MAX_ROTATION_DEG:
        return False, "rotation too large"
    if overlap_ratio < MIN_OVERLAP_RATIO:
        return False, "overlap ratio too low"
    return True, "kept"


def main():
    items = load_dataset(COARSE_POINT_FILE, COARSE_SEQ_FILE, "coarse")

    processed = []
    raw_counts = []
    proc_counts = []
    for item in items:
        raw_counts.append(len(item["points"]))
        pcd = preprocess(item["points"])
        processed.append(pcd)
        proc_counts.append(len(pcd.points))

    target_idx, target_selection = select_target_idx(items, raw_counts, proc_counts)
    fused = copy_pcd(processed[target_idx])
    estimate_normals(fused)

    records = []
    transforms = []

    print("========== Additive Coarse ICP target ==========")
    print("target:", items[target_idx]["global_name"])
    print("raw points:", raw_counts[target_idx])
    print("processed points:", proc_counts[target_idx])
    print("target selection:", target_selection)

    for i, source in enumerate(processed):
        T = np.eye(4)
        record = {
            "index": i,
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
            records.append(record)
            transforms.append(T)
            continue

        if len(source.points) == 0:
            record["reason"] = "empty cloud"
            records.append(record)
            transforms.append(T)
            continue

        print(f"\nICP source -> current fused: {items[i]['global_name']}")
        reg = run_icp(source, fused)
        T = np.asarray(reg.transformation, dtype=float)
        fitness = float(reg.fitness)
        rmse = float(reg.inlier_rmse)
        trans_norm = float(np.linalg.norm(T[:3, 3]))
        rot_deg = rotation_angle_deg(T)
        overlap_ratio = compute_overlap_ratio(source, fused, T)
        keep, reason = pass_filter(fitness, rmse, trans_norm, rot_deg, overlap_ratio)

        print(
            f"fitness: {fitness:.4f}, "
            f"rmse: {rmse:.6f}, "
            f"trans: {trans_norm:.5f}, "
            f"rot: {rot_deg:.2f} deg, "
            f"overlap: {overlap_ratio:.3f}, "
            f"keep: {keep}"
        )
        print(reason)

        record.update({
            "kept": bool(keep),
            "fitness": fitness,
            "inlier_rmse": rmse,
            "translation_norm_m": trans_norm,
            "rotation_angle_deg": rot_deg,
            "overlap_ratio": overlap_ratio,
            "transformation": T.tolist(),
            "reason": reason,
        })
        records.append(record)
        transforms.append(T)

        if keep:
            aligned = copy_pcd(source)
            aligned.transform(T)
            fused += aligned
            fused = fused.voxel_down_sample(FUSED_VOXEL_SIZE)
            if len(fused.points) > STAT_NB_NEIGHBORS:
                _, ind = fused.remove_statistical_outlier(
                    nb_neighbors=STAT_NB_NEIGHBORS,
                    std_ratio=STAT_STD_RATIO,
                    print_progress=False,
                )
                fused = fused.select_by_index(ind)
            estimate_normals(fused)

    o3d.io.write_point_cloud(str(OUTPUT_FUSED_PCD), fused)
    save_json(OUTPUT_RESULT_JSON, {
        "mode": "additive_icp",
        "coarse_point_file": str(COARSE_POINT_FILE),
        "coarse_seq_file": str(COARSE_SEQ_FILE),
        "output_fused_pcd": str(OUTPUT_FUSED_PCD),
        "target_index": int(target_idx),
        "target_name": items[target_idx]["name"],
        "target_global_name": items[target_idx]["global_name"],
        "target_selection": target_selection,
        "fused_points": int(len(fused.points)),
        "filter_rule": {
            "min_fitness": MIN_FITNESS,
            "max_inlier_rmse": MAX_RMSE,
            "max_translation_m": MAX_TRANSLATION,
            "max_rotation_deg": MAX_ROTATION_DEG,
            "min_overlap_ratio": MIN_OVERLAP_RATIO,
        },
        "records": records,
    })

    print("\n========== Additive Coarse ICP result ==========")
    print("saved fused pcd:", OUTPUT_FUSED_PCD)
    print("saved result json:", OUTPUT_RESULT_JSON)
    print("fused points:", len(fused.points))

    o3d.visualization.draw_geometries([fused], window_name="coarse_fuse.pcd")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--coarse-scan-dir", type=Path, default=None)
    parser.add_argument("--coarse-point-file", type=Path, default=None)
    parser.add_argument("--coarse-seq-file", type=Path, default=None)
    parser.add_argument("--output-pcd", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    configure_paths(parser.parse_args())
    main()

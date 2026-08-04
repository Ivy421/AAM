import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d

import fine_icp as base


OUTPUT_FUSED_PCD = base.FINE_DIR / "fine_fuse.pcd"  #fine_fuse_add
OUTPUT_RESULT_JSON = base.FINE_DIR / "fine_icp_result.json"  #fine_icp_add_result
OUTPUT_TRANSFORM_NPZ = base.FINE_DIR / "fine_icp_transforms.npz"


def configure_paths(args):
    global OUTPUT_FUSED_PCD, OUTPUT_RESULT_JSON, OUTPUT_TRANSFORM_NPZ

    base.configure_paths(args)

    OUTPUT_FUSED_PCD = Path(args.output_pcd) if args.output_pcd else base.FINE_DIR / "fine_fuse.pcd"
    OUTPUT_RESULT_JSON = Path(args.output_json) if args.output_json else base.FINE_DIR / "fine_icp_result.json"
    OUTPUT_TRANSFORM_NPZ = Path(args.output_transforms) if args.output_transforms else base.FINE_DIR / "fine_icp_transforms.npz"

    base.FINE_DIR.mkdir(parents=True, exist_ok=True)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def pass_filter(fitness, rmse, trans_norm, rot_deg, overlap_ratio):
    if fitness < base.MIN_FITNESS or rmse > base.MAX_RMSE:
        return False, "bad quality: fitness too low or rmse too large"
    if trans_norm > base.MAX_TRANSLATION:
        return False, "translation too large"
    if rot_deg > base.MAX_ROTATION_DEG:
        return False, "rotation too large"
    if overlap_ratio < base.MIN_OVERLAP_RATIO:
        return False, "overlap ratio too low"
    return True, "kept"


def main():
    base.FINE_DIR.mkdir(parents=True, exist_ok=True)

    items = []
    items += base.load_dataset(base.COARSE_POINT_FILE, base.COARSE_SEQ_FILE, "coarse")
    items += base.load_dataset(base.FINE_POINT_FILE, base.FINE_SEQ_FILE, "fine")

    processed = []
    raw_counts = []
    proc_counts = []

    for item in items:
        raw_counts.append(len(item["points"]))
        pcd = base.preprocess(item["points"])
        processed.append(pcd)
        proc_counts.append(len(pcd.points))

    target_idx, target_selection = base.select_target_idx(items, raw_counts, proc_counts)
    fused = base.copy_pcd(processed[target_idx])
    base.estimate_normals(fused)

    records = []
    transforms = []

    print("========== Additive Fine ICP target ==========")
    print("target:", items[target_idx]["global_name"])
    print("raw points:", raw_counts[target_idx])
    print("processed points:", proc_counts[target_idx])
    print("target selection:", target_selection)

    for i, source in enumerate(processed):
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
            records.append(record)
            transforms.append(T)
            continue

        if len(source.points) == 0:
            record["reason"] = "empty cloud"
            records.append(record)
            transforms.append(T)
            continue

        print(f"\nICP source -> current fused: {items[i]['global_name']}")

        reg = base.run_icp(source, fused)
        T = np.asarray(reg.transformation, dtype=float)
        fitness = float(reg.fitness)
        rmse = float(reg.inlier_rmse)
        trans_norm = float(np.linalg.norm(T[:3, 3]))
        rot_deg = base.rotation_angle_deg(T)
        overlap_ratio = base.compute_overlap_ratio(source, fused, T)

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
            aligned = base.copy_pcd(source)
            aligned.transform(T)
            fused += aligned
            fused = fused.voxel_down_sample(base.FUSED_VOXEL_SIZE)

            if len(fused.points) > base.STAT_NB_NEIGHBORS:
                _, ind = fused.remove_statistical_outlier(
                    nb_neighbors=base.STAT_NB_NEIGHBORS,
                    std_ratio=base.STAT_STD_RATIO,
                    print_progress=False,
                )
                fused = fused.select_by_index(ind)

            base.estimate_normals(fused)

    o3d.io.write_point_cloud(str(OUTPUT_FUSED_PCD), fused)
    np.savez(
        OUTPUT_TRANSFORM_NPZ,
        transforms=np.asarray(transforms, dtype=float),
        names=np.asarray([item["global_name"] for item in items], dtype=object),
    )

    save_json(OUTPUT_RESULT_JSON, {
        "mode": "additive_icp",
        "coarse_point_file": str(base.COARSE_POINT_FILE),
        "fine_point_file": str(base.FINE_POINT_FILE),
        "coarse_seq_file": str(base.COARSE_SEQ_FILE),
        "fine_seq_file": str(base.FINE_SEQ_FILE),
        "output_fused_pcd": str(OUTPUT_FUSED_PCD),
        "output_transforms_npz": str(OUTPUT_TRANSFORM_NPZ),
        "target_index": int(target_idx),
        "target_dataset": items[target_idx]["dataset"],
        "target_name": items[target_idx]["name"],
        "target_global_name": items[target_idx]["global_name"],
        "target_selection": target_selection,
        "fused_points": int(len(fused.points)),
        "filter_rule": {
            "min_fitness": base.MIN_FITNESS,
            "max_inlier_rmse": base.MAX_RMSE,
            "max_translation_m": base.MAX_TRANSLATION,
            "max_rotation_deg": base.MAX_ROTATION_DEG,
            "min_overlap_ratio": base.MIN_OVERLAP_RATIO,
            "overlap_distance_threshold_m": base.OVERLAP_DISTANCE_THRESHOLD,
        },
        "voxel_size": base.VOXEL_SIZE,
        "icp_distance_threshold": base.ICP_DISTANCE_THRESHOLD,
        "records": records,
    })

    print("\n========== Additive Fine ICP result ==========")
    print("saved fused pcd:", OUTPUT_FUSED_PCD)
    print("saved result json:", OUTPUT_RESULT_JSON)
    print("saved transforms:", OUTPUT_TRANSFORM_NPZ)
    print("fused points:", len(fused.points))

    if base.VISUALIZE:
        vis = base.copy_pcd(fused)
        vis.paint_uniform_color([0.2, 0.7, 1.0])
        o3d.visualization.draw_geometries([vis], window_name="fine_fuse.pcd")


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

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

import fine_icp as base


OUTPUT_FUSED_PCD = base.FINE_DIR / "fine_fuse_curvature.pcd"
OUTPUT_RESULT_JSON = base.FINE_DIR / "fine_icp_curvature_result.json"
OUTPUT_TRANSFORM_NPZ = base.FINE_DIR / "fine_icp_curvature_transforms.npz"

CURV_K = 30
CURV_PERCENTILE = 85
CURV_SIM_THRESHOLD = 0.85
CANDIDATE_K = 10
REFINE_DIST = 0.010
REFINE_MAX_ITER = 50
FEATURE_WEIGHT = 5.0
PLANE_WEIGHT = 1.0
PLANE_SAMPLE_STEP = 5
CONVERGE_TRANSLATION = 1e-5
CONVERGE_ROTATION_DEG = 0.01


def configure_paths(args):
    global OUTPUT_FUSED_PCD, OUTPUT_RESULT_JSON, OUTPUT_TRANSFORM_NPZ

    base.configure_paths(args)

    OUTPUT_FUSED_PCD = Path(args.output_pcd) if args.output_pcd else base.FINE_DIR / "fine_fuse_curvature.pcd"
    OUTPUT_RESULT_JSON = Path(args.output_json) if args.output_json else base.FINE_DIR / "fine_icp_curvature_result.json"
    OUTPUT_TRANSFORM_NPZ = Path(args.output_transforms) if args.output_transforms else base.FINE_DIR / "fine_icp_curvature_transforms.npz"


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def tangent_basis(n):
    ref = np.array([1., 0., 0.]) if abs(n[0]) < 0.9 else np.array([0., 1., 0.])
    e1 = np.cross(n, ref)
    e1 /= np.linalg.norm(e1) + 1e-12
    e2 = np.cross(n, e1)
    e2 /= np.linalg.norm(e2) + 1e-12
    return e1, e2


def curvature_descriptors(pcd):
    base.estimate_normals(pcd)

    pts = np.asarray(pcd.points)
    nrm = np.asarray(pcd.normals)
    k = min(CURV_K, len(pts))

    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=k, workers=-1)

    desc = np.zeros((len(pts), 4))
    score = np.zeros(len(pts))

    for i in range(len(pts)):
        e1, e2 = tangent_basis(nrm[i])
        d = pts[idx[i]] - pts[i]
        u, v, w = d @ e1, d @ e2, d @ nrm[i]

        A = np.column_stack([u*u, u*v, v*v, u, v, np.ones(len(u))])
        c = np.linalg.lstsq(A, w, rcond=None)[0]

        H = np.array([[2*c[0], c[1]], [c[1], 2*c[2]]])
        k1, k2 = np.linalg.eigvalsh(H)

        if abs(k1) < abs(k2):
            k1, k2 = k2, k1

        desc[i] = [k1*k2, (k1+k2)/2, k1, k2]
        score[i] = max(abs(k1), abs(k2))

    return desc, score, nrm


def curvature_similarity(a, b, flip=False):
    x = a.copy()

    if flip:
        x[1:] *= -1

    nx = np.linalg.norm(x)
    ny = np.linalg.norm(b)

    if nx < 1e-10 or ny < 1e-10:
        return 0.0

    return float(np.clip(np.dot(x, b) / (nx * ny), 0, 1))


def build_correspondences(sp, sn, sd, sh, tp, tn, td, tree):
    dist, ids = tree.query(sp, k=min(CANDIDATE_K, len(tp)), workers=-1)

    if ids.ndim == 1:
        ids = ids[:, None]
        dist = dist[:, None]

    si, ti, w = [], [], []

    for i in range(len(sp)):
        if not sh[i] and i % PLANE_SAMPLE_STEP:
            continue

        for d, j in zip(dist[i], ids[i]):
            if d > REFINE_DIST:
                break

            if sh[i]:
                sim = curvature_similarity(sd[i], td[j], np.dot(sn[i], tn[j]) < 0)

                if sim < CURV_SIM_THRESHOLD:
                    continue

                weight = FEATURE_WEIGHT
            else:
                weight = PLANE_WEIGHT

            si.append(i)
            ti.append(j)
            w.append(weight)
            break

    return np.asarray(si), np.asarray(ti), np.asarray(w)


def solve_point_to_plane(s, t, n, w):
    A = np.column_stack([np.cross(s, n), n])
    b = -np.sum(n * (s - t), axis=1)

    q = np.sqrt(w)
    x = np.linalg.lstsq(A * q[:, None], b * q, rcond=None)[0]

    T = np.eye(4)
    T[:3, :3] = R.from_rotvec(x[:3]).as_matrix()
    T[:3, 3] = x[3:]

    return T


def curvature_refine(source, target):
    work = base.copy_pcd(source)
    target = base.copy_pcd(target)

    sd, ss, _ = curvature_descriptors(work)
    td, ts, tn = curvature_descriptors(target)

    sh = ss >= np.percentile(ss, CURV_PERCENTILE)

    tp = np.asarray(target.points)
    tree = cKDTree(tp)

    T_total = np.eye(4)

    for iteration in range(REFINE_MAX_ITER):
        sp = np.asarray(work.points)
        sn = np.asarray(work.normals)

        si, ti, w = build_correspondences(sp, sn, sd, sh, tp, tn, td, tree)

        if len(si) < 6:
            break

        s = sp[si]
        t = tp[ti]
        n = tn[ti]

        dT = solve_point_to_plane(s, t, n, w)

        work.transform(dT)
        T_total = dT @ T_total

        dt = np.linalg.norm(dT[:3, 3])
        dr = np.degrees(R.from_matrix(dT[:3, :3]).magnitude())

        print(
            f"curv {iteration+1:02d}: "
            f"corr={len(si)}, "
            f"feature={np.sum(w > PLANE_WEIGHT)}, "
            f"plane={np.sum(w == PLANE_WEIGHT)}, "
            f"dt={dt*1000:.3f} mm, "
            f"dR={dr:.3f} deg"
        )

        if dt < CONVERGE_TRANSLATION and dr < CONVERGE_ROTATION_DEG:
            break

    return work, T_total


def pass_filter(fitness, rmse, trans_norm, rot_deg, overlap):
    if fitness < base.MIN_FITNESS or rmse > base.MAX_RMSE:
        return False, "bad quality: fitness too low or rmse too large"
    if trans_norm > base.MAX_TRANSLATION:
        return False, "translation too large"
    if rot_deg > base.MAX_ROTATION_DEG:
        return False, "rotation too large"
    if overlap < base.MIN_OVERLAP_RATIO:
        return False, "overlap ratio too low"
    return True, "kept"


def main():
    items = []
    items += base.load_dataset(base.COARSE_POINT_FILE, base.COARSE_SEQ_FILE, "coarse")
    items += base.load_dataset(base.FINE_POINT_FILE, base.FINE_SEQ_FILE, "fine")

    processed = [base.preprocess(x["points"]) for x in items]
    raw_counts = [len(x["points"]) for x in items]
    proc_counts = [len(x.points) for x in processed]

    target_idx, target_selection = base.select_target_idx(items, raw_counts, proc_counts)

    fused = base.copy_pcd(processed[target_idx])
    base.estimate_normals(fused)

    records = []
    transforms = []

    print("========== Fine ICP Curvature ==========")
    print("target:", items[target_idx]["global_name"])
    print("target selection:", target_selection)

    for i, source in enumerate(processed):
        T_final = np.eye(4)

        record = {
            "index": i,
            "dataset": items[i]["dataset"],
            "name": items[i]["name"],
            "global_name": items[i]["global_name"],
            "is_target": i == target_idx,
            "kept": False,
        }

        if i == target_idx:
            record["kept"] = True
            record["reason"] = "target frame"
            record["transformation"] = T_final.tolist()
            records.append(record)
            transforms.append(T_final)
            continue

        if len(source.points) == 0:
            record["reason"] = "empty cloud"
            record["transformation"] = T_final.tolist()
            records.append(record)
            transforms.append(T_final)
            continue

        print(f"\n========== {items[i]['global_name']} ==========")

        reg = base.run_icp(source, fused)
        T_global = np.asarray(reg.transformation)

        source_global = base.copy_pcd(source)
        source_global.transform(T_global)

        _, dT = curvature_refine(source_global, fused)
        T_final = dT @ T_global

        eva = o3d.pipelines.registration.evaluate_registration(
            source,
            fused,
            base.ICP_DISTANCE_THRESHOLD,
            T_final,
        )

        fitness = float(eva.fitness)
        rmse = float(eva.inlier_rmse)
        trans_norm = float(np.linalg.norm(T_final[:3, 3]))
        rot_deg = base.rotation_angle_deg(T_final)
        overlap = base.compute_overlap_ratio(source, fused, T_final)

        keep, reason = pass_filter(fitness, rmse, trans_norm, rot_deg, overlap)

        print(
            f"final fitness={fitness:.4f}, "
            f"rmse={rmse*1000:.3f} mm, "
            f"trans={trans_norm*1000:.3f} mm, "
            f"rot={rot_deg:.3f} deg, "
            f"overlap={overlap:.3f}, "
            f"keep={keep}"
        )

        record.update({
            "kept": keep,
            "fitness": fitness,
            "inlier_rmse": rmse,
            "translation_norm_m": trans_norm,
            "rotation_angle_deg": rot_deg,
            "overlap_ratio": overlap,
            "T_global": T_global.tolist(),
            "delta_T_curvature": dT.tolist(),
            "transformation": T_final.tolist(),
            "reason": reason,
        })

        records.append(record)
        transforms.append(T_final)

        if keep:
            aligned = base.copy_pcd(source)
            aligned.transform(T_final)

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
        transforms=np.asarray(transforms),
        names=np.asarray([x["global_name"] for x in items], dtype=object),
    )

    save_json(OUTPUT_RESULT_JSON, {
        "mode": "additive_global_icp_curvature",
        "target": items[target_idx]["global_name"],
        "target_selection": target_selection,
        "fused_points": len(fused.points),
        "records": records,
    })

    print("\nsaved:", OUTPUT_FUSED_PCD)
    print("fused points:", len(fused.points))

    if base.VISUALIZE:
        vis = base.copy_pcd(fused)
        vis.paint_uniform_color([0.2, 0.7, 1.0])
        o3d.visualization.draw_geometries([vis], window_name="fine_icp_curvature")


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
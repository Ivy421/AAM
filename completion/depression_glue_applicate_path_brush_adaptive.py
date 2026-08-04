"""Adaptive sponge-brush segmentation from a curved fix-surface point cloud."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


WORLD_X = np.array([1.0, 0.0, 0.0], dtype=float)

VOXEL_SIZE = 0.0008
NORMAL_RADII = np.arange(0.008, 0.041, 0.004)
MIN_NORMAL_NEIGHBORS = 10

POINT_CONNECT_RADIUS = 0.0050  # 0.003
REGION_NEIGHBOR_ANGLE_DEG = 15.0 # 10
REGION_CURVATURE_JUMP = 0.02 # 0.015
MAX_REGION_NORMAL_SPAN_DEG = 20.0 # 16
MAX_REGION_PLANE_RESIDUAL_M = 0.0025 # 0.0018

MERGE_DISTANCE_M = 0.006  # 0.0045
MERGE_NORMAL_ANGLE_DEG = 15.0  #7
MERGE_CURVATURE_JUMP = 0.015 # 0.010 
MIN_SEGMENT_POINTS = 10


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    #parser.add_argument("--fix-mask", type=Path, required=True)
    parser.add_argument("--fix-points", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def normalize(vector):
    vector = np.asarray(vector, dtype=float)
    return vector / (np.linalg.norm(vector) + 1e-12)


def angle_deg(a, b):
    cosine = np.clip(np.dot(normalize(a), normalize(b)), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def preprocess_points(points):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud = cloud.voxel_down_sample(VOXEL_SIZE)

    if len(cloud.points) >= 50:
        cloud, _ = cloud.remove_statistical_outlier(
            nb_neighbors=20,
            std_ratio=2.0,
        )
    return np.asarray(cloud.points, dtype=float)


def pca_features(local_points):
    center = local_points.mean(axis=0)
    centered = local_points - center
    covariance = centered.T @ centered / max(len(local_points), 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)

    inward_normal = normalize(eigenvectors[:, 0])
    if float(np.dot(inward_normal, WORLD_X)) < 0.0:
        inward_normal = -inward_normal

    curvature = float(
        eigenvalues[0] / max(float(eigenvalues.sum()), 1e-12)
    )
    residual = float(
        np.percentile(np.abs(centered @ inward_normal), 95.0)
    )
    return center, inward_normal, curvature, residual


def estimate_local_features(points, tree):
    point_count = len(points)
    normals = np.zeros((point_count, 3), dtype=float)
    curvatures = np.zeros(point_count, dtype=float)
    residuals = np.zeros(point_count, dtype=float)
    neighbor_counts = np.zeros(point_count, dtype=int)
    local_radii = np.zeros(point_count, dtype=float)
    local_neighbor_ids = []

    for index, point in enumerate(points):
        neighbor_ids = []
        used_radius = NORMAL_RADII[-1]

        for radius in NORMAL_RADII:
            neighbor_ids = tree.query_ball_point(point, float(radius))
            used_radius = float(radius)
            if len(neighbor_ids) >= MIN_NORMAL_NEIGHBORS:
                break

        if len(neighbor_ids) < MIN_NORMAL_NEIGHBORS:
            _, neighbor_ids = tree.query(
                point,
                k=min(MIN_NORMAL_NEIGHBORS, point_count),
            )
            neighbor_ids = np.atleast_1d(neighbor_ids).astype(int).tolist()

        _, normal, curvature, residual = pca_features(points[neighbor_ids])

        normals[index] = normal
        curvatures[index] = curvature
        residuals[index] = residual
        neighbor_counts[index] = len(neighbor_ids)
        local_radii[index] = used_radius
        local_neighbor_ids.append(np.asarray(neighbor_ids, dtype=int))

    return {
        "normals": normals,
        "curvatures": curvatures,
        "residuals": residuals,
        "neighbor_counts": neighbor_counts,
        "local_radii": local_radii,
        "neighbor_ids": local_neighbor_ids,
    }


def seed_scores(points, tree, features):
    dense_counts = np.asarray(
        [len(tree.query_ball_point(point, NORMAL_RADII[0])) for point in points],
        dtype=float,
    )
    density_term = 1.0 / np.maximum(dense_counts, 1.0)
    residual_term = (
        features["residuals"]
        / np.maximum(features["local_radii"], 1e-6)
    )
    return features["curvatures"] + residual_term + 0.1 * density_term


def region_metrics(indices, points, point_normals):
    indices = np.asarray(list(indices), dtype=int)
    local_points = points[indices]
    center, inward_normal, curvature, residual = pca_features(local_points)

    normal_span = max(
        angle_deg(normal, inward_normal)
        for normal in point_normals[indices]
    )

    return {
        "indices": indices,
        "center": center,
        "inward_normal": inward_normal,
        "outward_normal": -inward_normal,
        "curvature": curvature,
        "plane_residual_p95_m": residual,
        "normal_span_deg": float(normal_span),
        "point_count": int(len(indices)),
    }


def candidate_is_compatible(candidate, region_indices, points, features):
    current = region_metrics(
        region_indices,
        points,
        features["normals"],
    )

    if (
        angle_deg(
            features["normals"][candidate],
            current["inward_normal"],
        )
        > REGION_NEIGHBOR_ANGLE_DEG
    ):
        return False

    if (
        abs(features["curvatures"][candidate] - current["curvature"])
        > REGION_CURVATURE_JUMP
    ):
        return False

    proposed = region_metrics(
        list(region_indices) + [candidate],
        points,
        features["normals"],
    )
    return (
        proposed["normal_span_deg"] <= MAX_REGION_NORMAL_SPAN_DEG
        and proposed["plane_residual_p95_m"]
        <= MAX_REGION_PLANE_RESIDUAL_M
    )


def grow_one_region(seed, unassigned, points, tree, features):
    seed_normal = features["normals"][seed]
    initial_ids = [
        index
        for index in features["neighbor_ids"][seed]
        if unassigned[index]
        and angle_deg(features["normals"][index], seed_normal)
        <= REGION_NEIGHBOR_ANGLE_DEG
    ]

    region = set(initial_ids or [seed])
    unassigned[list(region)] = False
    queue = list(region)

    while queue:
        current = queue.pop()
        neighbor_ids = tree.query_ball_point(
            points[current],
            POINT_CONNECT_RADIUS,
        )

        for candidate in neighbor_ids:
            if not unassigned[candidate]:
                continue
            if not candidate_is_compatible(
                candidate,
                region,
                points,
                features,
            ):
                continue

            region.add(candidate)
            unassigned[candidate] = False
            queue.append(candidate)

    return sorted(region)


def build_initial_regions(points, tree, features):
    scores = seed_scores(points, tree, features)
    unassigned = np.ones(len(points), dtype=bool)
    regions = []

    while np.any(unassigned):
        candidates = np.where(unassigned)[0]
        seed = candidates[np.argmin(scores[candidates])]
        region = grow_one_region(
            seed,
            unassigned,
            points,
            tree,
            features,
        )
        regions.append(region)

    return regions


def regions_are_adjacent(region_a, region_b, points):
    tree_b = cKDTree(points[region_b])
    distances, _ = tree_b.query(points[region_a], k=1)
    return float(np.min(distances)) <= MERGE_DISTANCE_M


def can_merge(region_a, region_b, points, features):
    metrics_a = region_metrics(
        region_a,
        points,
        features["normals"],
    )
    metrics_b = region_metrics(
        region_b,
        points,
        features["normals"],
    )

    if (
        angle_deg(
            metrics_a["inward_normal"],
            metrics_b["inward_normal"],
        )
        > MERGE_NORMAL_ANGLE_DEG
    ):
        return False

    if (
        abs(metrics_a["curvature"] - metrics_b["curvature"])
        > MERGE_CURVATURE_JUMP
    ):
        return False

    combined = region_metrics(
        region_a + region_b,
        points,
        features["normals"],
    )
    return (
        combined["normal_span_deg"] <= MAX_REGION_NORMAL_SPAN_DEG
        and combined["plane_residual_p95_m"]
        <= MAX_REGION_PLANE_RESIDUAL_M
    )


def merge_regions(regions, points, features):
    regions = [list(region) for region in regions]

    changed = True
    while changed:
        changed = False

        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                if not regions_are_adjacent(
                    regions[i],
                    regions[j],
                    points,
                ):
                    continue
                if not can_merge(
                    regions[i],
                    regions[j],
                    points,
                    features,
                ):
                    continue

                regions[i] = sorted(set(regions[i] + regions[j]))
                del regions[j]
                changed = True
                break
            if changed:
                break

    return regions


def xoy_tangent_and_length(points):
    xy = np.asarray(points, dtype=float)[:, :2]
    center = xy.mean(axis=0)
    centered = xy - center

    if len(xy) <= 1:
        return np.array([1.0, 0.0, 0.0], dtype=float), 0.0

    covariance = centered.T @ centered / len(xy)
    _, eigenvectors = np.linalg.eigh(covariance)
    main_axis = eigenvectors[:, -1]
    if float(np.dot(main_axis, np.array([1.0, 0.0]))) < 0.0:
        main_axis = -main_axis
    coordinates = centered @ main_axis
    tangent = np.array([main_axis[0], main_axis[1], 0.0], dtype=float)
    return normalize(tangent), float(coordinates.max() - coordinates.min())


def build_segments(regions, points, features):
    segments = []

    for region in regions:
        if len(region) < MIN_SEGMENT_POINTS:
            continue

        metrics = region_metrics(
            region,
            points,
            features["normals"],
        )
        tangent, length = xoy_tangent_and_length(points[region])

        segments.append(
            {
                "id": len(segments),
                "coordinate_frame": "base",
                "point_count": metrics["point_count"],
                "center_point_base_m": np.round(
                    metrics["center"],
                    9,
                ).tolist(),
                "center_point_base_mm": np.round(
                    metrics["center"] * 1000.0,
                    3,
                ).tolist(),
                "inward_normal_unit": np.round(
                    metrics["inward_normal"],
                    9,
                ).tolist(),
                "outward_normal_unit": np.round(
                    metrics["outward_normal"],
                    9,
                ).tolist(),
                "xoy_length": round(length, 9),
                "tangent_unit": np.round(tangent, 9).tolist(),
                "quality": {
                    "curvature": round(metrics["curvature"], 9),
                    "normal_span_deg": round(
                        metrics["normal_span_deg"],
                        6,
                    ),
                    "plane_residual_p95_m": round(
                        metrics["plane_residual_p95_m"],
                        9,
                    ),
                },
            }
        )

    return segments


def save_outputs(out_dir, source_paths, points, segments, runtime_seconds):
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "glue_brush_adaptive_segments.json"
    npz_path = out_dir / "glue_brush_adaptive_segments.npz"
    centers_path = out_dir / "glue_brush_adaptive_centers.pcd"

    payload = {
        "coordinate_frame": "base",
        "position_unit": "m",
        "normal_definition": (
            "PCA inward normal has dot(normal, world_x) > 0; "
            "outward_normal_unit is its opposite."
        ),
        "runtime_seconds": round(runtime_seconds, 6),
        "parameters": {
            "voxel_size_m": VOXEL_SIZE,
            "normal_radii_m": NORMAL_RADII.tolist(),
            "min_normal_neighbors": MIN_NORMAL_NEIGHBORS,
            "point_connect_radius_m": POINT_CONNECT_RADIUS,
            "region_neighbor_angle_deg": REGION_NEIGHBOR_ANGLE_DEG,
            "region_curvature_jump": REGION_CURVATURE_JUMP,
            "max_region_normal_span_deg": MAX_REGION_NORMAL_SPAN_DEG,
            "max_region_plane_residual_m": MAX_REGION_PLANE_RESIDUAL_M,
            "merge_distance_m": MERGE_DISTANCE_M,
            "merge_normal_angle_deg": MERGE_NORMAL_ANGLE_DEG,
            "merge_curvature_jump": MERGE_CURVATURE_JUMP,
            "min_segment_points": MIN_SEGMENT_POINTS,
        },
        "segment_count": len(segments),
        "segments": segments,
    }

    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    centers = np.asarray(
        [segment["center_point_base_m"] for segment in segments],
        dtype=float,
    )
    outward_normals = np.asarray(
        [segment["outward_normal_unit"] for segment in segments],
        dtype=float,
    )
    xoy_lengths = np.asarray(
        [segment["xoy_length"] for segment in segments],
        dtype=float,
    )

    np.savez(
        npz_path,
        preprocessed_fix_points_base_m=points,
        segment_centers_base_m=centers,
        segment_outward_normals=outward_normals,
        segment_xoy_lengths_m=xoy_lengths,
    )

    center_cloud = o3d.geometry.PointCloud()
    center_cloud.points = o3d.utility.Vector3dVector(centers)
    center_cloud.normals = o3d.utility.Vector3dVector(outward_normals)
    o3d.io.write_point_cloud(str(centers_path), center_cloud)

    return json_path, npz_path, centers_path


def main():
    args = parse_args()
    start_time = time.perf_counter()

    source_cloud = o3d.io.read_point_cloud(str(args.fix_points))
    source_points = np.asarray(source_cloud.points, dtype=float)
    points = preprocess_points(source_points)

    tree = cKDTree(points)
    features = estimate_local_features(points, tree)

    regions = build_initial_regions(points, tree, features)
    regions = merge_regions(regions, points, features)
    segments = build_segments(regions, points, features)

    runtime_seconds = time.perf_counter() - start_time
    paths = save_outputs(
        args.out_dir,
        {
            "fix_points": args.fix_points,
        },
        points,
        segments,
        runtime_seconds,
    )

    print(f"Voxel points: {len(points)}")
    print(f"Adaptive regions: {len(regions)}")
    print(f"Adaptive segments: {len(segments)}")
    print(f"Runtime: {runtime_seconds:.3f} s")
    for path in paths:
        print(f"Saved: {path.resolve()}")


if __name__ == "__main__":
    main()

"""Split the depression fix mask into brush-sized surface segments."""

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


DEFAULT_SPACING = 0.03
MIN_SEGMENT_CELLS = 3
WORLD_X = np.array([1.0, 0.0, 0.0], dtype=float)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fix-mask", type=Path, required=True  )
    parser.add_argument("--fix-points", type=Path, required=True )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--spacing",
        type=float,
        default=DEFAULT_SPACING,
        help="Mask segment side length in metres.",
    )
    return parser.parse_args()


def normalize(vector):
    vector = np.asarray(vector, dtype=float).reshape(3)
    norm = np.linalg.norm(vector)
    if norm <= 1e-12:
        raise ValueError("Cannot normalize a zero vector.")
    return vector / norm


def fit_global_outward_normal(point_cloud):
    plane, _ = point_cloud.segment_plane(
        distance_threshold=0.002,
        ransac_n=3,
        num_iterations=3000,
    )
    normal = normalize(plane[:3])
    if float(np.dot(normal, WORLD_X)) > 0.0:
        normal = -normal
    return normal


def fit_segment_frame(points, global_outward, u_axis):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    centered = points - points.mean(axis=0)
    covariance = centered.T @ centered / max(len(points), 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    normal = normalize(eigenvectors[:, np.argmin(eigenvalues)])
    if float(np.dot(normal, global_outward)) < 0.0:
        normal = -normal
    if float(np.dot(normal, WORLD_X)) > 0.0:
        normal = -normal
    if abs(float(np.dot(normal, WORLD_X))) <= 1e-9:
        normal = global_outward.copy()

    tangent = eigenvectors[:, np.argmax(eigenvalues)]
    tangent = tangent - np.dot(tangent, normal) * normal
    if np.linalg.norm(tangent) <= 1e-9:
        tangent = u_axis - np.dot(u_axis, normal) * normal
    tangent = normalize(tangent)
    if float(np.dot(tangent, u_axis)) < 0.0:
        tangent = -tangent
    return normal, tangent


def cells_to_3d(cells_yx, z, data):
    cells_yx = np.asarray(cells_yx, dtype=float)
    u = float(data["u_min"]) + (cells_yx[:, 1] + 0.5) * float(data["grid_res"])
    v = float(data["v_min"]) + (cells_yx[:, 0] + 0.5) * float(data["grid_res"])
    return (
        np.asarray(data["origin"], dtype=float)
        + u[:, None] * np.asarray(data["u_axis"], dtype=float)
        + v[:, None] * np.asarray(data["v_axis"], dtype=float)
        + float(z) * np.asarray(data["n_axis"], dtype=float)
    )


def reconstruct_segment_points(segment_cells, masks, z_values, valid_flags, data):
    segment_lookup = {tuple(cell) for cell in np.asarray(segment_cells, dtype=int)}
    groups = []
    for mask, z, valid in zip(masks, z_values, valid_flags):
        if not valid:
            continue
        layer_cells = np.column_stack(np.where(mask))
        selected = np.asarray(
            [cell for cell in layer_cells if tuple(cell) in segment_lookup],
            dtype=int,
        )
        if len(selected):
            groups.append(cells_to_3d(selected, z, data))
    return np.vstack(groups) if groups else np.empty((0, 3), dtype=float)


def segment_fix_mask(data, surface_points, spacing):
    masks = np.asarray(data["fix_surface_masks"], dtype=bool)
    valid_flags = np.asarray(data["barrier_valid_flags"], dtype=bool)
    z_values = np.asarray(data["barrier_z_values"], dtype=float)
    if len(masks) == 0 or not np.any(valid_flags):
        raise RuntimeError("fix_mask.npz contains no valid fix surface masks.")

    union_mask = np.any(masks[valid_flags], axis=0)
    cells = np.column_stack(np.where(union_mask))
    if len(cells) == 0:
        raise RuntimeError("The union of valid fix surface masks is empty.")

    grid_res = float(data["grid_res"])
    spacing_cells = spacing / grid_res
    y_min, x_min = cells.min(axis=0)
    bin_y = np.floor((cells[:, 0] - y_min) / spacing_cells).astype(int)
    bin_x = np.floor((cells[:, 1] - x_min) / spacing_cells).astype(int)

    origin = np.asarray(data["origin"], dtype=float)
    u_axis = normalize(data["u_axis"])
    v_axis = normalize(data["v_axis"])
    surface_u = (surface_points - origin) @ u_axis
    surface_v = (surface_points - origin) @ v_axis

    segments = []
    for segment_id, (by, bx) in enumerate(np.unique(np.column_stack((bin_y, bin_x)), axis=0)):
        in_segment = (bin_y == by) & (bin_x == bx)
        segment_cells = cells[in_segment]
        if len(segment_cells) < MIN_SEGMENT_CELLS:
            continue

        u0 = float(data["u_min"]) + (x_min + bx * spacing_cells) * grid_res
        u1 = u0 + spacing
        v0 = float(data["v_min"]) + (y_min + by * spacing_cells) * grid_res
        v1 = v0 + spacing
        selected = (
            (surface_u >= u0)
            & (surface_u < u1)
            & (surface_v >= v0)
            & (surface_v < v1)
        )
        local_points = surface_points[selected]
        if len(local_points) < 3:
            local_points = reconstruct_segment_points(
                segment_cells,
                masks,
                z_values,
                valid_flags,
                data,
            )
        if len(local_points) < 3:
            continue

        segments.append(
            {
                "id": int(segment_id),
                "bin_yx": [int(by), int(bx)],
                "cell_count": int(len(segment_cells)),
                "area_m2": float(len(segment_cells) * grid_res * grid_res),
                "points": local_points,
            }
        )
    return segments, u_axis


def main():
    args = parse_args()
    if args.spacing <= 0.0:
        raise ValueError("--spacing must be positive.")
    for path in (args.fix_mask, args.fix_points):
        if not path.is_file():
            raise FileNotFoundError(path)

    data = np.load(args.fix_mask, allow_pickle=True)
    point_cloud = o3d.io.read_point_cloud(str(args.fix_points))
    surface_points = np.asarray(point_cloud.points, dtype=float)
    if len(surface_points) < 3:
        raise RuntimeError(f"Surface point cloud needs at least 3 points: {args.fix_points}")

    global_outward = fit_global_outward_normal(point_cloud)
    raw_segments, u_axis = segment_fix_mask(data, surface_points, args.spacing)

    results = []
    for segment in raw_segments:
        midpoint = segment["points"].mean(axis=0)
        normal, tangent = fit_segment_frame(
            segment["points"],
            global_outward,
            u_axis,
        )
        results.append(
            {
                "id": segment["id"],
                "bin_yx": segment["bin_yx"],
                "cell_count": segment["cell_count"],
                "area_m2": segment["area_m2"],
                "midpoint": midpoint.tolist(),
                "normal": normal.tolist(),
                "tangent": tangent.tolist(),
            }
        )

    if not results:
        raise RuntimeError("No valid brush segments were generated.")

    midpoints = np.asarray([item["midpoint"] for item in results], dtype=float)
    normals = np.asarray([item["normal"] for item in results], dtype=float)
    tangents = np.asarray([item["tangent"] for item in results], dtype=float)
    cell_counts = np.asarray([item["cell_count"] for item in results], dtype=int)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.out_dir / "glue_brush_segments.npz"
    json_path = args.out_dir / "glue_brush_segments.json"
    pcd_path = args.out_dir / "glue_brush_segments.pcd"

    np.savez(
        npz_path,
        segment_midpoints=midpoints,
        segment_normals=normals,
        segment_tangents=tangents,
        segment_cell_counts=cell_counts,
        global_outward_normal=global_outward,
        spacing=np.asarray(args.spacing),
        grid_res=np.asarray(float(data["grid_res"])),
    )
    json_path.write_text(
        json.dumps(
            {
                "spacing": args.spacing,
                "global_outward_normal": global_outward.tolist(),
                "segments": results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    output_cloud = o3d.geometry.PointCloud()
    output_cloud.points = o3d.utility.Vector3dVector(midpoints)
    output_cloud.normals = o3d.utility.Vector3dVector(normals)
    o3d.io.write_point_cloud(str(pcd_path), output_cloud)

    print(f"Generated brush segments: {len(results)}")
    print(f"Global outward normal: {global_outward.tolist()}")
    print(f"Saved: {npz_path}")
    print(f"Saved: {json_path}")
    print(f"Saved: {pcd_path}")


if __name__ == "__main__":
    main()

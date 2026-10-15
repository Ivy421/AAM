import os
import numpy as np
import open3d as o3d

# ============================================================
# Hard-coded input
# ============================================================
OUTPUT_DIR = r"data/runs/20261013_124017/completion/depression"
STATE_PATH = os.path.join(OUTPUT_DIR, "_part1_state.npz")
TOP_PCD_PATH = os.path.join(OUTPUT_DIR, "_part1_raw.pcd")

POINT_SIZE = 4.0
TURN_RADIUS = 0.0040
FIT_RADIUS = 0.0008


def normalize(v):
    v = np.asarray(v, dtype=float)
    return v / (np.linalg.norm(v) + 1e-12)


def uv_to_3d(uv, origin, u_axis, v_axis):
    uv = np.asarray(uv, dtype=float)
    return origin + uv[:, 0:1] * u_axis + uv[:, 1:2] * v_axis


def make_sphere(center, radius, color):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.compute_vertex_normals()
    sphere.translate(np.asarray(center, dtype=float))
    sphere.paint_uniform_color(color)
    return sphere


def make_tube(p0, p1, radius, color):
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    vec = p1 - p0
    length = np.linalg.norm(vec)
    if length < 1e-12:
        return None

    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length, resolution=20, split=4)
    cylinder.compute_vertex_normals()
    cylinder.paint_uniform_color(color)

    z_axis = np.array([0.0, 0.0, 1.0])
    direction = vec / length
    axis = np.cross(z_axis, direction)
    axis_norm = np.linalg.norm(axis)

    if axis_norm < 1e-12:
        if np.dot(z_axis, direction) < 0:
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([np.pi, 0.0, 0.0]))
        else:
            R = np.eye(3)
    else:
        axis = axis / axis_norm
        angle = np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)

    cylinder.rotate(R, center=np.zeros(3))
    cylinder.translate((p0 + p1) * 0.5)
    return cylinder


def add_polyline_tubes(geoms, points_3d, radius, color):
    for i in range(len(points_3d) - 1):
        tube = make_tube(points_3d[i], points_3d[i + 1], radius, color)
        if tube is not None:
            geoms.append(tube)


def fit_two_segments(points_uv, turn_idx, side):
    """
    Refit the same two segments using the saved Part-1 split index.

    max_u:
        main = v, normal = u
        u = a*v + b

    max_v / min_v:
        main = u, normal = v
        v = a*u + b
    """
    points_uv = np.asarray(points_uv, dtype=float)
    turn_idx = int(turn_idx)

    if side == "max_u":
        main = points_uv[:, 1]
        normal = points_uv[:, 0]
        normal_axis = "u"
    elif side in ("max_v", "min_v"):
        main = points_uv[:, 0]
        normal = points_uv[:, 1]
        normal_axis = "v"
    else:
        raise ValueError(f"Unsupported side: {side}")

    if turn_idx < 2 or len(points_uv) - turn_idx < 2:
        raise RuntimeError(f"Not enough points around turn_idx={turn_idx} for side={side}.")

    a1, b1 = np.polyfit(main[:turn_idx], normal[:turn_idx], 1)
    a2, b2 = np.polyfit(main[turn_idx:], normal[turn_idx:], 1)

    main1 = np.linspace(main[:turn_idx].min(), main[:turn_idx].max(), 100)
    main2 = np.linspace(main[turn_idx:].min(), main[turn_idx:].max(), 100)

    if normal_axis == "u":
        fit1 = np.column_stack([a1 * main1 + b1, main1])
        fit2 = np.column_stack([a2 * main2 + b2, main2])
    else:
        fit1 = np.column_stack([main1, a1 * main1 + b1])
        fit2 = np.column_stack([main2, a2 * main2 + b2])

    return fit1, fit2, (a1, b1), (a2, b2)


def main():
    data = np.load(STATE_PATH, allow_pickle=True)

    origin = data["origin"].astype(float)
    u_axis = normalize(data["u_axis"])
    v_axis = normalize(data["v_axis"])
    n_axis = normalize(data["n_axis"])

    u_side = str(data["u_side"])
    v_side = str(data["v_side"])

    # Raw discrete boundary points saved by Part 1.
    u_line_raw = data["u_line_raw"].astype(float)
    v_line_raw = data["v_line_raw"].astype(float)

    # Smoothed boundary used by TwoFit.
    u_line = data["u_line"].astype(float)
    v_line = data["v_line"].astype(float)

    turn_idx_u = int(data["turn_idx_u"])
    turn_idx_v = int(data["turn_idx_v"])
    turn_point_u = data["turn_point_u"].astype(float)
    turn_point_v = data["turn_point_v"].astype(float)

    # Refit the two straight-line segments from the saved split index.
    u_fit1, u_fit2, u_model1, u_model2 = fit_two_segments(u_line, turn_idx_u, u_side)
    v_fit1, v_fit2, v_model1, v_model2 = fit_two_segments(v_line, turn_idx_v, v_side)

    # UV -> 3D.
    u_raw_3d = uv_to_3d(u_line_raw, origin, u_axis, v_axis)
    v_raw_3d = uv_to_3d(v_line_raw, origin, u_axis, v_axis)
    u_fit1_3d = uv_to_3d(u_fit1, origin, u_axis, v_axis)
    u_fit2_3d = uv_to_3d(u_fit2, origin, u_axis, v_axis)
    v_fit1_3d = uv_to_3d(v_fit1, origin, u_axis, v_axis)
    v_fit2_3d = uv_to_3d(v_fit2, origin, u_axis, v_axis)
    turn_u_3d = uv_to_3d(turn_point_u.reshape(1, 2), origin, u_axis, v_axis)[0]
    turn_v_3d = uv_to_3d(turn_point_v.reshape(1, 2), origin, u_axis, v_axis)[0]

    top_pcd = o3d.io.read_point_cloud(TOP_PCD_PATH)
    top_pcd.paint_uniform_color([0.65, 0.65, 0.65])

    u_raw_pcd = o3d.geometry.PointCloud()
    u_raw_pcd.points = o3d.utility.Vector3dVector(u_raw_3d)
    u_raw_pcd.paint_uniform_color([0.0, 0.35, 1.0])

    v_raw_pcd = o3d.geometry.PointCloud()
    v_raw_pcd.points = o3d.utility.Vector3dVector(v_raw_3d)
    v_raw_pcd.paint_uniform_color([0.0, 0.85, 0.85])

    geoms = [top_pcd, u_raw_pcd, v_raw_pcd]

    # Stable-side fit: yellow.
    add_polyline_tubes(geoms, u_fit1_3d, FIT_RADIUS, [1.0, 0.75, 0.0])
    add_polyline_tubes(geoms, v_fit1_3d, FIT_RADIUS, [1.0, 0.75, 0.0])

    # Defect-side fit: magenta.
    add_polyline_tubes(geoms, u_fit2_3d, FIT_RADIUS, [1.0, 0.0, 1.0])
    add_polyline_tubes(geoms, v_fit2_3d, FIT_RADIUS, [1.0, 0.0, 1.0])

    # Turn points: large red spheres.
    geoms.append(make_sphere(turn_u_3d, TURN_RADIUS, [1.0, 0.0, 0.0]))
    geoms.append(make_sphere(turn_v_3d, TURN_RADIUS, [1.0, 0.0, 0.0]))

    # Filling-frame coordinate system.
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([u_axis, v_axis, n_axis])
    T[:3, 3] = origin
    frame.transform(T)
    geoms.append(frame)

    print("\n========== Part 1 TwoFit Visualization ==========")
    print("u_side:", u_side)
    print("v_side:", v_side)
    print("turn_idx_u:", turn_idx_u)
    print("turn_idx_v:", turn_idx_v)
    print("turn_point_u:", turn_point_u)
    print("turn_point_v:", turn_point_v)
    print("U-side stable fit (a,b):", u_model1)
    print("U-side defect fit (a,b):", u_model2)
    print("V-side stable fit (a,b):", v_model1)
    print("V-side defect fit (a,b):", v_model2)
    print("\nColors:")
    print("gray    = top plane")
    print("blue    = U-side raw boundary points")
    print("cyan    = V-side raw boundary points")
    print("yellow  = stable-side line fit")
    print("magenta = defect-side line fit")
    print("red     = turn points")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Part 1 Boundary + TwoFit + Turn Points", width=1400, height=900)

    for geom in geoms:
        vis.add_geometry(geom)

    render = vis.get_render_option()
    render.point_size = POINT_SIZE
    render.background_color = np.asarray([1.0, 1.0, 1.0])

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()

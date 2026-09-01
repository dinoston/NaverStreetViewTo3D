import numpy as np

from street3d.architectural import make_architectural_solid


def test_architectural_solid_expands_collapsed_depth_and_is_watertight(tmp_path):
    import open3d as o3d

    rng = np.random.default_rng(42)
    points = np.column_stack((
        rng.uniform(-5, 5, 2000),
        rng.normal(0, 0.08, 2000),
        rng.uniform(0, 4, 2000),
    ))
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    source = tmp_path / "building_clean.ply"
    o3d.io.write_point_cloud(str(source), cloud)

    point_path, ply_path, obj_path = make_architectural_solid(source, tmp_path / "mesh")
    mesh = o3d.io.read_triangle_mesh(str(ply_path))
    span = mesh.get_max_bound() - mesh.get_min_bound()

    assert point_path.exists() and obj_path.exists()
    assert mesh.is_watertight()
    assert span[1] >= span[0] * 0.51

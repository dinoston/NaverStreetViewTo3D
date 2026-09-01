from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def _z_up_basis(points: np.ndarray, world_up: np.ndarray) -> np.ndarray:
    world_up = world_up / np.linalg.norm(world_up)
    centered = points - np.median(points, axis=0)
    horizontal = centered - np.outer(centered @ world_up, world_up)
    _, _, axes = np.linalg.svd(horizontal, full_matrices=False)
    x_axis = axes[0] - world_up * np.dot(axes[0], world_up)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(world_up, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    return np.column_stack((x_axis, y_axis, world_up))


def mesh_gaussian_centers(
    gaussian_ply: Path,
    reference_ply: Path,
    destination_dir: Path,
    opacity_threshold: float = 0.10,
    max_scale: float = 0.50,
    poisson_depth: int = 9,
) -> tuple[Path, Path, Path]:
    """Filter a 3DGS PLY to the target bounds and produce a colored Poisson mesh."""
    try:
        import open3d as o3d
        from plyfile import PlyData
    except ImportError as exc:
        raise RuntimeError("Open3D and plyfile are required for Gaussian mesh extraction.") from exc

    gaussian = PlyData.read(gaussian_ply)["vertex"].data
    reference = PlyData.read(reference_ply)["vertex"].data
    xyz = np.column_stack((gaussian["x"], gaussian["y"], gaussian["z"])).astype(np.float64)
    reference_xyz = np.column_stack(
        (reference["x"], reference["y"], reference["z"])
    ).astype(np.float64)

    lower, upper = np.percentile(reference_xyz, [1.0, 99.0], axis=0)
    margin = np.maximum((upper - lower) * 0.20, 0.25)
    opacity = _sigmoid(np.asarray(gaussian["opacity"], dtype=np.float64))
    scales = np.exp(np.column_stack(
        (gaussian["scale_0"], gaussian["scale_1"], gaussian["scale_2"])
    ).astype(np.float64))
    keep = (
        np.isfinite(xyz).all(axis=1)
        & (xyz >= lower - margin).all(axis=1)
        & (xyz <= upper + margin).all(axis=1)
        & (opacity >= opacity_threshold)
        & (scales.max(axis=1) <= max_scale)
    )
    if keep.sum() < 2_000:
        raise RuntimeError(f"Too few usable Gaussian centers for meshing: {int(keep.sum())}")

    xyz = xyz[keep]
    # Official 3DGS stores the DC spherical-harmonic color coefficient.
    sh0 = 0.28209479177387814
    colors = 0.5 + sh0 * np.column_stack(
        (gaussian["f_dc_0"], gaussian["f_dc_1"], gaussian["f_dc_2"])
    ).astype(np.float64)[keep]
    colors = np.clip(colors, 0.0, 1.0)

    report_path = reference_ply.with_suffix(".json")
    world_up = np.array([0.0, 0.0, 1.0])
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        candidate = np.asarray(report.get("world_up"), dtype=np.float64)
        if candidate.shape == (3,) and np.linalg.norm(candidate) > 0.5:
            world_up = candidate
    basis = _z_up_basis(xyz, world_up)
    xyz = (xyz - np.median(xyz, axis=0)) @ basis
    xyz[:, 2] -= np.percentile(xyz[:, 2], 0.5)

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(xyz)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    diagonal = float(np.linalg.norm(cloud.get_max_bound() - cloud.get_min_bound()))
    cloud = cloud.voxel_down_sample(max(diagonal / 600.0, 1e-4))
    cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=24, std_ratio=2.0)
    if len(cloud.points) < 1_000:
        raise RuntimeError("Gaussian filtering removed too many surface points.")

    labels = np.asarray(
        cloud.cluster_dbscan(eps=max(diagonal / 100.0, 1e-3), min_points=12)
    )
    valid_labels = labels[labels >= 0]
    if len(valid_labels):
        counts = np.bincount(valid_labels)
        main_label = int(np.argmax(counts))
        main_indices = np.flatnonzero(labels == main_label).tolist()
        if len(main_indices) >= 1_000:
            cloud = cloud.select_by_index(main_indices)

    normal_radius = max(diagonal / 90.0, 1e-3)
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=60)
    )
    cloud.orient_normals_consistent_tangent_plane(40)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        cloud, depth=poisson_depth, scale=1.05, linear_fit=True
    )
    densities = np.asarray(densities)
    mesh.remove_vertices_by_mask(densities < np.percentile(densities, 8.0))
    mesh = mesh.crop(cloud.get_axis_aligned_bounding_box())
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    if len(mesh.triangles) > 350_000:
        mesh = mesh.simplify_quadric_decimation(350_000)
        mesh.compute_vertex_normals()

    destination_dir.mkdir(parents=True, exist_ok=True)
    cloud_path = destination_dir / "gaussian_building_surface_points.ply"
    ply_path = destination_dir / "gaussian_building_mesh.ply"
    obj_path = destination_dir / "gaussian_building_mesh.obj"
    o3d.io.write_point_cloud(str(cloud_path), cloud, write_ascii=False, compressed=False)
    o3d.io.write_triangle_mesh(str(ply_path), mesh, write_ascii=False, compressed=False)
    o3d.io.write_triangle_mesh(str(obj_path), mesh, write_ascii=False)

    metadata = {
        "source": str(gaussian_ply),
        "reference": str(reference_ply),
        "input_gaussians": len(gaussian),
        "filtered_centers": int(keep.sum()),
        "surface_points": len(cloud.points),
        "mesh_vertices": len(mesh.vertices),
        "mesh_triangles": len(mesh.triangles),
        "opacity_threshold": opacity_threshold,
        "max_scale": max_scale,
        "poisson_depth": poisson_depth,
        "coordinate_system": "Z-up, ground near Z=0",
    }
    ply_path.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[ok] Gaussian mesh: {len(cloud.points):,} surface points, "
        f"{len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles"
    )
    return cloud_path, ply_path, obj_path

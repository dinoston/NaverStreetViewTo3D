from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from .building import extract_target_building_masks
from .util import images_in


def _evenly_sample(items: list[Path], limit: int) -> list[Path]:
    if len(items) <= limit:
        return items
    indices = np.linspace(0, len(items) - 1, limit).round().astype(int)
    return [items[index] for index in indices]


def _latest_capture_session(items: list[Path], manifest_path: Path, gap_minutes: int = 30) -> tuple[list[Path], list[str]]:
    if not manifest_path.exists():
        return items, []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_by_frame = {str(row.get("image")): str(row.get("source", "")) for row in manifest}
    dated: list[tuple[Path, datetime]] = []
    undated: list[Path] = []
    for path in items:
        match = re.search(r"(\d{4}-\d{2}-\d{2})\s+(\d{6})", source_by_frame.get(path.name, ""))
        if match:
            dated.append((path, datetime.strptime(" ".join(match.groups()), "%Y-%m-%d %H%M%S")))
        else:
            undated.append(path)
    if len(dated) < 2:
        return items, []
    dated.sort(key=lambda item: item[1])
    sessions: list[list[tuple[Path, datetime]]] = [[]]
    for item in dated:
        if sessions[-1] and (item[1] - sessions[-1][-1][1]).total_seconds() > gap_minutes * 60:
            sessions.append([])
        sessions[-1].append(item)
    chosen = max(sessions, key=lambda group: (len(group), group[-1][1]))
    selected = [path for path, _ in chosen]
    ignored = [path.name for path in items if path not in selected]
    return selected, ignored


def _write_ply(path: Path, points: np.ndarray, colors: np.ndarray, confidence: np.ndarray) -> None:
    from plyfile import PlyData, PlyElement

    vertex = np.empty(
        len(points),
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
               ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("confidence", "f4")],
    )
    vertex["x"], vertex["y"], vertex["z"] = points.T
    vertex["red"], vertex["green"], vertex["blue"] = colors.T
    vertex["confidence"] = confidence
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(path)


def _write_projection_preview(cloud, destination: Path) -> None:
    """Write front/top/side point projections for quick geometry QA."""
    import cv2

    points = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors)
    if not len(points):
        return
    if len(colors) != len(points):
        colors = np.full((len(points), 3), 0.72, dtype=np.float64)
    panels = []
    for title, horizontal, vertical in (("FRONT  X-Z", 0, 2), ("TOP  X-Y", 0, 1), ("SIDE  Y-Z", 1, 2)):
        panel = np.full((620, 620, 3), 30, dtype=np.uint8)
        coordinates = points[:, [horizontal, vertical]]
        lower, upper = np.percentile(coordinates, [1, 99], axis=0)
        span = np.maximum(upper - lower, 1e-8)
        normalized = np.clip((coordinates - lower) / span, 0, 1)
        pixels = (normalized * 559 + 30).astype(np.int32)
        pixels[:, 1] = 619 - pixels[:, 1]
        bgr = np.clip(colors[:, ::-1] * 255, 0, 255).astype(np.uint8)
        for (x, y), color in zip(pixels, bgr):
            cv2.circle(panel, (int(x), int(y)), 1, tuple(map(int, color)), -1)
        cv2.putText(panel, title, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 245, 245), 2)
        panels.append(panel)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), np.concatenate(panels, axis=1))


def _load_padded_masks(paths: list[Path], target_size: int = 518):
    import torch
    from PIL import Image

    masks = []
    for path in paths:
        image = Image.open(path).convert("L")
        width, height = image.size
        if width >= height:
            new_width = target_size
            new_height = round(height * (new_width / width) / 14) * 14
        else:
            new_height = target_size
            new_width = round(width * (new_height / height) / 14) * 14
        image = image.resize((new_width, new_height), Image.Resampling.NEAREST)
        array = np.asarray(image) > 127
        canvas = np.zeros((target_size, target_size), dtype=bool)
        top = (target_size - new_height) // 2
        left = (target_size - new_width) // 2
        canvas[top:top + new_height, left:left + new_width] = array
        masks.append(canvas)
    return torch.from_numpy(np.stack(masks))


def reconstruct_vggt(
    frames_dir: Path,
    output_dir: Path,
    repo_dir: Path,
    model_name: str,
    max_images: int,
    confidence_percentile: float,
    pixel_stride: int,
    segmentation_model: str,
    manifest_path: Path,
    target_annotation_dir: Path | None = None,
) -> Path:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is missing from the project environment.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("Fast VGGT mode requires an NVIDIA CUDA GPU.")
    if not repo_dir.joinpath("vggt", "models", "vggt.py").exists():
        raise RuntimeError(f"VGGT repository is missing: {repo_dir}")

    started = time.perf_counter()
    sources = images_in(frames_dir)
    if len(sources) < 2:
        raise RuntimeError("Fast reconstruction needs at least two input images.")
    sources, ignored_sessions = _latest_capture_session(sources, manifest_path)
    print(f"[fast] capture session: {len(sources)} images ({len(ignored_sessions)} older/unrelated ignored)")
    building_mask_paths, segmentation_report = extract_target_building_masks(
        sources, output_dir / "building_masks", segmentation_model, target_annotation_dir
    )
    usable = [
        path for path, item in zip(sources, segmentation_report)
        if bool(item["usable"])
    ]
    if len(usable) < 2:
        raise RuntimeError("A dominant building was not detected in at least two images.")
    selected = _evenly_sample(usable, max_images)

    sys.path.insert(0, str(repo_dir))
    from vggt.models.vggt import VGGT
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    print(f"[fast] loading {model_name} ({len(selected)}/{len(sources)} images)")
    model = VGGT.from_pretrained(model_name, local_files_only=True).to(device).eval()
    images = load_and_preprocess_images([str(path) for path in selected], mode="pad").to(device)
    building_masks = _load_padded_masks([building_mask_paths[path] for path in selected])

    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=dtype):
        tokens, patch_start = model.aggregator(images[None])
        pose_encoding = model.camera_head(tokens)[-1]
        depth, confidence = model.depth_head(tokens, images=images[None], patch_start_idx=patch_start)
    # VGGT's authors recommend camera + depth unprojection for more accurate
    # geometry than the direct point-map branch.
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_encoding, images.shape[-2:])
    rotations = extrinsic[0, :, :3, :3].float().cpu().numpy()
    # COLMAP/VGGT camera Y points downward in the image.  Transform camera-up
    # into the world frame and average it to obtain a stable scene gravity axis.
    camera_up = np.einsum("sji,j->si", rotations, np.array([0.0, -1.0, 0.0], np.float32))
    reference_up = camera_up[0]
    camera_up[np.einsum("si,i->s", camera_up, reference_up) < 0] *= -1
    world_up = camera_up.mean(axis=0)
    world_up /= max(float(np.linalg.norm(world_up)), 1e-8)
    points = unproject_depth_map_to_point_map(
        depth.squeeze(0).float(), extrinsic.squeeze(0).float(), intrinsic.squeeze(0).float()
    )[None]

    points = np.asarray(points[0, :, ::pixel_stride, ::pixel_stride], dtype=np.float32)
    confidence = confidence[0, :, ::pixel_stride, ::pixel_stride].float().cpu().numpy()
    colors = images[:, :, ::pixel_stride, ::pixel_stride].permute(0, 2, 3, 1).cpu().numpy()
    target_mask = building_masks[:, ::pixel_stride, ::pixel_stride].numpy().reshape(-1)
    points = points.reshape(-1, 3)
    confidence = confidence.reshape(-1)
    colors = colors.reshape(-1, 3)

    finite = np.isfinite(points).all(axis=1) & np.isfinite(confidence)
    threshold_values = confidence[finite & target_mask]
    if not len(threshold_values):
        raise RuntimeError("Building masks did not overlap any valid reconstructed points.")
    threshold = float(np.percentile(threshold_values, confidence_percentile))
    # Padded/transparent vegetation pixels are white; map UI and empty areas are often black.
    useful_color = (colors.mean(axis=1) > 0.02) & (colors.mean(axis=1) < 0.98)
    keep = finite & useful_color & target_mask & (confidence >= threshold)
    points = points[keep].astype(np.float32)
    colors_u8 = np.clip(colors[keep] * 255.0, 0, 255).astype(np.uint8)
    confidence = confidence[keep].astype(np.float32)

    # Trim isolated depth catastrophes using a robust object-centric bounding volume.
    if len(points):
        median = np.median(points, axis=0)
        deviation = np.median(np.abs(points - median), axis=0)
        deviation = np.maximum(deviation, 1e-5)
        robust = (np.abs(points - median) <= 8.0 * deviation).all(axis=1)
        points, colors_u8, confidence = points[robust], colors_u8[robust], confidence[robust]

    destination = output_dir / "pointcloud" / "fast_building_points.ply"
    _write_ply(destination, points, colors_u8, confidence)
    report = {
        "engine": "VGGT",
        "geometry": "camera_depth_unprojection",
        "model": model_name,
        "input_images": len(sources),
        "ignored_other_sessions": ignored_sessions,
        "used_images": [path.name for path in selected],
        "points": len(points),
        "confidence_percentile": confidence_percentile,
        "confidence_threshold": threshold,
        "pixel_stride": pixel_stride,
        "world_up": world_up.tolist(),
        "building_segmentation": segmentation_report,
        "seconds": round(time.perf_counter() - started, 2),
        "warning": "Geometry still requires translational parallax between Street View positions.",
    }
    destination.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] fast point cloud: {destination} ({len(points):,} points, {report['seconds']} sec)")
    return destination


def clean_point_cloud_and_make_mesh(point_cloud_path: Path, mesh_dir: Path) -> tuple[Path, Path, Path] | None:
    """Keep the main spatial component and make a quick Poisson preview mesh."""
    try:
        import open3d as o3d
    except ImportError:
        print("[warn] Open3D is not installed; skipping clean point cloud and preview mesh.")
        return None

    started = time.perf_counter()
    cloud = o3d.io.read_point_cloud(str(point_cloud_path))
    points = np.asarray(cloud.points)
    if len(points) < 100:
        raise RuntimeError("Too few building points to clean or mesh.")
    diagonal = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    labels = np.asarray(cloud.cluster_dbscan(eps=diagonal * 0.018, min_points=20, print_progress=False))
    valid = labels >= 0
    if valid.any():
        cluster_ids, counts = np.unique(labels[valid], return_counts=True)
        largest = cluster_ids[int(np.argmax(counts))]
        cloud = cloud.select_by_index(np.flatnonzero(labels == largest))
    cloud = cloud.voxel_down_sample(max(diagonal / 350.0, 1e-6))
    cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.5)

    # VGGT's world frame is arbitrary.  Use camera-derived gravity for Z-up,
    # then choose the dominant horizontal extent as X.
    clean_points = np.asarray(cloud.points)
    center = np.mean(clean_points, axis=0)
    report_path = point_cloud_path.with_suffix(".json")
    world_up = None
    if report_path.exists():
        try:
            world_up = np.asarray(json.loads(report_path.read_text(encoding="utf-8")).get("world_up"), dtype=float)
        except (TypeError, ValueError, json.JSONDecodeError):
            world_up = None
    if world_up is None or world_up.shape != (3,) or np.linalg.norm(world_up) < 0.5:
        _, _, axes = np.linalg.svd(clean_points - center, full_matrices=False)
        world_up = axes[1]
    world_up = world_up / np.linalg.norm(world_up)
    centered = clean_points - center
    horizontal = centered - np.outer(centered @ world_up, world_up)
    _, _, horizontal_axes = np.linalg.svd(horizontal, full_matrices=False)
    x_axis = horizontal_axes[0]
    x_axis -= world_up * np.dot(x_axis, world_up)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(world_up, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    basis = np.column_stack((x_axis, y_axis, world_up))
    if np.linalg.det(basis) < 0:
        basis[:, 1] *= -1
    oriented = centered @ basis
    oriented[:, 2] -= oriented[:, 2].min()
    cloud.points = o3d.utility.Vector3dVector(oriented)

    stem = point_cloud_path.stem
    clean_path = point_cloud_path.with_name(f"{stem}_clean.ply")
    o3d.io.write_point_cloud(str(clean_path), cloud, write_ascii=False, compressed=False)
    _write_projection_preview(cloud, point_cloud_path.with_name(f"{stem}_preview.png"))
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=diagonal / 35.0, max_nn=50))
    cloud.orient_normals_consistent_tangent_plane(30)
    mesh, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud, depth=8, linear_fit=True)
    density = np.asarray(density)
    mesh.remove_vertices_by_mask(density < np.percentile(density, 5.0))
    mesh = mesh.crop(cloud.get_axis_aligned_bounding_box())
    mesh.compute_vertex_normals()
    mesh_dir.mkdir(parents=True, exist_ok=True)
    output_stem = stem.removesuffix("_points")
    mesh_path = mesh_dir / f"{output_stem}_mesh.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), mesh, write_ascii=False, compressed=False)

    # A clearly labelled architectural proxy closes unseen roof/back surfaces.
    lower, upper = np.percentile(oriented, [1.0, 99.0], axis=0)
    size = np.maximum(upper - lower, 1e-4)
    proxy = o3d.geometry.TriangleMesh.create_box(*size)
    proxy.translate(lower)
    proxy.compute_vertex_normals()
    proxy.paint_uniform_color([0.45, 0.48, 0.52])
    proxy_path = mesh_dir / f"{output_stem}_proxy_mesh.ply"
    o3d.io.write_triangle_mesh(str(proxy_path), proxy, write_ascii=False, compressed=False)
    print(
        f"[ok] clean cloud + preview mesh: {len(cloud.points):,} points, "
        f"{len(mesh.triangles):,} triangles ({time.perf_counter() - started:.2f} sec)"
    )
    return clean_path, mesh_path, proxy_path

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _box_mesh(lower: np.ndarray, upper: np.ndarray):
    """Create a consistently wound, watertight rectangular building shell."""
    import open3d as o3d

    lx, ly, lz = map(float, lower)
    ux, uy, uz = map(float, upper)
    vertices = np.asarray([
        [lx, ly, lz], [ux, ly, lz], [ux, uy, lz], [lx, uy, lz],
        [lx, ly, uz], [ux, ly, uz], [ux, uy, uz], [lx, uy, uz],
    ], dtype=np.float64)
    triangles = np.asarray([
        [0, 2, 1], [0, 3, 2],       # bottom
        [4, 5, 6], [4, 6, 7],       # roof
        [0, 1, 5], [0, 5, 4],       # front
        [1, 2, 6], [1, 6, 5],       # right
        [2, 3, 7], [2, 7, 6],       # back
        [3, 0, 4], [3, 4, 7],       # left
    ], dtype=np.int32)
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(triangles),
    )
    mesh.paint_uniform_color([0.52, 0.55, 0.59])
    mesh.compute_vertex_normals()
    return mesh


def _facade_texture(
    frame_path: Path, bbox: list[int], mask_path: Path | None, destination: Path
) -> None:
    """Crop a clean target view and neutralize pixels outside its building mask."""
    import cv2

    image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read facade frame: {frame_path}")
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1] - 1, x2), min(image.shape[0] - 1, y2)
    crop = image[y1:y2 + 1, x1:x2 + 1].copy()
    if mask_path and mask_path.exists():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            selected = mask[y1:y2 + 1, x1:x2 + 1] > 127
            if selected.shape == crop.shape[:2] and selected.any():
                # Keep the photo on the selected facade and replace cars, sky,
                # vegetation and map UI with a neutral tile-like background.
                neutral = np.full_like(crop, (118, 122, 128))
                crop = np.where(selected[..., None], crop, neutral)
    crop = cv2.resize(crop, (1536, 1024), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(destination), crop, [cv2.IMWRITE_PNG_COMPRESSION, 4])


def _write_textured_obj(
    lower: np.ndarray,
    upper: np.ndarray,
    obj_path: Path,
    long_texture: Path | None,
    short_texture: Path | None,
) -> None:
    lx, ly, lz = map(float, lower)
    ux, uy, uz = map(float, upper)
    vertices = [
        (lx, ly, lz), (ux, ly, lz), (ux, uy, lz), (lx, uy, lz),
        (lx, ly, uz), (ux, ly, uz), (ux, uy, uz), (lx, uy, uz),
    ]
    mtl_path = obj_path.with_suffix(".mtl")
    material_lines = [
        "newmtl Roof", "Kd 0.58 0.60 0.63", "Ka 0.10 0.10 0.10", "",
        "newmtl LongFacade", "Kd 0.72 0.72 0.72",
    ]
    if long_texture:
        material_lines.append(f"map_Kd {long_texture.name}")
    material_lines += ["", "newmtl ShortFacade", "Kd 0.72 0.72 0.72"]
    if short_texture:
        material_lines.append(f"map_Kd {short_texture.name}")
    mtl_path.write_text("\n".join(material_lines) + "\n", encoding="utf-8")

    lines = [f"mtllib {mtl_path.name}", "o FINAL_building_solid"]
    lines += [f"v {x:.9g} {y:.9g} {z:.9g}" for x, y, z in vertices]
    lines += ["vt 0 0", "vt 1 0", "vt 1 1", "vt 0 1"]
    lines += [
        "usemtl Roof", "f 1/1 3/3 2/2", "f 1/1 4/4 3/3",
        "f 5/1 6/2 7/3", "f 5/1 7/3 8/4",
        "usemtl LongFacade", "f 1/1 2/2 6/3", "f 1/1 6/3 5/4",
        "f 3/1 4/2 8/3", "f 3/1 8/3 7/4",
        "usemtl ShortFacade", "f 2/1 3/2 7/3", "f 2/1 7/3 6/4",
        "f 4/1 1/2 5/3", "f 4/1 5/3 8/4",
    ]
    obj_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_architectural_solid(
    clean_point_cloud: Path,
    destination_dir: Path,
    minimum_depth_ratio: float = 0.52,
    frames_dir: Path | None = None,
    target_report_path: Path | None = None,
    masks_dir: Path | None = None,
) -> tuple[Path, Path, Path]:
    """Regularize an uncertain reconstruction into a usable building solid.

    Street-view screenshots frequently constrain the facade but not the rear
    wall, which collapses depth and turns Poisson meshes into sheets.  Preserve
    robust observed width/height while enforcing a conservative building depth,
    then export a closed Manhattan-world shell and regularly sampled point cloud.
    """
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("Open3D is required for architectural solid export.") from exc

    cloud = o3d.io.read_point_cloud(str(clean_point_cloud))
    points = np.asarray(cloud.points)
    if len(points) < 100:
        raise RuntimeError("Too few points for architectural regularization.")

    lower, upper = np.percentile(points, [1.0, 99.0], axis=0)
    span = np.maximum(upper - lower, 1e-5)
    # clean_point_cloud is already Z-up. X is the dominant horizontal axis.
    # A depth smaller than this ratio is an observation failure, not a plausible
    # complete building, so expand symmetrically around the observed median.
    required_depth = max(float(span[1]), float(span[0]) * minimum_depth_ratio)
    depth_center = float(np.median(points[:, 1]))
    lower[1] = depth_center - required_depth / 2.0
    upper[1] = depth_center + required_depth / 2.0
    lower[2] = 0.0
    final_span = upper - lower

    solid = _box_mesh(lower, upper)
    destination_dir.mkdir(parents=True, exist_ok=True)
    ply_path = destination_dir / "FINAL_building_solid.ply"
    obj_path = destination_dir / "FINAL_building_solid.obj"
    o3d.io.write_triangle_mesh(str(ply_path), solid, write_ascii=False, compressed=False)
    long_texture = None
    short_texture = None
    texture_sources: dict[str, str] = {}
    if frames_dir and target_report_path and target_report_path.exists():
        target_rows = json.loads(target_report_path.read_text(encoding="utf-8"))
        # Inline copies and separately copied guides can describe the same view.
        unique = {str(row["matched_source"]): row for row in target_rows}.values()
        candidates = []
        for row in unique:
            bbox = list(map(int, row["source_bbox"]))
            width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            frame_path = frames_dir / str(row["matched_source"])
            if width > 20 and height > 20 and frame_path.exists():
                candidates.append((width / height, row, frame_path))
        if candidates:
            long_ratio = float(final_span[0] / final_span[2])
            short_ratio = float(final_span[1] / final_span[2])
            long_choice = min(candidates, key=lambda item: abs(np.log(item[0] / long_ratio)))
            short_choice = min(candidates, key=lambda item: abs(np.log(item[0] / short_ratio)))
            long_texture = destination_dir / "FINAL_facade_long.png"
            short_texture = destination_dir / "FINAL_facade_short.png"
            for choice, texture in ((long_choice, long_texture), (short_choice, short_texture)):
                _, row, frame_path = choice
                mask_path = masks_dir / f"{frame_path.stem}_building.png" if masks_dir else None
                _facade_texture(frame_path, row["source_bbox"], mask_path, texture)
            texture_sources = {
                "long_facade": long_choice[2].name,
                "short_facade": short_choice[2].name,
            }
    _write_textured_obj(lower, upper, obj_path, long_texture, short_texture)

    # A dense, perfectly planar point cloud is useful when the downstream tool
    # expects PLY points rather than a triangle mesh.
    target_points = int(np.clip(len(points) // 2, 20_000, 100_000))
    regularized = solid.sample_points_uniformly(number_of_points=target_points)
    point_path = destination_dir.parent / "pointcloud" / "FINAL_building_regularized_points.ply"
    point_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(point_path), regularized, write_ascii=False, compressed=False)

    report = {
        "method": "architectural_manhattan_solid",
        "source": str(clean_point_cloud),
        "observed_span": span.tolist(),
        "regularized_span": final_span.tolist(),
        "minimum_depth_ratio": minimum_depth_ratio,
        "watertight": bool(solid.is_watertight()),
        "vertices": len(solid.vertices),
        "triangles": len(solid.triangles),
        "regularized_points": len(regularized.points),
        "texture_sources": texture_sources,
        "note": "Rear wall and roof are inferred because screenshots do not observe them directly.",
    }
    (destination_dir / "FINAL_building_solid.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        "[ok] architectural solid: "
        f"{final_span[0]:.3f} x {final_span[1]:.3f} x {final_span[2]:.3f}, "
        f"watertight={report['watertight']}"
    )
    return point_path, ply_path, obj_path

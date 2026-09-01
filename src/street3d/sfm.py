from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from .fast import _write_ply


def _qvec_to_rotation(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = qvec / np.linalg.norm(qvec)
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ], dtype=np.float64)


def filter_colmap_building_points(
    text_model: Path,
    masks_dir: Path,
    used_images: list[str],
    destination: Path,
    min_observations: int = 2,
) -> Path | None:
    """Keep COLMAP points observed inside the selected building in 2+ views."""
    from PIL import Image

    images_file = text_model / "images.txt"
    points_file = text_model / "points3D.txt"
    if not images_file.exists() or not points_file.exists():
        return None

    used = set(used_images)
    masks: dict[str, np.ndarray] = {}
    for name in used:
        mask_path = masks_dir / f"{Path(name).stem}_building.png"
        if mask_path.exists():
            masks[name] = np.asarray(Image.open(mask_path).convert("L")) > 127

    lines = [line.strip() for line in images_file.read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.startswith("#")]
    observations: dict[int, int] = {}
    camera_up: list[np.ndarray] = []
    for index in range(0, len(lines), 2):
        header = lines[index].split()
        if len(header) < 10:
            continue
        name = header[9]
        if name not in masks:
            continue
        rotation = _qvec_to_rotation(np.asarray(header[1:5], dtype=float))
        camera_up.append(rotation.T @ np.array([0.0, -1.0, 0.0]))
        values = lines[index + 1].split()
        mask = masks[name]
        height, width = mask.shape
        for item in range(0, len(values) - 2, 3):
            point_id = int(values[item + 2])
            if point_id < 0:
                continue
            x, y = int(round(float(values[item]))), int(round(float(values[item + 1])))
            if 0 <= x < width and 0 <= y < height and mask[y, x]:
                observations[point_id] = observations.get(point_id, 0) + 1

    xyz, rgb, confidence = [], [], []
    for line in points_file.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        values = line.split()
        point_id = int(values[0])
        count = observations.get(point_id, 0)
        if count >= min_observations:
            xyz.append([float(value) for value in values[1:4]])
            rgb.append([int(value) for value in values[4:7]])
            confidence.append(float(count))
    if len(xyz) < 100:
        print(f"[warn] only {len(xyz)} target COLMAP points passed the multi-view filter")
        return None

    up = np.mean(camera_up, axis=0)
    up /= np.linalg.norm(up)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_ply(
        destination, np.asarray(xyz, np.float32), np.asarray(rgb, np.uint8),
        np.asarray(confidence, np.float32),
    )
    report = {
        "engine": "COLMAP sparse SfM",
        "used_images": sorted(masks),
        "points": len(xyz),
        "minimum_target_observations": min_observations,
        "world_up": up.tolist(),
        "purpose": "geometry-accurate sparse reference for the dense VGGT cloud",
    }
    destination.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[ok] target-filtered COLMAP cloud: {destination} ({len(xyz):,} points)")
    return destination


def write_filtered_colmap_text_model(
    text_model: Path,
    masks_dir: Path,
    used_images: list[str],
    destination: Path,
    min_observations: int = 2,
) -> tuple[int, int]:
    """Write a valid text COLMAP model containing only target cameras/points."""
    from PIL import Image

    used = set(used_images)
    masks = {
        name: np.asarray(Image.open(masks_dir / f"{Path(name).stem}_building.png").convert("L")) > 127
        for name in used
        if (masks_dir / f"{Path(name).stem}_building.png").exists()
    }
    source_lines = [
        line.strip() for line in (text_model / "images.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    kept_image_lines: list[str] = []
    observations: dict[int, int] = {}
    for index in range(0, len(source_lines), 2):
        header, points_line = source_lines[index], source_lines[index + 1]
        values = header.split()
        name = values[9] if len(values) >= 10 else ""
        if name not in masks:
            continue
        kept_image_lines.extend((header, points_line))
        mask = masks[name]
        height, width = mask.shape
        point_values = points_line.split()
        for item in range(0, len(point_values) - 2, 3):
            point_id = int(point_values[item + 2])
            if point_id < 0:
                continue
            x, y = int(round(float(point_values[item]))), int(round(float(point_values[item + 1])))
            if 0 <= x < width and 0 <= y < height and mask[y, x]:
                observations[point_id] = observations.get(point_id, 0) + 1

    point_lines = []
    for line in (text_model / "points3D.txt").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#") and observations.get(int(line.split()[0]), 0) >= min_observations:
            point_lines.append(line)
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(text_model / "cameras.txt", destination / "cameras.txt")
    (destination / "images.txt").write_text(
        "# Target-filtered image list\n" + "\n".join(kept_image_lines) + "\n", encoding="utf-8"
    )
    (destination / "points3D.txt").write_text(
        "# Target-filtered 3D points\n" + "\n".join(point_lines) + "\n", encoding="utf-8"
    )
    return len(kept_image_lines) // 2, len(point_lines)

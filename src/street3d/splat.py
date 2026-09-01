from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from .util import images_in


def prepare_building_splat_dataset(
    frames_dir: Path,
    building_masks_dir: Path,
    fast_report_path: Path,
    destination: Path,
) -> tuple[Path, int]:
    """Create an RGBA 3DGS dataset containing only the selected target building."""
    if not fast_report_path.exists():
        raise RuntimeError("Run the fast stage first so target building views can be selected.")
    report = json.loads(fast_report_path.read_text(encoding="utf-8"))
    used = set(report.get("used_images", []))
    if len(used) < 2:
        raise RuntimeError("Too few usable building views for Gaussian Splatting.")

    images_dir = destination / "images"
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    kept = 0
    for frame in images_in(frames_dir):
        rgba = Image.open(frame).convert("RGBA")
        array = np.asarray(rgba).copy()
        if frame.name in used:
            mask_path = building_masks_dir / f"{frame.stem}_building.png"
            if not mask_path.exists():
                raise RuntimeError(f"Building mask missing: {mask_path}")
            alpha = np.asarray(Image.open(mask_path).convert("L"))
            kept += 1
            array[..., 3] = np.minimum(array[..., 3], alpha)
        else:
            array[..., 3] = 0
        # The official trainer masks the rendered image with alpha but leaves
        # RGB ground truth untouched, so transparent RGB must also be black.
        alpha_float = array[..., 3:4].astype(np.float32) / 255.0
        array[..., :3] = np.round(array[..., :3] * alpha_float).astype(np.uint8)
        Image.fromarray(array, mode="RGBA").save(images_dir / frame.name)

    return destination, kept


def write_3dgs_initial_points(source: Path, destination: Path) -> None:
    """Convert the target-filtered COLMAP PLY to the 3DGS loader schema."""
    from plyfile import PlyData, PlyElement

    source_vertex = PlyData.read(source)["vertex"].data
    vertex = np.empty(
        len(source_vertex),
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
               ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
               ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    for field in ("x", "y", "z", "red", "green", "blue"):
        vertex[field] = source_vertex[field]
    vertex["nx"] = vertex["ny"] = vertex["nz"] = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(destination)

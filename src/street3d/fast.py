from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from .util import images_in


def _evenly_sample(items: list[Path], limit: int) -> list[Path]:
    if len(items) <= limit:
        return items
    indices = np.linspace(0, len(items) - 1, limit).round().astype(int)
    return [items[index] for index in indices]


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


def reconstruct_vggt(
    frames_dir: Path,
    output_dir: Path,
    repo_dir: Path,
    model_name: str,
    max_images: int,
    confidence_percentile: float,
    pixel_stride: int,
) -> Path:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is missing from the project environment.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("Fast VGGT mode requires an NVIDIA CUDA GPU.")
    if not repo_dir.joinpath("vggt", "models", "vggt.py").exists():
        raise RuntimeError(f"VGGT repository is missing: {repo_dir}")

    sources = images_in(frames_dir)
    if len(sources) < 2:
        raise RuntimeError("Fast reconstruction needs at least two input images.")
    selected = _evenly_sample(sources, max_images)

    sys.path.insert(0, str(repo_dir))
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images

    started = time.perf_counter()
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    print(f"[fast] loading {model_name} ({len(selected)}/{len(sources)} images)")
    model = VGGT.from_pretrained(model_name, local_files_only=True).to(device).eval()
    images = load_and_preprocess_images([str(path) for path in selected], mode="pad").to(device)

    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=dtype):
        tokens, patch_start = model.aggregator(images[None])
        points, confidence = model.point_head(tokens, images=images[None], patch_start_idx=patch_start)

    points = points[0, :, ::pixel_stride, ::pixel_stride].float().cpu().numpy()
    confidence = confidence[0, :, ::pixel_stride, ::pixel_stride].float().cpu().numpy()
    colors = images[:, :, ::pixel_stride, ::pixel_stride].permute(0, 2, 3, 1).cpu().numpy()
    points = points.reshape(-1, 3)
    confidence = confidence.reshape(-1)
    colors = colors.reshape(-1, 3)

    threshold = float(np.percentile(confidence[np.isfinite(confidence)], confidence_percentile))
    finite = np.isfinite(points).all(axis=1) & np.isfinite(confidence)
    # Padded/transparent vegetation pixels are white; map UI and empty areas are often black.
    useful_color = (colors.mean(axis=1) > 0.02) & (colors.mean(axis=1) < 0.98)
    keep = finite & useful_color & (confidence >= threshold)
    points = points[keep].astype(np.float32)
    colors_u8 = np.clip(colors[keep] * 255.0, 0, 255).astype(np.uint8)
    confidence = confidence[keep].astype(np.float32)

    destination = output_dir / "pointcloud" / "fast_building_points.ply"
    _write_ply(destination, points, colors_u8, confidence)
    report = {
        "engine": "VGGT",
        "model": model_name,
        "input_images": len(sources),
        "used_images": [path.name for path in selected],
        "points": len(points),
        "confidence_percentile": confidence_percentile,
        "confidence_threshold": threshold,
        "pixel_stride": pixel_stride,
        "seconds": round(time.perf_counter() - started, 2),
        "warning": "Geometry still requires translational parallax between Street View positions.",
    }
    destination.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] fast point cloud: {destination} ({len(points):,} points, {report['seconds']} sec)")
    return destination

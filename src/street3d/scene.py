from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

from .fast import _write_ply


def coherent_panorama_views(
    frames_dir: Path, registered_names: list[str], max_images: int = 16
) -> list[Path]:
    """Select yaw cohorts that COLMAP proved overlap across panorama stations."""
    view_ids = []
    for name in registered_names:
        match = re.search(r"_v(\d+)_", name)
        if match:
            view_ids.append(int(match.group(1)))
    if not view_ids:
        raise RuntimeError("No registered panorama yaw cohort was found.")
    counts = {view: view_ids.count(view) for view in set(view_ids)}
    peak = max(counts, key=counts.get)
    # Include the strongest direction and directly adjacent cube view. This
    # covers a building corner without mixing the opposite side of the street.
    chosen_views = {peak, (peak - 1) % 8, (peak + 1) % 8}
    candidates = sorted(
        path for path in frames_dir.glob("p*_v*.png")
        if (match := re.search(r"_v(\d+)_", path.name))
        and int(match.group(1)) in chosen_views
    )
    if len(candidates) <= max_images:
        return candidates
    indices = np.linspace(0, len(candidates) - 1, max_images).round().astype(int)
    return [candidates[index] for index in indices]


def reconstruct_vggt_scene(
    image_paths: list[Path], destination: Path, repo_dir: Path, model_name: str,
    confidence_percentile: float = 60.0, pixel_stride: int = 2,
) -> Path:
    """Reconstruct one coherent panorama direction without building masks."""
    import torch

    if len(image_paths) < 3:
        raise RuntimeError("Panorama scene reconstruction needs at least three coherent views.")
    if not torch.cuda.is_available():
        raise RuntimeError("Panorama scene reconstruction requires an NVIDIA CUDA GPU.")
    started = time.perf_counter()
    sys.path.insert(0, str(repo_dir))
    from vggt.models.vggt import VGGT
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    print(f"[panorama-fast] loading {model_name} ({len(image_paths)} coherent views)")
    model = VGGT.from_pretrained(model_name, local_files_only=True).to(device).eval()
    images = load_and_preprocess_images([str(path) for path in image_paths], mode="pad").to(device)
    # The loader converts RGBA to RGB, so preserve the preprocessing alpha mask
    # separately. It removes sky, vegetation and moving objects from the cloud.
    from PIL import Image
    alpha_masks = []
    target_size = 518
    for path in image_paths:
        rgba = Image.open(path).convert("RGBA")
        alpha = rgba.getchannel("A")
        width, height = alpha.size
        if width >= height:
            new_width = target_size
            new_height = round(height * (new_width / width) / 14) * 14
        else:
            new_height = target_size
            new_width = round(width * (new_height / height) / 14) * 14
        alpha = alpha.resize((new_width, new_height), Image.Resampling.NEAREST)
        canvas = np.zeros((target_size, target_size), dtype=bool)
        top, left = (target_size - new_height) // 2, (target_size - new_width) // 2
        canvas[top:top + new_height, left:left + new_width] = np.asarray(alpha) > 127
        alpha_masks.append(canvas)
    alpha_mask = np.stack(alpha_masks)[:, ::pixel_stride, ::pixel_stride].reshape(-1)
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=dtype):
        tokens, patch_start = model.aggregator(images[None])
        pose_encoding = model.camera_head(tokens)[-1]
        depth, confidence = model.depth_head(tokens, images=images[None], patch_start_idx=patch_start)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_encoding, images.shape[-2:])
    rotations = extrinsic[0, :, :3, :3].float().cpu().numpy()
    camera_up = np.einsum("sji,j->si", rotations, np.array([0.0, -1.0, 0.0], np.float32))
    reference_up = camera_up[0]
    camera_up[np.einsum("si,i->s", camera_up, reference_up) < 0] *= -1
    world_up = camera_up.mean(axis=0)
    world_up /= max(float(np.linalg.norm(world_up)), 1e-8)
    points = unproject_depth_map_to_point_map(
        depth.squeeze(0).float(), extrinsic.squeeze(0).float(), intrinsic.squeeze(0).float()
    )
    points = np.asarray(points[:, ::pixel_stride, ::pixel_stride], dtype=np.float32).reshape(-1, 3)
    confidence = confidence[0, :, ::pixel_stride, ::pixel_stride].float().cpu().numpy().reshape(-1)
    colors = images[:, :, ::pixel_stride, ::pixel_stride].permute(0, 2, 3, 1).cpu().numpy().reshape(-1, 3)
    finite = np.isfinite(points).all(axis=1) & np.isfinite(confidence)
    threshold = float(np.percentile(confidence[finite], confidence_percentile))
    useful_color = (colors.mean(axis=1) > 0.02) & (colors.mean(axis=1) < 0.98)
    keep = finite & useful_color & alpha_mask & (confidence >= threshold)
    points, colors, confidence = points[keep], colors[keep], confidence[keep]
    median = np.median(points, axis=0)
    deviation = np.maximum(np.median(np.abs(points - median), axis=0), 1e-5)
    robust = (np.abs(points - median) <= 10.0 * deviation).all(axis=1)
    points, colors, confidence = points[robust], colors[robust], confidence[robust]
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_ply(destination, points, np.clip(colors * 255, 0, 255).astype(np.uint8), confidence)
    report = {
        "engine": "VGGT", "mode": "coherent_panorama_scene",
        "used_images": [path.name for path in image_paths], "points": len(points),
        "confidence_percentile": confidence_percentile,
        "confidence_threshold": threshold, "pixel_stride": pixel_stride,
        "world_up": world_up.tolist(), "seconds": round(time.perf_counter() - started, 2),
    }
    destination.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[ok] panorama scene point cloud: {destination} ({len(points):,} points)")
    return destination

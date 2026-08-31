from __future__ import annotations

from pathlib import Path

import numpy as np


def extract_target_building_masks(
    image_paths: list[Path], output_dir: Path, model_name: str
) -> tuple[dict[Path, Path], list[dict[str, object]]]:
    """Segment the dominant, center-biased building instance in each screenshot."""
    try:
        import cv2
        import torch
        import torch.nn.functional as functional
        from PIL import Image
        from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
    except ImportError as exc:
        raise RuntimeError("Building segmentation dependencies are missing.") from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(model_name, local_files_only=True)
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name, use_safetensors=True, local_files_only=True
    ).to(device).eval()
    building_ids = [
        int(index) for index, label in model.config.id2label.items()
        if str(label).casefold() == "building"
    ]
    if not building_ids:
        raise RuntimeError(f"The segmentation model has no building label: {model_name}")

    output_dir.mkdir(parents=True, exist_ok=True)
    prepared: list[dict[str, object]] = []
    for image_path in image_paths:
        rgba = Image.open(image_path).convert("RGBA")
        rgb = rgba.convert("RGB")
        inputs = {key: value.to(device) for key, value in processor(images=rgb, return_tensors="pt").items()}
        with torch.inference_mode():
            logits = model(**inputs).logits
        labels = functional.interpolate(
            logits, size=(rgb.height, rgb.width), mode="bilinear", align_corners=False
        ).argmax(dim=1)[0].cpu().numpy()
        raw = np.isin(labels, building_ids).astype(np.uint8)

        # Remove thin noise, then select one dominant facade instead of every building.
        base = max(5, (min(rgb.size) // 100) | 1)
        kernel = np.ones((base, base), dtype=np.uint8)
        raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel)
        count, components, stats, centroids = cv2.connectedComponentsWithStats(raw, connectivity=8)
        height, width = raw.shape
        rgb_array = np.asarray(rgb)
        candidates: list[dict[str, object]] = []
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < height * width * 0.008:
                continue
            cx, cy = centroids[label]
            distance = np.hypot((cx - width / 2) / width, (cy - height / 2) / height)
            component_mask = components == label
            pixels = rgb_array[component_mask]
            # Median facade color is stable against windows, signs and sunlight.
            color = np.median(pixels[::max(1, len(pixels) // 10000)], axis=0)
            candidates.append({
                "label": label, "area": area, "distance": float(distance), "color": color,
                "central_score": area * max(0.25, 1.25 - distance),
            })
        prepared.append({
            "path": image_path, "rgba": rgba, "components": components,
            "raw": raw, "candidates": candidates, "shape": (height, width),
        })

    provisional_colors = []
    for item in prepared:
        candidates = item["candidates"]
        if candidates:
            provisional_colors.append(max(candidates, key=lambda candidate: candidate["central_score"])["color"])
    target_color = np.median(np.stack(provisional_colors), axis=0) if provisional_colors else np.array([128, 128, 128])

    results: dict[Path, Path] = {}
    report: list[dict[str, object]] = []
    for item in prepared:
        image_path = item["path"]
        rgba = item["rgba"]
        components = item["components"]
        raw = item["raw"]
        height, width = item["shape"]
        candidates = item["candidates"]
        best_label = 0
        best_color_distance = None
        if candidates:
            def candidate_score(candidate):
                color_distance = np.linalg.norm(candidate["color"] - target_color) / 441.7
                area_fraction = candidate["area"] / (height * width)
                return color_distance + 0.08 * candidate["distance"] - 0.12 * np.sqrt(area_fraction)

            best = min(candidates, key=candidate_score)
            best_label = int(best["label"])
            best_color_distance = float(np.linalg.norm(best["color"] - target_color))

        selected = (components == best_label).astype(np.uint8) if best_label else raw
        contours, _ = cv2.findContours(selected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        silhouette = np.zeros_like(selected)
        cv2.drawContours(silhouette, contours, -1, 1, thickness=cv2.FILLED)
        silhouette = cv2.dilate(silhouette, np.ones((5, 5), np.uint8), iterations=1)
        # Vegetation was made transparent in preprocessing; never restore it here.
        alpha = np.asarray(rgba.getchannel("A")) > 127
        silhouette = (silhouette.astype(bool) & alpha).astype(np.uint8) * 255

        destination = output_dir / f"{image_path.stem}_building.png"
        Image.fromarray(silhouette, mode="L").save(destination)
        coverage = float((silhouette > 0).mean())
        usable = coverage >= 0.08 and (best_color_distance is None or best_color_distance <= 50.0)
        results[image_path] = destination
        report.append({
            "image": image_path.name, "coverage": round(coverage, 4),
            "usable": usable,
            "appearance_distance": None if best_color_distance is None else round(best_color_distance, 2),
        })
    return results, report

from __future__ import annotations

from pathlib import Path

import numpy as np

from .target import load_target_guides, reference_evidence


def extract_target_building_masks(
    image_paths: list[Path], output_dir: Path, model_name: str,
    annotation_dir: Path | None = None,
) -> tuple[dict[Path, Path], list[dict[str, object]]]:
    """Segment one building, using red-box annotations when available."""
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
    guides = load_target_guides(
        image_paths, annotation_dir or Path("__no_target_annotations__"),
        output_dir.parent / "target_reference.json",
    )
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
        if guides:
            hits, projected_regions = reference_evidence(guides, rgb_array)
        else:
            hits, projected_regions = np.empty((0, 2), np.float32), []
        hit_labels = []
        for x, y in hits:
            ix, iy = int(round(x)), int(round(y))
            if 0 <= ix < width and 0 <= iy < height:
                hit_labels.append(int(components[iy, ix]))
        for candidate in candidates:
            candidate["reference_hits"] = hit_labels.count(int(candidate["label"]))
        prepared.append({
            "path": image_path, "rgba": rgba, "components": components,
            "raw": raw, "candidates": candidates, "shape": (height, width),
            "reference_matches": len(hits), "projected_regions": projected_regions,
        })

    provisional_colors = []
    # The matched annotation view fixes the target appearance and component.
    for guide in guides:
        for item in prepared:
            if item["path"] != guide["source"]:
                continue
            x1, y1, x2, y2 = guide["bbox"]
            for candidate in item["candidates"]:
                component = item["components"] == int(candidate["label"])
                overlap = int(component[y1:y2 + 1, x1:x2 + 1].sum())
                candidate["guide_overlap"] = overlap
            if item["candidates"]:
                provisional_colors.append(max(item["candidates"], key=lambda candidate: candidate.get("guide_overlap", 0))["color"])
    if not provisional_colors:
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
        best_reference_hits = 0
        if candidates:
            def candidate_score(candidate):
                color_distance = np.linalg.norm(candidate["color"] - target_color) / 441.7
                area_fraction = candidate["area"] / (height * width)
                return color_distance + 0.08 * candidate["distance"] - 0.12 * np.sqrt(area_fraction)

            guided = [candidate for candidate in candidates if candidate.get("reference_hits", 0) >= 4]
            if guided:
                best = max(
                    guided,
                    key=lambda candidate: (
                        candidate.get("reference_hits", 0),
                        -candidate_score(candidate),
                    ),
                )
            else:
                best = min(candidates, key=candidate_score)
            best_label = int(best["label"])
            best_color_distance = float(np.linalg.norm(best["color"] - target_color))
            best_reference_hits = int(best.get("reference_hits", 0))

        selected = (components == best_label).astype(np.uint8) if best_label else raw
        # Semantic segmentation can merge touching buildings.  The user box is
        # projected through feature geometry to cut the selected semantic blob
        # back to the requested physical building.
        projected_regions = item["projected_regions"]
        projection_clipped = False
        if guides and projected_regions:
            region = np.zeros_like(selected)
            for polygon in projected_regions:
                cv2.fillConvexPoly(region, np.round(polygon).astype(np.int32), 1)
            margin = max(9, (min(height, width) // 35) | 1)
            region = cv2.dilate(region, np.ones((margin, margin), np.uint8))
            clipped = selected & region
            if clipped.sum() >= height * width * 0.025:
                selected = clipped
                projection_clipped = True
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
        if guides:
            usable = (
                coverage >= 0.035
                and best_reference_hits >= 4
                and projection_clipped
                and (best_color_distance is None or best_color_distance <= 55.0)
            )
        else:
            usable = coverage >= 0.08 and (best_color_distance is None or best_color_distance <= 50.0)

        preview_dir = output_dir.parent / "building_previews"
        preview_dir.mkdir(parents=True, exist_ok=True)
        preview = np.asarray(rgba.convert("RGB")).copy()
        tint = np.array([40, 220, 80] if usable else [230, 55, 45], dtype=np.float32)
        selected_pixels = silhouette > 0
        preview[selected_pixels] = np.clip(
            preview[selected_pixels].astype(np.float32) * 0.45 + tint * 0.55, 0, 255
        ).astype(np.uint8)
        Image.fromarray(preview).save(preview_dir / f"{image_path.stem}_selection.jpg", quality=88)
        results[image_path] = destination
        report.append({
            "image": image_path.name, "coverage": round(coverage, 4),
            "usable": usable,
            "appearance_distance": None if best_color_distance is None else round(best_color_distance, 2),
            "target_feature_hits": best_reference_hits,
            "target_guided": bool(guides),
            "target_region_applied": projection_clipped,
        })
    return results, report

from __future__ import annotations

from pathlib import Path

from .util import images_in


def mask_vegetation(
    frames_dir: Path,
    masks_dir: Path,
    model_name: str,
    labels: list[str],
    dilation: int = 7,
) -> dict[str, str]:
    """Create COLMAP masks and RGBA training images with vegetation made transparent."""
    try:
        import cv2
        import numpy as np
        import torch
        import torch.nn.functional as functional
        from PIL import Image
        from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
    except ImportError as exc:
        raise RuntimeError("Masking dependencies missing. Run setup again.") from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name, use_safetensors=True
    ).to(device).eval()
    wanted = {label.casefold() for label in labels}
    excluded_ids = {
        int(class_id) for class_id, label in model.config.id2label.items()
        if str(label).casefold() in wanted
    }
    if not excluded_ids:
        raise RuntimeError(f"None of the requested mask labels exist in {model_name}: {labels}")

    masks_dir.mkdir(parents=True, exist_ok=True)
    renamed: dict[str, str] = {}
    for source in images_in(frames_dir):
        rgb = Image.open(source).convert("RGB")
        inputs = processor(images=rgb, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            logits = model(**inputs).logits
        segmentation = functional.interpolate(
            logits, size=(rgb.height, rgb.width), mode="bilinear", align_corners=False
        ).argmax(dim=1)[0].cpu().numpy()
        excluded = np.isin(segmentation, list(excluded_ids)).astype(np.uint8) * 255
        if dilation > 1:
            kernel = np.ones((dilation, dilation), dtype=np.uint8)
            excluded = cv2.dilate(excluded, kernel, iterations=1)
        valid = 255 - excluded

        destination = source.with_suffix(".png")
        rgba = np.dstack((np.asarray(rgb), valid))
        Image.fromarray(rgba, mode="RGBA").save(destination)
        if destination != source:
            source.unlink()
        colmap_mask = masks_dir / f"{destination.name}.png"
        cv2.imwrite(str(colmap_mask), valid)
        renamed[source.name] = destination.name
    return renamed

from __future__ import annotations

import math
from pathlib import Path

from .util import images_in


def _imports():
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Preprocessing dependencies missing. Run: pip install -e .") from exc
    return cv2, np


def _read_image(path: Path):
    """Read an image through bytes so Windows paths may contain Korean characters."""
    cv2, np = _imports()
    try:
        encoded = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def _write_jpeg(path: Path, image, quality: int) -> bool:
    cv2, np = _imports()
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if ok:
        encoded.tofile(str(path))
    return bool(ok)


def validate_panoramas(input_dir: Path) -> list[dict[str, object]]:
    cv2, _ = _imports()
    records = []
    for path in images_in(input_dir):
        image = _read_image(path)
        if image is None:
            records.append({"file": path.name, "valid": False, "reason": "decode_failed"})
            continue
        h, w = image.shape[:2]
        ratio = w / h
        records.append({
            "file": path.name,
            "width": w,
            "height": h,
            "ratio": round(ratio, 4),
            "valid": 1.85 <= ratio <= 2.15,
            "reason": "ok" if 1.85 <= ratio <= 2.15 else "expected_equirectangular_2_to_1",
        })
    return records


def validate_screenshots(input_dir: Path) -> list[dict[str, object]]:
    cv2, _ = _imports()
    records = []
    for path in images_in(input_dir):
        image = _read_image(path)
        if image is None:
            records.append({"file": path.name, "valid": False, "reason": "decode_failed"})
            continue
        h, w = image.shape[:2]
        valid = w >= 320 and h >= 240
        records.append({
            "file": path.name,
            "width": w,
            "height": h,
            "ratio": round(w / h, 4),
            "valid": valid,
            "reason": "ok" if valid else "image_too_small",
        })
    return records


def equirect_to_perspective(image, yaw: float, pitch: float, fov: float, size: int):
    cv2, np = _imports()
    h, w = image.shape[:2]
    half = math.tan(math.radians(fov) / 2.0)
    axis = np.linspace(-half, half, size, dtype=np.float32)
    x, y = np.meshgrid(axis, -axis)
    z = np.ones_like(x)
    rays = np.stack((x, y, z), axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)

    yr, pr = math.radians(yaw), math.radians(pitch)
    rotation_yaw = np.array([[math.cos(yr), 0, math.sin(yr)], [0, 1, 0], [-math.sin(yr), 0, math.cos(yr)]], dtype=np.float32)
    rotation_pitch = np.array([[1, 0, 0], [0, math.cos(pr), -math.sin(pr)], [0, math.sin(pr), math.cos(pr)]], dtype=np.float32)
    rays = rays @ (rotation_yaw @ rotation_pitch).T
    longitude = np.arctan2(rays[..., 0], rays[..., 2])
    latitude = np.arcsin(np.clip(rays[..., 1], -1.0, 1.0))
    map_x = ((longitude / (2 * np.pi) + 0.5) * w).astype(np.float32)
    map_y = ((0.5 - latitude / np.pi) * h).astype(np.float32)
    return cv2.remap(image, map_x, map_y, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_WRAP)


def decompose(input_dir: Path, output_dir: Path, face_size: int, fov: float, yaw_step: int,
              pitches: list[float], include_zenith: bool, jpeg_quality: int) -> list[dict[str, object]]:
    cv2, _ = _imports()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    views = [(float(yaw), float(pitch)) for pitch in pitches for yaw in range(0, 360, yaw_step)]
    if include_zenith:
        views += [(0.0, 90.0), (0.0, -90.0)]
    for pano_index, path in enumerate(images_in(input_dir)):
        image = _read_image(path)
        if image is None:
            continue
        for view_index, (yaw, pitch) in enumerate(views):
            frame = equirect_to_perspective(image, yaw, pitch, fov, face_size)
            name = f"p{pano_index:04d}_v{view_index:02d}_y{yaw:+06.1f}_p{pitch:+05.1f}.jpg"
            destination = output_dir / name
            _write_jpeg(destination, frame, jpeg_quality)
            manifest.append({"image": name, "source": path.name, "yaw": yaw, "pitch": pitch, "fov": fov})
    return manifest


def prepare_screenshots(
    input_dir: Path, output_dir: Path, jpeg_quality: int,
    target_dir: Path | None = None,
) -> list[dict[str, object]]:
    """Normalize ordinary perspective screenshots for COLMAP without reprojection."""
    cv2, np = _imports()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    for index, path in enumerate(images_in(input_dir)):
        image = _read_image(path)
        if image is None:
            continue
        h, w = image.shape[:2]
        if w < 320 or h < 240:
            continue
        # Allow the convenient workflow where annotated images are placed
        # directly in input/screenshots. Preserve the guide, but inpaint the red
        # stroke so it does not become facade texture or a COLMAP feature.
        from .target import _red_polygon

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        target_polygon = _red_polygon(rgb)
        inline_target = target_polygon is not None
        if inline_target:
            if target_dir is not None:
                import shutil

                target_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target_dir / f"inline_{path.name}")
            red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
            red_stroke = (
                (red > 190) & (red > green * 1.65) & (red > blue * 1.65) & (green < 135)
            ).astype(np.uint8) * 255
            red_stroke = cv2.dilate(red_stroke, np.ones((7, 7), np.uint8), iterations=1)
            image = cv2.inpaint(image, red_stroke, 5, cv2.INPAINT_TELEA)
        # Keep generated filenames ASCII-only for COLMAP and Windows console compatibility.
        name = f"s{index:04d}.jpg"
        destination = output_dir / name
        _write_jpeg(destination, image, jpeg_quality)
        manifest.append({
            "image": name,
            "source": path.name,
            "input_type": "perspective_screenshot",
            "width": w,
            "height": h,
            "inline_target_guide": inline_target,
        })
    return manifest

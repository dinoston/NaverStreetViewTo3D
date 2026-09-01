from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _rgb(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


def _red_box(image: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return the outer rectangle drawn by the user, if one is present."""
    red, green, blue = image[..., 0], image[..., 1], image[..., 2]
    pixels = (red > 190) & (red > green * 1.65) & (red > blue * 1.65) & (green < 125)
    # Prefer long straight border lines.  A plain min/max would accidentally
    # include red cars, signs or lettering inside the screenshot.
    columns = np.flatnonzero(pixels.sum(axis=0) > image.shape[0] * 0.20)
    rows = np.flatnonzero(pixels.sum(axis=1) > image.shape[1] * 0.12)
    if len(columns) >= 2 and len(rows) >= 2:
        x1, x2 = int(columns.min()), int(columns.max())
        y1, y2 = int(rows.min()), int(rows.max())
        if (x2 - x1) * (y2 - y1) >= image.shape[0] * image.shape[1] * 0.03:
            return x1, y1, x2, y2
    ys, xs = np.nonzero(pixels)
    if len(xs) < 150:
        return None
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    if (x2 - x1) * (y2 - y1) < image.shape[0] * image.shape[1] * 0.03:
        return None
    return x1, y1, x2, y2


def _red_polygon(image: np.ndarray) -> np.ndarray | None:
    """Return the user's largest red freehand outline as a simplified polygon."""
    import cv2

    red, green, blue = image[..., 0], image[..., 1], image[..., 2]
    pixels = (
        (red > 190) & (red > green * 1.65) & (red > blue * 1.65) & (green < 135)
    ).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(pixels, connectivity=8)
    if count <= 1:
        return None
    # The hand-drawn outline is normally the longest/largest connected red item;
    # isolated signs, cars and Korean lettering form much smaller components.
    component_id = max(
        range(1, count),
        key=lambda label: int(stats[label, cv2.CC_STAT_AREA])
        * max(int(stats[label, cv2.CC_STAT_WIDTH]), int(stats[label, cv2.CC_STAT_HEIGHT])),
    )
    component = (labels == component_id).astype(np.uint8) * 255
    gap = max(9, (min(image.shape[:2]) // 80) | 1)
    component = cv2.morphologyEx(
        component, cv2.MORPH_CLOSE, np.ones((gap, gap), np.uint8), iterations=2
    )
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    image_area = float(image.shape[0] * image.shape[1])
    if abs(cv2.contourArea(contour)) < image_area * 0.02:
        # A mouse/freehand rectangle is often left open at one corner.  Join all
        # long red strokes that belong to the outline before taking its hull;
        # this is more reliable than treating the longest U-shaped stroke alone.
        outline = np.zeros_like(component)
        minimum_span = min(image.shape[:2]) * 0.12
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            span = max(
                int(stats[label, cv2.CC_STAT_WIDTH]),
                int(stats[label, cv2.CC_STAT_HEIGHT]),
            )
            if area >= 25 and span >= minimum_span:
                outline[labels == label] = 255
        ys, xs = np.nonzero(outline if outline.any() else component)
        if len(xs) < 100:
            return None
        contour = cv2.convexHull(np.column_stack((xs, ys)).astype(np.int32))
    perimeter = cv2.arcLength(contour, True)
    polygon = cv2.approxPolyDP(contour, max(2.0, perimeter * 0.004), True)[:, 0]
    if len(polygon) < 3 or abs(cv2.contourArea(polygon)) < image_area * 0.02:
        return None
    return polygon.astype(np.float32)


def _features(image: np.ndarray, mask: np.ndarray | None = None):
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create(nfeatures=7000, contrastThreshold=0.02)
    return sift.detectAndCompute(gray, mask)


def _matches(query_descriptors, train_descriptors, ratio: float = 0.72):
    import cv2

    if query_descriptors is None or train_descriptors is None:
        return []
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(query_descriptors, train_descriptors, k=2)
    return [first for first, second in pairs if first.distance < ratio * second.distance]


def load_target_guides(
    image_paths: list[Path], annotation_dir: Path, report_path: Path
) -> list[dict[str, object]]:
    """Match red-box annotations to originals and build clean target descriptors."""
    import cv2

    annotations = sorted(
        path for path in annotation_dir.glob("*")
        if path.is_file() and path.suffix.casefold() in {".png", ".jpg", ".jpeg", ".webp"}
    ) if annotation_dir.exists() else []
    annotation_names = {path.name for path in annotations}
    annotations = [
        path for path in annotations
        if not (
            path.name.startswith("inline_")
            and path.name.removeprefix("inline_") in annotation_names
        )
    ]
    if not annotations:
        return []

    source_cache: dict[Path, tuple[np.ndarray, object, object]] = {}
    for source_path in image_paths:
        image = _rgb(source_path)
        keypoints, descriptors = _features(image)
        source_cache[source_path] = (image, keypoints, descriptors)

    guides: list[dict[str, object]] = []
    serializable: list[dict[str, object]] = []
    for annotation_path in annotations:
        annotated = _rgb(annotation_path)
        polygon = _red_polygon(annotated)
        if polygon is None:
            continue
        red, green, blue = annotated[..., 0], annotated[..., 1], annotated[..., 2]
        red_mask = ((red > 190) & (red > green * 1.65) & (red > blue * 1.65)).astype(np.uint8)
        feature_mask = (1 - cv2.dilate(red_mask, np.ones((9, 9), np.uint8))) * 255
        query_keypoints, query_descriptors = _features(annotated, feature_mask)

        best = None
        for source_path, (_, source_keypoints, source_descriptors) in source_cache.items():
            matches = _matches(query_descriptors, source_descriptors, ratio=0.70)
            if len(matches) < 8:
                continue
            query_points = np.float32([query_keypoints[item.queryIdx].pt for item in matches])
            source_points = np.float32([source_keypoints[item.trainIdx].pt for item in matches])
            homography, inliers = cv2.findHomography(query_points, source_points, cv2.RANSAC, 3.0)
            inlier_count = int(inliers.sum()) if inliers is not None else 0
            if homography is not None and (best is None or inlier_count > best[0]):
                best = (inlier_count, source_path, homography)
        if best is None or best[0] < 12:
            continue

        inlier_count, source_path, homography = best
        mapped = cv2.perspectiveTransform(polygon[None], homography)[0]
        source_image = source_cache[source_path][0]
        height, width = source_image.shape[:2]
        mapped[:, 0] = np.clip(mapped[:, 0], 0, width - 1)
        mapped[:, 1] = np.clip(mapped[:, 1], 0, height - 1)
        target_mask = np.zeros((height, width), np.uint8)
        cv2.fillConvexPoly(target_mask, np.round(mapped).astype(np.int32), 255)
        target_mask = cv2.erode(target_mask, np.ones((9, 9), np.uint8))
        target_keypoints, target_descriptors = _features(source_image, target_mask)
        if target_descriptors is None or len(target_descriptors) < 8:
            continue

        mapped_box = (
            int(mapped[:, 0].min()), int(mapped[:, 1].min()),
            int(mapped[:, 0].max()), int(mapped[:, 1].max()),
        )
        guides.append({
            "annotation": annotation_path,
            "source": source_path,
            "bbox": mapped_box,
            "polygon": mapped,
            "keypoints": target_keypoints,
            "descriptors": target_descriptors,
            "match_inliers": inlier_count,
        })
        serializable.append({
            "annotation": annotation_path.name,
            "matched_source": source_path.name,
            "source_bbox": list(mapped_box),
            "match_inliers": inlier_count,
            "target_features": len(target_descriptors),
            "outline_vertices": len(mapped),
        })

    if serializable:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"[target] {len(serializable)} red-box guide(s): "
            + ", ".join(f"{item['matched_source']} ({item['match_inliers']} inliers)" for item in serializable)
        )
    return guides


def reference_evidence(
    guides: list[dict[str, object]], image: np.ndarray
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return target-feature locations and projected target regions in a new view."""
    import cv2

    view_keypoints, view_descriptors = _features(image)
    if view_descriptors is None:
        return np.empty((0, 2), np.float32), []
    hit_points: list[np.ndarray] = []
    polygons: list[np.ndarray] = []
    height, width = image.shape[:2]
    for guide in guides:
        matches = _matches(guide["descriptors"], view_descriptors)
        if len(matches) < 5:
            continue
        query_points = np.float32([guide["keypoints"][item.queryIdx].pt for item in matches])
        view_points = np.float32([view_keypoints[item.trainIdx].pt for item in matches])
        homography, inliers = cv2.findHomography(query_points, view_points, cv2.RANSAC, 5.0)
        if homography is not None and inliers is not None and int(inliers.sum()) >= 5:
            view_points = view_points[inliers.ravel().astype(bool)]
            polygon = cv2.perspectiveTransform(
                np.asarray(guide["polygon"], dtype=np.float32)[None], homography
            )[0]
            polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
            area_fraction = abs(cv2.contourArea(polygon)) / float(width * height)
            if 0.02 <= area_fraction <= 0.92 and cv2.isContourConvex(np.round(polygon).astype(np.int32)):
                polygons.append(polygon)
        hit_points.append(view_points)
    points = np.concatenate(hit_points, axis=0) if hit_points else np.empty((0, 2), np.float32)
    return points, polygons

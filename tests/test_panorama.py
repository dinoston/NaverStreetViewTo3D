import numpy as np

import cv2

from street3d.panorama import equirect_to_perspective, prepare_screenshots, validate_screenshots


def test_projection_shape_and_center():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image[:, :100] = (10, 20, 30)
    image[:, 100:] = (200, 210, 220)
    output = equirect_to_perspective(image, yaw=0, pitch=0, fov=90, size=32)
    assert output.shape == (32, 32, 3)
    assert np.allclose(output[16, 16], image[50, 100], atol=10)


def test_prepare_ordinary_screenshot(tmp_path):
    source = tmp_path / "screenshots"
    output = tmp_path / "frames"
    source.mkdir()
    image = np.full((480, 640, 3), 127, dtype=np.uint8)
    cv2.imwrite(str(source / "0001.png"), image)

    checks = validate_screenshots(source)
    manifest = prepare_screenshots(source, output, jpeg_quality=95)

    assert checks[0]["valid"] is True
    assert manifest[0]["input_type"] == "perspective_screenshot"
    assert manifest[0]["inline_target_guide"] is False
    assert (output / manifest[0]["image"]).exists()


def test_inline_red_outline_is_saved_as_guide_and_removed_from_camera_image(tmp_path):
    source = tmp_path / "screenshots"
    output = tmp_path / "frames"
    targets = tmp_path / "target"
    source.mkdir()
    image = np.full((480, 640, 3), 180, dtype=np.uint8)
    outline = np.array([[120, 410], [150, 100], [480, 70], [540, 400]], np.int32)
    cv2.polylines(image, [outline], isClosed=True, color=(0, 0, 255), thickness=8)
    cv2.imwrite(str(source / "marked.png"), image)

    manifest = prepare_screenshots(source, output, jpeg_quality=95, target_dir=targets)
    prepared = cv2.imread(str(output / manifest[0]["image"]))
    red_pixels = (prepared[..., 2] > prepared[..., 1] * 1.5) & (prepared[..., 2] > 180)

    assert manifest[0]["inline_target_guide"] is True
    assert (targets / "inline_marked.png").exists()
    assert red_pixels.sum() < 100

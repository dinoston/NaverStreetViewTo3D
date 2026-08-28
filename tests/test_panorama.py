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
    assert (output / manifest[0]["image"]).exists()

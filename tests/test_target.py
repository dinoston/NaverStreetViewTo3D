import numpy as np

from street3d.target import _red_box, _red_polygon


def test_red_box_prefers_long_border_over_red_objects():
    image = np.zeros((600, 1000, 3), dtype=np.uint8)
    image[80:86, 200:801] = (255, 0, 0)
    image[500:506, 200:801] = (255, 0, 0)
    image[80:506, 200:206] = (255, 0, 0)
    image[80:506, 795:801] = (255, 0, 0)
    image[530:570, 20:100] = (255, 0, 0)  # red car/sign outside the box

    assert _red_box(image) == (200, 80, 800, 505)


def test_red_box_returns_none_without_annotation():
    image = np.full((300, 500, 3), (110, 120, 130), dtype=np.uint8)

    assert _red_box(image) is None


def test_red_polygon_keeps_freehand_shape_instead_of_bbox():
    import cv2

    image = np.zeros((500, 800, 3), dtype=np.uint8)
    outline = np.array([[180, 430], [220, 150], [410, 70], [650, 180], [590, 440]], np.int32)
    cv2.polylines(image, [outline], isClosed=True, color=(255, 0, 0), thickness=8)

    polygon = _red_polygon(image)

    assert polygon is not None
    assert cv2.contourArea(polygon) > 100_000
    assert cv2.contourArea(polygon) < (650 - 180) * (440 - 70) * 0.9

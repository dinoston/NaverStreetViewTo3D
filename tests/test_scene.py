from street3d.scene import coherent_panorama_views


def test_coherent_panorama_views_selects_registered_yaw_cohort(tmp_path):
    for pano in range(4):
        for view in range(8):
            (tmp_path / f"p{pano:04d}_v{view:02d}_y+000.0_p+00.0.png").touch()
    selected = coherent_panorama_views(
        tmp_path,
        ["p0000_v03_y+135.0_p+00.0.png", "p0001_v04_y+180.0_p+00.0.png",
         "p0002_v04_y+180.0_p+00.0.png"],
        max_images=16,
    )
    assert selected
    assert all(any(f"_v{view:02d}_" in path.name for view in (3, 4, 5)) for path in selected)

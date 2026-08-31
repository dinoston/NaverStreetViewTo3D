import json

from street3d.fast import _latest_capture_session


def test_latest_capture_session_ignores_older_unrelated_batch(tmp_path):
    frames = [tmp_path / f"s{i:04d}.png" for i in range(5)]
    manifest = [
        {"image": "s0000.png", "source": "스크린샷 2026-08-28 145626.png"},
        {"image": "s0001.png", "source": "스크린샷 2026-08-28 145638.png"},
        {"image": "s0002.png", "source": "스크린샷 2026-08-31 082129.png"},
        {"image": "s0003.png", "source": "스크린샷 2026-08-31 082136.png"},
        {"image": "s0004.png", "source": "스크린샷 2026-08-31 082140.png"},
    ]
    manifest_path = tmp_path / "frames.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    selected, ignored = _latest_capture_session(frames, manifest_path)

    assert [path.name for path in selected] == ["s0002.png", "s0003.png", "s0004.png"]
    assert ignored == ["s0000.png", "s0001.png"]

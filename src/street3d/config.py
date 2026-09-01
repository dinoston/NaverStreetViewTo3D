from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class PipelineConfig:
    input_dir: str = "input/panoramas"
    screenshot_dir: str = "input/screenshots"
    target_annotation_dir: str = "input/target"
    output_dir: str = "output"
    mask_vegetation: bool = True
    mask_model: str = "nvidia/segformer-b0-finetuned-ade-512-512"
    mask_labels: list[str] = field(default_factory=lambda: ["tree", "grass", "plant", "flower"])
    mask_dilation: int = 7
    face_size: int = 1600
    fov: float = 90.0
    yaw_step: int = 45
    pitches: list[float] = field(default_factory=lambda: [0.0])
    include_zenith: bool = False
    jpeg_quality: int = 95
    camera_model: str = "PINHOLE"
    matcher: str = "sequential"
    overlap: int = 12
    use_gpu: bool = True
    three_dgs_repo: str = "external/gaussian-splatting"
    sugar_repo: str = "external/SuGaR"
    iterations: int = 30000
    vggt_repo: str = "external/vggt"
    vggt_model: str = "facebook/VGGT-1B"
    fast_max_images: int = 16
    fast_confidence_percentile: float = 55.0
    fast_pixel_stride: int = 2
    splat_iterations: int = 7000

    @classmethod
    def load(cls, path: Path) -> "PipelineConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(**raw)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

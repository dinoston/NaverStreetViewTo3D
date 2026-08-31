"""Convert a 3DGS checkpoint PLY into a conventional XYZRGB point cloud."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--min-opacity", type=float, default=0.15)
    parser.add_argument("--max-scale-percentile", type=float, default=97.0)
    args = parser.parse_args()

    vertex = PlyData.read(args.source)["vertex"].data
    opacity = sigmoid(np.asarray(vertex["opacity"], dtype=np.float32))
    scales = np.column_stack(
        [np.exp(np.asarray(vertex[f"scale_{i}"], dtype=np.float32)) for i in range(3)]
    )
    max_scale = scales.max(axis=1)
    scale_limit = np.percentile(max_scale, args.max_scale_percentile)
    keep = (opacity >= args.min_opacity) & (max_scale <= scale_limit)

    sh_dc = np.column_stack(
        [np.asarray(vertex[f"f_dc_{i}"], dtype=np.float32) for i in range(3)]
    )
    rgb = np.clip((0.5 + 0.28209479177387814 * sh_dc) * 255.0, 0, 255).astype(np.uint8)
    count = int(keep.sum())
    result = np.empty(
        count,
        dtype=[
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
            ("opacity", "f4"), ("scale", "f4"),
        ],
    )
    for name in ("x", "y", "z"):
        result[name] = np.asarray(vertex[name], dtype=np.float32)[keep]
    result["red"], result["green"], result["blue"] = rgb[keep].T
    result["opacity"] = opacity[keep]
    result["scale"] = max_scale[keep]

    args.destination.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(result, "vertex")], text=False).write(args.destination)
    print(f"Exported {count:,} / {len(vertex):,} points to {args.destination}")
    print(f"Opacity >= {args.min_opacity}; max scale <= {scale_limit:.6g}")


if __name__ == "__main__":
    main()

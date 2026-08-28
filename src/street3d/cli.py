from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import PipelineConfig
from .pipeline import Pipeline


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="street3d", description="Panorama to 3DGS/mesh reconstruction pipeline")
    root.add_argument("--project", type=Path, default=Path.cwd(), help="Project directory")
    root.add_argument("--config", type=Path, help="Config JSON (default: <project>/config.json)")
    sub = root.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    sub.add_parser("doctor")
    for name in ("preprocess", "align", "train", "all"):
        item = sub.add_parser(name)
        item.add_argument("--force", action="store_true")
        if name == "all":
            item.add_argument("--stop-after", choices=("preprocess", "align", "train", "mesh"))
    sub.add_parser("mesh")
    return root


def main() -> None:
    args = parser().parse_args()
    project = args.project.resolve()
    config_path = args.config.resolve() if args.config else project / "config.json"
    if args.command == "init":
        project.mkdir(parents=True, exist_ok=True)
        (project / "input" / "panoramas").mkdir(parents=True, exist_ok=True)
        (project / "input" / "screenshots").mkdir(parents=True, exist_ok=True)
        (project / "external").mkdir(exist_ok=True)
        if not config_path.exists():
            PipelineConfig().save(config_path)
        print(f"Created project: {project}")
        print(f"Put 2:1 panoramas in: {project / 'input' / 'panoramas'}")
        print(f"Put ordinary screenshots in: {project / 'input' / 'screenshots'}")
        return
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}. Run `street3d --project PATH init` first.")
    pipeline = Pipeline(project, PipelineConfig.load(config_path))
    try:
        if args.command == "doctor":
            print(json.dumps(pipeline.doctor(), ensure_ascii=False, indent=2))
        elif args.command == "preprocess":
            pipeline.preprocess(args.force)
        elif args.command == "align":
            pipeline.align(args.force)
        elif args.command == "train":
            pipeline.train(args.force)
        elif args.command == "mesh":
            pipeline.mesh()
        elif args.command == "all":
            pipeline.all(args.force, args.stop_after)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()

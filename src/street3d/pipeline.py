from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

from .config import PipelineConfig
from .fast import clean_point_cloud_and_make_mesh, reconstruct_vggt
from .masking import mask_vegetation
from .panorama import decompose, prepare_screenshots, validate_panoramas, validate_screenshots
from .util import executable, images_in, run, write_status


class Pipeline:
    def __init__(self, project: Path, config: PipelineConfig):
        self.project = project.resolve()
        self.config = config
        self.input_dir = (self.project / config.input_dir).resolve()
        self.screenshot_dir = (self.project / config.screenshot_dir).resolve()
        self.output = (self.project / config.output_dir).resolve()
        self.frames = self.output / "frames"
        self.masks = self.output / "masks"
        self.colmap = self.output / "colmap"
        self.logs = self.output / "logs"

    def doctor(self) -> dict[str, object]:
        tools = {name: executable(name) for name in ("git", "ffmpeg", "nvidia-smi")}
        tools["python"] = sys.executable
        bundled_ffmpeg = Path(sys.prefix) / "Library" / "bin" / "ffmpeg.exe"
        if not tools["ffmpeg"] and bundled_ffmpeg.exists():
            tools["ffmpeg"] = str(bundled_ffmpeg)
        tools["colmap"] = self._colmap_executable()
        repos = {
            "3dgs": str((self.project / self.config.three_dgs_repo).resolve()),
            "sugar": str((self.project / self.config.sugar_repo).resolve()),
            "vggt": str((self.project / self.config.vggt_repo).resolve()),
        }
        result = {
            "python_supported": sys.version_info >= (3, 10),
            "python_version": sys.version.split()[0],
            "tools": tools,
            "repos": {k: {"path": v, "exists": Path(v).exists()} for k, v in repos.items()},
            "ready_preprocess": sys.version_info >= (3, 10),
            "ready_colmap": bool(tools["colmap"]),
            "ready_train": Path(repos["3dgs"]).joinpath("train.py").exists(),
            "ready_mesh": Path(repos["sugar"]).exists(),
            "ready_fast": Path(repos["vggt"]).joinpath("vggt", "models", "vggt.py").exists(),
        }
        return result

    def preprocess(self, force: bool = False) -> None:
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        pano_checks = validate_panoramas(self.input_dir)
        screenshot_checks = validate_screenshots(self.screenshot_dir)
        valid_panos = [x for x in pano_checks if x["valid"]]
        valid_screenshots = [x for x in screenshot_checks if x["valid"]]
        if not valid_panos and not valid_screenshots:
            raise RuntimeError(
                "No usable input images. Put 2:1 panoramas in input/panoramas or ordinary captures in input/screenshots."
            )
        if self.frames.exists() and images_in(self.frames) and not force:
            print(f"[skip] frames already exist: {self.frames}")
            return
        if force and self.frames.exists():
            shutil.rmtree(self.frames)
        manifest: list[dict[str, object]] = []
        if valid_panos:
            pano_manifest = decompose(
                self.input_dir, self.frames, self.config.face_size, self.config.fov,
                self.config.yaw_step, self.config.pitches, self.config.include_zenith,
                self.config.jpeg_quality,
            )
            for item in pano_manifest:
                item["input_type"] = "equirectangular_panorama"
            manifest.extend(pano_manifest)
        if valid_screenshots:
            manifest.extend(prepare_screenshots(
                self.screenshot_dir, self.frames, self.config.jpeg_quality
            ))
        if self.config.mask_vegetation:
            if force and self.masks.exists():
                shutil.rmtree(self.masks)
            renamed = mask_vegetation(
                self.frames, self.masks, self.config.mask_model,
                self.config.mask_labels, self.config.mask_dilation,
            )
            for item in manifest:
                item["image"] = renamed.get(str(item["image"]), item["image"])
                item["vegetation_masked"] = True
        self.output.mkdir(parents=True, exist_ok=True)
        report = {"panoramas": pano_checks, "screenshots": screenshot_checks}
        (self.output / "input_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        (self.output / "frames.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        write_status(self.output, "preprocess", "complete", panoramas=len(valid_panos),
                     screenshots=len(valid_screenshots), frames=len(manifest))
        print(f"[ok] {len(valid_panos)} panoramas + {len(valid_screenshots)} screenshots -> {len(manifest)} images")

    def align(self, force: bool = False) -> None:
        colmap_bin = self._colmap_executable()
        if not colmap_bin:
            raise RuntimeError("COLMAP is not installed or not in PATH. Run `street3d doctor`.")
        sparse = self.colmap / "sparse"
        if sparse.exists() and any(sparse.rglob("images.bin")) and not force:
            print(f"[skip] COLMAP model already exists: {sparse}")
            return
        database = self.colmap / "database.db"
        sparse.mkdir(parents=True, exist_ok=True)
        if force and database.exists():
            database.unlink()
        gpu = "1" if self.config.use_gpu else "0"
        has_screenshots = bool(images_in(self.screenshot_dir))
        single_camera = "0" if has_screenshots else "1"
        feature_command = [colmap_bin, "feature_extractor", "--database_path", database,
                           "--image_path", self.frames, "--ImageReader.camera_model", self.config.camera_model,
                           "--ImageReader.single_camera", single_camera, "--SiftExtraction.use_gpu", gpu]
        if self.config.mask_vegetation and self.masks.exists():
            feature_command += ["--ImageReader.mask_path", self.masks]
        run(feature_command, self.logs / "colmap.log")
        matcher = "sequential_matcher" if self.config.matcher == "sequential" else "exhaustive_matcher"
        command = [colmap_bin, matcher, "--database_path", database, "--SiftMatching.use_gpu", gpu]
        if matcher == "sequential_matcher":
            command += ["--SequentialMatching.overlap", str(self.config.overlap),
                        "--SequentialMatching.loop_detection", "0"]
        run(command, self.logs / "colmap.log")
        run([colmap_bin, "mapper", "--database_path", database, "--image_path", self.frames,
             "--output_path", sparse], self.logs / "colmap.log")
        models = sorted(p for p in sparse.iterdir() if p.is_dir() and (p / "images.bin").exists())
        if not models:
            write_status(self.output, "align", "failed", reason="no_sparse_model")
            raise RuntimeError("COLMAP could not register a model. Check image overlap and logs.")
        largest = max(models, key=lambda p: (p / "images.bin").stat().st_size)
        if largest.name != "0":
            target = sparse / "0"
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(largest, target)
        write_status(self.output, "align", "complete", model=str(sparse / "0"))

    def train(self, force: bool = False) -> None:
        repo = (self.project / self.config.three_dgs_repo).resolve()
        train_py = repo / "train.py"
        if not train_py.exists():
            raise RuntimeError(f"3DGS repository missing: {train_py}")
        model = self.output / "3dgs"
        point_cloud = model / "point_cloud" / f"iteration_{self.config.iterations}" / "point_cloud.ply"
        if point_cloud.exists() and not force:
            print(f"[skip] trained model already exists: {point_cloud}")
            return
        source = self.output / "dataset"
        source.mkdir(parents=True, exist_ok=True)
        images_link = source / "images"
        sparse_link = source / "sparse"
        self._link_or_copy(self.frames, images_link)
        self._link_or_copy(self.colmap / "sparse", sparse_link)
        run([sys.executable, train_py, "-s", source, "-m", model, "--iterations", str(self.config.iterations)],
            self.logs / "3dgs.log", cwd=repo)
        write_status(self.output, "train", "complete", model=str(model))

    def mesh(self) -> None:
        repo = (self.project / self.config.sugar_repo).resolve()
        script = repo / "train_full_pipeline.py"
        if not script.exists():
            raise RuntimeError(f"SuGaR pipeline missing: {script}")
        scene = self.output / "dataset"
        checkpoint = self.output / "3dgs"
        run([sys.executable, script, "-s", scene, "-r", "density", "--high_poly", "True",
             "--export_obj", "True", "--gs_output_dir", checkpoint,
             "--iteration_to_load", str(self.config.iterations),
             "--refinement_time", "short"],
            self.logs / "sugar.log", cwd=repo)
        write_status(self.output, "mesh", "complete")

    def fast(self, force: bool = False) -> None:
        destination = self.output / "pointcloud" / "fast_building_points.ply"
        if destination.exists() and not force:
            print(f"[skip] fast point cloud already exists: {destination}")
            return
        if not images_in(self.frames):
            self.preprocess(force=False)
        result = reconstruct_vggt(
            self.frames, self.output, (self.project / self.config.vggt_repo).resolve(),
            self.config.vggt_model, self.config.fast_max_images,
            self.config.fast_confidence_percentile, self.config.fast_pixel_stride,
            self.config.mask_model,
            self.output / "frames.json",
        )
        mesh_result = clean_point_cloud_and_make_mesh(result, self.output / "mesh")
        write_status(
            self.output, "fast", "complete", point_cloud=str(result),
            clean_point_cloud=str(mesh_result[0]) if mesh_result else None,
            preview_mesh=str(mesh_result[1]) if mesh_result else None,
            proxy_mesh=str(mesh_result[2]) if mesh_result else None,
        )

    def all(self, force: bool = False, stop_after: str | None = None) -> None:
        stages = [("preprocess", self.preprocess), ("align", self.align), ("train", self.train), ("mesh", self.mesh)]
        for name, method in stages:
            print(f"\n=== {name} ===")
            method(force=force) if name != "mesh" else method()
            if stop_after == name:
                break

    @staticmethod
    def _link_or_copy(source: Path, target: Path) -> None:
        if target.exists() or target.is_symlink():
            return
        try:
            os.symlink(source, target, target_is_directory=True)
        except OSError:
            shutil.copytree(source, target)

    @staticmethod
    def _colmap_executable() -> str | None:
        candidates = (
            Path(sys.prefix) / "Library" / "bin" / "colmap.exe",
            Path(sys.prefix) / "Library" / "COLMAP.bat",
        )
        return executable("colmap") or executable("COLMAP.bat") or next(
            (str(path) for path in candidates if path.exists()), None
        )

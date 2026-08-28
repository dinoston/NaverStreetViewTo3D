from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Iterable


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def images_in(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def executable(name: str) -> str | None:
    return shutil.which(name)


def run(command: Iterable[str], log_path: Path, cwd: Path | None = None) -> None:
    command = [str(x) for x in command]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {' '.join(command)}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        code = process.wait()
    if code:
        raise RuntimeError(f"Command failed ({code}). See {log_path}")


def write_status(root: Path, stage: str, state: str, **details: object) -> None:
    path = root / "status.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"stages": {}}
    data["stages"][stage] = {"state": state, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), **details}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


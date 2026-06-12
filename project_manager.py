from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any


def timestamp_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S_") + f"{int((time.time() % 1) * 1000):03d}"


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _file_checksum(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new("md5" if algorithm == "md5" else "sha256")
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def benchmark_write_speed(
    target_dir: Path,
    size_mb: float = 512.0,
    chunk_mb: float = 8.0,
    sample_seconds: float = 3.0,
) -> dict[str, Any]:
    target_dir.mkdir(parents=True, exist_ok=True)
    size_bytes = max(int(size_mb * 1024 * 1024), 8 * 1024 * 1024)
    chunk_size = max(int(chunk_mb * 1024 * 1024), 1024 * 1024)
    deadline_s = max(float(sample_seconds), 0.5)
    test_path = target_dir / f".mvss_write_benchmark_{os.getpid()}_{int(time.time() * 1000)}.bin"
    chunk = bytes(1024 * 1024)
    written = 0
    start = time.perf_counter()
    try:
        with test_path.open("wb", buffering=0) as fh:
            while written < size_bytes:
                for _ in range(max(chunk_size // len(chunk), 1)):
                    if written >= size_bytes:
                        break
                    fh.write(chunk)
                    written += len(chunk)
                if time.perf_counter() - start >= deadline_s and written >= 32 * 1024 * 1024:
                    break
            try:
                fh.flush()
                os.fsync(fh.fileno())
            except OSError:
                pass
    finally:
        elapsed = max(time.perf_counter() - start, 1e-9)
        try:
            test_path.unlink(missing_ok=True)
        except OSError:
            pass
    return {
        "target_dir": str(target_dir),
        "bytes_written": written,
        "elapsed_seconds": elapsed,
        "write_mbps": written / elapsed / 1024 / 1024,
        "sample_size_mb": written / 1024 / 1024,
    }


def _relative_or_str(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


class ProjectManager:
    def __init__(self, root_dir: Path, config: dict[str, Any]):
        self.root_dir = root_dir
        settings = config.setdefault("project", {})
        if not isinstance(settings, dict):
            settings = {}
            config["project"] = settings
        self.enabled = bool(settings.get("enabled", True))
        self.projects_subdir = str(settings.get("projects_subdir", "projects"))
        self.current_project_id = str(settings.get("current_project_id", "") or "")
        self.current_project_name = str(settings.get("current_project_name", "") or "")

    @property
    def projects_root(self) -> Path:
        return self.root_dir / self.projects_subdir

    @property
    def active_project_dir(self) -> Path:
        if not self.enabled:
            return self.root_dir
        if not self.current_project_id:
            self.create_project()
        return self.projects_root / self.current_project_id

    def create_project(self, name: str | None = None) -> Path:
        project_id = timestamp_id()
        self.current_project_id = project_id
        self.current_project_name = name or f"Project {project_id}"
        project_dir = self.projects_root / project_id
        (project_dir / "photos").mkdir(parents=True, exist_ok=True)
        (project_dir / "videos").mkdir(parents=True, exist_ok=True)
        (project_dir / "exports").mkdir(parents=True, exist_ok=True)
        self._write_project_json(project_dir, [])
        return project_dir

    def sync_config(self, config: dict[str, Any]) -> None:
        settings = config.setdefault("project", {})
        if not isinstance(settings, dict):
            settings = {}
            config["project"] = settings
        settings["enabled"] = self.enabled
        settings["projects_subdir"] = self.projects_subdir
        settings["current_project_id"] = self.current_project_id
        settings["current_project_name"] = self.current_project_name

    def output_root_for_mode(self, mode: str) -> Path:
        if not self.enabled:
            return self.root_dir / mode
        return self.active_project_dir / mode

    def register_session(
        self,
        mode: str,
        path: Path,
        metadata_path: Path | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        project_dir = self.active_project_dir
        project_json = self._project_json_path(project_dir)
        data = self._load_project_json(project_dir)
        sessions = data.setdefault("sessions", [])
        item = {
            "session_id": path.name,
            "mode": mode,
            "path": _relative_or_str(path, project_dir),
            "metadata_path": _relative_or_str(metadata_path, project_dir) if metadata_path else None,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        if extra:
            item.update(extra)
        sessions.append(item)
        data["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        with project_json.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2, default=_json_default)
            fh.write("\n")

    def project_meta(self) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        return {
            "project_id": self.current_project_id,
            "project_name": self.current_project_name,
            "project_dir": str(self.active_project_dir),
            "project_json": str(self._project_json_path(self.active_project_dir)),
        }

    def _project_json_path(self, project_dir: Path) -> Path:
        return project_dir / "project.json"

    def _load_project_json(self, project_dir: Path) -> dict[str, Any]:
        path = self._project_json_path(project_dir)
        if not path.exists():
            self._write_project_json(project_dir, [])
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}

    def _write_project_json(self, project_dir: Path, sessions: list[dict[str, Any]]) -> None:
        data = {
            "project_id": self.current_project_id,
            "project_name": self.current_project_name,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "sessions": sessions,
        }
        with self._project_json_path(project_dir).open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
            fh.write("\n")


def write_data_manifest(
    session_dir: Path,
    capture_summary: dict[str, Any],
    camera_settings: dict[str, Any],
    environment: dict[str, Any],
    algorithm: str = "sha256",
) -> dict[str, Any]:
    exports_dir = session_dir / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = exports_dir / "file_manifest.csv"
    summary_path = exports_dir / "capture_summary.json"
    rows: list[dict[str, Any]] = []
    for path in sorted(session_dir.rglob("*")):
        if not path.is_file():
            continue
        if path == csv_path or path == summary_path:
            continue
        stat = path.stat()
        rows.append(
            {
                "path": str(path.relative_to(session_dir)),
                "size_bytes": stat.st_size,
                "checksum_algorithm": "md5" if algorithm == "md5" else "sha256",
                "checksum": _file_checksum(path, algorithm),
                "modified_time": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(stat.st_mtime)),
            }
        )
    with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["path", "size_bytes", "checksum_algorithm", "checksum", "modified_time"],
        )
        writer.writeheader()
        writer.writerows(rows)
    summary_payload = {
        "session_dir": str(session_dir),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "capture_summary": capture_summary,
        "camera_settings": camera_settings,
        "environment": environment,
        "file_count": len(rows),
        "total_size_bytes": sum(int(row["size_bytes"]) for row in rows),
        "manifest_csv": str(csv_path),
    }
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary_payload, fh, ensure_ascii=False, indent=2, default=_json_default)
        fh.write("\n")
    return {
        "manifest_csv": str(csv_path),
        "summary_json": str(summary_path),
        "file_count": len(rows),
        "total_size_bytes": summary_payload["total_size_bytes"],
    }


def disk_free_gb(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    return usage.free / 1024**3

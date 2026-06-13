from __future__ import annotations

import hashlib
import csv
import html
import json
import logging
import os
from logging.handlers import RotatingFileHandler
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import ctypes
import traceback
from dataclasses import asdict, dataclass
from collections import deque
from pathlib import Path
from queue import Empty, Full, Queue
from tkinter import BOTH, BOTTOM, DISABLED, LEFT, NORMAL, RIGHT, TOP, X, BooleanVar, Canvas, Frame, StringVar, Tk, Toplevel, filedialog, messagebox
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from calibration_manager import StereoCalibration, load_stereo_calibration
from config_utils import config_bool, config_float, config_int
from image_quality import (
    DEFAULT_FOCUS_ROI,
    calibration_board_coverage,
    clamp_roi_frac,
    epipolar_alignment,
    exposure_metrics,
    focus_pair_metrics,
    make_anaglyph,
    make_focus_peaking_overlay,
    roi_from_pixels,
    speckle_quality,
)
from project_manager import ProjectManager, benchmark_write_speed, write_data_manifest

_MVS_IMPORT_ERROR: BaseException | None = None
try:
    from mvs_camera import Frame as CameraFrame
    from mvs_camera import FrameSyncError, FrameTimeoutError, MvsError, StereoCameraSystem, enumerate_cameras
except Exception as exc:
    _MVS_IMPORT_ERROR = exc

    @dataclass(frozen=True)
    class CameraFrame:  # type: ignore[no-redef]
        image: Image.Image
        frame_number: int
        width: int
        height: int
        host_timestamp: int
        camera_timestamp: int
        raw_data: bytes | None = None
        raw_frame_len: int = 0
        pixel_type: int = 0
        pixel_type_name: str = ""
        raw_bit_depth: int = 8
        raw_array_shape: tuple[int, ...] | None = None

    class MvsError(RuntimeError):  # type: ignore[no-redef]
        pass

    class FrameTimeoutError(MvsError):  # type: ignore[no-redef]
        pass

    class FrameSyncError(MvsError):  # type: ignore[no-redef]
        pass

    def _mvs_import_failure_message() -> str:
        return (
            "Cannot load mvs_camera. Install Hikrobot MVS manually and ensure the MvImport "
            "and MVS runtime directories are available in PYTHONPATH/PATH. "
            f"Original error: {type(_MVS_IMPORT_ERROR).__name__}: {_MVS_IMPORT_ERROR}"
        )

    class StereoCameraSystem:  # type: ignore[no-redef]
        def __init__(self, *_args, **_kwargs):
            raise MvsError(_mvs_import_failure_message())

    def enumerate_cameras() -> tuple[list[object], object]:  # type: ignore[no-redef]
        raise MvsError(_mvs_import_failure_message())

try:
    import winsound
except Exception:  # pragma: no cover - non-Windows fallback
    winsound = None


def _runtime_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _resource_dir() -> Path:
    bundle_dir = getattr(sys, "_MEIPASS", None)
    return Path(bundle_dir).resolve() if bundle_dir else Path(__file__).resolve().parent


BASE_DIR = _runtime_dir()
RESOURCE_DIR = _resource_dir()
CONFIG_PATH = BASE_DIR / "config.json"
DEFAULT_CONFIG_PATH = RESOURCE_DIR / "config.json"
LOG_DIR = BASE_DIR / "logs"
BIAODING_DIR = BASE_DIR.parent / "MVSS_Biaoding_CalibrationOnly"
CAPTURE_WIDTH = 5472
CAPTURE_HEIGHT = 3648
APP_VERSION = "V2.0_20260612"

BG_COLOR = "#0e1116"
SURFACE_COLOR = "#141a21"
PANEL_COLOR = "#1b232c"
PANEL_ELEVATED_COLOR = "#222c35"
CANVAS_COLOR = "#05070a"
CHART_COLOR = "#0c1117"
BORDER_COLOR = "#34414c"
BORDER_STRONG_COLOR = "#4d5e6b"
ACCENT_COLOR = "#12a8e8"
ACCENT_ACTIVE_COLOR = "#35c0ff"
SUCCESS_COLOR = "#31c784"
WARNING_COLOR = "#ffd166"
DANGER_COLOR = "#ff6b6b"
TEXT_COLOR = "#f2f6f9"
MUTED_TEXT_COLOR = "#aab6c0"
SUBTLE_TEXT_COLOR = "#71808d"
FONT_FAMILY = "Microsoft YaHei UI"
MONO_FONT_FAMILY = "Cascadia Mono"
BASE_FONT_SIZE = 10
TITLE_FONT_SIZE = 12
APP_TITLE_FONT_SIZE = 15
INFO_FONT_SIZE = 9
OVERLAY_FONT_SIZE = 10
WINDOW_ASPECT_RATIO = 16 / 10
TARGET_WINDOW_WIDTH = 1920
TARGET_WINDOW_HEIGHT = 1200
HIGH_RES_WINDOW_WIDTH = 2560
HIGH_RES_WINDOW_HEIGHT = 1600
MIN_WINDOW_WIDTH = 1440
MIN_WINDOW_HEIGHT = 900
CAMERA_ASPECT_RATIO = CAPTURE_WIDTH / CAPTURE_HEIGHT
CAMERA_GAP = 18
CAMERA_VERTICAL_PADDING = 0
PARAM_SIDEBAR_WIDTH = 400
QUALITY_MONITOR_MIN_HEIGHT = 210
QUALITY_MONITOR_HIGH_RES_MIN_HEIGHT = 260

FramePair = tuple[CameraFrame | None, CameraFrame | None]
LOGGER = logging.getLogger("mvss_capture")
UI_QUEUE_EVENT = "<<MvssUiQueue>>"
DEFAULT_HOST_TIMESTAMP_DELTA_NS = 10_000_000
_CONFIG_MISSING = object()
TRIGGER_SOURCE_CN = {
    "Software": "软触发",
    "Continuous": "连续采集",
    "Cascade": "硬触发级联（无功能）",
    "Line0": "外部硬触发（无功能）",
}
TRIGGER_SOURCE_FROM_CN = {v: k for k, v in TRIGGER_SOURCE_CN.items()}
TRIGGER_SOURCE_CN_ORDER = ("软触发", "连续采集", "硬触发级联（无功能）", "外部硬触发（无功能）")
ENABLED_TRIGGER_SOURCES = {"Software", "Continuous"}
DISABLED_TRIGGER_FALLBACK = "Software"
TRIGGER_CONFIG_SAFE_DEFAULTS = {
    "require_hardware_trigger": False,
    "hardware_sync_enabled": False,
    "hardware_sync_master": "left",
    "hardware_sync_master_line": "Line2",
    "hardware_sync_master_line_source": "ExposureActive",
    "hardware_sync_slave_line": "Line0",
    "hardware_sync_slave_activation": "RisingEdge",
    "hardware_sync_master_trigger_source": "Software",
}
TRIGGER_CONFIG_KEYS = {"trigger_source", *TRIGGER_CONFIG_SAFE_DEFAULTS.keys()}

def canonical_trigger_source(value: object) -> str:
    text = str(value or "").strip()
    return TRIGGER_SOURCE_FROM_CN.get(text, text)

def safe_capture_trigger_source(value: object) -> str:
    source = canonical_trigger_source(value)
    return source if source in ENABLED_TRIGGER_SOURCES else DISABLED_TRIGGER_FALLBACK

def safe_trigger_config(values: dict[str, object]) -> dict[str, object]:
    normalized = dict(values)
    normalized["trigger_source"] = safe_capture_trigger_source(normalized.get("trigger_source", DISABLED_TRIGGER_FALLBACK))
    normalized.update(TRIGGER_CONFIG_SAFE_DEFAULTS)
    return normalized

def display_trigger_source(value: object) -> str:
    text = canonical_trigger_source(value)
    return TRIGGER_SOURCE_CN.get(text, text)

DEFAULT_PREVIEW_FPS = 15.0
DEFAULT_RECORD_QUEUE_SECONDS = 10.0
RAW_FRAME_FORMATS = {"npy", "png16", "tiff16", "exr"}
DIC_CAPTURE_MODE = "dic_capture"
DIC_PIXEL_FORMATS = ("Mono16", "Mono12", "Mono10", "Mono8")
DIC_CAPTURE_CONFIG = {
    "trigger_source": "Continuous",
    "trigger_activation": "RisingEdge",
    "require_hardware_trigger": False,
    "hardware_sync_enabled": False,
    "hardware_sync_master": "left",
    "hardware_sync_master_line": "Line2",
    "hardware_sync_master_line_source": "ExposureActive",
    "hardware_sync_slave_line": "Line0",
    "hardware_sync_slave_activation": "RisingEdge",
    "hardware_sync_master_trigger_source": "Software",
    "pixel_format": "Mono8",
    "image_format": "png",
    "record_jpeg_quality": 100,
    "exposure_auto": "Off",
    "exposure_time_us": 20000.0,
    "gain_auto": "Off",
    "gain": 0.0,
    "roi_width": CAPTURE_WIDTH,
    "roi_height": CAPTURE_HEIGHT,
    "roi_offset_x": 0,
    "roi_offset_y": 0,
    "left_roi_width": CAPTURE_WIDTH,
    "left_roi_height": CAPTURE_HEIGHT,
    "left_roi_offset_x": 0,
    "left_roi_offset_y": 0,
    "right_roi_width": CAPTURE_WIDTH,
    "right_roi_height": CAPTURE_HEIGHT,
    "right_roi_offset_x": 0,
    "right_roi_offset_y": 0,
    "record_fps": 5.0,
    "interval_capture_seconds": 0.5,
    "interval_capture_count": None,
    "record_save_image_sequence": True,
    "record_realtime_mp4": True,
    "auto_make_mp4": False,
    "preview_fps": 5.0,
    "record_queue_max_items": 32,
    "record_queue_force_configured": True,
    "chunk_data_enabled": True,
    "timestamp_reject_enabled": False,
    "record_capture_priority_mode": False,
    "record_preview_during_capture": True,
    "record_preview_fps": 2.0,
    "preview_quality_analysis_enabled": False,
    "record_force_image_format": False,
    "save_raw_frames": True,
    "raw_frame_format": "tiff16",
    "viewable_sidecar_enabled": True,
    "viewable_sidecar_format": "png",
    "capture_quality_gate": {
        "enabled": False,
        "strict_mode": False,
        "checks": {
            "focus": True,
            "focus_consistency": True,
            "overexposure_max_pct": 5.0,
            "underexposure_max_pct": 5.0,
            "brightness_min": 40,
            "brightness_max": 220,
        },
    },
}


def default_presets() -> dict[str, dict[str, object]]:
    common_roi = {
        "roi_width": CAPTURE_WIDTH,
        "roi_height": CAPTURE_HEIGHT,
        "roi_offset_x": 0,
        "roi_offset_y": 0,
        "left_roi_width": CAPTURE_WIDTH,
        "left_roi_height": CAPTURE_HEIGHT,
        "left_roi_offset_x": 0,
        "left_roi_offset_y": 0,
        "right_roi_width": CAPTURE_WIDTH,
        "right_roi_height": CAPTURE_HEIGHT,
        "right_roi_offset_x": 0,
        "right_roi_offset_y": 0,
    }
    scientific_defaults = {
        "black_level": None,
        "digital_shift": None,
        "gamma": None,
        "save_raw_frames": True,
        "raw_frame_format": "tiff16",
        "image_format": "png",
        "record_force_image_format": False,
    }
    return {
        "室内低光": {
            **common_roi,
            **scientific_defaults,
            "trigger_source": "Continuous",
            "exposure_auto": "Off",
            "exposure_time_us": 20000.0,
            "auto_exposure_lower_limit": 1000.0,
            "auto_exposure_upper_limit": 100000.0,
            "gain_auto": "Off",
            "gain": 0.0,
            "auto_gain_lower_limit": 0.0,
            "auto_gain_upper_limit": 15.0,
            "balance_white_auto": "Off",
            "balance_ratio_red": None,
            "balance_ratio_green": None,
            "balance_ratio_blue": None,
            "pixel_format": "Mono8",
            "chunk_data_enabled": True,

(Showing lines 305-364 of 9682. Use offset=365 to continue.)            "pixel_format": "Mono8",
            "chunk_data_enabled": True,

(Showing lines 305-364 of 9682. Use offset=365 to continue.)            "pixel_format": "Mono8",
            "image_format": "png",
            "image_format": "png",
            "record_force_image_format": False,
            "save_raw_frames": True,
            "raw_frame_format": "tiff16",
            "record_save_image_sequence": True,
            "record_realtime_mp4": True,
            "auto_make_mp4": False,
            "exposure_auto": "Off",
            "exposure_time_us": 20000.0,
            "gain_auto": "Off",
            "gain": 0.0,
            "chunk_data_enabled": True,
            "chunk_selectors": ["Timestamp", "FrameCounter", "ExposureTime", "Gain"],
            "timestamp_reject_enabled": False,
        },
        "标定采集": {
            **common_roi,
            **scientific_defaults,
            "trigger_source": "Continuous",
            "pixel_format": "Mono8",
            "image_format": "png",
            "record_force_image_format": True,
            "save_raw_frames": False,
            "raw_frame_format": "npy",
            "exposure_auto": "Off",
            "exposure_time_us": 8000.0,
            "gain_auto": "Off",
            "gain": 0.0,
            "chunk_data_enabled": True,
            "chunk_selectors": ["Timestamp", "FrameCounter", "ExposureTime", "Gain"],
        },
    }


def dic_capture_defaults() -> dict[str, object]:
    values = dict(DIC_CAPTURE_CONFIG)
    gate = dict(DIC_CAPTURE_CONFIG["capture_quality_gate"])
    gate["checks"] = dict(gate["checks"])
    values["capture_quality_gate"] = gate
    return values


class UiEventQueue(Queue[tuple[str, object]]):
    def __init__(self, notify_callback):
        super().__init__()
        self._notify_callback = notify_callback

    def put(self, item, block=True, timeout=None):  # type: ignore[override]
        super().put(item, block=block, timeout=timeout)
        self._notify_callback()


class ThreadSafeConfig(dict):
    def __init__(self, *args, lock: threading.RLock | None = None, **kwargs):
        object.__setattr__(self, "_lock", lock or threading.RLock())
        super().__init__()
        self.update(*args, **kwargs)

    def _wrap(self, value):
        if isinstance(value, ThreadSafeConfig):
            return value
        if isinstance(value, dict):
            return ThreadSafeConfig(value, lock=self._lock)
        if isinstance(value, list):
            return [self._wrap(item) for item in value]
        return value

    def __getitem__(self, key):
        with self._lock:
            return super().__getitem__(key)

    def __setitem__(self, key, value) -> None:
        with self._lock:
            super().__setitem__(key, self._wrap(value))

    def __delitem__(self, key) -> None:
        with self._lock:
            super().__delitem__(key)

    def __contains__(self, key) -> bool:
        with self._lock:
            return super().__contains__(key)

    def __iter__(self):
        with self._lock:
            return iter(list(dict.keys(self)))

    def __len__(self) -> int:
        with self._lock:
            return super().__len__()

    def get(self, key, default=None):
        with self._lock:
            return super().get(key, default)

    def pop(self, key, default=_CONFIG_MISSING):
        with self._lock:
            if default is _CONFIG_MISSING:
                return super().pop(key)
            return super().pop(key, default)

    def clear(self) -> None:
        with self._lock:
            super().clear()

    def copy(self) -> dict:
        return self.snapshot()

    def keys(self):
        with self._lock:
            return list(dict.keys(self))

    def values(self):
        with self._lock:
            return [self._snapshot_value(value) for value in dict.values(self)]

    def items(self):
        with self._lock:
            return [(key, self._snapshot_value(value)) for key, value in dict.items(self)]

    def setdefault(self, key, default=None):
        with self._lock:
            if key not in self:
                super().__setitem__(key, self._wrap(default))
            value = super().__getitem__(key)
            wrapped = self._wrap(value)
            if wrapped is not value:
                super().__setitem__(key, wrapped)
            return wrapped

    def update(self, *args, **kwargs) -> None:
        with self._lock:
            data = dict(*args, **kwargs)
            for key, value in data.items():
                super().__setitem__(key, self._wrap(value))

    def snapshot(self) -> dict:
        with self._lock:
            return {key: self._snapshot_value(value) for key, value in dict.items(self)}

    def _snapshot_value(self, value):
        if isinstance(value, ThreadSafeConfig):
            return value.snapshot()
        if isinstance(value, dict):
            return {key: self._snapshot_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._snapshot_value(item) for item in value]
        return value


class RecordMetaWriter:
    def __init__(self, path: Path, flush_every: int = 32):
        self.path = path
        self.flush_every = max(flush_every, 1)
        self._lock = threading.Lock()
        self._fh = path.open("w", encoding="utf-8")
        self._fh.write("[\n")
        self._count = 0
        self._closed = False
        self._frames: list[dict] = []

    def append(self, frame_meta: dict) -> None:
        try:
            text = json.dumps(frame_meta, ensure_ascii=False, default=json_metadata_default)
            normalized = json.loads(text)
        except (TypeError, ValueError) as exc:
            LOGGER.exception("record frame metadata is not JSON serializable")
            raise RuntimeError(f"record frame metadata is not JSON serializable: {exc}") from exc
        with self._lock:
            if self._closed:
                raise RuntimeError("record frame metadata writer is already closed")
            if self._count:
                self._fh.write(",\n")
            self._fh.write(text)
            self._frames.append(normalized)
            self._count += 1
            if self._count % self.flush_every == 0:
                self._fh.flush()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._fh.write("\n]\n")
            self._fh.close()
            self._closed = True

    def load(self) -> list[dict]:
        with self._lock:
            return [dict(frame) for frame in self._frames]


def json_metadata_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def load_config() -> dict:
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except FileNotFoundError:
        try:
            with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except FileNotFoundError:
            LOGGER.warning("config.json not found at %s; starting with defaults.", CONFIG_PATH)
            payload = {}
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            LOGGER.warning("Default config.json at %s could not be loaded: %s", DEFAULT_CONFIG_PATH, exc)
            payload = {}
        if payload:
            try:
                save_config(payload)
            except OSError:
                LOGGER.warning("Could not create writable config.json at %s.", CONFIG_PATH, exc_info=True)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        backup_path: Path | None = None
        if CONFIG_PATH.exists():
            backup_path = CONFIG_PATH.with_name(f"{CONFIG_PATH.name}.bad.{timestamp_ms()}")
            try:
                shutil.copy2(CONFIG_PATH, backup_path)
            except OSError:
                backup_path = None
        backup_note = f" Backup: {backup_path}" if backup_path is not None else ""
        LOGGER.exception("Failed to load config.json; starting with defaults.%s", backup_note)
        messagebox.showwarning(
            "配置已重置",
            f"config.json 无法读取，程序将使用默认配置启动。\n原因: {exc}{backup_note}",
        )
        payload = {}
    if not isinstance(payload, dict):
        LOGGER.warning("config.json root is %s; starting with defaults.", type(payload).__name__)
        payload = {}
    payload = safe_trigger_config(payload)
    if isinstance(payload.get("presets"), dict):
        payload["presets"] = {
            name: safe_trigger_config(preset) if isinstance(preset, dict) else preset
            for name, preset in payload["presets"].items()
        }
    if isinstance(payload.get("dic_capture"), dict):
        payload["dic_capture"] = safe_trigger_config(payload["dic_capture"])
    return ThreadSafeConfig(payload)


def save_config(config: dict) -> None:
    payload = config.snapshot() if isinstance(config, ThreadSafeConfig) else dict(config)
    with CONFIG_PATH.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def timestamp_ms() -> str:
    return time.strftime("%Y%m%d_%H%M%S_") + f"{int((time.time() % 1) * 1000):03d}"


def safe_filename(text: object) -> str:
    value = str(text).strip()
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    return cleaned.strip("._") or "item"


def optional_float_text(text: str) -> float | None:
    value = text.strip()
    if not value:
        return None
    return float(value)


def optional_int_text(text: str) -> int | None:
    value = text.strip()
    if not value:
        return None
    return int(value)


def optional_positive_fps(text: object) -> float | None:
    value = str(text).strip()
    if not value:
        return None
    fps = float(value)
    return fps if fps > 0 else None


def configured_preview_fps(config: dict) -> float:
    return optional_positive_fps(config.get("preview_fps", DEFAULT_PREVIEW_FPS)) or DEFAULT_PREVIEW_FPS


def configured_record_queue_size(config: dict, fps: float | None = None) -> int:
    target_fps = fps if fps is not None and fps > 0 else configured_record_output_fps(config)
    required = int(round(max(target_fps, 0.1) * DEFAULT_RECORD_QUEUE_SECONDS))
    configured = config_int(config, "record_queue_max_items", required)
    if config_bool(config, "record_queue_force_configured", False, False):
        return max(configured, 1)
    return max(configured, required, 1)


def configured_record_outputs(config: dict, save_image_sequence: bool) -> dict[str, object]:
    post_make_mp4 = config_bool(config, "auto_make_mp4", True, True)
    realtime_mp4_enabled = config_bool(config, "record_realtime_mp4", not save_image_sequence, not save_image_sequence)
    forced_realtime = False
    if not save_image_sequence and not (post_make_mp4 or realtime_mp4_enabled):
        realtime_mp4_enabled = True
        forced_realtime = True
    elif not save_image_sequence and not realtime_mp4_enabled:
        realtime_mp4_enabled = True
        forced_realtime = True
    make_mp4_after = post_make_mp4 and save_image_sequence and not realtime_mp4_enabled
    use_realtime_mp4 = realtime_mp4_enabled
    return {
        "post_make_mp4": post_make_mp4,
        "record_realtime_mp4": realtime_mp4_enabled,
        "make_mp4_after": make_mp4_after,
        "use_realtime_mp4": use_realtime_mp4,
        "forced_realtime": forced_realtime,
        "mp4_generation": "ffmpeg_after_recording"
        if make_mp4_after
        else "opencv_realtime"
        if use_realtime_mp4
        else "disabled",
    }


def optional_config_text(config: dict, key: str, default: str = "") -> str:
    value = config.get(key, default)
    return "" if value is None else str(value)


def image_extension(config: dict) -> str:
    fmt = str(config.get("image_format", "bmp")).lower().strip()
    if fmt in {"jpg", "jpeg"}:
        return "jpg"
    if fmt == "png":
        return "png"
    return "bmp"


def raw_frame_format(config: dict) -> str:
    fmt = str(config.get("raw_frame_format", "npy")).lower().strip().replace("-", "")
    aliases = {
        "png": "png16",
        "16png": "png16",
        "png16bit": "png16",
        "tif": "tiff16",
        "tiff": "tiff16",
        "16tiff": "tiff16",
        "tiff16bit": "tiff16",
        "openexr": "exr",
    }
    fmt = aliases.get(fmt, fmt)
    return fmt if fmt in RAW_FRAME_FORMATS else "npy"


def raw_frame_extension(config: dict) -> str:
    fmt = raw_frame_format(config)
    if fmt == "png16":
        return "png"
    if fmt == "tiff16":
        return "tiff"
    return fmt


def config_uses_raw_frame_storage(config: dict) -> bool:
    if config_bool(config, "save_raw_frames", False, False):
        return True
    if config_bool(config, "record_force_image_format", False, False):
        return False
    pixel_format = str(config.get("pixel_format", "Mono8")).lower()
    return "bayer" in pixel_format or any(token in pixel_format for token in ("mono10", "mono12", "mono16"))


def resolve_output_root(config: dict) -> Path:
    configured = Path(str(config.get("save_dir", "captures")))
    return configured if configured.is_absolute() else BASE_DIR / configured


def estimate_frame_bytes(config: dict, width: int = CAPTURE_WIDTH, height: int = CAPTURE_HEIGHT) -> int:
    if config_uses_raw_frame_storage(config):
        pixel_format = str(config.get("pixel_format", "Mono8")).lower()
        channels = 3 if "rgb" in pixel_format or "bgr" in pixel_format else 1
        bit_depth = 16 if any(token in pixel_format for token in ("10", "12", "16", "bayer")) else 8
        raw_bytes = width * height * channels * (2 if bit_depth > 8 else 1)
        fmt = raw_frame_format(config)
        if fmt == "npy":
            return raw_bytes + 256
        if fmt in {"png16", "tiff16"}:
            return max(int(raw_bytes * 0.75), 1)
        return raw_bytes
    pixel_format = str(config.get("pixel_format", "Mono8")).lower()
    channels = 3 if "rgb" in pixel_format or "bgr" in pixel_format else 1
    raw_bytes = width * height * channels
    ext = image_extension(config)
    if ext == "jpg":
        quality = max(min(int(config.get("record_jpeg_quality", 95) or 95), 100), 1)
        ratio = 0.08 + 0.20 * (quality / 100.0)
        return max(int(raw_bytes * ratio), 1)
    if ext == "png":
        return max(int(raw_bytes * 0.60), 1)
    return raw_bytes


def configured_record_output_fps(config: dict) -> float:
    return optional_positive_fps(config.get("record_fps", 5.0)) or max(
        config_float(config, "record_output_fps_when_unlimited", 30.0),
        0.1,
    )


def record_preview_due(now: float, next_preview_time: float, preview_fps: float) -> tuple[bool, float]:
    preview_interval = 1.0 / max(float(preview_fps), 0.1)
    if now < next_preview_time:
        return False, next_preview_time
    while next_preview_time <= now:
        next_preview_time += preview_interval
    return True, next_preview_time


def effective_record_intervals(config: dict, capture_fps: float | None) -> tuple[float, float]:
    interval = 1.0 / capture_fps if capture_fps is not None and capture_fps > 0 else 0.0
    return interval, interval


def format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def image_estimated_bytes(image: Image.Image) -> int:
    channels = 1 if image.mode in {"1", "L", "P"} else len(image.getbands())
    return image.width * image.height * max(channels, 1)


def frame_raw_estimated_bytes(frame: CameraFrame | None) -> int:
    if frame is None:
        return 0
    raw_len = int(getattr(frame, "raw_frame_len", 0) or 0)
    if raw_len > 0:
        return raw_len
    if getattr(frame, "image", None) is None:
        return int(getattr(frame, "width", 0) or 0) * int(getattr(frame, "height", 0) or 0)
    return image_estimated_bytes(frame.image)


def contiguous_frame_buffer(data: bytes | bytearray | memoryview, count: int | None = None) -> bytes | bytearray | memoryview:
    if count is None and isinstance(data, (bytes, bytearray)):
        return data
    view = memoryview(data)
    if not view.c_contiguous:
        raw = view.tobytes()
        return raw[:count] if count is not None else raw
    if view.format != "B" or view.ndim != 1:
        view = view.cast("B")
    if count is not None:
        view = view[:count]
    if view.c_contiguous and view.ndim == 1:
        if count is not None and isinstance(data, (bytes, bytearray)) and count == len(data):
            return data
        return view
    return view.tobytes()


def format_bytes(value: object) -> str:
    try:
        size = float(value)
    except (TypeError, ValueError):
        return "--"
    units = ("B", "KB", "MB", "GB", "TB")
    for unit in units:
        if abs(size) < 1024.0 or unit == units[-1]:
            digits = 0 if unit == "B" else 2
            return f"{size:.{digits}f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"


def setup_logging(config: dict) -> None:
    if LOGGER.handlers:
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "capture.log"
    LOGGER.setLevel(logging.INFO)
    handler = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    LOGGER.addHandler(handler)
    if not config_bool(config, "logging_enabled", True, True):
        LOGGER.disabled = True


def enable_windows_dpi_awareness() -> None:
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def configure_tk_dpi_scaling(root: Tk) -> None:
    try:
        dpi = max(float(root.winfo_fpixels("1i")), 96.0)
        root.tk.call("tk", "scaling", dpi / 72.0)
    except Exception:
        pass


def apply_responsive_window_geometry(root: Tk) -> None:
    try:
        screen_width = max(int(root.winfo_screenwidth()), 1)
        screen_height = max(int(root.winfo_screenheight()), 1)
        use_high_res = screen_width >= 2400 and screen_height >= 1500
        desired_width = HIGH_RES_WINDOW_WIDTH if use_high_res else TARGET_WINDOW_WIDTH
        desired_height = HIGH_RES_WINDOW_HEIGHT if use_high_res else TARGET_WINDOW_HEIGHT
        side_margin = 80 if use_high_res else 32
        bottom_margin = 112 if use_high_res else 88
        available_width = max(1280, screen_width - side_margin)
        available_height = max(820, screen_height - bottom_margin)
        window_width = min(desired_width, available_width)
        window_height = min(desired_height, available_height)
        if window_width / max(window_height, 1) > WINDOW_ASPECT_RATIO:
            window_width = max(1280, int(window_height * WINDOW_ASPECT_RATIO))
        else:
            window_height = max(800, int(window_width / WINDOW_ASPECT_RATIO))
        x = max((screen_width - window_width) // 2, 0)
        y = max((screen_height - bottom_margin - window_height) // 2, 0)
        min_width = min(MIN_WINDOW_WIDTH, max(1280, screen_width - side_margin))
        min_height = min(MIN_WINDOW_HEIGHT, max(820, screen_height - bottom_margin))
        root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        root.minsize(min_width, min_height)
    except Exception:
        root.geometry(f"{TARGET_WINDOW_WIDTH}x{TARGET_WINDOW_HEIGHT - 88}")
        root.minsize(MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT)


def maximize_window(root: Tk) -> None:
    try:
        root.state("zoomed")
    except Exception:
        try:
            screen_width = max(int(root.winfo_screenwidth()), TARGET_WINDOW_WIDTH)
            screen_height = max(int(root.winfo_screenheight()), TARGET_WINDOW_HEIGHT)
            root.geometry(f"{screen_width}x{screen_height}+0+0")
        except Exception:
            pass


def camera_row_height(root: Tk) -> int:
    try:
        width = int(root.winfo_width())
        if width <= 1:
            width = int(root.winfo_screenwidth())
    except Exception:
        width = TARGET_WINDOW_WIDTH
    camera_width = max(width - PARAM_SIDEBAR_WIDTH - CAMERA_GAP * 2, 2)
    pane_width = max(camera_width / 2.0, 1.0)
    return int(pane_width / CAMERA_ASPECT_RATIO) + CAMERA_VERTICAL_PADDING * 2


class AspectRatioFrame(Frame):
    def __init__(self, master: Tk | Frame, aspect_ratio: float, **kwargs):
        super().__init__(master, **kwargs)
        self.aspect_ratio = max(float(aspect_ratio), 0.1)
        self._child: Frame | None = None
        self.bind("<Configure>", self._layout_child)

    def set_child(self, child: Frame) -> None:
        self._child = child
        child.place(in_=self, x=0, y=0)
        self._layout_child()

    def _layout_child(self, _event=None) -> None:
        if self._child is None:
            return
        width = max(self.winfo_width(), 1)
        height = max(self.winfo_height(), 1)
        child_width = min(width, int(height * self.aspect_ratio))
        child_height = min(height, int(child_width / self.aspect_ratio))
        x = max((width - child_width) // 2, 0)
        y = max((height - child_height) // 2, 0)
        self._child.place_configure(x=x, y=y, width=child_width, height=child_height)


class DualCameraStrip(Frame):
    def __init__(self, master: Tk | Frame, aspect_ratio: float, gap: int, vertical_padding: int, **kwargs):
        super().__init__(master, **kwargs)
        self.aspect_ratio = max(float(aspect_ratio), 0.1)
        self.gap = max(int(gap), 0)
        self.vertical_padding = max(int(vertical_padding), 0)
        self._left: Frame | None = None
        self._right: Frame | None = None
        self._sidebar: Frame | None = None
        self.bind("<Configure>", self._layout_children)

    def set_children(self, left: Frame, right: Frame) -> None:
        self._left = left
        self._right = right
        left.place(in_=self, x=0, y=0)
        right.place(in_=self, x=0, y=0)
        self._layout_children()

    def set_sidebar(self, sidebar: Frame) -> None:
        self._sidebar = sidebar
        sidebar.place(in_=self, x=0, y=0)
        self._layout_children()

    def _layout_children(self, _event=None) -> None:
        if self._left is None or self._right is None:
            return
        width = max(self.winfo_width(), 2 + self.gap)
        sidebar_width = 0
        camera_gap_count = 1
        if self._sidebar is not None:
            sidebar_width = PARAM_SIDEBAR_WIDTH
            camera_gap_count = 2
        camera_width = max(width - sidebar_width - self.gap * camera_gap_count, 2)
        left_width = max(camera_width // 2, 1)
        right_width = max(camera_width - left_width, 1)
        pane_height = max(int(min(left_width, right_width) / self.aspect_ratio), 1)
        sidebar_height = pane_height if self._sidebar is not None else 0
        strip_height = pane_height + self.vertical_padding * 2
        if abs(self.winfo_reqheight() - strip_height) > 1:
            self.configure(height=strip_height)
        y = self.vertical_padding
        self._left.place_configure(x=0, y=y, width=left_width, height=pane_height)
        self._right.place_configure(x=left_width + self.gap, y=y, width=right_width, height=pane_height)
        if self._sidebar is not None:
            sidebar_x = left_width + self.gap + right_width + self.gap
            self._sidebar.place_configure(x=sidebar_x, y=y, width=sidebar_width, height=sidebar_height)


class ZoomImagePane(Frame):
    def __init__(self, master: Tk | Frame, title: str, reset_command=None, roi_callback=None, side: str = ""):
        super().__init__(master, bg=BORDER_COLOR)
        self.title_var = StringVar(value=title)
        self.side = side
        self.info_var = StringVar(value="未连接")
        self.zoom_var = StringVar(value="100%")
        self._image_ref: ImageTk.PhotoImage | None = None
        self._last_image: Image.Image | None = None
        self._render_bounds: tuple[float, float, float, float] | None = None
        self._roi_start: tuple[int, int] | None = None
        self._roi_rect_id: int | None = None
        self._flash_id: int | None = None
        self._flash_after_id: str | None = None
        self._recording_active = False
        self._recording_after_id: str | None = None
        self._recording_dot_id: int | None = None
        self._recording_text_id: int | None = None
        self._performance_bg_id: int | None = None
        self._performance_text_id: int | None = None
        self._performance_status = "good"
        self._performance_text = "FPS -- | Drop -- | Delta --"
        self.performance_var = StringVar(value=self._performance_text)
        self._focus_peaking_enabled = False
        self._focus_peaking_overlay: Image.Image | None = None
        self._zebra_enabled = False
        self._zebra_period_seconds = 0.5
        self._zebra_mask_cache: dict[tuple[int, int], np.ndarray] = {}
        self._guide_mode = "off"
        self._focus_roi_frac: dict[str, float] | None = None
        self._focus_roi_rect_id: int | None = None
        self._magnifier_rect_frac: dict[str, float] | None = None
        self._magnifier_rect_id: int | None = None
        self._external_bindings: list[tuple[str, str]] = []
        self.roi_callback = roi_callback
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.roi_mode = False
        self._pan_start: tuple[int, int] | None = None
        self._pan_origin: tuple[float, float] = (0.0, 0.0)

        container = Frame(self, bg=CANVAS_COLOR, bd=0)
        container.pack(fill=BOTH, expand=True, padx=1, pady=1)

        header = ttk.Frame(container, style="PaneHeader.TFrame")
        header.pack(side=TOP, fill=X)
        ttk.Label(header, textvariable=self.title_var, style="PaneTitle.TLabel", padding=(12, 5), anchor="w").pack(
            side=LEFT, fill=X, expand=True
        )
        ttk.Button(header, text="还原", command=reset_command or self.reset_zoom, style="Pane.TButton", width=7).pack(
            side=RIGHT, padx=(4, 6), pady=3
        )
        ttk.Label(header, textvariable=self.zoom_var, style="PaneMeta.TLabel", padding=(7, 5), anchor="e").pack(
            side=RIGHT
        )

        footer = ttk.Frame(container, style="PaneHeader.TFrame")
        footer.pack(side=BOTTOM, fill=X)
        ttk.Label(footer, textvariable=self.info_var, style="PaneInfo.TLabel", padding=(10, 4), anchor="w").pack(
            side=LEFT, fill=X, expand=True
        )
        ttk.Label(
            footer,
            textvariable=self.performance_var,
            style="Performance.TLabel",
            padding=(10, 4),
            anchor="e",
        ).pack(side=RIGHT)

        self.canvas = Canvas(container, bg=CANVAS_COLOR, highlightthickness=0, bd=0)
        self.canvas.pack(side=TOP, fill=BOTH, expand=True)
        self._canvas_image_id: int | None = None
        self._canvas_text_id = self.canvas.create_text(
            0,
            0,
            text="无图像",
            fill=SUBTLE_TEXT_COLOR,
            font=(FONT_FAMILY, 18),
            anchor="center",
        )

        self.canvas.bind("<Configure>", self._on_configure)
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Button-4>", self._on_button4)
        self.canvas.bind("<Button-5>", self._on_button5)
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_press)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_release)
        self.canvas.bind("<Double-Button-1>", self._on_double_click)
        self._update_cursor()

    def _on_configure(self, _event=None) -> None:
        self._render()

    def _on_button4(self, _event=None) -> None:
        self.set_zoom(self.zoom * 1.1)

    def _on_button5(self, _event=None) -> None:
        self.set_zoom(self.zoom / 1.1)

    def _on_double_click(self, _event=None) -> None:
        self.reset_zoom()

    def set_title(self, text: str) -> None:
        self.title_var.set(text)

    def set_frame(self, frame: CameraFrame) -> None:
        self._last_image = frame.image if getattr(frame, "image", None) is not None else None
        self.info_var.set(
            f"{frame.width}x{frame.height}  Frame:{frame.frame_number}  CamTS:{frame.camera_timestamp}"
        )
        self._render()

    def set_display_image(self, image: Image.Image | None, info: str = "") -> None:
        self._last_image = image
        self.info_var.set(info or (f"{image.width}x{image.height}" if image is not None else "无信号"))
        self._render()

    def set_analysis_overlays(
        self,
        *,
        focus_peaking_enabled: bool | None = None,
        focus_peaking_overlay: Image.Image | None = None,
        zebra_enabled: bool | None = None,
        guide_mode: str | None = None,
        focus_roi_frac: dict[str, float] | None = None,
        magnifier_rect_frac: dict[str, float] | None = None,
    ) -> None:
        if focus_peaking_enabled is not None:
            self._focus_peaking_enabled = focus_peaking_enabled
        if focus_peaking_overlay is not None or not self._focus_peaking_enabled:
            self._focus_peaking_overlay = focus_peaking_overlay
        if zebra_enabled is not None:
            self._zebra_enabled = zebra_enabled
        if guide_mode is not None:
            self._guide_mode = guide_mode
        if focus_roi_frac is not None:
            self._focus_roi_frac = clamp_roi_frac(focus_roi_frac)
        self._magnifier_rect_frac = clamp_roi_frac(magnifier_rect_frac) if magnifier_rect_frac is not None else None

    def set_no_signal(self, reason: str = "无信号") -> None:
        self._last_image = None
        self.info_var.set(reason)
        if self._canvas_image_id is not None:
            self.canvas.delete(self._canvas_image_id)
            self._canvas_image_id = None
        self._image_ref = None
        self._render()

    def set_zoom(self, value: float) -> None:
        self.zoom = min(max(value, 0.2), 8.0)
        self.zoom_var.set(f"{self.zoom * 100:.0f}%")
        self._render()

    def reset_zoom(self) -> None:
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.set_zoom(1.0)

    def set_roi_mode(self, enabled: bool) -> None:
        self.roi_mode = enabled
        self._pan_start = None
        self._roi_start = None
        if not enabled:
            self.clear_roi_rectangle()
        self._update_cursor()

    def clear_roi_rectangle(self) -> None:
        if self._roi_rect_id is not None:
            self.canvas.delete(self._roi_rect_id)
            self._roi_rect_id = None

    def _update_cursor(self) -> None:
        self.canvas.configure(cursor="crosshair" if self.roi_mode else "fleur")

    def _render(self) -> None:
        width = max(self.canvas.winfo_width(), 100)
        height = max(self.canvas.winfo_height(), 100)
        if self._last_image is None:
            self._render_bounds = None
            self.canvas.delete("guide")
            if self._focus_roi_rect_id is not None:
                self.canvas.delete(self._focus_roi_rect_id)
                self._focus_roi_rect_id = None
            if self._magnifier_rect_id is not None:
                self.canvas.delete(self._magnifier_rect_id)
                self._magnifier_rect_id = None
            self.canvas.coords(self._canvas_text_id, width // 2, height // 2)
            self.canvas.itemconfigure(self._canvas_text_id, text="无信号", state="normal")
            self._place_performance_overlay()
            return

        target_width = max(1, int(width * self.zoom))
        target_height = max(1, int(height * self.zoom))
        source_width, source_height = self._last_image.size
        scale = min(target_width / max(source_width, 1), target_height / max(source_height, 1), 1.0)
        display_size = (max(1, int(source_width * scale)), max(1, int(source_height * scale)))
        image = self._last_image.resize(display_size, Image.Resampling.BILINEAR)
        image = image.convert("RGB")
        if self._focus_peaking_enabled and self._focus_peaking_overlay is not None:
            overlay = self._focus_peaking_overlay
            if overlay.size != image.size:
                overlay = overlay.resize(image.size, Image.Resampling.BILINEAR)
            image = Image.alpha_composite(image.convert("RGBA"), overlay.convert("RGBA")).convert("RGB")
        if self._zebra_enabled and self._zebra_visible():
            image = self._apply_zebra_overlay(image)
        image_width, image_height = image.size
        self._image_ref = ImageTk.PhotoImage(image)
        x = width // 2 + int(round(self.pan_x))
        y = height // 2 + int(round(self.pan_y))
        self._render_bounds = (x - image_width / 2, y - image_height / 2, image_width, image_height)
        if self._canvas_image_id is None:
            self._canvas_image_id = self.canvas.create_image(x, y, image=self._image_ref, anchor="center")
        else:
            self.canvas.itemconfigure(self._canvas_image_id, image=self._image_ref)
            self.canvas.coords(self._canvas_image_id, x, y)
        self.canvas.itemconfigure(self._canvas_text_id, state="hidden")
        self._draw_guides()
        self._draw_fraction_rects()
        self._place_performance_overlay()
        self._raise_overlays()

    def _apply_zebra_overlay(self, image: Image.Image) -> Image.Image:
        gray = np.asarray(image.convert("L"), dtype=np.uint8)
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
        height, width = gray.shape
        stripes = self._zebra_stripe_mask(width, height)
        over = (gray >= 254) & stripes
        under = (gray <= 2) & stripes
        rgb[over] = rgb[over] * 0.50 + np.array([255.0, 30.0, 30.0]) * 0.50
        rgb[under] = rgb[under] * 0.50 + np.array([35.0, 120.0, 255.0]) * 0.50
        return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")

    def _zebra_stripe_mask(self, width: int, height: int) -> np.ndarray:
        key = (width, height)
        mask = self._zebra_mask_cache.get(key)
        if mask is not None:
            return mask
        x_pattern = (np.arange(width, dtype=np.uint16) // 8).reshape(1, width)
        y_pattern = (np.arange(height, dtype=np.uint16) // 8).reshape(height, 1)
        mask = ((x_pattern + y_pattern) % 2 == 0)
        self._zebra_mask_cache[key] = mask
        if len(self._zebra_mask_cache) > 4:
            oldest_key = next(iter(self._zebra_mask_cache))
            if oldest_key != key:
                self._zebra_mask_cache.pop(oldest_key, None)
        return mask

    def _zebra_visible(self) -> bool:
        phase = time.monotonic() % self._zebra_period_seconds
        return phase < self._zebra_period_seconds / 2.0

    def _draw_guides(self) -> None:
        self.canvas.delete("guide")
        if self._last_image is None or self._render_bounds is None or self._guide_mode == "off":
            return
        left, top, width, height = self._render_bounds
        center_x = left + width / 2
        center_y = top + height / 2
        color = SUCCESS_COLOR
        if self._guide_mode in {"center", "full"}:
            self.canvas.create_line(
                left, center_y, left + width, center_y, fill=color, width=1, tags=("guide",), stipple="gray50"
            )
            self.canvas.create_line(
                center_x, top, center_x, top + height, fill=color, width=1, tags=("guide",), stipple="gray50"
            )
        if self._guide_mode in {"grid", "full"}:
            for frac in (1 / 3, 2 / 3):
                y = top + height * frac
                x = left + width * frac
                self.canvas.create_line(
                    left,
                    y,
                    left + width,
                    y,
                    fill=color,
                    width=1,
                    dash=(6, 5),
                    tags=("guide",),
                    stipple="gray50",
                )
                self.canvas.create_line(
                    x,
                    top,
                    x,
                    top + height,
                    fill=color,
                    width=1,
                    dash=(6, 5),
                    tags=("guide",),
                    stipple="gray50",
                )
        self._raise_overlays()

    def _draw_fraction_rects(self) -> None:
        if self._focus_roi_rect_id is not None:
            self.canvas.delete(self._focus_roi_rect_id)
            self._focus_roi_rect_id = None
        if self._magnifier_rect_id is not None:
            self.canvas.delete(self._magnifier_rect_id)
            self._magnifier_rect_id = None
        if self._last_image is None or self._render_bounds is None:
            return
        if self._focus_roi_frac is not None:
            self._focus_roi_rect_id = self._create_fraction_rect(
                self._focus_roi_frac, WARNING_COLOR, (5, 4), "focus_roi"
            )
        if self._magnifier_rect_frac is not None:
            self._magnifier_rect_id = self._create_fraction_rect(
                self._magnifier_rect_frac, ACCENT_ACTIVE_COLOR, (3, 3), "magnifier"
            )

    def _create_fraction_rect(self, roi: dict[str, float], color: str, dash: tuple[int, int], tag: str) -> int | None:
        if self._render_bounds is None:
            return None
        left, top, width, height = self._render_bounds
        x0 = left + roi["x_frac"] * width
        y0 = top + roi["y_frac"] * height
        x1 = x0 + roi["w_frac"] * width
        y1 = y0 + roi["h_frac"] * height
        return self.canvas.create_rectangle(x0, y0, x1, y1, outline=color, width=2, dash=dash, tags=(tag,))

    def _on_mouse_wheel(self, event) -> None:
        scale = 1.1 if event.delta > 0 else 1 / 1.1
        self.set_zoom(self.zoom * scale)

    def flash_shutter(self) -> None:
        width = max(self.canvas.winfo_width(), 1)
        height = max(self.canvas.winfo_height(), 1)
        if self._flash_id is not None:
            self.canvas.delete(self._flash_id)
        self._flash_id = self.canvas.create_rectangle(
            0,
            0,
            width,
            height,
            fill=TEXT_COLOR,
            outline="",
            stipple="gray50",
            tags=("flash",),
        )
        self.canvas.lift(self._flash_id)
        if self._flash_after_id is not None:
            try:
                self.canvas.after_cancel(self._flash_after_id)
            except Exception:
                pass
        self._flash_after_id = self.canvas.after(100, self._clear_flash)

    def _clear_flash(self) -> None:
        self._flash_after_id = None
        if self._flash_id is not None:
            self.canvas.delete(self._flash_id)
            self._flash_id = None

    def cancel_pending_callbacks(self) -> None:
        flash_after_id = self._flash_after_id
        self._flash_after_id = None
        if flash_after_id is not None:
            try:
                self.canvas.after_cancel(flash_after_id)
            except Exception:
                pass
        self._cancel_recording_blink()

    def set_recording(self, active: bool) -> None:
        if self._recording_active == active:
            return
        self._recording_active = active
        if active:
            self._cancel_recording_blink()
            self._blink_recording_overlay()
        else:
            self._cancel_recording_blink()
            self._show_recording_overlay(False)

    def set_performance_overlay(self, text: str, status: str) -> None:
        self._performance_text = text
        self._performance_status = status
        self.performance_var.set(text)
        if self._performance_text_id is not None:
            color = {"good": SUCCESS_COLOR, "warn": WARNING_COLOR, "bad": DANGER_COLOR}.get(status, SUCCESS_COLOR)
            self.canvas.itemconfigure(self._performance_text_id, text=text, fill=color)
        self._place_performance_overlay()

    def _blink_recording_overlay(self) -> None:
        if not self._recording_active:
            self._show_recording_overlay(False)
            return
        self._recording_after_id = None
        visible = False
        if self._recording_dot_id is not None:
            visible = self.canvas.itemcget(self._recording_dot_id, "state") != "hidden"
        self._show_recording_overlay(not visible)
        if self._recording_active:
            self._recording_after_id = self.canvas.after(500, self._blink_recording_overlay)

    def _cancel_recording_blink(self) -> None:
        after_id = self._recording_after_id
        self._recording_after_id = None
        if after_id is not None:
            try:
                self.canvas.after_cancel(after_id)
            except Exception:
                pass

    def _show_recording_overlay(self, visible: bool) -> None:
        state = "normal" if visible else "hidden"
        if self._recording_dot_id is None:
            self._recording_dot_id = self.canvas.create_oval(
                14,
                14,
                30,
                30,
                fill=DANGER_COLOR,
                outline="#ffb3b3",
                width=2,
                tags=("recording",),
            )
            self._recording_text_id = self.canvas.create_text(
                38,
                22,
                text="录像中",
                fill=TEXT_COLOR,
                font=(FONT_FAMILY, 12, "bold"),
                anchor="w",
                tags=("recording",),
            )
        self.canvas.itemconfigure(self._recording_dot_id, state=state)
        if self._recording_text_id is not None:
            self.canvas.itemconfigure(self._recording_text_id, state=state)
        self._raise_overlays()

    def _place_performance_overlay(self) -> None:
        self.performance_var.set(self._performance_text)
        for item_id in (self._performance_bg_id, self._performance_text_id):
            if item_id is not None:
                try:
                    self.canvas.delete(item_id)
                except Exception:
                    pass
        self._performance_bg_id = None
        self._performance_text_id = None
        canvas_width = max(self.canvas.winfo_width(), 1)
        canvas_height = max(self.canvas.winfo_height(), 1)
        x = 10
        y = max(canvas_height - 48, 10)
        text_width = max(180, min(canvas_width - 20, 620))
        text_height = 38 if "\n" in self._performance_text else 20
        color = {"good": SUCCESS_COLOR, "warn": WARNING_COLOR, "bad": DANGER_COLOR}.get(
            self._performance_status, SUCCESS_COLOR
        )
        if self._performance_bg_id is None:
            self._performance_bg_id = self.canvas.create_rectangle(
                x,
                y,
                x + text_width,
                y + text_height,
                fill=CANVAS_COLOR,
                outline=BORDER_COLOR,
                stipple="gray25",
                tags=("performance",),
            )
            self._performance_text_id = self.canvas.create_text(
                x + 7,
                y + 5,
                text=self._performance_text,
                fill=color,
                font=(MONO_FONT_FAMILY, max(8, OVERLAY_FONT_SIZE - 2), "bold"),
                anchor="nw",
                tags=("performance",),
            )
        else:
            self.canvas.coords(self._performance_bg_id, x, y, x + text_width, y + text_height)
            if self._performance_text_id is not None:
                self.canvas.coords(self._performance_text_id, x + 7, y + 5)
        self._raise_overlays()

    def _on_mouse_press(self, event) -> None:
        if self.roi_mode:
            self._on_roi_press(event)
            return
        if self._last_image is None:
            self._pan_start = None
            return
        self._pan_start = (event.x, event.y)
        self._pan_origin = (self.pan_x, self.pan_y)

    def _on_mouse_drag(self, event) -> None:
        if self.roi_mode:
            self._on_roi_drag(event)
            return
        if self._pan_start is None:
            return
        x0, y0 = self._pan_start
        origin_x, origin_y = self._pan_origin
        self.pan_x = origin_x + event.x - x0
        self.pan_y = origin_y + event.y - y0
        self._render()

    def _on_mouse_release(self, event) -> None:
        if self.roi_mode:
            self._on_roi_release(event)
            return
        self._pan_start = None

    def _on_roi_press(self, event) -> None:
        if self._last_image is None or self.roi_callback is None or not self._point_inside_rendered_image(event.x, event.y):
            self._roi_start = None
            return
        self._roi_start = (event.x, event.y)
        if self._roi_rect_id is not None:
            self.canvas.delete(self._roi_rect_id)
        self._roi_rect_id = self.canvas.create_rectangle(
            event.x,
            event.y,
            event.x,
            event.y,
            outline=ACCENT_ACTIVE_COLOR,
            width=2,
            dash=(5, 3),
            tags=("roi",),
        )

    def _on_roi_drag(self, event) -> None:
        if self._roi_start is None or self._roi_rect_id is None:
            return
        x0, y0 = self._roi_start
        x1, y1 = self._clamp_to_rendered_image(event.x, event.y)
        self.canvas.coords(self._roi_rect_id, x0, y0, x1, y1)

    def _on_roi_release(self, event) -> None:
        if self._roi_start is None or self._roi_rect_id is None:
            return
        x0, y0 = self._roi_start
        x1, y1 = self._clamp_to_rendered_image(event.x, event.y)
        self._roi_start = None
        if abs(x1 - x0) < 6 or abs(y1 - y0) < 6:
            self.canvas.delete(self._roi_rect_id)
            self._roi_rect_id = None
            return
        self.canvas.coords(self._roi_rect_id, x0, y0, x1, y1)
        roi = self._canvas_rect_to_image_roi(x0, y0, x1, y1)
        if roi is not None:
            self.roi_callback(roi, self.side)

    def _point_inside_rendered_image(self, x: float, y: float) -> bool:
        if self._render_bounds is None:
            return False
        left, top, width, height = self._render_bounds
        return left <= x <= left + width and top <= y <= top + height

    def bind_external(self, sequence: str, callback) -> None:
        bind_id = self.canvas.bind(sequence, callback, add="+")
        if bind_id:
            self._external_bindings.append((sequence, bind_id))

    def unbind_external_callbacks(self) -> None:
        self.cancel_pending_callbacks()
        for sequence, bind_id in self._external_bindings:
            try:
                self.canvas.unbind(sequence, bind_id)
            except Exception:
                pass
        self._external_bindings.clear()

    def _clamp_to_rendered_image(self, x: float, y: float) -> tuple[float, float]:
        if self._render_bounds is None:
            return x, y
        left, top, width, height = self._render_bounds
        return min(max(x, left), left + width), min(max(y, top), top + height)

    def _canvas_rect_to_image_roi(self, x0: float, y0: float, x1: float, y1: float) -> tuple[int, int, int, int] | None:
        if self._last_image is None or self._render_bounds is None:
            return None
        left, top, display_width, display_height = self._render_bounds
        if display_width <= 0 or display_height <= 0:
            return None
        x0, y0 = self._clamp_to_rendered_image(x0, y0)
        x1, y1 = self._clamp_to_rendered_image(x1, y1)
        min_x, max_x = sorted((x0, x1))
        min_y, max_y = sorted((y0, y1))
        image_width, image_height = self._last_image.size
        offset_x = int(round((min_x - left) * image_width / display_width))
        offset_y = int(round((min_y - top) * image_height / display_height))
        width = int(round((max_x - min_x) * image_width / display_width))
        height = int(round((max_y - min_y) * image_height / display_height))
        offset_x = min(max(offset_x, 0), image_width - 1)
        offset_y = min(max(offset_y, 0), image_height - 1)
        width = min(max(width, 1), image_width - offset_x)
        height = min(max(height, 1), image_height - offset_y)
        return offset_x, offset_y, width, height

    def _raise_overlays(self) -> None:
        self.canvas.lift("guide")
        if self._focus_roi_rect_id is not None:
            self.canvas.lift(self._focus_roi_rect_id)
        if self._magnifier_rect_id is not None:
            self.canvas.lift(self._magnifier_rect_id)
        if self._roi_rect_id is not None:
            self.canvas.lift(self._roi_rect_id)
        if self._recording_dot_id is not None:
            self.canvas.lift(self._recording_dot_id)
        if self._recording_text_id is not None:
            self.canvas.lift(self._recording_text_id)
        if self._performance_bg_id is not None:
            self.canvas.lift(self._performance_bg_id)
        if self._performance_text_id is not None:
            self.canvas.lift(self._performance_text_id)
        if self._flash_id is not None:
            self.canvas.lift(self._flash_id)


class ToolTip:
    def __init__(self, widget: ttk.Widget, text_provider):
        self.widget = widget
        self.text_provider = text_provider
        self.tip_window: Toplevel | None = None
        self.widget.bind("<Enter>", self.show, add="+")
        self.widget.bind("<Leave>", self.hide, add="+")
        self.widget.bind("<ButtonPress>", self.hide, add="+")

    def show(self, _event=None) -> None:
        text = str(self.text_provider()).strip()
        if not text or self.tip_window is not None:
            return
        x = self.widget.winfo_rootx() + 10
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8
        self.tip_window = Toplevel(self.widget)
        self.tip_window.wm_overrideredirect(True)
        self.tip_window.wm_geometry(f"+{x}+{y}")
        label = ttk.Label(
            self.tip_window,
            text=text,
            justify=LEFT,
            style="Tooltip.TLabel",
            padding=(8, 5),
        )
        label.pack()

    def hide(self, _event=None) -> None:
        if self.tip_window is not None:
            self.tip_window.destroy()
            self.tip_window = None


class HistogramCanvas(ttk.Frame):
    def __init__(self, master: Tk | Frame, title: str):
        super().__init__(master, style="Panel.TFrame", padding=(5, 3))
        self.title = title
        self.histogram: list[float] | None = None
        ttk.Label(self, text=title, style="Panel.TLabel").pack(side=TOP, anchor="w")
        self.canvas = Canvas(
            self, width=260, height=58, bg=CHART_COLOR, highlightthickness=1, highlightbackground=BORDER_COLOR
        )
        self.canvas.pack(side=TOP, fill=BOTH, expand=True)
        self.canvas.bind("<Configure>", self._on_configure)

    def _on_configure(self, _event=None) -> None:
        self.draw()

    def set_histogram(self, histogram: list[float] | None) -> None:
        self.histogram = histogram
        self.draw()

    def redraw_when_visible(self) -> None:
        self.update_idletasks()
        self.canvas.update_idletasks()
        self.draw()
        self.canvas.after_idle(self.draw)

    def draw(self) -> None:
        self.canvas.delete("all")
        width = max(self.canvas.winfo_width(), 120)
        height = max(self.canvas.winfo_height(), 58)
        margin_left = 24
        margin_right = 8
        margin_top = 8
        margin_bottom = 18
        plot_w = max(width - margin_left - margin_right, 1)
        plot_h = max(height - margin_top - margin_bottom, 1)
        x0 = margin_left
        y0 = margin_top
        x1 = margin_left + plot_w
        y1 = margin_top + plot_h

        self.canvas.create_rectangle(x0, y0, x1, y1, fill=CANVAS_COLOR, outline=BORDER_COLOR)
        self.canvas.create_rectangle(x0, y0, x0 + plot_w * 15 / 255, y1, fill="#102438", outline="")
        self.canvas.create_rectangle(x0 + plot_w * 240 / 255, y0, x1, y1, fill="#3b1f24", outline="")
        for value, color, dash in (
            (5, ACCENT_ACTIVE_COLOR, (2, 3)),
            (128, MUTED_TEXT_COLOR, (3, 3)),
            (250, DANGER_COLOR, (2, 3)),
        ):
            x = x0 + plot_w * value / 255
            self.canvas.create_line(x, y0, x, y1, fill=color, dash=dash)
        axis_font = (MONO_FONT_FAMILY, 8)
        self.canvas.create_text(x0, y1 + 3, text="0", fill=MUTED_TEXT_COLOR, anchor="nw", font=axis_font)
        self.canvas.create_text(x0 + plot_w / 2, y1 + 3, text="128", fill=MUTED_TEXT_COLOR, anchor="n", font=axis_font)
        self.canvas.create_text(x1, y1 + 3, text="255", fill=MUTED_TEXT_COLOR, anchor="ne", font=axis_font)
        if not self.histogram:
            self.canvas.create_text(
                (x0 + x1) / 2, (y0 + y1) / 2, text="暂无数据", fill=SUBTLE_TEXT_COLOR, anchor="center"
            )
            return
        values = np.asarray(self.histogram, dtype=np.float32)
        if values.size != 256:
            return
        max_value = float(np.max(values))
        if max_value <= 0:
            return
        values = np.sqrt(values / max_value)
        bar_w = max(plot_w / 256, 1.0)
        for index, value in enumerate(values):
            bar_h = float(value) * plot_h
            x = x0 + index * plot_w / 256
            color = "#c7ecff" if 15 <= index <= 240 else "#ffaaa0" if index > 240 else "#82d8ff"
            self.canvas.create_rectangle(x, y1 - bar_h, x + bar_w, y1, fill=color, outline=color)


class StereoCaptureOnlyApp:
    @property
    def previewing(self) -> bool:
        with self._state_lock:
            return bool(getattr(self, "_previewing", False))

    @previewing.setter
    def previewing(self, value: bool) -> None:
        with self._state_lock:
            self._previewing = bool(value)

    @property
    def recording(self) -> bool:
        with self._state_lock:
            return bool(getattr(self, "_recording", False))

    @recording.setter
    def recording(self, value: bool) -> None:
        with self._state_lock:
            self._recording = bool(value)

    @property
    def interval_capturing(self) -> bool:
        with self._state_lock:
            return bool(getattr(self, "_interval_capturing", False))

    @interval_capturing.setter
    def interval_capturing(self, value: bool) -> None:
        with self._state_lock:
            self._interval_capturing = bool(value)

    def _cached_trigger_source(self) -> str:
        with self._state_lock:
            raw = str(getattr(self, "_trigger_source_cache", self.config.get("trigger_source", "Software")))
            return canonical_trigger_source(raw)

    def _set_cached_trigger_source(self, value: str) -> None:
        with self._state_lock:
            self._trigger_source_cache = canonical_trigger_source(value)

    def _config_snapshot(self) -> dict:
        if isinstance(self.config, ThreadSafeConfig):
            return safe_trigger_config(self.config.snapshot())
        return safe_trigger_config(dict(self.config))

    def _update_config(self, values: dict[str, object], *, save: bool = True) -> dict:
        if any(key in values for key in TRIGGER_CONFIG_KEYS):
            values = safe_trigger_config(values)
        if isinstance(self.config, ThreadSafeConfig):
            with self.config._lock:
                self.config.update(values)
                snapshot = self.config.snapshot()
        else:
            self.config.update(values)
            snapshot = dict(self.config)
        if save:
            save_config(snapshot)
        return snapshot

    def _start_background_thread(self, target, name: str, *, report_errors: bool = True) -> threading.Thread:
        def runner() -> None:
            try:
                target()
            except Exception as exc:
                LOGGER.exception("Background thread %s failed.", name)
                if report_errors and not self._closing:
                    try:
                        self.ui_queue.put(("error", exc))
                    except Exception:
                        pass
            finally:
                self._forget_background_thread(threading.current_thread())

        thread = threading.Thread(target=runner, name=name, daemon=True)
        with self._background_threads_lock:
            self._background_threads.append(thread)
        thread.start()
        return thread

    def _forget_background_thread(self, thread: threading.Thread) -> None:
        if not hasattr(self, "_background_threads_lock"):
            return
        with self._background_threads_lock:
            try:
                self._background_threads.remove(thread)
            except ValueError:
                pass

    def _background_threads_snapshot(self) -> list[threading.Thread]:
        if not hasattr(self, "_background_threads_lock"):
            return []
        with self._background_threads_lock:
            self._background_threads = [thread for thread in self._background_threads if thread.is_alive()]
            return list(self._background_threads)

    def __init__(self, root: Tk):
        self.root = root
        self.root.title("双目同步采集")
        self.root.configure(bg=BG_COLOR)
        apply_responsive_window_geometry(self.root)
        self.root.option_add("*tearOff", False)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<F11>", self._on_fullscreen_key)
        self.root.bind("<Escape>", self._on_escape_key)

        self.config = load_config()
        self.config.update(safe_trigger_config(self.config.snapshot() if isinstance(self.config, ThreadSafeConfig) else dict(self.config)))
        setup_logging(self.config)
        self._ensure_default_full_resolution()
        self._ensure_recording_config_defaults()
        self._ensure_quality_config_defaults()
        self._ensure_reliability_config_defaults()
        self._ensure_preset_config_defaults()
        self._ensure_project_config_defaults()
        self.project_manager = ProjectManager(resolve_output_root(self.config), self.config)
        if self.project_manager.enabled:
            self.project_manager.create_project()
            self.project_manager.sync_config(self.config)
            save_config(self.config)
        self.calibration: StereoCalibration = load_stereo_calibration(self.config, BASE_DIR)
        self.camera_system: StereoCameraSystem | None = None
        self._ui_queue_event_lock = threading.Lock()
        self._ui_queue_event_pending = False
        self._ui_queue_fallback_after_id: str | None = None
        self.ui_queue: Queue[tuple[str, object]] = UiEventQueue(self._notify_ui_queue)

        self._state_lock = threading.RLock()
        self._background_threads_lock = threading.Lock()
        self._background_threads: list[threading.Thread] = []
        self._trigger_source_cache = str(self.config.get("trigger_source", "Software"))
        self.previewing = False
        self.preview_thread: threading.Thread | None = None
        self._preview_generation = 0
        self.interval_capturing = False
        self.interval_thread: threading.Thread | None = None
        self.interval_stop_event = threading.Event()
        self.interval_count = 0
        self.photo_count = 0
        self.recording = False
        self.record_thread: threading.Thread | None = None
        self.record_dir: Path | None = None
        self._resume_preview_after_recording = False
        self._closing = False
        self._record_stats_lock = threading.RLock()
        self.record_count = 0
        self.record_saved_count = 0
        self._record_next_saved_index = 0
        self.record_started_at: float | None = None
        self.record_started_wall_time: float | None = None
        self.record_stop_reason = "manual"
        self._record_last_frame_pair: tuple[CameraFrame | None, CameraFrame | None, float] | None = None
        self._record_last_frame_lock = threading.Lock()
        self._quality_metrics_lock = threading.Lock()
        self._record_write_lag = 0.0
        self._record_write_warning = ""
        self._record_skip_every_n = 1
        self._record_skip_keep_frames = 1
        self._record_split_index = 1
        self._record_segment_start_time = 0.0
        self._record_segment_start_saved = 0
        self._record_segment_sizes: dict[int, int] = {}
        self._record_video_segment_sizes: dict[int, int] = {}
        self._record_first_trigger_time: float | None = None
        self._record_second_stats: dict[int, dict[str, object]] = {}
        self._record_skipped_frames: list[dict[str, object]] = []
        self._record_skipped_count = 0
        self._record_skip_reasons: dict[str, int] = {}
        self._record_timeout_count = 0
        self._record_consecutive_timeouts = 0
        self._record_error_count = 0
        self._record_reconnect_count = 0
        self._record_disk_warning_count = 0
        self._record_last_camera_frame_numbers: dict[str, int] = {}
        self._record_frame_number_gap_count = 0
        self._record_disk_benchmark: dict[str, object] | None = None
        self._record_preflight_plan: dict[str, object] = {}
        self._record_last_disk_check = 0.0
        self._record_disk_usage_start = 0
        self._record_summary: dict[str, object] = {}
        self._dic_recording = False
        self._last_alert_times: dict[str, float] = {}
        self._last_reconnect_message = ""
        self._reconnecting = False
        self._interval_lamp_after_id: str | None = None

        self._last_preview_status_time = 0.0
        self._stat_last_time: float | None = None
        self._stat_frames = 0
        self._actual_fps = 0.0
        self._last_left_frame: int | None = None
        self._last_right_frame: int | None = None
        self._drop_count = 0
        self.roi_editing = False
        self.focus_roi_editing = False
        self._last_device_status = "尚未刷新设备。"
        self._last_video_sides: list[str] = []
        self._last_quality_metrics: dict[str, object] | None = None
        self._last_left_frame_obj: CameraFrame | None = None
        self._last_right_frame_obj: CameraFrame | None = None
        self._last_analysis_time = 0.0
        self._last_preview_analysis_gate_time = 0.0
        self._preview_frame_counter = 0
        self._cached_focus_roi: dict[str, float] = clamp_roi_frac(self.config.get("focus_roi"))
        self._cached_focus_roi_source: object = None
        self._last_focus_overlay_key: tuple[int | None, int | None, object] | None = None
        self._last_focus_overlay_time = 0.0
        self._last_focus_overlay_left: Image.Image | None = None
        self._last_focus_overlay_right: Image.Image | None = None
        self._last_reference_warning_score: float | None = None
        self._focus_drift_timer: threading.Timer | None = None
        self._focus_drift_warning_visible = False
        self._focus_drift_warning_text = ""
        self._magnifier_locked = False
        self._magnifier_roi_frac = clamp_roi_frac(self.config.get("focus_roi"))
        self._magnifier_zoom = 1
        self._magnifier_image_refs: list[ImageTk.PhotoImage] = []
        self._stereo_blink_phase = 0
        self._last_rectified_overlay_key: tuple[int | None, int | None] | None = None
        self._last_rectified_overlay_image: Image.Image | None = None
        self._field_correction_lock = threading.Lock()
        self._dark_frame_refs: dict[str, np.ndarray] = {}
        self._flat_field_refs: dict[str, np.ndarray] = {}
        self._device_versions: dict[str, str | None] = {}
        self._latest_temperatures: dict[str, float | None] = {}
        self._latest_link_throughput_mbps: dict[str, float | None] = {}
        self._latest_stream_stats: dict[str, dict[str, int | bool]] = {}
        self._temperature_samples: list[dict[str, object]] = []
        self._last_temperature_poll = 0.0
        self._last_stream_stats_poll = 0.0
        self._focus_history: list[tuple[float, float]] = []
        self._focus_peak_score = 0.0
        self._calibration_wizard: dict[str, object] = {}

        self.status_var = StringVar(value="准备就绪。请先连接相机。")
        self.save_dir_var = StringVar(value=str(self.config.get("save_dir", "captures")))
        self._init_control_vars()
        self._set_cached_trigger_source(str(self.config.get("trigger_source", "Software")))
        self._configure_style()
        self._build_ui()
        self.root.bind(UI_QUEUE_EVENT, self._on_ui_queue_event, add="+")
        self._schedule_ui_queue_fallback()

    def _ensure_default_full_resolution(self) -> None:
        if self.config.get("roi_width") in (None, ""):
            self.config["roi_width"] = CAPTURE_WIDTH
        if self.config.get("roi_height") in (None, ""):
            self.config["roi_height"] = CAPTURE_HEIGHT
        if self.config.get("roi_offset_x") in (None, ""):
            self.config["roi_offset_x"] = 0
        if self.config.get("roi_offset_y") in (None, ""):
            self.config["roi_offset_y"] = 0
        for side in ("left", "right"):
            for field, fallback in (
                ("width", self.config.get("roi_width", CAPTURE_WIDTH)),
                ("height", self.config.get("roi_height", CAPTURE_HEIGHT)),
                ("offset_x", self.config.get("roi_offset_x", 0)),
                ("offset_y", self.config.get("roi_offset_y", 0)),
            ):
                key = f"{side}_roi_{field}"
                if self.config.get(key) in (None, ""):
                    self.config[key] = fallback

    def _ensure_config_section(self, key: str) -> dict[str, object]:
        section = self.config.get(key)
        if not isinstance(section, dict):
            section = {}
            self.config[key] = section
        return section

    def _init_control_vars(self) -> None:
        presets = self.config.get("presets", {})
        default_preset = "室内低光" if "室内低光" in presets else next(iter(presets), "室内低光")
        self.preset_var = StringVar(value=default_preset)
        self.trigger_source_var = StringVar(value=display_trigger_source(self.config.get("trigger_source", "Software")))
        self.exposure_auto_var = StringVar(value=str(self.config.get("exposure_auto", "Off")))
        self.exposure_time_var = StringVar(value=str(self.config.get("exposure_time_us", 10000.0)))
        self.auto_exposure_lower_var = StringVar(value=optional_config_text(self.config, "auto_exposure_lower_limit", "100.0"))
        self.auto_exposure_upper_var = StringVar(
            value=optional_config_text(self.config, "auto_exposure_upper_limit", "100000.0")
        )
        self.gain_auto_var = StringVar(value=str(self.config.get("gain_auto", "Off")))
        self.gain_var = StringVar(value=str(self.config.get("gain", 0.0)))
        self.auto_gain_lower_var = StringVar(value=optional_config_text(self.config, "auto_gain_lower_limit", "0.0"))
        self.auto_gain_upper_var = StringVar(value=optional_config_text(self.config, "auto_gain_upper_limit", "15.0"))
        self.balance_auto_var = StringVar(value=str(self.config.get("balance_white_auto", "Off")))
        self.balance_red_var = StringVar(value=optional_config_text(self.config, "balance_ratio_red", ""))
        self.balance_green_var = StringVar(value=optional_config_text(self.config, "balance_ratio_green", ""))
        self.balance_blue_var = StringVar(value=optional_config_text(self.config, "balance_ratio_blue", ""))
        self.black_level_var = StringVar(value=optional_config_text(self.config, "black_level", ""))
        self.digital_shift_var = StringVar(value=optional_config_text(self.config, "digital_shift", ""))
        self.gamma_var = StringVar(value=optional_config_text(self.config, "gamma", ""))
        self.left_roi_width_var = StringVar(value=str(self.config.get("left_roi_width", self.config.get("roi_width", CAPTURE_WIDTH))))
        self.left_roi_height_var = StringVar(value=str(self.config.get("left_roi_height", self.config.get("roi_height", CAPTURE_HEIGHT))))
        self.left_roi_offset_x_var = StringVar(value=str(self.config.get("left_roi_offset_x", self.config.get("roi_offset_x", 0))))
        self.left_roi_offset_y_var = StringVar(value=str(self.config.get("left_roi_offset_y", self.config.get("roi_offset_y", 0))))
        self.right_roi_width_var = StringVar(value=str(self.config.get("right_roi_width", self.config.get("roi_width", CAPTURE_WIDTH))))
        self.right_roi_height_var = StringVar(value=str(self.config.get("right_roi_height", self.config.get("roi_height", CAPTURE_HEIGHT))))
        self.right_roi_offset_x_var = StringVar(value=str(self.config.get("right_roi_offset_x", self.config.get("roi_offset_x", 0))))
        self.right_roi_offset_y_var = StringVar(value=str(self.config.get("right_roi_offset_y", self.config.get("roi_offset_y", 0))))
        self.interval_seconds_var = StringVar(value=optional_config_text(self.config, "interval_capture_seconds", "5.0"))
        self.interval_limit_var = StringVar(value=optional_config_text(self.config, "interval_capture_count", ""))
        self.photo_count_var = StringVar(value="拍照次数 0")
        self.record_fps_var = StringVar(value=str(self.config.get("record_fps", 5.0)))
        dic_section = self.config.get("dic_capture", {})
        dic_record_fps = (
            dic_section.get("record_fps", DIC_CAPTURE_CONFIG["record_fps"])
            if isinstance(dic_section, dict)
            else DIC_CAPTURE_CONFIG["record_fps"]
        )
        self.dic_record_fps_var = StringVar(value=str(dic_record_fps))
        dic_pixel_format = (
            str(dic_section.get("pixel_format", DIC_CAPTURE_CONFIG["pixel_format"]))
            if isinstance(dic_section, dict)
            else str(DIC_CAPTURE_CONFIG["pixel_format"])
        )
        self.dic_pixel_format_var = StringVar(value=dic_pixel_format if dic_pixel_format in DIC_PIXEL_FORMATS else "Mono8")
        self.record_max_seconds_var = StringVar(value=optional_config_text(self.config, "record_max_seconds", "0"))
        exposure_monitor = self._ensure_config_section("exposure_monitor")
        self.preview_quality_analysis_var = BooleanVar(
            value=config_bool(self.config, "preview_quality_analysis_enabled", True, True)
        )
        self.focus_peaking_var = BooleanVar(value=False)
        self.zebra_var = BooleanVar(value=bool(exposure_monitor.get("zebra_enabled", False)))
        self.histogram_enabled_var = BooleanVar(value=bool(exposure_monitor.get("histogram_enabled", True)))
        self.param_panel_open_var = BooleanVar(value=True)
        self.focus_panel_open_var = BooleanVar(value=True)
        self.exposure_panel_open_var = BooleanVar(value=True)
        self.validation_panel_open_var = BooleanVar(value=True)
        self.magnifier_enabled_var = BooleanVar(value=False)
        self.project_id_var = StringVar(value=self.project_manager.current_project_id or "--")
        self.calibration_summary_var = StringVar(value=self.calibration.status_text())
        self.temperature_status_var = StringVar(value="Temp --")
        self.camera_health_var = StringVar(value="Health --")
        fixed_offset = self.config.get("camera_timestamp_offset_fixed")
        if fixed_offset not in (None, ""):
            offset_text = self._format_camera_timestamp_offset(int(fixed_offset))
        else:
            offset_text = "相机时基差 --"
        self.timestamp_offset_var = StringVar(value=offset_text)
        field_correction = self._ensure_config_section("field_correction")
        self.field_correction_enabled_var = BooleanVar(value=config_bool(field_correction, "enabled", False, False))
        self.field_correction_status_var = StringVar(value=self._field_correction_status_text())
        self.dic_quality_var = StringVar(value="DIC speckle --")
        self.focus_peak_var = StringVar(value="峰值 -- | 0%")
        hdr = self._ensure_config_section("hdr_bracketing")
        hdr_ev_offsets = hdr.get("ev_offsets", [-2, -1, 0, 1, 2])
        if not isinstance(hdr_ev_offsets, (list, tuple)):
            hdr_ev_offsets = [-2, -1, 0, 1, 2]
        self.hdr_enabled_var = BooleanVar(value=config_bool(hdr, "enabled", True, True))
        self.hdr_sequence_var = StringVar(
            value=", ".join(str(v) for v in hdr_ev_offsets)
        )
        self.focus_score_var = StringVar(value="对焦 --")
        self.focus_detail_var = StringVar(value="L: -- | R: -- | Δ: --")
        self.focus_status_var = StringVar(value="未标定")
        self.focus_roi_var = StringVar(value=self._format_focus_roi())
        self.exposure_status_var = StringVar(value="过曝: -- | 欠曝: -- | SNR: --")
        self.exposure_advice_var = StringVar(value="曝光建议: --")
        self.capture_gate_var = StringVar(value="采集检查: --")
        self.epipolar_status_var = StringVar(value="极线偏差: --")
        self.calibration_status_var = StringVar(value="标定板覆盖: --")
        self.stereo_preview_mode_var = StringVar(value="正常预览")
        self.guide_mode_var = StringVar(value="关闭")
        self.mp4_progress_var = StringVar(value="MP4 --")
        self._focus_peaking_enabled_setting = bool(self.focus_peaking_var.get())
        self._histogram_enabled_setting = bool(self.histogram_enabled_var.get())
        if self.project_manager.enabled:
            self.save_dir_var.set(str(self.project_manager.active_project_dir))

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self.root, style="Toolbar.TFrame", padding=(12, 6, 12, 5))
        toolbar.pack(side=TOP, fill=X)

        header = ttk.Frame(toolbar, style="Toolbar.TFrame")
        header.pack(side=TOP, fill=X)
        title_group = ttk.Frame(header, style="Toolbar.TFrame")
        title_group.pack(side=LEFT, fill=X, expand=True)
        ttk.Label(title_group, text="MVSS Capture", style="AppTitle.TLabel").pack(side=LEFT)
        ttk.Label(title_group, text="双目同步采集控制台", style="AppSubtitle.TLabel").pack(
            side=LEFT, padx=(12, 0)
        )
        ttk.Label(header, text=f"{CAPTURE_WIDTH} x {CAPTURE_HEIGHT}", style="HeaderValue.TLabel").pack(
            side=RIGHT
        )
        ttk.Label(header, text="默认采集尺寸", style="HeaderMeta.TLabel").pack(side=RIGHT, padx=(0, 8))

        actions = ttk.Frame(toolbar, style="Toolbar.TFrame")
        actions.pack(side=TOP, fill=X, pady=(5, 0))
        self.connect_button = ttk.Button(actions, text="连接相机", command=self.connect_cameras, style="Accent.TButton")
        self.preview_button = ttk.Button(
            actions, text="开始采集", command=self.toggle_preview, state=DISABLED, style="Capture.TButton"
        )
        self.photo_button = ttk.Button(actions, text="同步拍照", command=self.capture_photo, state=DISABLED)
        self.hdr_button = ttk.Button(actions, text="HDR包围", command=self.capture_hdr_bracket, state=DISABLED)
        self.interval_button = ttk.Button(actions, text="定时拍照", command=self.toggle_interval_capture, state=DISABLED)
        self.record_button = ttk.Button(
            actions, text="开始录像", command=self.toggle_recording, state=DISABLED, style="Record.TButton"
        )
        self.record_preflight_button = ttk.Button(
            actions, text="录像评估", command=self.show_record_preflight_wizard, style="Utility.TButton"
        )
        self.refresh_button = ttk.Button(actions, text="刷新设备", command=self.refresh_devices, style="Utility.TButton")
        self.choose_save_dir_button = ttk.Button(
            actions, text="保存路径", command=self.choose_save_dir, style="Utility.TButton"
        )
        self.project_export_button = ttk.Button(
            actions, text="项目/导出", command=self.show_project_export_popup, style="Utility.TButton"
        )
        self.calibration_wizard_button = ttk.Button(
            actions, text="标定向导", command=self.show_calibration_wizard, style="Utility.TButton"
        )
        self.exit_button = ttk.Button(actions, text="退出", command=self.close, style="Danger.TButton")

        for button in (
            self.connect_button,
            self.preview_button,
            self.photo_button,
            self.hdr_button,
            self.interval_button,
            self.record_button,
            self.record_preflight_button,
            self.refresh_button,
            self.choose_save_dir_button,
            self.project_export_button,
            self.calibration_wizard_button,
        ):
            button.pack(side=LEFT, padx=(0, 5), pady=0)
        self.exit_button.pack(side=RIGHT, pady=0)
        self.refresh_tooltip = ToolTip(self.refresh_button, self._device_tooltip_text)

        settings = ttk.Frame(toolbar, style="Toolbar.TFrame")
        settings.pack(side=TOP, fill=X, pady=(5, 0))

        trigger_panel = ttk.Frame(settings, style="Panel.TFrame", padding=(6, 4))
        trigger_panel.pack(side=LEFT, fill="y", padx=(0, 8))
        ttk.Label(trigger_panel, text="触发", style="Panel.TLabel").grid(row=0, column=0, padx=(0, 4), pady=2)
        ttk.OptionMenu(
            trigger_panel,
            self.trigger_source_var,
            self.trigger_source_var.get(),
            *TRIGGER_SOURCE_CN_ORDER,
        ).grid(row=0, column=1, padx=3, pady=2)
        self.apply_trigger_button = ttk.Button(
            trigger_panel, text="应用触发", command=self.apply_trigger_settings, state=DISABLED
        )
        self.apply_trigger_button.grid(row=0, column=2, padx=(6, 0), pady=2)

        preset_panel = ttk.Frame(settings, style="Panel.TFrame", padding=(6, 4))
        preset_panel.pack(side=LEFT, padx=(0, 8))
        ttk.Label(preset_panel, text="预设", style="Panel.TLabel").grid(row=0, column=0, padx=(0, 4), pady=2)
        preset_names = list(self.config.get("presets", {}).keys()) or [self.preset_var.get()]
        ttk.OptionMenu(preset_panel, self.preset_var, self.preset_var.get(), *preset_names).grid(
            row=0, column=1, padx=3, pady=2
        )
        ttk.Button(preset_panel, text="加载", command=self.load_preset).grid(row=0, column=2, padx=3, pady=2)
        ttk.Button(preset_panel, text="保存", command=self.save_preset).grid(row=0, column=3, padx=3, pady=2)

        interval_panel = ttk.Frame(settings, style="Panel.TFrame", padding=(6, 4))
        interval_panel.pack(side=LEFT, fill=X, expand=True)
        ttk.Label(interval_panel, text="定时拍照", style="Panel.TLabel").grid(row=0, column=0, padx=(0, 4), pady=2)
        self.interval_lamp = Canvas(interval_panel, width=18, height=18, bg=PANEL_COLOR, highlightthickness=0, bd=0)
        self.interval_lamp.grid(row=0, column=1, padx=(0, 8), pady=2)
        self.interval_lamp_id = self.interval_lamp.create_oval(3, 3, 15, 15, fill=SUBTLE_TEXT_COLOR, outline=CANVAS_COLOR)
        self._labeled_entry(interval_panel, "间隔s", self.interval_seconds_var, 7, 0, 2)
        self._labeled_entry(interval_panel, "张数", self.interval_limit_var, 7, 0, 4)
        self._labeled_entry(interval_panel, "录像fps", self.record_fps_var, 7, 0, 6)

        self._labeled_entry(interval_panel, "时长s", self.record_max_seconds_var, 7, 0, 8)
        ttk.Label(interval_panel, textvariable=self.photo_count_var, style="Panel.TLabel").grid(
            row=0, column=10, padx=(10, 4), pady=2, sticky="w"
        )
        ttk.Button(interval_panel, text="复位", command=self.reset_photo_count, width=5).grid(
            row=0, column=11, padx=(0, 4), pady=2
        )
        self.dic_capture_button = ttk.Button(
            interval_panel,
            text="DIC采集",
            command=self.toggle_dic_capture,
            state=DISABLED,
            style="Record.TButton",
            width=8,
        )
        self.dic_capture_button.grid(row=0, column=12, padx=(0, 4), pady=2)
        self.dic_record_fps_entry = self._labeled_entry(interval_panel, "DIC fps", self.dic_record_fps_var, 6, 0, 13)
        ttk.Label(interval_panel, text="输出", style="Panel.TLabel", anchor="e").grid(
            row=0, column=15, padx=(4, 1), pady=1, sticky="e"
        )
        self.dic_pixel_format_menu = ttk.OptionMenu(
            interval_panel,
            self.dic_pixel_format_var,
            self.dic_pixel_format_var.get(),
            *DIC_PIXEL_FORMATS,
        )
        self.dic_pixel_format_menu.grid(row=0, column=16, padx=(0, 2), pady=1, sticky="ew")

        info = ttk.Frame(toolbar, style="InfoBar.TFrame", padding=(8, 4))
        info.pack(side=TOP, fill=X, pady=(5, 0))
        ttk.Label(info, text="Project", style="InfoLabel.TLabel").pack(side=LEFT, padx=(0, 4))
        ttk.Label(info, textvariable=self.project_id_var, style="InfoValue.TLabel").pack(side=LEFT, padx=(0, 18))
        ttk.Label(info, textvariable=self.calibration_summary_var, style="InfoValue.TLabel").pack(
            side=LEFT, padx=(0, 18)
        )
        ttk.Label(info, textvariable=self.temperature_status_var, style="InfoValue.TLabel").pack(
            side=LEFT, padx=(0, 18)
        )
        ttk.Label(info, text="保存路径", style="InfoLabel.TLabel").pack(side=LEFT, padx=(0, 4))
        ttk.Label(info, textvariable=self.save_dir_var, style="InfoValue.TLabel").pack(side=LEFT, fill=X, expand=True)
        self.mp4_progress_label = ttk.Label(info, textvariable=self.mp4_progress_var, style="InfoValue.TLabel")
        self.mp4_progress = ttk.Progressbar(
            info,
            mode="determinate",
            maximum=100,
            length=180,
            style="Green.Horizontal.TProgressbar",
        )

        content = DualCameraStrip(
            self.root,
            CAMERA_ASPECT_RATIO,
            CAMERA_GAP,
            CAMERA_VERTICAL_PADDING,
            bg=BG_COLOR,
            height=camera_row_height(self.root),
        )
        self.camera_strip = content
        content.pack(side=TOP, fill=X)
        content.pack_propagate(False)
        self.left_pane = ZoomImagePane(content, "左相机", roi_callback=self.set_roi_from_preview, side="left")
        self.right_pane = ZoomImagePane(content, "右相机", roi_callback=self.set_roi_from_preview, side="right")
        self.camera_side_panel = ttk.Frame(content, style="Toolbar.TFrame")
        content.set_children(self.left_pane, self.right_pane)
        content.set_sidebar(self.camera_side_panel)

        self.param_panel = ttk.Frame(self.camera_side_panel, style="ParamPanel.TFrame", padding=(4, 4))
        self.param_panel.pack(side=TOP, fill=BOTH, expand=True)
        param_header = ttk.Frame(self.param_panel, style="Toolbar.TFrame")
        param_header.pack(side=TOP, fill=X)
        ttk.Label(param_header, text="参数设置", style="ParamTitle.TLabel").pack(side=LEFT, padx=(0, 8))
        ttk.Label(param_header, text="曝光/增益/白平衡/ROI", style="HeaderMeta.TLabel").pack(side=LEFT)

        self.param_panel_body = ttk.Frame(self.param_panel, style="Toolbar.TFrame")
        param_panel = self.param_panel_body
        param_panel.grid_columnconfigure(0, weight=1)

        gain_panel = ttk.Frame(param_panel, style="Panel.TFrame", padding=(4, 2))
        gain_panel.grid(row=0, column=0, sticky="ew", padx=0, pady=(0, 4))
        self._configure_parameter_grid(gain_panel)
        ttk.Label(gain_panel, text="增 益", style="Panel.TLabel").grid(row=0, column=0, padx=(0, 5), pady=1, sticky="w")
        ttk.OptionMenu(gain_panel, self.gain_auto_var, self.gain_auto_var.get(), "Off", "Once", "Continuous").grid(
            row=0, column=1, columnspan=2, padx=2, pady=1, sticky="w"
        )
        gain_panel.grid_columnconfigure(4, weight=1)
        gain_panel.grid_columnconfigure(5, minsize=70, weight=0)
        self.apply_gain_button = ttk.Button(gain_panel, text="应用", command=self.apply_gain_settings, state=DISABLED, width=6)
        self.apply_gain_button.grid(row=0, column=5, padx=(6, 0), pady=1, sticky="e")
        self._labeled_entry(gain_panel, "值", self.gain_var, 5, 1, 0, stretch=False)
        self._labeled_entry(gain_panel, "下限", self.auto_gain_lower_var, 5, 1, 2, stretch=False)
        self._labeled_entry(gain_panel, "上限", self.auto_gain_upper_var, 5, 1, 4, stretch=False)

        exposure_panel = ttk.Frame(param_panel, style="Panel.TFrame", padding=(4, 2))
        exposure_panel.grid(row=1, column=0, sticky="ew", padx=0, pady=(0, 4))
        self._configure_parameter_grid(exposure_panel)
        ttk.Label(exposure_panel, text="曝 光", style="Panel.TLabel").grid(
            row=0, column=0, padx=(0, 5), pady=1, sticky="w"
        )
        ttk.OptionMenu(
            exposure_panel,
            self.exposure_auto_var,
            self.exposure_auto_var.get(),
            "Off",
            "Once",
            "Continuous",
        ).grid(row=0, column=1, columnspan=2, padx=2, pady=1, sticky="w")
        exposure_panel.grid_columnconfigure(4, weight=1)
        exposure_panel.grid_columnconfigure(5, minsize=70, weight=0)
        self.apply_exposure_button = ttk.Button(
            exposure_panel, text="应用", command=self.apply_exposure_settings, state=DISABLED, width=6
        )
        self.apply_exposure_button.grid(row=0, column=5, padx=(6, 0), pady=1, sticky="e")
        self._labeled_entry(exposure_panel, "us", self.exposure_time_var, 7, 1, 0, stretch=False)
        self._labeled_entry(exposure_panel, "下限", self.auto_exposure_lower_var, 7, 1, 2, stretch=False)
        self._labeled_entry(exposure_panel, "上限", self.auto_exposure_upper_var, 7, 1, 4, stretch=False)

        wb_panel = ttk.Frame(param_panel, style="Panel.TFrame", padding=(4, 2))
        wb_panel.grid(row=2, column=0, sticky="ew", padx=0, pady=(0, 4))
        self._configure_parameter_grid(wb_panel)
        ttk.Label(wb_panel, text="白平衡", style="Panel.TLabel").grid(
            row=0, column=0, padx=(0, 5), pady=1, sticky="w"
        )
        ttk.OptionMenu(
            wb_panel,
            self.balance_auto_var,
            self.balance_auto_var.get(),
            "Off",
            "Once",
            "Continuous",
        ).grid(row=0, column=1, columnspan=2, padx=2, pady=1, sticky="w")
        wb_panel.grid_columnconfigure(4, weight=1)
        wb_panel.grid_columnconfigure(5, minsize=70, weight=0)
        self.apply_wb_button = ttk.Button(
            wb_panel, text="应用", command=self.apply_white_balance_settings, state=DISABLED, width=6
        )
        self.apply_wb_button.grid(row=0, column=5, padx=(6, 0), pady=1, sticky="e")

        for index, (label, variable) in enumerate(
            (("R", self.balance_red_var), ("G", self.balance_green_var), ("B", self.balance_blue_var))
        ):
            col = index * 2
            ttk.Label(wb_panel, text=label, style="Panel.TLabel", anchor="e").grid(
                row=1, column=col, padx=(2, 2), pady=1, sticky="e"
            )
            ttk.Entry(wb_panel, textvariable=variable, width=7).grid(
                row=1, column=col + 1, padx=(0, 6), pady=1, sticky="w"
            )

        correction_panel = ttk.Frame(param_panel, style="Panel.TFrame", padding=(4, 2))
        correction_panel.grid(row=3, column=0, sticky="ew", padx=0, pady=(0, 4))
        for column, width in {0: 44, 1: 50, 2: 38, 3: 50, 4: 48, 5: 50}.items():
            correction_panel.grid_columnconfigure(column, minsize=width, weight=0)
        correction_panel.grid_columnconfigure(6, weight=1)
        ttk.Label(correction_panel, text="图像校正", style="Panel.TLabel").grid(
            row=0, column=0, columnspan=3, padx=(0, 5), pady=1, sticky="w"
        )
        self.apply_correction_button = ttk.Button(
            correction_panel, text="应用", command=self.apply_image_correction_settings, state=DISABLED, width=6
        )
        self.apply_correction_button.grid(row=0, column=4, columnspan=3, padx=(6, 0), pady=1, sticky="e")
        for index, (label, variable) in enumerate(
            (("Black", self.black_level_var), ("Shift", self.digital_shift_var), ("Gamma", self.gamma_var))
        ):
            col = index * 2
            ttk.Label(correction_panel, text=label, style="Panel.TLabel", anchor="e").grid(
                row=1, column=col, padx=(4, 1), pady=1, sticky="e"
            )
            ttk.Entry(correction_panel, textvariable=variable, width=4).grid(
                row=1, column=col + 1, padx=(0, 2), pady=1, sticky="w"
            )

        roi_panel = ttk.Frame(param_panel, style="Panel.TFrame", padding=(4, 2))
        roi_panel.grid(row=4, column=0, sticky="ew", padx=0)
        for column, width in {
            0: 20,
            1: 18,
            2: 40,
            3: 18,
            4: 40,
            5: 18,
            6: 40,
            7: 18,
            8: 40,
        }.items():
            roi_panel.grid_columnconfigure(column, minsize=width, weight=0)
        roi_panel.grid_columnconfigure(9, weight=1)
        ttk.Label(roi_panel, text="ROI", style="Panel.TLabel").grid(
            row=0, column=0, columnspan=2, padx=(0, 5), pady=1, sticky="w"
        )
        self.edit_roi_button = ttk.Button(roi_panel, text="框选ROI", command=self.toggle_roi_edit_mode, width=8)
        self.edit_roi_button.grid(row=0, column=2, columnspan=2, padx=(0, 4), pady=1, sticky="ew")
        self.reset_roi_button = ttk.Button(roi_panel, text="重置", command=self.reset_roi_settings, width=5)
        self.reset_roi_button.grid(row=0, column=4, columnspan=2, padx=(0, 4), pady=1, sticky="ew")
        self.apply_roi_button = ttk.Button(roi_panel, text="应用", command=self.apply_roi_settings, state=DISABLED, width=5)
        self.apply_roi_button.grid(row=0, column=6, columnspan=2, padx=(0, 0), pady=1, sticky="ew")

        def grid_roi_entry(label: str, variable: StringVar, row: int, column: int) -> None:
            ttk.Label(roi_panel, text=label, style="Panel.TLabel", anchor="e").grid(
                row=row, column=column, padx=(2, 1), pady=1, sticky="e"
            )
            ttk.Entry(roi_panel, textvariable=variable, width=5, font=(MONO_FONT_FAMILY, max(BASE_FONT_SIZE - 1, 8))).grid(
                row=row, column=column + 1, padx=(0, 1), pady=1, sticky="w"
            )

        ttk.Label(roi_panel, text="左", style="Panel.TLabel").grid(row=1, column=0, padx=(0, 3), pady=1, sticky="w")
        grid_roi_entry("W", self.left_roi_width_var, 1, 1)
        grid_roi_entry("H", self.left_roi_height_var, 1, 3)
        grid_roi_entry("X", self.left_roi_offset_x_var, 1, 5)
        grid_roi_entry("Y", self.left_roi_offset_y_var, 1, 7)
        ttk.Label(roi_panel, text="右", style="Panel.TLabel").grid(row=2, column=0, padx=(0, 3), pady=1, sticky="w")
        grid_roi_entry("W", self.right_roi_width_var, 2, 1)
        grid_roi_entry("H", self.right_roi_height_var, 2, 3)
        grid_roi_entry("X", self.right_roi_offset_x_var, 2, 5)
        grid_roi_entry("Y", self.right_roi_offset_y_var, 2, 7)
        self.param_panel_body.pack(side=TOP, fill=X, pady=(5, 0))

        self.quality_panel = ttk.Frame(self.root, style="Toolbar.TFrame", padding=(12, 0, 12, 0))
        self.quality_panel.pack(side=TOP, fill=BOTH, expand=True)
        self._build_quality_panels(self.quality_panel)

        ttk.Separator(self.root, orient="horizontal").pack(side=TOP, fill=X)
        status_bar = ttk.Frame(self.root, style="StatusBar.TFrame")
        status_bar.pack(side=BOTTOM, fill=X)
        self.status_label = ttk.Label(
            status_bar,
            textvariable=self.status_var,
            style="Status.TLabel",
            anchor="w",
            padding=(8, 4),
        )
        self.status_label.pack(side=LEFT, fill=X, expand=True)
        ttk.Label(
            status_bar,
            text=APP_VERSION,
            style="StatusVersion.TLabel",
            anchor="e",
            padding=(8, 4),
        ).pack(side=RIGHT)

    def _configure_style(self) -> None:
        self.style = ttk.Style()
        try:
            self.style.theme_use("clam")
        except Exception:
            pass
        self.style.configure(".", background=BG_COLOR, foreground=TEXT_COLOR, font=(FONT_FAMILY, BASE_FONT_SIZE))
        self.style.configure("TFrame", background=BG_COLOR)
        self.style.configure("Toolbar.TFrame", background=SURFACE_COLOR)
        self.style.configure(
            "Panel.TFrame",
            background=PANEL_COLOR,
            bordercolor=BORDER_COLOR,
            borderwidth=1,
            relief="solid",
        )
        self.style.configure("InfoBar.TFrame", background=PANEL_ELEVATED_COLOR)
        self.style.configure("StatusBar.TFrame", background=SURFACE_COLOR)
        self.style.configure("PaneHeader.TFrame", background=PANEL_COLOR)
        self.style.configure("TLabel", background=BG_COLOR, foreground=TEXT_COLOR, font=(FONT_FAMILY, BASE_FONT_SIZE))
        self.style.configure(
            "StatusVersion.TLabel",
            background=SURFACE_COLOR,
            foreground=SUBTLE_TEXT_COLOR,
            font=(FONT_FAMILY, BASE_FONT_SIZE, "bold"),
        )
        self.style.configure(
            "Panel.TLabel", background=PANEL_COLOR, foreground=TEXT_COLOR, font=(FONT_FAMILY, BASE_FONT_SIZE)
        )
        self.style.configure(
            "CompactPanel.TLabel",
            background=PANEL_COLOR,
            foreground=TEXT_COLOR,
            font=(FONT_FAMILY, max(BASE_FONT_SIZE - 1, 8)),
        )
        self.style.configure(
            "AppTitle.TLabel",
            background=SURFACE_COLOR,
            foreground=TEXT_COLOR,
            font=(FONT_FAMILY, APP_TITLE_FONT_SIZE, "bold"),
        )
        self.style.configure(
            "AppSubtitle.TLabel",
            background=SURFACE_COLOR,
            foreground=MUTED_TEXT_COLOR,
            font=(FONT_FAMILY, BASE_FONT_SIZE),
        )
        self.style.configure(
            "HeaderMeta.TLabel",
            background=SURFACE_COLOR,
            foreground=SUBTLE_TEXT_COLOR,
            font=(FONT_FAMILY, BASE_FONT_SIZE),
        )
        self.style.configure(
            "ParamTitle.TLabel",
            background=SURFACE_COLOR,
            foreground=TEXT_COLOR,
            font=(FONT_FAMILY, TITLE_FONT_SIZE, "bold"),
        )
        self.style.configure(
            "HeaderValue.TLabel",
            background=SURFACE_COLOR,
            foreground=ACCENT_ACTIVE_COLOR,
            font=(MONO_FONT_FAMILY, BASE_FONT_SIZE, "bold"),
        )
        self.style.configure(
            "InfoLabel.TLabel",
            background=PANEL_ELEVATED_COLOR,
            foreground=SUBTLE_TEXT_COLOR,
            font=(FONT_FAMILY, BASE_FONT_SIZE),
        )
        self.style.configure(
            "InfoValue.TLabel",
            background=PANEL_ELEVATED_COLOR,
            foreground=TEXT_COLOR,
            font=(FONT_FAMILY, BASE_FONT_SIZE, "bold"),
        )
        self.style.configure(
            "Value.TLabel", background=BG_COLOR, foreground=TEXT_COLOR, font=(FONT_FAMILY, BASE_FONT_SIZE, "bold")
        )
        self.style.configure(
            "PaneTitle.TLabel",
            background=PANEL_COLOR,
            foreground=TEXT_COLOR,
            font=(FONT_FAMILY, TITLE_FONT_SIZE, "bold"),
        )
        self.style.configure(
            "PaneMeta.TLabel",
            background=PANEL_COLOR,
            foreground=ACCENT_ACTIVE_COLOR,
            font=(MONO_FONT_FAMILY, INFO_FONT_SIZE, "bold"),
        )
        self.style.configure(
            "PaneInfo.TLabel",
            background=PANEL_COLOR,
            foreground=MUTED_TEXT_COLOR,
            font=(MONO_FONT_FAMILY, INFO_FONT_SIZE),
        )
        self.style.configure(
            "Performance.TLabel",
            background=PANEL_COLOR,
            foreground=SUCCESS_COLOR,
            font=(MONO_FONT_FAMILY, INFO_FONT_SIZE, "bold"),
        )
        self.style.configure(
            "Status.TLabel", background=SURFACE_COLOR, foreground=MUTED_TEXT_COLOR, font=(FONT_FAMILY, BASE_FONT_SIZE)
        )
        self.style.configure(
            "Tooltip.TLabel",
            background=PANEL_ELEVATED_COLOR,
            foreground=TEXT_COLOR,
            font=(FONT_FAMILY, BASE_FONT_SIZE),
        )
        self.style.configure(
            "TButton",
            background=PANEL_ELEVATED_COLOR,
            foreground=TEXT_COLOR,
            bordercolor=BORDER_COLOR,
            borderwidth=1,
            focusthickness=1,
            focuscolor=ACCENT_COLOR,
            padding=(8, 4),
            relief="flat",
        )
        self.style.map(
            "TButton",
            background=[("pressed", SURFACE_COLOR), ("active", "#2a3641"), ("disabled", "#151b22")],
            bordercolor=[("active", BORDER_STRONG_COLOR), ("disabled", "#232c35")],
            foreground=[("disabled", SUBTLE_TEXT_COLOR)],
        )
        self.style.configure(
            "Accent.TButton",
            background=ACCENT_COLOR,
            foreground="#041016",
            bordercolor=ACCENT_COLOR,
            borderwidth=1,
            padding=(10, 4),
            relief="flat",
        )
        self.style.map(
            "Accent.TButton",
            background=[("pressed", "#0b7fb2"), ("active", ACCENT_ACTIVE_COLOR), ("disabled", "#17212a")],
            bordercolor=[("active", ACCENT_ACTIVE_COLOR), ("disabled", "#25313a")],
            foreground=[("disabled", SUBTLE_TEXT_COLOR)],
        )
        self.style.configure(
            "Capture.TButton",
            background=SUCCESS_COLOR,
            foreground="#03130c",
            bordercolor=SUCCESS_COLOR,
            borderwidth=1,
            padding=(10, 4),
            relief="flat",
        )
        self.style.map(
            "Capture.TButton",
            background=[("pressed", "#209967"), ("active", "#48d99a"), ("disabled", "#17221d")],
            bordercolor=[("active", "#48d99a"), ("disabled", "#25322d")],
            foreground=[("disabled", SUBTLE_TEXT_COLOR)],
        )
        self.style.configure(
            "Record.TButton",
            background="#b94343",
            foreground="#fff3f3",
            bordercolor="#d75a5a",
            borderwidth=1,
            padding=(10, 4),
            relief="flat",
        )
        self.style.map(
            "Record.TButton",
            background=[("pressed", "#923333"), ("active", DANGER_COLOR), ("disabled", "#241b1d")],
            bordercolor=[("active", DANGER_COLOR), ("disabled", "#35272a")],
            foreground=[("disabled", SUBTLE_TEXT_COLOR)],
        )
        self.style.configure("Utility.TButton", padding=(9, 4))
        self.style.configure(
            "Danger.TButton",
            background="#2a1d21",
            foreground="#ffb4b4",
            bordercolor="#6f373d",
            borderwidth=1,
            padding=(9, 4),
            relief="flat",
        )
        self.style.map("Danger.TButton", background=[("pressed", "#21171a"), ("active", "#3a2529")])
        self.style.configure("Pane.TButton", padding=(7, 3))
        self.style.configure(
            "TEntry",
            fieldbackground=CHART_COLOR,
            foreground=TEXT_COLOR,
            bordercolor=BORDER_COLOR,
            lightcolor=BORDER_COLOR,
            darkcolor=BORDER_COLOR,
            insertcolor=TEXT_COLOR,
            padding=(4, 2),
        )
        self.style.map("TEntry", bordercolor=[("focus", ACCENT_COLOR), ("disabled", "#252c34")])
        self.style.configure(
            "TMenubutton",
            background=PANEL_ELEVATED_COLOR,
            foreground=TEXT_COLOR,
            bordercolor=BORDER_COLOR,
            borderwidth=1,
            padding=(7, 3),
        )
        self.style.map(
            "TMenubutton",
            background=[("pressed", SURFACE_COLOR), ("active", "#2a3641"), ("disabled", "#151b22")],
            foreground=[("disabled", SUBTLE_TEXT_COLOR)],
        )
        self.style.configure(
            "TCheckbutton",
            background=PANEL_COLOR,
            foreground=TEXT_COLOR,
            indicatorcolor=CHART_COLOR,
            focuscolor=ACCENT_COLOR,
            padding=(2, 1),
        )
        self.style.map(
            "TCheckbutton",
            background=[("active", PANEL_COLOR)],
            foreground=[("disabled", SUBTLE_TEXT_COLOR)],
            indicatorcolor=[("selected", ACCENT_COLOR), ("disabled", "#252c34")],
        )
        self.style.configure(
            "Toolbar.TLabelframe",
            background=SURFACE_COLOR,
            bordercolor=BORDER_COLOR,
            borderwidth=1,
            relief="solid",
        )
        self.style.configure(
            "ParamPanel.TFrame",
            background=SURFACE_COLOR,
            bordercolor=BORDER_COLOR,
            borderwidth=1,
            relief="solid",
        )
        self.style.configure(
            "Toolbar.TLabelframe.Label",
            background=SURFACE_COLOR,
            foreground=MUTED_TEXT_COLOR,
            font=(FONT_FAMILY, BASE_FONT_SIZE, "bold"),
        )
        self.style.configure(
            "TLabelframe",
            background=BG_COLOR,
            bordercolor=BORDER_COLOR,
            borderwidth=1,
            relief="solid",
        )
        self.style.configure(
            "TLabelframe.Label",
            background=BG_COLOR,
            foreground=MUTED_TEXT_COLOR,
            font=(FONT_FAMILY, BASE_FONT_SIZE, "bold"),
        )
        self.style.configure("Horizontal.TSeparator", background=BORDER_COLOR)
        self.style.configure("Green.Horizontal.TProgressbar", troughcolor=CHART_COLOR, background=SUCCESS_COLOR)
        self.style.configure("Yellow.Horizontal.TProgressbar", troughcolor=CHART_COLOR, background=WARNING_COLOR)
        self.style.configure("Red.Horizontal.TProgressbar", troughcolor=CHART_COLOR, background=DANGER_COLOR)
        self.style.configure(
            "Good.TLabel", background=BG_COLOR, foreground=SUCCESS_COLOR, font=(FONT_FAMILY, BASE_FONT_SIZE, "bold")
        )
        self.style.configure(
            "Warn.TLabel", background=BG_COLOR, foreground=WARNING_COLOR, font=(FONT_FAMILY, BASE_FONT_SIZE, "bold")
        )
        self.style.configure(
            "Bad.TLabel", background=BG_COLOR, foreground=DANGER_COLOR, font=(FONT_FAMILY, BASE_FONT_SIZE, "bold")
        )
        self.style.configure(
            "PanelGood.TLabel",
            background=PANEL_COLOR,
            foreground=SUCCESS_COLOR,
            font=(FONT_FAMILY, BASE_FONT_SIZE, "bold"),
        )
        self.style.configure(
            "PanelWarn.TLabel",
            background=PANEL_COLOR,
            foreground=WARNING_COLOR,
            font=(FONT_FAMILY, BASE_FONT_SIZE, "bold"),
        )
        self.style.configure(
            "PanelBad.TLabel",
            background=PANEL_COLOR,
            foreground=DANGER_COLOR,
            font=(FONT_FAMILY, BASE_FONT_SIZE, "bold"),
        )

    def _build_quality_panels(self, parent: ttk.Frame) -> None:
        parent.grid_columnconfigure(0, weight=7, uniform="quality_panels")
        parent.grid_columnconfigure(1, weight=4, uniform="quality_panels")
        monitor_height = QUALITY_MONITOR_MIN_HEIGHT
        try:
            if int(self.root.winfo_screenheight()) >= 1200:
                monitor_height = QUALITY_MONITOR_HIGH_RES_MIN_HEIGHT
        except Exception:
            pass
        parent.grid_rowconfigure(0, minsize=monitor_height, weight=1)
        parent.grid_rowconfigure(1, weight=0)

        self.focus_panel = ttk.LabelFrame(parent, text="对焦辅助", padding=(6, 4))
        self.focus_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=(0, 5))
        self._build_focus_panel(self.focus_panel)

        self.exposure_panel = ttk.LabelFrame(parent, text="曝光监控", padding=(6, 4))
        self.exposure_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=(0, 5))
        self._build_exposure_panel(self.exposure_panel)

        self.validation_panel = ttk.Frame(parent, style="Panel.TFrame", padding=(6, 4))
        self.validation_panel.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=0, pady=(0, 5))
        self._build_validation_panel(self.validation_panel)

    def _build_camera_side_panels(self, parent: ttk.Frame) -> None:
        self.focus_panel = ttk.LabelFrame(parent, text="对焦辅助", padding=(6, 4))
        self.focus_panel.pack(side=TOP, fill=X, pady=(0, 8))
        self._build_focus_panel(self.focus_panel)

        self.exposure_panel = ttk.LabelFrame(parent, text="曝光监控", padding=(6, 4))
        self.exposure_panel.pack(side=TOP, fill=X)
        self._build_exposure_panel(self.exposure_panel)

        self.health_panel = ttk.LabelFrame(parent, text="相机健康", padding=(6, 4))
        self.health_panel.pack(side=TOP, fill=X, pady=(8, 0))
        self._build_health_panel(self.health_panel)

        self.science_panel = ttk.LabelFrame(parent, text="测量校正", padding=(6, 4))
        self.science_panel.pack(side=TOP, fill=X, pady=(8, 0))
        self._build_science_panel(self.science_panel)

    def _build_focus_panel(self, panel: ttk.LabelFrame) -> None:
        top = ttk.Frame(panel, style="Panel.TFrame")
        top.pack(side=TOP, fill=X)
        ttk.Checkbutton(top, text="峰值对焦", variable=self.focus_peaking_var, command=self._sync_quality_toggles).pack(
            side=LEFT, padx=(0, 6)
        )
        ttk.Checkbutton(top, text="放大镜", variable=self.magnifier_enabled_var, command=self._sync_quality_toggles).pack(
            side=LEFT, padx=(0, 6)
        )

        self.focus_panel_body = ttk.Frame(panel, style="Panel.TFrame")
        self.focus_panel_body.pack(side=TOP, fill=BOTH, expand=True)
        self.focus_panel_body.grid_columnconfigure(0, weight=3)
        self.focus_panel_body.grid_columnconfigure(1, weight=2, minsize=260)
        self.focus_panel_body.grid_rowconfigure(0, weight=1)

        focus_controls = ttk.Frame(self.focus_panel_body, style="Panel.TFrame")
        focus_controls.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        command_row = ttk.Frame(focus_controls, style="Panel.TFrame")
        command_row.pack(side=TOP, fill=X, pady=(3, 0))
        self.focus_roi_button = ttk.Button(command_row, text="框选对焦ROI", command=self.toggle_focus_roi_edit_mode)
        self.focus_roi_button.pack(side=LEFT, padx=(0, 4))
        ttk.Button(command_row, text="设为目标", command=self.set_focus_reference).pack(side=LEFT, padx=(0, 4))
        ttk.Button(command_row, text="保存基准", command=self.save_focus_reference_snapshot).pack(side=LEFT)
        score_row = ttk.Frame(focus_controls, style="Panel.TFrame")
        score_row.pack(side=TOP, fill=X, pady=(3, 0))
        ttk.Label(score_row, textvariable=self.focus_score_var).pack(side=LEFT, padx=(0, 8))
        self.focus_progress = ttk.Progressbar(score_row, mode="determinate", maximum=100, style="Yellow.Horizontal.TProgressbar")
        self.focus_progress.pack(side=LEFT, fill=X, expand=True, padx=(0, 8))
        self.focus_status_label = ttk.Label(score_row, textvariable=self.focus_status_var, style="Warn.TLabel", width=18)
        self.focus_status_label.pack(side=LEFT)

        detail_row = ttk.Frame(focus_controls, style="Panel.TFrame")
        detail_row.pack(side=TOP, fill=X, pady=(2, 0))
        self.focus_detail_label = ttk.Label(detail_row, textvariable=self.focus_detail_var, style="Panel.TLabel")
        self.focus_detail_label.pack(side=LEFT, fill=X, expand=True)
        ttk.Label(detail_row, textvariable=self.focus_roi_var, style="Panel.TLabel").pack(side=RIGHT)

        peak_row = ttk.Frame(focus_controls, style="Panel.TFrame")
        peak_row.pack(side=TOP, fill=X, pady=(2, 0))
        ttk.Label(peak_row, textvariable=self.focus_peak_var, style="Panel.TLabel").pack(side=LEFT, padx=(0, 8))
        self.focus_chart_canvas = Canvas(
            peak_row, width=240, height=28, bg=CHART_COLOR, highlightthickness=1, highlightbackground=BORDER_COLOR
        )
        self.focus_chart_canvas.pack(side=LEFT, fill=X, expand=True)

        self.magnifier_frame = ttk.LabelFrame(self.focus_panel_body, text="对焦放大镜", padding=(5, 3))
        self.magnifier_frame.grid(row=0, column=1, sticky="nsew")
        mag_top = ttk.Frame(self.magnifier_frame, style="Panel.TFrame")
        mag_top.pack(side=TOP, fill=X)
        self.magnifier_info_var = StringVar(value="倍率 100% | 点击预览锁定/解锁，滚轮调倍率")
        ttk.Label(mag_top, textvariable=self.magnifier_info_var, style="Panel.TLabel").pack(side=LEFT, fill=X, expand=True)
        self.magnifier_canvas = Canvas(
            self.magnifier_frame, width=260, height=96, bg=CHART_COLOR, highlightthickness=1, highlightbackground=BORDER_COLOR
        )
        self.magnifier_canvas.pack(side=TOP, fill=BOTH, expand=True, pady=(2, 0))
        self._magnifier_image_ref: ImageTk.PhotoImage | None = None
        self.left_pane.bind_external("<Motion>", self._on_magnifier_motion)
        self.left_pane.bind_external("<ButtonPress-1>", self._on_magnifier_click)
        self.left_pane.bind_external("<MouseWheel>", self._on_magnifier_wheel)
        self.left_pane.bind_external("<Button-4>", self._on_magnifier_wheel)
        self.left_pane.bind_external("<Button-5>", self._on_magnifier_wheel)
        self.right_pane.bind_external("<Motion>", self._on_magnifier_motion)
        self.right_pane.bind_external("<ButtonPress-1>", self._on_magnifier_click)
        self.right_pane.bind_external("<MouseWheel>", self._on_magnifier_wheel)
        self.right_pane.bind_external("<Button-4>", self._on_magnifier_wheel)
        self.right_pane.bind_external("<Button-5>", self._on_magnifier_wheel)
        self._update_quality_optional_sections()

    def _build_exposure_panel(self, panel: ttk.LabelFrame) -> None:
        top = ttk.Frame(panel, style="Panel.TFrame")
        top.pack(side=TOP, fill=X)
        ttk.Checkbutton(top, text="直方图", variable=self.histogram_enabled_var, command=self._sync_quality_toggles).pack(
            side=LEFT, padx=(0, 6)
        )
        ttk.Checkbutton(top, text="斑马纹", variable=self.zebra_var, command=self._sync_quality_toggles).pack(side=LEFT)

        self.exposure_panel_body = ttk.Frame(panel, style="Panel.TFrame")
        self.exposure_panel_body.pack(side=TOP, fill=BOTH, expand=True)
        self.exposure_status_label = ttk.Label(
            self.exposure_panel_body, textvariable=self.exposure_status_var, style="Panel.TLabel"
        )
        self.exposure_status_label.pack(side=TOP, fill=X, pady=(3, 0))
        ttk.Label(self.exposure_panel_body, textvariable=self.exposure_advice_var, style="Panel.TLabel").pack(
            side=TOP, fill=X, pady=(2, 0)
        )
        hist_row = ttk.Frame(self.exposure_panel_body, style="Panel.TFrame")
        hist_row.pack(side=TOP, fill=BOTH, expand=True, pady=(3, 0))
        hist_row.grid_columnconfigure(0, weight=1)
        hist_row.grid_columnconfigure(1, weight=1)
        hist_row.grid_rowconfigure(0, weight=1)
        self.left_hist_canvas = HistogramCanvas(hist_row, "左直方图")
        self.right_hist_canvas = HistogramCanvas(hist_row, "右直方图")
        self.left_hist_canvas.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        self.right_hist_canvas.grid(row=0, column=1, sticky="nsew", padx=(4, 0))

    def _build_health_panel(self, panel: ttk.LabelFrame) -> None:
        ttk.Label(panel, textvariable=self.camera_health_var, style="Panel.TLabel").pack(side=TOP, fill=X)
        self.health_chart_canvas = Canvas(
            panel,
            width=260,
            height=72,
            bg=CHART_COLOR,
            highlightthickness=1,
            highlightbackground=BORDER_COLOR,
        )
        self.health_chart_canvas.pack(side=TOP, fill=X, pady=(4, 0))

    def _build_science_panel(self, panel: ttk.LabelFrame) -> None:
        ttk.Label(panel, textvariable=self.timestamp_offset_var, style="Panel.TLabel").pack(side=TOP, fill=X)
        offset_buttons = ttk.Frame(panel, style="Panel.TFrame")
        offset_buttons.pack(side=TOP, fill=X, pady=(3, 0))
        self.timestamp_offset_button = ttk.Button(
            offset_buttons,
            text="计算时间偏置",
            command=self.calibrate_camera_timestamp_offset,
            state=DISABLED,
            width=12,
        )
        self.timestamp_offset_button.pack(side=LEFT, padx=(0, 4))
        self.timestamp_offset_clear_button = ttk.Button(
            offset_buttons,
            text="清除偏置",
            command=self.clear_camera_timestamp_offset,
            state=DISABLED,
            width=9,
        )
        self.timestamp_offset_clear_button.pack(side=LEFT)
        ttk.Checkbutton(
            panel,
            text="暗场/平场实时校正",
            variable=self.field_correction_enabled_var,
            command=self._sync_field_correction_enabled,
        ).pack(side=TOP, anchor="w", pady=(6, 0))
        ref_buttons = ttk.Frame(panel, style="Panel.TFrame")
        ref_buttons.pack(side=TOP, fill=X, pady=(3, 0))
        self.dark_frame_button = ttk.Button(
            ref_buttons,
            text="采集暗场",
            command=self.capture_dark_frame_reference,
            state=DISABLED,
            width=9,
        )
        self.dark_frame_button.pack(side=LEFT, padx=(0, 4))
        self.flat_field_button = ttk.Button(
            ref_buttons,
            text="采集平场",
            command=self.capture_flat_field_reference,
            state=DISABLED,
            width=9,
        )
        self.flat_field_button.pack(side=LEFT)
        ttk.Label(panel, textvariable=self.field_correction_status_var, style="Panel.TLabel").pack(side=TOP, fill=X, pady=(4, 0))
        ttk.Label(panel, textvariable=self.dic_quality_var, style="Panel.TLabel").pack(side=TOP, fill=X)

    def _build_validation_panel(self, panel: ttk.Frame) -> None:
        top = ttk.Frame(panel, style="Panel.TFrame")
        top.pack(side=TOP, fill=X)
        self.validation_panel_body = top
        ttk.Checkbutton(
            top,
            text="质量分析",
            variable=self.preview_quality_analysis_var,
            command=self._sync_quality_toggles,
        ).pack(side=LEFT, padx=(0, 6))
        self.validation_collapse_button = ttk.Button(top, text="校验", width=5, state=DISABLED)
        ttk.Label(top, text="预览模式", style="CompactPanel.TLabel").pack(side=LEFT, padx=(0, 4))
        ttk.OptionMenu(
            top,
            self.stereo_preview_mode_var,
            self.stereo_preview_mode_var.get(),
            "正常预览",
            "红蓝立体",
            "交替闪烁",
            "校正叠加",
            "位移叠加",
            command=self._on_quality_menu_changed,
        ).pack(side=LEFT, padx=(0, 6))
        ttk.Label(top, text="辅助线", style="CompactPanel.TLabel").pack(side=LEFT, padx=(0, 4))
        ttk.OptionMenu(
            top,
            self.guide_mode_var,
            self.guide_mode_var.get(),
            "关闭",
            "仅十字线",
            "仅网格线",
            "十字+网格",
            command=self._on_quality_menu_changed,
        ).pack(side=LEFT, padx=(0, 6))
        self.epipolar_button = ttk.Button(top, text="极线对准检查", command=self.run_epipolar_check)
        self.epipolar_button.pack(side=LEFT, padx=(0, 10))
        ttk.Label(
            top, textvariable=self.capture_gate_var, style="CompactPanel.TLabel"
        ).pack(side=LEFT, fill=X, expand=True, padx=(0, 8))
        self.epipolar_label = ttk.Label(
            top, textvariable=self.epipolar_status_var, style="CompactPanel.TLabel"
        )
        self.epipolar_label.pack(side=LEFT, fill=X, expand=True, padx=(0, 8))
        ttk.Label(
            top, textvariable=self.calibration_status_var, style="CompactPanel.TLabel"
        ).pack(side=LEFT, fill=X, expand=True)

    def _build_project_panel(self, panel: ttk.LabelFrame) -> None:
        top = ttk.Frame(panel, style="Panel.TFrame")
        top.pack(side=TOP, fill=X)
        ttk.Button(top, text="新建项目", command=self.create_new_project).pack(side=LEFT, padx=(0, 4))
        ttk.Button(top, text="重载标定", command=self.reload_calibration).pack(side=LEFT, padx=(0, 4))
        ttk.Button(top, text="标定向导", command=self.show_calibration_wizard).pack(side=LEFT, padx=(0, 4))
        ttk.Button(top, text="相机健康/校正", command=self.show_health_correction_popup).pack(side=LEFT, padx=(0, 4))
        ttk.Checkbutton(top, text="HDR", variable=self.hdr_enabled_var).pack(side=LEFT)

        body = ttk.Frame(panel, style="Panel.TFrame")
        body.pack(side=TOP, fill=X, pady=(3, 0))
        ttk.Label(body, text="项目", style="Panel.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 4), pady=1)
        ttk.Label(body, textvariable=self.project_id_var, style="Panel.TLabel").grid(row=0, column=1, sticky="ew", pady=1)
        ttk.Label(body, text="EV", style="Panel.TLabel").grid(row=1, column=0, sticky="w", padx=(0, 4), pady=1)
        ttk.Entry(body, textvariable=self.hdr_sequence_var, width=18).grid(row=1, column=1, sticky="ew", pady=1)
        ttk.Label(body, textvariable=self.calibration_summary_var, style="Panel.TLabel").grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(2, 0)
        )
        ttk.Label(body, textvariable=self.temperature_status_var, style="Panel.TLabel").grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=(2, 0)
        )
        body.grid_columnconfigure(1, weight=1)

    def _configure_parameter_grid(self, panel: ttk.Frame) -> None:
        widths = {
            0: 28,
            1: 86,
            2: 32,
            3: 86,
            4: 32,
            5: 86,
            6: 32,
            7: 86,
        }
        for column, width in widths.items():
            panel.grid_columnconfigure(column, minsize=width, weight=0)
        for col in (1, 3, 5, 7):
            panel.grid_columnconfigure(col, weight=1)

    def _labeled_entry(
        self,
        parent: ttk.Frame,
        label: str,
        variable: StringVar,
        width: int = 7,
        row: int = 0,
        column: int = 0,
        *,
        stretch: bool = True,
    ) -> ttk.Entry:
        ttk.Label(parent, text=label, style="Panel.TLabel", anchor="e").grid(
            row=row, column=column, padx=(4, 1), pady=1, sticky="e"
        )
        entry = ttk.Entry(parent, textvariable=variable, width=width)
        entry.grid(row=row, column=column + 1, padx=(0, 2), pady=1, sticky="ew" if stretch else "w")
        return entry

    def _toggle_param_panel(self) -> None:
        self.param_panel_open_var.set(True)
        if not self.param_panel_body.winfo_manager():
            self.param_panel_body.pack(side=TOP, fill=X, pady=(5, 0))
        if self.param_panel.winfo_manager():
            self.param_panel.pack_configure(fill=BOTH, expand=True)
        if hasattr(self, "camera_strip"):
            self.root.update_idletasks()
            self.camera_strip._layout_children()

    def show_project_export_popup(self) -> None:
        popup = getattr(self, "_project_export_popup", None)
        if popup is not None and popup.winfo_exists():
            popup.lift()
            popup.focus_force()
            return

        popup = Toplevel(self.root)
        self._project_export_popup = popup
        popup.title("项目/导出")
        popup.configure(bg=BG_COLOR)
        popup.transient(self.root)
        popup.geometry("+%d+%d" % (self.root.winfo_rootx() + 96, self.root.winfo_rooty() + 96))
        popup.protocol("WM_DELETE_WINDOW", popup.destroy)

        container = ttk.LabelFrame(popup, text="项目/导出", padding=(10, 8))
        container.pack(side=TOP, fill=BOTH, expand=True, padx=12, pady=12)
        self._build_project_panel(container)
        popup.update_idletasks()
        width = max(container.winfo_reqwidth() + 32, 520)
        height = max(container.winfo_reqheight() + 36, 190)
        popup.geometry(f"{width}x{height}+{self.root.winfo_rootx() + 96}+{self.root.winfo_rooty() + 96}")

    def show_health_correction_popup(self) -> None:
        popup = getattr(self, "_health_correction_popup", None)
        if popup is not None and popup.winfo_exists():
            popup.lift()
            popup.focus_force()
            return

        popup = Toplevel(self.root)
        self._health_correction_popup = popup
        popup.title("相机健康/测量校正")
        popup.configure(bg=BG_COLOR)
        popup.transient(self.root)
        popup.geometry("+%d+%d" % (self.root.winfo_rootx() + 128, self.root.winfo_rooty() + 128))

        def close_popup() -> None:
            self._clear_health_correction_popup_refs()
            if popup.winfo_exists():
                popup.destroy()

        popup.protocol("WM_DELETE_WINDOW", close_popup)

        container = ttk.Frame(popup, style="Toolbar.TFrame", padding=(12, 10))
        container.pack(side=TOP, fill=BOTH, expand=True)
        container.grid_columnconfigure(0, weight=1)
        container.grid_columnconfigure(1, weight=1)

        self.health_panel = ttk.LabelFrame(container, text="相机健康", padding=(8, 6))
        self.health_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self._build_health_panel(self.health_panel)

        self.science_panel = ttk.LabelFrame(container, text="测量校正", padding=(8, 6))
        self.science_panel.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        self._build_science_panel(self.science_panel)
        self._sync_health_correction_popup_state()
        self._update_camera_health_display(
            dict(getattr(self, "_latest_temperatures", {})),
            dict(getattr(self, "_latest_link_throughput_mbps", {})),
            dict(getattr(self, "_latest_stream_stats", {})),
        )

        popup.update_idletasks()
        width = max(container.winfo_reqwidth() + 32, 700)
        height = max(container.winfo_reqheight() + 28, 190)
        popup.geometry(f"{width}x{height}+{self.root.winfo_rootx() + 128}+{self.root.winfo_rooty() + 128}")

    def _sync_health_correction_popup_state(self) -> None:
        state = NORMAL if self.camera_system is not None else DISABLED
        for name in (
            "timestamp_offset_button",
            "timestamp_offset_clear_button",
            "dark_frame_button",
            "flat_field_button",
        ):
            button = getattr(self, name, None)
            if button is not None and button.winfo_exists():
                button.configure(state=state)

    def _clear_health_correction_popup_refs(self) -> None:
        for name in (
            "_health_correction_popup",
            "health_panel",
            "science_panel",
            "health_chart_canvas",
            "timestamp_offset_button",
            "timestamp_offset_clear_button",
            "dark_frame_button",
            "flat_field_button",
        ):
            if hasattr(self, name):
                delattr(self, name)

    def _toggle_panel(self, panel_name: str) -> None:
        mapping = {
            "validation": (self.validation_panel_body, self.validation_panel_open_var, self.validation_collapse_button),
        }
        if panel_name not in mapping:
            return
        body, variable, button = mapping[panel_name]
        is_open = bool(variable.get())
        if is_open:
            body.pack_forget()
            variable.set(False)
            button.configure(text="展开")
        else:
            body.pack(side=TOP, fill=X)
            variable.set(True)
            button.configure(text="收起")
            self._redraw_panel_after_expand(panel_name)

    def _redraw_panel_after_expand(self, panel_name: str) -> None:
        if panel_name != "exposure":
            return
        for histogram in (getattr(self, "left_hist_canvas", None), getattr(self, "right_hist_canvas", None)):
            if isinstance(histogram, HistogramCanvas):
                histogram.redraw_when_visible()

    def _toggle_focus_panel(self) -> None:
        return None

    def _toggle_exposure_panel(self) -> None:
        return None

    def _toggle_validation_panel(self) -> None:
        self.validation_panel_open_var.set(True)

    def _on_fullscreen_key(self, _event=None) -> None:
        self.toggle_fullscreen()

    def _on_escape_key(self, _event=None):
        if bool(self.root.attributes("-fullscreen")):
            self.root.attributes("-fullscreen", False)
            return "break"
        return None

    def _on_ui_queue_event(self, _event=None) -> None:
        self.process_ui_queue()

    def _on_quality_menu_changed(self, _value=None) -> None:
        self._sync_quality_toggles()
        if self._last_left_frame_obj is not None or self._last_right_frame_obj is not None:
            self._display_frames(self._last_left_frame_obj, self._last_right_frame_obj)

    def reload_calibration(self) -> None:
        self.calibration = load_stereo_calibration(self.config, BASE_DIR)
        self._last_rectified_overlay_key = None
        self._last_rectified_overlay_image = None
        self.calibration_summary_var.set(self.calibration.status_text())
        self.status_var.set(self.calibration.status_text())

    def create_new_project(self) -> None:
        project_dir = self.project_manager.create_project()
        self.project_manager.sync_config(self.config)
        save_config(self.config)
        self._set_project_status_vars()
        self.status_var.set(f"新建项目：{project_dir}")

    def _project_capture_paths(self, capture_id: str) -> tuple[Path, Path, Path]:
        project_dir = self.project_manager.active_project_dir
        left_dir = project_dir / "left"
        right_dir = project_dir / "right"
        meta_dir = project_dir / "exports" / "captures" / capture_id
        left_dir.mkdir(parents=True, exist_ok=True)
        right_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)
        return left_dir, right_dir, meta_dir

    def _set_project_status_vars(self) -> None:
        self.project_id_var.set(self.project_manager.current_project_id or "--")
        self.save_dir_var.set(str(self.project_manager.active_project_dir))

    def _calibration_wizard_settings(self) -> dict[str, object]:
        wizard = self._ensure_config_section("calibration_wizard")
        positions = wizard.get("positions")
        if not isinstance(positions, list) or not positions:
            positions = ["左上", "上中", "右上", "左中", "中心", "右中", "左下", "下中", "右下"]
            wizard["positions"] = positions
        return {
            "pattern": str(wizard.get("pattern", "chessboard")),
            "columns": int(wizard.get("columns", 9) or 9),
            "rows": int(wizard.get("rows", 6) or 6),
            "square_size_mm": float(wizard.get("square_size_mm", 40.0) or 40.0),
            "marker_size_mm": float(wizard.get("marker_size_mm", 30.0) or 30.0),
            "aruco_dictionary": str(wizard.get("aruco_dictionary", "DICT_5X5_1000")),
            "min_pairs": int(wizard.get("min_pairs", max(len(positions), 6)) or max(len(positions), 6)),
            "positions": [str(item) for item in positions],
        }

    def show_calibration_wizard(self) -> None:
        popup = getattr(self, "_calibration_wizard_popup", None)
        if popup is not None and popup.winfo_exists():
            popup.lift()
            popup.focus_force()
            return

        settings = self._calibration_wizard_settings()
        session_dir = self.project_manager.output_root_for_mode("calibration") / f"calibration_capture_{timestamp_ms()}"
        left_dir = session_dir / "left"
        right_dir = session_dir / "right"
        left_dir.mkdir(parents=True, exist_ok=True)
        right_dir.mkdir(parents=True, exist_ok=True)
        positions = list(settings["positions"])

        popup = Toplevel(self.root)
        self._calibration_wizard_popup = popup
        popup.title("标定采集与在线标定")
        popup.configure(bg=BG_COLOR)
        popup.transient(self.root)
        popup.geometry("+%d+%d" % (self.root.winfo_rootx() + 140, self.root.winfo_rooty() + 96))
        popup.protocol("WM_DELETE_WINDOW", popup.destroy)

        state = {
            "session_dir": session_dir,
            "left_dir": left_dir,
            "right_dir": right_dir,
            "settings": settings,
            "positions": positions,
            "captures": [],
            "running": False,
        }
        self._calibration_wizard = state

        container = ttk.LabelFrame(popup, text="标定向导", padding=(12, 10))
        container.pack(side=TOP, fill=BOTH, expand=True, padx=12, pady=12)
        title_var = StringVar(value="请将棋盘格/标定板移动到左上位置")
        status_var = StringVar(value=f"已采集 0 / {len(positions)} 组；目录：{session_dir}")
        result_var = StringVar(value="重投影误差：--")
        ttk.Label(container, textvariable=title_var, style="Panel.TLabel", font=(FONT_FAMILY, 12, "bold")).pack(
            side=TOP, fill=X
        )
        ttk.Label(container, textvariable=status_var, style="Panel.TLabel").pack(side=TOP, fill=X, pady=(6, 0))
        ttk.Label(container, textvariable=result_var, style="Panel.TLabel").pack(side=TOP, fill=X, pady=(4, 8))

        grid = ttk.Frame(container, style="Panel.TFrame")
        grid.pack(side=TOP, fill=X)
        position_labels: list[ttk.Label] = []
        for index, name in enumerate(positions):
            label = ttk.Label(grid, text=f"{index + 1}. {name}", style="Panel.TLabel", padding=(8, 5))
            label.grid(row=index // 3, column=index % 3, sticky="ew", padx=3, pady=3)
            position_labels.append(label)
        for column in range(3):
            grid.grid_columnconfigure(column, weight=1)

        config_panel = ttk.Frame(container, style="Panel.TFrame")
        config_panel.pack(side=TOP, fill=X, pady=(10, 0))
        pattern_var = StringVar(value=str(settings["pattern"]))
        columns_var = StringVar(value=str(settings["columns"]))
        rows_var = StringVar(value=str(settings["rows"]))
        square_var = StringVar(value=str(settings["square_size_mm"]))
        marker_var = StringVar(value=str(settings["marker_size_mm"]))
        min_pairs_var = StringVar(value=str(settings["min_pairs"]))
        dictionary_var = StringVar(value=str(settings["aruco_dictionary"]))
        ttk.Label(config_panel, text="类型").grid(row=0, column=0, padx=(0, 4), pady=3, sticky="w")
        ttk.OptionMenu(config_panel, pattern_var, pattern_var.get(), "chessboard", "charuco", "charuco_legacy", "circles", "acircles").grid(
            row=0, column=1, padx=(0, 8), pady=3, sticky="w"
        )
        self._labeled_entry(config_panel, "列", columns_var, 6, 0, 2)
        self._labeled_entry(config_panel, "行", rows_var, 6, 0, 4)
        self._labeled_entry(config_panel, "格mm", square_var, 8, 0, 6)
        self._labeled_entry(config_panel, "码mm", marker_var, 8, 1, 0)
        self._labeled_entry(config_panel, "最少组", min_pairs_var, 8, 1, 2)
        ttk.Label(config_panel, text="字典", style="Panel.TLabel").grid(row=1, column=4, padx=(4, 1), pady=1, sticky="e")
        ttk.Entry(config_panel, textvariable=dictionary_var, width=18).grid(row=1, column=5, columnspan=3, padx=(0, 2), pady=1, sticky="ew")

        buttons = ttk.Frame(container, style="Panel.TFrame")
        buttons.pack(side=BOTTOM, fill=X, pady=(12, 0))
        capture_button = ttk.Button(buttons, text="采集当前位置", style="Accent.TButton")
        calibrate_button = ttk.Button(buttons, text="开始在线标定", state=DISABLED)
        open_button = ttk.Button(buttons, text="打开目录", command=lambda: self._open_path(session_dir))
        close_button = ttk.Button(buttons, text="关闭", command=popup.destroy)
        capture_button.pack(side=LEFT, padx=(0, 6))
        calibrate_button.pack(side=LEFT, padx=(0, 6))
        open_button.pack(side=LEFT, padx=(0, 6))
        close_button.pack(side=RIGHT)

        def refresh() -> None:
            count = len(state["captures"])
            next_name = positions[min(count, len(positions) - 1)] if positions else "当前位置"
            title_var.set(f"请将棋盘格/标定板移动到 {next_name} 位置")
            status_var.set(f"已采集 {count} / {len(positions)} 组；目录：{session_dir}")
            for idx, label in enumerate(position_labels):
                if idx < count:
                    label.configure(text=f"{idx + 1}. {positions[idx]}  已采集")
                elif idx == count:
                    label.configure(text=f"{idx + 1}. {positions[idx]}  当前")
                else:
                    label.configure(text=f"{idx + 1}. {positions[idx]}")
            try:
                min_pairs_value = int(min_pairs_var.get() or 0)
            except ValueError:
                min_pairs_value = len(positions)
            calibrate_button.configure(state=NORMAL if count >= min_pairs_value and not state["running"] else DISABLED)

        def capture_current() -> None:
            if self.camera_system is None:
                messagebox.showwarning("相机未连接", "请先连接双目相机。")
                return
            if self.recording or self.interval_capturing:
                messagebox.showwarning("正在采集", "录像或定时拍照进行中，暂不能进行在线标定采集。")
                return
            index = len(state["captures"])
            position = positions[index] if index < len(positions) else f"补充{index + 1}"
            capture_button.configure(state=DISABLED)
            self.status_var.set(f"正在采集标定样本：{position}")

            def worker() -> None:
                try:
                    left, right, trigger_time = self._require_camera_system().capture_pair()
                    if left is None or right is None:
                        raise MvsError("在线标定需要左右相机同时采集到图像。")
                    ext = image_extension(self.config)
                    key = f"{index + 1:02d}_{safe_filename(position)}"
                    left_path = self._save_frame(left, left_dir / f"{key}_left.{ext}")
                    right_path = self._save_frame(right, right_dir / f"{key}_right.{ext}")
                    capture = {
                        "index": index + 1,
                        "position": position,
                        "trigger_time": trigger_time,
                        "left": str(left_path),
                        "right": str(right_path),
                        "left_frame": self._frame_meta(left),
                        "right_frame": self._frame_meta(right),
                    }
                    state["captures"].append(capture)
                    self._write_calibration_wizard_manifest(state)
                    self.ui_queue.put(("calibration_wizard_capture_done", (refresh, capture_button, position)))
                except Exception as exc:
                    self.ui_queue.put(("error", exc))
                    self.ui_queue.put(("calibration_wizard_capture_idle", capture_button))

            self._start_background_thread(worker, "calibration-capture")

        def run_calibration() -> None:
            if state["running"]:
                return
            try:
                updated_settings = {
                    "pattern": pattern_var.get().strip(),
                    "columns": int(columns_var.get()),
                    "rows": int(rows_var.get()),
                    "square_size_mm": float(square_var.get()),
                    "marker_size_mm": float(marker_var.get() or 0.0),
                    "aruco_dictionary": dictionary_var.get().strip() or "DICT_5X5_1000",
                    "min_pairs": int(min_pairs_var.get()),
                }
                if updated_settings["columns"] <= 0 or updated_settings["rows"] <= 0 or updated_settings["square_size_mm"] <= 0:
                    raise ValueError("标定板列、行、格尺寸必须大于 0。")
                if len(state["captures"]) < updated_settings["min_pairs"]:
                    raise ValueError("有效采集组数少于最少组数。")
            except ValueError as exc:
                messagebox.showerror("标定参数错误", str(exc))
                return
            state["settings"].update(updated_settings)
            wizard_cfg = self._ensure_config_section("calibration_wizard")
            wizard_cfg.update(updated_settings)
            save_config(self.config)
            state["running"] = True
            capture_button.configure(state=DISABLED)
            calibrate_button.configure(state=DISABLED)
            result_var.set("正在后台标定，请稍候...")
            self.status_var.set("正在执行在线标定...")

            def worker() -> None:
                try:
                    result, export_info = self._run_online_calibration_from_wizard(state)
                    self.ui_queue.put(("calibration_wizard_done", (result, export_info, result_var, capture_button, calibrate_button, refresh)))
                except Exception as exc:
                    self.ui_queue.put(("error", exc))
                    self.ui_queue.put(("calibration_wizard_failed", (result_var, capture_button, calibrate_button, refresh)))

            self._start_background_thread(worker, "calibration-run")

        capture_button.configure(command=capture_current)
        calibrate_button.configure(command=run_calibration)
        refresh()
        popup.update_idletasks()
        width = max(760, container.winfo_reqwidth() + 32)
        height = max(500, container.winfo_reqheight() + 36)
        popup.geometry(f"{width}x{height}+{self.root.winfo_rootx() + 140}+{self.root.winfo_rooty() + 96}")

    def _write_calibration_wizard_manifest(self, state: dict[str, object]) -> None:
        session_dir = Path(str(state["session_dir"]))
        payload = {
            "mode": "calibration_wizard_capture",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "settings": state.get("settings", {}),
            "positions": state.get("positions", []),
            "captures": state.get("captures", []),
        }
        with (session_dir / "calibration_capture_meta.json").open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, default=json_metadata_default)
            fh.write("\n")

    def _import_biaoding_api(self):
        if not BIAODING_DIR.exists():
            raise RuntimeError(f"标定程序目录不存在：{BIAODING_DIR}")
        biaoding_path = str(BIAODING_DIR)
        if biaoding_path in sys.path:
            sys.path.remove(biaoding_path)
        sys.path.insert(0, biaoding_path)
        try:
            from calibration import calibrate_stereo_from_folders
            from biaoding_app import export_capture_calibration
        except Exception as exc:
            raise RuntimeError(f"无法加载标定程序接口：{exc}") from exc
        return calibrate_stereo_from_folders, export_capture_calibration

    def _run_online_calibration_from_wizard(self, state: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
        calibrate_stereo_from_folders, export_capture_calibration = self._import_biaoding_api()
        session_dir = Path(str(state["session_dir"]))
        left_dir = Path(str(state["left_dir"]))
        right_dir = Path(str(state["right_dir"]))
        settings = state.get("settings", {})
        if not isinstance(settings, dict):
            raise RuntimeError("标定向导设置不可用。")
        output_dir = session_dir / "result"
        marker_value = settings.get("marker_size_mm")
        try:
            marker_size = float(marker_value) if marker_value not in (None, "") else None
        except (TypeError, ValueError):
            marker_size = None
        if marker_size is not None and marker_size <= 0:
            marker_size = None
        result = calibrate_stereo_from_folders(
            left_dir,
            right_dir,
            output_dir,
            pattern=str(settings.get("pattern", "chessboard")),
            columns=int(settings.get("columns", 9) or 9),
            rows=int(settings.get("rows", 6) or 6),
            square_size_mm=float(settings.get("square_size_mm", 40.0) or 40.0),
            marker_size_mm=marker_size,
            aruco_dictionary=str(settings.get("aruco_dictionary", "DICT_5X5_1000")),
            min_pairs=int(settings.get("min_pairs", 6) or 6),
            auto_resize_mismatched=False,
        )
        export_info = export_capture_calibration(
            result,
            BASE_DIR / "calib",
            capture_app_dir=BASE_DIR,
            update_capture_config=True,
        )
        calibration_cfg = self._ensure_config_section("calibration")
        calibration_cfg.update(
            {
                "enabled": True,
                "left_intrinsics": "calib/left.yaml",
                "right_intrinsics": "calib/right.yaml",
                "stereo_params": "calib/stereo.yaml",
            }
        )
        save_config(self.config)
        self.project_manager.register_session(
            "calibration",
            session_dir,
            output_dir / "calibration_result.json",
            {"export": export_info},
        )
        return result, export_info

    def _open_path(self, path: Path) -> None:
        try:
            if sys.platform == "win32":
                os.startfile(str(path))  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as exc:
            self.status_var.set(f"无法打开：{path}；{exc}")

    def toggle_fullscreen(self) -> None:
        current = bool(self.root.attributes("-fullscreen"))
        self.root.attributes("-fullscreen", not current)

    def _device_tooltip_text(self) -> str:
        connected: list[str] = []
        if self.camera_system is not None:
            if self.camera_system.left_info is not None:
                connected.append(f"左相机已连接：{self.camera_system.left_info.label}")
            if self.camera_system.right_info is not None:
                connected.append(f"右相机已连接：{self.camera_system.right_info.label}")
        connected_text = "\n".join(connected) if connected else "当前未连接相机。"
        return f"{self._last_device_status}\n{connected_text}"

    def _update_connected_titles(self, left_info, right_info) -> None:
        if left_info is not None:
            self.left_pane.set_title(f"左相机：{left_info.label}")
        else:
            self.left_pane.set_title("左相机：无信号")
            self.left_pane.set_no_signal()
        if right_info is not None:
            self.right_pane.set_title(f"右相机：{right_info.label}")
        else:
            self.right_pane.set_title("右相机：无信号")
            self.right_pane.set_no_signal()
        self._update_stereo_controls()

    def _update_stereo_controls(self) -> None:
        stereo_ready = self._connected_camera_count() >= 2
        if hasattr(self, "epipolar_button"):
            self.epipolar_button.configure(state=NORMAL if stereo_ready else DISABLED)
        if not stereo_ready:
            self.epipolar_status_var.set("极线偏差: 单相机模式不可用")
            self.capture_gate_var.set("采集检查: 单相机模式已禁用左右一致性检查")

    def _connected_camera_count(self) -> int:
        if self.camera_system is None:
            return 0
        return sum(info is not None for info in (self.camera_system.left_info, self.camera_system.right_info))

    @staticmethod
    def _format_camera_timestamp_offset(offset: int) -> str:
        abs_ns = abs(offset)
        if abs_ns >= 1_000_000_000:
            formatted = f"{abs_ns / 1_000_000_000:.2f} s"
        elif abs_ns >= 1_000_000:
            formatted = f"{abs_ns / 1_000_000:.2f} ms"
        elif abs_ns >= 1_000:
            formatted = f"{abs_ns / 1_000:.1f} μs"
        else:
            formatted = f"{abs_ns} ns"
        direction = "左比右早" if offset >= 0 else "右比左早"
        return f"{direction} {formatted}"

    def _apply_fixed_camera_timestamp_offset(self) -> None:
        if self.camera_system is None or not hasattr(self.camera_system, "set_camera_timestamp_offset"):
            return
        offset = self.config.get("camera_timestamp_offset_fixed")
        if offset in (None, ""):
            return
        try:
            self.camera_system.set_camera_timestamp_offset(int(offset))
            if hasattr(self, "timestamp_offset_var"):
                self.timestamp_offset_var.set(self._format_camera_timestamp_offset(int(offset)))
        except (TypeError, ValueError):
            LOGGER.warning("invalid fixed camera timestamp offset: %s", offset)

    def _display_frames(self, left: CameraFrame | None, right: CameraFrame | None) -> None:
        self._update_stats(left, right)
        self._update_performance_display()
        self._last_left_frame_obj = left
        self._last_right_frame_obj = right
        guide_mode = self._guide_mode_key()
        focus_roi = self._focus_roi()
        self.left_pane.set_analysis_overlays(
            focus_peaking_enabled=self.focus_peaking_var.get(),
            focus_peaking_overlay=self._last_focus_overlay_left,
            zebra_enabled=self.zebra_var.get(),
            guide_mode=guide_mode,
            focus_roi_frac=focus_roi,
            magnifier_rect_frac=self._magnifier_roi_frac if self.magnifier_enabled_var.get() else None,
        )
        self.right_pane.set_analysis_overlays(
            focus_peaking_enabled=self.focus_peaking_var.get(),
            focus_peaking_overlay=self._last_focus_overlay_right,
            zebra_enabled=self.zebra_var.get(),
            guide_mode=guide_mode,
            focus_roi_frac=focus_roi,
            magnifier_rect_frac=self._magnifier_roi_frac if self.magnifier_enabled_var.get() else None,
        )
        mode = self.stereo_preview_mode_var.get()
        if mode == "红蓝立体" and left is not None and right is not None:
            image = make_anaglyph(left.image, right.image)
            info = f"Anaglyph {image.width}x{image.height}"
            self.left_pane.set_display_image(image, info)
            self.right_pane.set_no_signal("红蓝立体模式使用左侧单窗显示")
        elif mode == "交替闪烁":
            self._stereo_blink_phase += 1
            frame = left if self._stereo_blink_phase % 2 == 0 else right
            side = "L" if self._stereo_blink_phase % 2 == 0 else "R"
            if frame is not None:
                self.left_pane.set_display_image(frame.image, f"Blink {side} {frame.width}x{frame.height}")
            else:
                self.left_pane.set_no_signal(f"闪烁 {side}: 无信号")
            self.right_pane.set_no_signal("交替闪烁模式使用左侧单窗显示")
        elif mode == "校正叠加" and left is not None and right is not None:
            image = self._rectified_overlay_for_frames(left, right)
            if image is not None:
                self.left_pane.set_display_image(image, f"Rectified overlay {image.width}x{image.height}")
                self.right_pane.set_no_signal("校正叠加模式使用左侧单窗显示")
            else:
                self.left_pane.set_frame(left)
                self.right_pane.set_frame(right)
                self.calibration_summary_var.set("Calibration: rectification unavailable")
        elif mode == "位移叠加" and left is not None:
            image = self._displacement_overlay_for_frame(left)
            if image is not None:
                self.left_pane.set_display_image(image, f"Displacement overlay {image.width}x{image.height}")
                if right is not None:
                    self.right_pane.set_frame(right)
                else:
                    self.right_pane.set_no_signal()
            else:
                self.left_pane.set_frame(left)
                if right is not None:
                    self.right_pane.set_frame(right)
                else:
                    self.right_pane.set_no_signal()
        else:
            if left is not None:
                self.left_pane.set_frame(left)
            else:
                self.left_pane.set_no_signal()
            if right is not None:
                self.right_pane.set_frame(right)
            else:
                self.right_pane.set_no_signal()
        self._update_magnifier()

    def _rectified_overlay_for_frames(self, left: CameraFrame, right: CameraFrame) -> Image.Image | None:
        key = (left.frame_number, right.frame_number)
        if key == self._last_rectified_overlay_key:
            return self._last_rectified_overlay_image
        settings = self.config.get("calibration", {})
        alpha = float(settings.get("rectified_overlay_alpha", 0.5)) if isinstance(settings, dict) else 0.5
        interval_px = int(settings.get("rectified_line_interval_px", 120)) if isinstance(settings, dict) else 120
        try:
            image = self.calibration.make_rectified_overlay(left.image, right.image, alpha=alpha, line_interval_px=interval_px)
        except Exception as exc:
            LOGGER.exception("rectified preview failed")
            self.calibration_summary_var.set(f"Calibration: preview failed ({exc})")
            image = None
        self._last_rectified_overlay_key = key
        self._last_rectified_overlay_image = image
        return image

    def _displacement_overlay_for_frame(self, frame: CameraFrame) -> Image.Image | None:
        image = getattr(frame, "image", None)
        if image is None:
            return None
        dic_analysis = self.config.get("dic_analysis", {})
        overlay_value = str(dic_analysis.get("overlay_path", "") or "").strip() if isinstance(dic_analysis, dict) else ""
        if not overlay_value:
            self.status_var.set("位移叠加未配置：请在 dic_analysis.overlay_path 指向位移模块输出图。")
            return None
        overlay_path = Path(overlay_value)
        if not overlay_path.is_absolute():
            overlay_path = BASE_DIR / overlay_path
        if not overlay_path.exists():
            self.status_var.set(f"位移叠加文件不存在：{overlay_path}")
            return None
        try:
            overlay = Image.open(overlay_path).convert("RGBA").resize(image.size, Image.Resampling.BILINEAR)
            base = image.convert("RGBA")
            return Image.alpha_composite(base, overlay).convert("RGB")
        except Exception as exc:
            LOGGER.exception("displacement overlay failed")
            self.status_var.set(f"位移叠加失败：{exc}")
            return None

    def _ensure_preview_thread_after_recording(self) -> None:
        if self.camera_system is None or self.recording or self.interval_capturing:
            return
        if not (self.previewing or self._resume_preview_after_recording):
            return
        if not self.camera_system.has_ready_camera():
            self.status_var.set("录像结束后未重启预览：相机连接不可用，请重新连接相机。")
            return
        self._resume_preview_after_recording = False
        self.previewing = True
        self.preview_button.configure(text="停止采集")
        self._reset_stats()
        self._start_preview_thread()

    def _start_preview_thread(self) -> None:
        with self._state_lock:
            if self.preview_thread and self.preview_thread.is_alive():
                return
            self._preview_generation += 1
            generation = self._preview_generation
            config_snapshot = self._config_snapshot()
            thread = threading.Thread(
                target=self._preview_loop,
                args=(generation, config_snapshot),
                daemon=True,
            )
            self.preview_thread = thread
            thread.start()

    def refresh_devices(self) -> None:
        def worker() -> None:
            try:
                cameras, _dev_list = enumerate_cameras()
                if not cameras:
                    self.ui_queue.put(("devices_refreshed", ([], "未检测到相机。")))
                    return
                summary = "; ".join(f"{cam.index}: {cam.label} [{cam.transport}]" for cam in cameras)
                self.ui_queue.put(("devices_refreshed", (cameras, f"检测到 {len(cameras)} 台相机：{summary}")))
            except Exception as exc:
                self.ui_queue.put(("error", exc))

        self._start_background_thread(worker, "refresh-devices")

    def connect_cameras(self) -> None:
        try:
            self._save_current_capture_settings()
        except ValueError:
            self.status_var.set("连接前请先检查参数：曝光、增益、ROI、定时拍照和录像 FPS 必须为数字。")
            return
        self.connect_button.configure(state=DISABLED)
        self.status_var.set("正在连接相机并应用采集参数...")

        def worker() -> None:
            try:
                system_config = self._config_snapshot()
                system_config["allow_single_camera"] = True
                system = StereoCameraSystem(system_config)
                left_info, right_info = system.connect()
                self.camera_system = system
                self._apply_fixed_camera_timestamp_offset()
                self._device_versions = system.device_versions()
                self._load_field_correction_references()
                self._poll_camera_temperatures(force=True)
                self.ui_queue.put(("connected", (left_info, right_info)))
            except Exception as exc:
                self.ui_queue.put(("error", exc))
                self.ui_queue.put(("connect_failed", None))

        self._start_background_thread(worker, "connect-cameras")

    def toggle_preview(self) -> None:
        if self.previewing:
            self.stop_preview()
        else:
            self.start_preview()

    def start_preview(self) -> None:
        if self.camera_system is None:
            return
        if self.previewing and self.preview_thread and self.preview_thread.is_alive():
            return
        self._reset_stats()
        self._set_last_quality_metrics(None)
        self.previewing = True
        self.preview_button.configure(text="停止采集")
        self.status_var.set("实时采集中。鼠标左键拖动画面平移，滚轮缩放；需要框选 ROI 时点击“框选ROI”。")
        self._set_capture_buttons(NORMAL)
        if self.recording or self.interval_capturing:
            return
        self._start_preview_thread()

    def stop_preview(self) -> None:
        self._set_last_quality_metrics(None)
        if self.recording or self.interval_capturing:
            self.previewing = False
            self.preview_button.configure(text="开始采集")
            self._set_capture_buttons(NORMAL)
            self.status_var.set("画面显示已停止，当前采集任务继续运行。")
            return
        self.previewing = False
        self.preview_button.configure(state=DISABLED)
        self.status_var.set("正在停止实时采集...")

    def _preview_loop(self, generation: int, config_snapshot: dict) -> None:
        fps = configured_preview_fps(config_snapshot)
        interval = 1.0 / fps
        timeout_ms = self._preview_capture_timeout_ms(config_snapshot)
        next_time = time.perf_counter()
        had_error = False
        consecutive_timeouts = 0

        try:
            while self.previewing and not self.recording and not self.interval_capturing:
                try:
                    left, right, _trigger_time = self._require_camera_system().capture_pair(timeout_ms=timeout_ms)
                except FrameTimeoutError as exc:
                    consecutive_timeouts += 1
                    self.ui_queue.put(("preview_stats_reset", None))
                    self._handle_capture_exception(exc, "preview", consecutive_timeouts)
                    now = time.perf_counter()
                    if now - self._last_preview_status_time >= 1.0:
                        self._last_preview_status_time = now
                        self.ui_queue.put(("status", self._capture_timeout_message(exc, consecutive_timeouts)))
                    next_time = time.perf_counter() + interval if interval > 0 else time.perf_counter()
                    continue
                except Exception as exc:
                    if self._handle_capture_exception(exc, "preview", 0):
                        next_time = time.perf_counter() + interval if interval > 0 else time.perf_counter()
                        continue
                    raise

                consecutive_timeouts = 0
                left, right = self._correct_frame_pair(left, right)
                if self.previewing:
                    self._preview_frame_counter += 1
                    self._poll_camera_temperatures()
                    if self._should_analyze_preview_frame(self._preview_frame_counter, config_snapshot):
                        analysis = self._analyze_preview_frames(left, right, self._preview_frame_counter)
                        self.ui_queue.put(("quality_metrics", analysis))
                    self.ui_queue.put(("frames", (left, right)))
                now = time.perf_counter()
                if now - self._last_preview_status_time >= 1.0:
                    self._last_preview_status_time = now
                    trigger_note = "等待 Line0 外触发" if self._cached_trigger_source() == "Line0" else "软件触发"
                    target_text = f"{fps:g} fps" if fps is not None else "max"
                    self.ui_queue.put(("status", self._status_with_stats(f"实时采集中：目标 {target_text}；{trigger_note}")))

                if interval > 0:
                    next_time += interval
                    sleep_s = next_time - time.perf_counter()
                    if sleep_s > 0:
                        time.sleep(sleep_s)
                    else:
                        next_time = time.perf_counter()
        except Exception as exc:
            had_error = True
            self.ui_queue.put(("error", exc))
        finally:
            self.ui_queue.put(("preview_done", (had_error, generation)))

    def _preview_capture_timeout_ms(self, config_snapshot: dict | None = None) -> int:
        config_snapshot = config_snapshot or self._config_snapshot()
        frame_timeout_ms = max(config_int(config_snapshot, "frame_timeout_ms", 3000), 1)
        default_timeout_ms = min(frame_timeout_ms, 500)
        preview_timeout_ms = max(config_int(config_snapshot, "preview_frame_timeout_ms", default_timeout_ms), 1)
        return min(preview_timeout_ms, frame_timeout_ms)

    def _should_analyze_preview_frame(self, frame_index: int, config_snapshot: dict | None = None) -> bool:
        config_snapshot = config_snapshot or self._config_snapshot()
        zebra_enabled = bool(self.zebra_var.get()) if hasattr(self, "zebra_var") else False
        analysis_required = (
            config_bool(config_snapshot, "preview_quality_analysis_enabled", True, True)
            or bool(getattr(self, "_histogram_enabled_setting", False))
            or bool(getattr(self, "_focus_peaking_enabled_setting", False))
            or zebra_enabled
        )
        if not analysis_required:
            return False
        fps = optional_positive_fps(config_snapshot.get("preview_fps", 15.0))
        analysis_fps = max(config_float(config_snapshot, "preview_analysis_fps", 1.0), 0.1)
        if fps is None:
            now = time.perf_counter()
            interval_s = max(1.0 / analysis_fps, 1.0)
            last_analysis = float(getattr(self, "_last_preview_analysis_gate_time", 0.0) or 0.0)
            if frame_index <= 1 or now - last_analysis >= interval_s:
                self._last_preview_analysis_gate_time = now
                return True
            return False
        every_n = max(int(round(fps / analysis_fps)), 1)
        return frame_index <= 1 or frame_index % every_n == 0

    def capture_photo(self) -> None:
        if self.camera_system is None or self.interval_capturing:
            return
        self.photo_button.configure(state=DISABLED)
        if self.recording:
            with self._record_last_frame_lock:
                latest = self._record_last_frame_pair
            if latest is None:
                self.status_var.set("录像刚开始，尚无可保存的快照帧。")
                self.photo_button.configure(state=NORMAL)
                return
            left, right, trigger_time = latest
            if (left is not None and getattr(left, "image", None) is None) or (
                right is not None and getattr(right, "image", None) is None
            ):
                try:
                    left, right, trigger_time = self._require_camera_system().capture_pair(convert_image=True)
                except Exception as exc:
                    self.ui_queue.put(("error", exc))
                    self.photo_button.configure(state=NORMAL)
                    return
            left_copy = self._clone_frame(left)
            right_copy = self._clone_frame(right)
            metrics = self._quality_metrics_for_pair(left_copy, right_copy)
            allow_capture, quality_report = self._capture_quality_gate_allows(metrics)
            self._apply_quality_report(quality_report)
            if not allow_capture:
                self.status_var.set("采集已取消：质量检查未通过。")
                self.photo_button.configure(state=NORMAL)
                return
            self.status_var.set("正在从录像原始帧保存同步快照...")

            def record_snapshot_worker() -> None:
                try:
                    photo_dir = self._save_photo_pair(
                        left_copy,
                        right_copy,
                        trigger_time,
                        mode="recording_photo",
                        quality_report=quality_report,
                    )
                    self.ui_queue.put(("shutter_flash", None))
                    self.ui_queue.put(("photo_done", ("photo", photo_dir)))
                except Exception as exc:
                    self.ui_queue.put(("error", exc))
                finally:
                    self.ui_queue.put(("capture_idle", None))

            self._start_background_thread(record_snapshot_worker, "record-snapshot")
            return

        if not self.previewing:
            self.preview_button.configure(state=DISABLED)
        if self.trigger_source_var.get() == "Line0":
            self.status_var.set("正在等待 Line0 外触发帧并保存...")
        else:
            self.status_var.set("正在同步拍照...")

        metrics = self._get_last_quality_metrics()
        if not metrics:
            self.status_var.set("正在获取现场质量检查帧...")

            def precheck_worker() -> None:
                try:
                    left, right, trigger_time = self._require_camera_system().capture_pair()
                    metrics = self._quality_metrics_for_pair(left, right)
                    self.ui_queue.put(("photo_quality_prefetched", (left, right, trigger_time, metrics)))
                except Exception as exc:
                    self.ui_queue.put(("error", exc))
                    self.ui_queue.put(("capture_idle", None))

            self._start_background_thread(precheck_worker, "photo-precheck")
            return

        allow_capture, quality_report = self._capture_quality_gate_allows(metrics)
        self._apply_quality_report(quality_report)
        if not allow_capture:
            self.status_var.set("采集已取消：质量检查未通过。")
            self._set_capture_buttons(NORMAL)
            return

        def worker() -> None:
            try:
                left, right, trigger_time = self._require_camera_system().capture_pair()
                fresh_metrics = self._quality_metrics_for_pair(left, right)
                fresh_report = self._quality_report_from_metrics(fresh_metrics)
                self.ui_queue.put(("quality_report", fresh_report))
                photo_dir = self._save_photo_pair(left, right, trigger_time, mode="photo", quality_report=fresh_report)
                if self.previewing:
                    self.ui_queue.put(("frames", (left, right)))
                self.ui_queue.put(("shutter_flash", None))
                self.ui_queue.put(("photo_done", ("photo", photo_dir)))
            except Exception as exc:
                self.ui_queue.put(("error", exc))
            finally:
                self.ui_queue.put(("capture_idle", None))

        self._start_background_thread(worker, "capture-photo")

    def capture_hdr_bracket(self) -> None:
        if self.camera_system is None or self.recording or self.interval_capturing:
            return
        if not self.hdr_enabled_var.get():
            self.status_var.set("HDR包围未启用。")
            return
        try:
            ev_offsets = self._parse_hdr_ev_offsets()
        except ValueError as exc:
            self.status_var.set(str(exc))
            return
        if not ev_offsets:
            self.status_var.set("HDR包围曝光序列为空。")
            return
        self._set_capture_buttons(DISABLED)
        self.status_var.set(f"HDR包围拍照中：{ev_offsets}")
        restore_lower = self._optional_entry_float(self.auto_exposure_lower_var)
        restore_upper = self._optional_entry_float(self.auto_exposure_upper_var)

        def worker() -> None:
            original_exposure = config_float(self.config, "exposure_time_us", 0.0)
            original_auto = str(self.config.get("exposure_auto", "Off"))
            try:
                camera_system = self._require_camera_system()
                captures: list[dict[str, object]] = []
                for ev in ev_offsets:
                    exposure_us = self._hdr_exposure_for_ev(original_exposure, ev)
                    camera_system.apply_exposure_settings("Off", exposure_us, None, None)
                    time.sleep(max(config_float(self.config.get("hdr_bracketing", {}), "settle_seconds", 0.10), 0.0))
                    left, right, trigger_time = camera_system.capture_pair()
                    captures.append(
                        {
                            "ev_offset": ev,
                            "exposure_time_us": exposure_us,
                            "left": left,
                            "right": right,
                            "trigger_time": trigger_time,
                        }
                    )
                    if self.previewing:
                        self.ui_queue.put(("frames", (left, right)))
                group_dir = self._save_hdr_bracket(captures, original_exposure)
                self.ui_queue.put(("shutter_flash", None))
                self.ui_queue.put(("photo_done", ("hdr", group_dir)))
            except Exception as exc:
                self.ui_queue.put(("error", exc))
            finally:
                try:
                    self._require_camera_system().apply_exposure_settings(
                        original_auto,
                        original_exposure,
                        restore_lower,
                        restore_upper,
                    )
                except Exception as exc:
                    LOGGER.info("failed to restore exposure after HDR bracket: %s", exc)
                self.ui_queue.put(("capture_idle", None))

        self._start_background_thread(worker, "hdr-bracket")

    def _parse_hdr_ev_offsets(self) -> list[float]:
        raw = self.hdr_sequence_var.get().replace(";", ",").split(",")
        values: list[float] = []
        for part in raw:
            value = part.strip()
            if not value:
                continue
            values.append(float(value))
        values = sorted(dict.fromkeys(values))
        hdr = self._ensure_config_section("hdr_bracketing")
        hdr["enabled"] = bool(self.hdr_enabled_var.get())
        hdr["ev_offsets"] = values
        save_config(self.config)
        return values

    def _hdr_exposure_for_ev(self, base_exposure_us: float, ev_offset: float) -> float:
        hdr = self.config.get("hdr_bracketing", {})
        min_us = config_float(hdr, "min_exposure_time_us", 50.0)
        max_us = config_float(hdr, "max_exposure_time_us", 1000000.0)
        base = base_exposure_us if base_exposure_us > 0 else config_float(self.config, "exposure_time_us", 20000.0)
        return min(max(base * (2.0**ev_offset), min_us), max_us)

    def _save_hdr_bracket(self, captures: list[dict[str, object]], base_exposure_us: float) -> Path:
        capture_id = timestamp_ms()
        left_dir, right_dir, meta_dir = self._project_capture_paths(f"{capture_id}_hdr")
        project_dir = self.project_manager.active_project_dir
        ext = image_extension(self.config)
        bracket_meta: list[dict[str, object]] = []
        for index, item in enumerate(captures, start=1):
            ev = float(item["ev_offset"])
            left = item.get("left")
            right = item.get("right")
            ev_token = f"{ev:+.1f}".replace("+", "p").replace("-", "m").replace(".", "p")
            left_path = left_dir / f"{capture_id}_hdr_ev_{ev_token}_left.{ext}"
            right_path = right_dir / f"{capture_id}_hdr_ev_{ev_token}_right.{ext}"
            if isinstance(left, CameraFrame):
                left_path = self._save_frame(left, left_path)
            if isinstance(right, CameraFrame):
                right_path = self._save_frame(right, right_path)
            bracket_meta.append(
                {
                    "index": index,
                    "ev_offset": ev,
                    "exposure_time_us": item.get("exposure_time_us"),
                    "trigger_time": item.get("trigger_time"),
                    "left": self._frame_meta(left) if isinstance(left, CameraFrame) else None,
                    "right": self._frame_meta(right) if isinstance(right, CameraFrame) else None,
                    "left_path": str(left_path) if left_path.exists() else None,
                    "right_path": str(right_path) if right_path.exists() else None,
                }
            )
        first_left = next((item.get("left") for item in captures if isinstance(item.get("left"), CameraFrame)), None)
        first_right = next((item.get("right") for item in captures if isinstance(item.get("right"), CameraFrame)), None)
        self._write_meta(
            meta_dir / "meta.json",
            mode="hdr_bracket",
            capture_id=capture_id,
            trigger_time=captures[0].get("trigger_time") if captures else time.time(),
            left=first_left,
            right=first_right,
            base_exposure_time_us=base_exposure_us,
            brackets=bracket_meta,
            data_manifest={
                "manifest_csv": str(meta_dir / "exports" / "file_manifest.csv"),
                "summary_json": str(meta_dir / "exports" / "capture_summary.json"),
            },
        )
        manifest = self._write_manifest_for_session(
            meta_dir, {"mode": "hdr_bracket", "capture_id": capture_id, "brackets": bracket_meta}
        )
        self.project_manager.register_session(
            "hdr_bracket",
            meta_dir,
            meta_dir / "meta.json",
            {"capture_id": capture_id, "image_root": str(project_dir), "manifest": manifest},
        )
        return meta_dir

    def toggle_interval_capture(self) -> None:
        if self.interval_capturing:
            self.stop_interval_capture()
        else:
            self.start_interval_capture()

    def start_interval_capture(self) -> None:
        if self.camera_system is None:
            return
        if self.recording:
            self.status_var.set("录像中不能启动定时拍照。")
            return
        try:
            interval_s = float(self.interval_seconds_var.get())
            limit = optional_int_text(self.interval_limit_var.get())
        except ValueError:
            self.status_var.set("定时拍照参数必须是数字。")
            return
        if interval_s <= 0:
            self.status_var.set("定时拍照间隔必须大于 0 秒。")
            return
        if limit is not None and limit <= 0:
            self.status_var.set("定时拍照张数必须为空或大于 0。")
            return

        self._update_config({"interval_capture_seconds": interval_s, "interval_capture_count": limit})
        self._reset_stats()
        display_enabled = self.previewing
        if display_enabled:
            self.previewing = False
            if self.preview_thread and self.preview_thread.is_alive():
                self.preview_thread.join(timeout=3)
        self.interval_capturing = True
        self.previewing = display_enabled
        self.interval_stop_event.clear()
        self.interval_count = 0
        self._set_interval_lamp(DANGER_COLOR)
        self.interval_button.configure(text="停止定时")
        self._set_capture_buttons(NORMAL)
        count_text = "持续拍照" if limit is None else f"拍 {limit} 组"
        self.status_var.set(f"定时拍照已启动：每 {interval_s:g} 秒保存一组图像，{count_text}。")
        self.interval_thread = threading.Thread(target=self._interval_capture_loop, args=(interval_s, limit), daemon=True)
        self.interval_thread.start()

    def stop_interval_capture(self) -> None:
        self.interval_capturing = False
        self.interval_stop_event.set()
        self.interval_button.configure(state=DISABLED)
        self._set_interval_lamp(SUBTLE_TEXT_COLOR)
        self.status_var.set("正在停止定时拍照...")

    def _interval_capture_loop(self, interval_s: float, limit: int | None) -> None:
        had_error = False
        started_at = time.perf_counter()
        next_time = time.perf_counter()
        try:
            while self.interval_capturing:
                try:
                    left, right, trigger_time = self._require_camera_system().capture_pair()
                except FrameTimeoutError as exc:
                    self._handle_capture_exception(exc, "interval", 1)
                    next_time = time.perf_counter() + interval_s
                    if self.interval_stop_event.wait(interval_s):
                        break
                    continue
                except Exception as exc:
                    if self._handle_capture_exception(exc, "interval", 0):
                        next_time = time.perf_counter() + interval_s
                        if self.interval_stop_event.wait(interval_s):
                            break
                        continue
                    raise
                left, right = self._correct_frame_pair(left, right)
                self.interval_count += 1
                metrics = self._quality_metrics_for_pair(left, right)
                report = self._quality_report_from_metrics(metrics)
                photo_dir = self._save_photo_pair(left, right, trigger_time, mode="interval_photo", quality_report=report)
                self.ui_queue.put(("interval_lamp_green", None))
                if self.previewing:
                    self._preview_frame_counter += 1
                    analysis = self._analyze_preview_frames(left, right, self._preview_frame_counter)
                    self.ui_queue.put(("quality_metrics", analysis))
                    self.ui_queue.put(("frames", (left, right)))
                self.ui_queue.put(
                    (
                        "status",
                        self._status_with_stats(
                            self._interval_status_text(
                                interval_s,
                                limit,
                                photo_dir.name,
                                time.perf_counter() - started_at,
                            )
                        ),
                    )
                )
                if limit is not None and self.interval_count >= limit:
                    break
                next_time += interval_s
                sleep_s = next_time - time.perf_counter()
                if sleep_s > 0:
                    if self.interval_stop_event.wait(sleep_s):
                        break
                else:
                    next_time = time.perf_counter()
        except Exception as exc:
            had_error = True
            self.ui_queue.put(("error", exc))
        finally:
            self.interval_capturing = False
            self.ui_queue.put(("interval_done", had_error))

    def toggle_recording(self) -> None:
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self) -> None:
        if self.camera_system is None:
            return
        if self.interval_capturing:
            self.status_var.set("定时拍照中不能开始录像。")
            return
        if self._dic_recording:
            self.status_var.set("DIC 图像采集中不能开始普通录像。")
            return
        self._ensure_recording_config_defaults()
        try:
            fps = optional_positive_fps(self.record_fps_var.get())
        except ValueError:
            self.status_var.set("录像 FPS 必须是数字；0 或留空表示不限速。")
            return
        record_updates: dict[str, object] = {"record_fps": fps or 0.0}
        try:
            record_updates["record_max_seconds"] = max(float(self.record_max_seconds_var.get() or 0), 0.0)
        except ValueError:
            self.status_var.set("录像时长必须是数字；0 表示不限时。")
            return
        try:
            record_updates.update(self._current_parameter_config())
        except ValueError:
            self.status_var.set("录像前请先检查相机参数：曝光、增益、ROI 必须为数字。")
            return
        config_snapshot = self._capture_priority_record_config({**self._config_snapshot(), **record_updates})
        persisted_updates = {key: config_snapshot[key] for key in config_snapshot if key in record_updates}
        persisted_updates.update(
            {
                key: config_snapshot[key]
                for key in (
                    "record_save_image_sequence",
                    "auto_make_mp4",
                    "record_realtime_mp4",
                    "record_preview_during_capture",
                    "record_clone_frames_for_writer",
                    "record_checksum_during_capture",
                    "record_queue_max_items",
                    "preview_quality_analysis_enabled",
                    "image_format",
                )
                if key in config_snapshot
            }
        )
        config_snapshot = self._update_config(persisted_updates)
        self._start_record_session(config_snapshot, mode="video")

    def toggle_dic_capture(self) -> None:
        if self._dic_recording and self.recording:
            self.stop_recording()
        else:
            self.start_dic_capture()

    def _dic_capture_config(self) -> dict[str, object]:
        dic_section = self._config_snapshot().get("dic_capture", {})
        dic_overrides = dic_section if isinstance(dic_section, dict) else {}
        snapshot = {**self._config_snapshot(), **dic_capture_defaults(), **dic_overrides}
        gate = dict(snapshot.get("capture_quality_gate", DIC_CAPTURE_CONFIG["capture_quality_gate"]))
        snapshot["capture_quality_gate"] = gate
        snapshot["image_format"] = image_extension(snapshot)
        return snapshot

    def _dic_record_fps_from_entry(self) -> float:
        fps = optional_positive_fps(self.dic_record_fps_var.get())
        if fps is None:
            raise ValueError("DIC FPS must be greater than 0")
        return fps

    def _dic_pixel_format_from_entry(self) -> str:
        if not hasattr(self, "dic_pixel_format_var"):
            section = self.config.get("dic_capture", {}) if hasattr(self, "config") else {}
            value = str(section.get("pixel_format", DIC_CAPTURE_CONFIG["pixel_format"]) if isinstance(section, dict) else DIC_CAPTURE_CONFIG["pixel_format"])
return value if value in DIC_PIXEL_FORMATS else "Mono8"
        value = str(self.dic_pixel_format_var.get() or "Mono8").strip()
        if value not in DIC_PIXEL_FORMATS:
            raise ValueError("unsupported DIC pixel format")
        return value

    def _apply_dic_ui_settings_to_config(self, config_snapshot: dict[str, object]) -> dict[str, object]:
        snapshot = dict(config_snapshot)
        fps = self._dic_record_fps_from_entry()
        pixel_format = self._dic_pixel_format_from_entry()
        snapshot["record_fps"] = fps
        snapshot["pixel_format"] = pixel_format
        high_bit_depth = pixel_format != "Mono8"
        snapshot["save_raw_frames"] = high_bit_depth
        snapshot["record_force_image_format"] = not high_bit_depth
        snapshot["image_format"] = "png"
        snapshot["viewable_sidecar_enabled"] = True
        snapshot["viewable_sidecar_format"] = "png"
        if high_bit_depth:
            snapshot["raw_frame_format"] = str(snapshot.get("raw_frame_format") or "tiff16")
        dic_section = dict(snapshot.get("dic_capture", {}) if isinstance(snapshot.get("dic_capture"), dict) else {})
        dic_section["record_fps"] = fps
        dic_section["pixel_format"] = pixel_format
        dic_section["save_raw_frames"] = high_bit_depth
        dic_section["record_force_image_format"] = not high_bit_depth
        dic_section["image_format"] = "png"
        dic_section["viewable_sidecar_enabled"] = True
        dic_section["viewable_sidecar_format"] = "png"
        if high_bit_depth:
            dic_section["raw_frame_format"] = str(snapshot.get("raw_frame_format") or "tiff16")
        snapshot["dic_capture"] = dic_section
        return snapshot

    def _apply_dic_record_fps_to_config(self, config_snapshot: dict[str, object]) -> dict[str, object]:
        return self._apply_dic_ui_settings_to_config(config_snapshot)

    def start_dic_capture(self) -> None:
        if self.camera_system is None:
            return
        if self.recording:
            self.status_var.set("录像中不能启动 DIC 图像采集。")
            return
        if self.interval_capturing:
            self.status_var.set("定时拍照中不能启动 DIC 图像采集。")
            return
        try:
            config_snapshot = self._apply_dic_ui_settings_to_config(self._dic_capture_config())
        except ValueError:
            self.status_var.set("DIC FPS 必须是大于 0 的数字，输出格式需为 Mono16/Mono12/Mono10/Mono8。")
            return
        self._load_vars_from_snapshot(config_snapshot, include_record_fps=False)
        self.status_var.set("正在应用 DIC 图像采集参数...")
        self.dic_capture_button.configure(state=DISABLED)
        self._set_parameter_buttons(DISABLED)

        def worker() -> None:
            try:
                warnings = self._apply_capture_config_to_camera(config_snapshot)
                persisted = self._update_config(config_snapshot)
                self._set_cached_trigger_source(str(persisted.get("trigger_source", "Software")))
                self.ui_queue.put(("dic_start", (persisted, warnings)))
            except Exception as exc:
                self.ui_queue.put(("error", exc))
                self.ui_queue.put(("dic_start_failed", None))

        self._start_background_thread(worker, "start-dic-capture")

    def _start_record_session(self, config_snapshot: dict, *, mode: str = "video", status_prefix: str = "正在录像") -> bool:
        if not self._check_disk_space_for_recording(config_snapshot):
            return False
        self._reset_stats()
        resume_preview_after_recording = self.previewing
        display_enabled = config_bool(config_snapshot, "record_preview_during_capture", True, True)
        self._resume_preview_after_recording = resume_preview_after_recording
        if self.previewing:
            self.previewing = False
            if self.preview_thread and self.preview_thread.is_alive():
                self.preview_thread.join(timeout=3)
        self.recording = True
        self.previewing = display_enabled
        record_started_at = time.perf_counter()
        self._reset_record_stats(record_started_at)
        self._reset_record_write_state()
        self._record_last_disk_check = 0.0
        self._record_disk_usage_start = self._disk_used_bytes(resolve_output_root(self.config))
        with self._record_last_frame_lock:
            self._record_last_frame_pair = None
        self.record_dir = self.project_manager.output_root_for_mode("videos") / time.strftime("%Y%m%d_%H%M%S")
        self.record_dir.mkdir(parents=True, exist_ok=True)
        config_snapshot = {**config_snapshot, "record_mode": mode}
        if mode == DIC_CAPTURE_MODE:
            self._dic_recording = True
            self.dic_capture_button.configure(text="停止DIC")
        else:
            self._dic_recording = False
        self.record_button.configure(text="停止录像")
        self._set_capture_buttons(NORMAL)
        self._set_recording_indicator(True)
        display_note = "并显示画面" if self.previewing else ""
        self.status_var.set(f"{status_prefix}{display_note}：{self.record_dir}")
        self.record_thread = threading.Thread(target=self._record_loop, args=(config_snapshot,), daemon=True)
        self.record_thread.start()
        return True

    def stop_recording(self) -> None:
        self.recording = False
        self.record_button.configure(state=DISABLED)
        if self._dic_recording and hasattr(self, "dic_capture_button"):
            self.dic_capture_button.configure(state=DISABLED)
        self._set_recording_indicator(False)
        self.status_var.set("正在停止录像并整理文件...")

    def _record_loop(self, config_snapshot: dict | None = None) -> None:
        """录制主循环（转入 V2 录制引擎）。"""
        self._record_loop_v2(config_snapshot or self._config_snapshot())

    def _require_camera_system(self) -> StereoCameraSystem:
        if self.camera_system is None:
            raise MvsError("camera system is not initialized")
        return self.camera_system

    def _require_record_dir(self) -> Path:
        if self.record_dir is None:
            raise RuntimeError("record directory is not initialized")
        return self.record_dir

    def _create_video_writer(self, path: Path, fps: float, image: Image.Image) -> cv2.VideoWriter:
        width, height = image.size
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height), True)
        if not writer.isOpened():
            raise RuntimeError(f"无法创建视频文件：{path}")
        return writer

    def _image_to_video_frame(self, image: Image.Image) -> np.ndarray:
        if image.mode != "RGB":
            image = image.convert("RGB")
        rgb = np.array(image)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def _frame_to_video_frame(self, frame: CameraFrame) -> np.ndarray:
        raw_frame = self._raw_frame_to_video_frame(frame)
        if raw_frame is not None:
            return raw_frame
        if getattr(frame, "image", None) is None:
            raise RuntimeError("raw-only frame cannot be converted to video with the current pixel format")
        return self._image_to_video_frame(frame.image)

    def _release_frame_raw(self, frame: CameraFrame | None) -> None:
        if frame is not None and hasattr(frame, "release_raw_data"):
            frame.release_raw_data()

    def _release_unsaved_record_frames(
        self,
        left: CameraFrame | None,
        right: CameraFrame | None,
        *,
        queued_for_writer: bool,
    ) -> None:
        if queued_for_writer:
            return
        self._release_frame_raw(left)
        self._release_frame_raw(right)

    def _raw_frame_to_video_frame(self, frame: CameraFrame) -> np.ndarray | None:
        raw_data = getattr(frame, "raw_data", None)
        width = int(getattr(frame, "width", 0) or 0)
        height = int(getattr(frame, "height", 0) or 0)
        if not raw_data or width <= 0 or height <= 0:
            return None
        pixel_name = str(getattr(frame, "pixel_type_name", "") or "").lower()
        raw_len = int(getattr(frame, "raw_frame_len", 0) or len(raw_data))
        if "rgb" in pixel_name and raw_len >= width * height * 3:
            payload = contiguous_frame_buffer(raw_data, width * height * 3)
            rgb = np.frombuffer(payload, dtype=np.uint8, count=width * height * 3).reshape((height, width, 3))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if "bgr" in pixel_name and raw_len >= width * height * 3:
            payload = contiguous_frame_buffer(raw_data, width * height * 3)
            return np.frombuffer(payload, dtype=np.uint8, count=width * height * 3).reshape((height, width, 3)).copy()
        if ("mono16" in pixel_name or "mono12" in pixel_name or "mono10" in pixel_name) and raw_len >= width * height * 2:
            payload = contiguous_frame_buffer(raw_data, width * height * 2)
            gray16 = np.frombuffer(payload, dtype="<u2", count=width * height).reshape((height, width))
            bit_depth = max(int(getattr(frame, "raw_bit_depth", 16) or 16), 9)
            scale = 255.0 / float((1 << min(bit_depth, 16)) - 1)
            gray8 = np.clip(gray16.astype(np.float32) * scale, 0, 255).astype(np.uint8)
            return cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)
        if "mono8" in pixel_name and raw_len >= width * height:
            payload = contiguous_frame_buffer(raw_data, width * height)
            gray = np.frombuffer(payload, dtype=np.uint8, count=width * height).reshape((height, width))
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if "bayer" in pixel_name:
            bayer = self._raw_bayer_to_8bit(frame, raw_data, raw_len, width, height, pixel_name)
            if bayer is None:
                return None
            return cv2.cvtColor(bayer, self._bayer_color_code(pixel_name))
        return None

    def _raw_bayer_to_8bit(
        self,
        frame: CameraFrame,
        raw_data: bytes | bytearray | memoryview,
        raw_len: int,
        width: int,
        height: int,
        pixel_name: str,
    ) -> np.ndarray | None:
        if raw_len >= width * height * 2 and any(token in pixel_name for token in ("16", "12", "10")):
            payload = contiguous_frame_buffer(raw_data, width * height * 2)
            bayer16 = np.frombuffer(payload, dtype="<u2", count=width * height).reshape((height, width))
            bit_depth = max(int(getattr(frame, "raw_bit_depth", 16) or 16), 9)
            scale = 255.0 / float((1 << min(bit_depth, 16)) - 1)
            return np.clip(bayer16.astype(np.float32) * scale, 0, 255).astype(np.uint8)
        if raw_len >= width * height:
            payload = contiguous_frame_buffer(raw_data, width * height)
            return np.frombuffer(payload, dtype=np.uint8, count=width * height).reshape((height, width))
        return None

    def _bayer_color_code(self, pixel_name: str) -> int:
        if "bayerrg" in pixel_name:
            return cv2.COLOR_BayerRG2BGR
        if "bayerbg" in pixel_name:
            return cv2.COLOR_BayerBG2BGR
        if "bayergb" in pixel_name:
            return cv2.COLOR_BayerGB2BGR
        return cv2.COLOR_BayerGR2BGR

    def _record_loop_v2(self, config_snapshot: dict) -> None:
        config_snapshot = self._capture_priority_record_config(config_snapshot)
        camera_system = self._require_camera_system()
        record_dir = self._require_record_dir()
        capture_fps = optional_positive_fps(config_snapshot.get("record_fps", 5.0))
        output_fps_setting = config_float(config_snapshot, "record_output_fps_when_unlimited", 30.0)
        fps = capture_fps or max(output_fps_setting, 0.1)
        capture_mode = str(getattr(camera_system, "trigger_source", config_snapshot.get("trigger_source", ""))).strip().lower()
        continuous_capture = capture_mode in {"continuous", "freerun", "free-run", "free run", "off", "none", "trigger off", "no trigger"}
        interval, image_interval = effective_record_intervals(config_snapshot, capture_fps)
        acquisition_interval = 0.0 if continuous_capture else interval
        save_interval = 1.0 / fps if fps > 0 else 0.0
        ext = image_extension(config_snapshot)
        save_image_sequence = config_bool(config_snapshot, "record_save_image_sequence", False, False)
        output_plan = configured_record_outputs(config_snapshot, save_image_sequence)
        post_make_mp4 = bool(output_plan["post_make_mp4"])
        realtime_mp4_enabled = bool(output_plan["record_realtime_mp4"])
        if bool(output_plan["forced_realtime"]):
            LOGGER.warning("Recording has no enabled output; forcing realtime MP4 output.")
        if save_image_sequence:
            for side in ("left", "right"):
                (record_dir / self._record_segment_dir(side, 1)).mkdir(parents=True, exist_ok=True)

        queue_size = configured_record_queue_size(config_snapshot, fps)
        image_queue: Queue[dict | None] | None = Queue(maxsize=queue_size) if save_image_sequence else None
        video_queue: Queue[dict | None] | None = Queue(maxsize=queue_size) if realtime_mp4_enabled else None
        meta_writer = RecordMetaWriter(record_dir / "frames.meta.json", config_int(config_snapshot, "record_meta_flush_every", 32))
        writer_errors: list[Exception] = []
        writer_errors_lock = threading.Lock()
        video_outputs: dict[str, list[str]] = {"left": [], "right": []}
        next_time = time.perf_counter()
        next_save_time = time.perf_counter()
        record_preview_fps = max(config_float(config_snapshot, "record_preview_fps", 2.0), 0.1)
        next_preview_time = time.perf_counter()
        last_status_time = 0.0
        max_seconds = max(config_float(config_snapshot, "record_max_seconds", 0.0), 0.0)
        make_mp4_after = bool(output_plan["make_mp4_after"])
        use_realtime_mp4 = bool(output_plan["use_realtime_mp4"])
        mp4_generation = str(output_plan["mp4_generation"])
        if save_image_sequence and not post_make_mp4 and not realtime_mp4_enabled:
            LOGGER.info("MP4 generation disabled by auto_make_mp4=false; recording %s sequence only.", ext.upper())

        workers: list[threading.Thread] = []
        if save_image_sequence and image_queue is not None:
            workers.append(
                threading.Thread(
                    target=self._record_image_writer_loop,
                    args=(image_queue, meta_writer, image_interval, ext, writer_errors, writer_errors_lock, config_snapshot),
                    daemon=True,
                )
            )
        if use_realtime_mp4 and video_queue is not None:
            workers.append(
                threading.Thread(
                    target=self._record_video_writer_loop,
                    args=(
                        video_queue,
                        meta_writer,
                        fps,
                        interval,
                        video_outputs,
                        writer_errors,
                        writer_errors_lock,
                        config_snapshot,
                        not save_image_sequence,
                    ),
                    daemon=True,
                )
            )
        for worker in workers:
            worker.start()

        try:
            while self.recording:
                record_started_at = self._record_started_at_snapshot()
                if max_seconds > 0 and record_started_at is not None:
                    if time.perf_counter() - record_started_at >= max_seconds:
                        self._set_record_stop_reason("time_limit")
                        self.recording = False
                        break

                loop_start = time.perf_counter()
                display_this_frame = False
                if self.previewing:
                    now_for_preview = time.perf_counter()
                    display_this_frame, next_preview_time = record_preview_due(
                        now_for_preview, next_preview_time, record_preview_fps
                    )
                try:
                    convert_for_preview = display_this_frame or save_image_sequence
                    left, right, trigger_time = self._require_camera_system().capture_pair(convert_image=convert_for_preview)
                except FrameTimeoutError as exc:
                    consecutive_timeouts = self._record_timeout_observed()
                    self._handle_capture_exception(exc, "record", consecutive_timeouts)
                    next_time = time.perf_counter() + acquisition_interval if acquisition_interval > 0 else time.perf_counter()
                    continue
                except Exception as exc:
                    self._add_record_errors()
                    if self._handle_capture_exception(exc, "record", 0):
                        next_time = time.perf_counter() + acquisition_interval if acquisition_interval > 0 else time.perf_counter()
                        continue
                    raise
                left, right = self._correct_frame_pair(left, right)
                self._reset_record_timeouts()
                record_count = self._increment_record_count()
                self._record_capture_observed(record_count, trigger_time)
                self._record_frame_numbers_observed(record_count, trigger_time, left, right)
                with self._record_last_frame_lock:
                    self._record_last_frame_pair = (left, right, trigger_time)

                should_sample_for_output = True
                if continuous_capture and save_interval > 0:
                    now_for_sample = time.perf_counter()
                    should_sample_for_output = now_for_sample >= next_save_time
                    if should_sample_for_output:
                        while next_save_time <= now_for_sample:
                            next_save_time += save_interval

                if should_sample_for_output and self._should_record_save_frame(record_count):
                    saved_index, segment_index = self._next_saved_frame_index()
                    item = {
                        "index": record_count,
                        "saved_index": saved_index,
                        "segment_index": segment_index,
                        "trigger_time": trigger_time,
                        "left": left,
                        "right": right,
                    }
                    queued = False
                    if save_image_sequence and image_queue is not None:
                        clone_frames = config_bool(config_snapshot, "record_clone_frames_for_writer", False, False)
                        queued = self._put_record_item(image_queue, item, clone_frames=clone_frames) or queued
                    if use_realtime_mp4 and video_queue is not None:
                        queued = self._put_record_item(video_queue, item, clone_frames=False) or queued
                    if make_mp4_after and not queued:
                        self._record_skipped("record_output_queue_unavailable", record_count)
                else:
                    queued = False
                    self._record_skipped("write_skip_strategy", record_count)
                if not self.previewing and not save_image_sequence:
                    self._release_unsaved_record_frames(left, right, queued_for_writer=queued)

                if self.previewing:
                    self._preview_frame_counter += 1
                    if display_this_frame and self._should_analyze_preview_frame(self._preview_frame_counter, config_snapshot):
                        analysis = self._analyze_preview_frames(left, right, self._preview_frame_counter)
                        self.ui_queue.put(("quality_metrics", analysis))
                    if display_this_frame:
                        self.ui_queue.put(("frames", (left, right)))
                else:
                    self._update_stats(left, right)

                now = time.perf_counter()
                if now - last_status_time >= 0.5:
                    last_status_time = now
                    self.ui_queue.put(
                        (
                            "status",
                            self._record_status_text(capture_fps, self._effective_record_fps(capture_fps), config_snapshot),
                        )
                    )
                    self.ui_queue.put(("record_progress", None))
                    self._check_disk_space_during_recording(capture_fps or fps, config_snapshot)
                writer_error = self._first_writer_error(writer_errors, writer_errors_lock)
                if writer_error is not None:
                    raise writer_error

                if acquisition_interval > 0:
                    next_time += acquisition_interval
                    sleep_s = next_time - time.perf_counter()
                    if sleep_s > 0:
                        time.sleep(sleep_s)
                    elif time.perf_counter() - loop_start > acquisition_interval * 2:
                        next_time = time.perf_counter()
        except Exception as exc:
            self.ui_queue.put(("error", exc))
        finally:
            self.recording = False
            queues: list[Queue] = []
            if image_queue is not None:
                queues.append(image_queue)
            if video_queue is not None:
                queues.append(video_queue)
            self._stop_record_workers(tuple(queues), workers, config_snapshot)
            meta_writer.close()
            frames_snapshot = meta_writer.load()
            writer_errors_snapshot = self._writer_errors_snapshot(writer_errors, writer_errors_lock)
            if writer_errors_snapshot:
                self.ui_queue.put(("error", writer_errors_snapshot[0]))
                self._add_record_errors(len(writer_errors_snapshot))
            output_fps = self._record_output_fps(capture_fps or fps)
            generated_video_names = self._finalize_recording_videos(
                record_dir,
                output_fps,
                frames_snapshot,
                video_outputs,
                config_snapshot,
            )
            summary = self._build_record_summary(record_dir, capture_fps or 0.0, output_fps, frames_snapshot)
            reports = self._write_record_reports(record_dir, summary, frames_snapshot, config_snapshot)
            summary["record_reports"] = reports
            write_lag, _write_warning, skip_every_n, skip_keep_frames = self._record_write_state_snapshot()
            stats = self._record_stats_snapshot()
            camera_system = self.camera_system
            record_mode = str(config_snapshot.get("record_mode", "video"))
            meta = {
                "mode": record_mode,
                "fps": fps,
                "effective_video_fps": output_fps,
                "frame_count": stats["record_count"],
                "saved_frame_count": stats["saved_frame_count"],
                "skipped_frame_count": stats["skipped_frame_count"],
                "skipped_frames": stats["skipped_frames"],
                "skip_reasons": stats["skip_reasons"],
                "timeout_count": stats["timeout_count"],
                "error_count": stats["error_count"],
                "reconnect_count": stats["reconnect_count"],
                "disk_warning_count": stats["disk_warning_count"],
                "image_format": ext,
                "record_save_image_sequence": save_image_sequence,
                "video_format": "mp4" if (post_make_mp4 or realtime_mp4_enabled) else None,
                "video_codec": config_snapshot.get("video_codec", "mp4v"),
                "video_bitrate_kbps": config_int(config_snapshot, "video_bitrate_kbps", 8000),
                "video_quality_crf": config_int(config_snapshot, "video_quality_crf", 23),
                "video_preset": config_snapshot.get("video_preset", "medium"),
                "use_nvenc": config_bool(config_snapshot, "use_nvenc", False, False),
                "auto_make_mp4": post_make_mp4,
                "record_realtime_mp4": realtime_mp4_enabled,
                "mp4_generation": mp4_generation,
                "record_mode": record_mode,
                "record_split_interval_seconds": config_float(config_snapshot, "record_split_interval_seconds", 600.0),
                "record_split_size_gb": config_float(config_snapshot, "record_split_size_gb", 4.0),
                "record_max_seconds": max_seconds,
                "stop_reason": stats["stop_reason"],
                "write_lag": write_lag,
                "disk_write_benchmark": self._record_disk_benchmark,
                "skip_every_n": skip_every_n,
                "skip_keep_frames": skip_keep_frames,
                "left_videos": [str(path) for path in video_outputs["left"]],
                "right_videos": [str(path) for path in video_outputs["right"]],
                "pixel_format": config_snapshot.get("pixel_format", "Mono8"),
                "left_camera": asdict(camera_system.left_info) if camera_system and camera_system.left_info else None,
                "right_camera": asdict(camera_system.right_info) if camera_system and camera_system.right_info else None,
                "device_versions": dict(self._device_versions),
                "temperatures_c": dict(self._latest_temperatures),
                "stream_stats": dict(getattr(self, "_latest_stream_stats", {})),
                "temperature_samples": list(self._temperature_samples),
                "calibration": self.calibration.meta(),
                "camera_timestamp_offset_fixed": config_snapshot.get("camera_timestamp_offset_fixed"),
                "field_correction": dict(config_snapshot.get("field_correction", {}))
                if isinstance(config_snapshot.get("field_correction"), dict)
                else {},
                "dic_analysis": dict(config_snapshot.get("dic_analysis", {}))
                if isinstance(config_snapshot.get("dic_analysis"), dict)
                else {},
                "project": self.project_manager.project_meta(),
                "data_manifest": {
                    "manifest_csv": str(record_dir / "exports" / "file_manifest.csv"),
                    "summary_json": str(record_dir / "exports" / "capture_summary.json"),
                },
                "record_reports": reports,
                "frames": frames_snapshot,
                "summary": summary,
            }
            with (record_dir / "meta.json").open("w", encoding="utf-8") as fh:
                json.dump(meta, fh, ensure_ascii=False, indent=2)
            manifest = self._write_manifest_for_session(record_dir, summary, config_snapshot)
            self.project_manager.register_session(record_mode, record_dir, record_dir / "meta.json", {"manifest": manifest})
            self.ui_queue.put(("record_done", (record_dir, generated_video_names, summary)))

    def _reset_record_write_state(self) -> None:
        with self._state_lock:
            self._record_write_lag = 0.0
            self._record_write_warning = ""
            self._record_skip_every_n = 1
            self._record_skip_keep_frames = 1

    def _record_write_state_snapshot(self) -> tuple[float, str, int, int]:
        with self._state_lock:
            return (
                self._record_write_lag,
                self._record_write_warning,
                max(int(self._record_skip_every_n), 1),
                max(int(self._record_skip_keep_frames), 1),
            )

    def _reset_record_stats(self, started_at: float) -> None:
        with self._record_stats_lock:
            self.record_count = 0
            self.record_saved_count = 0
            self._record_next_saved_index = 0
            self.record_started_at = started_at
            self.record_started_wall_time = time.time()
            self.record_stop_reason = "manual"
            self._record_split_index = 1
            self._record_segment_start_time = started_at
            self._record_segment_start_saved = 0
            self._record_segment_sizes = {}
            self._record_video_segment_sizes = {}
            self._record_first_trigger_time = None
            self._record_second_stats = {}
            self._record_skipped_frames = []
            self._record_skipped_count = 0
            self._record_skip_reasons = {}
            self._record_timeout_count = 0
            self._record_consecutive_timeouts = 0
            self._record_error_count = 0
            self._record_reconnect_count = 0
            self._record_disk_warning_count = 0
            self._record_last_camera_frame_numbers = {}
            self._record_frame_number_gap_count = 0
            self._record_summary = {}

    def _record_stats_snapshot(self) -> dict[str, object]:
        with self._record_stats_lock:
            return {
                "record_count": self.record_count,
                "saved_frame_count": self.record_saved_count,
                "skipped_frame_count": self._record_skipped_count,
                "skipped_frames": list(self._record_skipped_frames),
                "skip_reasons": dict(self._record_skip_reasons),
                "timeout_count": self._record_timeout_count,
                "error_count": self._record_error_count,
                "reconnect_count": self._record_reconnect_count,
                "disk_warning_count": self._record_disk_warning_count,
                "frame_number_gap_count": self._record_frame_number_gap_count,
                "record_started_at": self.record_started_at,
                "record_started_wall_time": self.record_started_wall_time,
                "per_second": [
                    {
                        **{key: val for key, val in value.items() if key != "drop_reasons"},
                        "drop_reasons": dict(value.get("drop_reasons", {})),
                    }
                    for _key, value in sorted(self._record_second_stats.items())
                ],
                "stop_reason": self.record_stop_reason,
            }

    def _record_counter_values(self) -> tuple[int, int]:
        with self._record_stats_lock:
            return self.record_count, self.record_saved_count

    def _increment_record_count(self) -> int:
        with self._record_stats_lock:
            self.record_count += 1
            return self.record_count

    def _record_second_index_locked(self, trigger_time: object | None = None) -> int:
        if trigger_time not in (None, ""):
            try:
                value = float(trigger_time)
            except (TypeError, ValueError):
                value = None
            if value is not None:
                if self._record_first_trigger_time is None:
                    self._record_first_trigger_time = value
                return int(max(value - self._record_first_trigger_time, 0.0))
        if self.record_started_at is None:
            return 0
        return int(max(time.perf_counter() - self.record_started_at, 0.0))

    def _record_second_bucket_locked(self, second: int) -> dict[str, object]:
        bucket = self._record_second_stats.get(second)
        if bucket is None:
            bucket = {
                "second": int(second),
                "captured_frames": 0,
                "saved_frames": 0,
                "skipped_frames": 0,
                "timeout_count": 0,
                "error_count": 0,
                "first_frame_index": None,
                "last_frame_index": None,
                "first_saved_index": None,
                "last_saved_index": None,
                "saved_bytes": 0,
                "write_seconds_total": 0.0,
                "write_samples": 0,
                "frame_number_gaps": 0,
                "drop_reasons": {},
            }
            self._record_second_stats[second] = bucket
        return bucket

    def _record_capture_observed(self, frame_index: int, trigger_time: object | None) -> None:
        with self._record_stats_lock:
            second = self._record_second_index_locked(trigger_time)
            bucket = self._record_second_bucket_locked(second)
            bucket["captured_frames"] = int(bucket.get("captured_frames", 0) or 0) + 1
            if bucket.get("first_frame_index") is None:
                bucket["first_frame_index"] = int(frame_index)
            bucket["last_frame_index"] = int(frame_index)

    def _record_frame_numbers_observed(
        self,
        frame_index: int,
        trigger_time: object | None,
        left: CameraFrame | None,
        right: CameraFrame | None,
    ) -> None:
        with self._record_stats_lock:
            second = self._record_second_index_locked(trigger_time)
            bucket = self._record_second_bucket_locked(second)
            for side, frame in (("left", left), ("right", right)):
                if frame is None:
                    continue
                try:
                    current = int(frame.frame_number)
                except (TypeError, ValueError):
                    continue
                previous = self._record_last_camera_frame_numbers.get(side)
                if previous is not None:
                    gap = current - previous
                    if gap > 1:
                        missing = gap - 1
                        self._record_frame_number_gap_count += missing
                        bucket["frame_number_gaps"] = int(bucket.get("frame_number_gaps", 0) or 0) + missing
                        reasons = bucket.setdefault("drop_reasons", {})
                        if isinstance(reasons, dict):
                            key = f"{side}_frame_number_gap"
                            reasons[key] = int(reasons.get(key, 0) or 0) + missing
                        self._notify_warning(
                            f"record_{side}_frame_number_gap",
                            f"{side} camera frame number gap before record frame {frame_index}: {previous} -> {current}",
                            log_only=True,
                        )
                self._record_last_camera_frame_numbers[side] = current

    def _record_save_observed(
        self,
        saved_index: int,
        frame_index: int | None,
        trigger_time: object | None,
        bytes_written: int,
        write_seconds: float,
    ) -> None:
        with self._record_stats_lock:
            second = self._record_second_index_locked(trigger_time)
            bucket = self._record_second_bucket_locked(second)
            bucket["saved_frames"] = int(bucket.get("saved_frames", 0) or 0) + 1
            if bucket.get("first_saved_index") is None:
                bucket["first_saved_index"] = int(saved_index)
            bucket["last_saved_index"] = int(saved_index)
            bucket["saved_bytes"] = int(bucket.get("saved_bytes", 0) or 0) + max(int(bytes_written), 0)
            bucket["write_seconds_total"] = float(bucket.get("write_seconds_total", 0.0) or 0.0) + max(
                float(write_seconds), 0.0
            )
            bucket["write_samples"] = int(bucket.get("write_samples", 0) or 0) + 1
            if frame_index is not None and bucket.get("last_frame_index") is None:
                bucket["last_frame_index"] = int(frame_index)

    def _next_saved_frame_index(self) -> tuple[int, int]:
        with self._record_stats_lock:
            self._record_next_saved_index += 1
            return self._record_next_saved_index, self._record_split_index

    def _record_timeout_observed(self) -> int:
        with self._record_stats_lock:
            self._record_timeout_count += 1
            self._record_consecutive_timeouts += 1
            bucket = self._record_second_bucket_locked(self._record_second_index_locked())
            bucket["timeout_count"] = int(bucket.get("timeout_count", 0) or 0) + 1
            reasons = bucket.setdefault("drop_reasons", {})
            if isinstance(reasons, dict):
                reasons["timeout"] = int(reasons.get("timeout", 0) or 0) + 1
            return self._record_consecutive_timeouts

    def _reset_record_timeouts(self) -> None:
        with self._record_stats_lock:
            self._record_consecutive_timeouts = 0

    def _add_record_errors(self, count: int = 1) -> None:
        if count <= 0:
            return
        with self._record_stats_lock:
            self._record_error_count += count
            bucket = self._record_second_bucket_locked(self._record_second_index_locked())
            bucket["error_count"] = int(bucket.get("error_count", 0) or 0) + count
            reasons = bucket.setdefault("drop_reasons", {})
            if isinstance(reasons, dict):
                reasons["error"] = int(reasons.get("error", 0) or 0) + count

    def _add_record_reconnect(self) -> None:
        with self._record_stats_lock:
            self._record_reconnect_count += 1

    def _add_record_disk_warning(self) -> None:
        with self._record_stats_lock:
            self._record_disk_warning_count += 1

    def _set_record_stop_reason(self, reason: str) -> None:
        with self._record_stats_lock:
            self.record_stop_reason = reason

    def _record_started_at_snapshot(self) -> float | None:
        with self._record_stats_lock:
            return self.record_started_at

    def _mark_record_saved(
        self,
        saved_index: int,
        segment_index: int,
        bytes_written: int,
        frame_index: int | None = None,
        trigger_time: object | None = None,
        write_seconds: float = 0.0,
    ) -> None:
        with self._record_stats_lock:
            self.record_saved_count = max(self.record_saved_count, saved_index)
            self._record_segment_sizes[segment_index] = self._record_segment_sizes.get(segment_index, 0) + bytes_written
        self._record_save_observed(saved_index, frame_index, trigger_time, bytes_written, write_seconds)

    def _mark_record_video_saved(
        self,
        saved_index: int,
        segment_index: int,
        segment_bytes: int,
        frame_index: int | None = None,
        trigger_time: object | None = None,
        write_seconds: float = 0.0,
    ) -> None:
        with self._record_stats_lock:
            previous = self._record_video_segment_sizes.get(segment_index, 0)
            current = max(int(segment_bytes), previous)
            delta_bytes = max(current - previous, 0)
            self._record_video_segment_sizes[segment_index] = current
            self._record_segment_sizes[segment_index] = current
            self.record_saved_count = max(self.record_saved_count, saved_index)
        self._record_save_observed(saved_index, frame_index, trigger_time, delta_bytes, write_seconds)

    def _record_video_segment_bytes(self, segment_index: int, video_outputs: dict[str, list[str]] | None = None) -> int:
        total = 0
        for side in ("left", "right"):
            paths = list((video_outputs or {}).get(side, []))
            expected = self._record_segment_video_path(side, segment_index)
            if str(expected) not in paths:
                paths.append(str(expected))
            for raw_path in paths:
                path = Path(raw_path)
                if path.name != expected.name or not path.exists():
                    continue
                try:
                    total += path.stat().st_size
                except OSError:
                    continue
        return total

    def _estimate_video_segment_bytes(self, saved_index: int, segment_index: int, config_snapshot: dict) -> int:
        bitrate_kbps = max(config_int(config_snapshot, "video_bitrate_kbps", 8000), 1)
        with self._record_stats_lock:
            segment_first_saved = self._record_segment_start_saved + 1 if segment_index == self._record_split_index else saved_index
        frames_in_segment = max(int(saved_index) - int(segment_first_saved) + 1, 1)
        fps = configured_record_output_fps(config_snapshot)
        per_side_bytes = frames_in_segment * bitrate_kbps * 1000 / 8 / fps
        return max(int(per_side_bytes * 2), 1)

    def _record_segment_snapshot(self) -> tuple[int, float, int, Path | None]:
        with self._record_stats_lock:
            return (
                self._record_split_index,
                self._record_segment_start_time,
                self._record_segment_sizes.get(self._record_split_index, 0),
                self.record_dir,
            )

    def _advance_record_segment_state(self, current_segment_index: int) -> tuple[int, Path | None] | None:
        with self._record_stats_lock:
            if current_segment_index != self._record_split_index:
                return None
            self._record_split_index += 1
            self._record_segment_start_time = time.perf_counter()
            self._record_segment_start_saved = self.record_saved_count
            return self._record_split_index, self.record_dir

    def _set_record_write_warning(self, warning: str) -> None:
        with self._state_lock:
            self._record_write_warning = warning

    def _raise_record_write_lag(self, minimum_lag: float, warning: str | None = None) -> str:
        with self._state_lock:
            if warning is not None:
                self._record_write_warning = warning
            self._record_write_lag = max(self._record_write_lag, minimum_lag)
            self._update_record_skip_strategy_locked()
            return self._record_write_warning

    def _observe_record_write_lag(self, lag: float) -> None:
        with self._state_lock:
            self._record_write_lag = 0.85 * self._record_write_lag + 0.15 * lag if self._record_write_lag else lag
            self._update_record_skip_strategy_locked()

    def _append_writer_error(
        self,
        writer_errors: list[Exception],
        writer_errors_lock: threading.Lock,
        exc: Exception,
    ) -> None:
        with writer_errors_lock:
            writer_errors.append(exc)

    def _first_writer_error(
        self,
        writer_errors: list[Exception],
        writer_errors_lock: threading.Lock,
    ) -> Exception | None:
        with writer_errors_lock:
            return writer_errors[0] if writer_errors else None

    def _writer_errors_snapshot(
        self,
        writer_errors: list[Exception],
        writer_errors_lock: threading.Lock,
    ) -> list[Exception]:
        with writer_errors_lock:
            return list(writer_errors)

    def _stop_record_workers(
        self,
        queues: tuple[Queue, ...],
        workers: list[threading.Thread],
        config_snapshot: dict,
    ) -> None:
        timeout_s = max(config_float(config_snapshot, "record_writer_stop_timeout_seconds", 10.0), 1.0)
        for queue in queues:
            try:
                queue.put(None, timeout=1.0)
            except Full:
                self._drain_queue(queue, "queue_drain_on_stop")
                queue.put(None, timeout=1.0)
        deadline = time.perf_counter() + timeout_s
        for queue in queues:
            while getattr(queue, "unfinished_tasks", 0) > 0 and time.perf_counter() < deadline:
                time.sleep(0.05)
        for worker in workers:
            remaining = max(deadline - time.perf_counter(), 0.1)
            worker.join(timeout=remaining)
            if worker.is_alive():
                self._notify_warning(
                    "record_worker_stop_timeout",
                    f"录像写入线程 {worker.name or worker.ident} 停止超时，已继续收尾。",
                )
        for queue in queues:
            if getattr(queue, "unfinished_tasks", 0) > 0:
                self._drain_queue(queue, "queue_drain_after_stop_timeout")

    def _drain_queue(self, queue: Queue, reason: str) -> None:
        while True:
            try:
                item = queue.get_nowait()
            except Empty:
                return
            else:
                if isinstance(item, dict):
                    self._record_skipped(reason, int(item.get("index", 0) or 0))
                queue.task_done()

    def _get_record_writer_item(self, writer_queue: Queue, timeout_s: float = 0.2) -> dict | None:
        while True:
            try:
                item = writer_queue.get(timeout=timeout_s)
            except Empty:
                if not self.recording and writer_queue.empty():
                    return None
            else:
                if item is None:
                    writer_queue.task_done()
                    return None
                return item

    def _put_record_item(self, queue: Queue, item: dict, *, clone_frames: bool = True) -> bool:
        frame_index = int(item.get("index", 0) or 0)
        if not self.recording:
            self._record_skipped("recording_stopped_before_queue", frame_index)
            return False
        if queue.full():
            warning = self._raise_record_write_lag(2.1, "写入队列已满，已丢弃当前录像帧以保护内存")
            self._notify_warning("record_queue_full", warning)
            self._record_skipped("record_queue_full", frame_index)
            self.ui_queue.put(("record_progress", None))
            return False
        queued = dict(item)
        if clone_frames:
            queued["left"] = self._clone_frame(item.get("left"))
            queued["right"] = self._clone_frame(item.get("right"))
        try:
            config = getattr(self, "config", {})
            queue.put(queued, timeout=max(config_float(config, "record_queue_put_timeout_seconds", 0.05), 0.0))
            return True
        except Full:
            warning = self._raise_record_write_lag(2.1, "写入队列已满，已丢弃当前录像帧以保护内存")
            self._notify_warning("record_queue_full", warning)
            self._record_skipped("record_queue_full", frame_index)
            self.ui_queue.put(("record_progress", None))
            return False

    def _clone_frame(self, frame: CameraFrame | None) -> CameraFrame | None:
        if frame is None:
            return None
        return CameraFrame(
            image=frame.image.copy() if getattr(frame, "image", None) is not None else None,
            frame_number=frame.frame_number,
            width=frame.width,
            height=frame.height,
            host_timestamp=frame.host_timestamp,
            camera_timestamp=frame.camera_timestamp,
            raw_data=getattr(frame, "raw_data", None),
            raw_frame_len=int(getattr(frame, "raw_frame_len", 0) or 0),
            pixel_type=int(getattr(frame, "pixel_type", 0) or 0),
            pixel_type_name=str(getattr(frame, "pixel_type_name", "") or ""),
            raw_bit_depth=int(getattr(frame, "raw_bit_depth", 8) or 8),
            raw_array_shape=getattr(frame, "raw_array_shape", None),
        )

    def _record_image_writer_loop(
        self,
        writer_queue: Queue[dict | None],
        meta_writer: RecordMetaWriter,
        interval: float,
        ext: str,
        writer_errors: list[Exception],
        writer_errors_lock: threading.Lock,
        config_snapshot: dict,
    ) -> None:
        self._require_record_dir()
        batch_size = max(config_int(config_snapshot, "record_writer_batch_size", 4), 1)
        while True:
            first_item = self._get_record_writer_item(writer_queue)
            if first_item is None:
                return
            batch: list[dict] = [first_item]
            for _ in range(batch_size - 1):
                try:
                    next_item = writer_queue.get_nowait()
                except Empty:
                    break
                if next_item is None:
                    writer_queue.task_done()
                    for item in batch:
                        self._write_record_image_item(
                            item,
                            meta_writer,
                            interval,
                            ext,
                            writer_errors,
                            writer_errors_lock,
                            config_snapshot,
                        )
                        writer_queue.task_done()
                    return
                batch.append(next_item)
            for item in batch:
                self._write_record_image_item(
                    item,
                    meta_writer,
                    interval,
                    ext,
                    writer_errors,
                    writer_errors_lock,
                    config_snapshot,
                )
                writer_queue.task_done()

    def _write_record_image_item(
        self,
        item: dict,
        meta_writer: RecordMetaWriter,
        interval: float,
        ext: str,
        writer_errors: list[Exception],
        writer_errors_lock: threading.Lock,
        config_snapshot: dict,
    ) -> None:
        record_dir = self._require_record_dir()
        started = time.perf_counter()
        paths: dict[str, str | None] = {"left": None, "right": None}
        checksums: dict[str, str | None] = {"left": None, "right": None}
        bytes_written = 0
        checksum_during_capture = config_bool(config_snapshot, "record_checksum_during_capture", False, False)
        try:
            saved_index = int(item["saved_index"])
            segment_index = int(item["segment_index"])
            name = f"{saved_index:06d}.{ext}"
            for side in ("left", "right"):
                frame = item.get(side)
                if frame is None:
                    continue
                path = record_dir / self._record_segment_dir(side, segment_index) / f"{side}_{name}"
                path = self._save_frame(frame, path, config_snapshot)
                paths[side] = str(path)
                bytes_written += path.stat().st_size if path.exists() else frame_raw_estimated_bytes(frame)
                if checksum_during_capture:
                    checksums[side] = self._file_checksum(path, config_snapshot)
            elapsed = time.perf_counter() - started
            lag = elapsed / interval if interval > 0 else 0.0
            self._observe_record_write_lag(lag)
            self._mark_record_saved(
                saved_index,
                segment_index,
                bytes_written,
                frame_index=int(item.get("index", 0) or 0),
                trigger_time=item.get("trigger_time"),
                write_seconds=elapsed,
            )
            self._poll_camera_temperatures()
            meta_writer.append(
                {
                    "index": item["index"],
                    "saved_index": saved_index,
                    "segment_index": segment_index,
                    "trigger_time": item["trigger_time"],
                    "left_frame": self._frame_meta(item["left"]) if item.get("left") is not None else None,
                    "right_frame": self._frame_meta(item["right"]) if item.get("right") is not None else None,
                    "left_path": paths["left"],
                    "right_path": paths["right"],
                    "left_checksum": checksums["left"],
                    "right_checksum": checksums["right"],
                    "checksum_algorithm": self._checksum_algorithm(config_snapshot) if checksum_during_capture else None,
                    "temperatures_c": dict(self._latest_temperatures),
                    "write_seconds": elapsed,
                    "write_lag": lag,
                }
            )
            self._advance_record_segment_if_needed(segment_index, config_snapshot)
        except Exception as exc:
            self._append_writer_error(writer_errors, writer_errors_lock, exc)
            self.recording = False

    def _record_video_writer_loop(
        self,
        writer_queue: Queue[dict | None],
        meta_writer: RecordMetaWriter,
        fps: float,
        interval: float,
        video_outputs: dict[str, list[str]],
        writer_errors: list[Exception],
        writer_errors_lock: threading.Lock,
        config_snapshot: dict,
        write_frame_meta: bool = False,
    ) -> None:
        writers: dict[tuple[str, int], cv2.VideoWriter] = {}
        try:
            while True:
                item = self._get_record_writer_item(writer_queue)
                if item is None:
                    break
                try:
                    started = time.perf_counter()
                    segment_index = int(item["segment_index"])
                    saved_index = int(item["saved_index"])
                    paths: dict[str, str | None] = {"left": None, "right": None}
                    raw_paths: dict[str, str | None] = {"left": None, "right": None}
                    frame_meta: dict[str, dict | None] = {"left": None, "right": None}
                    raw_sidecar_bytes = 0
                    for side in ("left", "right"):
                        frame = item.get(side)
                        if frame is None:
                            continue
                        frame_meta[side] = self._frame_meta(frame)
                        key = (side, segment_index)
                        if key not in writers:
                            path = self._record_segment_video_path(side, segment_index)
                            writer, codec_name = self._create_video_writer_v2(path, fps, frame, config_snapshot)
                            writers[key] = writer
                            video_outputs[side].append(str(path))
                            if codec_name != str(config_snapshot.get("video_codec", "mp4v")):
                                self._set_record_write_warning(f"Video codec fallback: {codec_name}")
                        writers[key].write(self._frame_to_video_frame(frame))
                        paths[side] = str(self._record_segment_video_path(side, segment_index))
                        if self._should_save_raw_frame(frame, config_snapshot) and not config_bool(
                            config_snapshot, "record_force_image_format", False, False
                        ):
                            raw_path = (
                                self._require_record_dir()
                                / "raw"
                                / side
                                / f"part_{segment_index:03d}"
                                / f"{side}_{saved_index:06d}.{raw_frame_extension(config_snapshot)}"
                            )
                            raw_path.parent.mkdir(parents=True, exist_ok=True)
                            raw_path = self._save_raw_frame(frame, raw_path, config_snapshot)
                            raw_paths[side] = str(raw_path)
                            raw_sidecar_bytes += raw_path.stat().st_size if raw_path.exists() else frame_raw_estimated_bytes(frame)
                        if write_frame_meta and hasattr(frame, "release_raw_data"):
                            frame.release_raw_data()
                    elapsed = time.perf_counter() - started
                    lag = elapsed / interval if interval > 0 else 0.0
                    if write_frame_meta:
                        bytes_written = max(
                            self._record_video_segment_bytes(segment_index, video_outputs),
                            self._estimate_video_segment_bytes(saved_index, segment_index, config_snapshot),
                        ) + raw_sidecar_bytes
                        self._observe_record_write_lag(lag)
                        self._mark_record_video_saved(
                            saved_index,
                            segment_index,
                            bytes_written,
                            frame_index=int(item.get("index", 0) or 0),
                            trigger_time=item.get("trigger_time"),
                            write_seconds=elapsed,
                        )
                        meta_writer.append(
                            {
                                "index": item["index"],
                                "saved_index": saved_index,
                                "segment_index": segment_index,
                                "trigger_time": item["trigger_time"],
                                "left_frame": frame_meta["left"],
                                "right_frame": frame_meta["right"],
                                "left_path": None,
                                "right_path": None,
                                "left_video_path": paths["left"],
                                "right_video_path": paths["right"],
                                "left_raw_path": raw_paths["left"],
                                "right_raw_path": raw_paths["right"],
                                "left_checksum": None,
                                "right_checksum": None,
                                "checksum_algorithm": None,
                                "temperatures_c": dict(self._latest_temperatures),
                                "write_seconds": elapsed,
                                "write_lag": lag,
                            }
                        )
                        self._advance_record_segment_if_needed(segment_index, config_snapshot)
                except Exception as exc:
                    self._append_writer_error(writer_errors, writer_errors_lock, exc)
                    self.recording = False
                finally:
                    writer_queue.task_done()
        finally:
            for writer in writers.values():
                writer.release()

    def _create_video_writer_v2(
        self,
        path: Path,
        fps: float,
        frame_or_image: CameraFrame | Image.Image,
        config_snapshot: dict,
    ) -> tuple[cv2.VideoWriter, str]:
        if isinstance(frame_or_image, Image.Image):
            width, height = frame_or_image.size
        else:
            width = int(getattr(frame_or_image, "width", 0) or 0)
            height = int(getattr(frame_or_image, "height", 0) or 0)
            if (width <= 0 or height <= 0) and getattr(frame_or_image, "image", None) is not None:
                width, height = frame_or_image.image.size
        if width <= 0 or height <= 0:
            raise RuntimeError("invalid video frame size")
        codec = str(config_snapshot.get("video_codec", "mp4v")).strip() or "mp4v"
        candidates = self._opencv_fourcc_candidates(codec, config_bool(config_snapshot, "use_nvenc", False, False))
        for candidate, fourcc_text in candidates:
            fourcc = cv2.VideoWriter_fourcc(*fourcc_text)
            writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height), True)
            if writer.isOpened():
                return writer, f"{candidate}/{fourcc_text}"
            writer.release()
        raise RuntimeError(f"无法创建视频文件：{path}")

    def _opencv_fourcc_candidates(self, codec: str, use_nvenc: bool) -> list[tuple[str, str]]:
        requested = codec.strip().lower()
        candidates: list[tuple[str, str]] = []
        if use_nvenc or requested in {"h264", "h264_nvenc", "avc1", "libx264", "x264"}:
            candidates.extend(
                [
                    ("h264", "avc1"),
                    ("h264", "H264"),
                    ("h264", "X264"),
                ]
            )
        if requested and requested not in {"h264_nvenc", "libx264"}:
            fourcc = requested[:4]
            if len(fourcc) == 4:
                candidates.append((requested, fourcc))
        candidates.extend([("mp4v", "mp4v"), ("mpeg4", "FMP4"), ("mpeg4", "XVID")])
        unique: list[tuple[str, str]] = []
        seen: set[str] = set()
        for candidate in candidates:
            if candidate[1] in seen:
                continue
            seen.add(candidate[1])
            unique.append(candidate)
        return unique

    def choose_save_dir(self) -> None:
        selected = filedialog.askdirectory(title="选择保存路径", initialdir=str(resolve_output_root(self.config).resolve()))
        if not selected:
            return
        path = Path(selected)
        try:
            if path.is_relative_to(BASE_DIR):
                value = str(path.relative_to(BASE_DIR))
            else:
                value = str(path)
        except ValueError:
            value = str(path)
        self.config["save_dir"] = value
        self.project_manager = ProjectManager(resolve_output_root(self.config), self.config)
        if self.project_manager.enabled:
            self.project_manager.create_project()
            self.project_manager.sync_config(self.config)
            self._set_project_status_vars()
        else:
            self.save_dir_var.set(value)
        save_config(self.config)
        self.status_var.set(f"保存路径已设置：{self.save_dir_var.get()}")

    def reset_view(self) -> None:
        self.left_pane.reset_zoom()
        self.right_pane.reset_zoom()
        self.status_var.set("画面缩放已还原。")

    def _set_interval_lamp(self, color: str) -> None:
        self.interval_lamp.itemconfigure(self.interval_lamp_id, fill=color)

    def _flash_interval_lamp_green(self) -> None:
        if self._interval_lamp_after_id is not None:
            try:
                self.root.after_cancel(self._interval_lamp_after_id)
            except Exception:
                pass
            self._interval_lamp_after_id = None
        self._set_interval_lamp(SUCCESS_COLOR)

        def restore() -> None:
            self._interval_lamp_after_id = None
            if self._closing:
                return
            self._set_interval_lamp(DANGER_COLOR if self.interval_capturing else SUBTLE_TEXT_COLOR)

        self._interval_lamp_after_id = self.root.after(1000, restore)

    def _flash_shutter_feedback(self) -> None:
        self.left_pane.flash_shutter()
        self.right_pane.flash_shutter()

    def reset_photo_count(self) -> None:
        self.photo_count = 0
        self.photo_count_var.set("拍照次数 0")
        self.status_var.set("同步拍照计数已复位。")

    def _increment_photo_count(self) -> None:
        self.photo_count += 1
        self.photo_count_var.set(f"拍照次数 {self.photo_count}")

    def _set_recording_indicator(self, active: bool) -> None:
        self.left_pane.set_recording(active)
        self.right_pane.set_recording(active)

    def toggle_roi_edit_mode(self) -> None:
        self._set_roi_edit_mode(not self.roi_editing)
        if self.roi_editing:
            self.status_var.set("ROI 修改模式已开启：在任一画面中按住左键拖动框选 ROI。")
        else:
            self.status_var.set("ROI 修改模式已关闭：鼠标左键拖动画面平移。")

    def _set_roi_edit_mode(self, enabled: bool) -> None:
        self.roi_editing = enabled
        if enabled:
            self.focus_roi_editing = False
            if hasattr(self, "focus_roi_button"):
                self.focus_roi_button.configure(text="框选对焦ROI")
        self.edit_roi_button.configure(text="退出ROI" if enabled else "框选ROI")
        self.left_pane.set_roi_mode(enabled)
        self.right_pane.set_roi_mode(enabled)

    def toggle_focus_roi_edit_mode(self) -> None:
        self.focus_roi_editing = not self.focus_roi_editing
        if self.focus_roi_editing:
            self._set_roi_edit_mode(False)
            self.left_pane.set_roi_mode(True)
            self.right_pane.set_roi_mode(True)
            self.focus_roi_button.configure(text="退出对焦ROI")
            self.status_var.set("对焦 ROI 框选模式已开启：在任一画面拖拽框选对焦目标区域。")
        else:
            self.left_pane.set_roi_mode(False)
            self.right_pane.set_roi_mode(False)
            self.focus_roi_button.configure(text="框选对焦ROI")
            self.status_var.set("对焦 ROI 框选模式已关闭。")

    def set_roi_from_preview(self, roi: tuple[int, int, int, int], side: str = "left") -> None:
        if self.focus_roi_editing:
            x, y, width, height = roi
            source_frame = self._last_right_frame_obj if side == "right" else self._last_left_frame_obj
            image_width = source_frame.width if source_frame is not None else CAPTURE_WIDTH
            image_height = source_frame.height if source_frame is not None else CAPTURE_HEIGHT
            focus_roi = roi_from_pixels(x, y, width, height, image_width, image_height)
            self.config["focus_roi"] = focus_roi
            self._magnifier_roi_frac = focus_roi
            save_config(self.config)
            self.focus_roi_var.set(self._format_focus_roi())
            self._cached_focus_roi = clamp_roi_frac(focus_roi)
            self._cached_focus_roi_source = self.config.get("focus_roi")
            self._last_focus_overlay_key = None
            self._last_focus_overlay_time = 0.0
            self.focus_roi_editing = False
            self.left_pane.set_roi_mode(False)
            self.right_pane.set_roi_mode(False)
            self.focus_roi_button.configure(text="框选对焦ROI")
            self.status_var.set(
                f"已设置对焦 ROI：x={focus_roi['x_frac']:.2f}, y={focus_roi['y_frac']:.2f}, "
                f"w={focus_roi['w_frac']:.2f}, h={focus_roi['h_frac']:.2f}。"
            )
            return
        offset_x, offset_y, width, height = roi
        side = "right" if side == "right" else "left"
        if side == "left":
            self.left_roi_width_var.set(str(width))
            self.left_roi_height_var.set(str(height))
            self.left_roi_offset_x_var.set(str(offset_x))
            self.left_roi_offset_y_var.set(str(offset_y))
        else:
            self.right_roi_width_var.set(str(width))
            self.right_roi_height_var.set(str(height))
            self.right_roi_offset_x_var.set(str(offset_x))
            self.right_roi_offset_y_var.set(str(offset_y))
        label = "左" if side == "left" else "右"
        self.status_var.set(f"已从{label}相机预览框选 ROI：W={width}, H={height}, X={offset_x}, Y={offset_y}。")
        self._set_roi_edit_mode(False)
        if self.camera_system is not None:
            self.apply_roi_settings()

    def reset_roi_settings(self) -> None:
        self.left_roi_width_var.set(str(CAPTURE_WIDTH))
        self.left_roi_height_var.set(str(CAPTURE_HEIGHT))
        self.left_roi_offset_x_var.set("0")
        self.left_roi_offset_y_var.set("0")
        self.right_roi_width_var.set(str(CAPTURE_WIDTH))
        self.right_roi_height_var.set(str(CAPTURE_HEIGHT))
        self.right_roi_offset_x_var.set("0")
        self.right_roi_offset_y_var.set("0")
        self.left_pane.clear_roi_rectangle()
        self.right_pane.clear_roi_rectangle()
        self._set_roi_edit_mode(False)
        self.status_var.set(f"ROI 已还原为满幅：{CAPTURE_WIDTH}x{CAPTURE_HEIGHT}, X=0, Y=0。")
        if self.camera_system is not None:
            self.apply_roi_settings()

    def load_preset(self) -> None:
        presets = self.config.get("presets", {})
        preset = presets.get(self.preset_var.get())
        if not preset:
            self.status_var.set(f"未找到预设：{self.preset_var.get()}")
            return
        self.config.update(safe_trigger_config(preset))
        self._ensure_default_full_resolution()
        self._load_vars_from_config()
        save_config(self.config)
        self.status_var.set(f"已加载预设：{self.preset_var.get()}；连接相机后可应用参数。")

    def save_preset(self) -> None:
        try:
            preset = self._current_parameter_config()
        except ValueError:
            self.status_var.set("预设保存失败：参数必须是数字。")
            return
        presets = self.config.setdefault("presets", {})
        presets[self.preset_var.get()] = preset
        save_config(self.config)
        self.status_var.set(f"已保存预设：{self.preset_var.get()}")

    def apply_gain_settings(self) -> None:
        if self.camera_system is None:
            return
        try:
            gain_auto = self.gain_auto_var.get()
            gain = float(self.gain_var.get())
            lower = self._optional_entry_float(self.auto_gain_lower_var)
            upper = self._optional_entry_float(self.auto_gain_upper_var)
        except ValueError:
            self.status_var.set("增益参数必须是数字。")
            return
        self.apply_gain_button.configure(state=DISABLED)
        self.status_var.set("正在应用增益设置...")

        def worker() -> None:
            try:
                warnings = self.camera_system.apply_gain_settings(gain_auto, gain, lower, upper)
                self._update_config(
                    {
                        "gain_auto": gain_auto,
                        "gain": gain,
                        "auto_gain_lower_limit": lower,
                        "auto_gain_upper_limit": upper,
                    }
                )
                self.ui_queue.put(("status", self._format_apply_result("增益已应用", warnings)))
            except Exception as exc:
                self.ui_queue.put(("error", exc))
            finally:
                self.ui_queue.put(("param_idle", None))

        self._start_background_thread(worker, "apply-gain")

    def apply_exposure_settings(self) -> None:
        if self.camera_system is None:
            return
        try:
            exposure_auto = self.exposure_auto_var.get()
            exposure_time = float(self.exposure_time_var.get())
            lower = self._optional_entry_float(self.auto_exposure_lower_var)
            upper = self._optional_entry_float(self.auto_exposure_upper_var)
        except ValueError:
            self.status_var.set("曝光参数必须是数字。")
            return
        self.apply_exposure_button.configure(state=DISABLED)
        self.status_var.set("正在应用曝光设置...")

        def worker() -> None:
            try:
                warnings = self.camera_system.apply_exposure_settings(exposure_auto, exposure_time, lower, upper)
                self._update_config(
                    {
                        "exposure_auto": exposure_auto,
                        "exposure_time_us": exposure_time,
                        "auto_exposure_lower_limit": lower,
                        "auto_exposure_upper_limit": upper,
                    }
                )
                self.ui_queue.put(("status", self._format_apply_result("曝光已应用", warnings)))
            except Exception as exc:
                self.ui_queue.put(("error", exc))
            finally:
                self.ui_queue.put(("param_idle", None))

        self._start_background_thread(worker, "apply-exposure")

    def apply_white_balance_settings(self) -> None:
        if self.camera_system is None:
            return
        try:
            balance_auto = self.balance_auto_var.get()
            red = self._optional_entry_float(self.balance_red_var)
            green = self._optional_entry_float(self.balance_green_var)
            blue = self._optional_entry_float(self.balance_blue_var)
        except ValueError:
            self.status_var.set("白平衡参数必须是数字。")
            return
        self.apply_wb_button.configure(state=DISABLED)
        self.status_var.set("正在应用白平衡设置...")

        def worker() -> None:
            try:
                warnings = self.camera_system.apply_white_balance_settings(balance_auto, red, green, blue)
                self._update_config(
                    {
                        "balance_white_auto": balance_auto,
                        "balance_ratio_red": red,
                        "balance_ratio_green": green,
                        "balance_ratio_blue": blue,
                    }
                )
                self.ui_queue.put(("status", self._format_apply_result("白平衡已应用", warnings)))
            except Exception as exc:
                self.ui_queue.put(("error", exc))
            finally:
                self.ui_queue.put(("param_idle", None))

        self._start_background_thread(worker, "apply-white-balance")

    def apply_image_correction_settings(self) -> None:
        if self.camera_system is None:
            return
        try:
            black_level = self._optional_entry_float(self.black_level_var)
            digital_shift = self._optional_entry_float(self.digital_shift_var)
            gamma = self._optional_entry_float(self.gamma_var)
        except ValueError:
            self.status_var.set("图像校正参数必须是数字或留空。")
            return
        self.apply_correction_button.configure(state=DISABLED)
        self.status_var.set("正在应用图像校正设置...")

        def worker() -> None:
            try:
                warnings = self.camera_system.apply_image_correction_settings(black_level, digital_shift, gamma)
                self._update_config(
                    {
                        "black_level": black_level,
                        "digital_shift": digital_shift,
                        "gamma": gamma,
                    }
                )
                self.ui_queue.put(("status", self._format_apply_result("图像校正已应用", warnings)))
            except Exception as exc:
                self.ui_queue.put(("error", exc))
            finally:
                self.ui_queue.put(("param_idle", None))

        self._start_background_thread(worker, "apply-image-correction")

    def _field_correction_status_text(self) -> str:
        section = self.config.get("field_correction", {})
        if not isinstance(section, dict):
            return "Field correction --"
        enabled = "on" if config_bool(section, "enabled", False, False) else "off"
        dark = "dark" if section.get("dark_frame_path") else "no dark"
        flat = "flat" if section.get("flat_field_path") else "no flat"
        return f"Field correction {enabled} | {dark} | {flat}"

    def _sync_field_correction_enabled(self) -> None:
        section = self._ensure_config_section("field_correction")
        section["enabled"] = bool(self.field_correction_enabled_var.get())
        save_config(self.config)
        self._load_field_correction_references()
        self.field_correction_status_var.set(self._field_correction_status_text())

    def _field_correction_dir(self) -> Path:
        path = BASE_DIR / "calib" / "field_correction"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _load_field_correction_references(self) -> None:
        section = self.config.get("field_correction", {})
        if not isinstance(section, dict):
            return
        refs: dict[str, dict[str, np.ndarray]] = {"dark": {}, "flat": {}}
        for kind, key in (("dark", "dark_frame_path"), ("flat", "flat_field_path")):
            value = str(section.get(key, "") or "").strip()
            if not value:
                continue
            path = Path(value)
            if not path.is_absolute():
                path = BASE_DIR / path
            if not path.exists():
                continue
            try:
                with np.load(path) as data:
                    for side in ("left", "right"):
                        if side in data:
                            refs[kind][side] = np.asarray(data[side], dtype=np.float32)
            except Exception:
                LOGGER.exception("field correction reference load failed: %s", path)
        with self._field_correction_lock:
            self._dark_frame_refs = refs["dark"]
            self._flat_field_refs = refs["flat"]

    def _relative_to_base(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(BASE_DIR.resolve()))
        except ValueError:
            return str(path)

    def _capture_field_reference_arrays(self, sample_count: int) -> dict[str, np.ndarray]:
        sums: dict[str, np.ndarray] = {}
        counts: dict[str, int] = {}
        timeout_ms = self._preview_capture_timeout_ms()
        for _ in range(max(int(sample_count), 1)):
            left, right, _trigger_time = self._require_camera_system().capture_pair(timeout_ms=timeout_ms, convert_image=True)
            for side, frame in (("left", left), ("right", right)):
                array = self._frame_to_correction_array(frame)
                if array is None:
                    continue
                if side not in sums:
                    sums[side] = np.zeros_like(array, dtype=np.float64)
                    counts[side] = 0
                if sums[side].shape != array.shape:
                    raise MvsError(f"{side} field correction frame size changed during reference capture")
                sums[side] += array.astype(np.float64, copy=False)
                counts[side] += 1
        if not sums:
            raise MvsError("未采集到可用于场校正的图像帧")
        return {side: (array / max(counts[side], 1)).astype(np.float32) for side, array in sums.items()}

    def _save_field_reference(self, kind: str, arrays: dict[str, np.ndarray]) -> Path:
        path = self._field_correction_dir() / f"{kind}_{time.strftime('%Y%m%d_%H%M%S')}.npz"
        np.savez_compressed(path, **arrays)
        section = self._ensure_config_section("field_correction")
        path_key = "dark_frame_path" if kind == "dark" else "flat_field_path"
        section[path_key] = self._relative_to_base(path)
        save_config(self.config)
        self._load_field_correction_references()
        if hasattr(self, "field_correction_status_var"):
            self.field_correction_status_var.set(self._field_correction_status_text())
        return path

    def _capture_field_reference_async(self, kind: str) -> None:
        if self.camera_system is None:
            return
        section = self._ensure_config_section("field_correction")
        sample_count = max(config_int(section, "sample_count", 16), 1)
        if hasattr(self, "dark_frame_button"):
            self.dark_frame_button.configure(state=DISABLED)
        if hasattr(self, "flat_field_button"):
            self.flat_field_button.configure(state=DISABLED)
        self.status_var.set(("正在采集暗场参考，请盖上镜头盖..." if kind == "dark" else "正在采集平场参考，请对准均匀光源..."))

        def worker() -> None:
            try:
                arrays = self._capture_field_reference_arrays(sample_count)
                path = self._save_field_reference(kind, arrays)
                label = "暗场" if kind == "dark" else "平场"
                self.ui_queue.put(("status", f"{label}参考已保存：{path}"))
            except Exception as exc:
                self.ui_queue.put(("error", exc))
            finally:
                self.ui_queue.put(("param_idle", None))

        self._start_background_thread(worker, f"capture-{kind}-field")

    def capture_dark_frame_reference(self) -> None:
        self._capture_field_reference_async("dark")

    def capture_flat_field_reference(self) -> None:
        self._capture_field_reference_async("flat")

    def calibrate_camera_timestamp_offset(self) -> None:
        if self.camera_system is None:
            return
        if hasattr(self, "timestamp_offset_button"):
            self.timestamp_offset_button.configure(state=DISABLED)
        self.status_var.set("正在计算 CameraTimestamp Offset...")

        def worker() -> None:
            try:
                offset = self.camera_system.calibrate_camera_timestamp_offset()
                self._update_config({"camera_timestamp_offset_fixed": offset})
                self.ui_queue.put(("timestamp_offset", offset))
                self.ui_queue.put(("status", f"CameraTimestamp Offset 已固定为 {offset}"))
            except Exception as exc:
                self.ui_queue.put(("error", exc))
            finally:
                self.ui_queue.put(("param_idle", None))

        self._start_background_thread(worker, "camera-timestamp-offset")

    def clear_camera_timestamp_offset(self) -> None:
        if self.camera_system is not None and hasattr(self.camera_system, "set_camera_timestamp_offset"):
            self.camera_system.set_camera_timestamp_offset(None)
        self._update_config({"camera_timestamp_offset_fixed": None})
        self.timestamp_offset_var.set("相机时基差 --")
        self.status_var.set("相机时间偏置已清除。")

    def apply_roi_settings(self, restart_stream: bool = True) -> None:
        if self.camera_system is None:
            return
        try:
            rois = {
                "left": (
                    optional_int_text(self.left_roi_width_var.get()) or CAPTURE_WIDTH,
                    optional_int_text(self.left_roi_height_var.get()) or CAPTURE_HEIGHT,
                    int(self.left_roi_offset_x_var.get() or 0),
                    int(self.left_roi_offset_y_var.get() or 0),
                ),
                "right": (
                    optional_int_text(self.right_roi_width_var.get()) or CAPTURE_WIDTH,
                    optional_int_text(self.right_roi_height_var.get()) or CAPTURE_HEIGHT,
                    int(self.right_roi_offset_x_var.get() or 0),
                    int(self.right_roi_offset_y_var.get() or 0),
                ),
            }
        except ValueError:
            self.status_var.set("ROI 参数必须是整数。")
            return
        self.apply_roi_button.configure(state=DISABLED)
        self.status_var.set("正在应用 ROI 设置...")

        def worker() -> None:
            try:
                results, warnings = self.camera_system.apply_side_roi_settings(rois, restart_stream=restart_stream)
                actual: dict[str, tuple[int, int, int, int]] = {}
                for side, requested in rois.items():
                    result = results.get(side)
                    actual[side] = result.actual_roi if result is not None and result.actual_roi else requested
                left_width, left_height, left_offset_x, left_offset_y = actual["left"]
                right_width, right_height, right_offset_x, right_offset_y = actual["right"]
                self._update_config(
                    {
                        "roi_width": left_width,
                        "roi_height": left_height,
                        "roi_offset_x": left_offset_x,
                        "roi_offset_y": left_offset_y,
                        "left_roi_width": left_width,
                        "left_roi_height": left_height,
                        "left_roi_offset_x": left_offset_x,
                        "left_roi_offset_y": left_offset_y,
                        "right_roi_width": right_width,
                        "right_roi_height": right_height,
                        "right_roi_offset_x": right_offset_x,
                        "right_roi_offset_y": right_offset_y,
                    }
                )
                self.ui_queue.put(("roi_applied", (actual, list(warnings))))
            except Exception as exc:
                self.ui_queue.put(("error", exc))
            finally:
                self.ui_queue.put(("param_idle", None))

        self._start_background_thread(worker, "apply-roi")

    def apply_trigger_settings(self) -> None:
        if self.camera_system is None:
            return
        requested_source = canonical_trigger_source(self.trigger_source_var.get())
        trigger_source = safe_capture_trigger_source(requested_source)
        if requested_source != trigger_source:
            self.trigger_source_var.set(display_trigger_source(trigger_source))
            self.status_var.set("硬触发级联和外部硬触发已禁用，当前仅支持软触发和连续采集；已切换为软触发。")
            self._update_config({"trigger_source": trigger_source})
            self._set_cached_trigger_source(trigger_source)
            return
        self._set_cached_trigger_source(trigger_source)
        self.apply_trigger_button.configure(state=DISABLED)
        if trigger_source == "Continuous":
            self.status_var.set("正在应用连续采集；相机会按曝光和带宽能力自由运行。")
        else:
            self.status_var.set("正在应用软触发模式...")

        def worker() -> None:
            try:
                warnings = self.camera_system.apply_trigger_settings(trigger_source)
                self._update_config({"trigger_source": trigger_source})
                self._set_cached_trigger_source(trigger_source)
                message = self._format_apply_result("触发模式已应用", warnings)
                if trigger_source == "Continuous":
                    message += "；连续采集关闭帧触发，预览/录像读取相机连续输出的最新帧。"
                self.ui_queue.put(("status", message))
            except Exception as exc:
                self.ui_queue.put(("error", exc))
            finally:
                self.ui_queue.put(("param_idle", None))

        self._start_background_thread(worker, "apply-trigger")

    def process_ui_queue(self) -> None:
        with self._ui_queue_event_lock:
            self._ui_queue_event_pending = False
        pending_frames: object | None = None
        pending_quality_metrics: object | None = None
        deferred: deque[tuple[str, object]] = deque()
        try:
            while True:
                kind, payload = self.ui_queue.get_nowait()
                if kind == "frames":
                    pending_frames = payload
                    continue
                if kind == "quality_metrics":
                    pending_quality_metrics = payload
                    continue
                deferred.append((kind, payload))
        except Empty:
            pass

        if pending_quality_metrics is not None:
            self._apply_quality_metrics(pending_quality_metrics)
        if pending_frames is not None:
            left, right = pending_frames
            self._display_frames(left, right)

        try:
            while deferred:
                kind, payload = deferred.popleft()
                if kind == "status":
                    self.status_var.set(str(payload))
                elif kind == "devices_refreshed":
                    _cameras, message = payload
                    self._last_device_status = str(message)
                    self.status_var.set(str(message))
                elif kind == "connected":
                    left_info, right_info = payload
                    self._update_connected_titles(left_info, right_info)
                    self._set_capture_buttons(NORMAL)
                    self._set_parameter_buttons(NORMAL)
                    count = self._connected_camera_count()
                    mode_text = "单相机" if count == 1 else "双相机"
                    connected_names = []
                    if left_info is not None:
                        connected_names.append(f"左: {left_info.label}")
                    if right_info is not None:
                        connected_names.append(f"右: {right_info.label}")
                    self._last_device_status = f"当前已连接 {count} 台相机：" + "；".join(connected_names)
                    self.status_var.set(
                        f"{mode_text}连接成功。当前 ROI：左 {self.config.get('left_roi_width', self.config.get('roi_width'))}x{self.config.get('left_roi_height', self.config.get('roi_height'))}；右 {self.config.get('right_roi_width', self.config.get('roi_width'))}x{self.config.get('right_roi_height', self.config.get('roi_height'))}。"
                    )
                    if count >= 2 and config_bool(self.config, "timestamp_reject_enabled", False, False):
                        cam_delta = int(self.config.get("max_camera_timestamp_delta", 0) or 0)
                        host_delta = int(self.config.get("max_host_timestamp_delta", 0) or 0)
                        if cam_delta <= 0 and host_delta <= 0:
                            self._notify_warning(
                                "timestamp_threshold_disabled",
                                "时间戳同步校验已开启，但阈值为 0，当前不会拒绝不同步帧。",
                                log_only=True,
                            )
                            self.status_var.set(
                                "双相机连接成功；时间戳同步阈值为 0，当前仅显示帧差，不会拒绝不同步帧。"
                            )
                    self._start_focus_reference_check()
                    self._schedule_focus_drift_check()
                elif kind == "connect_failed":
                    self.connect_button.configure(state=NORMAL)
                elif kind == "temperature":
                    self._update_temperature_display(payload if isinstance(payload, dict) else {})
                elif kind == "focus_reference_check":
                    self._handle_focus_reference_check(payload)
                elif kind == "epipolar_done":
                    self._handle_epipolar_result(payload)
                elif kind == "focus_ref_done":
                    self.status_var.set(str(payload))
                elif kind == "quality_report":
                    self._apply_quality_report(payload)
                elif kind == "calibration_board":
                    self._apply_calibration_board(payload)
                elif kind == "photo_quality_prefetched":
                    self._handle_photo_quality_prefetched(payload)
                elif kind == "record_stats":
                    left, right = payload
                    self._update_stats(left, right)
                    self._update_performance_display()
                elif kind == "preview_stats_reset":
                    self._reset_stats()
                elif kind == "record_progress":
                    self._update_performance_display()
                elif kind == "mp4_progress":
                    if isinstance(payload, dict):
                        percent = float(payload.get("percent", 0.0) or 0.0)
                        message = str(payload.get("message", "MP4 --"))
                        self._set_mp4_progress(percent, message, visible=True)
                elif kind == "timestamp_offset":
                    try:
                        formatted = self._format_camera_timestamp_offset(int(payload))
                    except (TypeError, ValueError):
                        formatted = f"Camera offset {payload}"
                    self.timestamp_offset_var.set(formatted)
                elif kind == "shutter_flash":
                    self._flash_shutter_feedback()
                elif kind == "photo_done":
                    mode, path = payload if isinstance(payload, tuple) and len(payload) == 2 else ("photo", payload)
                    if mode == "photo":
                        self._increment_photo_count()
                    self.status_var.set(f"拍照完成：{path}")
                elif kind == "calibration_wizard_capture_done":
                    refresh, button, position = payload
                    if callable(refresh):
                        refresh()
                    try:
                        button.configure(state=NORMAL)
                    except Exception:
                        pass
                    self.status_var.set(f"标定样本已采集：{position}")
                elif kind == "calibration_wizard_capture_idle":
                    try:
                        payload.configure(state=NORMAL)
                    except Exception:
                        pass
                elif kind == "calibration_wizard_done":
                    result, export_info, result_var, capture_button, calibrate_button, refresh = payload
                    self._calibration_wizard["running"] = False
                    self.reload_calibration()
                    stereo = result.get("stereo", {}) if isinstance(result, dict) else {}
                    left = result.get("left", {}) if isinstance(result, dict) else {}
                    right = result.get("right", {}) if isinstance(result, dict) else {}
                    stereo_rms = float(stereo.get("rms_reprojection_error_px", 0.0) or 0.0) if isinstance(stereo, dict) else 0.0
                    baseline = float(stereo.get("baseline_mm", 0.0) or 0.0) if isinstance(stereo, dict) else 0.0
                    left_rms = float(left.get("rms_reprojection_error_px", 0.0) or 0.0) if isinstance(left, dict) else 0.0
                    right_rms = float(right.get("rms_reprojection_error_px", 0.0) or 0.0) if isinstance(right, dict) else 0.0
                    result_var.set(
                        f"重投影误差：左 {left_rms:.4f}px，右 {right_rms:.4f}px，双目 {stereo_rms:.4f}px；基线 {baseline:.3f} mm"
                    )
                    try:
                        capture_button.configure(state=NORMAL)
                        calibrate_button.configure(state=NORMAL)
                    except Exception:
                        pass
                    if callable(refresh):
                        refresh()
                    self.status_var.set(f"在线标定完成，已导出到 {export_info.get('directory', BASE_DIR / 'calib')}")
                elif kind == "calibration_wizard_failed":
                    result_var, capture_button, calibrate_button, refresh = payload
                    self._calibration_wizard["running"] = False
                    result_var.set("标定失败，请查看错误信息或补采更多角度。")
                    try:
                        capture_button.configure(state=NORMAL)
                        calibrate_button.configure(state=NORMAL)
                    except Exception:
                        pass
                    if callable(refresh):
                        refresh()
                elif kind == "roi_applied":
                    self._reset_stats()
                    if isinstance(payload, tuple) and len(payload) == 2 and isinstance(payload[0], dict):
                        actual, warnings = payload
                        left = actual.get("left", (CAPTURE_WIDTH, CAPTURE_HEIGHT, 0, 0))
                        right = actual.get("right", (CAPTURE_WIDTH, CAPTURE_HEIGHT, 0, 0))
                        self.left_roi_width_var.set(str(left[0]))
                        self.left_roi_height_var.set(str(left[1]))
                        self.left_roi_offset_x_var.set(str(left[2]))
                        self.left_roi_offset_y_var.set(str(left[3]))
                        self.right_roi_width_var.set(str(right[0]))
                        self.right_roi_height_var.set(str(right[1]))
                        self.right_roi_offset_x_var.set(str(right[2]))
                        self.right_roi_offset_y_var.set(str(right[3]))
                        self.status_var.set(
                            self._format_apply_result(
                                f"实际应用 ROI：左 W={left[0]},H={left[1]},X={left[2]},Y={left[3]}；右 W={right[0]},H={right[1]},X={right[2]},Y={right[3]}",
                                list(warnings),
                            )
                        )
                    else:
                        width, height, offset_x, offset_y, warnings = payload
                        self.left_roi_width_var.set(str(width))
                        self.left_roi_height_var.set(str(height))
                        self.left_roi_offset_x_var.set(str(offset_x))
                        self.left_roi_offset_y_var.set(str(offset_y))
                        self.right_roi_width_var.set(str(width))
                        self.right_roi_height_var.set(str(height))
                        self.right_roi_offset_x_var.set(str(offset_x))
                        self.right_roi_offset_y_var.set(str(offset_y))
                        self.status_var.set(
                            self._format_apply_result(
                                f"实际应用 ROI：W={width}, H={height}, X={offset_x}, Y={offset_y}",
                                warnings,
                            )
                        )
                elif kind == "capture_idle":
                    if self.camera_system is not None and not self.interval_capturing:
                        self._set_capture_buttons(NORMAL)
                elif kind == "dic_start":
                    config_snapshot, warnings = payload if isinstance(payload, tuple) and len(payload) == 2 else (self._dic_capture_config(), [])
                    self._set_parameter_buttons(NORMAL)
                    started = self._start_record_session(
                        dict(config_snapshot),
                        mode=DIC_CAPTURE_MODE,
                        status_prefix="DIC 图像采集中",
                    )
                    if not started:
                        self._dic_recording = False
                        if hasattr(self, "dic_capture_button"):
                            self.dic_capture_button.configure(text="DIC采集")
                        if self.camera_system is not None:
                            self._set_capture_buttons(NORMAL)
                        continue
                    if warnings:
                        self.status_var.set("DIC 参数已应用；" + "；".join(str(item) for item in warnings))
                elif kind == "dic_start_failed":
                    self._dic_recording = False
                    if hasattr(self, "dic_capture_button"):
                        self.dic_capture_button.configure(text="DIC采集", state=NORMAL if self.camera_system is not None else DISABLED)
                    self._set_parameter_buttons(NORMAL)
                    if self.camera_system is not None:
                        self._set_capture_buttons(NORMAL)
                elif kind == "interval_done":
                    self.interval_capturing = False
                    self.interval_button.configure(text="定时拍照")
                    self.preview_button.configure(text="停止采集" if self.previewing else "开始采集")
                    if self._interval_lamp_after_id is None:
                        self._set_interval_lamp(SUBTLE_TEXT_COLOR)
                    if self.camera_system is not None and not self.recording:
                        self._set_capture_buttons(NORMAL)
                    if not payload:
                        self.status_var.set(f"定时拍照已停止，共保存 {self.interval_count} 组。")
                elif kind == "interval_lamp_green":
                    self._flash_interval_lamp_green()
                elif kind == "preview_done":
                    had_error, generation = payload
                    with self._state_lock:
                        if generation != self._preview_generation:
                            continue
                    should_continue_preview = bool(had_error and self.previewing and self.camera_system is not None)
                    if self.recording or self.interval_capturing:
                        self.preview_button.configure(text="停止采集" if self.previewing else "开始采集")
                        if self.camera_system is not None:
                            self._set_capture_buttons(NORMAL)
                    elif should_continue_preview:
                        self.preview_button.configure(text="停止采集")
                        self._set_capture_buttons(NORMAL)
                        self._start_preview_thread()
                    else:
                        self.previewing = False
                        self.preview_button.configure(text="开始采集")
                    if self.camera_system is not None and not self.recording and not self.interval_capturing:
                        self._set_capture_buttons(NORMAL)
                        if not had_error:
                            self.status_var.set("实时采集已停止。")
                elif kind == "record_done":
                    record_dir, video_names, summary = payload
                    was_dic = self._dic_recording
                    self._dic_recording = False
                    self.recording = False
                    if not self._resume_preview_after_recording:
                        self.previewing = False
                    self._hide_mp4_progress()
                    self.record_button.configure(text="开始录像")
                    if hasattr(self, "dic_capture_button"):
                        self.dic_capture_button.configure(text="DIC采集")
                    self.preview_button.configure(text="停止采集" if self.previewing else "开始采集")
                    self._set_recording_indicator(False)
                    self._set_capture_buttons(NORMAL)
                    self._last_video_sides = list(video_names)
                    self._ensure_preview_thread_after_recording()
                    videos = "、".join(self._last_video_sides) if self._last_video_sides else "未生成 MP4，仅保存 BMP 序列"
                    report_note = ""
                    if isinstance(summary, dict):
                        reports = summary.get("record_reports")
                        if isinstance(reports, dict) and reports.get("html"):
                            report_note = f"；报告 {Path(str(reports['html'])).name}"
                    done_label = "DIC 图像采集完成" if was_dic else "录像完成"
                    self.status_var.set(f"{done_label}：{record_dir}，{videos}；{self._format_record_summary(summary)}{report_note}")
                elif kind == "param_idle":
                    if self.camera_system is not None:
                        self._set_parameter_buttons(NORMAL)
                    if hasattr(self, "field_correction_status_var"):
                        self.field_correction_status_var.set(self._field_correction_status_text())
                elif kind == "error":
                    self._show_error(payload)
        except Empty:
            pass

    def _notify_ui_queue(self) -> None:
        with self._ui_queue_event_lock:
            if self._ui_queue_event_pending:
                return
            self._ui_queue_event_pending = True
        try:
            self.root.event_generate(UI_QUEUE_EVENT, when="tail")
        except Exception:
            try:
                self.root.after(0, self.process_ui_queue)
            except Exception:
                pass

    def _schedule_ui_queue_fallback(self) -> None:
        self.process_ui_queue()
        self._ui_queue_fallback_after_id = self.root.after(250, self._schedule_ui_queue_fallback)

    def _set_capture_buttons(self, state: str) -> None:
        def set_dic_options_enabled(enabled: bool) -> None:
            if hasattr(self, "dic_record_fps_entry"):
                self.dic_record_fps_entry.configure(state=NORMAL if enabled else DISABLED)
            if hasattr(self, "dic_pixel_format_menu"):
                self.dic_pixel_format_menu.configure(state=NORMAL if enabled else DISABLED)

        if self.camera_system is None:
            self.connect_button.configure(state=NORMAL)
            self.preview_button.configure(state=DISABLED)
            self.photo_button.configure(state=DISABLED)
            self.hdr_button.configure(state=DISABLED)
            self.interval_button.configure(state=DISABLED)
            self.record_button.configure(state=DISABLED)
            if hasattr(self, "dic_capture_button"):
                self.dic_capture_button.configure(state=DISABLED)
            set_dic_options_enabled(False)
            self.record_preflight_button.configure(state=NORMAL)
            self.calibration_wizard_button.configure(state=NORMAL)
            return

        self.connect_button.configure(state=DISABLED)
        self.record_preflight_button.configure(state=DISABLED if self.recording or self.interval_capturing else NORMAL)
        self.calibration_wizard_button.configure(state=DISABLED if self.recording or self.interval_capturing else NORMAL)
        if self.recording:
            self.preview_button.configure(state=state)
            self.photo_button.configure(state=state)
            self.hdr_button.configure(state=DISABLED)
            self.interval_button.configure(state=DISABLED)
            self.record_button.configure(state=DISABLED if self._dic_recording else state)
            if hasattr(self, "dic_capture_button"):
                self.dic_capture_button.configure(state=state if self._dic_recording else DISABLED)
            set_dic_options_enabled(False)
        elif self.interval_capturing:
            self.preview_button.configure(state=state)
            self.photo_button.configure(state=DISABLED)
            self.hdr_button.configure(state=DISABLED)
            self.interval_button.configure(state=state)
            self.record_button.configure(state=DISABLED)
            if hasattr(self, "dic_capture_button"):
                self.dic_capture_button.configure(state=DISABLED)
            set_dic_options_enabled(False)
        elif self.previewing:
            self.preview_button.configure(state=state)
            self.photo_button.configure(state=state)
            self.hdr_button.configure(state=state)
            self.interval_button.configure(state=state)
            self.record_button.configure(state=state)
            if hasattr(self, "dic_capture_button"):
                self.dic_capture_button.configure(state=state)
            set_dic_options_enabled(True)
        else:
            self.preview_button.configure(state=state)
            self.photo_button.configure(state=state)
            self.hdr_button.configure(state=state)
            self.interval_button.configure(state=state)
            self.record_button.configure(state=state)
            if hasattr(self, "dic_capture_button"):
                self.dic_capture_button.configure(state=state)
            set_dic_options_enabled(True)

    def _set_parameter_buttons(self, state: str) -> None:
        self.apply_gain_button.configure(state=state)
        self.apply_exposure_button.configure(state=state)
        self.apply_wb_button.configure(state=state)
        self.apply_correction_button.configure(state=state)
        self.apply_roi_button.configure(state=state)
        self.apply_trigger_button.configure(state=state)
        for name in (
            "timestamp_offset_button",
            "timestamp_offset_clear_button",
            "dark_frame_button",
            "flat_field_button",
        ):
            button = getattr(self, name, None)
            if button is not None:
                button.configure(state=state)

    def _set_mp4_progress(self, percent: float, message: str, *, visible: bool = True) -> None:
        if not hasattr(self, "mp4_progress"):
            return
        if visible and not self.mp4_progress.winfo_ismapped():
            self.mp4_progress_label.pack(side=LEFT, padx=(12, 4))
            self.mp4_progress.pack(side=LEFT, padx=(0, 4))
        value = min(max(float(percent), 0.0), 100.0)
        self.mp4_progress.configure(value=value)
        self.mp4_progress_var.set(message)

    def _hide_mp4_progress(self) -> None:
        if not hasattr(self, "mp4_progress"):
            return
        self.mp4_progress.pack_forget()
        self.mp4_progress_label.pack_forget()
        self.mp4_progress.configure(value=0)
        self.mp4_progress_var.set("MP4 --")

    def _ensure_quality_config_defaults(self) -> None:
        self.config.setdefault("focus_roi", dict(DEFAULT_FOCUS_ROI))
        self.config["focus_roi"] = clamp_roi_frac(self.config.get("focus_roi"))
        self.config.setdefault("focus_method", "laplacian")
        self.config.setdefault("focus_reference_score", None)
        self.config.setdefault("focus_drift_check_interval_minutes", 30)
        self.config.setdefault("focus_drift_warning_threshold", 0.15)
        exposure_monitor = self._ensure_config_section("exposure_monitor")
        exposure_monitor.setdefault("zebra_enabled", False)
        exposure_monitor.setdefault("histogram_enabled", True)
        exposure_monitor.setdefault("snr_warning_threshold_db", 30)
        calibration_check = self._ensure_config_section("calibration_check")
        calibration_check.setdefault("board_coverage_enabled", False)
        calibration_check.setdefault("board_min_area_frac", 0.05)
        calibration_check.setdefault("board_max_area_frac", 0.40)
        calibration_check.setdefault("board_grid_rows", 3)
        calibration_check.setdefault("board_grid_cols", 3)
        calibration_check.setdefault("board_pattern_cols", 9)
        calibration_check.setdefault("board_pattern_rows", 6)
        quality_gate = self._ensure_config_section("capture_quality_gate")
        quality_gate.setdefault("enabled", True)
        quality_gate.setdefault("strict_mode", False)
        checks = quality_gate.get("checks")
        if not isinstance(checks, dict):
            checks = {}
            quality_gate["checks"] = checks
        checks.setdefault("focus", True)
        checks.setdefault("focus_consistency", True)
        checks.setdefault("overexposure_max_pct", 5.0)
        checks.setdefault("underexposure_max_pct", 5.0)
        checks.setdefault("brightness_min", 40)
        checks.setdefault("brightness_max", 220)

    def _ensure_reliability_config_defaults(self) -> None:
        defaults = {
            "logging_enabled": True,
            "sound_alert_enabled": True,
            "auto_reconnect_enabled": True,
            "auto_reconnect_max_attempts": 5,
            "auto_reconnect_initial_delay_seconds": 1.0,
            "auto_reconnect_max_delay_seconds": 16.0,
            "consecutive_timeout_alert_threshold": 3,
            "preview_fps": DEFAULT_PREVIEW_FPS,
            "frame_timeout_ms": 800,
            "preview_frame_timeout_ms": 300,
            "roi_restart_settle_seconds": 0.20,
            "roi_warmup_frames": 2,
            "roi_warmup_timeout_ms": 800,
            "record_checksum_algorithm": "sha256",
            "record_disk_check_interval_seconds": 10.0,
            "record_disk_warning_minutes": 2.0,
            "record_disk_min_free_gb": 2.0,
            "record_stop_on_low_disk": True,
            "record_writer_stop_timeout_seconds": 10.0,
            "close_thread_join_timeout_seconds": 10.0,
            "close_total_thread_join_timeout_seconds": 10.0,
            "software_trigger_barrier_timeout_seconds": 1.0,
            "record_queue_max_items": 200,
            "record_queue_force_configured": False,
            "record_queue_put_timeout_seconds": 0.05,
            "record_output_fps_when_unlimited": 30.0,
            "ffmpeg_sequence_gap_warning_threshold": 300,
            "record_writer_batch_size": 4,
            "record_meta_flush_every": 32,
            "preview_quality_analysis_enabled": True,
            "preview_analysis_fps": 1.0,
            "record_capture_priority_mode": True,
            "record_preview_during_capture": True,
            "record_preview_fps": 2.0,
            "record_realtime_mp4": True,
            "record_clone_frames_for_writer": False,
            "record_drop_frames_on_write_lag": False,
            "record_checksum_during_capture": False,
            "record_force_image_format": False,
            "focus_peaking_overlay_interval_seconds": 0.20,
            "raw_frame_format": "npy",
            "record_disk_benchmark_enabled": True,
            "record_disk_benchmark_size_mb": 512.0,
            "record_disk_benchmark_seconds": 3.0,
            "record_disk_benchmark_margin": 1.25,
            "record_preflight_prompt_enabled": True,
            "timestamp_reject_enabled": True,
            "max_camera_timestamp_delta": 0,
            "max_host_timestamp_delta": DEFAULT_HOST_TIMESTAMP_DELTA_NS,
            "camera_timestamp_offset_samples": 5,
            "camera_timestamp_offset_window": 64,
            "stream_buffer_size": 256,
            "raw_buffer_pool_size": 64,
            "continuous_pair_buffer_size": 256,
            "continuous_pair_match_timeout_ms": 200,
            "chunk_data_enabled": True,
            "chunk_selectors": ["Timestamp", "FrameCounter", "ExposureTime", "Gain"],
            "require_hardware_trigger": False,
            "hardware_sync_enabled": False,
            "hardware_sync_master": "left",
            "hardware_sync_master_line": "Line2",
            "hardware_sync_master_line_source": "ExposureActive",
            "hardware_sync_slave_line": "Line0",
            "hardware_sync_slave_activation": "RisingEdge",
            "hardware_sync_master_trigger_source": "Software",
            "acquisition_frame_rate": None,
            "trigger_delay_us": 0.0,
            "line_debouncer_time_us": 0.0,
            "trigger_activation": "RisingEdge",
            "black_level": None,
            "digital_shift": None,
            "gamma": None,
            "save_raw_frames": True,
            "image_format": "png",
            "field_correction": {
                "enabled": False,
                "dark_frame_path": "",
                "flat_field_path": "",
                "sample_count": 16,
            },
            "dic_analysis": {
                "enabled": False,
                "displacement_module": "",
                "output_schema_version": 1,
                "overlay_path": "",
            },
        }
        for key, value in defaults.items():
            self.config.setdefault(key, value)
        if str(self.config.get("pixel_format", "")).lower() == "mono16":
            if str(self.config.get("image_format", "")).lower() in {"jpg", "jpeg"}:
                self.config["image_format"] = "png"
            self.config["save_raw_frames"] = True
        field_correction = self._ensure_config_section("field_correction")
        field_correction.setdefault("enabled", False)
        field_correction.setdefault("dark_frame_path", "")
        field_correction.setdefault("flat_field_path", "")
        field_correction.setdefault("sample_count", 16)
        dic_analysis = self._ensure_config_section("dic_analysis")
        dic_analysis.setdefault("enabled", False)
        dic_analysis.setdefault("displacement_module", "")
        dic_analysis.setdefault("output_schema_version", 1)
        dic_analysis.setdefault("overlay_path", "")
        if (
            config_bool(self.config, "timestamp_reject_enabled", False, False)
            and int(self.config.get("max_camera_timestamp_delta", 0) or 0) <= 0
            and int(self.config.get("max_host_timestamp_delta", 0) or 0) <= 0
        ):
            self.config["timestamp_reject_enabled"] = False

    def _ensure_preset_config_defaults(self) -> None:
        presets = self.config.setdefault("presets", {})
        if not isinstance(presets, dict):
            presets = {}
            self.config["presets"] = presets
        for name, defaults in default_presets().items():
            preset = presets.get(name)
            if not isinstance(preset, dict):
                presets[name] = dict(defaults)
                continue
            for key, value in defaults.items():
                preset.setdefault(key, value)

    def _ensure_project_config_defaults(self) -> None:
        dic_capture = self._ensure_config_section("dic_capture")
        for key, value in dic_capture_defaults().items():
            if isinstance(value, dict):
                section = dic_capture.get(key)
                if not isinstance(section, dict):
                    section = {}
                    dic_capture[key] = section
                for sub_key, sub_value in value.items():
                    section.setdefault(sub_key, sub_value)
            else:
                dic_capture.setdefault(key, value)
        project = self._ensure_config_section("project")
        project.setdefault("enabled", True)
        project.setdefault("projects_subdir", "projects")
        project.setdefault("current_project_id", "")
        project.setdefault("current_project_name", "")
        calibration = self._ensure_config_section("calibration")
        calibration.setdefault("enabled", False)
        calibration.setdefault("left_intrinsics", "calib/left.yaml")
        calibration.setdefault("right_intrinsics", "calib/right.yaml")
        calibration.setdefault("stereo_params", "calib/stereo.yaml")
        calibration.setdefault("rectified_overlay_alpha", 0.5)
        calibration.setdefault("rectified_line_interval_px", 120)
        wizard = self._ensure_config_section("calibration_wizard")
        wizard.setdefault("pattern", "chessboard")
        wizard.setdefault("columns", 9)
        wizard.setdefault("rows", 6)
        wizard.setdefault("square_size_mm", 40.0)
        wizard.setdefault("marker_size_mm", 30.0)
        wizard.setdefault("aruco_dictionary", "DICT_5X5_1000")
        wizard.setdefault("min_pairs", 9)
        wizard.setdefault(
            "positions",
            [
                "左上",
                "上中",
                "右上",
                "左中",
                "中心",
                "右中",
                "左下",
                "下中",
                "右下",
            ],
        )
        temperature = self._ensure_config_section("temperature_monitor")
        temperature.setdefault("enabled", True)
        temperature.setdefault("interval_seconds", 30.0)
        temperature.setdefault("stream_stats_interval_seconds", 1.0)
        temperature.setdefault("warning_threshold_c", 65.0)
        temperature.setdefault("critical_threshold_c", 75.0)
        hdr = self._ensure_config_section("hdr_bracketing")
        hdr.setdefault("enabled", True)
        hdr.setdefault("ev_offsets", [-2, -1, 0, 1, 2])
        hdr.setdefault("settle_seconds", 0.10)
        hdr.setdefault("min_exposure_time_us", 50.0)
        hdr.setdefault("max_exposure_time_us", 1000000.0)

    def _format_focus_roi(self) -> str:
        roi = clamp_roi_frac(self.config.get("focus_roi"))
        return f"ROI {roi['x_frac']:.2f},{roi['y_frac']:.2f},{roi['w_frac']:.2f},{roi['h_frac']:.2f}"

    def _guide_mode_key(self) -> str:
        value = self.guide_mode_var.get()
        if value in {"中心十字", "仅十字线"}:
            return "center"
        if value in {"仅网格线", "网格线"}:
            return "grid"
        if value in {"全部网格线", "十字+网格"}:
            return "full"
        return "off"

    def _record_roi_sizes_from_config(self, config_snapshot: dict) -> tuple[tuple[int, int], tuple[int, int]]:
        left_size = (
            int(config_snapshot.get("left_roi_width", config_snapshot.get("roi_width", CAPTURE_WIDTH)) or CAPTURE_WIDTH),
            int(config_snapshot.get("left_roi_height", config_snapshot.get("roi_height", CAPTURE_HEIGHT)) or CAPTURE_HEIGHT),
        )
        right_size = (
            int(config_snapshot.get("right_roi_width", config_snapshot.get("roi_width", CAPTURE_WIDTH)) or CAPTURE_WIDTH),
            int(config_snapshot.get("right_roi_height", config_snapshot.get("roi_height", CAPTURE_HEIGHT)) or CAPTURE_HEIGHT),
        )
        return left_size, right_size

    def _record_pair_bytes_from_config(self, config_snapshot: dict) -> int:
        if not config_bool(config_snapshot, "record_save_image_sequence", False, False):
            bitrate_kbps = max(config_int(config_snapshot, "video_bitrate_kbps", 8000), 1)
            fps = configured_record_output_fps(config_snapshot)
            return max(int((bitrate_kbps * 1000 / 8) / fps) * 2, 1)
        left_size, right_size = self._record_roi_sizes_from_config(config_snapshot)
        return estimate_frame_bytes(config_snapshot, left_size[0], left_size[1]) + estimate_frame_bytes(
            config_snapshot, right_size[0], right_size[1]
        )

    def _capture_priority_record_config(self, config_snapshot: dict) -> dict:
        snapshot = dict(config_snapshot)
        if not config_bool(snapshot, "record_capture_priority_mode", True, True):
            return snapshot

        snapshot["record_save_image_sequence"] = False
        snapshot["auto_make_mp4"] = True
        snapshot["record_realtime_mp4"] = True
        snapshot["record_preview_during_capture"] = True
        snapshot["record_preview_fps"] = 2.0
        snapshot["record_clone_frames_for_writer"] = False
        snapshot["record_checksum_during_capture"] = False
        snapshot["record_split_interval_seconds"] = 0
        snapshot["record_split_size_gb"] = 0.0
        snapshot["record_queue_max_items"] = configured_record_queue_size(
            snapshot,
            optional_positive_fps(snapshot.get("record_fps", 0.0)) or None,
        )
        snapshot["preview_quality_analysis_enabled"] = False
        snapshot["focus_peaking_overlay_interval_seconds"] = max(
            config_float(snapshot, "focus_peaking_overlay_interval_seconds", 0.20),
            1.0,
        )
        return snapshot

    def _collect_record_config_for_preflight(self) -> dict[str, object] | None:
        self._ensure_recording_config_defaults()
        try:
            fps = optional_positive_fps(self.record_fps_var.get())
            max_seconds = max(float(self.record_max_seconds_var.get() or 0), 0.0)
            updates: dict[str, object] = {"record_fps": fps or 0.0, "record_max_seconds": max_seconds}
            updates.update(self._current_parameter_config())
        except ValueError as exc:
            self.status_var.set(f"录像评估失败：{exc}")
            messagebox.showerror("录像评估失败", f"请先检查 FPS、时长、曝光、增益、ROI 等参数。\n\n{exc}")
            return None
        snapshot = self._config_snapshot()
        snapshot.update(updates)
        snapshot["image_format"] = image_extension(snapshot)
        return self._capture_priority_record_config(snapshot)

    def _build_record_preflight_plan(
        self,
        config_snapshot: dict,
        benchmark: dict[str, object] | None = None,
    ) -> dict[str, object]:
        save_root = self.project_manager.output_root_for_mode("videos")
        save_root.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(save_root)
        fps = configured_record_output_fps(config_snapshot)
        left_size, right_size = self._record_roi_sizes_from_config(config_snapshot)
        pair_bytes = self._record_pair_bytes_from_config(config_snapshot)
        required_mbps = pair_bytes * fps / 1024 / 1024
        margin = max(config_float(config_snapshot, "record_disk_benchmark_margin", 1.25), 1.0)
        measured_mbps = float((benchmark or {}).get("write_mbps") or 0.0)
        usable_mbps = measured_mbps / margin if measured_mbps > 0 else 0.0
        suggested_fps = usable_mbps * 1024 * 1024 / pair_bytes if pair_bytes > 0 and usable_mbps > 0 else 0.0
        max_seconds = max(float(config_snapshot.get("record_max_seconds", 0.0) or 0.0), 0.0)
        one_minute_bytes = pair_bytes * fps * 60
        planned_seconds = max_seconds if max_seconds > 0 else 60.0
        planned_bytes = pair_bytes * fps * planned_seconds
        disk_ok = usage.free >= one_minute_bytes
        speed_ok = measured_mbps <= 0 or measured_mbps >= required_mbps * margin
        status = "ok" if disk_ok and speed_ok else "warn"
        if measured_mbps <= 0:
            status = "unknown" if disk_ok else "warn"
        return {
            "status": status,
            "save_root": str(save_root),
            "free_bytes": int(usage.free),
            "left_size": list(left_size),
            "right_size": list(right_size),
            "pixel_format": str(config_snapshot.get("pixel_format", "Mono8")),
            "image_format": image_extension(config_snapshot),
            "record_save_image_sequence": config_bool(config_snapshot, "record_save_image_sequence", False, False),
            "target_fps": fps,
            "pair_bytes": int(pair_bytes),
            "required_mbps": required_mbps,
            "margin": margin,
            "benchmark": benchmark,
            "measured_mbps": measured_mbps,
            "usable_mbps": usable_mbps,
            "suggested_max_fps": suggested_fps,
            "one_minute_bytes": int(one_minute_bytes),
            "planned_seconds": planned_seconds,
            "planned_bytes": int(planned_bytes),
            "disk_ok": disk_ok,
            "speed_ok": speed_ok,
        }

    def _record_preflight_text(self, plan: dict[str, object]) -> str:
        left_size = plan.get("left_size") if isinstance(plan.get("left_size"), list) else [0, 0]
        right_size = plan.get("right_size") if isinstance(plan.get("right_size"), list) else [0, 0]
        measured_mbps = float(plan.get("measured_mbps") or 0.0)
        suggested_fps = float(plan.get("suggested_max_fps") or 0.0)
        speed_text = f"{measured_mbps:.1f} MB/s" if measured_mbps > 0 else "未测试"
        suggested_text = f"{suggested_fps:.1f} fps" if suggested_fps > 0 else "请先运行磁盘测速"
        save_sequence = bool(plan.get("record_save_image_sequence"))
        output_mode = "MP4视频" + (" | 图像序列：开" if save_sequence else " | 图像序列：关")
        status = str(plan.get("status", "unknown"))
        if status == "ok":
            verdict = "当前设置预计可以稳定写入。"
        elif status == "warn":
            verdict = "当前设置可能无法稳定写入。建议降低 FPS、缩小 ROI、降低码率，或关闭图像序列。"
        else:
            verdict = "磁盘测速尚未执行。进行高码率录像前请先执行测速。"
        return "\n".join(
            [
                "录像性能评估",
                "",
                f"ROI: 左 {left_size[0]}x{left_size[1]}, 右 {right_size[0]}x{right_size[1]}",
                f"像素格式: {plan.get('pixel_format', '--')}; 输出: {output_mode}",
                f"目标帧率: {float(plan.get('target_fps') or 0.0):.1f} fps",
                f"估计双目组帧大小: {format_bytes(plan.get('pair_bytes'))}",
                f"估计写入带宽: {float(plan.get('required_mbps') or 0.0):.1f} MB/s",
                f"实测磁盘写入: {speed_text}",
                f"安全系数: {float(plan.get('margin') or 1.0):.2f}x",
                f"建议最大帧率: {suggested_text}",
                f"估计 1 分钟数据量: {format_bytes(plan.get('one_minute_bytes'))}",
                f"剩余空间: {format_bytes(plan.get('free_bytes'))}",
                "",
                verdict,
            ]
        )

    def _run_record_preflight(self, config_snapshot: dict, *, run_benchmark: bool = True) -> dict[str, object]:
        benchmark = None
        if run_benchmark and config_bool(config_snapshot, "record_disk_benchmark_enabled", True, True):
            save_root = self.project_manager.output_root_for_mode("videos")
            benchmark = benchmark_write_speed(
                save_root,
                size_mb=config_float(config_snapshot, "record_disk_benchmark_size_mb", 512.0),
                sample_seconds=config_float(config_snapshot, "record_disk_benchmark_seconds", 3.0),
            )
            self._record_disk_benchmark = benchmark
        plan = self._build_record_preflight_plan(config_snapshot, benchmark)
        self._record_preflight_plan = dict(plan)
        return plan

    def show_record_preflight_wizard(self) -> None:
        config_snapshot = self._collect_record_config_for_preflight()
        if config_snapshot is None:
            return
        self.status_var.set("正在进行录像性能评估...")
        try:
            plan = self._run_record_preflight(config_snapshot, run_benchmark=True)
        except Exception as exc:
            self._show_error(exc)
            return

        popup = Toplevel(self.root)
        popup.title("录像前性能评估")
        popup.configure(bg=BG_COLOR)
        popup.transient(self.root)
        popup.geometry("+%d+%d" % (self.root.winfo_rootx() + 128, self.root.winfo_rooty() + 128))

        body = ttk.LabelFrame(popup, text="当前配置", padding=(12, 10))
        body.pack(side=TOP, fill=BOTH, expand=True, padx=12, pady=12)
        text = ttk.Label(body, text=self._record_preflight_text(plan), style="Panel.TLabel", justify=LEFT)
        text.pack(side=TOP, fill=BOTH, expand=True)
        buttons = ttk.Frame(body, style="Panel.TFrame")
        buttons.pack(side=BOTTOM, fill=X, pady=(12, 0))
        ttk.Button(buttons, text="重新测速", command=lambda: self._refresh_record_preflight_popup(config_snapshot, text)).pack(
            side=LEFT
        )
        ttk.Button(buttons, text="关闭", command=popup.destroy).pack(side=RIGHT)
        popup.update_idletasks()
        width = max(560, body.winfo_reqwidth() + 32)
        height = max(360, body.winfo_reqheight() + 36)
        popup.geometry(f"{width}x{height}+{self.root.winfo_rootx() + 128}+{self.root.winfo_rooty() + 128}")
        self.status_var.set("录像性能评估完成。")

    def _refresh_record_preflight_popup(self, config_snapshot: dict, label: ttk.Label) -> None:
        self.status_var.set("正在重新测速...")
        try:
            plan = self._run_record_preflight(config_snapshot, run_benchmark=True)
            label.configure(text=self._record_preflight_text(plan))
            self.status_var.set("录像性能评估完成。")
        except Exception as exc:
            self._show_error(exc)

    def _sync_quality_toggles(self) -> None:
        self.config["preview_quality_analysis_enabled"] = bool(self.preview_quality_analysis_var.get())
        exposure_monitor = self._ensure_config_section("exposure_monitor")
        exposure_monitor["zebra_enabled"] = bool(self.zebra_var.get())
        exposure_monitor["histogram_enabled"] = bool(self.histogram_enabled_var.get())
        self._focus_peaking_enabled_setting = bool(self.focus_peaking_var.get())
        self._histogram_enabled_setting = bool(self.histogram_enabled_var.get())
        if not self.preview_quality_analysis_var.get():
            self._set_last_quality_metrics(None)
            self._last_focus_overlay_left = None
            self._last_focus_overlay_right = None
            self._last_focus_overlay_key = None
            self._last_focus_overlay_time = 0.0
        self._update_quality_optional_sections()
        save_config(self.config)
        if hasattr(self, "left_pane"):
            if self._histogram_enabled_setting and (
                self._last_left_frame_obj is not None or self._last_right_frame_obj is not None
            ):
                metrics = self._analyze_preview_frames(
                    self._last_left_frame_obj,
                    self._last_right_frame_obj,
                    self._preview_frame_counter + 1,
                )
                self._apply_quality_metrics(metrics)
            self._display_frames(self._last_left_frame_obj, self._last_right_frame_obj)

    def _update_quality_optional_sections(self) -> None:
        if not hasattr(self, "magnifier_frame"):
            return
        if self.magnifier_enabled_var.get() and not self.magnifier_frame.winfo_manager():
            self.magnifier_frame.grid(row=0, column=1, sticky="nsew")
        elif not self.magnifier_enabled_var.get() and self.magnifier_frame.winfo_manager():
            self.magnifier_frame.grid_remove()
        self._update_magnifier()

    def _set_last_quality_metrics(self, metrics: dict[str, object] | None) -> None:
        with self._quality_metrics_lock:
            self._last_quality_metrics = metrics

    def _get_last_quality_metrics(self) -> dict[str, object] | None:
        with self._quality_metrics_lock:
            return dict(self._last_quality_metrics) if isinstance(self._last_quality_metrics, dict) else None

    def _focus_roi(self) -> dict[str, float]:
        source = self.config.get("focus_roi")
        if source is not self._cached_focus_roi_source:
            self._cached_focus_roi = clamp_roi_frac(source)
            self._cached_focus_roi_source = source
        return self._cached_focus_roi

    def _analyze_preview_frames(self, left: CameraFrame | None, right: CameraFrame | None, frame_index: int) -> dict[str, object]:
        roi = self._focus_roi()
        method = str(self.config.get("focus_method", "laplacian"))
        left_image = left.image if left is not None and getattr(left, "image", None) is not None else None
        right_image = right.image if right is not None and getattr(right, "image", None) is not None else None
        metrics: dict[str, object] = {
            "focus": focus_pair_metrics(left_image, right_image, roi, method),
            "focus_roi": roi,
            "temperatures_c": dict(self._latest_temperatures),
            "timestamp": time.time(),
        }
        update_histogram = bool(self._histogram_enabled_setting)
        metrics["left_exposure"] = exposure_metrics(left_image, include_histogram=update_histogram)
        metrics["right_exposure"] = exposure_metrics(right_image, include_histogram=update_histogram)
        metrics["dic_speckle"] = {
            "left": speckle_quality(left_image, roi),
            "right": speckle_quality(right_image, roi),
        }
        focus = metrics["focus"]
        if self._focus_peaking_enabled_setting and isinstance(focus, dict):
            now = time.perf_counter()
            interval_s = max(config_float(self.config, "focus_peaking_overlay_interval_seconds", 0.20), 0.05)
            overlay_key = (
                left.frame_number if left is not None else None,
                right.frame_number if right is not None else None,
                (
                    roi["x_frac"],
                    roi["y_frac"],
                    roi["w_frac"],
                    roi["h_frac"],
                    method,
                ),
            )
            stale = now - self._last_focus_overlay_time >= interval_s
            if self._last_focus_overlay_key is None or (overlay_key != self._last_focus_overlay_key and stale):
                self._last_focus_overlay_left = make_focus_peaking_overlay(left_image) if left_image is not None else None
                self._last_focus_overlay_right = make_focus_peaking_overlay(right_image) if right_image is not None else None
                self._last_focus_overlay_key = overlay_key
                self._last_focus_overlay_time = now
        elif not self._focus_peaking_enabled_setting:
            self._last_focus_overlay_left = None
            self._last_focus_overlay_right = None
            self._last_focus_overlay_key = None
            self._last_focus_overlay_time = 0.0
        return metrics

    def _apply_quality_metrics(self, metrics: object) -> None:
        if not isinstance(metrics, dict):
            return
        now = time.perf_counter()
        self._set_last_quality_metrics(metrics)
        if now - self._last_analysis_time < 0.20:
            return
        self._last_analysis_time = now
        focus = metrics.get("focus")
        if isinstance(focus, dict):
            self._update_focus_display(focus)
        left_exposure = metrics.get("left_exposure") if isinstance(metrics.get("left_exposure"), dict) else None
        right_exposure = metrics.get("right_exposure") if isinstance(metrics.get("right_exposure"), dict) else None
        self._update_exposure_display(left_exposure, right_exposure)
        self._update_dic_quality_display(metrics.get("dic_speckle"))
        self._update_capture_gate_preview()

    def _update_dic_quality_display(self, payload: object) -> None:
        if not hasattr(self, "dic_quality_var"):
            return
        if not isinstance(payload, dict):
            self.dic_quality_var.set("DIC speckle --")
            return
        scores: list[float] = []
        ratings: list[str] = []
        for side in ("left", "right"):
            item = payload.get(side)
            if not isinstance(item, dict):
                continue
            scores.append(float(item.get("score") or 0.0))
            ratings.append(f"{side}:{item.get('rating', '--')}")
        if not scores:
            self.dic_quality_var.set("DIC speckle --")
            return
        self.dic_quality_var.set(f"DIC speckle {min(scores):.2f} | {' '.join(ratings)}")

    def _poll_camera_temperatures(self, force: bool = False) -> None:
        if self.camera_system is None:
            return
        monitor = self.config.get("temperature_monitor", {})
        interval_s = max(config_float(monitor, "interval_seconds", 30.0), 1.0)
        stream_interval_s = max(config_float(monitor, "stream_stats_interval_seconds", 1.0), 0.2)
        now = time.perf_counter()
        poll_temperature = config_bool(monitor, "enabled", True, True) and (force or now - self._last_temperature_poll >= interval_s)
        poll_stream_stats = force or now - getattr(self, "_last_stream_stats_poll", 0.0) >= stream_interval_s
        if not poll_temperature and not poll_stream_stats:
            return
        readings = dict(getattr(self, "_latest_temperatures", {}))
        throughput = dict(getattr(self, "_latest_link_throughput_mbps", {}))
        stream_stats = dict(getattr(self, "_latest_stream_stats", {}))
        if poll_temperature:
            self._last_temperature_poll = now
            try:
                readings = self.camera_system.sensor_temperatures()
            except Exception as exc:
                LOGGER.info("temperature read failed: %s", exc)
                readings = dict(getattr(self, "_latest_temperatures", {}))
            try:
                throughput = self.camera_system.link_throughput_mbps()
            except Exception as exc:
                LOGGER.debug("link throughput read failed: %s", exc, exc_info=True)
                throughput = dict(getattr(self, "_latest_link_throughput_mbps", {}))
        if poll_stream_stats:
            self._last_stream_stats_poll = now
            try:
                stream_stats = self.camera_system.stream_stats()
            except Exception as exc:
                LOGGER.debug("stream stats read failed: %s", exc, exc_info=True)
                stream_stats = dict(getattr(self, "_latest_stream_stats", {}))
        self._latest_temperatures = readings
        self._latest_link_throughput_mbps = throughput
        self._latest_stream_stats = stream_stats
        if poll_temperature:
            sample = {
                "time": time.time(),
                "temperatures_c": dict(readings),
                "link_throughput_mbps": dict(throughput),
                "stream_stats": dict(stream_stats),
            }
            self._temperature_samples.append(sample)
            if len(self._temperature_samples) > 10000:
                self._temperature_samples = self._temperature_samples[-10000:]
        self.ui_queue.put(
            (
                "temperature",
                {
                    "temperatures_c": dict(readings),
                    "link_throughput_mbps": dict(throughput),
                    "stream_stats": dict(stream_stats),
                },
            )
        )

    def _update_temperature_display(self, readings: dict[str, float | None]) -> None:
        throughput: dict[str, float | None] = {}
        stream_stats: dict[str, dict[str, int | bool]] = {}
        if "temperatures_c" in readings or "link_throughput_mbps" in readings or "stream_stats" in readings:
            throughput = readings.get("link_throughput_mbps", {}) if isinstance(readings.get("link_throughput_mbps"), dict) else {}
            stream_stats = readings.get("stream_stats", {}) if isinstance(readings.get("stream_stats"), dict) else {}
            readings = readings.get("temperatures_c", {}) if isinstance(readings.get("temperatures_c"), dict) else {}
        parts: list[str] = []
        values = {side: value for side, value in readings.items() if value is not None}
        if values:
            parts.append(" | ".join(f"{side}:{value:.1f}C" for side, value in values.items()))
        link_values = {side: value for side, value in throughput.items() if value is not None}
        if link_values:
            parts.append("Link " + " | ".join(f"{side}:{value:.0f}Mbps" for side, value in link_values.items()))
        drop_values: dict[str, int] = {}
        for side, stats in stream_stats.items():
            if not isinstance(stats, dict):
                continue
            try:
                dropped = int(stats.get("dropped_frames", 0) or 0)
            except (TypeError, ValueError, OverflowError):
                continue
            if dropped > 0:
                drop_values[side] = dropped
        if drop_values:
            parts.append("StreamDrop " + " | ".join(f"{side}:{value}" for side, value in drop_values.items()))
        text = " | ".join(parts) if parts else "Temp unavailable"
        self.temperature_status_var.set(text)
        self._update_camera_health_display(readings, throughput, stream_stats)
        if not values:
            return
        monitor = self.config.get("temperature_monitor", {})
        warning = config_float(monitor, "warning_threshold_c", 65.0)
        critical = config_float(monitor, "critical_threshold_c", 75.0)
        max_temp = max(values.values())
        if max_temp >= critical:
            self._notify_warning("temperature_critical", f"相机传感器温度过高：{text}")
        elif max_temp >= warning:
            self._notify_warning("temperature_warning", f"相机传感器温度偏高：{text}", log_only=True)

    def _update_camera_health_display(
        self,
        temperatures: dict[str, float | None],
        throughput: dict[str, float | None],
        stream_stats: dict[str, dict[str, int | bool]],
    ) -> None:
        parts: list[str] = []
        versions = {side: version for side, version in getattr(self, "_device_versions", {}).items() if version}
        if versions:
            version_text = "FW " + " | ".join(f"{side}:{version}" for side, version in sorted(versions.items()))
            if len(set(versions.values())) > 1:
                version_text += " | 版本不一致"
            parts.append(version_text)
        link_parts: list[str] = []
        for side, stats in sorted(stream_stats.items()):
            if not isinstance(stats, dict):
                continue
            link_errors = stats.get("link_error_count")
            resends = stats.get("resend_packet_count")
            values = []
            if link_errors is not None:
                values.append(f"err {link_errors}")
            if resends is not None:
                values.append(f"resend {resends}")
            if values:
                link_parts.append(f"{side}:{'/'.join(values)}")
        if link_parts:
            parts.append("LinkCnt " + " | ".join(link_parts))
        elif throughput:
            parts.append("LinkCnt unavailable")
        if hasattr(self, "camera_health_var"):
            self.camera_health_var.set("；".join(parts) if parts else "Health --")
        self._update_temperature_trend_chart(temperatures)

    def _update_temperature_trend_chart(self, current: dict[str, float | None]) -> None:
        if not hasattr(self, "health_chart_canvas"):
            return
        canvas = self.health_chart_canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), 120)
        height = max(canvas.winfo_height(), 60)
        canvas.create_rectangle(0, 0, width, height, fill=CHART_COLOR, outline=BORDER_COLOR)
        samples = list(self._temperature_samples[-120:])
        if not samples and current:
            samples = [{"time": time.time(), "temperatures_c": dict(current)}]
        series: dict[str, list[float]] = {}
        for sample in samples:
            values = sample.get("temperatures_c") if isinstance(sample, dict) else None
            if not isinstance(values, dict):
                continue
            for side, value in values.items():
                if value is None:
                    continue
                try:
                    series.setdefault(str(side), []).append(float(value))
                except (TypeError, ValueError):
                    continue
        all_values = [value for values in series.values() for value in values]
        if not all_values:
            canvas.create_text(width // 2, height // 2, text="No temp trend", fill=SUBTLE_TEXT_COLOR)
            return
        min_v = min(all_values)
        max_v = max(all_values)
        if max_v <= min_v:
            max_v = min_v + 1.0
        colors = {"left": ACCENT_ACTIVE_COLOR, "right": WARNING_COLOR}
        left_pad, right_pad, top_pad, bottom_pad = 28, 8, 8, 16
        plot_w = max(width - left_pad - right_pad, 1)
        plot_h = max(height - top_pad - bottom_pad, 1)
        for tick in (min_v, max_v):
            y = top_pad + (max_v - tick) / (max_v - min_v) * plot_h
            canvas.create_line(left_pad, y, width - right_pad, y, fill="#26313b")
            canvas.create_text(3, y, text=f"{tick:.0f}", fill=SUBTLE_TEXT_COLOR, anchor="w", font=(MONO_FONT_FAMILY, 7))
        for side, values in sorted(series.items()):
            if len(values) == 1:
                x = left_pad + plot_w
                y = top_pad + (max_v - values[0]) / (max_v - min_v) * plot_h
                canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill=colors.get(side, SUCCESS_COLOR), outline="")
                continue
            points: list[float] = []
            for index, value in enumerate(values):
                x = left_pad + index / max(len(values) - 1, 1) * plot_w
                y = top_pad + (max_v - value) / (max_v - min_v) * plot_h
                points.extend([x, y])
            canvas.create_line(*points, fill=colors.get(side, SUCCESS_COLOR), width=2)
            canvas.create_text(
                width - right_pad,
                points[-1],
                text=side,
                fill=colors.get(side, SUCCESS_COLOR),
                anchor="e",
                font=(MONO_FONT_FAMILY, 7, "bold"),
            )

    def _update_focus_chart(self, score: float) -> None:
        now = time.time()
        self._focus_history.append((now, score))
        self._focus_history = self._focus_history[-240:]
        if score > self._focus_peak_score:
            self._focus_peak_score = score
        peak = max(self._focus_peak_score, 1e-9)
        pct = max(min(score / peak * 100.0, 100.0), 0.0)
        self.focus_peak_var.set(f"峰值 {self._focus_peak_score:.1f} | {pct:.0f}%")
        if not hasattr(self, "focus_chart_canvas"):
            return
        canvas = self.focus_chart_canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), 40)
        height = max(canvas.winfo_height(), 32)
        canvas.create_rectangle(0, 0, width, height, fill=CHART_COLOR, outline="")
        if len(self._focus_history) < 2:
            return
        scores = [item[1] for item in self._focus_history]
        max_score = max(max(scores), 1e-9)
        points: list[float] = []
        count = len(scores)
        for index, value in enumerate(scores):
            x = 4 + index * (width - 8) / max(count - 1, 1)
            y = height - 4 - (value / max_score) * (height - 8)
            points.extend([x, y])
        canvas.create_line(*points, fill=SUCCESS_COLOR, width=2, smooth=True)
        canvas.create_line(
            4,
            height - 4 - (score / max_score) * (height - 8),
            width - 4,
            height - 4 - (score / max_score) * (height - 8),
            fill=WARNING_COLOR,
        )

    def _update_focus_display(self, focus: dict[str, object]) -> None:
        score = float(focus.get("score") or 0.0)
        left_score = focus.get("left")
        right_score = focus.get("right")
        delta = focus.get("delta")
        reference = self.config.get("focus_reference_score")
        if reference in (None, "") or float(reference) <= 0:
            pct = 0.0
            status = "未标定"
            style = "Yellow.Horizontal.TProgressbar"
            label_style = "Warn.TLabel"
        else:
            pct = min(max(score / max(float(reference), 1e-9) * 100.0, 0.0), 150.0)
            if pct >= 80:
                status = "对焦良好"
                style = "Green.Horizontal.TProgressbar"
                label_style = "Good.TLabel"
            elif pct >= 40:
                status = "对焦一般，建议微调"
                style = "Yellow.Horizontal.TProgressbar"
                label_style = "Warn.TLabel"
            else:
                status = "对焦不足，需重新对焦"
                style = "Red.Horizontal.TProgressbar"
                label_style = "Bad.TLabel"
        self.focus_score_var.set(f"Focus {score:.1f}")
        self._update_focus_chart(score)
        self.focus_progress.configure(value=min(pct, 100.0), style=style)
        self.focus_status_var.set(status)
        self.focus_status_label.configure(style=label_style)
        l_text = "--" if left_score is None else f"{float(left_score):.1f}"
        r_text = "--" if right_score is None else f"{float(right_score):.1f}"
        d_text = "--" if delta is None else f"{float(delta):.1f}"
        self.focus_detail_var.set(f"L: {l_text} | R: {r_text} | Δ: {d_text}")
        consistency_warning = bool(focus.get("consistency_warning"))
        self.focus_detail_label.configure(style="PanelBad.TLabel" if consistency_warning else "Panel.TLabel")
        if consistency_warning:
            self.focus_status_var.set("左右对焦不一致")
            self.focus_status_label.configure(style="Bad.TLabel")

    def _update_exposure_display(self, left_exposure: dict[str, object] | None, right_exposure: dict[str, object] | None) -> None:
        exposures = [item for item in (left_exposure, right_exposure) if item]
        if not exposures:
            self.exposure_status_var.set("过曝: -- | 欠曝: -- | SNR: --")
            return
        over_pct = max(float(item.get("over_pct") or 0.0) for item in exposures)
        under_pct = max(float(item.get("under_pct") or 0.0) for item in exposures)
        snr_values = [float(item["snr_db"]) for item in exposures if item.get("snr_db") is not None]
        snr_text = "无法估算"
        snr_style = "PanelBad.TLabel"
        if snr_values:
            snr = min(snr_values)
            snr_text = f"{snr:.1f}dB"
            if snr > 40:
                snr_style = "PanelGood.TLabel"
            elif snr >= config_float(self.config.get("exposure_monitor", {}), "snr_warning_threshold_db", 30.0):
                snr_style = "PanelWarn.TLabel"
            else:
                snr_style = "PanelBad.TLabel"
        self.exposure_status_var.set(f"过曝: {over_pct:.1f}% | 欠曝: {under_pct:.1f}% | SNR: {snr_text}")
        self.exposure_status_label.configure(style=snr_style)
        advice = left_exposure.get("advice") if left_exposure else None
        if not advice and right_exposure:
            advice = right_exposure.get("advice")
        gain_limit = config_float(self.config, "auto_gain_upper_limit", config_float(self.config, "gain", 0.0))
        current_gain = config_float(self.config, "gain", 0.0)
        if snr_values and current_gain > min(gain_limit, 6.0) and min(snr_values) < 30.0:
            advice = "建议降低增益或增加光照"
        self.exposure_advice_var.set(f"曝光建议: {advice or '--'}")
        if self.histogram_enabled_var.get():
            if left_exposure and left_exposure.get("histogram"):
                self.left_hist_canvas.set_histogram(left_exposure.get("histogram"))
            if right_exposure and right_exposure.get("histogram"):
                self.right_hist_canvas.set_histogram(right_exposure.get("histogram"))

    def _quality_report_from_metrics(self, metrics: dict[str, object] | None = None) -> dict[str, object]:
        if metrics is None:
            metrics = self._get_last_quality_metrics()
        if not isinstance(metrics, dict) or not metrics:
            return {
                "results": [],
                "passed": 0,
                "failed": 0,
                "ok": True,
                "text": "暂无实时质量数据",
            }
        gate = self.config.get("capture_quality_gate", {})
        checks = gate.get("checks", {}) if isinstance(gate, dict) else {}
        if not isinstance(checks, dict):
            checks = {}
        focus = metrics.get("focus") if isinstance(metrics, dict) else None
        left_exposure = metrics.get("left_exposure") if isinstance(metrics, dict) else None
        right_exposure = metrics.get("right_exposure") if isinstance(metrics, dict) else None
        results: list[dict[str, object]] = []
        if checks.get("focus", True):
            reference = self.config.get("focus_reference_score")
            score = float(focus.get("score") or 0.0) if isinstance(focus, dict) else 0.0
            ok = True if reference in (None, "") or float(reference) <= 0 else score >= float(reference) * 0.40
            results.append({"name": "对焦", "ok": ok, "detail": f"{score:.1f}"})
        if checks.get("focus_consistency", True):
            if self._connected_camera_count() >= 2:
                warning = bool(focus.get("consistency_warning")) if isinstance(focus, dict) else False
                results.append({"name": "左右一致", "ok": not warning, "detail": ""})
            else:
                results.append({"name": "左右一致", "ok": True, "detail": "单相机跳过"})
        exposures = [item for item in (left_exposure, right_exposure) if isinstance(item, dict)]
        if exposures:
            over_pct = max(float(item.get("over_pct") or 0.0) for item in exposures)
            under_pct = max(float(item.get("under_pct") or 0.0) for item in exposures)
            mean = float(np.mean([float(item.get("mean") or 0.0) for item in exposures]))
        else:
            over_pct = under_pct = mean = 0.0
        results.append(
            {
                "name": "过曝",
                "ok": over_pct <= float(checks.get("overexposure_max_pct", 5.0)),
                "detail": f"{over_pct:.1f}%",
            }
        )
        results.append(
            {
                "name": "欠曝",
                "ok": under_pct <= float(checks.get("underexposure_max_pct", 5.0)),
                "detail": f"{under_pct:.1f}%",
            }
        )
        brightness_ok = float(checks.get("brightness_min", 40)) <= mean <= float(checks.get("brightness_max", 220))
        results.append({"name": "亮度", "ok": brightness_ok, "detail": f"{mean:.1f}"})
        results.append({"name": "相机连接", "ok": self.camera_system is not None and self._connected_camera_count() > 0, "detail": ""})
        passed = sum(1 for item in results if item["ok"])
        failed = len(results) - passed
        text_parts = [f"{'✓' if item['ok'] else '✗'} {item['name']}{item['detail'] if item['detail'] else ''}" for item in results]
        return {
            "results": results,
            "passed": passed,
            "failed": failed,
            "ok": failed == 0,
            "text": " | ".join(text_parts) + f" | {passed}/{len(results)} 通过",
        }

    def _update_capture_gate_preview(self) -> None:
        report = self._quality_report_from_metrics()
        self._apply_quality_report(report)

    def _apply_quality_report(self, report: object) -> None:
        if not isinstance(report, dict):
            return
        self.capture_gate_var.set(f"采集检查: {report.get('text', '--')}")

    def _apply_calibration_board(self, board: object) -> None:
        if not isinstance(board, dict):
            return
        area = float(board.get("area_frac") or 0.0) * 100.0
        grid_icon = str(board.get("grid_icon") or "")
        suffix = f" | {grid_icon.replace(chr(10), '/')}" if grid_icon else ""
        self.calibration_status_var.set(
            f"标定板覆盖: {area:.1f}% | 位置: {board.get('position', '--')} | {board.get('suggestion', '--')}{suffix}"
        )

    def _handle_photo_quality_prefetched(self, payload: object) -> None:
        if not isinstance(payload, tuple) or len(payload) != 4:
            self._set_capture_buttons(NORMAL)
            return
        left, right, trigger_time, metrics = payload
        if not isinstance(metrics, dict):
            self._set_capture_buttons(NORMAL)
            return
        allow_capture, quality_report = self._capture_quality_gate_allows(metrics)
        self._apply_quality_report(quality_report)
        if not allow_capture:
            self.status_var.set("采集已取消：质量检查未通过。")
            self._set_capture_buttons(NORMAL)
            return
        self.status_var.set("质量检查通过，正在保存现场采样帧...")

        def worker() -> None:
            try:
                photo_dir = self._save_photo_pair(left, right, trigger_time, mode="photo", quality_report=quality_report)
                if self.previewing:
                    self.ui_queue.put(("frames", (left, right)))
                self.ui_queue.put(("shutter_flash", None))
                self.ui_queue.put(("photo_done", ("photo", photo_dir)))
            except Exception as exc:
                self.ui_queue.put(("error", exc))
            finally:
                self.ui_queue.put(("capture_idle", None))

        self._start_background_thread(worker, "save-photo-prefetched")

    def set_focus_reference(self) -> None:
        metrics = self._get_last_quality_metrics()
        focus = metrics.get("focus") if isinstance(metrics, dict) else None
        if not isinstance(focus, dict) or float(focus.get("score") or 0.0) <= 0:
            self.status_var.set("暂无可用对焦分数，请先开启预览并对准目标。")
            return
        score = float(focus.get("score") or 0.0)
        self.config["focus_reference_score"] = score
        save_config(self.config)
        self.status_var.set(f"已设为对焦目标：{score:.1f}")
        self._update_focus_display(focus)

    def save_focus_reference_snapshot(self) -> None:
        left = self._last_left_frame_obj
        right = self._last_right_frame_obj
        if left is None and right is None:
            self.status_var.set("暂无可保存的对焦基准帧，请先开启预览。")
            return
        focus = focus_pair_metrics(
            left.image if left is not None else None,
            right.image if right is not None else None,
            self.config.get("focus_roi"),
            str(self.config.get("focus_method", "laplacian")),
        )
        score = float(focus.get("score") or 0.0)
        self.config["focus_reference_score"] = score
        save_config(self.config)
        root = resolve_output_root(self.config) / "focus_refs"
        root.mkdir(parents=True, exist_ok=True)
        left_dir = root / "left"
        right_dir = root / "right"
        left_dir.mkdir(parents=True, exist_ok=True)
        right_dir.mkdir(parents=True, exist_ok=True)
        capture_id = timestamp_ms()
        ext = image_extension(self.config)
        if left is not None:
            self._save_image(left.image, left_dir / f"focus_ref_{capture_id}_left.{ext}")
        if right is not None:
            self._save_image(right.image, right_dir / f"focus_ref_{capture_id}_right.{ext}")
        self._update_focus_display(focus)
        self.status_var.set(f"已保存对焦基准：{score:.1f}")

    def _start_focus_reference_check(self) -> None:
        reference = self.config.get("focus_reference_score")
        if reference in (None, "") or float(reference) <= 0 or self.camera_system is None:
            return

        def worker() -> None:
            try:
                camera_system = self._require_camera_system()
                left, right, _trigger_time = camera_system.capture_pair()
                focus = focus_pair_metrics(
                    left.image if left is not None else None,
                    right.image if right is not None else None,
                    self.config.get("focus_roi"),
                    str(self.config.get("focus_method", "laplacian")),
                )
                self.ui_queue.put(("focus_reference_check", focus))
            except Exception as exc:
                self.ui_queue.put(("status", f"对焦启动检查未完成：{exc}"))

        self._start_background_thread(worker, "focus-reference-check", report_errors=False)

    def _handle_focus_reference_check(self, focus: object) -> None:
        if not isinstance(focus, dict):
            return
        reference = self.config.get("focus_reference_score")
        if reference in (None, "") or float(reference) <= 0:
            return
        score = float(focus.get("score") or 0.0)
        deviation = abs(score - float(reference)) / max(float(reference), 1e-9)
        if deviation > 0.20 and score != self._last_reference_warning_score:
            self._last_reference_warning_score = score
            messagebox.showwarning("对焦偏移提醒", "检测到对焦可能偏移，建议重新对焦并更新基准")

    def _schedule_focus_drift_check(self) -> None:
        if self._closing:
            return
        if self._focus_drift_timer is not None:
            self._focus_drift_timer.cancel()
            self._focus_drift_timer = None
        minutes = max(config_float(self.config, "focus_drift_check_interval_minutes", 30.0), 1.0)
        self._focus_drift_timer = threading.Timer(minutes * 60.0, self._focus_drift_timer_worker)
        self._focus_drift_timer.daemon = True
        self._focus_drift_timer.start()

    def _focus_drift_timer_worker(self) -> None:
        try:
            reference = self.config.get("focus_reference_score")
            if reference not in (None, "") and float(reference) > 0 and self.camera_system is not None and not self.recording:
                left, right, _trigger_time = self._require_camera_system().capture_pair()
                focus = focus_pair_metrics(
                    left.image if left is not None else None,
                    right.image if right is not None else None,
                    self.config.get("focus_roi"),
                    str(self.config.get("focus_method", "laplacian")),
                )
                score = float(focus.get("score") or 0.0)
                threshold = config_float(self.config, "focus_drift_warning_threshold", 0.15)
                deviation = abs(score - float(reference)) / max(float(reference), 1e-9)
                if deviation > threshold:
                    self._focus_drift_warning_text = f"对焦漂移 {deviation * 100:.0f}%：建议重新对焦"
                    self.ui_queue.put(("status", self._focus_drift_warning_text))
        except Exception as exc:
            if not self._closing:
                self.ui_queue.put(("status", f"对焦漂移检查未完成：{exc}"))
        finally:
            if not self._closing and self.camera_system is not None:
                self._schedule_focus_drift_check()

    def run_epipolar_check(self) -> None:
        left = self._clone_frame(self._last_left_frame_obj)
        right = self._clone_frame(self._last_right_frame_obj)
        if left is None or right is None:
            self.status_var.set("需要左右两路预览帧才能执行极线对准检查。")
            return
        self.epipolar_status_var.set("极线偏差: 正在检查...")

        def worker() -> None:
            result = epipolar_alignment(left.image, right.image)
            self.ui_queue.put(("epipolar_done", result))

        self._start_background_thread(worker, "epipolar-check", report_errors=False)

    def _handle_epipolar_result(self, result: object) -> None:
        if not isinstance(result, dict):
            return
        message = str(result.get("message", "极线偏差: --"))
        if result.get("ok") and result.get("match_count"):
            message += f" | 匹配点 {result.get('match_count')}"
        self.epipolar_status_var.set(message)
        self.epipolar_label.configure(style="PanelBad.TLabel" if result.get("warning") else "PanelGood.TLabel")
        if result.get("warning"):
            self.status_var.set("相机可能存在旋转偏差，请调整后重新采图")

    def _on_magnifier_motion(self, event) -> None:
        if not self.magnifier_enabled_var.get() or self._magnifier_locked:
            return
        roi = self._canvas_point_to_magnifier_roi(event.x, event.y, self._pane_from_event(event))
        if roi is not None:
            self._magnifier_roi_frac = roi
            self.config["focus_roi"] = roi
            self.focus_roi_var.set(self._format_focus_roi())
            self._update_magnifier()

    def _on_magnifier_click(self, event) -> None:
        if not self.magnifier_enabled_var.get() or self.roi_editing:
            return
        roi = self._canvas_point_to_magnifier_roi(event.x, event.y, self._pane_from_event(event))
        if roi is not None:
            self._magnifier_roi_frac = roi
            self.config["focus_roi"] = roi
            save_config(self.config)
            self.focus_roi_var.set(self._format_focus_roi())
            self._magnifier_locked = not self._magnifier_locked
            self._update_magnifier()

    def _on_magnifier_wheel(self, event) -> None:
        if not self.magnifier_enabled_var.get():
            return
        values = [1, 2, 4]
        index = values.index(self._magnifier_zoom) if self._magnifier_zoom in values else 0
        delta = getattr(event, "delta", 0)
        button_num = getattr(event, "num", None)
        if delta > 0 or button_num == 4:
            index = min(index + 1, len(values) - 1)
        elif delta < 0 or button_num == 5:
            index = max(index - 1, 0)
        else:
            return
        self._magnifier_zoom = values[index]
        self._update_magnifier()

    def _pane_from_event(self, event) -> ZoomImagePane:
        widget = getattr(event, "widget", None)
        if hasattr(self, "right_pane") and widget is self.right_pane.canvas and self.right_pane._last_image is not None:
            return self.right_pane
        return self.left_pane

    def _canvas_point_to_magnifier_roi(self, x: int, y: int, pane: ZoomImagePane | None = None) -> dict[str, float] | None:
        pane = pane or self.left_pane
        if pane._last_image is None or pane._render_bounds is None or not pane._point_inside_rendered_image(x, y):
            return None
        left, top, display_width, display_height = pane._render_bounds
        image_width, image_height = pane._last_image.size
        image_x = int(round((x - left) * image_width / display_width))
        image_y = int(round((y - top) * image_height / display_height))
        box_w = min(200, image_width)
        box_h = min(200, image_height)
        x0 = min(max(image_x - box_w // 2, 0), max(image_width - box_w, 0))
        y0 = min(max(image_y - box_h // 2, 0), max(image_height - box_h, 0))
        return roi_from_pixels(x0, y0, box_w, box_h, image_width, image_height)

    @staticmethod
    def _magnifier_crop_for_roi(
        image: Image.Image,
        roi: dict[str, float],
    ) -> tuple[Image.Image, tuple[int, int, int, int]]:
        safe_roi = clamp_roi_frac(roi)
        x = min(max(int(safe_roi["x_frac"] * image.width), 0), max(image.width - 1, 0))
        y = min(max(int(safe_roi["y_frac"] * image.height), 0), max(image.height - 1, 0))
        w = max(1, int(safe_roi["w_frac"] * image.width))
        h = max(1, int(safe_roi["h_frac"] * image.height))
        x1 = min(max(x + w, x + 1), image.width)
        y1 = min(max(y + h, y + 1), image.height)
        return image.crop((x, y, x1, y1)).convert("RGB"), (x, y, x1, y1)

    @staticmethod
    def _image_luma_stats(image: Image.Image) -> tuple[float, int, int]:
        gray = np.asarray(image.convert("L"), dtype=np.uint8)
        if gray.size == 0:
            return 0.0, 0, 0
        return float(np.mean(gray)), int(np.min(gray)), int(np.max(gray))

    def _update_magnifier(self) -> None:
        if not hasattr(self, "magnifier_canvas"):
            return
        self.magnifier_canvas.delete("all")
        self._magnifier_image_refs = []
        if not self.magnifier_enabled_var.get():
            self.magnifier_canvas.create_text(120, 90, text="Magnifier off", fill=SUBTLE_TEXT_COLOR, anchor="center")
            return
        frames = (("L", self._last_left_frame_obj), ("R", self._last_right_frame_obj))
        available = [(label, frame) for label, frame in frames if frame is not None]
        if not available:
            self.magnifier_canvas.create_text(120, 90, text="No frame", fill=SUBTLE_TEXT_COLOR, anchor="center")
            return
        roi = clamp_roi_frac(self._magnifier_roi_frac)
        canvas_w = max(self.magnifier_canvas.winfo_width(), 120)
        canvas_h = max(self.magnifier_canvas.winfo_height(), 80)
        pane_gap = 8 if len(available) > 1 else 0
        pane_w = max((canvas_w - pane_gap) // max(len(available), 1), 1)
        image_h = max(canvas_h - 18, 1)
        dark_count = 0
        for index, (label, frame) in enumerate(available):
            crop, box = self._magnifier_crop_for_roi(frame.image, roi)
            mean_luma, _min_luma, max_luma = self._image_luma_stats(crop)
            crop_is_dark = mean_luma <= 2.0 and max_luma <= 5
            if crop_is_dark:
                dark_count += 1
            if self._magnifier_zoom > 1:
                crop = crop.resize(
                    (crop.width * self._magnifier_zoom, crop.height * self._magnifier_zoom),
                    Image.Resampling.NEAREST,
                )
            preview = crop.copy()
            preview.thumbnail((pane_w, image_h), Image.Resampling.NEAREST)
            image_ref = ImageTk.PhotoImage(preview)
            self._magnifier_image_refs.append(image_ref)
            left = index * (pane_w + pane_gap)
            center_x = left + pane_w // 2
            center_y = 18 + image_h // 2
            self.magnifier_canvas.create_image(center_x, center_y, image=image_ref, anchor="center")
            self.magnifier_canvas.create_rectangle(left, 0, left + pane_w, canvas_h, outline=BORDER_COLOR)
            self.magnifier_canvas.create_text(
                center_x,
                3,
                text=f"{label} mean {mean_luma:.0f}",
                fill=WARNING_COLOR if crop_is_dark else TEXT_COLOR,
                anchor="n",
                font=(FONT_FAMILY, 9, "bold"),
            )
            x0, y0, x1, y1 = box
            self.magnifier_canvas.create_text(
                center_x,
                canvas_h - 3,
                text=f"{x0},{y0} {x1 - x0}x{y1 - y0}",
                fill=SUBTLE_TEXT_COLOR,
                anchor="s",
                font=(MONO_FONT_FAMILY, 8),
            )
        lock_text = "locked" if self._magnifier_locked else "live"
        status = " | dark ROI" if dark_count == len(available) else ""
        self.magnifier_info_var.set(
            f"Zoom {self._magnifier_zoom * 100}% | {lock_text} | ROI x={roi['x_frac']:.3f}, y={roi['y_frac']:.3f}{status}"
        )

    def _capture_quality_gate_allows(self, metrics: dict[str, object] | None = None) -> tuple[bool, dict[str, object]]:
        gate = self.config.get("capture_quality_gate", {})
        if not config_bool(gate, "enabled", True, True):
            return True, {"ok": True, "text": "采集检查已关闭"}
        report = self._quality_report_from_metrics(metrics)
        strict = config_bool(gate, "strict_mode", False, False)
        if report["ok"] or not strict:
            return True, report
        failed_names = [str(item["name"]) for item in report["results"] if not item["ok"]]
        allow = messagebox.askyesno("采集质量检查", f"{'、'.join(failed_names)} 未通过，是否仍要采集？")
        return allow, report

    def _quality_metrics_for_pair(self, left: CameraFrame | None, right: CameraFrame | None) -> dict[str, object]:
        metrics = self._analyze_preview_frames(left, right, self._preview_frame_counter + 1)
        self._set_last_quality_metrics(metrics)
        calibration_cfg = self.config.get("calibration_check", {})
        if config_bool(calibration_cfg, "board_coverage_enabled", False, False):
            board = calibration_board_coverage(
                left.image if left is not None else None,
                calibration_cfg if isinstance(calibration_cfg, dict) else {},
            )
            metrics["calibration_board"] = board
            if board is not None:
                self.ui_queue.put(("calibration_board", board))
                area = float(board.get("area_frac") or 0.0) * 100.0
                self.ui_queue.put(
                    (
                        "status",
                        f"标定板覆盖: {area:.1f}% | 位置: {board.get('position')} | {board.get('suggestion')}",
                    )
                )
        return metrics

    def _current_parameter_config(self) -> dict:
        left_width = optional_int_text(self.left_roi_width_var.get()) or CAPTURE_WIDTH
        left_height = optional_int_text(self.left_roi_height_var.get()) or CAPTURE_HEIGHT
        left_offset_x = int(self.left_roi_offset_x_var.get() or 0)
        left_offset_y = int(self.left_roi_offset_y_var.get() or 0)
        right_width = optional_int_text(self.right_roi_width_var.get()) or CAPTURE_WIDTH
        right_height = optional_int_text(self.right_roi_height_var.get()) or CAPTURE_HEIGHT
        right_offset_x = int(self.right_roi_offset_x_var.get() or 0)
        right_offset_y = int(self.right_roi_offset_y_var.get() or 0)
        return {
            "trigger_source": safe_capture_trigger_source(self.trigger_source_var.get()),
            "exposure_auto": self.exposure_auto_var.get(),
            "exposure_time_us": float(self.exposure_time_var.get() or 0),
            "auto_exposure_lower_limit": optional_float_text(self.auto_exposure_lower_var.get()),
            "auto_exposure_upper_limit": optional_float_text(self.auto_exposure_upper_var.get()),
            "gain_auto": self.gain_auto_var.get(),
            "gain": float(self.gain_var.get() or 0),
            "auto_gain_lower_limit": optional_float_text(self.auto_gain_lower_var.get()),
            "auto_gain_upper_limit": optional_float_text(self.auto_gain_upper_var.get()),
            "balance_white_auto": self.balance_auto_var.get(),
            "balance_ratio_red": optional_float_text(self.balance_red_var.get()),
            "balance_ratio_green": optional_float_text(self.balance_green_var.get()),
            "balance_ratio_blue": optional_float_text(self.balance_blue_var.get()),
            "black_level": optional_float_text(self.black_level_var.get()),
            "digital_shift": optional_float_text(self.digital_shift_var.get()),
            "gamma": optional_float_text(self.gamma_var.get()),
            "pixel_format": self.config.get("pixel_format", "Mono8"),
            "image_format": image_extension(self.config),
            "record_force_image_format": config_bool(self.config, "record_force_image_format", False, False),
            "save_raw_frames": config_bool(self.config, "save_raw_frames", False, False),
            "raw_frame_format": raw_frame_format(self.config),
            "camera_timestamp_offset_fixed": self.config.get("camera_timestamp_offset_fixed"),
            "field_correction": dict(self.config.get("field_correction", {}))
            if isinstance(self.config.get("field_correction"), dict)
            else {},
            "dic_analysis": dict(self.config.get("dic_analysis", {}))
            if isinstance(self.config.get("dic_analysis"), dict)
            else {},
            "chunk_data_enabled": config_bool(self.config, "chunk_data_enabled", False, False),
            "chunk_selectors": list(self.config.get("chunk_selectors", []))
            if isinstance(self.config.get("chunk_selectors"), list)
            else self.config.get("chunk_selectors"),
            **TRIGGER_CONFIG_SAFE_DEFAULTS,
            "roi_width": left_width,
            "roi_height": left_height,
            "roi_offset_x": left_offset_x,
            "roi_offset_y": left_offset_y,
            "left_roi_width": left_width,
            "left_roi_height": left_height,
            "left_roi_offset_x": left_offset_x,
            "left_roi_offset_y": left_offset_y,
            "right_roi_width": right_width,
            "right_roi_height": right_height,
            "right_roi_offset_x": right_offset_x,
            "right_roi_offset_y": right_offset_y,
        }

    def _load_vars_from_snapshot(self, snapshot: dict[str, object], *, include_record_fps: bool = True) -> None:
        snapshot = safe_trigger_config(snapshot)
        self.trigger_source_var.set(display_trigger_source(snapshot.get("trigger_source", "Software")))
        self._set_cached_trigger_source(str(snapshot.get("trigger_source", "Software")))
        self.exposure_auto_var.set(str(snapshot.get("exposure_auto", "Off")))
        self.exposure_time_var.set(str(snapshot.get("exposure_time_us", 10000.0)))
        self.gain_auto_var.set(str(snapshot.get("gain_auto", "Off")))
        self.gain_var.set(str(snapshot.get("gain", 0.0)))
        self.black_level_var.set(optional_config_text(snapshot, "black_level", ""))
        self.digital_shift_var.set(optional_config_text(snapshot, "digital_shift", ""))
        self.gamma_var.set(optional_config_text(snapshot, "gamma", ""))
        self.left_roi_width_var.set(str(snapshot.get("left_roi_width", snapshot.get("roi_width", CAPTURE_WIDTH))))
        self.left_roi_height_var.set(str(snapshot.get("left_roi_height", snapshot.get("roi_height", CAPTURE_HEIGHT))))
        self.left_roi_offset_x_var.set(str(snapshot.get("left_roi_offset_x", snapshot.get("roi_offset_x", 0))))
        self.left_roi_offset_y_var.set(str(snapshot.get("left_roi_offset_y", snapshot.get("roi_offset_y", 0))))
        self.right_roi_width_var.set(str(snapshot.get("right_roi_width", snapshot.get("roi_width", CAPTURE_WIDTH))))
        self.right_roi_height_var.set(str(snapshot.get("right_roi_height", snapshot.get("roi_height", CAPTURE_HEIGHT))))
        self.right_roi_offset_x_var.set(str(snapshot.get("right_roi_offset_x", snapshot.get("roi_offset_x", 0))))
        self.right_roi_offset_y_var.set(str(snapshot.get("right_roi_offset_y", snapshot.get("roi_offset_y", 0))))
        self.interval_seconds_var.set(optional_config_text(snapshot, "interval_capture_seconds", ""))
        self.interval_limit_var.set(optional_config_text(snapshot, "interval_capture_count", ""))
        if include_record_fps:
            self.record_fps_var.set(str(snapshot.get("record_fps", 5.0)))
        self.dic_record_fps_var.set(str(snapshot.get("record_fps", DIC_CAPTURE_CONFIG["record_fps"])))
        if hasattr(self, "dic_pixel_format_var"):
            pixel_format = str(snapshot.get("pixel_format", DIC_CAPTURE_CONFIG["pixel_format"]))
            self.dic_pixel_format_var.set(pixel_format if pixel_format in DIC_PIXEL_FORMATS else "Mono8")

    def _apply_capture_config_to_camera(self, config_snapshot: dict[str, object]) -> list[str]:
        config_snapshot = safe_trigger_config(config_snapshot)
        camera_system = self._require_camera_system()
        warnings: list[str] = []

        def optional_config_float(key: str) -> float | None:
            value = config_snapshot.get(key)
            return optional_float_text("" if value is None else str(value))

        pixel_format = str(config_snapshot.get("pixel_format", "Mono8"))
        apply_pixel_format = getattr(camera_system, "apply_pixel_format_settings", None)
        if callable(apply_pixel_format):
            warnings.extend(apply_pixel_format(pixel_format))
        warnings.extend(camera_system.apply_trigger_settings(str(config_snapshot.get("trigger_source", "Software"))))
        warnings.extend(
            camera_system.apply_exposure_settings(
                str(config_snapshot.get("exposure_auto", "Off")),
                float(config_snapshot.get("exposure_time_us", 0.0) or 0.0),
                optional_config_float("auto_exposure_lower_limit"),
                optional_config_float("auto_exposure_upper_limit"),
            )
        )
        warnings.extend(
            camera_system.apply_gain_settings(
                str(config_snapshot.get("gain_auto", "Off")),
                float(config_snapshot.get("gain", 0.0) or 0.0),
                optional_config_float("auto_gain_lower_limit"),
                optional_config_float("auto_gain_upper_limit"),
            )
        )
        apply_correction = getattr(camera_system, "apply_image_correction_settings", None)
        if callable(apply_correction):
            warnings.extend(
                apply_correction(
                    optional_config_float("black_level"),
                    optional_config_float("digital_shift"),
                    optional_config_float("gamma"),
                )
            )
        rois = {
            "left": (
                int(config_snapshot.get("left_roi_width", config_snapshot.get("roi_width", CAPTURE_WIDTH)) or CAPTURE_WIDTH),
                int(config_snapshot.get("left_roi_height", config_snapshot.get("roi_height", CAPTURE_HEIGHT)) or CAPTURE_HEIGHT),
                int(config_snapshot.get("left_roi_offset_x", config_snapshot.get("roi_offset_x", 0)) or 0),
                int(config_snapshot.get("left_roi_offset_y", config_snapshot.get("roi_offset_y", 0)) or 0),
            ),
            "right": (
                int(config_snapshot.get("right_roi_width", config_snapshot.get("roi_width", CAPTURE_WIDTH)) or CAPTURE_WIDTH),
                int(config_snapshot.get("right_roi_height", config_snapshot.get("roi_height", CAPTURE_HEIGHT)) or CAPTURE_HEIGHT),
                int(config_snapshot.get("right_roi_offset_x", config_snapshot.get("roi_offset_x", 0)) or 0),
                int(config_snapshot.get("right_roi_offset_y", config_snapshot.get("roi_offset_y", 0)) or 0),
            ),
        }
        _results, roi_warnings = camera_system.apply_side_roi_settings(rois, restart_stream=True)
        warnings.extend(roi_warnings)
        apply_chunk = getattr(camera_system, "apply_chunk_settings", None)
        if callable(apply_chunk):
            warnings.extend(
                apply_chunk(
                    config_bool(config_snapshot, "chunk_data_enabled", False, False),
                    config_snapshot.get("chunk_selectors"),
                )
            )
        if hasattr(camera_system, "config"):
            camera_system.config.update(config_snapshot)
        camera_system.trigger_source = str(config_snapshot.get("trigger_source", camera_system.trigger_source))
        camera_system.require_hardware_trigger = False
        camera_system.hardware_sync_enabled = False
        camera_system.timestamp_reject_enabled = config_bool(config_snapshot, "timestamp_reject_enabled", True, False)
        camera_system.max_camera_timestamp_delta = int(config_snapshot.get("max_camera_timestamp_delta", 0) or 0)
        camera_system.max_host_timestamp_delta = int(
            config_snapshot.get("max_host_timestamp_delta", DEFAULT_HOST_TIMESTAMP_DELTA_NS) or 0
        )
        if camera_system.max_camera_timestamp_delta <= 0 and camera_system.max_host_timestamp_delta <= 0:
            camera_system.timestamp_reject_enabled = False
        return warnings

    def _save_current_capture_settings(self) -> None:
        values = self._current_parameter_config()
        values["interval_capture_seconds"] = float(self.interval_seconds_var.get() or 0)
        values["interval_capture_count"] = optional_int_text(self.interval_limit_var.get())
        values["record_fps"] = optional_positive_fps(self.record_fps_var.get()) or 0.0
        values["record_max_seconds"] = max(float(self.record_max_seconds_var.get() or 0), 0.0)
        try:
            dic_record_fps = self._dic_record_fps_from_entry()
        except ValueError:
            dic_record_fps = config_float(self.config.get("dic_capture", {}), "record_fps", DIC_CAPTURE_CONFIG["record_fps"])
        try:
            dic_pixel_format = self._dic_pixel_format_from_entry()
        except ValueError:
            dic_pixel_format = str(self.config.get("dic_capture", {}).get("pixel_format", DIC_CAPTURE_CONFIG["pixel_format"]))
        dic_capture = dict(self.config.get("dic_capture", {}) if isinstance(self.config.get("dic_capture"), dict) else {})
        dic_capture["record_fps"] = dic_record_fps
        dic_capture["pixel_format"] = dic_pixel_format
        high_bit_depth = dic_pixel_format != "Mono8"
        dic_capture["save_raw_frames"] = high_bit_depth
        dic_capture["record_force_image_format"] = not high_bit_depth
        dic_capture["image_format"] = "png"
        dic_capture["viewable_sidecar_enabled"] = True
        dic_capture["viewable_sidecar_format"] = "png"
        if high_bit_depth:
            dic_capture["raw_frame_format"] = str(dic_capture.get("raw_frame_format") or "tiff16")
        values["dic_capture"] = dic_capture
        self._update_config(values, save=False)
        self._set_cached_trigger_source(str(values.get("trigger_source", "Software")))
        self._ensure_recording_config_defaults()
        save_config(self.config)

    def _ensure_recording_config_defaults(self) -> None:
        defaults = {
            "auto_make_mp4": True,
            "video_codec": "mp4v",
            "video_bitrate_kbps": 8000,
            "video_quality_crf": 23,
            "video_preset": "medium",
            "use_nvenc": False,
            "record_save_image_sequence": False,
            "record_split_interval_seconds": 0,
            "record_split_size_gb": 1.0,
            "record_max_seconds": 0,
        }
        for key, value in defaults.items():
            self.config.setdefault(key, value)
        self._ensure_reliability_config_defaults()

    def _load_vars_from_config(self) -> None:
        self.config.update(safe_trigger_config(self._config_snapshot()))
        self._ensure_default_full_resolution()
        self.trigger_source_var.set(display_trigger_source(self.config.get("trigger_source", "Software")))
        self._set_cached_trigger_source(str(self.config.get("trigger_source", "Software")))
        self.exposure_auto_var.set(str(self.config.get("exposure_auto", "Off")))
        self.exposure_time_var.set(str(self.config.get("exposure_time_us", 10000.0)))
        self.auto_exposure_lower_var.set(optional_config_text(self.config, "auto_exposure_lower_limit", "100.0"))
        self.auto_exposure_upper_var.set(optional_config_text(self.config, "auto_exposure_upper_limit", "100000.0"))
        self.gain_auto_var.set(str(self.config.get("gain_auto", "Off")))
        self.gain_var.set(str(self.config.get("gain", 0.0)))
        self.auto_gain_lower_var.set(optional_config_text(self.config, "auto_gain_lower_limit", "0.0"))
        self.auto_gain_upper_var.set(optional_config_text(self.config, "auto_gain_upper_limit", "15.0"))
        self.black_level_var.set(optional_config_text(self.config, "black_level", ""))
        self.digital_shift_var.set(optional_config_text(self.config, "digital_shift", ""))
        self.gamma_var.set(optional_config_text(self.config, "gamma", ""))
        self.balance_auto_var.set(str(self.config.get("balance_white_auto", "Off")))
        self.balance_red_var.set(optional_config_text(self.config, "balance_ratio_red", ""))
        self.balance_green_var.set(optional_config_text(self.config, "balance_ratio_green", ""))
        self.balance_blue_var.set(optional_config_text(self.config, "balance_ratio_blue", ""))
        self.left_roi_width_var.set(str(self.config.get("left_roi_width", self.config.get("roi_width", CAPTURE_WIDTH))))
        self.left_roi_height_var.set(str(self.config.get("left_roi_height", self.config.get("roi_height", CAPTURE_HEIGHT))))
        self.left_roi_offset_x_var.set(str(self.config.get("left_roi_offset_x", self.config.get("roi_offset_x", 0))))
        self.left_roi_offset_y_var.set(str(self.config.get("left_roi_offset_y", self.config.get("roi_offset_y", 0))))
        self.right_roi_width_var.set(str(self.config.get("right_roi_width", self.config.get("roi_width", CAPTURE_WIDTH))))
        self.right_roi_height_var.set(str(self.config.get("right_roi_height", self.config.get("roi_height", CAPTURE_HEIGHT))))
        self.right_roi_offset_x_var.set(str(self.config.get("right_roi_offset_x", self.config.get("roi_offset_x", 0))))
        self.right_roi_offset_y_var.set(str(self.config.get("right_roi_offset_y", self.config.get("roi_offset_y", 0))))
        self.record_max_seconds_var.set(optional_config_text(self.config, "record_max_seconds", "0"))

    def _format_apply_result(self, prefix: str, warnings: list[str]) -> str:
        if warnings:
            return prefix + "；" + "；".join(warnings)
        return prefix + f"到 {self._connected_camera_count()} 台相机。"

    def _side_roi_dimensions_from_config(self, config_snapshot: dict | None = None) -> tuple[tuple[int, int], tuple[int, int]]:
        config_snapshot = config_snapshot or self._config_snapshot()
        left = (
            int(config_snapshot.get("left_roi_width") or config_snapshot.get("roi_width") or CAPTURE_WIDTH),
            int(config_snapshot.get("left_roi_height") or config_snapshot.get("roi_height") or CAPTURE_HEIGHT),
        )
        right = (
            int(config_snapshot.get("right_roi_width") or config_snapshot.get("roi_width") or CAPTURE_WIDTH),
            int(config_snapshot.get("right_roi_height") or config_snapshot.get("roi_height") or CAPTURE_HEIGHT),
        )
        return left, right

    def _reset_stats(self) -> None:
        self._stat_last_time = time.perf_counter()
        self._stat_frames = 0
        self._actual_fps = 0.0
        self._last_left_frame = None
        self._last_right_frame = None
        self._drop_count = 0
        if hasattr(self, "left_pane") and hasattr(self, "right_pane"):
            self._set_performance_overlay("FPS -- | Drop -- | Delta --", "good")

    def _update_stats(self, left: CameraFrame | None, right: CameraFrame | None) -> None:
        now = time.perf_counter()
        self._stat_frames += 1
        if self._stat_last_time is not None:
            elapsed = now - self._stat_last_time
            if elapsed >= 1.0:
                self._actual_fps = self._stat_frames / elapsed
                self._stat_frames = 0
                self._stat_last_time = now
        left_dropped = 0
        right_dropped = 0
        if left is not None and self._last_left_frame is not None:
            left_step = left.frame_number - self._last_left_frame
            if left_step > 1:
                left_dropped = left_step - 1
        if right is not None and self._last_right_frame is not None:
            right_step = right.frame_number - self._last_right_frame
            if right_step > 1:
                right_dropped = right_step - 1
        self._drop_count += max(left_dropped, right_dropped)
        if left is not None:
            self._last_left_frame = left.frame_number
        if right is not None:
            self._last_right_frame = right.frame_number

    def _update_performance_display(self) -> None:
        write_lag, _write_warning, _skip_every_n, _skip_keep_frames = self._record_write_state_snapshot()
        if self._last_left_frame is None or self._last_right_frame is None:
            side = "L" if self._last_left_frame is not None else "R" if self._last_right_frame is not None else "--"
            text = f"FPS {self._actual_fps:4.1f} | Drop {self._drop_count} | {side} only"
            if self.recording:
                text += "\n" + self._record_overlay_suffix()
            self._set_performance_overlay(text, "warn" if self.recording and write_lag > 1.5 else "good")
            return
        frame_delta = self._last_left_frame - self._last_right_frame
        if self._drop_count > 0 or abs(frame_delta) > 3:
            status = "bad"
        elif abs(frame_delta) > 1 or self._actual_fps <= 0.5:
            status = "warn"
        else:
            status = "good"
        if self.recording and write_lag > 1.5:
            status = "warn" if write_lag <= 2.0 else "bad"
        text = f"FPS {self._actual_fps:4.1f} | Drop {self._drop_count} | Delta {frame_delta}"
        if self.recording:
            text += "\n" + self._record_overlay_suffix()
        self._set_performance_overlay(text, status)

    def _set_performance_overlay(self, text: str, status: str) -> None:
        self.left_pane.set_performance_overlay(text, status)
        self.right_pane.set_performance_overlay(text, status)

    def _status_with_stats(self, prefix: str) -> str:
        return prefix

    def _interval_status_text(
        self,
        interval_s: float,
        limit: int | None,
        latest_dir_name: str,
        elapsed_seconds: float,
    ) -> str:
        prefix = f"定时拍照中：已保存 {self.interval_count} 组"
        if limit is not None:
            remaining = max(int(limit) - int(self.interval_count), 0)
            eta = remaining * max(float(interval_s), 0.0)
            if self.interval_count > 0 and elapsed_seconds > 0:
                average_interval = elapsed_seconds / max(self.interval_count, 1)
                eta = remaining * max(average_interval, 0.0)
            prefix = f"定时拍照中：已保存 {self.interval_count}/{int(limit)} 组，剩余 {remaining} 组，预计 {format_duration(eta)}"
        return f"{prefix}；最近 {latest_dir_name}；间隔 {interval_s:g} 秒"

    def _record_overlay_suffix(self) -> str:
        elapsed = self._record_elapsed_seconds()
        free_gb = self._record_free_space_gb()
        write_lag, write_warning, skip_every_n, _skip_keep_frames = self._record_write_state_snapshot()
        suffix = f"已录 {format_duration(elapsed)} / 剩余空间 {free_gb:.1f} GB"
        max_seconds = max(config_float(self.config, "record_max_seconds", 0.0), 0.0)
        if max_seconds > 0:
            suffix += f" | 剩余时长 {format_duration(max_seconds - elapsed)}"
        if write_lag > 1.5:
            suffix += f" | Write lag {write_lag:.1f}x skip {skip_every_n}"
        if write_warning:
            suffix += f" | {write_warning}"
        return suffix

    def _record_status_text(self, target_fps: float | None, effective_fps: float, config_snapshot: dict | None = None) -> str:
        config_snapshot = config_snapshot or self._config_snapshot()
        elapsed = self._record_elapsed_seconds()
        free_gb = self._record_free_space_gb()
        write_lag, write_warning, skip_every_n, _skip_keep_frames = self._record_write_state_snapshot()
        record_count, record_saved_count = self._record_counter_values()
        target_text = f"{target_fps:g} fps" if target_fps is not None else "max"
        parts = [
            f"录像中：采集 {record_count} 组，保存 {record_saved_count} 组",
            f"目标 {target_text}，实际写入约 {effective_fps:g} fps",
            f"已录 {format_duration(elapsed)} / 剩余空间 {free_gb:.1f} GB",
        ]
        max_seconds = max(config_float(config_snapshot, "record_max_seconds", 0.0), 0.0)
        if max_seconds > 0:
            parts.append(f"剩余时长 {format_duration(max_seconds - elapsed)}")
        if write_lag > 1.5:
            parts.append(f"写入滞后 {write_lag:.1f}x，跳帧策略 {skip_every_n}")
        if write_warning:
            parts.append(write_warning)
        return "；".join(parts)

    def _notify_warning(self, key: str, message: str, log_only: bool = False) -> None:
        now = time.perf_counter()
        last = self._last_alert_times.get(key, 0.0)
        if now - last < 5.0:
            return
        self._last_alert_times[key] = now
        LOGGER.warning(message)
        if not log_only:
            self.ui_queue.put(("status", message))
            if config_bool(self.config, "sound_alert_enabled", True, True):
                self._play_alert_sound()

    def _play_alert_sound(self) -> None:
        try:
            if winsound is not None:
                winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        except Exception:
            LOGGER.exception("声音告警失败")

    def _handle_capture_exception(self, exc: Exception, mode: str, consecutive_timeouts: int) -> bool:
        message = f"{mode} 采集异常：{exc}"
        if isinstance(exc, FrameSyncError):
            self._disable_timestamp_reject_after_sync_error(str(exc))
            self._notify_warning(f"{mode}_sync_error", message)
            return True
        if isinstance(exc, FrameTimeoutError):
            threshold = max(config_int(self.config, "consecutive_timeout_alert_threshold", 3), 1)
            if consecutive_timeouts >= threshold:
                detail = f"{mode} 连续超时 {consecutive_timeouts} 次：{exc}"
                if self._cached_trigger_source() == "Line0":
                    detail += "；当前为 Line0 外触发模式，请检查触发线、共地、触发电平/上升沿和脉冲频率。"
                self._notify_warning(f"{mode}_timeouts", detail)
                if mode == "preview" or self._cached_trigger_source() != "Line0":
                    return self._attempt_reconnect(mode)
            LOGGER.warning(message)
            return True
        self._notify_warning(f"{mode}_capture_error", message)
        return self._attempt_reconnect(mode)

    def _disable_timestamp_reject_after_sync_error(self, detail: str) -> None:
        if self.camera_system is not None:
            self.camera_system.timestamp_reject_enabled = False
        if config_bool(self.config, "timestamp_reject_enabled", False, False):
            self._update_config(
                {
                    "timestamp_reject_enabled": False,
                    "max_camera_timestamp_delta": 0,
                    "max_host_timestamp_delta": 0,
                }
            )
        LOGGER.warning("Timestamp reject disabled after FrameSyncError; continuing without reconnect: %s", detail)

    def _attempt_reconnect(self, mode: str) -> bool:
        if self._closing or not config_bool(self.config, "auto_reconnect_enabled", True, True):
            return False
        if self._reconnecting:
            return False
        self._reconnecting = True
        max_attempts = max(config_int(self.config, "auto_reconnect_max_attempts", 5), 1)
        delay = max(config_float(self.config, "auto_reconnect_initial_delay_seconds", 1.0), 0.1)
        max_delay = max(config_float(self.config, "auto_reconnect_max_delay_seconds", 16.0), delay)
        try:
            for attempt in range(1, max_attempts + 1):
                if self._closing:
                    return False
                if mode == "record" and not self.recording:
                    return False
                if mode == "preview" and not self.previewing:
                    return False
                if mode == "interval" and not self.interval_capturing:
                    return False
                self._last_reconnect_message = f"相机异常，正在自动重连 {attempt}/{max_attempts}..."
                self.ui_queue.put(("status", self._last_reconnect_message))
                LOGGER.warning(self._last_reconnect_message)
                if self._wait_reconnect_delay(delay):
                    return False
                try:
                    if self.camera_system is not None:
                        try:
                            self.camera_system.close()
                        except Exception:
                            LOGGER.exception("重连前关闭旧相机失败")
                    system_config = self._config_snapshot()
                    system_config["allow_single_camera"] = True
                    system = StereoCameraSystem(system_config)
                    left_info, right_info = system.connect()
                    self.camera_system = system
                    self._apply_fixed_camera_timestamp_offset()
                    self._load_field_correction_references()
                    self._add_record_reconnect()
                    self.ui_queue.put(("connected", (left_info, right_info)))
                    self.ui_queue.put(("status", "相机自动重连成功，采集任务继续。"))
                    LOGGER.info("相机自动重连成功")
                    return True
                except Exception as reconnect_exc:
                    LOGGER.exception("相机自动重连失败")
                    self.ui_queue.put(("status", f"自动重连失败 {attempt}/{max_attempts}：{reconnect_exc}"))
                    delay = min(delay * 2.0, max_delay)
            self._notify_warning("auto_reconnect_failed", "自动重连已达到最大次数，采集已停止。")
            if mode == "record":
                self._set_record_stop_reason("reconnect_failed")
                self.recording = False
            elif mode == "preview":
                self.previewing = False
            elif mode == "interval":
                self.interval_capturing = False
            return False
        finally:
            self._reconnecting = False

    def _wait_reconnect_delay(self, delay: float) -> bool:
        deadline = time.perf_counter() + max(delay, 0.0)
        while time.perf_counter() < deadline:
            if self._closing:
                return True
            time.sleep(min(0.1, max(deadline - time.perf_counter(), 0.0)))
        return self._closing

    def _record_elapsed_seconds(self) -> float:
        record_started_at = self._record_started_at_snapshot()
        if record_started_at is None:
            return 0.0
        return max(time.perf_counter() - record_started_at, 0.0)

    def _record_free_space_gb(self) -> float:
        root = self.record_dir if self.record_dir is not None else resolve_output_root(self.config)
        try:
            usage = shutil.disk_usage(root)
        except FileNotFoundError:
            usage = shutil.disk_usage(resolve_output_root(self.config))
        return usage.free / 1024**3

    def _should_record_save_frame(self, frame_index: int) -> bool:
        _write_lag, _write_warning, skip_every_n, skip_keep_frames = self._record_write_state_snapshot()
        if skip_every_n <= 1:
            return True
        keep = min(max(skip_keep_frames, 1), skip_every_n)
        return (frame_index - 1) % skip_every_n < keep

    def _update_record_skip_strategy(self) -> None:
        with self._state_lock:
            self._update_record_skip_strategy_locked()

    def _update_record_skip_strategy_locked(self) -> None:
        if not config_bool(self.config, "record_drop_frames_on_write_lag", False, False):
            self._record_skip_every_n = 1
            self._record_skip_keep_frames = 1
            if self._record_write_lag > 2.5:
                self._record_write_warning = "Disk writes are severely behind; frame dropping is disabled."
            elif self._record_write_lag > 1.5:
                self._record_write_warning = "Disk writes are behind; frame dropping is disabled."
            elif self._record_write_warning.startswith("Disk writes are"):
                self._record_write_warning = ""
            return
        if self._record_write_lag > 2.5:
            self._record_skip_every_n = 3
            self._record_skip_keep_frames = 1
            self._record_write_warning = "磁盘写入严重滞后，已每3帧写1帧"
        elif self._record_write_lag > 1.5:
            self._record_skip_every_n = 2
            self._record_skip_keep_frames = 1
            self._record_write_warning = "磁盘写入跟不上，已每2帧写1帧"
        else:
            self._record_skip_every_n = 1
            self._record_skip_keep_frames = 1
            if self._record_write_warning.startswith("磁盘写入"):
                self._record_write_warning = ""

    def _effective_record_fps(self, target_fps: float | None) -> float:
        if target_fps is None:
            elapsed = self._record_elapsed_seconds()
            _record_count, record_saved_count = self._record_counter_values()
            return record_saved_count / elapsed if elapsed > 0 else 0.0
        _write_lag, _write_warning, skip_every_n, skip_keep_frames = self._record_write_state_snapshot()
        if skip_every_n > 1:
            keep = min(max(skip_keep_frames, 1), skip_every_n)
            return target_fps * keep / skip_every_n
        return target_fps

    def _record_output_fps(self, target_fps: float) -> float:
        elapsed = self._record_elapsed_seconds()
        _record_count, record_saved_count = self._record_counter_values()
        if elapsed > 0 and record_saved_count > 0:
            return max(record_saved_count / elapsed, 0.1)
        if record_saved_count <= 0:
            return 0.0
        return self._effective_record_fps(target_fps)

    def _record_skipped(self, reason: str, frame_index: int) -> None:
        if frame_index <= 0:
            return
        with self._record_stats_lock:
            self._record_skipped_count += 1
            self._record_skip_reasons[reason] = self._record_skip_reasons.get(reason, 0) + 1
            bucket = self._record_second_bucket_locked(self._record_second_index_locked())
            bucket["skipped_frames"] = int(bucket.get("skipped_frames", 0) or 0) + 1
            if bucket.get("first_frame_index") is None:
                bucket["first_frame_index"] = int(frame_index)
            bucket["last_frame_index"] = int(frame_index)
            reasons = bucket.setdefault("drop_reasons", {})
            if isinstance(reasons, dict):
                reasons[reason] = int(reasons.get(reason, 0) or 0) + 1
            if len(self._record_skipped_frames) < 1000:
                self._record_skipped_frames.append(
                    {
                        "index": frame_index,
                        "reason": reason,
                        "time": time.time(),
                    }
                )
        if reason == "write_skip_strategy":
            self._notify_warning("record_frame_skip", f"录像写入压力较大，已记录跳过帧 {frame_index}", log_only=True)

    def _check_disk_space_during_recording(self, fps: float, config_snapshot: dict) -> None:
        now = time.perf_counter()
        interval_s = max(config_float(config_snapshot, "record_disk_check_interval_seconds", 10.0), 1.0)
        if now - self._record_last_disk_check < interval_s:
            return
        self._record_last_disk_check = now
        root = self.record_dir if self.record_dir is not None else resolve_output_root(config_snapshot)
        usage = shutil.disk_usage(root)
        free_gb = usage.free / 1024**3
        min_free_gb = max(config_float(config_snapshot, "record_disk_min_free_gb", 2.0), 0.0)
        warning_minutes = max(config_float(config_snapshot, "record_disk_warning_minutes", 2.0), 0.1)
        left_size, right_size = self._side_roi_dimensions_from_config(config_snapshot)
        bytes_per_second = (
            estimate_frame_bytes(config_snapshot, left_size[0], left_size[1])
            + estimate_frame_bytes(config_snapshot, right_size[0], right_size[1])
        ) * fps
        seconds_left = usage.free / max(bytes_per_second, 1)
        if free_gb <= min_free_gb or seconds_left <= warning_minutes * 60.0:
            self._add_record_disk_warning()
            message = f"磁盘空间预警：剩余 {free_gb:.1f} GB，按当前设置约可录 {format_duration(seconds_left)}。"
            self._notify_warning("record_low_disk", message)
            if config_bool(config_snapshot, "record_stop_on_low_disk", True, True) and free_gb <= min_free_gb:
                self._set_record_stop_reason("low_disk_space")
                self.recording = False

    def _record_segment_dir(self, side: str, segment_index: int) -> str:
        if segment_index <= 1:
            return side
        return f"{side}_part{segment_index:03d}"

    def _record_segment_video_path(self, side: str, segment_index: int) -> Path:
        record_dir = self._require_record_dir()
        suffix = "" if segment_index <= 1 else f"_part{segment_index:03d}"
        return record_dir / f"{side}{suffix}.mp4"

    def _advance_record_segment_if_needed(self, current_segment_index: int, config_snapshot: dict) -> None:
        active_segment_index, segment_started_at, segment_bytes, _record_dir = self._record_segment_snapshot()
        if current_segment_index != active_segment_index:
            return
        split_seconds = max(config_float(config_snapshot, "record_split_interval_seconds", 600.0), 0.0)
        split_size_gb = max(config_float(config_snapshot, "record_split_size_gb", 4.0), 0.0)
        elapsed = time.perf_counter() - segment_started_at
        should_split = (split_seconds > 0 and elapsed >= split_seconds) or (
            split_size_gb > 0 and segment_bytes >= split_size_gb * 1024**3
        )
        if not should_split:
            return
        advanced = self._advance_record_segment_state(current_segment_index)
        if advanced is None:
            return
        new_segment_index, record_dir = advanced
        if record_dir is not None and config_bool(config_snapshot, "record_save_image_sequence", False, False):
            for side in ("left", "right"):
                (record_dir / self._record_segment_dir(side, new_segment_index)).mkdir(parents=True, exist_ok=True)

    def _finalize_recording_videos(
        self,
        record_dir: Path,
        fps: float,
        frames: list[dict],
        video_outputs: dict[str, list[str]],
        config_snapshot: dict,
    ) -> list[str]:
        if config_bool(config_snapshot, "auto_make_mp4", True, True) and config_bool(
            config_snapshot, "record_save_image_sequence", False, False
        ):
            total_units = self._mp4_progress_total_units(frames)
            progress_done = 0

            def progress(current: int = 0, message: str = "") -> None:
                units = min(max(progress_done + max(int(current), 0), 0), max(total_units, 1))
                percent = units / max(total_units, 1) * 100.0
                self.ui_queue.put(
                    (
                        "mp4_progress",
                        {
                            "percent": percent,
                            "message": message or f"合成 MP4 {percent:.0f}%",
                        },
                    )
                )

            def segment_done(count: int, message: str = "") -> None:
                nonlocal progress_done
                progress_done = min(progress_done + max(int(count), 0), max(total_units, 1))
                progress(0, message)

            progress(0, "MP4 0%")
            ffmpeg_outputs = self._try_make_mp4_from_frames(
                record_dir,
                fps,
                frames,
                config_snapshot,
                progress_callback=progress,
                segment_done_callback=segment_done,
            )
            for side, paths in ffmpeg_outputs.items():
                if paths:
                    video_outputs[side] = [str(path) for path in paths]
            missing_sides = [side for side in ("left", "right") if not video_outputs[side]]
            if missing_sides:
                opencv_outputs = self._try_make_mp4_from_frames_opencv(
                    record_dir,
                    fps,
                    frames,
                    config_snapshot,
                    missing_sides,
                    progress_callback=progress,
                    segment_done_callback=segment_done,
                )
                for side, paths in opencv_outputs.items():
                    if paths:
                        video_outputs[side] = [str(path) for path in paths]
            progress_done = max(total_units, 1)
            progress(0, "MP4 100%")
        names: list[str] = []
        for side in ("left", "right"):
            for path in video_outputs[side]:
                names.append(Path(path).name)
        return names

    def _mp4_progress_total_units(self, frames: list[dict]) -> int:
        total = 0
        for frame in frames:
            for side in ("left", "right"):
                if frame.get(f"{side}_path") is not None:
                    total += 1
        return max(total, 1)

    def _try_make_mp4_from_frames_opencv(
        self,
        record_dir: Path,
        fps: float,
        frames: list[dict],
        config_snapshot: dict,
        sides: list[str],
        progress_callback=None,
        segment_done_callback=None,
    ) -> dict[str, list[Path]]:
        if not frames:
            return {side: [] for side in sides}
        ext = image_extension(config_snapshot)
        outputs: dict[str, list[Path]] = {side: [] for side in sides}
        segment_indices = sorted({int(frame["segment_index"]) for frame in frames})
        for segment_index in segment_indices:
            for side in sides:
                segment_frames = sorted(
                    (
                        frame
                        for frame in frames
                        if int(frame["segment_index"]) == segment_index and frame.get(f"{side}_path") is not None
                    ),
                    key=lambda frame: int(frame.get("saved_index", 0) or 0),
                )
                if not segment_frames:
                    continue
                first_path = Path(str(segment_frames[0].get(f"{side}_path")))
                first_image = self._cv2_imread_unicode(first_path, cv2.IMREAD_UNCHANGED)
                if first_image is None:
                    LOGGER.warning("OpenCV MP4 generation could not read first frame: %s", first_path)
                    continue
                height, width = first_image.shape[:2]
                output = self._record_segment_video_path(side, segment_index)
                output.parent.mkdir(parents=True, exist_ok=True)
                temp_output = self._opencv_temp_video_path(output)
                writer, codec_name = self._create_video_writer_v2(
                    temp_output,
                    fps,
                    Image.new("L", (width, height)),
                    config_snapshot,
                )
                started = time.perf_counter()
                written = 0
                segment_total = len(segment_frames)
                try:
                    for frame in segment_frames:
                        path = Path(str(frame.get(f"{side}_path")))
                        if path.suffix.lower().lstrip(".") != ext:
                            continue
                        image = self._cv2_imread_unicode(path, cv2.IMREAD_UNCHANGED)
                        if image is None:
                            continue
                        if image.ndim == 2:
                            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                        elif image.shape[2] == 4:
                            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                        writer.write(image)
                        written += 1
                        if progress_callback is not None and (written % 5 == 0 or written == segment_total):
                            progress_callback(
                                written,
                                f"MP4 {side} part {segment_index} {written}/{segment_total}",
                            )
                finally:
                    writer.release()
                if written > 0 and temp_output.exists():
                    try:
                        output.unlink(missing_ok=True)
                        shutil.move(str(temp_output), str(output))
                    except OSError:
                        LOGGER.exception("OpenCV MP4 generation could not move %s to %s.", temp_output, output)
                        continue
                else:
                    temp_output.unlink(missing_ok=True)
                if written > 0 and output.exists():
                    outputs[side].append(output)
                    if codec_name != str(config_snapshot.get("video_codec", "mp4v")):
                        self._set_record_write_warning(f"Video codec fallback: {codec_name}")
                    LOGGER.info(
                        "OpenCV MP4 generation wrote %s from %d %s frames in %.1fs.",
                        output,
                        written,
                        ext.upper(),
                        time.perf_counter() - started,
                    )
                if segment_done_callback is not None:
                    segment_done_callback(written, f"MP4 {side} part {segment_index} done")
        return outputs

    def _cv2_imread_unicode(self, path: Path, flags: int) -> np.ndarray | None:
        try:
            data = np.fromfile(str(path), dtype=np.uint8)
        except OSError:
            return None
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)

    def _opencv_temp_video_path(self, final_path: Path) -> Path:
        temp_dir = Path(tempfile.gettempdir()) / "mvss_capture_mp4"
        temp_dir.mkdir(parents=True, exist_ok=True)
        return temp_dir / f"{timestamp_ms()}_{final_path.stem}.mp4"

    def _try_make_mp4_from_frames(
        self,
        record_dir: Path,
        fps: float,
        frames: list[dict],
        config_snapshot: dict,
        progress_callback=None,
        segment_done_callback=None,
    ) -> dict[str, list[Path]]:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg or not frames:
            return {"left": [], "right": []}
        ext = image_extension(config_snapshot)
        outputs: dict[str, list[Path]] = {"left": [], "right": []}
        segment_indices = sorted({int(frame["segment_index"]) for frame in frames})
        for segment_index in segment_indices:
            for side in ("left", "right"):
                segment_frames = [
                    frame
                    for frame in frames
                    if int(frame["segment_index"]) == segment_index and frame.get(f"{side}_path") is not None
                ]
                if not segment_frames:
                    continue
                if progress_callback is not None:
                    progress_callback(0, f"MP4 {side} part {segment_index}")
                frame_dir = record_dir / self._record_segment_dir(side, segment_index)
                if not frame_dir.exists() or not any(frame_dir.glob(f"{side}_*.{ext}")):
                    continue
                start_number = min(int(frame["saved_index"]) for frame in segment_frames)
                contiguous, temp_dir, skipped = self._prepare_ffmpeg_sequence(frame_dir, side, ext, segment_frames)
                if skipped:
                    continue
                input_dir = temp_dir or frame_dir
                pattern = input_dir / f"{side}_%06d.{ext}"
                if contiguous:
                    start_number = 1
                output = self._record_segment_video_path(side, segment_index)
                try:
                    command = self._ffmpeg_mp4_command(ffmpeg, pattern, output, fps, start_number, config_snapshot)
                    result = subprocess.run(
                        command,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        check=False,
                    )
                    if result.returncode == 0 and output.exists():
                        outputs[side].append(output)
                        if segment_done_callback is not None:
                            segment_done_callback(len(segment_frames), f"MP4 {side} part {segment_index} done")
                    elif config_bool(config_snapshot, "use_nvenc", False, False):
                        if result.stderr:
                            LOGGER.warning("ffmpeg NVENC MP4 generation failed for %s: %s", output, result.stderr[-2000:])
                        fallback = self._ffmpeg_mp4_command(
                            ffmpeg,
                            pattern,
                            output,
                            fps,
                            start_number,
                            config_snapshot,
                            force_software=True,
                        )
                        result = subprocess.run(
                            fallback,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE,
                            text=True,
                            encoding="utf-8",
                            errors="replace",
                            check=False,
                        )
                        if result.returncode == 0 and output.exists():
                            outputs[side].append(output)
                            self._set_record_write_warning("NVENC失败，已回退软件编码")
                            if segment_done_callback is not None:
                                segment_done_callback(len(segment_frames), f"MP4 {side} part {segment_index} done")
                        elif result.stderr:
                            LOGGER.warning("ffmpeg fallback MP4 generation failed for %s: %s", output, result.stderr[-2000:])
                    elif result.stderr:
                        LOGGER.warning("ffmpeg MP4 generation failed for %s: %s", output, result.stderr[-2000:])
                finally:
                    if temp_dir is not None:
                        shutil.rmtree(temp_dir, ignore_errors=True)
        return outputs

    def _prepare_ffmpeg_sequence(
        self,
        frame_dir: Path,
        side: str,
        ext: str,
        segment_frames: list[dict],
    ) -> tuple[bool, Path | None, bool]:
        saved_indices = sorted(int(frame["saved_index"]) for frame in segment_frames)
        if not saved_indices:
            return False, None, True
        expected = list(range(saved_indices[0], saved_indices[0] + len(saved_indices)))
        if saved_indices == expected:
            return False, None, False
        missing_count = saved_indices[-1] - saved_indices[0] + 1 - len(saved_indices)
        gap_warning_threshold = max(config_int(self.config, "ffmpeg_sequence_gap_warning_threshold", 300), 0)
        if gap_warning_threshold and missing_count >= gap_warning_threshold:
            self._notify_warning(
                "ffmpeg_sequence_many_gaps",
                f"录像帧序列缺号 {missing_count} 帧，MP4 合成需要临时重排文件，可能占用较多磁盘空间。",
                log_only=True,
            )
        sources = [frame_dir / f"{side}_{saved_index:06d}.{ext}" for saved_index in saved_indices]
        existing_sources = [src for src in sources if src.exists()]
        required_bytes = sum(src.stat().st_size for src in existing_sources)
        free_bytes = shutil.disk_usage(frame_dir).free
        if required_bytes > 0 and free_bytes < required_bytes * 1.2:
            LOGGER.warning(
                "Not enough free space to renumber ffmpeg sequence for %s: need %.1f GB, free %.1f GB; skipping MP4 generation.",
                frame_dir,
                required_bytes * 1.2 / 1024**3,
                free_bytes / 1024**3,
            )
            self._set_record_write_warning("磁盘空间不足，已跳过不连续帧序列的 MP4 合成")
            return False, None, True
        temp_dir = frame_dir / "_mp4_sequence_tmp"
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        temp_dir.mkdir(parents=True, exist_ok=True)
        for output_index, saved_index in enumerate(saved_indices, start=1):
            src = frame_dir / f"{side}_{saved_index:06d}.{ext}"
            dst = temp_dir / f"{side}_{output_index:06d}.{ext}"
            if src.exists():
                shutil.copy2(src, dst)
        return True, temp_dir, False

    def _ffmpeg_mp4_command(
        self,
        ffmpeg: str,
        input_pattern: Path,
        output_path: Path,
        fps: float,
        start_number: int = 1,
        config_snapshot: dict | None = None,
        force_software: bool = False,
    ) -> list[str]:
        config_snapshot = config_snapshot or self._config_snapshot()
        bitrate = max(config_int(config_snapshot, "video_bitrate_kbps", 8000), 1)
        crf = max(config_int(config_snapshot, "video_quality_crf", 23), 0)
        preset = str(config_snapshot.get("video_preset", "medium"))
        codec = str(config_snapshot.get("video_codec", "mp4v")).strip().lower()
        use_nvenc = config_bool(config_snapshot, "use_nvenc", False, False) and not force_software
        command = [
            ffmpeg,
            "-y",
            "-framerate",
            f"{fps:g}",
            "-start_number",
            str(start_number),
            "-i",
            str(input_pattern),
            "-pix_fmt",
            "yuv420p",
        ]
        if force_software:
            command.extend(["-c:v", "mpeg4", "-q:v", "3", "-b:v", f"{bitrate}k", "-movflags", "+faststart"])
        elif use_nvenc:
            command.extend(["-c:v", "h264_nvenc", "-b:v", f"{bitrate}k", "-preset", preset, "-movflags", "+faststart"])
        elif codec in {"h264", "h264_nvenc", "avc1", "libx264"}:
            command.extend(["-c:v", "libx264", "-crf", str(crf), "-preset", preset, "-movflags", "+faststart"])
        else:
            command.extend(["-c:v", "mpeg4", "-q:v", "3", "-b:v", f"{bitrate}k", "-movflags", "+faststart"])
        command.append(str(output_path))
        return command

    def _capture_timeout_message(self, exc: object, count: int) -> str:
        if self._cached_trigger_source() == "Line0":
            return (
                f"实时采集中：等待 Line0 外触发；连续超时 {count} 次。"
                "请检查外部脉冲、Line0 接线、共地和触发边沿设置。"
            )
        return f"实时采集中：软件触发后未收到图像；连续超时 {count} 次。{exc}"

    def _check_disk_space_for_recording(self, config_snapshot: dict) -> bool:
        save_root = self.project_manager.output_root_for_mode("videos")
        save_root.mkdir(parents=True, exist_ok=True)
        plan = self._run_record_preflight(
            config_snapshot,
            run_benchmark=config_bool(config_snapshot, "record_disk_benchmark_enabled", True, True),
        )
        usage_free = int(plan.get("free_bytes", 0) or 0)
        required_mbps = float(plan.get("required_mbps") or 0.0)
        estimated_one_minute = int(plan.get("one_minute_bytes", 0) or 0)
        if usage_free < estimated_one_minute:
            messagebox.showerror(
                "磁盘空间不足",
                f"当前可用空间约 {usage_free / 1024**3:.1f} GB，按当前设置录制 1 分钟预计需要 "
                f"{estimated_one_minute / 1024**3:.1f} GB。",
            )
            return False
        if usage_free < estimated_one_minute * 3:
            if not messagebox.askyesno(
                "磁盘空间偏低",
                f"当前可用空间约 {usage_free / 1024**3:.1f} GB，按当前设置录制 1 分钟预计需要 "
                f"{estimated_one_minute / 1024**3:.1f} GB。是否继续？",
            ):
                return False
        measured_mbps = float(plan.get("measured_mbps") or 0.0)
        margin = float(plan.get("margin") or 1.0)
        if config_bool(config_snapshot, "record_preflight_prompt_enabled", True, True):
            speed_text = f"{measured_mbps:.1f} MB/s" if measured_mbps > 0 else "未测速"
            suggested = float(plan.get("suggested_max_fps") or 0.0)
            suggested_text = f"{suggested:.1f} fps" if suggested > 0 else "需先测速"
            if not messagebox.askyesno(
                "录像前性能评估",
                "当前配置评估结果：\n"
                f"- 预计需要：{required_mbps:.1f} MB/s\n"
                f"- 磁盘测速：{speed_text}\n"
                f"- 建议最高 FPS：{suggested_text}\n"
                f"- 录制 1 分钟约占用：{estimated_one_minute / 1024**3:.2f} GB\n\n"
                "是否开始录像？",
            ):
                return False
        if measured_mbps > 0 and measured_mbps < required_mbps * margin:
            if not messagebox.askyesno(
                "写入速度可能不足",
                f"当前设置预计需要 {required_mbps:.1f} MB/s，实测约 {measured_mbps:.1f} MB/s。"
                f"\n建议最高 FPS：{float(plan.get('suggested_max_fps') or 0.0):.1f}。继续录像可能丢帧，是否继续？",
            ):
                return False
        return True

    def _checksum_algorithm(self, config_snapshot: dict | None = None) -> str:
        config_snapshot = config_snapshot or self._config_snapshot()
        algorithm = str(config_snapshot.get("record_checksum_algorithm", "sha256")).strip().lower()
        return "md5" if algorithm == "md5" else "sha256"

    def _file_checksum(self, path: Path, config_snapshot: dict | None = None) -> str:
        digest = hashlib.new(self._checksum_algorithm(config_snapshot))
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _disk_used_bytes(self, path: Path) -> int:
        try:
            root = path if path.exists() else path.parent
            usage = shutil.disk_usage(root)
            return usage.total - usage.free
        except Exception:
            return 0

    def _directory_size_bytes(self, path: Path) -> int:
        if not path.exists():
            return 0
        total = 0
        for item in path.rglob("*"):
            if item.is_file():
                try:
                    total += item.stat().st_size
                except OSError:
                    continue
        return total

    def _build_record_summary(
        self,
        record_dir: Path,
        target_fps: float,
        output_fps: float,
        frames: list[dict],
    ) -> dict[str, object]:
        elapsed = self._record_elapsed_seconds()
        dir_bytes = self._directory_size_bytes(record_dir)
        disk_used_delta = max(self._disk_used_bytes(record_dir) - self._record_disk_usage_start, 0)
        stats = self._record_stats_snapshot()
        record_count = int(stats["record_count"])
        summary = {
            "total_frame_count": record_count,
            "saved_frame_count": stats["saved_frame_count"],
            "valid_frame_count": len(frames),
            "skipped_frame_count": stats["skipped_frame_count"],
            "skipped_frames": stats["skipped_frames"],
            "timeout_count": stats["timeout_count"],
            "error_count": stats["error_count"],
            "reconnect_count": stats["reconnect_count"],
            "disk_warning_count": stats["disk_warning_count"],
            "frame_number_gap_count": stats["frame_number_gap_count"],
            "target_fps": target_fps,
            "average_capture_fps": record_count / elapsed if elapsed > 0 else 0.0,
            "effective_video_fps": output_fps,
            "elapsed_seconds": elapsed,
            "directory_size_bytes": dir_bytes,
            "disk_used_delta_bytes": disk_used_delta,
            "stop_reason": stats["stop_reason"],
            "skip_reasons": stats["skip_reasons"],
            "per_second": stats.get("per_second", []),
            "average_write_seconds": self._average_record_write_seconds(frames),
            "disk_write_benchmark": self._record_disk_benchmark,
            "preflight": dict(self._record_preflight_plan),
        }
        with self._record_stats_lock:
            self._record_summary = summary
        return summary

    def _average_record_write_seconds(self, frames: list[dict]) -> float:
        values = []
        for frame in frames:
            try:
                value = float(frame.get("write_seconds") or 0.0)
            except (TypeError, ValueError):
                continue
            if value > 0:
                values.append(value)
        return float(np.mean(values)) if values else 0.0

    def _record_report_second_rows(self, summary: dict[str, object], frames: list[dict]) -> list[dict[str, object]]:
        rows_by_second: dict[int, dict[str, object]] = {}
        for raw in summary.get("per_second", []) if isinstance(summary.get("per_second"), list) else []:
            if not isinstance(raw, dict):
                continue
            second = int(raw.get("second", 0) or 0)
            drop_reasons = raw.get("drop_reasons") if isinstance(raw.get("drop_reasons"), dict) else {}
            write_samples = int(raw.get("write_samples", 0) or 0)
            write_total = float(raw.get("write_seconds_total", 0.0) or 0.0)
            rows_by_second[second] = {
                "second": second,
                "captured_frames": int(raw.get("captured_frames", 0) or 0),
                "saved_frames": int(raw.get("saved_frames", 0) or 0),
                "skipped_frames": int(raw.get("skipped_frames", 0) or 0),
                "timeout_count": int(raw.get("timeout_count", 0) or 0),
                "error_count": int(raw.get("error_count", 0) or 0),
                "frame_number_gaps": int(raw.get("frame_number_gaps", 0) or 0),
                "first_frame_index": raw.get("first_frame_index"),
                "last_frame_index": raw.get("last_frame_index"),
                "first_saved_index": raw.get("first_saved_index"),
                "last_saved_index": raw.get("last_saved_index"),
                "saved_mb": float(raw.get("saved_bytes", 0) or 0) / 1024 / 1024,
                "avg_write_ms": (write_total / write_samples * 1000.0) if write_samples > 0 else 0.0,
                "drop_reasons": "; ".join(f"{key}: {value}" for key, value in sorted(drop_reasons.items())),
            }
        for frame in frames:
            try:
                trigger = float(frame.get("trigger_time") or 0.0)
            except (TypeError, ValueError):
                trigger = 0.0
            second = int(max(trigger - float(frames[0].get("trigger_time") or trigger), 0.0)) if frames else 0
            rows_by_second.setdefault(
                second,
                {
                    "second": second,
                    "captured_frames": 0,
                    "saved_frames": 0,
                    "skipped_frames": 0,
                    "timeout_count": 0,
                    "error_count": 0,
                    "frame_number_gaps": 0,
                    "first_frame_index": None,
                    "last_frame_index": None,
                    "first_saved_index": None,
                    "last_saved_index": None,
                    "saved_mb": 0.0,
                    "avg_write_ms": 0.0,
                    "drop_reasons": "",
                },
            )
        return [rows_by_second[key] for key in sorted(rows_by_second)]

    def _write_record_reports(
        self,
        record_dir: Path,
        summary: dict[str, object],
        frames: list[dict],
        config_snapshot: dict,
    ) -> dict[str, str]:
        second_rows = self._record_report_second_rows(summary, frames)
        skipped_frames = summary.get("skipped_frames")
        if not isinstance(skipped_frames, list):
            skipped_frames = self._record_stats_snapshot().get("skipped_frames", [])
        csv_path = record_dir / "record_report.csv"
        html_path = record_dir / "record_report.html"
        csv_fields = [
            "section",
            "second",
            "captured_frames",
            "saved_frames",
            "skipped_frames",
            "timeout_count",
            "error_count",
            "frame_number_gaps",
            "first_frame_index",
            "last_frame_index",
            "first_saved_index",
            "last_saved_index",
            "saved_mb",
            "avg_write_ms",
            "drop_reasons",
            "frame_index",
            "reason",
            "time",
            "metric",
            "value",
        ]
        with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=csv_fields)
            writer.writeheader()
            summary_rows = {
                "total_frame_count": summary.get("total_frame_count", 0),
                "saved_frame_count": summary.get("saved_frame_count", 0),
                "skipped_frame_count": summary.get("skipped_frame_count", 0),
                "timeout_count": summary.get("timeout_count", 0),
                "frame_number_gap_count": summary.get("frame_number_gap_count", 0),
                "average_capture_fps": f"{float(summary.get('average_capture_fps') or 0.0):.4f}",
                "effective_video_fps": f"{float(summary.get('effective_video_fps') or 0.0):.4f}",
                "average_write_ms": f"{float(summary.get('average_write_seconds') or 0.0) * 1000.0:.4f}",
                "directory_size_bytes": summary.get("directory_size_bytes", 0),
                "stop_reason": summary.get("stop_reason", ""),
            }
            for metric_name, value in summary_rows.items():
                writer.writerow({"section": "summary", "metric": metric_name, "value": value})
            for row in second_rows:
                writer.writerow({"section": "per_second", **row})
            for skipped in skipped_frames if isinstance(skipped_frames, list) else []:
                if not isinstance(skipped, dict):
                    continue
                writer.writerow(
                    {
                        "section": "skipped_frame",
                        "frame_index": skipped.get("index"),
                        "reason": skipped.get("reason"),
                        "time": self._format_wall_time(skipped.get("time")),
                    }
                )

        html_path.write_text(
            self._record_report_html(record_dir, summary, second_rows, skipped_frames, config_snapshot),
            encoding="utf-8",
        )
        return {"csv": str(csv_path), "html": str(html_path)}

    def _format_wall_time(self, value: object) -> str:
        try:
            return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(value)))
        except (TypeError, ValueError, OSError):
            return ""

    def _record_report_html(
        self,
        record_dir: Path,
        summary: dict[str, object],
        second_rows: list[dict[str, object]],
        skipped_frames: object,
        config_snapshot: dict,
    ) -> str:
        benchmark = summary.get("disk_write_benchmark") if isinstance(summary.get("disk_write_benchmark"), dict) else {}
        preflight = summary.get("preflight") if isinstance(summary.get("preflight"), dict) else {}
        skip_reasons = summary.get("skip_reasons") if isinstance(summary.get("skip_reasons"), dict) else {}
        generated_at = time.strftime("%Y-%m-%d %H:%M:%S")

        def esc(value: object) -> str:
            return html.escape(str(value if value is not None else ""))

        def metric(label: str, value: object, suffix: str = "") -> str:
            return f"<div class=\"metric\"><span>{esc(label)}</span><strong>{esc(value)}{suffix}</strong></div>"

        second_body = "\n".join(
            "<tr>"
            f"<td>{int(row.get('second', 0))}</td>"
            f"<td>{int(row.get('captured_frames', 0))}</td>"
            f"<td>{int(row.get('saved_frames', 0))}</td>"
            f"<td>{int(row.get('skipped_frames', 0))}</td>"
            f"<td>{int(row.get('timeout_count', 0))}</td>"
            f"<td>{int(row.get('error_count', 0))}</td>"
            f"<td>{int(row.get('frame_number_gaps', 0))}</td>"
            f"<td>{esc(row.get('first_frame_index'))}</td>"
            f"<td>{esc(row.get('last_frame_index'))}</td>"
            f"<td>{float(row.get('saved_mb') or 0.0):.2f}</td>"
            f"<td>{float(row.get('avg_write_ms') or 0.0):.2f}</td>"
            f"<td>{esc(row.get('drop_reasons'))}</td>"
            "</tr>"
            for row in second_rows
        )
        skipped_body = "\n".join(
            "<tr>"
            f"<td>{esc(item.get('index'))}</td>"
            f"<td>{esc(item.get('reason'))}</td>"
            f"<td>{esc(self._format_wall_time(item.get('time')))}</td>"
            "</tr>"
            for item in skipped_frames
            if isinstance(item, dict)
        ) if isinstance(skipped_frames, list) else ""
        if not skipped_body:
            skipped_body = "<tr><td colspan=\"3\">无具体跳帧记录</td></tr>"
        reason_text = ", ".join(f"{key}: {value}" for key, value in sorted(skip_reasons.items())) if skip_reasons else "无"
        benchmark_text = (
            f"{float(benchmark.get('write_mbps') or 0.0):.1f} MB/s"
            if benchmark
            else "未测速或未记录"
        )
        suggested_text = (
            f"{float(preflight.get('suggested_max_fps') or 0.0):.1f} fps"
            if preflight and float(preflight.get("suggested_max_fps") or 0.0) > 0
            else "--"
        )
        return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>MVSS 录像报告 - {esc(record_dir.name)}</title>
<style>
body {{ margin: 0; background: #f5f7fa; color: #1f2933; font-family: "Microsoft YaHei UI", Arial, sans-serif; }}
main {{ max-width: 1180px; margin: 0 auto; padding: 28px; }}
h1 {{ margin: 0 0 4px; font-size: 24px; }}
h2 {{ margin: 28px 0 10px; font-size: 18px; }}
.sub {{ color: #66788a; margin-bottom: 18px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; }}
.metric {{ background: #ffffff; border: 1px solid #d8e0e8; border-radius: 8px; padding: 12px 14px; }}
.metric span {{ display: block; color: #66788a; font-size: 12px; margin-bottom: 6px; }}
.metric strong {{ font-size: 18px; }}
.panel {{ background: #ffffff; border: 1px solid #d8e0e8; border-radius: 8px; padding: 14px; margin-top: 12px; }}
table {{ width: 100%; border-collapse: collapse; background: #ffffff; border: 1px solid #d8e0e8; }}
th, td {{ padding: 8px 9px; border-bottom: 1px solid #e6edf3; text-align: left; font-size: 13px; }}
th {{ background: #edf3f8; color: #344250; position: sticky; top: 0; }}
.table-wrap {{ max-height: 520px; overflow: auto; border-radius: 8px; }}
.warn {{ color: #a15c00; }}
.ok {{ color: #14784e; }}
code {{ background: #edf3f8; padding: 2px 5px; border-radius: 4px; }}
</style>
</head>
<body>
<main>
<h1>MVSS 录像报告</h1>
<div class="sub">生成时间：{esc(generated_at)} | 目录：<code>{esc(record_dir)}</code></div>
<section class="grid">
{metric("总采集帧", summary.get("total_frame_count", 0))}
{metric("保存帧", summary.get("saved_frame_count", 0))}
{metric("跳帧/未保存", summary.get("skipped_frame_count", 0))}
{metric("超时", summary.get("timeout_count", 0))}
{metric("帧号缺口", summary.get("frame_number_gap_count", 0))}
{metric("平均采集 FPS", f"{float(summary.get('average_capture_fps') or 0.0):.2f}")}
{metric("有效视频 FPS", f"{float(summary.get('effective_video_fps') or 0.0):.2f}")}
{metric("平均写入耗时", f"{float(summary.get('average_write_seconds') or 0.0) * 1000.0:.2f}", " ms")}
{metric("磁盘测速", benchmark_text)}
{metric("建议最高 FPS", suggested_text)}
{metric("目录大小", format_bytes(summary.get("directory_size_bytes")))}
</section>
<section class="panel">
<strong>结论：</strong>
<span class="{esc('ok' if not skip_reasons and int(summary.get('timeout_count', 0) or 0) == 0 else 'warn')}">
停止原因 {esc(summary.get('stop_reason', '--'))}；跳帧原因：{esc(reason_text)}。
</span>
<br>
Output {esc('MP4 + image sequence' if config_bool(config_snapshot, 'record_save_image_sequence', False, False) else 'MP4 video')}; target FPS {float(summary.get('target_fps') or 0.0):.2f}; pixel format {esc(config_snapshot.get('pixel_format', '--'))}.
</section>
<h2>每秒采集/保存统计</h2>
<div class="table-wrap">
<table>
<thead><tr><th>秒</th><th>采集帧</th><th>保存帧</th><th>跳帧</th><th>超时</th><th>错误</th><th>帧号缺口</th><th>首帧</th><th>末帧</th><th>保存 MB</th><th>平均写入 ms</th><th>原因</th></tr></thead>
<tbody>{second_body or '<tr><td colspan="12">无秒级统计</td></tr>'}</tbody>
</table>
</div>
<h2>跳帧明细</h2>
<div class="table-wrap">
<table>
<thead><tr><th>帧号</th><th>原因</th><th>时间</th></tr></thead>
<tbody>{skipped_body}</tbody>
</table>
</div>
</main>
</body>
</html>
"""

    def _format_record_summary(self, summary: object) -> str:
        if not isinstance(summary, dict):
            return "摘要不可用"
        return (
            f"总帧 {summary.get('total_frame_count', 0)}，"
            f"有效 {summary.get('valid_frame_count', 0)}，"
            f"跳帧 {summary.get('skipped_frame_count', 0)}，"
            f"超时 {summary.get('timeout_count', 0)}，"
            f"平均 {float(summary.get('average_capture_fps') or 0.0):.1f} fps"
        )

    def _optional_entry_float(self, value: StringVar) -> float | None:
        return optional_float_text(value.get())

    def _save_photo_pair(
        self,
        left: CameraFrame | None,
        right: CameraFrame | None,
        trigger_time: float,
        mode: str,
        quality_report: dict[str, object] | None = None,
    ) -> Path:
        capture_id = timestamp_ms()
        left_dir, right_dir, meta_dir = self._project_capture_paths(capture_id)
        project_dir = self.project_manager.active_project_dir
        ext = image_extension(self.config)

        group_left = left_dir / f"{capture_id}_left.{ext}"
        group_right = right_dir / f"{capture_id}_right.{ext}"

        if left is not None:
            group_left = self._save_frame(left, group_left)
        if right is not None:
            group_right = self._save_frame(right, group_right)
        quality_metrics = self._quality_metrics_for_pair(left, right)
        focus = quality_metrics.get("focus") if isinstance(quality_metrics.get("focus"), dict) else {}
        left_exposure = quality_metrics.get("left_exposure") if isinstance(quality_metrics.get("left_exposure"), dict) else None
        right_exposure = quality_metrics.get("right_exposure") if isinstance(quality_metrics.get("right_exposure"), dict) else None
        calibration_board = quality_metrics.get("calibration_board")
        dic_speckle = quality_metrics.get("dic_speckle")
        self._write_meta(
            meta_dir / "meta.json",
            mode=mode,
            capture_id=capture_id,
            trigger_time=trigger_time,
            left=left,
            right=right,
            left_path=str(group_left) if left is not None else None,
            right_path=str(group_right) if right is not None else None,
            group_left_path=str(group_left) if left is not None else None,
            group_right_path=str(group_right) if right is not None else None,
            focus_left=focus.get("left"),
            focus_right=focus.get("right"),
            focus_score=focus.get("score"),
            focus_consistency_warning=bool(focus.get("consistency_warning")),
            exposure_left=self._meta_exposure(left_exposure),
            exposure_right=self._meta_exposure(right_exposure),
            dic_speckle=dic_speckle,
            calibration_board=calibration_board,
            capture_quality_report=quality_report or self._quality_report_from_metrics(quality_metrics),
            data_manifest={
                "manifest_csv": str(meta_dir / "exports" / "file_manifest.csv"),
                "summary_json": str(meta_dir / "exports" / "capture_summary.json"),
            },
        )
        manifest = self._write_manifest_for_session(
            meta_dir,
            {
                "mode": mode,
                "capture_id": capture_id,
                "dic_speckle": dic_speckle,
                "quality_report": quality_report or self._quality_report_from_metrics(quality_metrics),
            },
        )
        self.project_manager.register_session(
            mode,
            meta_dir,
            meta_dir / "meta.json",
            {"capture_id": capture_id, "image_root": str(project_dir), "manifest": manifest},
        )
        return meta_dir

    def _save_frame(self, frame: CameraFrame, path: Path, config_snapshot: dict | None = None) -> Path:
        config_snapshot = config_snapshot or self._config_snapshot()
        force_image = config_bool(config_snapshot, "record_force_image_format", False, False)
        if self._should_save_raw_frame(frame, config_snapshot) and not force_image:
            return self._save_raw_frame(frame, path, config_snapshot)
        if getattr(frame, "image", None) is None:
            return self._save_raw_frame(frame, path, config_snapshot)
        return self._save_image(frame.image, path, config_snapshot)

    def _should_save_raw_frame(self, frame: CameraFrame, config_snapshot: dict) -> bool:
        if config_bool(config_snapshot, "save_raw_frames", False, False):
            return getattr(frame, "raw_data", None) is not None
        pixel_name = str(getattr(frame, "pixel_type_name", "") or "").lower()
        bit_depth = int(getattr(frame, "raw_bit_depth", 8) or 8)
        return bool(getattr(frame, "raw_data", None)) and (bit_depth > 8 or "bayer" in pixel_name)

    def _raw_frame_array(self, frame: CameraFrame) -> np.ndarray:
        raw_data = getattr(frame, "raw_data", None)
        if raw_data is None:
            raise MvsError("raw-only frame has no raw payload to save")
        width = int(getattr(frame, "width", 0) or 0)
        height = int(getattr(frame, "height", 0) or 0)
        bit_depth = int(getattr(frame, "raw_bit_depth", 8) or 8)
        pixel_name = str(getattr(frame, "pixel_type_name", "") or "").lower()
        raw_len = int(getattr(frame, "raw_frame_len", 0) or len(raw_data))
        if width > 0 and height > 0 and raw_len >= width * height * 2 and bit_depth > 8:
            payload = contiguous_frame_buffer(raw_data, width * height * 2)
            return np.frombuffer(payload, dtype="<u2", count=width * height).reshape((height, width)).copy()
        if width > 0 and height > 0 and raw_len >= width * height and ("mono" in pixel_name or "bayer" in pixel_name):
            payload = contiguous_frame_buffer(raw_data, width * height)
            return np.frombuffer(payload, dtype=np.uint8, count=width * height).reshape((height, width)).copy()
        return np.frombuffer(contiguous_frame_buffer(raw_data), dtype=np.uint8).copy()

    def _frame_to_correction_array(self, frame: CameraFrame | None) -> np.ndarray | None:
        if frame is None:
            return None
        raw_data = getattr(frame, "raw_data", None)
        if raw_data is not None:
            array = self._raw_frame_array(frame)
            if array.ndim == 2:
                return array.astype(np.float32, copy=False)
        image = getattr(frame, "image", None)
        if image is None:
            return None
        return np.asarray(image.convert("L"), dtype=np.float32)

    def _correct_frame(self, frame: CameraFrame | None, side: str) -> CameraFrame | None:
        if frame is None:
            return None
        field_correction = self.config.get("field_correction", {})
        if not config_bool(field_correction if isinstance(field_correction, dict) else {}, "enabled", False, False):
            return frame
        array = self._frame_to_correction_array(frame)
        if array is None:
            return frame
        with self._field_correction_lock:
            dark = self._dark_frame_refs.get(side)
            flat = self._flat_field_refs.get(side)
        corrected = array.astype(np.float32, copy=True)
        if dark is not None and dark.shape == corrected.shape:
            corrected -= dark
        if flat is not None and flat.shape == corrected.shape:
            flat_work = flat.astype(np.float32, copy=False)
            if dark is not None and dark.shape == flat_work.shape:
                flat_work = flat_work - dark
            gain = float(np.mean(flat_work[flat_work > 1.0])) if np.any(flat_work > 1.0) else 1.0
            corrected = corrected / np.maximum(flat_work, 1.0) * gain
        bit_depth = int(getattr(frame, "raw_bit_depth", 8) or 8)
        max_value = float((1 << min(max(bit_depth, 8), 16)) - 1)
        if bit_depth > 8:
            corrected_u16 = np.clip(corrected, 0.0, max_value).astype(np.uint16)
            image = Image.fromarray((corrected_u16 / max(max_value, 1.0) * 255.0).astype(np.uint8), "L")
            raw_data = corrected_u16.tobytes()
            raw_len = len(raw_data)
        else:
            corrected_u8 = np.clip(corrected, 0.0, 255.0).astype(np.uint8)
            image = Image.fromarray(corrected_u8, "L")
            raw_data = corrected_u8.tobytes()
            raw_len = len(raw_data)
        return CameraFrame(
            image=image,
            frame_number=frame.frame_number,
            width=frame.width,
            height=frame.height,
            host_timestamp=frame.host_timestamp,
            camera_timestamp=frame.camera_timestamp,
            raw_data=raw_data,
            raw_frame_len=raw_len,
            pixel_type=frame.pixel_type,
            pixel_type_name=frame.pixel_type_name,
            raw_bit_depth=frame.raw_bit_depth,
            raw_array_shape=getattr(frame, "raw_array_shape", None),
        )

    def _correct_frame_pair(
        self,
        left: CameraFrame | None,
        right: CameraFrame | None,
    ) -> tuple[CameraFrame | None, CameraFrame | None]:
        return self._correct_frame(left, "left"), self._correct_frame(right, "right")

    def _validate_standard_raw_image_array(self, frame: CameraFrame, array: np.ndarray, fmt: str) -> None:
        width = int(getattr(frame, "width", 0) or 0)
        height = int(getattr(frame, "height", 0) or 0)
        bit_depth = int(getattr(frame, "raw_bit_depth", 8) or 8)
        if bit_depth <= 8:
            return
        if width <= 0 or height <= 0 or array.shape != (height, width) or array.dtype != np.uint16:
            raise MvsError(
                f"{fmt} requires unpacked high-bit-depth mono data. "
                "Choose raw_frame_format='npy' to keep the original bytes, or set the camera PixelFormat to unpacked Mono16."
            )

    def _save_raw_frame(self, frame: CameraFrame, path: Path, config_snapshot: dict | None = None) -> Path:
        config_snapshot = config_snapshot or self._config_snapshot()
        if getattr(frame, "raw_data", None) is None:
            if getattr(frame, "image", None) is None:
                raise MvsError("raw-only frame has no raw payload to save")
            return self._save_image(frame.image, path, config_snapshot)
        fmt = raw_frame_format(config_snapshot)
        raw_path = path.with_suffix(f".{raw_frame_extension(config_snapshot)}")
        try:
            array = self._raw_frame_array(frame)
            if fmt == "npy":
                np.save(raw_path, array)
            elif fmt == "png16":
                self._validate_standard_raw_image_array(frame, array, fmt)
                if array.dtype != np.uint16:
                    array = array.astype(np.uint16, copy=False)
                Image.fromarray(array).save(raw_path, format="PNG")
            elif fmt == "tiff16":
                self._validate_standard_raw_image_array(frame, array, fmt)
                if array.dtype != np.uint16:
                    array = array.astype(np.uint16, copy=False)
                Image.fromarray(array).save(raw_path, format="TIFF")
            elif fmt == "exr":
                self._validate_standard_raw_image_array(frame, array, fmt)
                exr_array = array.astype(np.float32, copy=False)
                ok = cv2.imwrite(str(raw_path), exr_array)
                if not ok:
                    raise MvsError("OpenCV EXR writer returned failure; enable OpenEXR support or choose png16/tiff16")
            else:
                raise MvsError(f"unsupported raw_frame_format: {fmt}")
            self._save_viewable_sidecar(array, raw_path, config_snapshot)
        except Exception as exc:
            LOGGER.exception("原始帧保存失败: %s", raw_path)
            try:
                raw_path.unlink(missing_ok=True)
            except OSError:
                LOGGER.warning("原始帧保存失败后清理残留文件失败: %s", raw_path, exc_info=True)
            raise MvsError(f"原始帧保存失败：{raw_path}；{exc}") from exc
        return raw_path

    def _save_viewable_sidecar(self, array: np.ndarray, raw_path: Path, config_snapshot: dict) -> None:
        if not config_bool(config_snapshot, "viewable_sidecar_enabled", True, True):
            return
        if array.size == 0 or array.ndim < 2:
            return
        fmt = str(config_snapshot.get("viewable_sidecar_format", config_snapshot.get("image_format", "png"))).lower().strip()
        if fmt in {"jpg", "jpeg"}:
            ext = "jpg"
            quality = max(min(int(config_snapshot.get("record_jpeg_quality", 95) or 95), 100), 1)
        else:
            ext = "png"
            quality = None
        view_path = raw_path.with_suffix(f".view.{ext}")
        if array.dtype == np.uint8:
            work = array
        elif array.dtype in (np.uint16, np.int32, np.int64):
            vmin = int(array.min())
            vmax = int(array.max())
            value_range = vmax - vmin
            if value_range <= 0:
                view8 = np.full(array.shape, 128, dtype=np.uint8)
            else:
                view8 = np.clip((array.astype(np.float64) - vmin) * 255.0 / value_range, 0, 255).astype(np.uint8)
            work = view8
        else:
            work = np.clip(array.astype(np.float64) * 255.0, 0, 255).astype(np.uint8)
        try:
            image = Image.fromarray(work)
            if ext == "jpg":
                image.save(view_path, format="JPEG", quality=quality)
            else:
                image.save(view_path, format="PNG")
        except Exception as exc:
            LOGGER.debug("可查看侧车图保存失败: %s (%s)", view_path, exc, exc_info=True)

    def _save_image(self, image: Image.Image, path: Path, config_snapshot: dict | None = None) -> Path:
        config_snapshot = config_snapshot or self._config_snapshot()
        ext = path.suffix.lower().lstrip(".")
        if ext == "jpeg":
            ext = "jpg"
            path = path.with_suffix(".jpg")
        if ext not in {"bmp", "jpg", "png"}:
            ext = image_extension(config_snapshot)
            path = path.with_suffix(f".{ext}")
        if image.mode not in ("L", "RGB"):
            image = image.convert("RGB")
        try:
            if ext == "jpg":
                quality = int(config_snapshot.get("record_jpeg_quality", 95))
                image.save(path, format="JPEG", quality=quality)
            elif ext == "png":
                image.save(path, format="PNG")
            else:
                image.save(path, format="BMP")
        except OSError as exc:
            LOGGER.exception("图像保存失败: %s", path)
            try:
                path.unlink(missing_ok=True)
            except OSError:
                LOGGER.warning("图像保存失败后清理残留文件失败: %s", path, exc_info=True)
            raise MvsError(f"图像保存失败：{path}；{exc}") from exc
        return path

    def _meta_exposure(self, exposure: dict[str, object] | None) -> dict[str, object] | None:
        if exposure is None:
            return None
        return {
            "mean": exposure.get("mean"),
            "over_pct": exposure.get("over_pct"),
            "under_pct": exposure.get("under_pct"),
            "snr_db": exposure.get("snr_db"),
            "advice": exposure.get("advice"),
        }

    def _write_manifest_for_session(
        self,
        session_dir: Path,
        capture_summary: dict[str, object],
        config_snapshot: dict | None = None,
    ) -> dict[str, object]:
        config_snapshot = config_snapshot or self._config_snapshot()
        camera_settings = self._capture_settings_snapshot(config_snapshot)
        environment = {
            "temperatures_c": dict(self._latest_temperatures),
            "stream_stats": dict(getattr(self, "_latest_stream_stats", {})),
            "temperature_samples": list(self._temperature_samples[-1000:]),
            "save_dir": str(resolve_output_root(config_snapshot)),
            "project": self.project_manager.project_meta(),
        }
        return write_data_manifest(
            session_dir,
            capture_summary=capture_summary,
            camera_settings=camera_settings,
            environment=environment,
            algorithm=self._checksum_algorithm(config_snapshot),
        )

    def _capture_settings_snapshot(self, config_snapshot: dict | None = None) -> dict[str, object]:
        config_snapshot = config_snapshot or self._config_snapshot()
        keys = (
            "trigger_source",
            "exposure_auto",
            "exposure_time_us",
            "auto_exposure_lower_limit",
            "auto_exposure_upper_limit",
            "gain_auto",
            "gain",
            "auto_gain_lower_limit",
            "auto_gain_upper_limit",
            "balance_white_auto",
            "balance_ratio_red",
            "balance_ratio_green",
            "balance_ratio_blue",
            "roi_width",
            "roi_height",
            "roi_offset_x",
            "roi_offset_y",
            "left_roi_width",
            "left_roi_height",
            "left_roi_offset_x",
            "left_roi_offset_y",
            "right_roi_width",
            "right_roi_height",
            "right_roi_offset_x",
            "right_roi_offset_y",
            "pixel_format",
            "image_format",
            "record_save_image_sequence",
            "record_fps",
            "record_split_interval_seconds",
            "record_split_size_gb",
            "record_max_seconds",
            "chunk_data_enabled",
            "chunk_selectors",
            "timestamp_reject_enabled",
            "max_camera_timestamp_delta",
            "max_host_timestamp_delta",
            "camera_timestamp_offset_fixed",
            "require_hardware_trigger",
            "hardware_sync_enabled",
            "hardware_sync_master",
            "hardware_sync_master_line",
            "hardware_sync_master_line_source",
            "hardware_sync_slave_line",
            "hardware_sync_slave_activation",
            "hardware_sync_master_trigger_source",
            "acquisition_frame_rate",
            "trigger_delay_us",
            "line_debouncer_time_us",
            "trigger_activation",
            "black_level",
            "digital_shift",
            "gamma",
            "save_raw_frames",
            "raw_frame_format",
            "field_correction",
            "dic_analysis",
        )
        snapshot = {key: config_snapshot.get(key) for key in keys}
        snapshot["image_format"] = image_extension(config_snapshot)
        snapshot["device_versions"] = dict(self._device_versions)
        snapshot["calibration"] = self.calibration.meta()
        return snapshot

    def _write_meta(self, path: Path, **data) -> None:
        left_frame = data.get("left")
        right_frame = data.get("right")
        payload = {key: value for key, value in data.items() if key not in {"left", "right"}}
        payload["left"] = self._frame_meta(left_frame) if left_frame is not None else None
        payload["right"] = self._frame_meta(right_frame) if right_frame is not None else None
        payload["image_format"] = image_extension(self.config)
        payload["pixel_format"] = self.config.get("pixel_format", "Mono8")
        payload["left_camera"] = asdict(self.camera_system.left_info) if self.camera_system and self.camera_system.left_info else None
        payload["right_camera"] = (
            asdict(self.camera_system.right_info) if self.camera_system and self.camera_system.right_info else None
        )
        payload["device_versions"] = dict(self._device_versions)
        payload["temperatures_c"] = dict(self._latest_temperatures)
        payload["stream_stats"] = dict(getattr(self, "_latest_stream_stats", {}))
        payload["temperature_samples"] = list(self._temperature_samples[-1000:])
        payload["calibration"] = self.calibration.meta()
        payload["camera_timestamp_offset_fixed"] = self.config.get("camera_timestamp_offset_fixed")
        payload["field_correction"] = dict(self.config.get("field_correction", {})) if isinstance(
            self.config.get("field_correction"), dict
        ) else {}
        payload["dic_analysis"] = dict(self.config.get("dic_analysis", {})) if isinstance(
            self.config.get("dic_analysis"), dict
        ) else {}
        payload["project"] = self.project_manager.project_meta()
        tmp_path = path.with_name(f"{path.name}.tmp")
        with tmp_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        tmp_path.replace(path)

    def _frame_meta(self, frame: CameraFrame) -> dict:
        return {
            "frame_number": frame.frame_number,
            "width": frame.width,
            "height": frame.height,
            "host_timestamp": frame.host_timestamp,
            "camera_timestamp": frame.camera_timestamp,
            "pixel_type": int(getattr(frame, "pixel_type", 0) or 0),
            "pixel_type_name": str(getattr(frame, "pixel_type_name", "") or ""),
            "raw_frame_len": int(getattr(frame, "raw_frame_len", 0) or 0),
            "raw_bit_depth": int(getattr(frame, "raw_bit_depth", 8) or 8),
            "raw_array_shape": list(getattr(frame, "raw_array_shape", None) or []),
        }

    def _show_error(self, exc: object) -> None:
        details = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)) if isinstance(exc, BaseException) else str(exc)
        self.status_var.set(str(exc))
        messagebox.showerror("错误", details)

    def close(self) -> None:
        self._closing = True
        self.previewing = False
        self.recording = False
        self.interval_capturing = False
        self._dic_recording = False
        self.interval_stop_event.set()
        if self._focus_drift_timer is not None:
            self._focus_drift_timer.cancel()
            self._focus_drift_timer = None
        if self._interval_lamp_after_id is not None:
            try:
                self.root.after_cancel(self._interval_lamp_after_id)
            except Exception:
                pass
            self._interval_lamp_after_id = None
        if self._ui_queue_fallback_after_id is not None:
            try:
                self.root.after_cancel(self._ui_queue_fallback_after_id)
            except Exception:
                pass
            self._ui_queue_fallback_after_id = None
        if hasattr(self, "left_pane") and hasattr(self, "right_pane"):
            self.left_pane.unbind_external_callbacks()
            self.right_pane.unbind_external_callbacks()
            self._set_recording_indicator(False)
        join_timeout = max(config_float(self.config, "close_thread_join_timeout_seconds", 10.0), 0.1)
        join_budget = max(config_float(self.config, "close_total_thread_join_timeout_seconds", join_timeout), 0.1)
        join_deadline = time.perf_counter() + join_budget
        self._join_thread_on_close(self.preview_thread, join_timeout, join_deadline)
        self._join_thread_on_close(self.interval_thread, join_timeout, join_deadline)
        self._join_thread_on_close(self.record_thread, join_timeout, join_deadline)
        for thread in self._background_threads_snapshot():
            self._join_thread_on_close(thread, join_timeout, join_deadline)
        if self.camera_system is not None:
            try:
                self.camera_system.close()
            except MvsError as exc:
                self.status_var.set(str(exc))
        self.root.destroy()

    def _join_thread_on_close(
        self,
        thread: threading.Thread | None,
        timeout: float,
        deadline: float | None = None,
    ) -> None:
        if thread is None or not thread.is_alive():
            return
        wait_timeout = timeout
        if deadline is not None:
            wait_timeout = min(wait_timeout, max(deadline - time.perf_counter(), 0.0))
        if wait_timeout > 0:
            thread.join(timeout=wait_timeout)
        if thread.is_alive():
            LOGGER.warning("Thread %s did not stop within %.1f seconds during close.", thread.name or thread.ident, wait_timeout)


def _log_startup_exception() -> None:
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        LOGGER.disabled = False
        if not LOGGER.handlers:
            handler = RotatingFileHandler(LOG_DIR / "capture.log", maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            LOGGER.addHandler(handler)
        LOGGER.exception("application startup failed")
    except Exception:
        pass


def _show_startup_error(root: Tk | None, details: str) -> None:
    fallback_root: Tk | None = None
    try:
        parent = root
        if parent is None:
            fallback_root = Tk()
            fallback_root.withdraw()
            parent = fallback_root
        messagebox.showerror("启动失败", details, parent=parent)
    except Exception:
        print(details, file=sys.stderr)
    finally:
        if fallback_root is not None:
            try:
                fallback_root.destroy()
            except Exception:
                pass


def main() -> None:
    root: Tk | None = None
    try:
        enable_windows_dpi_awareness()
        root = Tk()
        configure_tk_dpi_scaling(root)
        StereoCaptureOnlyApp(root)
        root.mainloop()
    except Exception as exc:
        details = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        _log_startup_exception()
        _show_startup_error(root, details)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

from __future__ import annotations

import hashlib
import json
import logging
import os
from logging.handlers import RotatingFileHandler
import shutil
import subprocess
import sys
import threading
import time
import ctypes
import traceback
from dataclasses import asdict, dataclass
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
        host_timestamp: float
        camera_timestamp: int

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


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
LOG_DIR = BASE_DIR / "logs"
CAPTURE_WIDTH = 5472
CAPTURE_HEIGHT = 3648

BG_COLOR = "#262626"
PANEL_COLOR = "#3a3a3a"
CANVAS_COLOR = "#101010"
BORDER_COLOR = "#4b4b4b"
ACCENT_COLOR = "#2f80ed"
TEXT_COLOR = "#f0f0f0"
MUTED_TEXT_COLOR = "#b8b8b8"
FONT_FAMILY = "Microsoft YaHei UI"
BASE_FONT_SIZE = 10
TITLE_FONT_SIZE = 14
INFO_FONT_SIZE = 10
OVERLAY_FONT_SIZE = 11

FramePair = tuple[CameraFrame | None, CameraFrame | None]
LOGGER = logging.getLogger("mvss_capture")
UI_QUEUE_EVENT = "<<MvssUiQueue>>"
DEFAULT_HOST_TIMESTAMP_DELTA_NS = 0
_CONFIG_MISSING = object()


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
            return super().setdefault(key, self._wrap(default))

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
        with self._lock:
            if self._closed:
                raise RuntimeError("record frame metadata writer is already closed")
            if self._count:
                self._fh.write(",\n")
            json.dump(frame_meta, self._fh, ensure_ascii=False)
            self._frames.append(dict(frame_meta))
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


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return ThreadSafeConfig(json.load(fh))


def save_config(config: dict) -> None:
    payload = config.snapshot() if isinstance(config, ThreadSafeConfig) else dict(config)
    with CONFIG_PATH.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def timestamp_ms() -> str:
    return time.strftime("%Y%m%d_%H%M%S_") + f"{int((time.time() % 1) * 1000):03d}"


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


def resolve_output_root(config: dict) -> Path:
    configured = Path(str(config.get("save_dir", "captures")))
    return configured if configured.is_absolute() else BASE_DIR / configured


def estimate_frame_bytes(config: dict, width: int = CAPTURE_WIDTH, height: int = CAPTURE_HEIGHT) -> int:
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


def setup_logging(config: dict) -> None:
    if LOGGER.handlers:
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "capture.log"
    LOGGER.setLevel(logging.INFO)
    handler = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    LOGGER.addHandler(handler)
    if not config_bool(config, "logging_enabled", True):
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


class ZoomImagePane(Frame):
    def __init__(self, master: Tk | Frame, title: str, reset_command=None, roi_callback=None):
        super().__init__(master, bg=BORDER_COLOR)
        self.title_var = StringVar(value=title)
        self.info_var = StringVar(value="未连接")
        self.zoom_var = StringVar(value="100%")
        self._image_ref: ImageTk.PhotoImage | None = None
        self._last_image: Image.Image | None = None
        self._render_bounds: tuple[float, float, float, float] | None = None
        self._roi_start: tuple[int, int] | None = None
        self._roi_rect_id: int | None = None
        self._flash_id: int | None = None
        self._recording_active = False
        self._recording_after_id: str | None = None
        self._recording_dot_id: int | None = None
        self._recording_text_id: int | None = None
        self._performance_bg_id: int | None = None
        self._performance_text_id: int | None = None
        self._performance_status = "good"
        self._performance_text = "FPS -- | Drop -- | Delta --"
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
        container.pack(fill=BOTH, expand=True, padx=2, pady=2)

        header = ttk.Frame(container, style="PaneHeader.TFrame")
        header.pack(side=TOP, fill=X)
        ttk.Label(header, textvariable=self.title_var, style="PaneTitle.TLabel", padding=(12, 6), anchor="w").pack(
            side=LEFT, fill=X, expand=True
        )
        ttk.Button(header, text="还原", command=reset_command or self.reset_zoom, width=6).pack(
            side=RIGHT, padx=(4, 8), pady=4
        )
        ttk.Label(header, textvariable=self.zoom_var, style="PaneTitle.TLabel", padding=(8, 6), anchor="e").pack(
            side=RIGHT
        )

        self.canvas = Canvas(container, bg=CANVAS_COLOR, highlightthickness=0, bd=0)
        self.canvas.pack(side=TOP, fill=BOTH, expand=True)
        self._canvas_image_id: int | None = None
        self._canvas_text_id = self.canvas.create_text(
            0,
            0,
            text="无图像",
            fill="#777777",
            font=(FONT_FAMILY, 18),
            anchor="center",
        )

        ttk.Label(container, textvariable=self.info_var, style="PaneInfo.TLabel", padding=(10, 4), anchor="w").pack(
            side=BOTTOM, fill=X
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
        self._last_image = frame.image
        self.info_var.set(
            f"{frame.width}x{frame.height}  Frame:{frame.frame_number}  CamTS:{frame.camera_timestamp}"
        )
        self._render()

    def set_display_image(self, image: Image.Image | None, info: str = "") -> None:
        self._last_image = image
        self.info_var.set(info or (f"{image.width}x{image.height}" if image is not None else "No Signal"))
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

    def set_no_signal(self, reason: str = "No Signal") -> None:
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
            self.canvas.itemconfigure(self._canvas_text_id, text="No Signal", state="normal")
            self._place_performance_overlay()
            return

        target_width = max(1, int(width * self.zoom))
        target_height = max(1, int(height * self.zoom))
        source_width, source_height = self._last_image.size
        scale = min(target_width / max(source_width, 1), target_height / max(source_height, 1), 1.0)
        display_size = (max(1, int(source_width * scale)), max(1, int(source_height * scale)))
        image = self._last_image.resize(display_size, Image.Resampling.LANCZOS)
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
        color = "#48e07b"
        self.canvas.create_line(left, center_y, left + width, center_y, fill=color, width=1, tags=("guide",), stipple="gray50")
        self.canvas.create_line(center_x, top, center_x, top + height, fill=color, width=1, tags=("guide",), stipple="gray50")
        if self._guide_mode == "full":
            for frac in (1 / 3, 2 / 3):
                y = top + height * frac
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
            self._focus_roi_rect_id = self._create_fraction_rect(self._focus_roi_frac, "#ffd166", (5, 4), "focus_roi")
        if self._magnifier_rect_frac is not None:
            self._magnifier_rect_id = self._create_fraction_rect(self._magnifier_rect_frac, "#00d4ff", (3, 3), "magnifier")

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
            fill="#ffffff",
            outline="",
            stipple="gray50",
            tags=("flash",),
        )
        self.canvas.lift(self._flash_id)
        self.canvas.after(100, self._clear_flash)

    def _clear_flash(self) -> None:
        if self._flash_id is not None:
            self.canvas.delete(self._flash_id)
            self._flash_id = None

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
        if self._performance_text_id is not None:
            color = {"good": "#7bd88f", "warn": "#ffd166", "bad": "#ff6b6b"}.get(status, "#7bd88f")
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
                fill="#e53935",
                outline="#ffb3b3",
                width=2,
                tags=("recording",),
            )
            self._recording_text_id = self.canvas.create_text(
                38,
                22,
                text="Recording",
                fill="#ffffff",
                font=("Arial", 12, "bold"),
                anchor="w",
                tags=("recording",),
            )
        self.canvas.itemconfigure(self._recording_dot_id, state=state)
        if self._recording_text_id is not None:
            self.canvas.itemconfigure(self._recording_text_id, state=state)
        self._raise_overlays()

    def _place_performance_overlay(self) -> None:
        canvas_width = max(self.canvas.winfo_width(), 1)
        canvas_height = max(self.canvas.winfo_height(), 1)
        x = 10
        y = max(canvas_height - 48, 10)
        text_width = max(180, min(canvas_width - 20, 620))
        text_height = 38 if "\n" in self._performance_text else 20
        if self._performance_bg_id is None:
            self._performance_bg_id = self.canvas.create_rectangle(
                x,
                y,
                x + text_width,
                y + text_height,
                fill="#000000",
                outline="#1f1f1f",
                stipple="gray25",
                tags=("performance",),
            )
            self._performance_text_id = self.canvas.create_text(
                x + 7,
                y + 5,
                text=self._performance_text,
                fill="#7bd88f",
                font=("Consolas", max(8, OVERLAY_FONT_SIZE - 2), "bold"),
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
            outline="#00d4ff",
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
            self.roi_callback(roi)

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
        super().__init__(master, style="Panel.TFrame", padding=(6, 4))
        self.title = title
        self.histogram: list[float] | None = None
        ttk.Label(self, text=title, style="Panel.TLabel").pack(side=TOP, anchor="w")
        self.canvas = Canvas(self, width=280, height=130, bg="#151515", highlightthickness=1, highlightbackground="#555555")
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
        height = max(self.canvas.winfo_height(), 80)
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

        self.canvas.create_rectangle(x0, y0, x1, y1, fill="#1f1f1f", outline="#555555")
        self.canvas.create_rectangle(x0, y0, x0 + plot_w * 15 / 255, y1, fill="#20334d", outline="")
        self.canvas.create_rectangle(x0 + plot_w * 240 / 255, y0, x1, y1, fill="#4a2525", outline="")
        for value, color, dash in ((5, "#4c8dff", (2, 3)), (128, "#a8a8a8", (3, 3)), (250, "#ff6b6b", (2, 3))):
            x = x0 + plot_w * value / 255
            self.canvas.create_line(x, y0, x, y1, fill=color, dash=dash)
        self.canvas.create_text(x0, y1 + 3, text="0", fill="#a8a8a8", anchor="nw", font=("Consolas", 8))
        self.canvas.create_text(x0 + plot_w / 2, y1 + 3, text="128", fill="#a8a8a8", anchor="n", font=("Consolas", 8))
        self.canvas.create_text(x1, y1 + 3, text="255", fill="#a8a8a8", anchor="ne", font=("Consolas", 8))
        if not self.histogram:
            self.canvas.create_text((x0 + x1) / 2, (y0 + y1) / 2, text="No Data", fill="#777777", anchor="center")
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
            color = "#c7d7ff" if 15 <= index <= 240 else "#ffb0a6" if index > 240 else "#94bdff"
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
            return str(getattr(self, "_trigger_source_cache", self.config.get("trigger_source", "Software")))

    def _set_cached_trigger_source(self, value: str) -> None:
        with self._state_lock:
            self._trigger_source_cache = str(value)

    def _config_snapshot(self) -> dict:
        if isinstance(self.config, ThreadSafeConfig):
            return self.config.snapshot()
        return dict(self.config)

    def _update_config(self, values: dict[str, object], *, save: bool = True) -> dict:
        if isinstance(self.config, ThreadSafeConfig):
            with self.config._lock:
                self.config.update(values)
                snapshot = self.config.snapshot()
        else:
            self.config.update(values)
            snapshot = dict(self.config)
        if save:
            with CONFIG_PATH.open("w", encoding="utf-8") as fh:
                json.dump(snapshot, fh, ensure_ascii=False, indent=2)
                fh.write("\n")
        return snapshot

    def __init__(self, root: Tk):
        self.root = root
        self.root.title("双目同步采集")
        self.root.configure(bg=BG_COLOR)
        self.root.geometry("1660x980")
        self.root.minsize(1280, 820)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<F11>", self._on_fullscreen_key)
        self.root.bind("<Escape>", self._on_escape_key)

        self.config = load_config()
        setup_logging(self.config)
        self._ensure_default_full_resolution()
        self._ensure_recording_config_defaults()
        self._ensure_quality_config_defaults()
        self._ensure_reliability_config_defaults()
        self._ensure_project_config_defaults()
        self.project_manager = ProjectManager(resolve_output_root(self.config), self.config)
        if self.project_manager.enabled and not self.project_manager.current_project_id:
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
        self._trigger_source_cache = str(self.config.get("trigger_source", "Software"))
        self.previewing = False
        self.preview_thread: threading.Thread | None = None
        self._preview_generation = 0
        self.interval_capturing = False
        self.interval_thread: threading.Thread | None = None
        self.interval_stop_event = threading.Event()
        self.interval_count = 0
        self.recording = False
        self.record_thread: threading.Thread | None = None
        self.record_dir: Path | None = None
        self._closing = False
        self.record_count = 0
        self.record_saved_count = 0
        self._record_next_saved_index = 0
        self.record_started_at: float | None = None
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
        self._record_skipped_frames: list[dict[str, object]] = []
        self._record_skipped_count = 0
        self._record_skip_reasons: dict[str, int] = {}
        self._record_timeout_count = 0
        self._record_consecutive_timeouts = 0
        self._record_error_count = 0
        self._record_reconnect_count = 0
        self._record_disk_warning_count = 0
        self._record_disk_benchmark: dict[str, object] | None = None
        self._record_last_disk_check = 0.0
        self._record_disk_usage_start = 0
        self._record_summary: dict[str, object] = {}
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
        self._stereo_blink_phase = 0
        self._last_rectified_overlay_key: tuple[int | None, int | None] | None = None
        self._last_rectified_overlay_image: Image.Image | None = None
        self._device_versions: dict[str, str | None] = {}
        self._latest_temperatures: dict[str, float | None] = {}
        self._temperature_samples: list[dict[str, object]] = []
        self._last_temperature_poll = 0.0
        self._focus_history: list[tuple[float, float]] = []
        self._focus_peak_score = 0.0

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
        self.trigger_source_var = StringVar(value=str(self.config.get("trigger_source", "Software")))
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
        self.roi_width_var = StringVar(value=str(self.config.get("roi_width", CAPTURE_WIDTH)))
        self.roi_height_var = StringVar(value=str(self.config.get("roi_height", CAPTURE_HEIGHT)))
        self.roi_offset_x_var = StringVar(value=str(self.config.get("roi_offset_x", 0)))
        self.roi_offset_y_var = StringVar(value=str(self.config.get("roi_offset_y", 0)))
        self.interval_seconds_var = StringVar(value=optional_config_text(self.config, "interval_capture_seconds", "5.0"))
        self.interval_limit_var = StringVar(value=optional_config_text(self.config, "interval_capture_count", ""))
        self.record_fps_var = StringVar(value=str(self.config.get("record_fps", 5.0)))
        self.record_max_seconds_var = StringVar(value=optional_config_text(self.config, "record_max_seconds", "0"))
        exposure_monitor = self._ensure_config_section("exposure_monitor")
        self.focus_peaking_var = BooleanVar(value=False)
        self.zebra_var = BooleanVar(value=bool(exposure_monitor.get("zebra_enabled", False)))
        self.histogram_enabled_var = BooleanVar(value=bool(exposure_monitor.get("histogram_enabled", True)))
        self.focus_panel_open_var = BooleanVar(value=True)
        self.exposure_panel_open_var = BooleanVar(value=True)
        self.validation_panel_open_var = BooleanVar(value=True)
        self.magnifier_enabled_var = BooleanVar(value=False)
        self.project_id_var = StringVar(value=self.project_manager.current_project_id or "--")
        self.calibration_summary_var = StringVar(value=self.calibration.status_text())
        self.temperature_status_var = StringVar(value="Temp --")
        self.focus_peak_var = StringVar(value="Peak -- | 0%")
        hdr = self._ensure_config_section("hdr_bracketing")
        hdr_ev_offsets = hdr.get("ev_offsets", [-2, -1, 0, 1, 2])
        if not isinstance(hdr_ev_offsets, (list, tuple)):
            hdr_ev_offsets = [-2, -1, 0, 1, 2]
        self.hdr_enabled_var = BooleanVar(value=config_bool(hdr, "enabled", True))
        self.hdr_sequence_var = StringVar(
            value=", ".join(str(v) for v in hdr_ev_offsets)
        )
        self.focus_score_var = StringVar(value="Focus --")
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
        self._focus_peaking_enabled_setting = bool(self.focus_peaking_var.get())
        self._histogram_enabled_setting = bool(self.histogram_enabled_var.get())

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self.root, padding=(10, 8))
        toolbar.pack(side=TOP, fill=X)

        actions = ttk.Frame(toolbar)
        actions.pack(side=TOP, fill=X)
        self.connect_button = ttk.Button(actions, text="连接相机", command=self.connect_cameras, style="Accent.TButton")
        self.preview_button = ttk.Button(actions, text="开始采集", command=self.toggle_preview, state=DISABLED)
        self.photo_button = ttk.Button(actions, text="同步拍照", command=self.capture_photo, state=DISABLED)
        self.hdr_button = ttk.Button(actions, text="HDR包围", command=self.capture_hdr_bracket, state=DISABLED)
        self.interval_button = ttk.Button(actions, text="定时拍照", command=self.toggle_interval_capture, state=DISABLED)
        self.record_button = ttk.Button(actions, text="开始录像", command=self.toggle_recording, state=DISABLED)
        self.new_project_button = ttk.Button(actions, text="新建项目", command=self.create_new_project)
        self.refresh_button = ttk.Button(actions, text="刷新设备", command=self.refresh_devices)
        self.choose_save_dir_button = ttk.Button(actions, text="保存路径", command=self.choose_save_dir)
        self.exit_button = ttk.Button(actions, text="退出", command=self.close)

        for button in (
            self.connect_button,
            self.preview_button,
            self.photo_button,
            self.hdr_button,
            self.interval_button,
            self.record_button,
            self.new_project_button,
            self.refresh_button,
            self.choose_save_dir_button,
        ):
            button.pack(side=LEFT, padx=(0, 8))
        self.exit_button.pack(side=RIGHT)
        self.refresh_tooltip = ToolTip(self.refresh_button, self._device_tooltip_text)

        settings = ttk.Frame(toolbar)
        settings.pack(side=TOP, fill=X, pady=(8, 0))

        trigger_panel = ttk.Frame(settings, style="Panel.TFrame", padding=(8, 6))
        trigger_panel.pack(side=LEFT, padx=(0, 8))
        ttk.Label(trigger_panel, text="触发", style="Panel.TLabel").grid(row=0, column=0, padx=(0, 4), pady=2)
        ttk.OptionMenu(
            trigger_panel,
            self.trigger_source_var,
            self.trigger_source_var.get(),
            "Software",
            "Line0",
        ).grid(row=0, column=1, padx=3, pady=2)
        self.apply_trigger_button = ttk.Button(
            trigger_panel, text="应用触发", command=self.apply_trigger_settings, state=DISABLED
        )
        self.apply_trigger_button.grid(row=0, column=2, padx=(6, 0), pady=2)

        preset_panel = ttk.Frame(settings, style="Panel.TFrame", padding=(8, 6))
        preset_panel.pack(side=LEFT, padx=(0, 8))
        ttk.Label(preset_panel, text="预设", style="Panel.TLabel").grid(row=0, column=0, padx=(0, 4), pady=2)
        preset_names = list(self.config.get("presets", {}).keys()) or [self.preset_var.get()]
        ttk.OptionMenu(preset_panel, self.preset_var, self.preset_var.get(), *preset_names).grid(
            row=0, column=1, padx=3, pady=2
        )
        ttk.Button(preset_panel, text="加载", command=self.load_preset).grid(row=0, column=2, padx=3, pady=2)
        ttk.Button(preset_panel, text="保存", command=self.save_preset).grid(row=0, column=3, padx=3, pady=2)

        interval_panel = ttk.Frame(settings, style="Panel.TFrame", padding=(8, 6))
        interval_panel.pack(side=LEFT, fill=X, expand=True)
        ttk.Label(interval_panel, text="定时拍照", style="Panel.TLabel").grid(row=0, column=0, padx=(0, 4), pady=2)
        self.interval_lamp = Canvas(interval_panel, width=18, height=18, bg=PANEL_COLOR, highlightthickness=0, bd=0)
        self.interval_lamp.grid(row=0, column=1, padx=(0, 8), pady=2)
        self.interval_lamp_id = self.interval_lamp.create_oval(3, 3, 15, 15, fill="#666666", outline="#2a2a2a")
        self._labeled_entry(interval_panel, "间隔s", self.interval_seconds_var, 7, 0, 2)
        self._labeled_entry(interval_panel, "张数", self.interval_limit_var, 7, 0, 4)
        self._labeled_entry(interval_panel, "录像fps", self.record_fps_var, 7, 0, 6)

        self._labeled_entry(interval_panel, "时长s", self.record_max_seconds_var, 7, 0, 8)

        info = ttk.Frame(toolbar)
        info.pack(side=TOP, fill=X, pady=(8, 0))
        ttk.Label(info, text="Project").pack(side=LEFT, padx=(0, 4))
        ttk.Label(info, textvariable=self.project_id_var, style="Value.TLabel").pack(side=LEFT, padx=(0, 18))
        ttk.Label(info, textvariable=self.calibration_summary_var, style="Value.TLabel").pack(side=LEFT, padx=(0, 18))
        ttk.Label(info, textvariable=self.temperature_status_var, style="Value.TLabel").pack(side=LEFT, padx=(0, 18))
        ttk.Label(info, text="默认采集尺寸").pack(side=LEFT, padx=(0, 4))
        ttk.Label(info, text=f"{CAPTURE_WIDTH} x {CAPTURE_HEIGHT}", style="Value.TLabel").pack(side=LEFT, padx=(0, 18))
        ttk.Label(info, text="保存路径").pack(side=LEFT, padx=(0, 4))
        ttk.Label(info, textvariable=self.save_dir_var, style="Value.TLabel").pack(side=LEFT, fill=X, expand=True)

        param_panel = ttk.LabelFrame(toolbar, text="参数设置", padding=(10, 8))
        param_panel.pack(side=TOP, fill=X, pady=(8, 0))
        param_panel.grid_columnconfigure(0, weight=1)
        param_panel.grid_columnconfigure(1, weight=1)

        gain_panel = ttk.Frame(param_panel, style="Panel.TFrame", padding=(8, 6))
        gain_panel.grid(row=0, column=0, sticky="ew", padx=(0, 6), pady=(0, 6))
        self._configure_parameter_grid(gain_panel)
        ttk.Label(gain_panel, text="增益", style="Panel.TLabel").grid(row=0, column=0, padx=(0, 6), pady=3, sticky="w")
        ttk.OptionMenu(gain_panel, self.gain_auto_var, self.gain_auto_var.get(), "Off", "Once", "Continuous").grid(
            row=0, column=1, padx=3, pady=3, sticky="w"
        )
        self._labeled_entry(gain_panel, "值", self.gain_var, 6, 0, 2)
        self._labeled_entry(gain_panel, "下限", self.auto_gain_lower_var, 6, 0, 4)
        self._labeled_entry(gain_panel, "上限", self.auto_gain_upper_var, 6, 0, 6)
        self.apply_gain_button = ttk.Button(gain_panel, text="应用增益", command=self.apply_gain_settings, state=DISABLED)
        self.apply_gain_button.grid(row=0, column=10, padx=(8, 0), pady=3, sticky="e")

        exposure_panel = ttk.Frame(param_panel, style="Panel.TFrame", padding=(8, 6))
        exposure_panel.grid(row=0, column=1, sticky="ew", padx=(6, 0), pady=(0, 6))
        self._configure_parameter_grid(exposure_panel)
        ttk.Label(exposure_panel, text="曝光", style="Panel.TLabel").grid(
            row=0, column=0, padx=(0, 6), pady=3, sticky="w"
        )
        ttk.OptionMenu(
            exposure_panel,
            self.exposure_auto_var,
            self.exposure_auto_var.get(),
            "Off",
            "Once",
            "Continuous",
        ).grid(row=0, column=1, padx=3, pady=3, sticky="w")
        self.apply_exposure_button = ttk.Button(
            exposure_panel, text="应用曝光", command=self.apply_exposure_settings, state=DISABLED
        )
        self._labeled_entry(exposure_panel, "us", self.exposure_time_var, 8, 0, 2)
        self._labeled_entry(exposure_panel, "下限", self.auto_exposure_lower_var, 8, 0, 4)
        self._labeled_entry(exposure_panel, "上限", self.auto_exposure_upper_var, 8, 0, 6)
        self.apply_exposure_button.grid(row=0, column=10, padx=(8, 0), pady=3, sticky="e")

        wb_panel = ttk.Frame(param_panel, style="Panel.TFrame", padding=(8, 6))
        wb_panel.grid(row=1, column=0, sticky="ew", padx=(0, 6))
        self._configure_parameter_grid(wb_panel)
        ttk.Label(wb_panel, text="白平衡", style="Panel.TLabel").grid(
            row=0, column=0, padx=(0, 6), pady=3, sticky="w"
        )
        ttk.OptionMenu(
            wb_panel,
            self.balance_auto_var,
            self.balance_auto_var.get(),
            "Off",
            "Once",
            "Continuous",
        ).grid(row=0, column=1, padx=3, pady=3, sticky="w")
        self._labeled_entry(wb_panel, "R", self.balance_red_var, 5, 0, 2)
        self._labeled_entry(wb_panel, "G", self.balance_green_var, 5, 0, 4)
        self._labeled_entry(wb_panel, "B", self.balance_blue_var, 5, 0, 6)
        self.apply_wb_button = ttk.Button(wb_panel, text="应用白平衡", command=self.apply_white_balance_settings, state=DISABLED)
        self.apply_wb_button.grid(row=0, column=10, padx=(8, 0), pady=3, sticky="e")

        roi_panel = ttk.Frame(param_panel, style="Panel.TFrame", padding=(8, 6))
        roi_panel.grid(row=1, column=1, sticky="ew", padx=(6, 0))
        self._configure_parameter_grid(roi_panel)
        roi_panel.grid_columnconfigure(10, minsize=90, weight=0)
        roi_panel.grid_columnconfigure(11, minsize=90, weight=1)
        ttk.Label(roi_panel, text="ROI", style="Panel.TLabel").grid(row=0, column=0, padx=(0, 6), pady=3, sticky="w")
        self._labeled_entry(roi_panel, "W", self.roi_width_var, 6, 0, 1)
        self._labeled_entry(roi_panel, "H", self.roi_height_var, 6, 0, 3)
        self._labeled_entry(roi_panel, "X", self.roi_offset_x_var, 5, 0, 5)
        self._labeled_entry(roi_panel, "Y", self.roi_offset_y_var, 5, 0, 7)
        self.edit_roi_button = ttk.Button(roi_panel, text="修改ROI", command=self.toggle_roi_edit_mode)
        self.edit_roi_button.grid(row=0, column=9, padx=(8, 0), pady=3, sticky="e")
        self.reset_roi_button = ttk.Button(roi_panel, text="还原ROI", command=self.reset_roi_settings)
        self.reset_roi_button.grid(row=0, column=10, padx=(8, 0), pady=3, sticky="e")
        self.apply_roi_button = ttk.Button(roi_panel, text="应用ROI", command=self.apply_roi_settings, state=DISABLED)
        self.apply_roi_button.grid(row=0, column=11, padx=(8, 0), pady=3, sticky="e")

        content = Frame(self.root, bg=BG_COLOR)
        content.pack(side=TOP, fill=BOTH, expand=True)
        content.grid_columnconfigure(0, weight=1, uniform="camera")
        content.grid_columnconfigure(1, weight=1, uniform="camera")
        content.grid_rowconfigure(0, weight=1)

        self.left_pane = ZoomImagePane(content, "左相机", roi_callback=self.set_roi_from_preview)
        self.right_pane = ZoomImagePane(content, "右相机", roi_callback=self.set_roi_from_preview)
        self.left_pane.grid(row=0, column=0, sticky="nsew", padx=(8, 4), pady=8)
        self.right_pane.grid(row=0, column=1, sticky="nsew", padx=(4, 8), pady=8)

        self.quality_panel = ttk.Frame(self.root, padding=(8, 0))
        self.quality_panel.pack(side=TOP, fill=X)
        self._build_quality_panels(self.quality_panel)

        ttk.Separator(self.root, orient="horizontal").pack(side=TOP, fill=X)
        status_bar = ttk.Frame(self.root)
        status_bar.pack(side=BOTTOM, fill=X)
        self.status_label = ttk.Label(
            status_bar,
            textvariable=self.status_var,
            style="Status.TLabel",
            anchor="w",
            padding=(10, 6),
        )
        self.status_label.pack(side=LEFT, fill=X, expand=True)

    def _configure_style(self) -> None:
        self.style = ttk.Style()
        try:
            self.style.theme_use("clam")
        except Exception:
            pass
        self.style.configure(".", background=BG_COLOR, foreground=TEXT_COLOR, font=(FONT_FAMILY, BASE_FONT_SIZE))
        self.style.configure("TFrame", background=BG_COLOR)
        self.style.configure("Panel.TFrame", background=PANEL_COLOR)
        self.style.configure("PaneHeader.TFrame", background=PANEL_COLOR)
        self.style.configure("TLabel", background=BG_COLOR, foreground=TEXT_COLOR, font=(FONT_FAMILY, BASE_FONT_SIZE))
        self.style.configure("Panel.TLabel", background=PANEL_COLOR, foreground=TEXT_COLOR, font=(FONT_FAMILY, BASE_FONT_SIZE))
        self.style.configure("Value.TLabel", background=BG_COLOR, foreground="#ffffff", font=(FONT_FAMILY, BASE_FONT_SIZE, "bold"))
        self.style.configure("PaneTitle.TLabel", background=PANEL_COLOR, foreground="white", font=(FONT_FAMILY, TITLE_FONT_SIZE, "bold"))
        self.style.configure("PaneInfo.TLabel", background=PANEL_COLOR, foreground="#d7d7d7", font=("Consolas", INFO_FONT_SIZE))
        self.style.configure("Status.TLabel", background=BG_COLOR, foreground=MUTED_TEXT_COLOR, font=(FONT_FAMILY, BASE_FONT_SIZE))
        self.style.configure("Tooltip.TLabel", background="#202020", foreground="#f2f2f2", font=(FONT_FAMILY, BASE_FONT_SIZE))
        self.style.configure("TButton", background=PANEL_COLOR, foreground="white", borderwidth=0, padding=(10, 6))
        self.style.map(
            "TButton",
            background=[("active", "#505050"), ("disabled", "#303030")],
            foreground=[("disabled", "#777777")],
        )
        self.style.configure("Accent.TButton", background=ACCENT_COLOR, foreground="white", borderwidth=0, padding=(12, 7))
        self.style.map("Accent.TButton", background=[("active", "#4a90f5"), ("disabled", "#303030")])
        self.style.configure("TEntry", fieldbackground="#3d3d3d", foreground="white", bordercolor="#555555")
        self.style.configure("TMenubutton", background=PANEL_COLOR, foreground="white", borderwidth=0, padding=(8, 5))
        self.style.map("TMenubutton", background=[("active", "#505050"), ("disabled", "#303030")])
        self.style.configure("TLabelframe", background=BG_COLOR, bordercolor="#555555", relief="solid")
        self.style.configure("TLabelframe.Label", background=BG_COLOR, foreground="#dcdcdc", font=(FONT_FAMILY, BASE_FONT_SIZE, "bold"))
        self.style.configure("Horizontal.TSeparator", background="#555555")
        self.style.configure("Green.Horizontal.TProgressbar", troughcolor="#1f1f1f", background="#41c46d")
        self.style.configure("Yellow.Horizontal.TProgressbar", troughcolor="#1f1f1f", background="#ffd166")
        self.style.configure("Red.Horizontal.TProgressbar", troughcolor="#1f1f1f", background="#ff6b6b")
        self.style.configure("Good.TLabel", background=BG_COLOR, foreground="#7bd88f", font=(FONT_FAMILY, BASE_FONT_SIZE, "bold"))
        self.style.configure("Warn.TLabel", background=BG_COLOR, foreground="#ffd166", font=(FONT_FAMILY, BASE_FONT_SIZE, "bold"))
        self.style.configure("Bad.TLabel", background=BG_COLOR, foreground="#ff6b6b", font=(FONT_FAMILY, BASE_FONT_SIZE, "bold"))
        self.style.configure("PanelGood.TLabel", background=PANEL_COLOR, foreground="#7bd88f", font=(FONT_FAMILY, BASE_FONT_SIZE, "bold"))
        self.style.configure("PanelWarn.TLabel", background=PANEL_COLOR, foreground="#ffd166", font=(FONT_FAMILY, BASE_FONT_SIZE, "bold"))
        self.style.configure("PanelBad.TLabel", background=PANEL_COLOR, foreground="#ff6b6b", font=(FONT_FAMILY, BASE_FONT_SIZE, "bold"))

    def _build_quality_panels(self, parent: ttk.Frame) -> None:
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_columnconfigure(1, weight=1)
        parent.grid_columnconfigure(2, weight=1)
        parent.grid_columnconfigure(3, weight=1)

        self.focus_panel = ttk.LabelFrame(parent, text="对焦辅助", padding=(8, 6))
        self.focus_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=(0, 8))
        self._build_focus_panel(self.focus_panel)

        self.exposure_panel = ttk.LabelFrame(parent, text="曝光监控", padding=(8, 6))
        self.exposure_panel.grid(row=0, column=1, sticky="nsew", padx=6, pady=(0, 8))
        self._build_exposure_panel(self.exposure_panel)

        self.validation_panel = ttk.LabelFrame(parent, text="采集校验", padding=(8, 6))
        self.validation_panel.grid(row=0, column=2, sticky="nsew", padx=(6, 0), pady=(0, 8))
        self._build_validation_panel(self.validation_panel)

        self.project_panel = ttk.LabelFrame(parent, text="项目/导出", padding=(8, 6))
        self.project_panel.grid(row=0, column=3, sticky="nsew", padx=(6, 0), pady=(0, 8))
        self._build_project_panel(self.project_panel)

    def _build_focus_panel(self, panel: ttk.LabelFrame) -> None:
        top = ttk.Frame(panel)
        top.pack(side=TOP, fill=X)
        self.focus_collapse_button = ttk.Button(top, text="v", width=3, command=self._toggle_focus_panel)
        self.focus_collapse_button.pack(side=LEFT, padx=(0, 6))
        ttk.Checkbutton(top, text="峰值对焦", variable=self.focus_peaking_var, command=self._sync_quality_toggles).pack(
            side=LEFT, padx=(0, 8)
        )
        ttk.Checkbutton(top, text="放大镜", variable=self.magnifier_enabled_var, command=self._sync_quality_toggles).pack(
            side=LEFT, padx=(0, 8)
        )
        self.focus_roi_button = ttk.Button(top, text="框选对焦ROI", command=self.toggle_focus_roi_edit_mode)
        self.focus_roi_button.pack(side=LEFT, padx=(0, 6))
        ttk.Button(top, text="设为目标", command=self.set_focus_reference).pack(side=LEFT, padx=(0, 6))
        ttk.Button(top, text="保存对焦基准", command=self.save_focus_reference_snapshot).pack(side=LEFT, padx=(0, 6))

        self.focus_panel_body = ttk.Frame(panel)
        self.focus_panel_body.pack(side=TOP, fill=X)
        score_row = ttk.Frame(self.focus_panel_body)
        score_row.pack(side=TOP, fill=X, pady=(6, 0))
        ttk.Label(score_row, textvariable=self.focus_score_var).pack(side=LEFT, padx=(0, 8))
        self.focus_progress = ttk.Progressbar(score_row, mode="determinate", maximum=100, style="Yellow.Horizontal.TProgressbar")
        self.focus_progress.pack(side=LEFT, fill=X, expand=True, padx=(0, 8))
        self.focus_status_label = ttk.Label(score_row, textvariable=self.focus_status_var, style="Warn.TLabel", width=18)
        self.focus_status_label.pack(side=LEFT)

        detail_row = ttk.Frame(self.focus_panel_body)
        detail_row.pack(side=TOP, fill=X, pady=(5, 0))
        self.focus_detail_label = ttk.Label(detail_row, textvariable=self.focus_detail_var, style="Panel.TLabel")
        self.focus_detail_label.pack(side=LEFT, fill=X, expand=True)
        ttk.Label(detail_row, textvariable=self.focus_roi_var, style="Panel.TLabel").pack(side=RIGHT)

        peak_row = ttk.Frame(self.focus_panel_body)
        peak_row.pack(side=TOP, fill=X, pady=(5, 0))
        ttk.Label(peak_row, textvariable=self.focus_peak_var, style="Panel.TLabel").pack(side=LEFT, padx=(0, 8))
        self.focus_chart_canvas = Canvas(
            peak_row, width=260, height=44, bg="#151515", highlightthickness=1, highlightbackground="#555555"
        )
        self.focus_chart_canvas.pack(side=LEFT, fill=X, expand=True)

        self.magnifier_frame = ttk.LabelFrame(self.focus_panel_body, text="对焦放大镜", padding=(6, 4))
        self.magnifier_frame.pack(side=TOP, fill=X, pady=(6, 0))
        mag_top = ttk.Frame(self.magnifier_frame)
        mag_top.pack(side=TOP, fill=X)
        self.magnifier_info_var = StringVar(value="倍率 100% | 点击预览锁定/解锁，滚轮调倍率")
        ttk.Label(mag_top, textvariable=self.magnifier_info_var, style="Panel.TLabel").pack(side=LEFT, fill=X, expand=True)
        self.magnifier_canvas = Canvas(self.magnifier_frame, width=240, height=180, bg="#111111", highlightthickness=1, highlightbackground="#555555")
        self.magnifier_canvas.pack(side=TOP, fill=BOTH, expand=True, pady=(4, 0))
        self._magnifier_image_ref: ImageTk.PhotoImage | None = None
        self.left_pane.bind_external("<Motion>", self._on_magnifier_motion)
        self.left_pane.bind_external("<ButtonPress-1>", self._on_magnifier_click)
        self.left_pane.bind_external("<MouseWheel>", self._on_magnifier_wheel)
        self.left_pane.bind_external("<Button-4>", self._on_magnifier_wheel)
        self.left_pane.bind_external("<Button-5>", self._on_magnifier_wheel)

    def _build_exposure_panel(self, panel: ttk.LabelFrame) -> None:
        top = ttk.Frame(panel)
        top.pack(side=TOP, fill=X)
        self.exposure_collapse_button = ttk.Button(top, text="v", width=3, command=self._toggle_exposure_panel)
        self.exposure_collapse_button.pack(side=LEFT, padx=(0, 6))
        ttk.Checkbutton(top, text="直方图", variable=self.histogram_enabled_var, command=self._sync_quality_toggles).pack(
            side=LEFT, padx=(0, 8)
        )
        ttk.Checkbutton(top, text="斑马纹", variable=self.zebra_var, command=self._sync_quality_toggles).pack(side=LEFT)
        self.exposure_status_label = ttk.Label(top, textvariable=self.exposure_status_var, style="Panel.TLabel")
        self.exposure_status_label.pack(side=RIGHT)

        self.exposure_panel_body = ttk.Frame(panel)
        self.exposure_panel_body.pack(side=TOP, fill=X)
        hist_row = ttk.Frame(self.exposure_panel_body)
        hist_row.pack(side=TOP, fill=X, pady=(6, 0))
        hist_row.grid_columnconfigure(0, weight=1)
        hist_row.grid_columnconfigure(1, weight=1)
        self.left_hist_canvas = HistogramCanvas(hist_row, "左直方图")
        self.right_hist_canvas = HistogramCanvas(hist_row, "右直方图")
        self.left_hist_canvas.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        self.right_hist_canvas.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        ttk.Label(self.exposure_panel_body, textvariable=self.exposure_advice_var, style="Panel.TLabel").pack(side=TOP, fill=X, pady=(5, 0))

    def _build_validation_panel(self, panel: ttk.LabelFrame) -> None:
        top = ttk.Frame(panel)
        top.pack(side=TOP, fill=X)
        self.validation_collapse_button = ttk.Button(top, text="v", width=3, command=self._toggle_validation_panel)
        self.validation_collapse_button.pack(side=LEFT, padx=(0, 6))
        ttk.Label(top, text="预览模式", style="Panel.TLabel").pack(side=LEFT, padx=(0, 4))
        ttk.OptionMenu(
            top,
            self.stereo_preview_mode_var,
            self.stereo_preview_mode_var.get(),
            "正常预览",
            "红蓝立体",
            "交替闪烁",
            "校正叠加",
            command=self._on_quality_menu_changed,
        ).pack(side=LEFT, padx=(0, 10))
        ttk.Label(top, text="辅助线", style="Panel.TLabel").pack(side=LEFT, padx=(0, 4))
        ttk.OptionMenu(
            top,
            self.guide_mode_var,
            self.guide_mode_var.get(),
            "关闭",
            "中心十字",
            "全部网格线",
            command=self._on_quality_menu_changed,
        ).pack(side=LEFT, padx=(0, 10))
        self.epipolar_button = ttk.Button(top, text="极线对准检查", command=self.run_epipolar_check)
        self.epipolar_button.pack(side=LEFT)

        self.validation_panel_body = ttk.Frame(panel)
        self.validation_panel_body.pack(side=TOP, fill=X)
        ttk.Label(self.validation_panel_body, textvariable=self.capture_gate_var, style="Panel.TLabel").pack(side=TOP, fill=X, pady=(6, 0))
        self.epipolar_label = ttk.Label(self.validation_panel_body, textvariable=self.epipolar_status_var, style="Panel.TLabel")
        self.epipolar_label.pack(side=TOP, fill=X, pady=(4, 0))
        ttk.Label(self.validation_panel_body, textvariable=self.calibration_status_var, style="Panel.TLabel").pack(side=TOP, fill=X, pady=(4, 0))

    def _build_project_panel(self, panel: ttk.LabelFrame) -> None:
        top = ttk.Frame(panel)
        top.pack(side=TOP, fill=X)
        ttk.Button(top, text="新建项目", command=self.create_new_project).pack(side=LEFT, padx=(0, 6))
        ttk.Button(top, text="重载标定", command=self.reload_calibration).pack(side=LEFT, padx=(0, 6))
        ttk.Checkbutton(top, text="HDR", variable=self.hdr_enabled_var).pack(side=LEFT)

        body = ttk.Frame(panel)
        body.pack(side=TOP, fill=X, pady=(6, 0))
        ttk.Label(body, text="项目", style="Panel.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 4), pady=2)
        ttk.Label(body, textvariable=self.project_id_var, style="Panel.TLabel").grid(row=0, column=1, sticky="ew", pady=2)
        ttk.Label(body, text="EV", style="Panel.TLabel").grid(row=1, column=0, sticky="w", padx=(0, 4), pady=2)
        ttk.Entry(body, textvariable=self.hdr_sequence_var, width=18).grid(row=1, column=1, sticky="ew", pady=2)
        ttk.Label(body, textvariable=self.calibration_summary_var, style="Panel.TLabel").grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(4, 0)
        )
        ttk.Label(body, textvariable=self.temperature_status_var, style="Panel.TLabel").grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=(4, 0)
        )
        body.grid_columnconfigure(1, weight=1)

    def _configure_parameter_grid(self, panel: ttk.Frame) -> None:
        widths = {
            0: 54,
            1: 96,
            2: 36,
            3: 74,
            4: 42,
            5: 74,
            6: 42,
            7: 74,
            8: 42,
            9: 70,
            10: 90,
        }
        for column, width in widths.items():
            panel.grid_columnconfigure(column, minsize=width, weight=0)
        panel.grid_columnconfigure(10, weight=1)

    def _labeled_entry(
        self, parent: ttk.Frame, label: str, variable: StringVar, width: int = 7, row: int = 0, column: int = 0
    ) -> ttk.Entry:
        ttk.Label(parent, text=label, style="Panel.TLabel", anchor="e").grid(
            row=row, column=column, padx=(8, 3), pady=3, sticky="e"
        )
        entry = ttk.Entry(parent, textvariable=variable, width=width)
        entry.grid(row=row, column=column + 1, padx=(0, 4), pady=3, sticky="ew")
        return entry

    def _toggle_panel(self, panel_name: str) -> None:
        mapping = {
            "focus": (self.focus_panel_body, self.focus_panel_open_var, self.focus_collapse_button),
            "exposure": (self.exposure_panel_body, self.exposure_panel_open_var, self.exposure_collapse_button),
            "validation": (self.validation_panel_body, self.validation_panel_open_var, self.validation_collapse_button),
        }
        body, variable, button = mapping[panel_name]
        is_open = bool(variable.get())
        if is_open:
            body.pack_forget()
            variable.set(False)
            button.configure(text=">")
        else:
            body.pack(side=TOP, fill=X)
            variable.set(True)
            button.configure(text="v")
            self._redraw_panel_after_expand(panel_name)

    def _redraw_panel_after_expand(self, panel_name: str) -> None:
        if panel_name != "exposure":
            return
        for histogram in (getattr(self, "left_hist_canvas", None), getattr(self, "right_hist_canvas", None)):
            if isinstance(histogram, HistogramCanvas):
                histogram.redraw_when_visible()

    def _toggle_focus_panel(self) -> None:
        self._toggle_panel("focus")

    def _toggle_exposure_panel(self) -> None:
        self._toggle_panel("exposure")

    def _toggle_validation_panel(self) -> None:
        self._toggle_panel("validation")

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
        self.project_id_var.set(self.project_manager.current_project_id)
        self.status_var.set(f"新建项目：{project_dir}")

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
            self.left_pane.set_title("左相机：No Signal")
            self.left_pane.set_no_signal()
        if right_info is not None:
            self.right_pane.set_title(f"右相机：{right_info.label}")
        else:
            self.right_pane.set_title("右相机：No Signal")
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
            magnifier_rect_frac=None,
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
                self.left_pane.set_no_signal(f"Blink {side}: No Signal")
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

    def _ensure_preview_thread_after_recording(self) -> None:
        if self.camera_system is None or not self.previewing or self.recording or self.interval_capturing:
            return
        if not self.camera_system.has_ready_camera():
            self.status_var.set("录像结束后未重启预览：相机连接不可用，请重新连接相机。")
            return
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

        threading.Thread(target=worker, daemon=True).start()

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
                self._device_versions = system.device_versions()
                self._poll_camera_temperatures(force=True)
                self.ui_queue.put(("connected", (left_info, right_info)))
            except Exception as exc:
                self.ui_queue.put(("error", exc))
                self.ui_queue.put(("connect_failed", None))

        threading.Thread(target=worker, daemon=True).start()

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
        self.status_var.set("实时采集中。鼠标左键拖动画面平移，滚轮缩放；需要框选 ROI 时点击“修改ROI”。")
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
        assert self.camera_system is not None
        fps = max(float(config_snapshot.get("preview_fps", 15.0)), 0.1)
        interval = 1.0 / fps
        next_time = time.perf_counter()
        had_error = False
        consecutive_timeouts = 0

        try:
            while self.previewing and not self.recording and not self.interval_capturing:
                try:
                    left, right, _trigger_time = self.camera_system.capture_pair()
                except FrameTimeoutError as exc:
                    consecutive_timeouts += 1
                    self._handle_capture_exception(exc, "preview", consecutive_timeouts)
                    now = time.perf_counter()
                    if now - self._last_preview_status_time >= 1.0:
                        self._last_preview_status_time = now
                        self.ui_queue.put(("status", self._capture_timeout_message(exc, consecutive_timeouts)))
                    next_time = time.perf_counter() + interval
                    continue
                except Exception as exc:
                    if self._handle_capture_exception(exc, "preview", 0):
                        next_time = time.perf_counter() + interval
                        continue
                    raise

                consecutive_timeouts = 0
                if self.previewing:
                    self._preview_frame_counter += 1
                    self._poll_camera_temperatures()
                    analysis = self._analyze_preview_frames(left, right, self._preview_frame_counter)
                    self.ui_queue.put(("quality_metrics", analysis))
                    self.ui_queue.put(("frames", (left, right)))
                now = time.perf_counter()
                if now - self._last_preview_status_time >= 1.0:
                    self._last_preview_status_time = now
                    trigger_note = "等待 Line0 外触发" if self._cached_trigger_source() == "Line0" else "软件触发"
                    self.ui_queue.put(("status", self._status_with_stats(f"实时采集中：目标 {fps:g} fps；{trigger_note}")))

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
                    self.ui_queue.put(("photo_done", photo_dir))
                except Exception as exc:
                    self.ui_queue.put(("error", exc))
                finally:
                    self.ui_queue.put(("capture_idle", None))

            threading.Thread(target=record_snapshot_worker, daemon=True).start()
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
                    left, right, trigger_time = self.camera_system.capture_pair()
                    metrics = self._quality_metrics_for_pair(left, right)
                    self.ui_queue.put(("photo_quality_prefetched", (left, right, trigger_time, metrics)))
                except Exception as exc:
                    self.ui_queue.put(("error", exc))
                    self.ui_queue.put(("capture_idle", None))

            threading.Thread(target=precheck_worker, daemon=True).start()
            return

        allow_capture, quality_report = self._capture_quality_gate_allows(metrics)
        self._apply_quality_report(quality_report)
        if not allow_capture:
            self.status_var.set("采集已取消：质量检查未通过。")
            self._set_capture_buttons(NORMAL)
            return

        def worker() -> None:
            try:
                left, right, trigger_time = self.camera_system.capture_pair()
                fresh_metrics = self._quality_metrics_for_pair(left, right)
                fresh_report = self._quality_report_from_metrics(fresh_metrics)
                self.ui_queue.put(("quality_report", fresh_report))
                photo_dir = self._save_photo_pair(left, right, trigger_time, mode="photo", quality_report=fresh_report)
                if self.previewing:
                    self.ui_queue.put(("frames", (left, right)))
                self.ui_queue.put(("shutter_flash", None))
                self.ui_queue.put(("photo_done", photo_dir))
            except Exception as exc:
                self.ui_queue.put(("error", exc))
            finally:
                self.ui_queue.put(("capture_idle", None))

        threading.Thread(target=worker, daemon=True).start()

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
                captures: list[dict[str, object]] = []
                for ev in ev_offsets:
                    exposure_us = self._hdr_exposure_for_ev(original_exposure, ev)
                    self.camera_system.apply_exposure_settings("Off", exposure_us, None, None)
                    time.sleep(max(config_float(self.config.get("hdr_bracketing", {}), "settle_seconds", 0.10), 0.0))
                    left, right, trigger_time = self.camera_system.capture_pair()
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
                self.ui_queue.put(("photo_done", group_dir))
            except Exception as exc:
                self.ui_queue.put(("error", exc))
            finally:
                try:
                    self.camera_system.apply_exposure_settings(
                        original_auto,
                        original_exposure,
                        restore_lower,
                        restore_upper,
                    )
                except Exception as exc:
                    LOGGER.info("failed to restore exposure after HDR bracket: %s", exc)
                self.ui_queue.put(("capture_idle", None))

        threading.Thread(target=worker, daemon=True).start()

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
        group_dir = self.project_manager.output_root_for_mode("photos") / f"{capture_id}_hdr"
        group_dir.mkdir(parents=True, exist_ok=True)
        ext = image_extension(self.config)
        bracket_meta: list[dict[str, object]] = []
        for index, item in enumerate(captures, start=1):
            ev = float(item["ev_offset"])
            left = item.get("left")
            right = item.get("right")
            left_path = group_dir / f"ev_{ev:+.1f}_left.{ext}"
            right_path = group_dir / f"ev_{ev:+.1f}_right.{ext}"
            if isinstance(left, CameraFrame):
                self._save_image(left.image, left_path)
            if isinstance(right, CameraFrame):
                self._save_image(right.image, right_path)
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
            group_dir / "meta.json",
            mode="hdr_bracket",
            capture_id=capture_id,
            trigger_time=captures[0].get("trigger_time") if captures else time.time(),
            left=first_left,
            right=first_right,
            base_exposure_time_us=base_exposure_us,
            brackets=bracket_meta,
            data_manifest={
                "manifest_csv": str(group_dir / "exports" / "file_manifest.csv"),
                "summary_json": str(group_dir / "exports" / "capture_summary.json"),
            },
        )
        manifest = self._write_manifest_for_session(
            group_dir, {"mode": "hdr_bracket", "capture_id": capture_id, "brackets": bracket_meta}
        )
        self.project_manager.register_session("hdr_bracket", group_dir, group_dir / "meta.json", {"manifest": manifest})
        return group_dir

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
        self._set_interval_lamp("#d32f2f")
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
        self._set_interval_lamp("#666666")
        self.status_var.set("正在停止定时拍照...")

    def _interval_capture_loop(self, interval_s: float, limit: int | None) -> None:
        assert self.camera_system is not None
        had_error = False
        next_time = time.perf_counter()
        try:
            while self.interval_capturing:
                try:
                    left, right, trigger_time = self.camera_system.capture_pair()
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
                            f"定时拍照中：已保存 {self.interval_count} 组；最近 {photo_dir.name}；间隔 {interval_s:g} 秒"
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
        self._ensure_recording_config_defaults()
        try:
            fps = max(float(self.record_fps_var.get()), 0.1)
        except ValueError:
            self.status_var.set("录像 FPS 必须是数字。")
            return
        record_updates: dict[str, object] = {"record_fps": fps}
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
        config_snapshot = self._update_config(record_updates)
        if not self._check_disk_space_for_recording(config_snapshot):
            return
        self._reset_stats()
        display_enabled = self.previewing
        if display_enabled:
            self.previewing = False
            if self.preview_thread and self.preview_thread.is_alive():
                self.preview_thread.join(timeout=3)
        self.recording = True
        self.previewing = display_enabled
        self.record_count = 0
        self.record_saved_count = 0
        self._record_next_saved_index = 0
        self.record_started_at = time.perf_counter()
        self.record_stop_reason = "manual"
        self._reset_record_write_state()
        self._record_split_index = 1
        self._record_segment_start_time = self.record_started_at
        self._record_segment_start_saved = 0
        self._record_segment_sizes = {}
        self._record_skipped_frames = []
        self._record_skipped_count = 0
        self._record_skip_reasons = {}
        self._record_timeout_count = 0
        self._record_consecutive_timeouts = 0
        self._record_error_count = 0
        self._record_reconnect_count = 0
        self._record_disk_warning_count = 0
        self._record_disk_benchmark = None
        self._record_last_disk_check = 0.0
        self._record_disk_usage_start = self._disk_used_bytes(resolve_output_root(self.config))
        self._record_summary = {}
        with self._record_last_frame_lock:
            self._record_last_frame_pair = None
        self.record_dir = self.project_manager.output_root_for_mode("videos") / time.strftime("%Y%m%d_%H%M%S")
        self.record_dir.mkdir(parents=True, exist_ok=True)
        self.record_button.configure(text="停止录像")
        self._set_capture_buttons(NORMAL)
        self._set_recording_indicator(True)
        display_note = "并显示画面" if self.previewing else ""
        self.status_var.set(f"正在录像{display_note}：{self.record_dir}")
        self.record_thread = threading.Thread(target=self._record_loop, args=(config_snapshot,), daemon=True)
        self.record_thread.start()

    def stop_recording(self) -> None:
        self.recording = False
        self.record_button.configure(state=DISABLED)
        self._set_recording_indicator(False)
        self.status_var.set("正在停止录像并整理文件...")

    def _record_loop(self, config_snapshot: dict | None = None) -> None:
        """录制主循环（转入 V2 录制引擎）。"""
        self._record_loop_v2(config_snapshot or self._config_snapshot())

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

    def _record_loop_v2(self, config_snapshot: dict) -> None:
        assert self.camera_system is not None
        assert self.record_dir is not None
        fps = max(float(config_snapshot.get("record_fps", 5.0)), 0.1)
        interval = 1.0 / fps
        ext = image_extension(config_snapshot)
        record_dir = self.record_dir
        for side in ("left", "right"):
            (record_dir / self._record_segment_dir(side, 1)).mkdir(parents=True, exist_ok=True)

        image_queue: Queue[dict | None] = Queue(maxsize=max(8, int(fps * 4)))
        video_queue: Queue[dict | None] = Queue(maxsize=max(8, int(fps * 4)))
        meta_writer = RecordMetaWriter(record_dir / "frames.meta.json", config_int(config_snapshot, "record_meta_flush_every", 32))
        writer_errors: list[BaseException] = []
        writer_errors_lock = threading.Lock()
        video_outputs: dict[str, list[str]] = {"left": [], "right": []}
        next_time = time.perf_counter()
        last_status_time = 0.0
        max_seconds = max(config_float(config_snapshot, "record_max_seconds", 0.0), 0.0)
        auto_make_mp4 = config_bool(config_snapshot, "auto_make_mp4", True)
        make_mp4_after = auto_make_mp4 and shutil.which("ffmpeg") is not None
        use_realtime_mp4 = auto_make_mp4 and not make_mp4_after
        mp4_generation = "ffmpeg_after_recording" if make_mp4_after else "opencv_realtime" if use_realtime_mp4 else "disabled"
        if not auto_make_mp4:
            LOGGER.info("MP4 generation disabled by auto_make_mp4=false; recording %s sequence only.", ext.upper())

        workers = [
            threading.Thread(
                target=self._record_image_writer_loop,
                args=(image_queue, meta_writer, interval, ext, writer_errors, writer_errors_lock, config_snapshot),
                daemon=True,
            )
        ]
        if use_realtime_mp4:
            workers.append(
                threading.Thread(
                    target=self._record_video_writer_loop,
                    args=(video_queue, fps, video_outputs, writer_errors, writer_errors_lock, config_snapshot),
                    daemon=True,
                )
            )
        for worker in workers:
            worker.start()

        try:
            while self.recording:
                if max_seconds > 0 and self.record_started_at is not None:
                    if time.perf_counter() - self.record_started_at >= max_seconds:
                        self.record_stop_reason = "time_limit"
                        self.recording = False
                        break

                loop_start = time.perf_counter()
                try:
                    left, right, trigger_time = self.camera_system.capture_pair()
                except FrameTimeoutError as exc:
                    self._record_timeout_count += 1
                    self._record_consecutive_timeouts += 1
                    self._handle_capture_exception(exc, "record", self._record_consecutive_timeouts)
                    next_time = time.perf_counter() + interval
                    continue
                except Exception as exc:
                    self._record_error_count += 1
                    if self._handle_capture_exception(exc, "record", 0):
                        next_time = time.perf_counter() + interval
                        continue
                    raise
                self._record_consecutive_timeouts = 0
                self.record_count += 1
                with self._record_last_frame_lock:
                    self._record_last_frame_pair = (left, right, trigger_time)

                if self._should_record_save_frame(self.record_count):
                    self._record_next_saved_index += 1
                    item = {
                        "index": self.record_count,
                        "saved_index": self._record_next_saved_index,
                        "segment_index": self._record_split_index,
                        "trigger_time": trigger_time,
                        "left": left,
                        "right": right,
                    }
                    self._put_record_item(image_queue, item)
                    if use_realtime_mp4:
                        self._put_record_item(video_queue, item)
                else:
                    self._record_skipped("write_skip_strategy", self.record_count)

                if self.previewing:
                    self._preview_frame_counter += 1
                    analysis = self._analyze_preview_frames(left, right, self._preview_frame_counter)
                    self.ui_queue.put(("quality_metrics", analysis))
                    self.ui_queue.put(("frames", (left, right)))
                else:
                    self.ui_queue.put(("record_stats", (left, right)))

                now = time.perf_counter()
                if now - last_status_time >= 0.5:
                    last_status_time = now
                    self.ui_queue.put(
                        (
                            "status",
                            self._record_status_text(fps, self._effective_record_fps(fps), config_snapshot),
                        )
                    )
                    self.ui_queue.put(("record_progress", None))
                    self._check_disk_space_during_recording(fps, config_snapshot)
                writer_error = self._first_writer_error(writer_errors, writer_errors_lock)
                if writer_error is not None:
                    raise writer_error

                next_time += interval
                sleep_s = next_time - time.perf_counter()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                elif time.perf_counter() - loop_start > interval * 2:
                    next_time = time.perf_counter()
        except Exception as exc:
            self.ui_queue.put(("error", exc))
        finally:
            self.recording = False
            queues: list[Queue] = [image_queue]
            if use_realtime_mp4:
                queues.append(video_queue)
            self._stop_record_workers(tuple(queues), workers, config_snapshot)
            meta_writer.close()
            frames_snapshot = meta_writer.load()
            writer_errors_snapshot = self._writer_errors_snapshot(writer_errors, writer_errors_lock)
            if writer_errors_snapshot:
                self.ui_queue.put(("error", writer_errors_snapshot[0]))
                self._record_error_count += len(writer_errors_snapshot)
            output_fps = self._record_output_fps(fps)
            generated_video_names = self._finalize_recording_videos(
                record_dir,
                output_fps,
                frames_snapshot,
                video_outputs,
                config_snapshot,
            )
            summary = self._build_record_summary(record_dir, fps, output_fps, frames_snapshot)
            write_lag, _write_warning, skip_every_n, skip_keep_frames = self._record_write_state_snapshot()
            meta = {
                "mode": "video",
                "fps": fps,
                "effective_video_fps": output_fps,
                "frame_count": self.record_count,
                "saved_frame_count": self.record_saved_count,
                "skipped_frame_count": self._record_skipped_count,
                "skipped_frames": list(self._record_skipped_frames),
                "skip_reasons": dict(self._record_skip_reasons),
                "timeout_count": self._record_timeout_count,
                "error_count": self._record_error_count,
                "reconnect_count": self._record_reconnect_count,
                "disk_warning_count": self._record_disk_warning_count,
                "image_format": ext,
                "video_format": "mp4" if auto_make_mp4 else None,
                "video_codec": config_snapshot.get("video_codec", "mp4v"),
                "video_bitrate_kbps": config_int(config_snapshot, "video_bitrate_kbps", 8000),
                "video_quality_crf": config_int(config_snapshot, "video_quality_crf", 23),
                "video_preset": config_snapshot.get("video_preset", "medium"),
                "use_nvenc": config_bool(config_snapshot, "use_nvenc", False),
                "auto_make_mp4": config_bool(config_snapshot, "auto_make_mp4", True),
                "mp4_generation": mp4_generation,
                "record_split_interval_seconds": config_float(config_snapshot, "record_split_interval_seconds", 600.0),
                "record_split_size_gb": config_float(config_snapshot, "record_split_size_gb", 4.0),
                "record_max_seconds": max_seconds,
                "stop_reason": self.record_stop_reason,
                "write_lag": write_lag,
                "disk_write_benchmark": self._record_disk_benchmark,
                "skip_every_n": skip_every_n,
                "skip_keep_frames": skip_keep_frames,
                "left_videos": [str(path) for path in video_outputs["left"]],
                "right_videos": [str(path) for path in video_outputs["right"]],
                "pixel_format": config_snapshot.get("pixel_format", "Mono8"),
                "left_camera": asdict(self.camera_system.left_info) if self.camera_system.left_info else None,
                "right_camera": asdict(self.camera_system.right_info) if self.camera_system.right_info else None,
                "device_versions": dict(self._device_versions),
                "temperatures_c": dict(self._latest_temperatures),
                "temperature_samples": list(self._temperature_samples),
                "calibration": self.calibration.meta(),
                "project": self.project_manager.project_meta(),
                "data_manifest": {
                    "manifest_csv": str(record_dir / "exports" / "file_manifest.csv"),
                    "summary_json": str(record_dir / "exports" / "capture_summary.json"),
                },
                "frames": frames_snapshot,
                "summary": summary,
            }
            with (record_dir / "meta.json").open("w", encoding="utf-8") as fh:
                json.dump(meta, fh, ensure_ascii=False, indent=2)
            manifest = self._write_manifest_for_session(record_dir, summary, config_snapshot)
            self.project_manager.register_session("video", record_dir, record_dir / "meta.json", {"manifest": manifest})
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
        writer_errors: list[BaseException],
        writer_errors_lock: threading.Lock,
        exc: BaseException,
    ) -> None:
        with writer_errors_lock:
            writer_errors.append(exc)

    def _first_writer_error(
        self,
        writer_errors: list[BaseException],
        writer_errors_lock: threading.Lock,
    ) -> BaseException | None:
        with writer_errors_lock:
            return writer_errors[0] if writer_errors else None

    def _writer_errors_snapshot(
        self,
        writer_errors: list[BaseException],
        writer_errors_lock: threading.Lock,
    ) -> list[BaseException]:
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

    def _put_record_item(self, queue: Queue, item: dict) -> None:
        queued = dict(item)
        queued["left"] = self._clone_frame(item.get("left"))
        queued["right"] = self._clone_frame(item.get("right"))
        queued_ok = False
        while self.recording:
            try:
                queue.put(queued, timeout=0.25)
                queued_ok = True
                return
            except Full:
                warning = self._raise_record_write_lag(2.1, "写入队列拥堵，正在等待磁盘")
                self._notify_warning("record_queue_full", warning)
                self.ui_queue.put(("record_progress", None))
        if not queued_ok:
            self._record_skipped("recording_stopped_before_queue", int(item.get("index", 0) or 0))

    def _clone_frame(self, frame: CameraFrame | None) -> CameraFrame | None:
        if frame is None:
            return None
        return CameraFrame(
            image=frame.image.copy(),
            frame_number=frame.frame_number,
            width=frame.width,
            height=frame.height,
            host_timestamp=frame.host_timestamp,
            camera_timestamp=frame.camera_timestamp,
        )

    def _record_image_writer_loop(
        self,
        writer_queue: Queue[dict | None],
        meta_writer: RecordMetaWriter,
        interval: float,
        ext: str,
        writer_errors: list[BaseException],
        writer_errors_lock: threading.Lock,
        config_snapshot: dict,
    ) -> None:
        assert self.record_dir is not None
        batch_size = max(config_int(config_snapshot, "record_writer_batch_size", 4), 1)
        while True:
            first_item = writer_queue.get()
            if first_item is None:
                writer_queue.task_done()
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
        writer_errors: list[BaseException],
        writer_errors_lock: threading.Lock,
        config_snapshot: dict,
    ) -> None:
        assert self.record_dir is not None
        started = time.perf_counter()
        paths: dict[str, str | None] = {"left": None, "right": None}
        checksums: dict[str, str | None] = {"left": None, "right": None}
        bytes_written = 0
        try:
            saved_index = int(item["saved_index"])
            segment_index = int(item["segment_index"])
            name = f"{saved_index:06d}.{ext}"
            for side in ("left", "right"):
                frame = item.get(side)
                if frame is None:
                    continue
                path = self.record_dir / self._record_segment_dir(side, segment_index) / f"{side}_{name}"
                self._save_image(frame.image, path, config_snapshot)
                paths[side] = str(path)
                bytes_written += path.stat().st_size if path.exists() else image_estimated_bytes(frame.image)
                checksums[side] = self._file_checksum(path, config_snapshot)
            elapsed = time.perf_counter() - started
            lag = elapsed / interval if interval > 0 else 0.0
            self._observe_record_write_lag(lag)
            self.record_saved_count = saved_index
            self._record_segment_sizes[segment_index] = self._record_segment_sizes.get(segment_index, 0) + bytes_written
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
                    "checksum_algorithm": self._checksum_algorithm(config_snapshot),
                    "temperatures_c": dict(self._latest_temperatures),
                    "write_seconds": elapsed,
                    "write_lag": lag,
                }
            )
            self._advance_record_segment_if_needed(segment_index, config_snapshot)
        except BaseException as exc:
            self._append_writer_error(writer_errors, writer_errors_lock, exc)
            self.recording = False

    def _record_video_writer_loop(
        self,
        writer_queue: Queue[dict | None],
        fps: float,
        video_outputs: dict[str, list[str]],
        writer_errors: list[BaseException],
        writer_errors_lock: threading.Lock,
        config_snapshot: dict,
    ) -> None:
        writers: dict[tuple[str, int], cv2.VideoWriter] = {}
        try:
            while True:
                item = writer_queue.get()
                if item is None:
                    writer_queue.task_done()
                    break
                try:
                    segment_index = int(item["segment_index"])
                    for side in ("left", "right"):
                        frame = item.get(side)
                        if frame is None:
                            continue
                        key = (side, segment_index)
                        if key not in writers:
                            path = self._record_segment_video_path(side, segment_index)
                            writer, codec_name = self._create_video_writer_v2(path, fps, frame.image, config_snapshot)
                            writers[key] = writer
                            video_outputs[side].append(str(path))
                            if codec_name != str(config_snapshot.get("video_codec", "mp4v")):
                                self._set_record_write_warning(f"编码器回退到 {codec_name}")
                        writers[key].write(self._image_to_video_frame(frame.image))
                except BaseException as exc:
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
        image: Image.Image,
        config_snapshot: dict,
    ) -> tuple[cv2.VideoWriter, str]:
        width, height = image.size
        codec = str(config_snapshot.get("video_codec", "mp4v")).strip() or "mp4v"
        candidates = self._opencv_fourcc_candidates(codec, config_bool(config_snapshot, "use_nvenc", False))
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
        self.save_dir_var.set(value)
        self.config["save_dir"] = value
        save_config(self.config)
        self.status_var.set(f"保存路径已设置：{value}")

    def reset_view(self) -> None:
        self.left_pane.reset_zoom()
        self.right_pane.reset_zoom()
        self.status_var.set("画面缩放已还原。")

    def _set_interval_lamp(self, color: str) -> None:
        self.interval_lamp.itemconfigure(self.interval_lamp_id, fill=color)

    def _flash_interval_lamp_green(self) -> None:
        if self._interval_lamp_after_id is not None:
            self.root.after_cancel(self._interval_lamp_after_id)
            self._interval_lamp_after_id = None
        self._set_interval_lamp("#2e7d32")

        def restore() -> None:
            self._interval_lamp_after_id = None
            self._set_interval_lamp("#d32f2f" if self.interval_capturing else "#666666")

        self._interval_lamp_after_id = self.root.after(1000, restore)

    def _flash_shutter_feedback(self) -> None:
        self.left_pane.flash_shutter()
        self.right_pane.flash_shutter()

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
        self.edit_roi_button.configure(text="退出ROI" if enabled else "修改ROI")
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

    def set_roi_from_preview(self, roi: tuple[int, int, int, int]) -> None:
        if self.focus_roi_editing:
            x, y, width, height = roi
            image_width = self._last_left_frame_obj.width if self._last_left_frame_obj is not None else CAPTURE_WIDTH
            image_height = self._last_left_frame_obj.height if self._last_left_frame_obj is not None else CAPTURE_HEIGHT
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
        self.roi_width_var.set(str(width))
        self.roi_height_var.set(str(height))
        self.roi_offset_x_var.set(str(offset_x))
        self.roi_offset_y_var.set(str(offset_y))
        self.status_var.set(f"已从预览框选 ROI：W={width}, H={height}, X={offset_x}, Y={offset_y}。")
        self._set_roi_edit_mode(False)
        if self.camera_system is not None:
            self.apply_roi_settings()

    def reset_roi_settings(self) -> None:
        self.roi_width_var.set(str(CAPTURE_WIDTH))
        self.roi_height_var.set(str(CAPTURE_HEIGHT))
        self.roi_offset_x_var.set("0")
        self.roi_offset_y_var.set("0")
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
        self.config.update(preset)
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

        threading.Thread(target=worker, daemon=True).start()

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

        threading.Thread(target=worker, daemon=True).start()

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

        threading.Thread(target=worker, daemon=True).start()

    def apply_roi_settings(self, restart_stream: bool = True) -> None:
        if self.camera_system is None:
            return
        try:
            width = optional_int_text(self.roi_width_var.get()) or CAPTURE_WIDTH
            height = optional_int_text(self.roi_height_var.get()) or CAPTURE_HEIGHT
            offset_x = int(self.roi_offset_x_var.get() or 0)
            offset_y = int(self.roi_offset_y_var.get() or 0)
        except ValueError:
            self.status_var.set("ROI 参数必须是整数。")
            return
        self.apply_roi_button.configure(state=DISABLED)
        self.status_var.set("正在应用 ROI 设置...")

        def worker() -> None:
            try:
                result = self.camera_system.apply_roi_settings(width, height, offset_x, offset_y, restart_stream=restart_stream)
                actual_width, actual_height, actual_offset_x, actual_offset_y = result.actual_roi or (
                    width,
                    height,
                    offset_x,
                    offset_y,
                )
                self._update_config(
                    {
                        "roi_width": actual_width,
                        "roi_height": actual_height,
                        "roi_offset_x": actual_offset_x,
                        "roi_offset_y": actual_offset_y,
                    }
                )
                self.ui_queue.put(("roi_applied", (actual_width, actual_height, actual_offset_x, actual_offset_y, list(result))))
            except Exception as exc:
                self.ui_queue.put(("error", exc))
            finally:
                self.ui_queue.put(("param_idle", None))

        threading.Thread(target=worker, daemon=True).start()

    def apply_trigger_settings(self) -> None:
        if self.camera_system is None:
            return
        trigger_source = self.trigger_source_var.get()
        self._set_cached_trigger_source(trigger_source)
        self.apply_trigger_button.configure(state=DISABLED)
        if trigger_source == "Line0":
            self.status_var.set("正在应用 Line0 外触发模式；请确认两台相机 Line0 已接入同一个上升沿触发脉冲。")
        else:
            self.status_var.set("正在应用触发模式...")

        def worker() -> None:
            try:
                warnings = self.camera_system.apply_trigger_settings(trigger_source)
                self._update_config({"trigger_source": trigger_source})
                self._set_cached_trigger_source(trigger_source)
                message = self._format_apply_result("触发模式已应用", warnings)
                if trigger_source == "Line0":
                    message += "；Line0 模式不会发送软件触发，若没有外部脉冲会持续超时。"
                self.ui_queue.put(("status", message))
            except Exception as exc:
                self.ui_queue.put(("error", exc))
            finally:
                self.ui_queue.put(("param_idle", None))

        threading.Thread(target=worker, daemon=True).start()

    def process_ui_queue(self) -> None:
        with self._ui_queue_event_lock:
            self._ui_queue_event_pending = False
        try:
            while True:
                kind, payload = self.ui_queue.get_nowait()
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
                        f"{mode_text}连接成功。当前 ROI：{self.config.get('roi_width')}x{self.config.get('roi_height')}。"
                    )
                    if count >= 2 and config_bool(self.config, "timestamp_reject_enabled", False):
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
                elif kind == "frames":
                    left, right = payload
                    self._display_frames(left, right)
                elif kind == "quality_metrics":
                    self._apply_quality_metrics(payload)
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
                elif kind == "record_progress":
                    self._update_performance_display()
                elif kind == "shutter_flash":
                    self._flash_shutter_feedback()
                elif kind == "photo_done":
                    self.status_var.set(f"拍照完成：{payload}")
                elif kind == "roi_applied":
                    width, height, offset_x, offset_y, warnings = payload
                    self.roi_width_var.set(str(width))
                    self.roi_height_var.set(str(height))
                    self.roi_offset_x_var.set(str(offset_x))
                    self.roi_offset_y_var.set(str(offset_y))
                    self.status_var.set(
                        self._format_apply_result(
                            f"实际应用 ROI：W={width}, H={height}, X={offset_x}, Y={offset_y}",
                            warnings,
                        )
                    )
                elif kind == "capture_idle":
                    if self.camera_system is not None and not self.interval_capturing:
                        self._set_capture_buttons(NORMAL)
                elif kind == "interval_done":
                    self.interval_capturing = False
                    self.interval_button.configure(text="定时拍照")
                    self.preview_button.configure(text="停止采集" if self.previewing else "开始采集")
                    if self._interval_lamp_after_id is None:
                        self._set_interval_lamp("#666666")
                    if self.camera_system is not None and not self.recording:
                        self._set_capture_buttons(NORMAL)
                    if not payload:
                        self.status_var.set(f"定时拍照已停止，共保存 {self.interval_count} 组。")
                elif kind == "interval_lamp_green":
                    self._flash_interval_lamp_green()
                elif kind == "preview_done":
                    had_error, generation = payload
                    if generation != self._preview_generation:
                        continue
                    if self.recording or self.interval_capturing:
                        self.preview_button.configure(text="停止采集" if self.previewing else "开始采集")
                        if self.camera_system is not None:
                            self._set_capture_buttons(NORMAL)
                    else:
                        self.previewing = False
                        self.preview_button.configure(text="开始采集")
                    if self.camera_system is not None and not self.recording and not self.interval_capturing:
                        self._set_capture_buttons(NORMAL)
                        if not had_error:
                            self.status_var.set("实时采集已停止。")
                elif kind == "record_done":
                    record_dir, video_names, summary = payload
                    self.recording = False
                    self.record_button.configure(text="开始录像")
                    self.preview_button.configure(text="停止采集" if self.previewing else "开始采集")
                    self._set_recording_indicator(False)
                    self._set_capture_buttons(NORMAL)
                    self._last_video_sides = list(video_names)
                    self._ensure_preview_thread_after_recording()
                    videos = "、".join(self._last_video_sides) if self._last_video_sides else "未生成 MP4，仅保存 BMP 序列"
                    self.status_var.set(f"录像完成：{record_dir}，{videos}；{self._format_record_summary(summary)}")
                elif kind == "param_idle":
                    if self.camera_system is not None:
                        self._set_parameter_buttons(NORMAL)
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
        if self.camera_system is None:
            self.connect_button.configure(state=NORMAL)
            self.preview_button.configure(state=DISABLED)
            self.photo_button.configure(state=DISABLED)
            self.hdr_button.configure(state=DISABLED)
            self.interval_button.configure(state=DISABLED)
            self.record_button.configure(state=DISABLED)
            return

        self.connect_button.configure(state=DISABLED)
        if self.recording:
            self.preview_button.configure(state=state)
            self.photo_button.configure(state=state)
            self.hdr_button.configure(state=DISABLED)
            self.interval_button.configure(state=DISABLED)
            self.record_button.configure(state=state)
        elif self.interval_capturing:
            self.preview_button.configure(state=state)
            self.photo_button.configure(state=DISABLED)
            self.hdr_button.configure(state=DISABLED)
            self.interval_button.configure(state=state)
            self.record_button.configure(state=DISABLED)
        elif self.previewing:
            self.preview_button.configure(state=state)
            self.photo_button.configure(state=state)
            self.hdr_button.configure(state=state)
            self.interval_button.configure(state=state)
            self.record_button.configure(state=state)
        else:
            self.preview_button.configure(state=state)
            self.photo_button.configure(state=state)
            self.hdr_button.configure(state=state)
            self.interval_button.configure(state=state)
            self.record_button.configure(state=state)

    def _set_parameter_buttons(self, state: str) -> None:
        self.apply_gain_button.configure(state=state)
        self.apply_exposure_button.configure(state=state)
        self.apply_wb_button.configure(state=state)
        self.apply_roi_button.configure(state=state)
        self.apply_trigger_button.configure(state=state)

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
            "record_checksum_algorithm": "sha256",
            "record_disk_check_interval_seconds": 10.0,
            "record_disk_warning_minutes": 2.0,
            "record_disk_min_free_gb": 2.0,
            "record_stop_on_low_disk": True,
            "record_writer_stop_timeout_seconds": 10.0,
            "close_thread_join_timeout_seconds": 1.0,
            "record_writer_batch_size": 4,
            "record_meta_flush_every": 32,
            "focus_peaking_overlay_interval_seconds": 0.20,
            "record_disk_benchmark_enabled": True,
            "record_disk_benchmark_size_mb": 512.0,
            "record_disk_benchmark_seconds": 3.0,
            "record_disk_benchmark_margin": 1.25,
            "timestamp_reject_enabled": False,
            "max_camera_timestamp_delta": 0,
            "max_host_timestamp_delta": DEFAULT_HOST_TIMESTAMP_DELTA_NS,
        }
        for key, value in defaults.items():
            self.config.setdefault(key, value)
        if (
            config_bool(self.config, "timestamp_reject_enabled", False)
            and int(self.config.get("max_camera_timestamp_delta", 0) or 0) <= 0
            and int(self.config.get("max_host_timestamp_delta", 0) or 0) <= 0
        ):
            self.config["timestamp_reject_enabled"] = False

    def _ensure_project_config_defaults(self) -> None:
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
        temperature = self._ensure_config_section("temperature_monitor")
        temperature.setdefault("enabled", True)
        temperature.setdefault("interval_seconds", 30.0)
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
        if value == "中心十字":
            return "center"
        if value == "全部网格线":
            return "full"
        return "off"

    def _sync_quality_toggles(self) -> None:
        exposure_monitor = self._ensure_config_section("exposure_monitor")
        exposure_monitor["zebra_enabled"] = bool(self.zebra_var.get())
        exposure_monitor["histogram_enabled"] = bool(self.histogram_enabled_var.get())
        self._focus_peaking_enabled_setting = bool(self.focus_peaking_var.get())
        self._histogram_enabled_setting = bool(self.histogram_enabled_var.get())
        save_config(self.config)
        if hasattr(self, "left_pane"):
            self._display_frames(self._last_left_frame_obj, self._last_right_frame_obj)

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
        left_image = left.image if left is not None else None
        right_image = right.image if right is not None else None
        metrics: dict[str, object] = {
            "focus": focus_pair_metrics(left_image, right_image, roi, method),
            "focus_roi": roi,
            "temperatures_c": dict(self._latest_temperatures),
            "timestamp": time.time(),
        }
        update_histogram = self._histogram_enabled_setting and frame_index % 4 == 0
        metrics["left_exposure"] = exposure_metrics(left_image, include_histogram=update_histogram)
        metrics["right_exposure"] = exposure_metrics(right_image, include_histogram=update_histogram)
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
        self._update_capture_gate_preview()

    def _poll_camera_temperatures(self, force: bool = False) -> None:
        monitor = self.config.get("temperature_monitor", {})
        if not config_bool(monitor, "enabled", True):
            return
        if self.camera_system is None:
            return
        interval_s = max(config_float(monitor, "interval_seconds", 30.0), 1.0)
        now = time.perf_counter()
        if not force and now - self._last_temperature_poll < interval_s:
            return
        self._last_temperature_poll = now
        try:
            readings = self.camera_system.sensor_temperatures()
        except Exception as exc:
            LOGGER.info("temperature read failed: %s", exc)
            return
        self._latest_temperatures = readings
        sample = {"time": time.time(), "temperatures_c": dict(readings)}
        self._temperature_samples.append(sample)
        if len(self._temperature_samples) > 10000:
            self._temperature_samples = self._temperature_samples[-10000:]
        self.ui_queue.put(("temperature", dict(readings)))

    def _update_temperature_display(self, readings: dict[str, float | None]) -> None:
        values = {side: value for side, value in readings.items() if value is not None}
        if not values:
            self.temperature_status_var.set("Temp unavailable")
            return
        text = " | ".join(f"{side}:{value:.1f}C" for side, value in values.items())
        self.temperature_status_var.set(text)
        monitor = self.config.get("temperature_monitor", {})
        warning = config_float(monitor, "warning_threshold_c", 65.0)
        critical = config_float(monitor, "critical_threshold_c", 75.0)
        max_temp = max(values.values())
        if max_temp >= critical:
            self._notify_warning("temperature_critical", f"相机传感器温度过高：{text}")
        elif max_temp >= warning:
            self._notify_warning("temperature_warning", f"相机传感器温度偏高：{text}", log_only=True)

    def _update_focus_chart(self, score: float) -> None:
        now = time.time()
        self._focus_history.append((now, score))
        self._focus_history = self._focus_history[-240:]
        if score > self._focus_peak_score:
            self._focus_peak_score = score
        peak = max(self._focus_peak_score, 1e-9)
        pct = max(min(score / peak * 100.0, 100.0), 0.0)
        self.focus_peak_var.set(f"Peak {self._focus_peak_score:.1f} | {pct:.0f}%")
        if not hasattr(self, "focus_chart_canvas"):
            return
        canvas = self.focus_chart_canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), 40)
        height = max(canvas.winfo_height(), 32)
        canvas.create_rectangle(0, 0, width, height, fill="#151515", outline="")
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
        canvas.create_line(*points, fill="#7bd88f", width=2, smooth=True)
        canvas.create_line(4, height - 4 - (score / max_score) * (height - 8), width - 4, height - 4 - (score / max_score) * (height - 8), fill="#ffd166")

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
                self.ui_queue.put(("photo_done", photo_dir))
            except Exception as exc:
                self.ui_queue.put(("error", exc))
            finally:
                self.ui_queue.put(("capture_idle", None))

        threading.Thread(target=worker, daemon=True).start()

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
        capture_id = timestamp_ms()
        ext = image_extension(self.config)
        if left is not None:
            self._save_image(left.image, root / f"focus_ref_{capture_id}_left.{ext}")
        if right is not None:
            self._save_image(right.image, root / f"focus_ref_{capture_id}_right.{ext}")
        self._update_focus_display(focus)
        self.status_var.set(f"已保存对焦基准：{score:.1f}")

    def _start_focus_reference_check(self) -> None:
        reference = self.config.get("focus_reference_score")
        if reference in (None, "") or float(reference) <= 0 or self.camera_system is None:
            return

        def worker() -> None:
            try:
                assert self.camera_system is not None
                left, right, _trigger_time = self.camera_system.capture_pair()
                focus = focus_pair_metrics(
                    left.image if left is not None else None,
                    right.image if right is not None else None,
                    self.config.get("focus_roi"),
                    str(self.config.get("focus_method", "laplacian")),
                )
                self.ui_queue.put(("focus_reference_check", focus))
            except Exception as exc:
                self.ui_queue.put(("status", f"对焦启动检查未完成：{exc}"))

        threading.Thread(target=worker, daemon=True).start()

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
                left, right, _trigger_time = self.camera_system.capture_pair()
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

        threading.Thread(target=worker, daemon=True).start()

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
        roi = self._canvas_point_to_magnifier_roi(event.x, event.y)
        if roi is not None:
            self._magnifier_roi_frac = roi
            self.config["focus_roi"] = roi
            self.focus_roi_var.set(self._format_focus_roi())
            self._update_magnifier()

    def _on_magnifier_click(self, event) -> None:
        if not self.magnifier_enabled_var.get() or self.roi_editing:
            return
        roi = self._canvas_point_to_magnifier_roi(event.x, event.y)
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

    def _canvas_point_to_magnifier_roi(self, x: int, y: int) -> dict[str, float] | None:
        pane = self.left_pane
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

    def _update_magnifier(self) -> None:
        if not hasattr(self, "magnifier_canvas"):
            return
        self.magnifier_canvas.delete("all")
        if not self.magnifier_enabled_var.get() or self._last_left_frame_obj is None:
            self.magnifier_canvas.create_text(120, 90, text="未开启", fill="#777777", anchor="center")
            return
        image = self._last_left_frame_obj.image
        roi = clamp_roi_frac(self._magnifier_roi_frac)
        x = int(roi["x_frac"] * image.width)
        y = int(roi["y_frac"] * image.height)
        w = max(1, int(roi["w_frac"] * image.width))
        h = max(1, int(roi["h_frac"] * image.height))
        crop = image.crop((x, y, min(x + w, image.width), min(y + h, image.height))).convert("RGB")
        if self._magnifier_zoom > 1:
            crop = crop.resize((crop.width * self._magnifier_zoom, crop.height * self._magnifier_zoom), Image.Resampling.NEAREST)
        canvas_w = max(self.magnifier_canvas.winfo_width(), 120)
        canvas_h = max(self.magnifier_canvas.winfo_height(), 80)
        preview = crop.copy()
        preview.thumbnail((canvas_w, canvas_h), Image.Resampling.NEAREST)
        self._magnifier_image_ref = ImageTk.PhotoImage(preview)
        self.magnifier_canvas.create_image(canvas_w // 2, canvas_h // 2, image=self._magnifier_image_ref, anchor="center")
        lock_text = "锁定" if self._magnifier_locked else "跟随"
        self.magnifier_info_var.set(f"倍率 {self._magnifier_zoom * 100}% | {lock_text}")

    def _capture_quality_gate_allows(self, metrics: dict[str, object] | None = None) -> tuple[bool, dict[str, object]]:
        gate = self.config.get("capture_quality_gate", {})
        if not config_bool(gate, "enabled", True):
            return True, {"ok": True, "text": "采集检查已关闭"}
        report = self._quality_report_from_metrics(metrics)
        strict = config_bool(gate, "strict_mode", False)
        if report["ok"] or not strict:
            return True, report
        failed_names = [str(item["name"]) for item in report["results"] if not item["ok"]]
        allow = messagebox.askyesno("采集质量检查", f"{'、'.join(failed_names)} 未通过，是否仍要采集？")
        return allow, report

    def _quality_metrics_for_pair(self, left: CameraFrame | None, right: CameraFrame | None) -> dict[str, object]:
        metrics = self._analyze_preview_frames(left, right, self._preview_frame_counter + 1)
        self._set_last_quality_metrics(metrics)
        calibration_cfg = self.config.get("calibration_check", {})
        if config_bool(calibration_cfg, "board_coverage_enabled", False):
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
        width = optional_int_text(self.roi_width_var.get()) or CAPTURE_WIDTH
        height = optional_int_text(self.roi_height_var.get()) or CAPTURE_HEIGHT
        return {
            "trigger_source": self.trigger_source_var.get(),
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
            "roi_width": width,
            "roi_height": height,
            "roi_offset_x": int(self.roi_offset_x_var.get() or 0),
            "roi_offset_y": int(self.roi_offset_y_var.get() or 0),
        }

    def _save_current_capture_settings(self) -> None:
        values = self._current_parameter_config()
        values["interval_capture_seconds"] = float(self.interval_seconds_var.get() or 0)
        values["interval_capture_count"] = optional_int_text(self.interval_limit_var.get())
        values["record_fps"] = max(float(self.record_fps_var.get() or 0), 0.1)
        values["record_max_seconds"] = max(float(self.record_max_seconds_var.get() or 0), 0.0)
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
            "record_split_interval_seconds": 600,
            "record_split_size_gb": 4.0,
            "record_max_seconds": 0,
        }
        for key, value in defaults.items():
            self.config.setdefault(key, value)
        self._ensure_reliability_config_defaults()

    def _load_vars_from_config(self) -> None:
        self._ensure_default_full_resolution()
        self.trigger_source_var.set(str(self.config.get("trigger_source", "Software")))
        self._set_cached_trigger_source(str(self.config.get("trigger_source", "Software")))
        self.exposure_auto_var.set(str(self.config.get("exposure_auto", "Off")))
        self.exposure_time_var.set(str(self.config.get("exposure_time_us", 10000.0)))
        self.auto_exposure_lower_var.set(optional_config_text(self.config, "auto_exposure_lower_limit", "100.0"))
        self.auto_exposure_upper_var.set(optional_config_text(self.config, "auto_exposure_upper_limit", "100000.0"))
        self.gain_auto_var.set(str(self.config.get("gain_auto", "Off")))
        self.gain_var.set(str(self.config.get("gain", 0.0)))
        self.auto_gain_lower_var.set(optional_config_text(self.config, "auto_gain_lower_limit", "0.0"))
        self.auto_gain_upper_var.set(optional_config_text(self.config, "auto_gain_upper_limit", "15.0"))
        self.balance_auto_var.set(str(self.config.get("balance_white_auto", "Off")))
        self.balance_red_var.set(optional_config_text(self.config, "balance_ratio_red", ""))
        self.balance_green_var.set(optional_config_text(self.config, "balance_ratio_green", ""))
        self.balance_blue_var.set(optional_config_text(self.config, "balance_ratio_blue", ""))
        self.roi_width_var.set(str(self.config.get("roi_width", CAPTURE_WIDTH)))
        self.roi_height_var.set(str(self.config.get("roi_height", CAPTURE_HEIGHT)))
        self.roi_offset_x_var.set(str(self.config.get("roi_offset_x", 0)))
        self.roi_offset_y_var.set(str(self.config.get("roi_offset_y", 0)))
        self.record_max_seconds_var.set(optional_config_text(self.config, "record_max_seconds", "0"))

    def _format_apply_result(self, prefix: str, warnings: list[str]) -> str:
        if warnings:
            return prefix + "；" + "；".join(warnings)
        return prefix + f"到 {self._connected_camera_count()} 台相机。"

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

    def _record_status_text(self, target_fps: float, effective_fps: float, config_snapshot: dict | None = None) -> str:
        config_snapshot = config_snapshot or self._config_snapshot()
        elapsed = self._record_elapsed_seconds()
        free_gb = self._record_free_space_gb()
        write_lag, write_warning, skip_every_n, _skip_keep_frames = self._record_write_state_snapshot()
        parts = [
            f"录像中：采集 {self.record_count} 组，保存 {self.record_saved_count} 组",
            f"目标 {target_fps:g} fps，实际写入约 {effective_fps:g} fps",
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
            if config_bool(self.config, "sound_alert_enabled", True):
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
                if self._cached_trigger_source() != "Line0":
                    return self._attempt_reconnect(mode)
            LOGGER.warning(message)
            return True
        self._notify_warning(f"{mode}_capture_error", message)
        return self._attempt_reconnect(mode)

    def _disable_timestamp_reject_after_sync_error(self, detail: str) -> None:
        if self.camera_system is not None:
            self.camera_system.timestamp_reject_enabled = False
        if config_bool(self.config, "timestamp_reject_enabled", False):
            self._update_config(
                {
                    "timestamp_reject_enabled": False,
                    "max_camera_timestamp_delta": 0,
                    "max_host_timestamp_delta": 0,
                }
            )
        LOGGER.warning("Timestamp reject disabled after FrameSyncError; continuing without reconnect: %s", detail)

    def _attempt_reconnect(self, mode: str) -> bool:
        if self._closing or not config_bool(self.config, "auto_reconnect_enabled", True):
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
                    self._record_reconnect_count += 1
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
                self.record_stop_reason = "reconnect_failed"
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
        if self.record_started_at is None:
            return 0.0
        return max(time.perf_counter() - self.record_started_at, 0.0)

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

    def _effective_record_fps(self, target_fps: float) -> float:
        _write_lag, _write_warning, skip_every_n, skip_keep_frames = self._record_write_state_snapshot()
        if skip_every_n > 1:
            keep = min(max(skip_keep_frames, 1), skip_every_n)
            return target_fps * keep / skip_every_n
        return target_fps

    def _record_output_fps(self, target_fps: float) -> float:
        elapsed = self._record_elapsed_seconds()
        if elapsed > 0 and self.record_saved_count > 0:
            return max(self.record_saved_count / elapsed, 0.1)
        if self.record_saved_count <= 0:
            return 0.0
        return self._effective_record_fps(target_fps)

    def _record_skipped(self, reason: str, frame_index: int) -> None:
        if frame_index <= 0:
            return
        self._record_skipped_count += 1
        self._record_skip_reasons[reason] = self._record_skip_reasons.get(reason, 0) + 1
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
        width = int(config_snapshot.get("roi_width") or CAPTURE_WIDTH)
        height = int(config_snapshot.get("roi_height") or CAPTURE_HEIGHT)
        bytes_per_second = estimate_frame_bytes(config_snapshot, width, height) * max(self._connected_camera_count(), 1) * fps
        seconds_left = usage.free / max(bytes_per_second, 1)
        if free_gb <= min_free_gb or seconds_left <= warning_minutes * 60.0:
            self._record_disk_warning_count += 1
            message = f"磁盘空间预警：剩余 {free_gb:.1f} GB，按当前设置约可录 {format_duration(seconds_left)}。"
            self._notify_warning("record_low_disk", message)
            if config_bool(config_snapshot, "record_stop_on_low_disk", True) and free_gb <= min_free_gb:
                self.record_stop_reason = "low_disk_space"
                self.recording = False

    def _record_segment_dir(self, side: str, segment_index: int) -> str:
        if segment_index <= 1:
            return side
        return f"{side}_part{segment_index:03d}"

    def _record_segment_video_path(self, side: str, segment_index: int) -> Path:
        assert self.record_dir is not None
        suffix = "" if segment_index <= 1 else f"_part{segment_index:03d}"
        return self.record_dir / f"{side}{suffix}.mp4"

    def _advance_record_segment_if_needed(self, current_segment_index: int, config_snapshot: dict) -> None:
        if current_segment_index != self._record_split_index:
            return
        split_seconds = max(config_float(config_snapshot, "record_split_interval_seconds", 600.0), 0.0)
        split_size_gb = max(config_float(config_snapshot, "record_split_size_gb", 4.0), 0.0)
        elapsed = time.perf_counter() - self._record_segment_start_time
        segment_bytes = self._record_segment_sizes.get(current_segment_index, 0)
        should_split = (split_seconds > 0 and elapsed >= split_seconds) or (
            split_size_gb > 0 and segment_bytes >= split_size_gb * 1024**3
        )
        if not should_split:
            return
        self._record_split_index += 1
        self._record_segment_start_time = time.perf_counter()
        self._record_segment_start_saved = self.record_saved_count
        if self.record_dir is not None:
            for side in ("left", "right"):
                (self.record_dir / self._record_segment_dir(side, self._record_split_index)).mkdir(parents=True, exist_ok=True)

    def _finalize_recording_videos(
        self,
        record_dir: Path,
        fps: float,
        frames: list[dict],
        video_outputs: dict[str, list[str]],
        config_snapshot: dict,
    ) -> list[str]:
        if config_bool(config_snapshot, "auto_make_mp4", True):
            ffmpeg_outputs = self._try_make_mp4_from_frames(record_dir, fps, frames, config_snapshot)
            for side, paths in ffmpeg_outputs.items():
                if paths:
                    video_outputs[side] = [str(path) for path in paths]
        names: list[str] = []
        for side in ("left", "right"):
            for path in video_outputs[side]:
                names.append(Path(path).name)
        return names

    def _try_make_mp4_from_frames(
        self,
        record_dir: Path,
        fps: float,
        frames: list[dict],
        config_snapshot: dict,
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
                    elif config_bool(config_snapshot, "use_nvenc", False):
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
        use_nvenc = config_bool(config_snapshot, "use_nvenc", False) and not force_software
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
        usage = shutil.disk_usage(save_root)
        width = optional_int_text(self.roi_width_var.get()) or CAPTURE_WIDTH
        height = optional_int_text(self.roi_height_var.get()) or CAPTURE_HEIGHT
        frame_bytes = estimate_frame_bytes(config_snapshot, width, height)
        camera_count = max(self._connected_camera_count(), 1)
        pair_bytes = frame_bytes * camera_count
        fps = max(float(config_snapshot.get("record_fps", 5.0)), 0.1)
        estimated_one_minute = int(pair_bytes * fps * 60)
        if usage.free < estimated_one_minute:
            messagebox.showerror(
                "磁盘空间不足",
                f"当前可用空间约 {usage.free / 1024**3:.1f} GB，按当前设置录制 1 分钟预计需要 "
                f"{estimated_one_minute / 1024**3:.1f} GB。",
            )
            return False
        if usage.free < estimated_one_minute * 3:
            if not messagebox.askyesno(
                "磁盘空间偏低",
                f"当前可用空间约 {usage.free / 1024**3:.1f} GB，按当前设置录制 1 分钟预计需要 "
                f"{estimated_one_minute / 1024**3:.1f} GB。是否继续？",
            ):
                return False
        if config_bool(config_snapshot, "record_disk_benchmark_enabled", True):
            required_mbps = pair_bytes * fps / 1024 / 1024
            benchmark = benchmark_write_speed(
                save_root,
                size_mb=config_float(config_snapshot, "record_disk_benchmark_size_mb", 512.0),
                sample_seconds=config_float(config_snapshot, "record_disk_benchmark_seconds", 3.0),
            )
            self._record_disk_benchmark = benchmark
            margin = max(config_float(config_snapshot, "record_disk_benchmark_margin", 1.25), 1.0)
            if float(benchmark.get("write_mbps") or 0.0) < required_mbps * margin:
                if not messagebox.askyesno(
                    "写入速度可能不足",
                    f"当前设置预计需要 {required_mbps:.1f} MB/s，实测约 "
                    f"{float(benchmark.get('write_mbps') or 0.0):.1f} MB/s。继续录像可能丢帧，是否继续？",
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
        summary = {
            "total_frame_count": self.record_count,
            "saved_frame_count": self.record_saved_count,
            "valid_frame_count": len(frames),
            "skipped_frame_count": self._record_skipped_count,
            "timeout_count": self._record_timeout_count,
            "error_count": self._record_error_count,
            "reconnect_count": self._record_reconnect_count,
            "disk_warning_count": self._record_disk_warning_count,
            "target_fps": target_fps,
            "average_capture_fps": self.record_count / elapsed if elapsed > 0 else 0.0,
            "effective_video_fps": output_fps,
            "elapsed_seconds": elapsed,
            "directory_size_bytes": dir_bytes,
            "disk_used_delta_bytes": disk_used_delta,
            "stop_reason": self.record_stop_reason,
            "skip_reasons": dict(self._record_skip_reasons),
        }
        self._record_summary = summary
        return summary

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
        photo_root = self.project_manager.output_root_for_mode("photos")
        group_dir = photo_root / capture_id
        group_dir.mkdir(parents=True, exist_ok=True)
        ext = image_extension(self.config)

        group_left = group_dir / f"left.{ext}"
        group_right = group_dir / f"right.{ext}"

        if left is not None:
            self._save_image(left.image, group_left)
        if right is not None:
            self._save_image(right.image, group_right)
        quality_metrics = self._quality_metrics_for_pair(left, right)
        focus = quality_metrics.get("focus") if isinstance(quality_metrics.get("focus"), dict) else {}
        left_exposure = quality_metrics.get("left_exposure") if isinstance(quality_metrics.get("left_exposure"), dict) else None
        right_exposure = quality_metrics.get("right_exposure") if isinstance(quality_metrics.get("right_exposure"), dict) else None
        calibration_board = quality_metrics.get("calibration_board")
        self._write_meta(
            group_dir / "meta.json",
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
            calibration_board=calibration_board,
            capture_quality_report=quality_report or self._quality_report_from_metrics(quality_metrics),
            data_manifest={
                "manifest_csv": str(group_dir / "exports" / "file_manifest.csv"),
                "summary_json": str(group_dir / "exports" / "capture_summary.json"),
            },
        )
        manifest = self._write_manifest_for_session(
            group_dir,
            {
                "mode": mode,
                "capture_id": capture_id,
                "quality_report": quality_report or self._quality_report_from_metrics(quality_metrics),
            },
        )
        self.project_manager.register_session(mode, group_dir, group_dir / "meta.json", {"manifest": manifest})
        return group_dir

    def _save_image(self, image: Image.Image, path: Path, config_snapshot: dict | None = None) -> None:
        config_snapshot = config_snapshot or self._config_snapshot()
        ext = path.suffix.lower().lstrip(".")
        if ext == "jpeg":
            ext = "jpg"
        if ext not in {"bmp", "jpg", "png"}:
            ext = image_extension(config_snapshot)
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
            raise MvsError(f"图像保存失败：{path}；{exc}") from exc

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
            "pixel_format",
            "image_format",
            "record_fps",
            "record_max_seconds",
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
        payload["temperature_samples"] = list(self._temperature_samples[-1000:])
        payload["calibration"] = self.calibration.meta()
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
        self.interval_stop_event.set()
        if self._focus_drift_timer is not None:
            self._focus_drift_timer.cancel()
            self._focus_drift_timer = None
        if self._ui_queue_fallback_after_id is not None:
            try:
                self.root.after_cancel(self._ui_queue_fallback_after_id)
            except Exception:
                pass
            self._ui_queue_fallback_after_id = None
        if hasattr(self, "left_pane") and hasattr(self, "right_pane"):
            self.left_pane.unbind_external_callbacks()
            self._set_recording_indicator(False)
        join_timeout = max(config_float(self.config, "close_thread_join_timeout_seconds", 1.0), 0.1)
        self._join_thread_on_close(self.preview_thread, join_timeout)
        self._join_thread_on_close(self.interval_thread, join_timeout)
        self._join_thread_on_close(self.record_thread, join_timeout)
        if self.camera_system is not None:
            try:
                self.camera_system.close()
            except MvsError as exc:
                self.status_var.set(str(exc))
        self.root.destroy()

    def _join_thread_on_close(self, thread: threading.Thread | None, timeout: float) -> None:
        if thread is None or not thread.is_alive():
            return
        thread.join(timeout=timeout)
        if thread.is_alive():
            LOGGER.warning("Thread %s did not stop within %.1f seconds during close.", thread.name or thread.ident, timeout)


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

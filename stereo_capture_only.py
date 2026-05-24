from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
import time
import ctypes
import traceback
from dataclasses import asdict
from pathlib import Path
from queue import Empty, Full, Queue
from tkinter import BOTH, BOTTOM, DISABLED, LEFT, NORMAL, RIGHT, TOP, X, Canvas, Frame, StringVar, Tk, Toplevel, filedialog, messagebox
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from mvs_camera import Frame as CameraFrame
from mvs_camera import FrameTimeoutError, MvsError, StereoCameraSystem, enumerate_cameras


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
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


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_config(config: dict) -> None:
    with CONFIG_PATH.open("w", encoding="utf-8") as fh:
        json.dump(config, fh, ensure_ascii=False, indent=2)
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
    return width * height * channels


def config_bool(config: dict, key: str, default: bool) -> bool:
    value = config.get(key, default)
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "enabled"}


def config_float(config: dict, key: str, default: float) -> float:
    value = config.get(key, default)
    if value in (None, ""):
        return default
    return float(value)


def config_int(config: dict, key: str, default: int) -> int:
    value = config.get(key, default)
    if value in (None, ""):
        return default
    return int(value)


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

        self.canvas.bind("<Configure>", lambda _event: self._render())
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Button-4>", lambda _event: self.set_zoom(self.zoom * 1.1))
        self.canvas.bind("<Button-5>", lambda _event: self.set_zoom(self.zoom / 1.1))
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_press)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_release)
        self.canvas.bind("<Double-Button-1>", lambda _event: self.reset_zoom())
        self._update_cursor()

    def set_title(self, text: str) -> None:
        self.title_var.set(text)

    def set_frame(self, frame: CameraFrame) -> None:
        self._last_image = frame.image
        self.info_var.set(
            f"{frame.width}x{frame.height}  Frame:{frame.frame_number}  CamTS:{frame.camera_timestamp}"
        )
        self._render()

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
            self.canvas.coords(self._canvas_text_id, width // 2, height // 2)
            self.canvas.itemconfigure(self._canvas_text_id, text="No Signal", state="normal")
            self._place_performance_overlay()
            return

        image = self._last_image.copy()
        target_width = max(1, int(width * self.zoom))
        target_height = max(1, int(height * self.zoom))
        image.thumbnail((target_width, target_height), Image.Resampling.LANCZOS)
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
        self._place_performance_overlay()
        self._raise_overlays()

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
            self._blink_recording_overlay()
        else:
            if self._recording_after_id is not None:
                self.canvas.after_cancel(self._recording_after_id)
                self._recording_after_id = None
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
        visible = False
        if self._recording_dot_id is not None:
            visible = self.canvas.itemcget(self._recording_dot_id, "state") != "hidden"
        self._show_recording_overlay(not visible)
        self._recording_after_id = self.canvas.after(500, self._blink_recording_overlay)

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


class StereoCaptureOnlyApp:
    def __init__(self, root: Tk):
        self.root = root
        self.root.title("双目同步采集")
        self.root.configure(bg=BG_COLOR)
        self.root.geometry("1660x980")
        self.root.minsize(1280, 820)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<F11>", lambda _event: self.toggle_fullscreen())
        self.root.bind("<Escape>", lambda _event: self.root.attributes("-fullscreen", False))

        self.config = load_config()
        self._ensure_default_full_resolution()
        self._ensure_recording_config_defaults()
        self.camera_system: StereoCameraSystem | None = None
        self.ui_queue: Queue[tuple[str, object]] = Queue()

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
        self.record_count = 0
        self.record_saved_count = 0
        self._record_next_saved_index = 0
        self.record_started_at: float | None = None
        self.record_stop_reason = "manual"
        self._record_last_frame_pair: tuple[CameraFrame | None, CameraFrame | None, float] | None = None
        self._record_last_frame_lock = threading.Lock()
        self._record_write_lag = 0.0
        self._record_write_warning = ""
        self._record_skip_every_n = 1
        self._record_split_index = 1
        self._record_segment_start_time = 0.0
        self._record_segment_start_saved = 0
        self._record_segment_sizes: dict[int, int] = {}
        self._interval_lamp_after_id: str | None = None

        self._last_preview_status_time = 0.0
        self._stat_last_time: float | None = None
        self._stat_frames = 0
        self._actual_fps = 0.0
        self._last_left_frame: int | None = None
        self._last_right_frame: int | None = None
        self._drop_count = 0
        self.roi_editing = False
        self._last_device_status = "尚未刷新设备。"
        self._last_video_sides: list[str] = []

        self.status_var = StringVar(value="准备就绪。请先连接相机。")
        self.save_dir_var = StringVar(value=str(self.config.get("save_dir", "captures")))
        self._init_control_vars()
        self._configure_style()
        self._build_ui()
        self.root.after(100, self.process_ui_queue)

    def _ensure_default_full_resolution(self) -> None:
        if self.config.get("roi_width") in (None, ""):
            self.config["roi_width"] = CAPTURE_WIDTH
        if self.config.get("roi_height") in (None, ""):
            self.config["roi_height"] = CAPTURE_HEIGHT
        if self.config.get("roi_offset_x") in (None, ""):
            self.config["roi_offset_x"] = 0
        if self.config.get("roi_offset_y") in (None, ""):
            self.config["roi_offset_y"] = 0

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

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self.root, padding=(10, 8))
        toolbar.pack(side=TOP, fill=X)

        actions = ttk.Frame(toolbar)
        actions.pack(side=TOP, fill=X)
        self.connect_button = ttk.Button(actions, text="连接相机", command=self.connect_cameras, style="Accent.TButton")
        self.preview_button = ttk.Button(actions, text="开始采集", command=self.toggle_preview, state=DISABLED)
        self.photo_button = ttk.Button(actions, text="同步拍照", command=self.capture_photo, state=DISABLED)
        self.interval_button = ttk.Button(actions, text="定时拍照", command=self.toggle_interval_capture, state=DISABLED)
        self.record_button = ttk.Button(actions, text="开始录像", command=self.toggle_recording, state=DISABLED)
        self.refresh_button = ttk.Button(actions, text="刷新设备", command=self.refresh_devices)
        self.choose_save_dir_button = ttk.Button(actions, text="保存路径", command=self.choose_save_dir)
        self.exit_button = ttk.Button(actions, text="退出", command=self.close)

        for button in (
            self.connect_button,
            self.preview_button,
            self.photo_button,
            self.interval_button,
            self.record_button,
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

    def _connected_camera_count(self) -> int:
        if self.camera_system is None:
            return 0
        return sum(info is not None for info in (self.camera_system.left_info, self.camera_system.right_info))

    def _display_frames(self, left: CameraFrame | None, right: CameraFrame | None) -> None:
        self._update_stats(left, right)
        self._update_performance_display()
        if left is not None:
            self.left_pane.set_frame(left)
        else:
            self.left_pane.set_no_signal()
        if right is not None:
            self.right_pane.set_frame(right)
        else:
            self.right_pane.set_no_signal()

    def _ensure_preview_thread_after_recording(self) -> None:
        if self.camera_system is None or not self.previewing or self.recording or self.interval_capturing:
            return
        if self.preview_thread is not None and self.preview_thread.is_alive():
            return
        self._reset_stats()
        self._preview_generation += 1
        self.preview_thread = threading.Thread(target=self._preview_loop, args=(self._preview_generation,), daemon=True)
        self.preview_thread.start()

    def _start_preview_thread(self) -> None:
        self._preview_generation += 1
        self.preview_thread = threading.Thread(target=self._preview_loop, args=(self._preview_generation,), daemon=True)
        self.preview_thread.start()

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
                system_config = dict(self.config)
                system_config["allow_single_camera"] = True
                system = StereoCameraSystem(system_config)
                left_info, right_info = system.connect()
                self.camera_system = system
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
        self._reset_stats()
        self.previewing = True
        self.preview_button.configure(text="停止采集")
        self.status_var.set("实时采集中。鼠标左键拖动画面平移，滚轮缩放；需要框选 ROI 时点击“修改ROI”。")
        self._set_capture_buttons(NORMAL)
        if self.recording or self.interval_capturing:
            return
        self._start_preview_thread()

    def stop_preview(self) -> None:
        if self.recording or self.interval_capturing:
            self.previewing = False
            self.preview_button.configure(text="开始采集")
            self._set_capture_buttons(NORMAL)
            self.status_var.set("画面显示已停止，当前采集任务继续运行。")
            return
        self.previewing = False
        self.preview_button.configure(state=DISABLED)
        self.status_var.set("正在停止实时采集...")

    def _preview_loop(self, generation: int) -> None:
        assert self.camera_system is not None
        fps = max(float(self.config.get("preview_fps", 15.0)), 0.1)
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
                    now = time.perf_counter()
                    if now - self._last_preview_status_time >= 1.0:
                        self._last_preview_status_time = now
                        self.ui_queue.put(("status", self._capture_timeout_message(exc, consecutive_timeouts)))
                    next_time = time.perf_counter() + interval
                    continue

                consecutive_timeouts = 0
                if self.previewing:
                    self.ui_queue.put(("frames", (left, right)))
                now = time.perf_counter()
                if now - self._last_preview_status_time >= 1.0:
                    self._last_preview_status_time = now
                    trigger_note = "等待 Line0 外触发" if self.trigger_source_var.get() == "Line0" else "软件触发"
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
            self.status_var.set("正在从录像原始帧保存同步快照...")

            def record_snapshot_worker() -> None:
                try:
                    photo_dir = self._save_photo_pair(left_copy, right_copy, trigger_time, mode="recording_photo")
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

        def worker() -> None:
            try:
                left, right, trigger_time = self.camera_system.capture_pair()
                photo_dir = self._save_photo_pair(left, right, trigger_time, mode="photo")
                if self.previewing:
                    self.ui_queue.put(("frames", (left, right)))
                self.ui_queue.put(("shutter_flash", None))
                self.ui_queue.put(("photo_done", photo_dir))
            except Exception as exc:
                self.ui_queue.put(("error", exc))
            finally:
                self.ui_queue.put(("capture_idle", None))

        threading.Thread(target=worker, daemon=True).start()

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

        self.config["interval_capture_seconds"] = interval_s
        self.config["interval_capture_count"] = limit
        save_config(self.config)
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
                left, right, trigger_time = self.camera_system.capture_pair()
                self.interval_count += 1
                photo_dir = self._save_photo_pair(left, right, trigger_time, mode="interval_photo")
                self.ui_queue.put(("interval_lamp_green", None))
                if self.previewing:
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
        self.config["record_fps"] = fps
        try:
            self.config["record_max_seconds"] = max(float(self.record_max_seconds_var.get() or 0), 0.0)
        except ValueError:
            self.status_var.set("录像时长必须是数字；0 表示不限时。")
            return
        save_config(self.config)
        if not self._check_disk_space_for_recording():
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
        self._record_write_lag = 0.0
        self._record_write_warning = ""
        self._record_skip_every_n = 1
        self._record_split_index = 1
        self._record_segment_start_time = self.record_started_at
        self._record_segment_start_saved = 0
        self._record_segment_sizes = {}
        with self._record_last_frame_lock:
            self._record_last_frame_pair = None
        self.record_dir = resolve_output_root(self.config) / "videos" / time.strftime("%Y%m%d_%H%M%S")
        self.record_dir.mkdir(parents=True, exist_ok=True)
        self.record_button.configure(text="停止录像")
        self._set_capture_buttons(NORMAL)
        self._set_recording_indicator(True)
        display_note = "并显示画面" if self.previewing else ""
        self.status_var.set(f"正在录像{display_note}：{self.record_dir}")
        self.record_thread = threading.Thread(target=self._record_loop, daemon=True)
        self.record_thread.start()

    def stop_recording(self) -> None:
        self.recording = False
        self.record_button.configure(state=DISABLED)
        self._set_recording_indicator(False)
        self.status_var.set("正在停止录像并整理文件...")

    def _record_loop(self) -> None:
        self._record_loop_v2()
        return
        assert self.camera_system is not None
        assert self.record_dir is not None
        fps = max(float(self.config.get("record_fps", 5.0)), 0.1)
        interval = 1.0 / fps
        meta_frames = []
        next_time = time.perf_counter()
        writers: dict[str, cv2.VideoWriter] = {}
        video_paths = {
            "left": self.record_dir / "left.mp4",
            "right": self.record_dir / "right.mp4",
        }

        try:
            while self.recording:
                loop_start = time.perf_counter()
                left, right, trigger_time = self.camera_system.capture_pair()
                self.record_count += 1
                for side, frame in (("left", left), ("right", right)):
                    if frame is None:
                        continue
                    if side not in writers:
                        writers[side] = self._create_video_writer(video_paths[side], fps, frame.image)
                    writers[side].write(self._image_to_video_frame(frame.image))
                meta_frames.append(
                    {
                        "index": self.record_count,
                        "trigger_time": trigger_time,
                        "left_frame": self._frame_meta(left) if left is not None else None,
                        "right_frame": self._frame_meta(right) if right is not None else None,
                    }
                )
                if self.previewing:
                    self.ui_queue.put(("frames", (left, right)))
                self.ui_queue.put(("status", self._status_with_stats(f"录像中：已保存 {self.record_count} 组，目标 {fps:g} fps")))

                next_time += interval
                sleep_s = next_time - time.perf_counter()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                elif time.perf_counter() - loop_start > interval * 2:
                    next_time = time.perf_counter()
        except Exception as exc:
            self.ui_queue.put(("error", exc))
        finally:
            for writer in writers.values():
                writer.release()
            record_dir = self.record_dir
            if record_dir is not None:
                generated_video_names = [video_paths[side].name for side in ("left", "right") if side in writers]
                meta = {
                    "mode": "video",
                    "fps": fps,
                    "frame_count": self.record_count,
                    "video_format": "mp4",
                    "left_video": str(video_paths["left"]) if "left" in writers else None,
                    "right_video": str(video_paths["right"]) if "right" in writers else None,
                    "pixel_format": self.config.get("pixel_format", "Mono8"),
                    "left_camera": asdict(self.camera_system.left_info) if self.camera_system.left_info else None,
                    "right_camera": asdict(self.camera_system.right_info) if self.camera_system.right_info else None,
                    "frames": meta_frames,
                }
                with (record_dir / "meta.json").open("w", encoding="utf-8") as fh:
                    json.dump(meta, fh, ensure_ascii=False, indent=2)
                self.ui_queue.put(("record_done", (record_dir, generated_video_names)))

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

    def _record_loop_v2(self) -> None:
        assert self.camera_system is not None
        assert self.record_dir is not None
        fps = max(float(self.config.get("record_fps", 5.0)), 0.1)
        interval = 1.0 / fps
        ext = "bmp"
        record_dir = self.record_dir
        for side in ("left", "right"):
            (record_dir / self._record_segment_dir(side, 1)).mkdir(parents=True, exist_ok=True)

        bmp_queue: Queue[dict | None] = Queue(maxsize=max(8, int(fps * 4)))
        video_queue: Queue[dict | None] = Queue(maxsize=max(8, int(fps * 4)))
        meta_frames: list[dict] = []
        meta_lock = threading.Lock()
        writer_errors: list[BaseException] = []
        video_outputs: dict[str, list[str]] = {"left": [], "right": []}
        next_time = time.perf_counter()
        last_status_time = 0.0
        max_seconds = max(config_float(self.config, "record_max_seconds", 0.0), 0.0)
        make_mp4_after = config_bool(self.config, "auto_make_mp4", True) and shutil.which("ffmpeg") is not None
        use_realtime_mp4 = not make_mp4_after

        workers = [
            threading.Thread(
                target=self._record_bmp_writer_loop,
                args=(bmp_queue, meta_frames, meta_lock, interval, ext, writer_errors),
                daemon=True,
            )
        ]
        if use_realtime_mp4:
            workers.append(
                threading.Thread(
                    target=self._record_video_writer_loop,
                    args=(video_queue, fps, video_outputs, writer_errors),
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
                left, right, trigger_time = self.camera_system.capture_pair()
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
                    self._put_record_item(bmp_queue, item)
                    if use_realtime_mp4:
                        self._put_record_item(video_queue, item)

                if self.previewing:
                    self.ui_queue.put(("frames", (left, right)))
                else:
                    self.ui_queue.put(("record_stats", (left, right)))

                now = time.perf_counter()
                if now - last_status_time >= 0.5:
                    last_status_time = now
                    self.ui_queue.put(("status", self._record_status_text(fps, self._effective_record_fps(fps))))
                    self.ui_queue.put(("record_progress", None))
                if writer_errors:
                    raise writer_errors[0]

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
            queues: list[Queue] = [bmp_queue]
            if use_realtime_mp4:
                queues.append(video_queue)
            self._stop_record_workers(tuple(queues), workers)
            if writer_errors:
                self.ui_queue.put(("error", writer_errors[0]))
            with meta_lock:
                frames_snapshot = list(meta_frames)
            output_fps = self._record_output_fps(fps)
            generated_video_names = self._finalize_recording_videos(record_dir, output_fps, frames_snapshot, video_outputs)
            meta = {
                "mode": "video",
                "fps": fps,
                "effective_video_fps": output_fps,
                "frame_count": self.record_count,
                "saved_frame_count": self.record_saved_count,
                "image_format": ext,
                "video_format": "mp4",
                "video_codec": self.config.get("video_codec", "mp4v"),
                "video_bitrate_kbps": config_int(self.config, "video_bitrate_kbps", 8000),
                "video_quality_crf": config_int(self.config, "video_quality_crf", 23),
                "video_preset": self.config.get("video_preset", "medium"),
                "use_nvenc": config_bool(self.config, "use_nvenc", False),
                "auto_make_mp4": config_bool(self.config, "auto_make_mp4", True),
                "mp4_generation": "ffmpeg_after_recording" if make_mp4_after else "opencv_realtime",
                "record_split_interval_seconds": config_float(self.config, "record_split_interval_seconds", 600.0),
                "record_split_size_gb": config_float(self.config, "record_split_size_gb", 4.0),
                "record_max_seconds": max_seconds,
                "stop_reason": self.record_stop_reason,
                "write_lag": self._record_write_lag,
                "skip_every_n": self._record_skip_every_n,
                "left_videos": [str(path) for path in video_outputs["left"]],
                "right_videos": [str(path) for path in video_outputs["right"]],
                "pixel_format": self.config.get("pixel_format", "Mono8"),
                "left_camera": asdict(self.camera_system.left_info) if self.camera_system.left_info else None,
                "right_camera": asdict(self.camera_system.right_info) if self.camera_system.right_info else None,
                "frames": frames_snapshot,
            }
            with (record_dir / "meta.json").open("w", encoding="utf-8") as fh:
                json.dump(meta, fh, ensure_ascii=False, indent=2)
            self.ui_queue.put(("record_done", (record_dir, generated_video_names)))

    def _stop_record_workers(self, queues: tuple[Queue, ...], workers: list[threading.Thread]) -> None:
        for queue in queues:
            queue.join()
            queue.put(None)
        for worker in workers:
            worker.join()

    def _put_record_item(self, queue: Queue, item: dict) -> None:
        queued = dict(item)
        queued["left"] = self._clone_frame(item.get("left"))
        queued["right"] = self._clone_frame(item.get("right"))
        while self.recording:
            try:
                queue.put(queued, timeout=0.25)
                return
            except Full:
                self._record_write_warning = "写入队列拥堵，正在等待磁盘"
                self._record_write_lag = max(self._record_write_lag, 2.1)
                self._update_record_skip_strategy()
                self.ui_queue.put(("record_progress", None))

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

    def _record_bmp_writer_loop(
        self,
        writer_queue: Queue[dict | None],
        meta_frames: list[dict],
        meta_lock: threading.Lock,
        interval: float,
        ext: str,
        writer_errors: list[BaseException],
    ) -> None:
        assert self.record_dir is not None
        while True:
            item = writer_queue.get()
            if item is None:
                writer_queue.task_done()
                return
            started = time.perf_counter()
            paths: dict[str, str | None] = {"left": None, "right": None}
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
                    self._save_image(frame.image, path)
                    paths[side] = str(path)
                    bytes_written += path.stat().st_size if path.exists() else image_estimated_bytes(frame.image)
                elapsed = time.perf_counter() - started
                lag = elapsed / interval if interval > 0 else 0.0
                self._record_write_lag = 0.85 * self._record_write_lag + 0.15 * lag if self._record_write_lag else lag
                self._update_record_skip_strategy()
                self.record_saved_count = saved_index
                self._record_segment_sizes[segment_index] = self._record_segment_sizes.get(segment_index, 0) + bytes_written
                with meta_lock:
                    meta_frames.append(
                        {
                            "index": item["index"],
                            "saved_index": saved_index,
                            "segment_index": segment_index,
                            "trigger_time": item["trigger_time"],
                            "left_frame": self._frame_meta(item["left"]) if item.get("left") is not None else None,
                            "right_frame": self._frame_meta(item["right"]) if item.get("right") is not None else None,
                            "left_path": paths["left"],
                            "right_path": paths["right"],
                            "write_seconds": elapsed,
                            "write_lag": lag,
                        }
                    )
                self._advance_record_segment_if_needed(segment_index)
            except BaseException as exc:
                writer_errors.append(exc)
                self.recording = False
            finally:
                writer_queue.task_done()

    def _record_video_writer_loop(
        self,
        writer_queue: Queue[dict | None],
        fps: float,
        video_outputs: dict[str, list[str]],
        writer_errors: list[BaseException],
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
                            writer, codec_name = self._create_video_writer_v2(path, fps, frame.image)
                            writers[key] = writer
                            video_outputs[side].append(str(path))
                            if codec_name != str(self.config.get("video_codec", "mp4v")):
                                self._record_write_warning = f"编码器回退到 {codec_name}"
                        writers[key].write(self._image_to_video_frame(frame.image))
                except BaseException as exc:
                    writer_errors.append(exc)
                    self.recording = False
                finally:
                    writer_queue.task_done()
        finally:
            for writer in writers.values():
                writer.release()

    def _create_video_writer_v2(self, path: Path, fps: float, image: Image.Image) -> tuple[cv2.VideoWriter, str]:
        width, height = image.size
        codec = str(self.config.get("video_codec", "mp4v")).strip() or "mp4v"
        candidates = []
        if config_bool(self.config, "use_nvenc", False):
            candidates.append("h264")
        candidates.append(codec)
        if codec.lower() != "mp4v":
            candidates.append("mp4v")
        for candidate in dict.fromkeys(candidates):
            fourcc_text = "avc1" if candidate.lower() in {"h264", "h264_nvenc", "avc1"} else candidate[:4]
            fourcc = cv2.VideoWriter_fourcc(*fourcc_text)
            writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height), True)
            if writer.isOpened():
                return writer, candidate
            writer.release()
        raise RuntimeError(f"无法创建视频文件：{path}")

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
        self.edit_roi_button.configure(text="退出ROI" if enabled else "修改ROI")
        self.left_pane.set_roi_mode(enabled)
        self.right_pane.set_roi_mode(enabled)

    def set_roi_from_preview(self, roi: tuple[int, int, int, int]) -> None:
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
                self.config["gain_auto"] = gain_auto
                self.config["gain"] = gain
                self.config["auto_gain_lower_limit"] = lower
                self.config["auto_gain_upper_limit"] = upper
                save_config(self.config)
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
                self.config["exposure_auto"] = exposure_auto
                self.config["exposure_time_us"] = exposure_time
                self.config["auto_exposure_lower_limit"] = lower
                self.config["auto_exposure_upper_limit"] = upper
                save_config(self.config)
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
                self.config["balance_white_auto"] = balance_auto
                self.config["balance_ratio_red"] = red
                self.config["balance_ratio_green"] = green
                self.config["balance_ratio_blue"] = blue
                save_config(self.config)
                self.ui_queue.put(("status", self._format_apply_result("白平衡已应用", warnings)))
            except Exception as exc:
                self.ui_queue.put(("error", exc))
            finally:
                self.ui_queue.put(("param_idle", None))

        threading.Thread(target=worker, daemon=True).start()

    def apply_roi_settings(self) -> None:
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
                result = self.camera_system.apply_roi_settings(width, height, offset_x, offset_y)
                actual_width, actual_height, actual_offset_x, actual_offset_y = result.actual_roi or (
                    width,
                    height,
                    offset_x,
                    offset_y,
                )
                self.config["roi_width"] = actual_width
                self.config["roi_height"] = actual_height
                self.config["roi_offset_x"] = actual_offset_x
                self.config["roi_offset_y"] = actual_offset_y
                save_config(self.config)
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
        self.apply_trigger_button.configure(state=DISABLED)
        self.status_var.set("正在应用触发模式...")

        def worker() -> None:
            try:
                warnings = self.camera_system.apply_trigger_settings(trigger_source)
                self.config["trigger_source"] = trigger_source
                save_config(self.config)
                self.ui_queue.put(("status", self._format_apply_result("触发模式已应用", warnings)))
            except Exception as exc:
                self.ui_queue.put(("error", exc))
            finally:
                self.ui_queue.put(("param_idle", None))

        threading.Thread(target=worker, daemon=True).start()

    def process_ui_queue(self) -> None:
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
                elif kind == "connect_failed":
                    self.connect_button.configure(state=NORMAL)
                elif kind == "frames":
                    left, right = payload
                    self._display_frames(left, right)
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
                    record_dir, video_names = payload
                    self.recording = False
                    self.record_button.configure(text="开始录像")
                    self.preview_button.configure(text="停止采集" if self.previewing else "开始采集")
                    self._set_recording_indicator(False)
                    self._set_capture_buttons(NORMAL)
                    self._last_video_sides = list(video_names)
                    self._ensure_preview_thread_after_recording()
                    videos = "、".join(self._last_video_sides) if self._last_video_sides else "视频文件"
                    self.status_var.set(f"录像完成：{record_dir}，已生成 {videos}。")
                elif kind == "param_idle":
                    if self.camera_system is not None:
                        self._set_parameter_buttons(NORMAL)
                elif kind == "error":
                    self._show_error(payload)
        except Empty:
            pass
        self.root.after(100, self.process_ui_queue)

    def _set_capture_buttons(self, state: str) -> None:
        if self.camera_system is None:
            self.connect_button.configure(state=NORMAL)
            self.preview_button.configure(state=DISABLED)
            self.photo_button.configure(state=DISABLED)
            self.interval_button.configure(state=DISABLED)
            self.record_button.configure(state=DISABLED)
            return

        self.connect_button.configure(state=DISABLED)
        if self.recording:
            self.preview_button.configure(state=state)
            self.photo_button.configure(state=state)
            self.interval_button.configure(state=DISABLED)
            self.record_button.configure(state=state)
        elif self.interval_capturing:
            self.preview_button.configure(state=state)
            self.photo_button.configure(state=DISABLED)
            self.interval_button.configure(state=state)
            self.record_button.configure(state=DISABLED)
        elif self.previewing:
            self.preview_button.configure(state=state)
            self.photo_button.configure(state=state)
            self.interval_button.configure(state=state)
            self.record_button.configure(state=state)
        else:
            self.preview_button.configure(state=state)
            self.photo_button.configure(state=state)
            self.interval_button.configure(state=state)
            self.record_button.configure(state=state)

    def _set_parameter_buttons(self, state: str) -> None:
        self.apply_gain_button.configure(state=state)
        self.apply_exposure_button.configure(state=state)
        self.apply_wb_button.configure(state=state)
        self.apply_roi_button.configure(state=state)
        self.apply_trigger_button.configure(state=state)

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
        self.config.update(self._current_parameter_config())
        self.config["interval_capture_seconds"] = float(self.interval_seconds_var.get() or 0)
        self.config["interval_capture_count"] = optional_int_text(self.interval_limit_var.get())
        self.config["record_fps"] = max(float(self.record_fps_var.get() or 0), 0.1)
        self.config["record_max_seconds"] = max(float(self.record_max_seconds_var.get() or 0), 0.0)
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

    def _load_vars_from_config(self) -> None:
        self._ensure_default_full_resolution()
        self.trigger_source_var.set(str(self.config.get("trigger_source", "Software")))
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
        if left is not None and self._last_left_frame is not None:
            left_step = left.frame_number - self._last_left_frame
            if left_step > 1:
                self._drop_count += left_step - 1
        if right is not None and self._last_right_frame is not None:
            right_step = right.frame_number - self._last_right_frame
            if right_step > 1:
                self._drop_count += right_step - 1
        if left is not None:
            self._last_left_frame = left.frame_number
        if right is not None:
            self._last_right_frame = right.frame_number

    def _update_performance_display(self) -> None:
        if self._last_left_frame is None or self._last_right_frame is None:
            side = "L" if self._last_left_frame is not None else "R" if self._last_right_frame is not None else "--"
            text = f"FPS {self._actual_fps:4.1f} | Drop {self._drop_count} | {side} only"
            if self.recording:
                text += "\n" + self._record_overlay_suffix()
            self._set_performance_overlay(text, "warn" if self.recording and self._record_write_lag > 1.5 else "good")
            return
        frame_delta = self._last_left_frame - self._last_right_frame
        if self._drop_count > 0 or abs(frame_delta) > 3:
            status = "bad"
        elif abs(frame_delta) > 1 or self._actual_fps <= 0.5:
            status = "warn"
        else:
            status = "good"
        if self.recording and self._record_write_lag > 1.5:
            status = "warn" if self._record_write_lag <= 2.0 else "bad"
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
        suffix = f"已录 {format_duration(elapsed)} / 剩余空间 {free_gb:.1f} GB"
        max_seconds = max(config_float(self.config, "record_max_seconds", 0.0), 0.0)
        if max_seconds > 0:
            suffix += f" | 剩余时长 {format_duration(max_seconds - elapsed)}"
        if self._record_write_lag > 1.5:
            suffix += f" | Write lag {self._record_write_lag:.1f}x skip {self._record_skip_every_n}"
        if self._record_write_warning:
            suffix += f" | {self._record_write_warning}"
        return suffix

    def _record_status_text(self, target_fps: float, effective_fps: float) -> str:
        elapsed = self._record_elapsed_seconds()
        free_gb = self._record_free_space_gb()
        parts = [
            f"录像中：采集 {self.record_count} 组，保存 {self.record_saved_count} 组",
            f"目标 {target_fps:g} fps，实际写入约 {effective_fps:g} fps",
            f"已录 {format_duration(elapsed)} / 剩余空间 {free_gb:.1f} GB",
        ]
        max_seconds = max(config_float(self.config, "record_max_seconds", 0.0), 0.0)
        if max_seconds > 0:
            parts.append(f"剩余时长 {format_duration(max_seconds - elapsed)}")
        if self._record_write_lag > 1.5:
            parts.append(f"写入滞后 {self._record_write_lag:.1f}x，跳帧策略 {self._record_skip_every_n}")
        if self._record_write_warning:
            parts.append(self._record_write_warning)
        return "；".join(parts)

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
        if self._record_skip_every_n <= 1:
            return True
        if self._record_skip_every_n == 2:
            return frame_index % 2 == 1
        if self._record_skip_every_n == 3:
            return frame_index % 3 != 0
        return frame_index % self._record_skip_every_n == 1

    def _update_record_skip_strategy(self) -> None:
        if self._record_write_lag > 2.0:
            self._record_skip_every_n = 2
            self._record_write_warning = "磁盘写入跟不上，已每2帧写1帧"
        elif self._record_write_lag > 1.5:
            self._record_skip_every_n = 3
            self._record_write_warning = "磁盘写入偏慢，已每3帧写2帧"
        else:
            self._record_skip_every_n = 1
            if self._record_write_warning.startswith("磁盘写入"):
                self._record_write_warning = ""

    def _effective_record_fps(self, target_fps: float) -> float:
        if self._record_skip_every_n == 2:
            return target_fps / 2.0
        if self._record_skip_every_n == 3:
            return target_fps * 2.0 / 3.0
        if self._record_skip_every_n > 1:
            return target_fps / self._record_skip_every_n
        return target_fps

    def _record_output_fps(self, target_fps: float) -> float:
        elapsed = self._record_elapsed_seconds()
        if elapsed > 0 and self.record_saved_count > 0:
            return max(self.record_saved_count / elapsed, 0.1)
        return self._effective_record_fps(target_fps)

    def _record_segment_dir(self, side: str, segment_index: int) -> str:
        if segment_index <= 1:
            return side
        return f"{side}_part{segment_index:03d}"

    def _record_segment_video_path(self, side: str, segment_index: int) -> Path:
        assert self.record_dir is not None
        suffix = "" if segment_index <= 1 else f"_part{segment_index:03d}"
        return self.record_dir / f"{side}{suffix}.mp4"

    def _advance_record_segment_if_needed(self, current_segment_index: int) -> None:
        if current_segment_index != self._record_split_index:
            return
        split_seconds = max(config_float(self.config, "record_split_interval_seconds", 600.0), 0.0)
        split_size_gb = max(config_float(self.config, "record_split_size_gb", 4.0), 0.0)
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
    ) -> list[str]:
        if config_bool(self.config, "auto_make_mp4", True):
            ffmpeg_outputs = self._try_make_mp4_from_frames(record_dir, fps, frames)
            for side, paths in ffmpeg_outputs.items():
                if paths:
                    video_outputs[side] = [str(path) for path in paths]
        names: list[str] = []
        for side in ("left", "right"):
            for path in video_outputs[side]:
                names.append(Path(path).name)
        return names

    def _try_make_mp4_from_frames(self, record_dir: Path, fps: float, frames: list[dict]) -> dict[str, list[Path]]:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg or not frames:
            return {"left": [], "right": []}
        ext = "bmp"
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
                contiguous, temp_dir = self._prepare_ffmpeg_sequence(frame_dir, side, ext, segment_frames)
                input_dir = temp_dir or frame_dir
                pattern = input_dir / f"{side}_%06d.{ext}"
                if contiguous:
                    start_number = 1
                output = self._record_segment_video_path(side, segment_index)
                try:
                    command = self._ffmpeg_mp4_command(ffmpeg, pattern, output, fps, start_number)
                    result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                    if result.returncode == 0 and output.exists():
                        outputs[side].append(output)
                    elif config_bool(self.config, "use_nvenc", False):
                        fallback = self._ffmpeg_mp4_command(
                            ffmpeg,
                            pattern,
                            output,
                            fps,
                            start_number,
                            force_software=True,
                        )
                        result = subprocess.run(fallback, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                        if result.returncode == 0 and output.exists():
                            outputs[side].append(output)
                            self._record_write_warning = "NVENC失败，已回退软件编码"
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
    ) -> tuple[bool, Path | None]:
        saved_indices = sorted(int(frame["saved_index"]) for frame in segment_frames)
        if not saved_indices:
            return False, None
        expected = list(range(saved_indices[0], saved_indices[0] + len(saved_indices)))
        if saved_indices == expected:
            return False, None
        temp_dir = frame_dir / "_mp4_sequence_tmp"
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        temp_dir.mkdir(parents=True, exist_ok=True)
        for output_index, saved_index in enumerate(saved_indices, start=1):
            src = frame_dir / f"{side}_{saved_index:06d}.{ext}"
            dst = temp_dir / f"{side}_{output_index:06d}.{ext}"
            if src.exists():
                shutil.copy2(src, dst)
        return True, temp_dir

    def _ffmpeg_mp4_command(
        self,
        ffmpeg: str,
        input_pattern: Path,
        output_path: Path,
        fps: float,
        start_number: int = 1,
        force_software: bool = False,
    ) -> list[str]:
        bitrate = max(config_int(self.config, "video_bitrate_kbps", 8000), 1)
        crf = max(config_int(self.config, "video_quality_crf", 23), 0)
        preset = str(self.config.get("video_preset", "medium"))
        codec = str(self.config.get("video_codec", "mp4v")).strip().lower()
        use_nvenc = config_bool(self.config, "use_nvenc", False) and not force_software
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
            command.extend(["-c:v", "mpeg4", "-b:v", f"{bitrate}k"])
        elif use_nvenc:
            command.extend(["-c:v", "h264_nvenc", "-b:v", f"{bitrate}k", "-preset", preset])
        elif codec in {"h264", "h264_nvenc", "avc1", "libx264"}:
            command.extend(["-c:v", "libx264", "-crf", str(crf), "-preset", preset])
        else:
            command.extend(["-c:v", "mpeg4", "-b:v", f"{bitrate}k"])
        command.append(str(output_path))
        return command

    def _capture_timeout_message(self, exc: object, count: int) -> str:
        trigger_note = "等待 Line0 外触发" if self.trigger_source_var.get() == "Line0" else "软件触发后未收到图像"
        return f"实时采集中：{trigger_note}；连续超时 {count} 次。{exc}"

    def _check_disk_space_for_recording(self) -> bool:
        save_root = resolve_output_root(self.config)
        save_root.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(save_root)
        width = optional_int_text(self.roi_width_var.get()) or CAPTURE_WIDTH
        height = optional_int_text(self.roi_height_var.get()) or CAPTURE_HEIGHT
        frame_bytes = estimate_frame_bytes(self.config, width, height)
        camera_count = max(self._connected_camera_count(), 1)
        pair_bytes = frame_bytes * camera_count
        fps = max(float(self.config.get("record_fps", 5.0)), 0.1)
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
        return True

    def _optional_entry_float(self, value: StringVar) -> float | None:
        return optional_float_text(value.get())

    def _save_photo_pair(
        self,
        left: CameraFrame | None,
        right: CameraFrame | None,
        trigger_time: float,
        mode: str,
    ) -> Path:
        capture_id = timestamp_ms()
        photo_root = resolve_output_root(self.config) / "photos"
        group_dir = photo_root / capture_id
        left_dir = photo_root / "left"
        right_dir = photo_root / "right"
        group_dir.mkdir(parents=True, exist_ok=True)
        ext = image_extension(self.config)

        group_left = group_dir / f"left.{ext}"
        group_right = group_dir / f"right.{ext}"
        left_path = left_dir / f"{capture_id}_left.{ext}"
        right_path = right_dir / f"{capture_id}_right.{ext}"

        if left is not None:
            left_dir.mkdir(parents=True, exist_ok=True)
            self._save_image(left.image, group_left)
            self._save_image(left.image, left_path)
        if right is not None:
            right_dir.mkdir(parents=True, exist_ok=True)
            self._save_image(right.image, group_right)
            self._save_image(right.image, right_path)
        self._write_meta(
            group_dir / "meta.json",
            mode=mode,
            capture_id=capture_id,
            trigger_time=trigger_time,
            left=left,
            right=right,
            left_path=str(left_path) if left is not None else None,
            right_path=str(right_path) if right is not None else None,
            group_left_path=str(group_left) if left is not None else None,
            group_right_path=str(group_right) if right is not None else None,
        )
        return group_dir

    def _save_image(self, image: Image.Image, path: Path) -> None:
        ext = path.suffix.lower()
        if image.mode not in ("L", "RGB"):
            image = image.convert("RGB")
        if ext in {".jpg", ".jpeg"}:
            quality = int(self.config.get("record_jpeg_quality", 95))
            image.save(path, format="JPEG", quality=quality)
        elif ext == ".png":
            image.save(path, format="PNG")
        else:
            image.save(path, format="BMP")

    def _write_meta(self, path: Path, **data) -> None:
        payload = dict(data)
        payload["left"] = self._frame_meta(data["left"]) if data["left"] is not None else None
        payload["right"] = self._frame_meta(data["right"]) if data["right"] is not None else None
        payload["image_format"] = image_extension(self.config)
        payload["pixel_format"] = self.config.get("pixel_format", "Mono8")
        payload["left_camera"] = asdict(self.camera_system.left_info) if self.camera_system and self.camera_system.left_info else None
        payload["right_camera"] = (
            asdict(self.camera_system.right_info) if self.camera_system and self.camera_system.right_info else None
        )
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)

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
        self.previewing = False
        self.recording = False
        self.interval_capturing = False
        self.interval_stop_event.set()
        if hasattr(self, "left_pane") and hasattr(self, "right_pane"):
            self._set_recording_indicator(False)
        if self.preview_thread and self.preview_thread.is_alive():
            self.preview_thread.join(timeout=3)
        if self.interval_thread and self.interval_thread.is_alive():
            self.interval_thread.join(timeout=3)
        if self.record_thread and self.record_thread.is_alive():
            self.record_thread.join(timeout=3)
        if self.camera_system is not None:
            try:
                self.camera_system.close()
            except MvsError as exc:
                self.status_var.set(str(exc))
        self.root.destroy()


def main() -> None:
    enable_windows_dpi_awareness()
    root = Tk()
    configure_tk_dpi_scaling(root)
    StereoCaptureOnlyApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

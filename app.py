from __future__ import annotations

import json
import math
import shutil
import subprocess
import threading
import time
import traceback
import webbrowser
from dataclasses import asdict
from pathlib import Path
from queue import Empty, Queue
from tkinter import BOTH, BOTTOM, DISABLED, LEFT, NORMAL, RIGHT, TOP, X, Canvas, DoubleVar, Frame, Scrollbar, StringVar, Text, Tk, Toplevel, filedialog, messagebox
from tkinter import ttk

import numpy as np
from PIL import Image, ImageTk

from mvs_camera import Frame as CameraFrame
from mvs_camera import MvsError, StereoCameraSystem, enumerate_cameras


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
PATTERN_GENERATOR_DIR = Path(r"E:\Desktop\SAM3\calibration-pattern-generator")
BG_COLOR = "#2d2d2d"
PANEL_COLOR = "#404040"
CANVAS_COLOR = "#111111"
BORDER_COLOR = "#505050"
ACCENT_COLOR = "#3498db"
TEXT_COLOR = "#e6e6e6"
MUTED_TEXT_COLOR = "#aaaaaa"
FONT_FAMILY = "Microsoft YaHei UI"


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_config(config: dict) -> None:
    with CONFIG_PATH.open("w", encoding="utf-8") as fh:
        json.dump(config, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def timestamp_ms() -> str:
    return time.strftime("%Y%m%d_%H%M%S_") + f"{int((time.time() % 1) * 1000):03d}"


def image_extension(config: dict) -> str:
    fmt = str(config.get("image_format", "bmp")).lower().strip()
    if fmt in {"jpg", "jpeg"}:
        return "jpg"
    if fmt == "png":
        return "png"
    return "bmp"


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


def estimate_frame_bytes(config: dict, width: int = 5472, height: int = 3648) -> int:
    pixel_format = str(config.get("pixel_format", "Mono8")).lower()
    channels = 3 if "rgb" in pixel_format or "bgr" in pixel_format else 1
    return width * height * channels


def configure_matplotlib_chinese_font(matplotlib) -> None:
    candidates = ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC", "Arial Unicode MS"]
    matplotlib.rcParams["font.sans-serif"] = candidates + list(matplotlib.rcParams.get("font.sans-serif", []))
    matplotlib.rcParams["axes.unicode_minus"] = False


def resolve_output_root(config: dict) -> Path:
    configured = Path(str(config.get("save_dir", "captures")))
    return configured if configured.is_absolute() else BASE_DIR / configured


def resolve_app_path(path_text: str | Path) -> Path:
    path = Path(str(path_text).strip())
    return path if path.is_absolute() else BASE_DIR / path


def optional_config_text(config: dict, key: str, default: str = "") -> str:
    value = config.get(key, default)
    return "" if value is None else str(value)


class ImagePane(Frame):
    def __init__(self, master: Tk | Frame, title: str):
        super().__init__(master, bg=BORDER_COLOR)
        self.title_var = StringVar(value=title)
        self.info_var = StringVar(value="未连接")
        self._image_ref: ImageTk.PhotoImage | None = None
        self._last_image: Image.Image | None = None
        self.zoom = 1.0

        container = Frame(self, bg=CANVAS_COLOR, bd=0)
        container.pack(fill=BOTH, expand=True, padx=2, pady=2)
        ttk.Label(
            container,
            textvariable=self.title_var,
            style="PaneTitle.TLabel",
            padding=(12, 6),
            anchor="w",
        ).pack(side=TOP, fill=X)
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
        ttk.Label(
            container,
            textvariable=self.info_var,
            style="PaneInfo.TLabel",
            padding=(10, 4),
            anchor="w",
        ).pack(side=BOTTOM, fill=X)

        self.canvas.bind("<Configure>", lambda _event: self._render())
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Button-4>", lambda _event: self.set_zoom(self.zoom * 1.1))
        self.canvas.bind("<Button-5>", lambda _event: self.set_zoom(self.zoom / 1.1))

    def set_title(self, text: str) -> None:
        self.title_var.set(text)

    def set_frame(self, frame: CameraFrame) -> None:
        self._last_image = frame.image
        self.info_var.set(
            f"{frame.width}x{frame.height}  Frame:{frame.frame_number}  CamTS:{frame.camera_timestamp}"
        )
        self._render()

    def _render(self) -> None:
        width = max(self.canvas.winfo_width(), 100)
        height = max(self.canvas.winfo_height(), 100)
        if self._last_image is None:
            self.canvas.coords(self._canvas_text_id, width // 2, height // 2)
            return

        image = self._last_image.copy()
        target_width = max(1, int(width * self.zoom))
        target_height = max(1, int(height * self.zoom))
        image.thumbnail((target_width, target_height), Image.Resampling.BILINEAR)
        self._image_ref = ImageTk.PhotoImage(image)
        x = width // 2
        y = height // 2
        if self._canvas_image_id is None:
            self._canvas_image_id = self.canvas.create_image(x, y, image=self._image_ref, anchor="center")
        else:
            self.canvas.itemconfigure(self._canvas_image_id, image=self._image_ref)
            self.canvas.coords(self._canvas_image_id, x, y)
        self.canvas.itemconfigure(self._canvas_text_id, state="hidden")

    def _on_mouse_wheel(self, event) -> None:
        if event.delta > 0:
            self.set_zoom(self.zoom * 1.1)
        else:
            self.set_zoom(self.zoom / 1.1)

    def set_zoom(self, value: float) -> None:
        self.zoom = min(8.0, max(0.1, value))
        self._render()

    def reset_zoom(self) -> None:
        self.set_zoom(1.0)


class CalibrationImagePane(Frame):
    def __init__(self, master: Tk | Toplevel | Frame, title: str):
        super().__init__(master, bg=BORDER_COLOR)
        self.title_var = StringVar(value=title)
        self.info_var = StringVar(value="未加载")
        self._image_ref: ImageTk.PhotoImage | None = None
        self._last_image: Image.Image | None = None
        self._detected_points: list[tuple[float, float]] = []
        self._reprojected_points: list[tuple[float, float]] = []
        self.zoom = 1.0

        container = Frame(self, bg=CANVAS_COLOR, bd=0)
        container.pack(fill=BOTH, expand=True, padx=2, pady=2)
        header = ttk.Frame(container, style="Panel.TFrame")
        header.pack(side=TOP, fill=X)
        ttk.Label(header, textvariable=self.title_var, style="PaneTitle.TLabel", padding=(10, 5), anchor="w").pack(side=LEFT, fill=X, expand=True)
        ttk.Button(header, text="+", width=3, command=lambda: self.set_zoom(self.zoom * 1.2)).pack(side=LEFT, padx=(0, 4), pady=3)
        ttk.Button(header, text="-", width=3, command=lambda: self.set_zoom(self.zoom / 1.2)).pack(side=LEFT, padx=(0, 4), pady=3)
        ttk.Button(header, text="1:1", width=5, command=self.reset_zoom).pack(side=LEFT, padx=(0, 6), pady=3)

        self.canvas = Canvas(container, bg=CANVAS_COLOR, highlightthickness=0, bd=0)
        self.canvas.pack(side=TOP, fill=BOTH, expand=True)
        self._canvas_text_id = self.canvas.create_text(
            0,
            0,
            text="无图像",
            fill="#777777",
            font=(FONT_FAMILY, 14),
            anchor="center",
        )
        ttk.Label(container, textvariable=self.info_var, style="PaneInfo.TLabel", padding=(8, 3), anchor="w").pack(side=BOTTOM, fill=X)

        self.canvas.bind("<Configure>", lambda _event: self._render())
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Button-4>", lambda _event: self.set_zoom(self.zoom * 1.1))
        self.canvas.bind("<Button-5>", lambda _event: self.set_zoom(self.zoom / 1.1))

    def set_title(self, text: str) -> None:
        self.title_var.set(text)

    def set_image(
        self,
        path: str | Path,
        detected_points: list[list[float]] | None = None,
        reprojected_points: list[list[float]] | None = None,
        error_px: float | None = None,
    ) -> None:
        image = Image.open(path)
        self._last_image = image.convert("RGB")
        self._detected_points = [tuple(map(float, point)) for point in (detected_points or [])]
        self._reprojected_points = [tuple(map(float, point)) for point in (reprojected_points or [])]
        suffix = f"；重投影误差 {error_px:.3f}px" if error_px is not None else ""
        self.info_var.set(f"{Path(path).name}  {self._last_image.width}x{self._last_image.height}{suffix}")
        self._render()

    def _on_mouse_wheel(self, event) -> None:
        self.set_zoom(self.zoom * 1.1 if event.delta > 0 else self.zoom / 1.1)

    def set_zoom(self, value: float) -> None:
        self.zoom = min(8.0, max(0.1, value))
        self._render()

    def reset_zoom(self) -> None:
        self.set_zoom(1.0)

    def _render(self) -> None:
        width = max(self.canvas.winfo_width(), 100)
        height = max(self.canvas.winfo_height(), 100)
        self.canvas.delete("render")
        if self._last_image is None:
            self.canvas.coords(self._canvas_text_id, width // 2, height // 2)
            self.canvas.itemconfigure(self._canvas_text_id, state="normal")
            return

        image = self._last_image.copy()
        source_width, source_height = image.size
        target_width = max(1, int(width * self.zoom))
        target_height = max(1, int(height * self.zoom))
        image.thumbnail((target_width, target_height), Image.Resampling.BILINEAR)
        self._image_ref = ImageTk.PhotoImage(image)
        image_width, image_height = image.size
        x0 = (width - image_width) // 2
        y0 = (height - image_height) // 2
        self.canvas.create_image(width // 2, height // 2, image=self._image_ref, anchor="center", tags="render")
        self.canvas.itemconfigure(self._canvas_text_id, state="hidden")

        scale_x = image_width / max(source_width, 1)
        scale_y = image_height / max(source_height, 1)
        radius = 4
        for x, y in self._detected_points:
            cx = x0 + x * scale_x
            cy = y0 + y * scale_y
            self.canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius, outline="#00ff4c", width=2, tags="render")
        for x, y in self._reprojected_points:
            cx = x0 + x * scale_x
            cy = y0 + y * scale_y
            self.canvas.create_line(cx - radius, cy, cx + radius, cy, fill="#ff3030", width=2, tags="render")
            self.canvas.create_line(cx, cy - radius, cx, cy + radius, fill="#ff3030", width=2, tags="render")


class StereoCaptureApp:
    def __init__(self, root: Tk):
        self.root = root
        self.root.title("海康双目同步采集")
        self.root.configure(bg=BG_COLOR)
        self.root.geometry("1600x980")
        self.root.minsize(1280, 800)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._configure_style()

        self.config = load_config()
        self.camera_system: StereoCameraSystem | None = None
        self.ui_queue: Queue[tuple[str, object]] = Queue()
        self.previewing = False
        self.preview_thread: threading.Thread | None = None
        self.recording = False
        self.record_thread: threading.Thread | None = None
        self.record_dir: Path | None = None
        self.record_count = 0
        self.interval_capturing = False
        self.interval_thread: threading.Thread | None = None
        self.interval_stop_event = threading.Event()
        self.interval_count = 0
        self.calibrating = False
        self.calibration_window: Toplevel | None = None
        self.calibration_summary_vars: dict[str, StringVar] = {}
        self.last_calibration_result: dict | None = None
        self.calibration_view_pairs: list[dict] = []
        self.calibration_pair_index = 0
        self.calibration_3d_image_ref: ImageTk.PhotoImage | None = None
        self.calibration_progress_var = DoubleVar(value=0.0)
        self.calibration_progress_text_var = StringVar(value="等待开始。")
        self._last_preview_status_time = 0.0
        self._stat_last_time: float | None = None
        self._stat_frames = 0
        self._actual_fps = 0.0
        self._last_left_frame: int | None = None
        self._last_right_frame: int | None = None
        self._drop_count = 0

        self.status_var = StringVar(value="准备就绪。先连接相机，再点击开始采集进行实时预览。预览中可同步拍照。")
        self.gain_auto_var = StringVar(value=str(self.config.get("gain_auto", "Off")))
        self.gain_var = StringVar(value=str(self.config.get("gain", 0.0)))
        self.auto_gain_lower_var = StringVar(value=str(self.config.get("auto_gain_lower_limit", 0.0)))
        self.auto_gain_upper_var = StringVar(value=str(self.config.get("auto_gain_upper_limit", 15.0)))
        self.exposure_auto_var = StringVar(value=str(self.config.get("exposure_auto", "Off")))
        self.exposure_time_var = StringVar(value=str(self.config.get("exposure_time_us", 10000.0)))
        self.auto_exposure_lower_var = StringVar(value=str(self.config.get("auto_exposure_lower_limit", 100.0)))
        self.auto_exposure_upper_var = StringVar(value=str(self.config.get("auto_exposure_upper_limit", 100000.0)))
        self.balance_auto_var = StringVar(value=str(self.config.get("balance_white_auto", "Off")))
        self.balance_red_var = StringVar(value=optional_config_text(self.config, "balance_ratio_red", ""))
        self.balance_green_var = StringVar(value=optional_config_text(self.config, "balance_ratio_green", ""))
        self.balance_blue_var = StringVar(value=optional_config_text(self.config, "balance_ratio_blue", ""))
        self.roi_width_var = StringVar(value=optional_config_text(self.config, "roi_width", ""))
        self.roi_height_var = StringVar(value=optional_config_text(self.config, "roi_height", ""))
        self.roi_offset_x_var = StringVar(value=str(self.config.get("roi_offset_x", 0)))
        self.roi_offset_y_var = StringVar(value=str(self.config.get("roi_offset_y", 0)))
        self.trigger_source_var = StringVar(value=str(self.config.get("trigger_source", "Software")))
        self.save_dir_var = StringVar(value=str(self.config.get("save_dir", "captures")))
        self.preset_var = StringVar(value="室内低光")
        self.interval_seconds_var = StringVar(value=optional_config_text(self.config, "interval_capture_seconds", "5.0"))
        self.interval_limit_var = StringVar(value=optional_config_text(self.config, "interval_capture_count", ""))
        default_left_dir = str(resolve_output_root(self.config) / "photos" / "left")
        default_right_dir = str(resolve_output_root(self.config) / "photos" / "right")
        default_calibration_dir = str(resolve_output_root(self.config) / "calibration")
        self.calib_left_dir_var = StringVar(value=optional_config_text(self.config, "calibration_left_dir", default_left_dir))
        self.calib_right_dir_var = StringVar(value=optional_config_text(self.config, "calibration_right_dir", default_right_dir))
        self.calib_output_dir_var = StringVar(value=optional_config_text(self.config, "calibration_output_dir", default_calibration_dir))
        self.calib_pattern_var = StringVar(value=optional_config_text(self.config, "calibration_pattern", "chessboard"))
        self.calib_columns_var = StringVar(value=optional_config_text(self.config, "calibration_columns", "9"))
        self.calib_rows_var = StringVar(value=optional_config_text(self.config, "calibration_rows", "6"))
        self.calib_square_size_var = StringVar(value=optional_config_text(self.config, "calibration_square_size_mm", "20.0"))
        self.calib_marker_size_var = StringVar(value=optional_config_text(self.config, "calibration_marker_size_mm", "15.0"))
        self.calib_dictionary_var = StringVar(value=optional_config_text(self.config, "calibration_aruco_dictionary", "DICT_4X4_50"))

        toolbar = ttk.Frame(root, padding=(10, 8))
        toolbar.pack(side=TOP, fill=X)

        control_panel = ttk.LabelFrame(toolbar, text="控制面板", padding=(10, 8))
        control_panel.pack(side=TOP, fill=X, pady=(0, 8))
        actions_panel = ttk.Frame(control_panel)
        actions_panel.pack(side=TOP, fill=X)
        settings_panel = ttk.Frame(control_panel)
        settings_panel.pack(side=TOP, fill=X, pady=(8, 0))

        self.connect_button = ttk.Button(actions_panel, text="连接相机", command=self.connect_cameras, style="Accent.TButton")
        self.preview_button = ttk.Button(actions_panel, text="开始采集", command=self.toggle_preview, state=DISABLED)
        self.photo_button = ttk.Button(actions_panel, text="同步拍照", command=self.capture_photo, state=DISABLED)
        self.interval_button = ttk.Button(actions_panel, text="定时拍照", command=self.toggle_interval_capture, state=DISABLED)
        self.record_button = ttk.Button(actions_panel, text="开始录像", command=self.toggle_recording, state=DISABLED)
        self.reset_view_button = ttk.Button(actions_panel, text="还原画面", command=self.reset_view)
        self.open_calibration_button = ttk.Button(actions_panel, text="相机标定", command=self.open_calibration_page)
        self.refresh_button = ttk.Button(actions_panel, text="刷新设备", command=self.refresh_devices)
        self.choose_save_dir_button = ttk.Button(actions_panel, text="保存路径", command=self.choose_save_dir)
        self.exit_button = ttk.Button(actions_panel, text="退出", command=self.close)
        for button in (
            self.connect_button,
            self.preview_button,
            self.photo_button,
            self.interval_button,
            self.record_button,
            self.reset_view_button,
            self.open_calibration_button,
            self.refresh_button,
            self.choose_save_dir_button,
        ):
            button.pack(side=LEFT, padx=(0, 8))
        self.exit_button.pack(side=RIGHT)

        trigger_panel = ttk.Frame(settings_panel)
        trigger_panel.pack(side=LEFT, padx=(0, 16))
        ttk.Label(trigger_panel, text="触发").grid(row=0, column=0, padx=(0, 4), pady=2)
        ttk.OptionMenu(trigger_panel, self.trigger_source_var, self.trigger_source_var.get(), "Software", "Line0").grid(row=0, column=1, padx=3, pady=2)
        self.apply_trigger_button = ttk.Button(trigger_panel, text="应用触发", command=self.apply_trigger_settings, state=DISABLED)
        self.apply_trigger_button.grid(row=0, column=2, padx=(6, 0), pady=2)

        preset_panel = ttk.Frame(settings_panel)
        preset_panel.pack(side=LEFT, padx=(0, 16))
        ttk.Label(preset_panel, text="预设").grid(row=0, column=0, padx=(0, 4), pady=2)
        ttk.OptionMenu(preset_panel, self.preset_var, self.preset_var.get(), "室内低光", "室外强光").grid(row=0, column=1, padx=3, pady=2)
        ttk.Button(preset_panel, text="加载", command=self.load_preset).grid(row=0, column=2, padx=3, pady=2)
        ttk.Button(preset_panel, text="保存", command=self.save_preset).grid(row=0, column=3, padx=3, pady=2)

        interval_panel = ttk.Frame(settings_panel)
        interval_panel.pack(side=LEFT)
        ttk.Label(interval_panel, text="定时").grid(row=0, column=0, padx=(0, 4), pady=2)
        self._labeled_entry(interval_panel, "秒", self.interval_seconds_var, 6, 0, 1)
        self._labeled_entry(interval_panel, "张数", self.interval_limit_var, 6, 0, 3)

        param_panel = ttk.LabelFrame(toolbar, text="参数设置", padding=(10, 8))
        param_panel.pack(side=TOP, fill=X)
        for i in range(2):
            param_panel.grid_columnconfigure(i, weight=1)

        gain_panel = ttk.Frame(param_panel, style="Panel.TFrame", padding=(8, 6))
        gain_panel.grid(row=0, column=0, sticky="ew", padx=(0, 8), pady=4)
        ttk.Label(gain_panel, text="增益", style="Panel.TLabel").grid(row=0, column=0, padx=(0, 6), pady=3, sticky="w")
        ttk.OptionMenu(gain_panel, self.gain_auto_var, self.gain_auto_var.get(), "Off", "Once", "Continuous").grid(row=0, column=1, padx=3, pady=3, sticky="w")
        self.apply_gain_button = ttk.Button(gain_panel, text="应用增益", command=self.apply_gain_settings, state=DISABLED)
        self.apply_gain_button.grid(row=0, column=2, padx=(8, 0), pady=3, sticky="w")
        self._labeled_entry(gain_panel, "值", self.gain_var, 6, 1, 0)
        self._labeled_entry(gain_panel, "下限", self.auto_gain_lower_var, 6, 1, 2)
        self._labeled_entry(gain_panel, "上限", self.auto_gain_upper_var, 6, 1, 4)

        exposure_panel = ttk.Frame(param_panel, style="Panel.TFrame", padding=(8, 6))
        exposure_panel.grid(row=0, column=1, sticky="ew", padx=(8, 0), pady=4)
        ttk.Label(exposure_panel, text="曝光", style="Panel.TLabel").grid(row=0, column=0, padx=(0, 6), pady=3, sticky="w")
        ttk.OptionMenu(exposure_panel, self.exposure_auto_var, self.exposure_auto_var.get(), "Off", "Once", "Continuous").grid(row=0, column=1, padx=3, pady=3, sticky="w")
        self.apply_exposure_button = ttk.Button(exposure_panel, text="应用曝光", command=self.apply_exposure_settings, state=DISABLED)
        self.apply_exposure_button.grid(row=0, column=2, padx=(8, 0), pady=3, sticky="w")
        self._labeled_entry(exposure_panel, "us", self.exposure_time_var, 8, 1, 0)
        self._labeled_entry(exposure_panel, "下限", self.auto_exposure_lower_var, 8, 1, 2)
        self._labeled_entry(exposure_panel, "上限", self.auto_exposure_upper_var, 8, 1, 4)

        wb_panel = ttk.Frame(param_panel, style="Panel.TFrame", padding=(8, 6))
        wb_panel.grid(row=1, column=0, sticky="ew", padx=(0, 8), pady=4)
        ttk.Label(wb_panel, text="白平衡", style="Panel.TLabel").grid(row=0, column=0, padx=(0, 6), pady=3, sticky="w")
        ttk.OptionMenu(wb_panel, self.balance_auto_var, self.balance_auto_var.get(), "Off", "Once", "Continuous").grid(row=0, column=1, padx=3, pady=3, sticky="w")
        self.apply_wb_button = ttk.Button(wb_panel, text="应用白平衡", command=self.apply_white_balance_settings, state=DISABLED)
        self.apply_wb_button.grid(row=0, column=2, padx=(8, 0), pady=3, sticky="w")
        self._labeled_entry(wb_panel, "R", self.balance_red_var, 5, 1, 0)
        self._labeled_entry(wb_panel, "G", self.balance_green_var, 5, 1, 2)
        self._labeled_entry(wb_panel, "B", self.balance_blue_var, 5, 1, 4)

        roi_panel = ttk.Frame(param_panel, style="Panel.TFrame", padding=(8, 6))
        roi_panel.grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=4)
        ttk.Label(roi_panel, text="ROI", style="Panel.TLabel").grid(row=0, column=0, padx=(0, 6), pady=3, sticky="w")
        self.apply_roi_button = ttk.Button(roi_panel, text="应用ROI", command=self.apply_roi_settings, state=DISABLED)
        self.apply_roi_button.grid(row=0, column=1, padx=(8, 0), pady=3, sticky="w")
        self._labeled_entry(roi_panel, "W", self.roi_width_var, 6, 1, 0)
        self._labeled_entry(roi_panel, "H", self.roi_height_var, 6, 1, 2)
        self._labeled_entry(roi_panel, "X", self.roi_offset_x_var, 5, 1, 4)
        self._labeled_entry(roi_panel, "Y", self.roi_offset_y_var, 5, 1, 6)

        self.calibrate_button: ttk.Button | None = None

        content = Frame(root, bg=BG_COLOR)
        content.pack(side=TOP, fill=BOTH, expand=True)
        content.grid_columnconfigure(0, weight=1, uniform="camera")
        content.grid_columnconfigure(1, weight=1, uniform="camera")
        content.grid_rowconfigure(0, weight=1)
        self.left_pane = ImagePane(content, "左相机")
        self.right_pane = ImagePane(content, "右相机")
        self.left_pane.grid(row=0, column=0, sticky="nsew", padx=(8, 4), pady=8)
        self.right_pane.grid(row=0, column=1, sticky="nsew", padx=(4, 8), pady=8)

        ttk.Separator(root, orient="horizontal").pack(side=TOP, fill=X)
        self.status_bar = ttk.Label(root, textvariable=self.status_var, style="Status.TLabel", anchor="w", padding=(10, 6))
        self.status_bar.pack(side=BOTTOM, fill=X)

        self.root.after(100, self.process_ui_queue)

    def _configure_style(self) -> None:
        self.style = ttk.Style()
        try:
            self.style.theme_use("clam")
        except Exception:
            pass
        self.style.configure(".", background=BG_COLOR, foreground=TEXT_COLOR, font=(FONT_FAMILY, 9))
        self.style.configure("TFrame", background=BG_COLOR)
        self.style.configure("Panel.TFrame", background=PANEL_COLOR)
        self.style.configure("TLabel", background=BG_COLOR, foreground=TEXT_COLOR, font=(FONT_FAMILY, 9))
        self.style.configure("Panel.TLabel", background=PANEL_COLOR, foreground=TEXT_COLOR)
        self.style.configure("Muted.TLabel", background=BG_COLOR, foreground=MUTED_TEXT_COLOR)
        self.style.configure("PaneTitle.TLabel", background=PANEL_COLOR, foreground="white", font=(FONT_FAMILY, 13, "bold"))
        self.style.configure("PaneInfo.TLabel", background=PANEL_COLOR, foreground="#d7d7d7", font=("Consolas", 10))
        self.style.configure("Status.TLabel", background=BG_COLOR, foreground=MUTED_TEXT_COLOR, font=(FONT_FAMILY, 9))
        self.style.configure("TButton", background=PANEL_COLOR, foreground="white", borderwidth=0, padding=(10, 6))
        self.style.map(
            "TButton",
            background=[("active", "#505050"), ("disabled", "#303030")],
            foreground=[("disabled", "#777777")],
        )
        self.style.configure("Accent.TButton", background=ACCENT_COLOR, foreground="white", borderwidth=0, padding=(12, 7))
        self.style.map("Accent.TButton", background=[("active", "#4aa3df"), ("disabled", "#303030")])
        self.style.configure("TEntry", fieldbackground="#3d3d3d", foreground="white", bordercolor="#555555", lightcolor="#555555", darkcolor="#555555", insertcolor="white")
        self.style.configure("TMenubutton", background=PANEL_COLOR, foreground="white", borderwidth=0, padding=(8, 5))
        self.style.map("TMenubutton", background=[("active", "#505050"), ("disabled", "#303030")])
        self.style.configure("TLabelframe", background=BG_COLOR, bordercolor="#555555", relief="solid")
        self.style.configure("TLabelframe.Label", background=BG_COLOR, foreground="#dcdcdc", font=(FONT_FAMILY, 9, "bold"))
        self.style.configure("Horizontal.TSeparator", background="#555555")

    def _labeled_entry(self, parent, label: str, variable: StringVar, width: int = 7, row: int = 0, column: int = 0) -> ttk.Entry:
        ttk.Label(parent, text=label, style="Panel.TLabel").grid(row=row, column=column, padx=(8, 3), pady=3, sticky="w")
        entry = ttk.Entry(parent, textvariable=variable, width=width)
        entry.grid(row=row, column=column + 1, padx=(0, 4), pady=3)
        return entry

    def reset_view(self) -> None:
        self.left_pane.reset_zoom()
        self.right_pane.reset_zoom()
        self.status_var.set("画面缩放已还原。")

    def open_calibration_page(self) -> None:
        if self.calibration_window is not None and self.calibration_window.winfo_exists():
            self.calibration_window.lift()
            self.calibration_window.focus_force()
            return

        window = Toplevel(self.root)
        self.calibration_window = window
        window.title("相机标定")
        window.configure(bg=BG_COLOR)
        window.geometry("1500x900")
        window.minsize(1180, 720)
        window.protocol("WM_DELETE_WINDOW", self.close_calibration_page)

        container = ttk.Frame(window, padding=(14, 12))
        container.pack(side=TOP, fill=BOTH, expand=True)
        container.grid_columnconfigure(0, weight=1)

        source_panel = ttk.LabelFrame(container, text="标定图像", padding=(10, 8))
        source_panel.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        source_panel.grid_columnconfigure(1, weight=1)
        source_panel.grid_columnconfigure(4, weight=1)
        source_panel.grid_columnconfigure(7, weight=1)
        ttk.Label(source_panel, text="左图").grid(row=0, column=0, padx=(0, 4), pady=4, sticky="w")
        ttk.Entry(source_panel, textvariable=self.calib_left_dir_var, width=34).grid(row=0, column=1, padx=(0, 4), pady=4, sticky="ew")
        ttk.Button(source_panel, text="选择", command=lambda: self.choose_calibration_dir(self.calib_left_dir_var)).grid(row=0, column=2, padx=(0, 12), pady=4)
        ttk.Label(source_panel, text="右图").grid(row=0, column=3, padx=(0, 4), pady=4, sticky="w")
        ttk.Entry(source_panel, textvariable=self.calib_right_dir_var, width=34).grid(row=0, column=4, padx=(0, 4), pady=4, sticky="ew")
        ttk.Button(source_panel, text="选择", command=lambda: self.choose_calibration_dir(self.calib_right_dir_var)).grid(row=0, column=5, padx=(0, 12), pady=4)
        ttk.Label(source_panel, text="输出").grid(row=0, column=6, padx=(0, 4), pady=4, sticky="w")
        ttk.Entry(source_panel, textvariable=self.calib_output_dir_var, width=34).grid(row=0, column=7, padx=(0, 4), pady=4, sticky="ew")
        ttk.Button(source_panel, text="选择", command=lambda: self.choose_calibration_dir(self.calib_output_dir_var)).grid(row=0, column=8, padx=(0, 0), pady=4)

        board_panel = ttk.LabelFrame(container, text="标定板", padding=(10, 8))
        board_panel.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        for column in (1, 3, 5, 7, 9):
            board_panel.grid_columnconfigure(column, weight=1)
        ttk.Label(board_panel, text="类型").grid(row=0, column=0, padx=(0, 4), pady=4, sticky="w")
        ttk.OptionMenu(
            board_panel,
            self.calib_pattern_var,
            self.calib_pattern_var.get(),
            "chessboard",
            "charuco",
            "charuco_legacy",
            "circles",
            "acircles",
        ).grid(row=0, column=1, padx=(0, 12), pady=4, sticky="w")
        self._labeled_entry(board_panel, "列", self.calib_columns_var, 7, 0, 2)
        self._labeled_entry(board_panel, "行", self.calib_rows_var, 7, 0, 4)
        self._labeled_entry(board_panel, "格mm", self.calib_square_size_var, 8, 0, 6)
        self._labeled_entry(board_panel, "码mm", self.calib_marker_size_var, 8, 0, 8)
        ttk.Label(board_panel, text="字典").grid(row=1, column=0, padx=(0, 4), pady=4, sticky="w")
        ttk.OptionMenu(
            board_panel,
            self.calib_dictionary_var,
            self.calib_dictionary_var.get(),
            "DICT_4X4_50",
            "DICT_4X4_100",
            "DICT_5X5_100",
            "DICT_6X6_250",
            "DICT_7X7_1000",
            "DICT_ARUCO_ORIGINAL",
            "DICT_APRILTAG_36h11",
        ).grid(row=1, column=1, columnspan=2, padx=(0, 12), pady=4, sticky="w")
        ttk.Button(board_panel, text="导入标定板图片", command=self.import_calibration_board_image).grid(row=1, column=3, padx=(0, 8), pady=4, sticky="w")
        ttk.Button(board_panel, text="生成标定板", command=self.open_pattern_generator).grid(row=1, column=4, padx=(0, 8), pady=4, sticky="w")
        self.calibrate_button = ttk.Button(board_panel, text="开始标定", command=self.start_calibration, style="Accent.TButton")
        self.calibrate_button.grid(row=1, column=5, padx=(0, 8), pady=4, sticky="w")

        summary_panel = ttk.LabelFrame(container, text="标定结果摘要", padding=(10, 8))
        summary_panel.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        summary_panel.grid_columnconfigure(0, weight=1)
        self.calibration_summary_vars = {}
        summary_items = [
            ("mono_error", "单目重投影误差"),
            ("valid_pairs", "单目有效图像"),
            ("stereo_rms", "双目 RMS"),
            ("baseline", "基线"),
            ("intrinsics", "内参 / 畸变"),
            ("calibration_date", "标定日期"),
        ]
        for column, (key, label) in enumerate(summary_items):
            frame = ttk.Frame(summary_panel)
            frame.grid(row=0, column=column, padx=(10 if column else 0, 22), pady=2, sticky="w")
            ttk.Label(frame, text=label).pack(side=TOP, anchor="w")
            value_var = StringVar(value="--")
            self.calibration_summary_vars[key] = value_var
            ttk.Label(frame, textvariable=value_var, font=(FONT_FAMILY, 12, "bold")).pack(side=TOP, anchor="w")
        ttk.Button(summary_panel, text="刷新标定摘要", command=self.refresh_calibration_summary).grid(row=0, column=len(summary_items), padx=(12, 8), pady=2)
        ttk.Button(summary_panel, text="打开 ChArUco 三维图", command=self.open_calibration_3d_view).grid(row=0, column=len(summary_items) + 1, padx=(0, 0), pady=2)
        summary_detail_frame = ttk.Frame(summary_panel)
        summary_detail_frame.grid(row=1, column=0, columnspan=len(summary_items) + 2, sticky="ew", pady=(8, 0))
        summary_detail_frame.grid_rowconfigure(0, weight=1)
        summary_detail_frame.grid_columnconfigure(0, weight=1)
        self.calibration_summary_text = Text(
            summary_detail_frame,
            bg=CANVAS_COLOR,
            fg=TEXT_COLOR,
            insertbackground=TEXT_COLOR,
            relief="flat",
            wrap="word",
            font=("Consolas", 10),
            height=8,
        )
        self.calibration_summary_text.grid(row=0, column=0, sticky="ew")
        summary_scroll = Scrollbar(summary_detail_frame, command=self.calibration_summary_text.yview)
        summary_scroll.grid(row=0, column=1, sticky="ns")
        self.calibration_summary_text.configure(yscrollcommand=summary_scroll.set, state="disabled")

        progress_panel = ttk.LabelFrame(container, text="标定进度", padding=(10, 8))
        progress_panel.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        progress_panel.grid_columnconfigure(0, weight=1)
        self.calibration_progress_bar = ttk.Progressbar(progress_panel, maximum=100.0, variable=self.calibration_progress_var)
        self.calibration_progress_bar.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        ttk.Label(progress_panel, textvariable=self.calibration_progress_text_var).grid(row=0, column=1, sticky="e")

        result_panel = ttk.LabelFrame(container, text="标定可视化", padding=(10, 8))
        result_panel.grid(row=4, column=0, sticky="nsew")
        result_panel.grid_columnconfigure(0, weight=1)
        result_panel.grid_columnconfigure(1, weight=1)
        result_panel.grid_columnconfigure(2, weight=1)
        result_panel.grid_rowconfigure(1, weight=1)
        container.grid_rowconfigure(4, weight=1)
        self.calibration_status_var = StringVar(value="选择左右图目录，确认标定板参数后开始标定。")
        ttk.Label(result_panel, textvariable=self.calibration_status_var, style="Status.TLabel", anchor="w", padding=(0, 4)).grid(row=0, column=0, columnspan=3, sticky="ew")

        self.calibration_left_image_pane = CalibrationImagePane(result_panel, "左相机识别")
        self.calibration_right_image_pane = CalibrationImagePane(result_panel, "右相机识别")
        self.calibration_left_image_pane.grid(row=1, column=0, sticky="nsew", padx=(0, 6), pady=(4, 0))
        self.calibration_right_image_pane.grid(row=1, column=1, sticky="nsew", padx=(6, 6), pady=(4, 0))

        right_result = ttk.Frame(result_panel)
        right_result.grid(row=1, column=2, sticky="nsew", padx=(6, 0), pady=(4, 0))
        right_result.grid_rowconfigure(1, weight=1)
        right_result.grid_rowconfigure(3, weight=1)
        right_result.grid_columnconfigure(0, weight=1)
        nav = ttk.Frame(right_result)
        nav.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ttk.Button(nav, text="上一张", command=lambda: self.show_calibration_pair(self.calibration_pair_index - 1)).pack(side=LEFT, padx=(0, 6))
        self.calibration_pair_var = StringVar(value="--")
        ttk.Label(nav, textvariable=self.calibration_pair_var).pack(side=LEFT, padx=(0, 6))
        ttk.Button(nav, text="下一张", command=lambda: self.show_calibration_pair(self.calibration_pair_index + 1)).pack(side=LEFT, padx=(0, 12))
        ttk.Button(nav, text="重绘三维图", command=self.refresh_calibration_3d_plot).pack(side=LEFT)

        self.calibration_3d_canvas = Canvas(right_result, bg="white", highlightthickness=0, height=260)
        self.calibration_3d_canvas.grid(row=1, column=0, sticky="nsew")
        detail_frame = ttk.Frame(right_result)
        detail_frame.grid(row=3, column=0, sticky="nsew", pady=(8, 0))
        detail_frame.grid_rowconfigure(0, weight=1)
        detail_frame.grid_columnconfigure(0, weight=1)
        self.calibration_result_text = Text(
            detail_frame,
            bg=CANVAS_COLOR,
            fg=TEXT_COLOR,
            insertbackground=TEXT_COLOR,
            relief="flat",
            wrap="word",
            font=("Consolas", 10),
            height=10,
        )
        self.calibration_result_text.grid(row=0, column=0, sticky="nsew")
        result_scroll = Scrollbar(detail_frame, command=self.calibration_result_text.yview)
        result_scroll.grid(row=0, column=1, sticky="ns")
        self.calibration_result_text.configure(yscrollcommand=result_scroll.set, state="disabled")
        self._set_calibration_status("选择左右图目录，确认标定板参数后开始标定。")
        if self.last_calibration_result is not None:
            self.render_calibration_result(self.last_calibration_result)

    def close_calibration_page(self) -> None:
        if self.calibration_window is not None and self.calibration_window.winfo_exists():
            self.calibration_window.destroy()
        self.calibration_window = None
        self.calibrate_button = None

    def _set_calibration_status(self, text: str) -> None:
        if hasattr(self, "calibration_status_var"):
            self.calibration_status_var.set(text)
        self.status_var.set(text)

    def _set_calibration_result_text(self, text: str) -> None:
        if hasattr(self, "calibration_result_text"):
            self.calibration_result_text.configure(state="normal")
            self.calibration_result_text.delete("1.0", "end")
            self.calibration_result_text.insert("1.0", text)
            self.calibration_result_text.configure(state="disabled")

    def _set_calibration_summary_text(self, text: str) -> None:
        if hasattr(self, "calibration_summary_text"):
            self.calibration_summary_text.configure(state="normal")
            self.calibration_summary_text.delete("1.0", "end")
            self.calibration_summary_text.insert("1.0", text)
            self.calibration_summary_text.configure(state="disabled")

    def _set_calibration_progress(self, value: float, text: str) -> None:
        if hasattr(self, "calibration_progress_var"):
            self.calibration_progress_var.set(max(0.0, min(100.0, float(value))))
        if hasattr(self, "calibration_progress_text_var"):
            self.calibration_progress_text_var.set(text)

    def refresh_calibration_summary(self) -> None:
        if not hasattr(self, "calibration_summary_vars") or not self.calibration_summary_vars:
            return
        if self.calibration_view_pairs:
            self._set_calibration_status("标定摘要已刷新。")
        else:
            self._set_calibration_status("暂无标定结果。请先开始标定。")
        if self.last_calibration_result is not None:
            self._set_calibration_summary_text(self._format_calibration_detail(self.last_calibration_result))

    def open_calibration_3d_view(self) -> None:
        if not self.calibration_view_pairs:
            self._set_calibration_status("暂无三维位姿数据。请先完成标定。")
            return
        self.refresh_calibration_3d_plot()
        if self.calibration_window is not None and self.calibration_window.winfo_exists():
            self.calibration_window.lift()
            self.calibration_window.focus_force()

    def show_calibration_pair(self, index: int) -> None:
        if not self.calibration_view_pairs:
            if hasattr(self, "calibration_pair_var"):
                self.calibration_pair_var.set("--")
            return
        index = max(0, min(index, len(self.calibration_view_pairs) - 1))
        self.calibration_pair_index = index
        pair = self.calibration_view_pairs[index]
        if hasattr(self, "calibration_pair_var"):
            self.calibration_pair_var.set(f"{index + 1}/{len(self.calibration_view_pairs)}  {pair.get('key', '')}")
        if hasattr(self, "calibration_left_image_pane"):
            self.calibration_left_image_pane.set_image(
                pair["left"],
                pair.get("left_points"),
                pair.get("left_reprojected_points"),
                pair.get("left_reprojection_error_px"),
            )
        if hasattr(self, "calibration_right_image_pane"):
            self.calibration_right_image_pane.set_image(
                pair["right"],
                pair.get("right_points"),
                pair.get("right_reprojected_points"),
                pair.get("right_reprojection_error_px"),
            )
        detail = (
            f"图像 {index + 1}/{len(self.calibration_view_pairs)}\n"
            f"匹配键：{pair.get('key', '')}\n"
            f"角点数量：{pair.get('point_count', '--')}\n"
            f"左图重投影误差：{pair.get('left_reprojection_error_px', 0):.4f} px\n"
            f"右图重投影误差：{pair.get('right_reprojection_error_px', 0):.4f} px\n"
        )
        if self.last_calibration_result is not None:
            left = self.last_calibration_result["left"]["matlab_like_intrinsics"]
            right = self.last_calibration_result["right"]["matlab_like_intrinsics"]
            stereo = self.last_calibration_result["stereo"]
            detail += (
                "\n中文参数摘要\n"
                f"左相机焦距 fx/fy：{left['focal_length_px'][0]:.3f} / {left['focal_length_px'][1]:.3f} px\n"
                f"左相机主点 cx/cy：{left['principal_point_px'][0]:.3f} / {left['principal_point_px'][1]:.3f} px\n"
                f"左相机径向畸变：{left['radial_distortion']}\n"
                f"左相机切向畸变：{left['tangential_distortion']}\n"
                f"右相机焦距 fx/fy：{right['focal_length_px'][0]:.3f} / {right['focal_length_px'][1]:.3f} px\n"
                f"右相机主点 cx/cy：{right['principal_point_px'][0]:.3f} / {right['principal_point_px'][1]:.3f} px\n"
                f"基线：{stereo['baseline_mm']:.3f} mm\n"
                f"双目平移向量 T：{stereo['translation_vector']}\n"
            )
        self._set_calibration_result_text(detail)

    def render_calibration_result(self, result: dict) -> None:
        self.last_calibration_result = result
        self.calibration_view_pairs = list(result.get("accepted_pairs", []))
        self.calibration_pair_index = 0
        summary = self._format_calibration_summary(result)
        self._update_calibration_summary_vars(result)
        self._set_calibration_status(summary)
        self._set_calibration_summary_text(self._format_calibration_detail(result))
        self.show_calibration_pair(0)
        self.refresh_calibration_3d_plot(result)

    def _format_calibration_summary(self, result: dict) -> str:
        left = result["left"]
        right = result["right"]
        stereo = result["stereo"]
        return (
            f"标定完成：有效 {result['accepted_pair_count']}/{result['total_pairs']} 对；"
            f"左 RMS {left['rms_reprojection_error_px']:.4f}px，"
            f"右 RMS {right['rms_reprojection_error_px']:.4f}px，"
            f"双目 RMS {stereo['rms_reprojection_error_px']:.4f}px，"
            f"基线 {stereo['baseline_mm']:.3f} mm；"
            f"标定日期 {result.get('calibration_date', '--')}；"
            f"详细参数见下方摘要文本。"
        )

    def _update_calibration_summary_vars(self, result: dict) -> None:
        if not self.calibration_summary_vars:
            return
        left_rms = result["left"]["rms_reprojection_error_px"]
        right_rms = result["right"]["rms_reprojection_error_px"]
        self.calibration_summary_vars["mono_error"].set(f"左 {left_rms:.3f}px / 右 {right_rms:.3f}px")
        self.calibration_summary_vars["valid_pairs"].set(f"{result['accepted_pair_count']} / {result['total_pairs']} 对")
        self.calibration_summary_vars["stereo_rms"].set(f"{result['stereo']['rms_reprojection_error_px']:.3f}px")
        self.calibration_summary_vars["baseline"].set(f"{result['stereo']['baseline_mm']:.3f} mm")
        left_intr = result["left"]["matlab_like_intrinsics"]
        self.calibration_summary_vars["intrinsics"].set(
            f"fx {left_intr['focal_length_px'][0]:.1f}, fy {left_intr['focal_length_px'][1]:.1f}"
        )

    def _format_calibration_detail(self, result: dict) -> str:
        left = result["left"]["matlab_like_intrinsics"]
        right = result["right"]["matlab_like_intrinsics"]
        stereo = result["stereo"]
        rect = stereo.get("rectification", {})
        reconstruction = result.get("artifacts", {}).get("reconstruction", {})
        return (
            f"标定日期: {result.get('calibration_date', '--')}\n"
            f"分辨率: {result.get('image_size', [])}\n"
            f"左K: {left['camera_matrix_opencv']}\n"
            f"左D: {left['distortion_coefficients_opencv']}\n"
            f"右K: {right['camera_matrix_opencv']}\n"
            f"右D: {right['distortion_coefficients_opencv']}\n"
            f"R: {stereo['rotation_matrix']}\n"
            f"T: {stereo['translation_vector']}\n"
            f"R1: {rect.get('R1', [])}\n"
            f"R2: {rect.get('R2', [])}\n"
            f"P1: {rect.get('P1', [])}\n"
            f"P2: {rect.get('P2', [])}\n"
            f"Q: {rect.get('Q', [])}\n"
            f"重投影误差: 左 {result['left']['rms_reprojection_error_px']:.4f}px / 右 {result['right']['rms_reprojection_error_px']:.4f}px / 双目 {stereo['rms_reprojection_error_px']:.4f}px\n"
            f"基线: {stereo['baseline_mm']:.3f} mm\n"
            f"覆盖热图: {result.get('artifacts', {}).get('board_coverage_heatmap', {}).get('image', '--')}\n"
            f"误差分布图: {result.get('artifacts', {}).get('reprojection_error_distribution', {}).get('image', '--')}\n"
            f"深度误差曲线: {result.get('artifacts', {}).get('depth_error_curve', {}).get('image', '--')}\n"
            f"重建输出: {reconstruction.get('reconstruction_result', '--')}\n"
        )

    def refresh_calibration_3d_plot(self, result: dict | None = None) -> None:
        if result is None:
            if not self.calibration_view_pairs:
                return
            result = {"accepted_pairs": self.calibration_view_pairs}
        if not hasattr(self, "calibration_3d_canvas"):
            return
        try:
            image = self._make_calibration_3d_image(result)
        except Exception as exc:
            self._set_calibration_result_text(f"三维位姿图生成失败：{exc}")
            return
        canvas_width = max(self.calibration_3d_canvas.winfo_width(), 360)
        canvas_height = max(self.calibration_3d_canvas.winfo_height(), 240)
        image.thumbnail((canvas_width, canvas_height), Image.Resampling.BILINEAR)
        self.calibration_3d_image_ref = ImageTk.PhotoImage(image)
        self.calibration_3d_canvas.delete("all")
        self.calibration_3d_canvas.create_image(canvas_width // 2, canvas_height // 2, image=self.calibration_3d_image_ref, anchor="center")

    def _make_calibration_3d_image(self, result: dict) -> Image.Image:
        import matplotlib

        matplotlib.use("Agg")
        configure_matplotlib_chinese_font(matplotlib)
        import matplotlib.pyplot as plt

        pairs = result.get("accepted_pairs", [])
        fig = plt.figure(figsize=(6.2, 4.2), dpi=120)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title("标定板三维位置（左相机坐标系）")
        ax.set_xlabel("X 向右 (mm)")
        ax.set_ylabel("Z 向前 (mm)")
        ax.set_zlabel("-Y 向上 (mm)")
        ax.scatter([0], [0], [0], c="blue", marker="o", s=60)
        ax.text(0, 0, 0, "左相机", color="blue")

        colors = plt.cm.tab20(np.linspace(0, 1, max(len(pairs), 1)))
        all_points = [np.array([[0.0, 0.0, 0.0]])]
        for index, pair in enumerate(pairs):
            pose = pair.get("board_pose_left_camera", {})
            t = np.asarray(pose.get("translation_vector_mm", [0, 0, 0]), dtype=float).reshape(3)
            obj = np.asarray(pair.get("object_points", []), dtype=float)
            color = colors[index % len(colors)]
            if obj.size >= 12:
                min_xy = obj[:, :2].min(axis=0)
                max_xy = obj[:, :2].max(axis=0)
                corners = np.array(
                    [
                        [min_xy[0], min_xy[1], 0.0],
                        [max_xy[0], min_xy[1], 0.0],
                        [max_xy[0], max_xy[1], 0.0],
                        [min_xy[0], max_xy[1], 0.0],
                        [min_xy[0], min_xy[1], 0.0],
                    ]
                )
                rvec = np.asarray(pose.get("rotation_vector", [0, 0, 0]), dtype=float).reshape(3)
                rotated = self._rotate_points_rodrigues(corners, rvec) + t
                plot_points = self._camera_points_to_plot_points(rotated)
                ax.plot(plot_points[:, 0], plot_points[:, 1], plot_points[:, 2], color=color, linewidth=1.5)
                all_points.append(plot_points)
                center = rotated[:4].mean(axis=0)
            else:
                center = t
                plot_t = self._camera_points_to_plot_points(t.reshape(1, 3))
                ax.scatter([plot_t[0, 0]], [plot_t[0, 1]], [plot_t[0, 2]], color=color)
                all_points.append(plot_t)
            plot_center = self._camera_points_to_plot_points(center.reshape(1, 3))[0]
            ax.text(plot_center[0], plot_center[1], plot_center[2], str(index + 1), color=color)

        ax.grid(True)
        self._set_3d_axes_equal(ax, np.vstack(all_points))
        ax.view_init(elev=24, azim=-58)
        fig.tight_layout()
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        image = Image.fromarray(rgba).convert("RGB")
        plt.close(fig)
        return image

    def _camera_points_to_plot_points(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        return np.column_stack((pts[:, 0], pts[:, 2], -pts[:, 1]))

    def _set_3d_axes_equal(self, ax, points: np.ndarray) -> None:
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        centers = (mins + maxs) / 2.0
        radius = max(float(np.max(maxs - mins)) / 2.0, 1.0)
        ax.set_xlim(centers[0] - radius, centers[0] + radius)
        ax.set_ylim(centers[1] - radius, centers[1] + radius)
        ax.set_zlim(centers[2] - radius, centers[2] + radius)

    def _rotate_points_rodrigues(self, points: np.ndarray, rvec: np.ndarray) -> np.ndarray:
        theta = float(np.linalg.norm(rvec))
        if theta < 1e-12:
            return points
        k = rvec / theta
        kx = np.array(
            [
                [0, -k[2], k[1]],
                [k[2], 0, -k[0]],
                [-k[1], k[0], 0],
            ],
            dtype=float,
        )
        rotation = np.eye(3) + math.sin(theta) * kx + (1 - math.cos(theta)) * (kx @ kx)
        return points @ rotation.T

    def refresh_devices(self) -> None:
        def worker() -> None:
            try:
                cameras, _dev_list = enumerate_cameras()
                if not cameras:
                    self.ui_queue.put(("status", "未检测到相机。"))
                    return
                summary = "；".join(f"{cam.index}: {cam.label} [{cam.transport}]" for cam in cameras)
                self.ui_queue.put(("status", f"检测到 {len(cameras)} 台相机：{summary}"))
            except Exception as exc:
                self.ui_queue.put(("error", exc))

        threading.Thread(target=worker, daemon=True).start()

    def connect_cameras(self) -> None:
        self.connect_button.configure(state=DISABLED)
        self.status_var.set("正在连接两台相机...")

        def worker() -> None:
            try:
                system = StereoCameraSystem(self.config)
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
        self.photo_button.configure(state=NORMAL)
        self.record_button.configure(state=DISABLED)
        self.status_var.set("实时采集中：可同步拍照；录像前请先停止采集。")
        self.preview_thread = threading.Thread(target=self._preview_loop, daemon=True)
        self.preview_thread.start()

    def stop_preview(self) -> None:
        self.previewing = False
        self.preview_button.configure(state=DISABLED)
        self.status_var.set("正在停止实时采集...")

    def _preview_loop(self) -> None:
        assert self.camera_system is not None
        fps = max(float(self.config.get("preview_fps", self.config.get("record_fps", 5.0))), 0.1)
        interval = 1.0 / fps
        next_time = time.perf_counter()
        had_error = False

        try:
            while self.previewing:
                left, right, _trigger_time = self.camera_system.capture_pair()
                self.ui_queue.put(("frames", (left, right)))
                now = time.perf_counter()
                if now - self._last_preview_status_time >= 1.0:
                    self._last_preview_status_time = now
                    trigger_note = "等待 Line0 外触发。" if self.trigger_source_var.get() == "Line0" else "可同步拍照。"
                    self.ui_queue.put(("status", self._status_with_stats(f"实时采集中：目标 {fps:g} fps，{trigger_note}")))

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
            self.ui_queue.put(("preview_done", had_error))

    def capture_photo(self) -> None:
        if self.camera_system is None:
            return
        self.photo_button.configure(state=DISABLED)
        self.interval_button.configure(state=DISABLED)
        if not self.previewing:
            self.preview_button.configure(state=DISABLED)
            self.record_button.configure(state=DISABLED)
        if self.trigger_source_var.get() == "Line0":
            self.status_var.set("正在等待 Line0 外触发帧并保存...")
        else:
            self.status_var.set("正在同步拍照...")

        def worker() -> None:
            try:
                left, right, trigger_time = self.camera_system.capture_pair()
                photo_dir = self._save_photo_pair(left, right, trigger_time, mode="photo")
                self.ui_queue.put(("frames", (left, right)))
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
        if self.previewing:
            self.status_var.set("请先停止实时采集，再启动定时拍照。定时拍照会显示每次保存的画面。")
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
        self.interval_capturing = True
        self.interval_stop_event.clear()
        self.interval_count = 0
        self.interval_button.configure(text="停止定时")
        self.preview_button.configure(state=DISABLED)
        self.photo_button.configure(state=DISABLED)
        self.record_button.configure(state=DISABLED)
        self.status_var.set(f"定时拍照已启动：每 {interval_s:g} 秒保存一组左右图。")
        self.interval_thread = threading.Thread(target=self._interval_capture_loop, args=(interval_s, limit), daemon=True)
        self.interval_thread.start()

    def stop_interval_capture(self) -> None:
        self.interval_capturing = False
        self.interval_stop_event.set()
        self.interval_button.configure(state=DISABLED)
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
        if self.previewing:
            self.status_var.set("请先停止实时采集，再开始录像。")
            return
        if not self._check_disk_space_for_recording():
            return
        self._reset_stats()
        self.recording = True
        self.record_count = 0
        self.record_dir = resolve_output_root(self.config) / "videos" / time.strftime("%Y%m%d_%H%M%S")
        (self.record_dir / "left").mkdir(parents=True, exist_ok=True)
        (self.record_dir / "right").mkdir(parents=True, exist_ok=True)
        self.record_button.configure(text="停止录像")
        self.photo_button.configure(state=DISABLED)
        self.status_var.set(f"正在录像：{self.record_dir}")
        self.record_thread = threading.Thread(target=self._record_loop, daemon=True)
        self.record_thread.start()

    def stop_recording(self) -> None:
        self.recording = False
        self.record_button.configure(state=DISABLED)
        self.status_var.set("正在停止录像并整理文件...")

    def _record_loop(self) -> None:
        assert self.camera_system is not None
        assert self.record_dir is not None
        fps = max(float(self.config.get("record_fps", 5.0)), 0.1)
        interval = 1.0 / fps
        meta_frames = []
        next_time = time.perf_counter()

        try:
            while self.recording:
                loop_start = time.perf_counter()
                left, right, trigger_time = self.camera_system.capture_pair()
                self.record_count += 1
                ext = image_extension(self.config)
                name = f"{self.record_count:06d}.{ext}"
                self._save_image(left.image, self.record_dir / "left" / f"left_{name}")
                self._save_image(right.image, self.record_dir / "right" / f"right_{name}")
                meta_frames.append(
                    {
                        "index": self.record_count,
                        "trigger_time": trigger_time,
                        "left_frame": self._frame_meta(left),
                        "right_frame": self._frame_meta(right),
                    }
                )
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
            meta = {
                "mode": "video",
                "fps": fps,
                "frame_count": self.record_count,
                "image_format": image_extension(self.config),
                "pixel_format": self.config.get("pixel_format", "Mono8"),
                "left_camera": asdict(self.camera_system.left_info) if self.camera_system.left_info else None,
                "right_camera": asdict(self.camera_system.right_info) if self.camera_system.right_info else None,
                "frames": meta_frames,
            }
            with (self.record_dir / "meta.json").open("w", encoding="utf-8") as fh:
                json.dump(meta, fh, ensure_ascii=False, indent=2)
            if self.config.get("auto_make_mp4", True):
                self._try_make_mp4(self.record_dir, fps)
            self.ui_queue.put(("record_done", self.record_dir))

    def _try_make_mp4(self, record_dir: Path, fps: float) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            return
        ext = image_extension(self.config)
        commands = [
            [
                ffmpeg,
                "-y",
                "-framerate",
                f"{fps:g}",
                "-i",
                str(record_dir / "left" / f"left_%06d.{ext}"),
                "-pix_fmt",
                "yuv420p",
                str(record_dir / "left.mp4"),
            ],
            [
                ffmpeg,
                "-y",
                "-framerate",
                f"{fps:g}",
                "-i",
                str(record_dir / "right" / f"right_%06d.{ext}"),
                "-pix_fmt",
                "yuv420p",
                str(record_dir / "right.mp4"),
            ],
        ]
        for command in commands:
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

    def process_ui_queue(self) -> None:
        try:
            while True:
                kind, payload = self.ui_queue.get_nowait()
                if kind == "status":
                    self.status_var.set(str(payload))
                elif kind == "connected":
                    left_info, right_info = payload
                    self.left_pane.set_title(f"左相机：{left_info.label}")
                    self.right_pane.set_title(f"右相机：{right_info.label}")
                    self.status_var.set("相机连接成功。点击开始采集可实时预览；预览中可同步拍照。")
                    self._set_capture_buttons(NORMAL)
                    self._set_parameter_buttons(NORMAL)
                elif kind == "connect_failed":
                    self.connect_button.configure(state=NORMAL)
                elif kind == "frames":
                    left, right = payload
                    self._update_stats(left, right)
                    self.left_pane.set_frame(left)
                    self.right_pane.set_frame(right)
                elif kind == "photo_done":
                    self.status_var.set(f"拍照完成：{payload}")
                elif kind == "capture_idle":
                    if self.camera_system is not None and not self.recording:
                        self._set_capture_buttons(NORMAL)
                elif kind == "interval_done":
                    self.interval_capturing = False
                    self.interval_button.configure(text="定时拍照")
                    if self.camera_system is not None and not self.recording:
                        self._set_capture_buttons(NORMAL)
                    if not payload:
                        self.status_var.set(f"定时拍照已停止，共保存 {self.interval_count} 组。")
                elif kind == "preview_done":
                    self.previewing = False
                    self.preview_button.configure(text="开始采集")
                    if self.camera_system is not None and not self.recording:
                        self._set_capture_buttons(NORMAL)
                        if not payload:
                            self.status_var.set("实时采集已停止。可以同步拍照或开始录像。")
                elif kind == "record_done":
                    self.recording = False
                    self.record_button.configure(text="开始录像")
                    self._set_capture_buttons(NORMAL)
                    self.status_var.set(f"录像完成：{payload}")
                elif kind == "gain_idle":
                    if self.camera_system is not None:
                        self._set_parameter_buttons(NORMAL)
                elif kind == "param_idle":
                    if self.camera_system is not None:
                        self._set_parameter_buttons(NORMAL)
                elif kind == "calibration_done":
                    self.calibrating = False
                    if self.calibrate_button is not None:
                        self.calibrate_button.configure(state=NORMAL)
                    self._set_calibration_progress(100.0, "标定完成")
                    self.render_calibration_result(payload)
                elif kind == "calibration_idle":
                    self.calibrating = False
                    if self.calibrate_button is not None:
                        self.calibrate_button.configure(state=NORMAL)
                    self._set_calibration_progress(0.0, "等待开始")
                elif kind == "calibration_progress":
                    value, text = payload
                    self._set_calibration_progress(float(value), str(text))
                elif kind == "error":
                    self._show_error(payload)
        except Empty:
            pass
        self.root.after(100, self.process_ui_queue)

    def _set_capture_buttons(self, state: str) -> None:
        preview_state = state if self.camera_system is not None else DISABLED
        if self.interval_capturing:
            self.preview_button.configure(state=DISABLED)
            self.photo_button.configure(state=DISABLED)
            self.interval_button.configure(state=state)
            self.record_button.configure(state=DISABLED)
        elif self.previewing:
            self.preview_button.configure(state=preview_state)
            self.photo_button.configure(state=state)
            self.interval_button.configure(state=state)
            self.record_button.configure(state=DISABLED)
        else:
            self.preview_button.configure(state=preview_state)
            self.photo_button.configure(state=state)
            self.interval_button.configure(state=state)
            self.record_button.configure(state=state)
        self.connect_button.configure(state=DISABLED if self.camera_system is not None else NORMAL)

    def _set_parameter_buttons(self, state: str) -> None:
        self.apply_gain_button.configure(state=state)
        self.apply_exposure_button.configure(state=state)
        self.apply_wb_button.configure(state=state)
        self.apply_roi_button.configure(state=state)
        self.apply_trigger_button.configure(state=state)
        if self.calibrate_button is None:
            return
        if self.calibrating:
            self.calibrate_button.configure(state=DISABLED)
        else:
            self.calibrate_button.configure(state=NORMAL)

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
                if warnings:
                    self.ui_queue.put(("status", "增益已应用；" + "；".join(warnings)))
                else:
                    self.ui_queue.put(("status", "增益设置已应用到左右相机。"))
            except Exception as exc:
                self.ui_queue.put(("error", exc))
            finally:
                self.ui_queue.put(("gain_idle", None))

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
            width = optional_int_text(self.roi_width_var.get())
            height = optional_int_text(self.roi_height_var.get())
            offset_x = int(self.roi_offset_x_var.get() or 0)
            offset_y = int(self.roi_offset_y_var.get() or 0)
        except ValueError:
            self.status_var.set("ROI 参数必须是整数。")
            return
        self.apply_roi_button.configure(state=DISABLED)
        self.status_var.set("正在应用 ROI 设置...")

        def worker() -> None:
            try:
                warnings = self.camera_system.apply_roi_settings(width, height, offset_x, offset_y)
                self.config["roi_width"] = width
                self.config["roi_height"] = height
                self.config["roi_offset_x"] = offset_x
                self.config["roi_offset_y"] = offset_y
                save_config(self.config)
                self.ui_queue.put(("status", self._format_apply_result("ROI 已应用", warnings)))
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

    def choose_save_dir(self) -> None:
        selected = filedialog.askdirectory(
            title="选择保存路径",
            initialdir=str(resolve_output_root(self.config).resolve()),
        )
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

    def load_preset(self) -> None:
        presets = self.config.get("presets", {})
        preset = presets.get(self.preset_var.get())
        if not preset:
            self.status_var.set(f"未找到预设：{self.preset_var.get()}")
            return
        self.config.update(preset)
        self._load_vars_from_config()
        save_config(self.config)
        self.status_var.set(f"已加载预设：{self.preset_var.get()}；连接相机后点击应用参数。")

    def save_preset(self) -> None:
        presets = self.config.setdefault("presets", {})
        presets[self.preset_var.get()] = self._current_parameter_config()
        save_config(self.config)
        self.status_var.set(f"已保存预设：{self.preset_var.get()}")

    def choose_calibration_dir(self, variable: StringVar) -> None:
        initial = variable.get().strip() or str(resolve_output_root(self.config))
        selected = filedialog.askdirectory(title="选择标定目录", initialdir=str(resolve_app_path(initial).resolve()))
        if selected:
            variable.set(selected)

    def open_pattern_generator(self) -> None:
        index_path = PATTERN_GENERATOR_DIR / "index.html"
        if not index_path.exists():
            messagebox.showerror("标定板生成器不存在", f"未找到：{index_path}")
            return
        webbrowser.open(index_path.resolve().as_uri())
        self.status_var.set(f"已打开标定板生成器：{index_path}")

    def import_calibration_board_image(self) -> None:
        selected = filedialog.askopenfilename(
            title="导入标定板图片",
            filetypes=[
                ("Image files", "*.bmp *.dib *.jpg *.jpeg *.png *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not selected:
            return
        try:
            from calibration import infer_board_from_image

            square_default = optional_float_text(self.calib_square_size_var.get())
            marker_default = optional_float_text(self.calib_marker_size_var.get())
            info = infer_board_from_image(
                selected,
                default_square_size_mm=square_default,
                default_marker_size_mm=marker_default,
            )
        except Exception as exc:
            self._show_error(exc)
            return

        self.calib_pattern_var.set(str(info["pattern"]))
        self.calib_columns_var.set(str(info["columns"]))
        self.calib_rows_var.set(str(info["rows"]))
        if info.get("square_size_mm") is not None:
            self.calib_square_size_var.set(str(info["square_size_mm"]))
        if info.get("marker_size_mm") is not None:
            self.calib_marker_size_var.set(str(info["marker_size_mm"]))
        if info.get("aruco_dictionary"):
            self.calib_dictionary_var.set(str(info["aruco_dictionary"]))

        summary = (
            f"已导入标定板：{Path(selected).name}；类型 {info['pattern']}，"
            f"列 {info['columns']}，行 {info['rows']}。{info.get('note', '')}"
        )
        self._set_calibration_status(summary)
        self._set_calibration_result_text(json.dumps(info, ensure_ascii=False, indent=2))

    def start_calibration(self) -> None:
        if self.calibrating:
            return
        try:
            columns = int(self.calib_columns_var.get())
            rows = int(self.calib_rows_var.get())
            square_size = float(self.calib_square_size_var.get())
            marker_size = optional_float_text(self.calib_marker_size_var.get())
        except ValueError:
            self.status_var.set("标定参数中的行、列、尺寸必须是数字。")
            return
        if columns <= 0 or rows <= 0 or square_size <= 0:
            self.status_var.set("标定板行列数和格尺寸必须大于 0。")
            return
        pattern = self.calib_pattern_var.get().strip()
        if pattern in {"charuco", "charuco_legacy"} and (marker_size is None or marker_size <= 0):
            self.status_var.set("ChArUco 标定需要填写码尺寸。")
            return
        try:
            from calibration import normalize_aruco_dictionary_name

            aruco_dictionary = normalize_aruco_dictionary_name(self.calib_dictionary_var.get())
            self.calib_dictionary_var.set(aruco_dictionary)
        except Exception:
            aruco_dictionary = self.calib_dictionary_var.get()

        self.calibrating = True
        if self.calibrate_button is not None:
            self.calibrate_button.configure(state=DISABLED)
        self._set_calibration_status("正在标定，请等待...")
        self._set_calibration_result_text("")
        self._set_calibration_progress(0.0, "准备开始")
        self.config.update(
            {
                "calibration_left_dir": self.calib_left_dir_var.get(),
                "calibration_right_dir": self.calib_right_dir_var.get(),
                "calibration_output_dir": self.calib_output_dir_var.get(),
                "calibration_pattern": pattern,
                "calibration_columns": columns,
                "calibration_rows": rows,
                "calibration_square_size_mm": square_size,
                "calibration_marker_size_mm": marker_size,
                "calibration_aruco_dictionary": aruco_dictionary,
            }
        )
        save_config(self.config)

        def worker() -> None:
            try:
                from calibration import calibrate_stereo_from_folders, summarize_result

                result = calibrate_stereo_from_folders(
                    resolve_app_path(self.calib_left_dir_var.get()),
                    resolve_app_path(self.calib_right_dir_var.get()),
                    resolve_app_path(self.calib_output_dir_var.get()),
                    pattern=pattern,
                    columns=columns,
                    rows=rows,
                    square_size_mm=square_size,
                    marker_size_mm=marker_size,
                    aruco_dictionary=aruco_dictionary,
                    legacy_charuco=pattern == "charuco_legacy",
                    progress_callback=lambda value, text: self.ui_queue.put(("calibration_progress", (value, text))),
                )
                result["summary_text"] = summarize_result(result)
                self.ui_queue.put(("calibration_done", result))
            except Exception as exc:
                self.ui_queue.put(("error", exc))
                self.ui_queue.put(("calibration_idle", None))

        threading.Thread(target=worker, daemon=True).start()

    def _current_parameter_config(self) -> dict:
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
            "roi_width": optional_int_text(self.roi_width_var.get()),
            "roi_height": optional_int_text(self.roi_height_var.get()),
            "roi_offset_x": int(self.roi_offset_x_var.get() or 0),
            "roi_offset_y": int(self.roi_offset_y_var.get() or 0),
        }

    def _load_vars_from_config(self) -> None:
        self.trigger_source_var.set(str(self.config.get("trigger_source", "Software")))
        self.exposure_auto_var.set(str(self.config.get("exposure_auto", "Off")))
        self.exposure_time_var.set(str(self.config.get("exposure_time_us", 10000.0)))
        self.auto_exposure_lower_var.set(str(self.config.get("auto_exposure_lower_limit", 100.0)))
        self.auto_exposure_upper_var.set(str(self.config.get("auto_exposure_upper_limit", 100000.0)))
        self.gain_auto_var.set(str(self.config.get("gain_auto", "Off")))
        self.gain_var.set(str(self.config.get("gain", 0.0)))
        self.auto_gain_lower_var.set(str(self.config.get("auto_gain_lower_limit", 0.0)))
        self.auto_gain_upper_var.set(str(self.config.get("auto_gain_upper_limit", 15.0)))
        self.balance_auto_var.set(str(self.config.get("balance_white_auto", "Off")))
        self.balance_red_var.set("" if self.config.get("balance_ratio_red") is None else str(self.config.get("balance_ratio_red")))
        self.balance_green_var.set("" if self.config.get("balance_ratio_green") is None else str(self.config.get("balance_ratio_green")))
        self.balance_blue_var.set("" if self.config.get("balance_ratio_blue") is None else str(self.config.get("balance_ratio_blue")))
        self.roi_width_var.set("" if self.config.get("roi_width") is None else str(self.config.get("roi_width")))
        self.roi_height_var.set("" if self.config.get("roi_height") is None else str(self.config.get("roi_height")))
        self.roi_offset_x_var.set(str(self.config.get("roi_offset_x", 0)))
        self.roi_offset_y_var.set(str(self.config.get("roi_offset_y", 0)))

    def _format_apply_result(self, prefix: str, warnings: list[str]) -> str:
        if warnings:
            return prefix + "；" + "；".join(warnings)
        return prefix + "到左右相机。"

    def _reset_stats(self) -> None:
        self._stat_last_time = time.perf_counter()
        self._stat_frames = 0
        self._actual_fps = 0.0
        self._last_left_frame = None
        self._last_right_frame = None
        self._drop_count = 0

    def _update_stats(self, left: CameraFrame, right: CameraFrame) -> None:
        now = time.perf_counter()
        self._stat_frames += 1
        if self._stat_last_time is not None:
            elapsed = now - self._stat_last_time
            if elapsed >= 1.0:
                self._actual_fps = self._stat_frames / elapsed
                self._stat_frames = 0
                self._stat_last_time = now
        if self._last_left_frame is not None:
            left_step = left.frame_number - self._last_left_frame
            if left_step > 1:
                self._drop_count += left_step - 1
        if self._last_right_frame is not None:
            right_step = right.frame_number - self._last_right_frame
            if right_step > 1:
                self._drop_count += right_step - 1
        self._last_left_frame = left.frame_number
        self._last_right_frame = right.frame_number

    def _status_with_stats(self, prefix: str) -> str:
        if self._last_left_frame is None or self._last_right_frame is None:
            return prefix
        frame_delta = self._last_left_frame - self._last_right_frame
        return f"{prefix} 实际 {self._actual_fps:.1f} fps；丢帧 {self._drop_count}；左右帧差 {frame_delta}"

    def _check_disk_space_for_recording(self) -> bool:
        save_root = resolve_output_root(self.config)
        save_root.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(save_root)
        width = optional_int_text(self.roi_width_var.get()) or 5472
        height = optional_int_text(self.roi_height_var.get()) or 3648
        frame_bytes = estimate_frame_bytes(self.config, width, height)
        pair_bytes = frame_bytes * 2
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
        text = value.get().strip()
        if not text:
            return None
        return float(text)

    def _save_photo_pair(self, left: CameraFrame, right: CameraFrame, trigger_time: float, mode: str = "photo") -> Path:
        capture_id = timestamp_ms()
        photo_root = resolve_output_root(self.config) / "photos"
        group_dir = photo_root / capture_id
        left_dir = photo_root / "left"
        right_dir = photo_root / "right"
        group_dir.mkdir(parents=True, exist_ok=True)
        left_dir.mkdir(parents=True, exist_ok=True)
        right_dir.mkdir(parents=True, exist_ok=True)
        ext = image_extension(self.config)

        group_left = group_dir / f"left.{ext}"
        group_right = group_dir / f"right.{ext}"
        left_path = left_dir / f"{capture_id}_left.{ext}"
        right_path = right_dir / f"{capture_id}_right.{ext}"

        self._save_image(left.image, group_left)
        self._save_image(right.image, group_right)
        self._save_image(left.image, left_path)
        self._save_image(right.image, right_path)
        self._write_meta(
            group_dir / "meta.json",
            mode=mode,
            capture_id=capture_id,
            trigger_time=trigger_time,
            left=left,
            right=right,
            left_path=str(left_path),
            right_path=str(right_path),
            group_left_path=str(group_left),
            group_right_path=str(group_right),
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
        payload["left"] = self._frame_meta(data["left"])
        payload["right"] = self._frame_meta(data["right"])
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
        if self.preview_thread and self.preview_thread.is_alive():
            self.preview_thread.join(timeout=3)
        if self.record_thread and self.record_thread.is_alive():
            self.record_thread.join(timeout=3)
        if self.interval_thread and self.interval_thread.is_alive():
            self.interval_thread.join(timeout=3)
        if self.camera_system is not None:
            try:
                self.camera_system.close()
            except MvsError as exc:
                self.status_var.set(str(exc))
        self.root.destroy()


def main() -> None:
    root = Tk()
    app = StereoCaptureApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

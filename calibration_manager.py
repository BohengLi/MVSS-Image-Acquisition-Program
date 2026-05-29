from __future__ import annotations

import json
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw


LOGGER = logging.getLogger("mvss_capture")

MATRIX_KEYS = (
    "K",
    "K1",
    "K2",
    "D",
    "D1",
    "D2",
    "R",
    "T",
    "E",
    "F",
    "R1",
    "R2",
    "P1",
    "P2",
    "Q",
    "M",
    "M1",
    "M2",
    "camera_matrix",
    "distortion_coefficients",
    "dist_coeffs",
)
MAX_RECTIFICATION_MAP_CACHE = 3


def _resolve_path(base_dir: Path, value: object) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    return path if path.is_absolute() else base_dir / path


def _to_serializable_matrix(value: np.ndarray | None) -> list | None:
    if value is None:
        return None
    return np.asarray(value, dtype=float).tolist()


def _matrix_from_json(value: object) -> np.ndarray | None:
    if value in (None, ""):
        return None
    if isinstance(value, dict):
        data = value.get("data", value.get("values"))
        rows = value.get("rows")
        cols = value.get("cols")
        if data is not None:
            arr = np.asarray(data, dtype=float)
            if rows and cols and arr.size == int(rows) * int(cols):
                return arr.reshape((int(rows), int(cols)))
            return arr
    try:
        return np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None


def _read_cv_node_matrix(fs: cv2.FileStorage, key: str) -> np.ndarray | None:
    node = fs.getNode(key)
    if node.empty():
        return None
    try:
        mat = node.mat()
        if mat is not None:
            return np.asarray(mat, dtype=float)
    except Exception as exc:
        LOGGER.debug("Failed to read calibration matrix node %s via mat(): %s", key, exc, exc_info=True)
    try:
        if node.isSeq():
            values = [node.at(i).real() for i in range(node.size())]
            return np.asarray(values, dtype=float)
    except Exception as exc:
        LOGGER.debug("Failed to read calibration matrix node %s via sequence: %s", key, exc, exc_info=True)
    try:
        return np.asarray([node.real()], dtype=float)
    except Exception as exc:
        LOGGER.debug("Failed to read calibration matrix node %s via scalar fallback: %s", key, exc, exc_info=True)
        return None


def _read_cv_node_scalar(fs: cv2.FileStorage, key: str) -> float | None:
    node = fs.getNode(key)
    if node.empty():
        return None
    try:
        return float(node.real())
    except Exception as exc:
        LOGGER.debug("Failed to read calibration scalar node %s: %s", key, exc, exc_info=True)
        return None


def _load_calibration_file(path: Path) -> dict[str, np.ndarray | float | str]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        result: dict[str, np.ndarray | float | str] = {}
        if isinstance(data, dict):
            for key in MATRIX_KEYS:
                if key in data:
                    matrix = _matrix_from_json(data[key])
                    if matrix is not None:
                        result[key] = matrix
            for key in ("image_width", "image_height", "width", "height"):
                if key in data:
                    try:
                        result[key] = float(data[key])
                    except (TypeError, ValueError):
                        pass
            if "image_size" in data:
                matrix = _matrix_from_json(data["image_size"])
                if matrix is not None:
                    result["image_size"] = matrix
        return result

    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise ValueError(f"unable to open calibration file: {path}")
    try:
        result = {}
        for key in MATRIX_KEYS:
            value = _read_cv_node_matrix(fs, key)
            if value is not None:
                result[key] = value
        for key in ("image_width", "image_height", "width", "height"):
            value = _read_cv_node_scalar(fs, key)
            if value is not None:
                result[key] = value
        value = _read_cv_node_matrix(fs, "image_size")
        if value is not None:
            result["image_size"] = value
        return result
    finally:
        fs.release()


def _first_matrix(data: dict[str, Any], keys: tuple[str, ...]) -> np.ndarray | None:
    for key in keys:
        value = data.get(key)
        if value is not None:
            arr = np.asarray(value, dtype=float)
            if arr.size:
                return arr
    return None


def _coalesce_matrix(*values: np.ndarray | None) -> np.ndarray | None:
    for value in values:
        if value is not None:
            return value
    return None


def _normalize_distortion(value: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value, dtype=float).reshape(-1, 1)


def _normalize_vector(value: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    return arr.reshape(3, 1) if arr.size == 3 else arr


def _image_size_from_data(*items: dict[str, Any]) -> tuple[int, int] | None:
    for data in items:
        size = data.get("image_size")
        if size is not None:
            arr = np.asarray(size, dtype=float).reshape(-1)
            if arr.size >= 2:
                return int(arr[0]), int(arr[1])
        width = data.get("image_width", data.get("width"))
        height = data.get("image_height", data.get("height"))
        if width and height:
            return int(float(width)), int(float(height))
    return None


@dataclass
class StereoCalibration:
    source_files: dict[str, str] = field(default_factory=dict)
    K1: np.ndarray | None = None
    D1: np.ndarray | None = None
    K2: np.ndarray | None = None
    D2: np.ndarray | None = None
    R: np.ndarray | None = None
    T: np.ndarray | None = None
    R1: np.ndarray | None = None
    R2: np.ndarray | None = None
    P1: np.ndarray | None = None
    P2: np.ndarray | None = None
    Q: np.ndarray | None = None
    image_size: tuple[int, int] | None = None
    enabled: bool = True
    warnings: list[str] = field(default_factory=list)
    _map_cache: OrderedDict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = field(
        default_factory=OrderedDict, repr=False
    )

    @property
    def intrinsics_loaded(self) -> bool:
        return self.K1 is not None and self.D1 is not None and self.K2 is not None and self.D2 is not None

    @property
    def stereo_loaded(self) -> bool:
        return self.R is not None and self.T is not None

    @property
    def rectification_ready(self) -> bool:
        return self.enabled and self.intrinsics_loaded and self.stereo_loaded

    def status_text(self) -> str:
        if not self.enabled:
            return "Calibration: disabled"
        if self.rectification_ready:
            return "Calibration: loaded, rectification ready"
        if self.intrinsics_loaded:
            return "Calibration: intrinsics loaded, missing stereo R/T"
        if self.warnings:
            return "Calibration: not loaded (" + "; ".join(self.warnings[:2]) + ")"
        return "Calibration: not loaded"

    def meta(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "source_files": dict(self.source_files),
            "intrinsics_loaded": self.intrinsics_loaded,
            "stereo_loaded": self.stereo_loaded,
            "rectification_ready": self.rectification_ready,
            "image_size": list(self.image_size) if self.image_size else None,
            "K1": _to_serializable_matrix(self.K1),
            "D1": _to_serializable_matrix(self.D1),
            "K2": _to_serializable_matrix(self.K2),
            "D2": _to_serializable_matrix(self.D2),
            "R": _to_serializable_matrix(self.R),
            "T": _to_serializable_matrix(self.T),
            "R1": _to_serializable_matrix(self.R1),
            "R2": _to_serializable_matrix(self.R2),
            "P1": _to_serializable_matrix(self.P1),
            "P2": _to_serializable_matrix(self.P2),
            "Q": _to_serializable_matrix(self.Q),
            "warnings": list(self.warnings),
        }

    def _ensure_maps(self, image_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        if not self.rectification_ready:
            return None
        if image_size in self._map_cache:
            maps = self._map_cache[image_size]
            self._map_cache.move_to_end(image_size)
            return maps
        width, height = image_size
        K1 = np.asarray(self.K1, dtype=float)
        D1 = _normalize_distortion(self.D1)
        K2 = np.asarray(self.K2, dtype=float)
        D2 = _normalize_distortion(self.D2)
        R = np.asarray(self.R, dtype=float)
        T = _normalize_vector(self.T)
        if self.R1 is None or self.R2 is None or self.P1 is None or self.P2 is None:
            R1, R2, P1, P2, Q, _roi1, _roi2 = cv2.stereoRectify(
                K1,
                D1,
                K2,
                D2,
                (width, height),
                R,
                T,
                flags=cv2.CALIB_ZERO_DISPARITY,
                alpha=0,
            )
            self.R1, self.R2, self.P1, self.P2, self.Q = R1, R2, P1, P2, Q
        map1_left, map2_left = cv2.initUndistortRectifyMap(
            K1, D1, self.R1, self.P1, (width, height), cv2.CV_16SC2
        )
        map1_right, map2_right = cv2.initUndistortRectifyMap(
            K2, D2, self.R2, self.P2, (width, height), cv2.CV_16SC2
        )
        maps = (map1_left, map2_left, map1_right, map2_right)
        self._map_cache[image_size] = maps
        self._map_cache.move_to_end(image_size)
        while len(self._map_cache) > MAX_RECTIFICATION_MAP_CACHE:
            self._map_cache.popitem(last=False)
        return maps

    def rectify_pair(self, left: Image.Image, right: Image.Image) -> tuple[Image.Image, Image.Image] | None:
        image_size = left.size
        maps = self._ensure_maps(image_size)
        if maps is None:
            return None
        map1_left, map2_left, map1_right, map2_right = maps
        left_mode = "L" if left.mode == "L" else "RGB"
        right_mode = "L" if right.mode == "L" else "RGB"
        left_arr = np.asarray(left.convert(left_mode))
        right_arr = np.asarray(right.convert(right_mode).resize(image_size, Image.Resampling.BILINEAR))
        rect_left = cv2.remap(left_arr, map1_left, map2_left, cv2.INTER_LINEAR)
        rect_right = cv2.remap(right_arr, map1_right, map2_right, cv2.INTER_LINEAR)
        return Image.fromarray(rect_left, left_mode), Image.fromarray(rect_right, right_mode)

    def make_rectified_overlay(
        self,
        left: Image.Image,
        right: Image.Image,
        alpha: float = 0.50,
        line_interval_px: int = 120,
    ) -> Image.Image | None:
        rectified = self.rectify_pair(left, right)
        if rectified is None:
            return None
        rect_left, rect_right = rectified
        left_rgb = rect_left.convert("RGB")
        right_rgb = rect_right.convert("RGB").resize(left_rgb.size, Image.Resampling.BILINEAR)
        overlay = Image.blend(left_rgb, right_rgb, min(max(alpha, 0.0), 1.0))
        draw = ImageDraw.Draw(overlay)
        width, height = overlay.size
        interval = max(int(line_interval_px), 20)
        for y in range(interval, height, interval):
            draw.line((0, y, width, y), fill=(255, 230, 70), width=1)
        draw.line((0, height // 2, width, height // 2), fill=(70, 220, 255), width=2)
        return overlay


def load_stereo_calibration(config: dict[str, Any], base_dir: Path) -> StereoCalibration:
    settings = config.get("calibration", {})
    enabled = bool(settings.get("enabled", True)) if isinstance(settings, dict) else False
    calibration = StereoCalibration(enabled=enabled)
    if not enabled or not isinstance(settings, dict):
        return calibration

    paths = {
        "left_intrinsics": _resolve_path(base_dir, settings.get("left_intrinsics")),
        "right_intrinsics": _resolve_path(base_dir, settings.get("right_intrinsics")),
        "stereo_params": _resolve_path(base_dir, settings.get("stereo_params")),
    }
    loaded: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        if path is None:
            continue
        calibration.source_files[name] = str(path)
        if not path.exists():
            calibration.warnings.append(f"{name} missing: {path}")
            continue
        try:
            loaded[name] = _load_calibration_file(path)
        except Exception as exc:
            calibration.warnings.append(f"{name} load failed: {exc}")
            LOGGER.warning("Calibration file %s failed to load from %s: %s", name, path, exc, exc_info=True)

    left = loaded.get("left_intrinsics", {})
    right = loaded.get("right_intrinsics", {})
    stereo = loaded.get("stereo_params", {})

    calibration.K1 = _coalesce_matrix(
        _first_matrix(stereo, ("K1", "M1")),
        _first_matrix(left, ("K", "camera_matrix", "M", "K1", "M1")),
    )
    calibration.D1 = _normalize_distortion(
        _coalesce_matrix(
            _first_matrix(stereo, ("D1",)),
            _first_matrix(left, ("D", "distortion_coefficients", "dist_coeffs", "D1")),
        )
    )
    calibration.K2 = _coalesce_matrix(
        _first_matrix(stereo, ("K2", "M2")),
        _first_matrix(right, ("K", "camera_matrix", "M", "K2", "M2")),
    )
    calibration.D2 = _normalize_distortion(
        _coalesce_matrix(
            _first_matrix(stereo, ("D2",)),
            _first_matrix(right, ("D", "distortion_coefficients", "dist_coeffs", "D2")),
        )
    )
    calibration.R = _first_matrix(stereo, ("R",))
    calibration.T = _normalize_vector(_first_matrix(stereo, ("T",)))
    calibration.R1 = _first_matrix(stereo, ("R1",))
    calibration.R2 = _first_matrix(stereo, ("R2",))
    calibration.P1 = _first_matrix(stereo, ("P1",))
    calibration.P2 = _first_matrix(stereo, ("P2",))
    calibration.Q = _first_matrix(stereo, ("Q",))
    calibration.image_size = _image_size_from_data(stereo, left, right)

    if not calibration.intrinsics_loaded:
        calibration.warnings.append("K1/D1/K2/D2 are incomplete")
    if calibration.intrinsics_loaded and not calibration.stereo_loaded:
        calibration.warnings.append("stereo R/T are incomplete")
    return calibration

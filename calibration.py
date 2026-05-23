from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np


IMAGE_SUFFIXES = {".bmp", ".dib", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
ARUCO_DICT_NAMES = (
    "DICT_4X4_50",
    "DICT_4X4_100",
    "DICT_4X4_250",
    "DICT_4X4_1000",
    "DICT_5X5_50",
    "DICT_5X5_100",
    "DICT_5X5_250",
    "DICT_5X5_1000",
    "DICT_6X6_50",
    "DICT_6X6_100",
    "DICT_6X6_250",
    "DICT_6X6_1000",
    "DICT_7X7_50",
    "DICT_7X7_100",
    "DICT_7X7_250",
    "DICT_7X7_1000",
    "DICT_ARUCO_ORIGINAL",
    "DICT_APRILTAG_16h5",
    "DICT_APRILTAG_25h9",
    "DICT_APRILTAG_36h10",
    "DICT_APRILTAG_36h11",
)


class CalibrationError(RuntimeError):
    pass


class CREStereoFallbackRequired(CalibrationError):
    def __init__(
        self,
        reason: str,
        *,
        stage: str = "inference",
        right_disparity_policy: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.stage = stage
        self.right_disparity_policy = right_disparity_policy or {}


BASE_DIR = Path(__file__).resolve().parent
SAM3_MASK_SCRIPT = BASE_DIR / "sam3_mask_inference.py"
_CRESTEREO_MODEL_CACHE: dict[tuple[str, tuple[str, ...] | None], Any] = {}


def _load_cv2():
    os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")
    try:
        import cv2
    except Exception as exc:
        raise CalibrationError(
            "未安装 OpenCV 标定依赖。请执行 `python -m pip install -r requirements.txt`，"
            "其中需要 opencv-contrib-python 才能使用 ChArUco/ArUco 标定。"
        ) from exc
    try:
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass
    return cv2


def _read_gray(cv2, path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if image is None:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise CalibrationError(f"无法读取图像：{path}")
    return image


def _list_images(directory: Path) -> list[Path]:
    if not directory.exists():
        raise CalibrationError(f"图像目录不存在：{directory}")
    if not directory.is_dir():
        raise CalibrationError(f"不是图像目录：{directory}")
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file())


def _pair_key(path: Path, side: str) -> str:
    stem = path.stem.lower()
    stem = re.sub(rf"(^|[_\-\s]){side}([_\-\s]|$)", "_", stem)
    stem = re.sub(r"[_\-\s]+", "_", stem).strip("_")
    return stem or path.stem.lower()


def find_stereo_pairs(left_dir: str | Path, right_dir: str | Path) -> list[tuple[str, Path, Path]]:
    left_images = _list_images(Path(left_dir))
    right_images = _list_images(Path(right_dir))
    left_by_key = {_pair_key(path, "left"): path for path in left_images}
    right_by_key = {_pair_key(path, "right"): path for path in right_images}
    keys = sorted(set(left_by_key) & set(right_by_key))
    return [(key, left_by_key[key], right_by_key[key]) for key in keys]


def _grid_object_points(pattern: str, columns: int, rows: int, spacing: float) -> np.ndarray:
    if columns <= 0 or rows <= 0:
        raise CalibrationError("标定板列数和行数必须大于 0。")
    if spacing <= 0:
        raise CalibrationError("标定板尺寸必须大于 0。")

    if pattern == "acircles":
        points = []
        for row in range(rows):
            for column in range(columns):
                points.append([(2 * column + row % 2) * spacing, row * spacing, 0.0])
        return np.asarray(points, dtype=np.float32)

    objp = np.zeros((columns * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    objp *= float(spacing)
    return objp


def _detect_chessboard(cv2, gray, columns: int, rows: int) -> np.ndarray | None:
    pattern_size = (columns, rows)
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE
    if hasattr(cv2, "CALIB_CB_EXHAUSTIVE"):
        flags |= cv2.CALIB_CB_EXHAUSTIVE
    if hasattr(cv2, "CALIB_CB_ACCURACY"):
        flags |= cv2.CALIB_CB_ACCURACY

    if hasattr(cv2, "findChessboardCornersSB"):
        ok, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags)
        if ok:
            return corners.reshape(-1, 2).astype(np.float32)

    classic_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, pattern_size, classic_flags)
    if not ok:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return corners.reshape(-1, 2).astype(np.float32)


def _circle_blob_detector(cv2, image_shape: tuple[int, int]):
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 0
    params.filterByArea = True
    params.minArea = 10
    params.maxArea = max(100, image_shape[0] * image_shape[1] / 4)
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    return cv2.SimpleBlobDetector_create(params)


def _detect_circles(cv2, gray, columns: int, rows: int, asymmetric: bool) -> np.ndarray | None:
    flags = cv2.CALIB_CB_ASYMMETRIC_GRID if asymmetric else cv2.CALIB_CB_SYMMETRIC_GRID
    detector = _circle_blob_detector(cv2, gray.shape)
    try:
        ok, centers = cv2.findCirclesGrid(gray, (columns, rows), flags, blobDetector=detector)
    except TypeError:
        ok, centers = cv2.findCirclesGrid(gray, (columns, rows), flags)
    if not ok:
        return None
    return centers.reshape(-1, 2).astype(np.float32)


def _aruco_dictionary(cv2, name: str):
    if not hasattr(cv2, "aruco"):
        raise CalibrationError("当前 OpenCV 不包含 aruco 模块，请安装 opencv-contrib-python。")
    aruco = cv2.aruco
    dict_name = normalize_aruco_dictionary_name(name)
    if not hasattr(aruco, dict_name):
        valid_names = ", ".join(ARUCO_DICT_NAMES)
        raise CalibrationError(f"OpenCV 不支持 ArUco 字典：{dict_name}。可用字典：{valid_names}")
    dict_id = getattr(aruco, dict_name)
    if hasattr(aruco, "getPredefinedDictionary"):
        return aruco.getPredefinedDictionary(dict_id)
    return aruco.Dictionary_get(dict_id)


def _create_charuco_board(
    cv2,
    columns: int,
    rows: int,
    square_size: float,
    marker_size: float,
    dictionary_name: str,
    legacy: bool,
):
    if marker_size <= 0:
        raise CalibrationError("ChArUco 标记尺寸必须大于 0。")
    if marker_size >= square_size:
        raise CalibrationError("ChArUco 标记尺寸必须小于方格尺寸。")

    aruco = cv2.aruco
    dictionary = _aruco_dictionary(cv2, dictionary_name)
    if hasattr(aruco, "CharucoBoard_create"):
        board = aruco.CharucoBoard_create(columns, rows, square_size, marker_size, dictionary)
    else:
        board = aruco.CharucoBoard((columns, rows), square_size, marker_size, dictionary)
    if legacy and hasattr(board, "setLegacyPattern"):
        board.setLegacyPattern(True)
    return board, dictionary


def _charuco_board_corners(board) -> np.ndarray:
    if hasattr(board, "getChessboardCorners"):
        corners = board.getChessboardCorners()
    elif hasattr(board, "chessboardCorners"):
        corners = board.chessboardCorners
    else:
        raise CalibrationError("无法从当前 OpenCV ChArUcoBoard 读取角点坐标。")
    return np.asarray(corners, dtype=np.float32).reshape(-1, 3)


def _detect_charuco(cv2, gray, board, dictionary) -> tuple[np.ndarray, np.ndarray] | None:
    aruco = cv2.aruco
    if hasattr(aruco, "DetectorParameters"):
        parameters = aruco.DetectorParameters()
    else:
        parameters = aruco.DetectorParameters_create()

    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(dictionary, parameters)
        marker_corners, marker_ids, rejected = detector.detectMarkers(gray)
    else:
        marker_corners, marker_ids, rejected = aruco.detectMarkers(gray, dictionary, parameters=parameters)

    if marker_ids is None or len(marker_ids) == 0:
        return None

    try:
        aruco.refineDetectedMarkers(gray, board, marker_corners, marker_ids, rejected)
    except Exception:
        pass

    ok, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(marker_corners, marker_ids, gray, board)
    if charuco_ids is None or charuco_corners is None or int(ok) < 4:
        return None
    return charuco_ids.reshape(-1).astype(np.int32), charuco_corners.reshape(-1, 2).astype(np.float32)


def infer_board_from_image(
    image_path: str | Path,
    *,
    max_squares_x: int = 18,
    max_squares_y: int = 14,
    default_square_size_mm: float | None = None,
    default_marker_size_mm: float | None = None,
) -> dict[str, Any]:
    path = Path(image_path)
    result = _infer_board_from_filename(path.name)
    if result is None:
        cv2 = _load_cv2()
        gray = _read_gray(cv2, path)
        result = _infer_board_from_pixels(cv2, gray, max_squares_x, max_squares_y)
    if result is None:
        raise CalibrationError("无法从该标定板图片识别规格。请换用清晰的标定板原图，或手动填写行列数。")
    if default_square_size_mm is not None and result.get("square_size_mm") is None:
        result["square_size_mm"] = float(default_square_size_mm)
    if default_marker_size_mm is not None and result.get("marker_size_mm") is None:
        result["marker_size_mm"] = float(default_marker_size_mm)
    result["source_image"] = str(path)
    return result


def _infer_board_from_filename(filename: str) -> dict[str, Any] | None:
    name = Path(filename).stem
    lower = name.lower()
    if lower.startswith("chessboard_"):
        match = re.search(r"chessboard_(\d+)x(\d+)_([0-9.]+)mm", lower)
        if match:
            rows_squares, columns_squares, square_size = match.groups()
            rows_squares_i = int(rows_squares)
            columns_squares_i = int(columns_squares)
            return {
                "pattern": "chessboard",
                "columns": max(columns_squares_i - 1, 1),
                "rows": max(rows_squares_i - 1, 1),
                "square_size_mm": float(square_size),
                "marker_size_mm": None,
                "aruco_dictionary": None,
                "confidence": "filename",
                "note": "从标定板生成器文件名解析。棋盘格标定使用内角点数量。",
            }
    if lower.startswith("charuco_"):
        match = re.search(r"charuco_(\d+)x(\d+)_([0-9.]+)mm_([0-9.]+)mm_(.+)$", name, re.IGNORECASE)
        if match:
            columns, rows, square_size, marker_size, dictionary = match.groups()
            return {
                "pattern": "charuco",
                "columns": int(columns),
                "rows": int(rows),
                "square_size_mm": float(square_size),
                "marker_size_mm": float(marker_size),
                "aruco_dictionary": _normalize_dictionary_name(dictionary),
                "confidence": "filename",
                "note": "从标定板生成器文件名解析。ChArUco 标定使用方格数量。",
            }
    if lower.startswith("circles_"):
        match = re.search(r"circles_(\d+)x(\d+)_([0-9.]+)mm_([0-9.]+)mm", lower)
        if match:
            rows, columns, _diameter, spacing = match.groups()
            return {
                "pattern": "circles",
                "columns": int(columns),
                "rows": int(rows),
                "square_size_mm": float(spacing),
                "marker_size_mm": None,
                "aruco_dictionary": None,
                "confidence": "filename",
                "note": "从标定板生成器文件名解析。圆点阵尺寸使用圆心间距。",
            }
    return None


def _normalize_dictionary_name(value: str) -> str:
    return normalize_aruco_dictionary_name(value)


def normalize_aruco_dictionary_name(value: str) -> str:
    name = str(value).strip()
    if not name:
        return "DICT_4X4_50"
    upper = name.upper()
    if not upper.startswith("DICT_"):
        upper = "DICT_" + upper

    canonical_by_upper = {candidate.upper(): candidate for candidate in ARUCO_DICT_NAMES}
    if upper in canonical_by_upper:
        return canonical_by_upper[upper]

    # Some generated or copied filenames append a serial suffix, e.g.
    # DICT_5X5_1000_01. Keep only the OpenCV dictionary prefix.
    patterns = (
        r"DICT_[4-7]X[4-7]_(?:1000|250|100|50)",
        r"DICT_APRILTAG_(?:16H5|25H9|36H10|36H11)",
        r"DICT_ARUCO_ORIGINAL",
    )
    for pattern in patterns:
        match = re.search(pattern, upper)
        if match:
            matched = match.group(0)
            return canonical_by_upper.get(matched, matched)

    return upper


def _infer_board_from_pixels(cv2, gray, max_squares_x: int, max_squares_y: int) -> dict[str, Any] | None:
    charuco = _infer_charuco_from_pixels(cv2, gray, max_squares_x, max_squares_y)
    if charuco is not None:
        return charuco
    chessboard = _infer_chessboard_from_pixels(cv2, gray, max_squares_x, max_squares_y)
    if chessboard is not None:
        return chessboard
    circles = _infer_circles_from_pixels(cv2, gray, max_squares_x, max_squares_y)
    if circles is not None:
        return circles
    return None


def _infer_charuco_from_pixels(cv2, gray, max_squares_x: int, max_squares_y: int) -> dict[str, Any] | None:
    if not hasattr(cv2, "aruco"):
        return None
    aruco = cv2.aruco
    best: dict[str, Any] | None = None
    for dict_name in ARUCO_DICT_NAMES:
        if not hasattr(aruco, dict_name):
            continue
        try:
            dictionary = _aruco_dictionary(cv2, dict_name)
            if hasattr(aruco, "DetectorParameters"):
                parameters = aruco.DetectorParameters()
            else:
                parameters = aruco.DetectorParameters_create()
            if hasattr(aruco, "ArucoDetector"):
                detector = aruco.ArucoDetector(dictionary, parameters)
                marker_corners, marker_ids, _rejected = detector.detectMarkers(gray)
            else:
                marker_corners, marker_ids, _rejected = aruco.detectMarkers(gray, dictionary, parameters=parameters)
        except Exception:
            continue
        if marker_ids is None or len(marker_ids) == 0:
            continue
        points = np.concatenate([corner.reshape(-1, 2) for corner in marker_corners], axis=0)
        columns, rows = _estimate_grid_from_points(points, max_squares_x, max_squares_y)
        if columns <= 0 or rows <= 0:
            continue
        candidate = {
            "pattern": "charuco",
            "columns": int(columns),
            "rows": int(rows),
            "square_size_mm": None,
            "marker_size_mm": None,
            "aruco_dictionary": dict_name,
            "confidence": "image",
            "detected_markers": int(len(marker_ids)),
            "note": "从 ArUco 标记外接网格估计 ChArUco 方格数；请核对格尺寸和码尺寸。",
        }
        if best is None or candidate["detected_markers"] > best.get("detected_markers", 0):
            best = candidate
    return best


def _estimate_grid_from_points(points: np.ndarray, max_x: int, max_y: int) -> tuple[int, int]:
    if len(points) == 0:
        return 0, 0
    x_values = np.sort(points[:, 0])
    y_values = np.sort(points[:, 1])
    x_unique = _count_clusters(x_values, max(2.0, float(np.ptp(x_values)) / max(max_x * 3, 1)))
    y_unique = _count_clusters(y_values, max(2.0, float(np.ptp(y_values)) / max(max_y * 3, 1)))
    columns = max(1, min(max_x, x_unique - 1))
    rows = max(1, min(max_y, y_unique - 1))
    return columns, rows


def _count_clusters(values: np.ndarray, threshold: float) -> int:
    if len(values) == 0:
        return 0
    clusters = 1
    current = float(values[0])
    for value in values[1:]:
        if abs(float(value) - current) > threshold:
            clusters += 1
            current = float(value)
        else:
            current = (current + float(value)) / 2.0
    return clusters


def _infer_chessboard_from_pixels(cv2, gray, max_squares_x: int, max_squares_y: int) -> dict[str, Any] | None:
    best: tuple[int, float, int, int] | None = None
    for columns in range(3, max_squares_x):
        for rows in range(3, max_squares_y):
            corners = _detect_chessboard(cv2, gray, columns, rows)
            if corners is None:
                continue
            score = columns * rows
            aspect_error = _grid_aspect_error(corners, columns, rows)
            if best is None or score > best[0] or (score == best[0] and aspect_error < best[1]):
                best = (score, aspect_error, columns, rows)
    if best is None:
        return None
    _score, _aspect_error, columns, rows = best
    return {
        "pattern": "chessboard",
        "columns": int(columns),
        "rows": int(rows),
        "square_size_mm": None,
        "marker_size_mm": None,
        "aruco_dictionary": None,
        "confidence": "image",
        "note": "从图片识别棋盘格内角点数量；格尺寸仍需按实际标定板填写。",
    }


def _grid_aspect_error(points: np.ndarray, columns: int, rows: int) -> float:
    pts = points.reshape(-1, 2)
    width = max(float(np.ptp(pts[:, 0])), 1.0)
    height = max(float(np.ptp(pts[:, 1])), 1.0)
    image_aspect = width / height
    grid_aspect = max(float(columns - 1), 1.0) / max(float(rows - 1), 1.0)
    return abs(math.log(max(image_aspect, 1e-6) / max(grid_aspect, 1e-6)))


def _infer_circles_from_pixels(cv2, gray, max_squares_x: int, max_squares_y: int) -> dict[str, Any] | None:
    for pattern, asymmetric in (("acircles", True), ("circles", False)):
        best: tuple[int, int, int] | None = None
        for columns in range(3, max_squares_x + 1):
            for rows in range(3, max_squares_y + 1):
                centers = _detect_circles(cv2, gray, columns, rows, asymmetric)
                if centers is None:
                    continue
                score = columns * rows
                if best is None or score > best[0]:
                    best = (score, columns, rows)
        if best is not None:
            _score, columns, rows = best
            return {
                "pattern": pattern,
                "columns": int(columns),
                "rows": int(rows),
                "square_size_mm": None,
                "marker_size_mm": None,
                "aruco_dictionary": None,
                "confidence": "image",
                "note": "从图片识别圆点阵规格；格mm请填写圆心间距。",
            }
    return None


def _mono_error(cv2, object_points, image_points, rvecs, tvecs, camera_matrix, dist_coeffs) -> tuple[float, list[float]]:
    total_sq_error = 0.0
    total_points = 0
    per_view = []
    for obj, img, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, camera_matrix, dist_coeffs)
        error = cv2.norm(img, projected, cv2.NORM_L2)
        count = len(obj)
        per_view.append(float(error / math.sqrt(count)))
        total_sq_error += float(error * error)
        total_points += count
    return float(math.sqrt(total_sq_error / max(total_points, 1))), per_view


def _projected_points(cv2, object_points, rvecs, tvecs, camera_matrix, dist_coeffs) -> list[np.ndarray]:
    projections = []
    for obj, rvec, tvec in zip(object_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, camera_matrix, dist_coeffs)
        projections.append(projected.reshape(-1, 2).astype(np.float32))
    return projections


def _solve_view_poses(cv2, object_points, image_points, camera_matrix, dist_coeffs) -> tuple[list[np.ndarray], list[np.ndarray]]:
    rvecs: list[np.ndarray] = []
    tvecs: list[np.ndarray] = []
    for obj, img in zip(object_points, image_points):
        ok, rvec, tvec = cv2.solvePnP(
            obj.reshape(-1, 3).astype(np.float32),
            img.reshape(-1, 2).astype(np.float32),
            camera_matrix,
            dist_coeffs,
        )
        if not ok:
            rvec = np.zeros((3, 1), dtype=np.float64)
            tvec = np.zeros((3, 1), dtype=np.float64)
        rvecs.append(rvec)
        tvecs.append(tvec)
    return rvecs, tvecs


def _array(value: Any) -> list:
    return np.asarray(value).tolist()


def _matlab_like_parameters(camera_matrix, dist_coeffs) -> dict[str, Any]:
    matrix = np.asarray(camera_matrix, dtype=float)
    dist = np.asarray(dist_coeffs, dtype=float).reshape(-1)
    radial = [float(dist[0])] if len(dist) > 0 else []
    if len(dist) > 1:
        radial.append(float(dist[1]))
    if len(dist) > 4:
        radial.append(float(dist[4]))
    tangential = [float(dist[2]), float(dist[3])] if len(dist) > 3 else []
    return {
        "focal_length_px": [float(matrix[0, 0]), float(matrix[1, 1])],
        "principal_point_px": [float(matrix[0, 2]), float(matrix[1, 2])],
        "skew": float(matrix[0, 1]),
        "radial_distortion": radial,
        "tangential_distortion": tangential,
        "camera_matrix_opencv": _array(matrix),
        "distortion_coefficients_opencv": _array(dist),
    }


def _write_opencv_yaml(cv2, path: Path, result: dict[str, Any]) -> None:
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    if not storage.isOpened():
        return
    try:
        storage.write("image_width", int(result["image_size"][0]))
        storage.write("image_height", int(result["image_size"][1]))
        storage.write("left_camera_matrix", np.asarray(result["left"]["camera_matrix"], dtype=np.float64))
        storage.write("left_distortion", np.asarray(result["left"]["distortion_coefficients"], dtype=np.float64))
        storage.write("right_camera_matrix", np.asarray(result["right"]["camera_matrix"], dtype=np.float64))
        storage.write("right_distortion", np.asarray(result["right"]["distortion_coefficients"], dtype=np.float64))
        storage.write("rotation_matrix", np.asarray(result["stereo"]["rotation_matrix"], dtype=np.float64))
        storage.write("translation_vector", np.asarray(result["stereo"]["translation_vector"], dtype=np.float64))
        storage.write("essential_matrix", np.asarray(result["stereo"]["essential_matrix"], dtype=np.float64))
        storage.write("fundamental_matrix", np.asarray(result["stereo"]["fundamental_matrix"], dtype=np.float64))
        if "rectification" in result["stereo"]:
            rectification = result["stereo"]["rectification"]
            storage.write("R1", np.asarray(rectification["R1"], dtype=np.float64))
            storage.write("R2", np.asarray(rectification["R2"], dtype=np.float64))
            storage.write("P1", np.asarray(rectification["P1"], dtype=np.float64))
            storage.write("P2", np.asarray(rectification["P2"], dtype=np.float64))
            storage.write("Q", np.asarray(rectification["Q"], dtype=np.float64))
    finally:
        storage.release()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _invoke_progress(progress_callback: Callable[[float, str], None] | None, value: float, message: str) -> None:
    if progress_callback is None:
        return
    progress_callback(float(max(0.0, min(100.0, value))), str(message))


def _read_color(cv2, path: str | Path):
    image_path = Path(path)
    data = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise CalibrationError(f"无法读取图像：{image_path}")
    return image


def _write_image(cv2, path: str | Path, image, quality: int = 92) -> None:
    image_path = Path(path)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = image_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(max(90, quality))]
    elif suffix == ".png":
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), 0]
    else:
        params = []
    ok, encoded = cv2.imencode(image_path.suffix or ".png", image, params)
    if not ok:
        raise CalibrationError(f"无法编码图像：{image_path}")
    encoded.tofile(str(image_path))


def _safe_name(text: Any) -> str:
    name = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(text).strip())
    return name.strip("._") or "pair"


def _link_or_copy_file(source: str | Path, destination: str | Path) -> None:
    source_path = Path(source)
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists():
        destination_path.unlink()
    try:
        os.link(source_path, destination_path)
    except OSError:
        shutil.copy2(source_path, destination_path)


def _resize_to_height(cv2, image, target_height: int):
    height, width = image.shape[:2]
    if height == target_height:
        return image
    scale = target_height / max(height, 1)
    return cv2.resize(image, (max(1, int(round(width * scale))), target_height), interpolation=cv2.INTER_AREA)


def _resize_max(cv2, image, max_width: int = 2400, max_height: int = 1600):
    height, width = image.shape[:2]
    scale = min(1.0, max_width / max(width, 1), max_height / max(height, 1))
    if scale >= 1.0:
        return image
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def _label_image(cv2, image, label: str):
    bar_height = max(34, image.shape[0] // 32)
    bar = np.full((bar_height, image.shape[1], 3), 32, dtype=np.uint8)
    font_scale = max(0.7, min(1.1, image.shape[1] / 1800.0))
    thickness = max(1, int(round(font_scale * 2)))
    cv2.putText(
        bar,
        label,
        (14, int(bar_height * 0.68)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (235, 235, 235),
        thickness,
        cv2.LINE_AA,
    )
    return np.vstack([bar, image])


def _compose_pair(cv2, left, right, left_label: str, right_label: str, max_width: int = 2600):
    target_height = min(left.shape[0], right.shape[0])
    left_fit = _resize_to_height(cv2, left, target_height)
    right_fit = _resize_to_height(cv2, right, target_height)
    combined = np.hstack([_label_image(cv2, left_fit, left_label), _label_image(cv2, right_fit, right_label)])
    return _resize_max(cv2, combined, max_width=max_width, max_height=1600)


def _draw_detection_overlay(cv2, image, detected_points, reprojected_points=None):
    overlay = image.copy()
    radius = max(4, min(12, int(round(max(image.shape[:2]) / 650))))
    thickness = max(2, radius // 2)
    points = np.asarray(detected_points, dtype=float).reshape(-1, 2)
    for x, y in points:
        cv2.circle(overlay, (int(round(x)), int(round(y))), radius, (0, 210, 0), thickness, cv2.LINE_AA)
    if len(points) >= 2:
        origin = points[0]
        x_axis = points[min(len(points) - 1, max(1, len(points) // 2))]
        y_axis = points[min(len(points) - 1, max(2, len(points) - 1))]
        origin_pt = tuple(map(int, np.round(origin)))
        x_pt = tuple(map(int, np.round(x_axis)))
        y_pt = tuple(map(int, np.round(y_axis)))
        cv2.arrowedLine(overlay, origin_pt, x_pt, (255, 80, 80), thickness, cv2.LINE_AA, 0, 0.08)
        cv2.arrowedLine(overlay, origin_pt, y_pt, (80, 120, 255), thickness, cv2.LINE_AA, 0, 0.08)
        cv2.putText(overlay, "X", (x_pt[0] + 6, x_pt[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 80, 80), max(1, thickness), cv2.LINE_AA)
        cv2.putText(overlay, "Y", (y_pt[0] + 6, y_pt[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 120, 255), max(1, thickness), cv2.LINE_AA)
        cv2.circle(overlay, origin_pt, radius + 4, (255, 255, 0), max(1, thickness), cv2.LINE_AA)
        cv2.putText(overlay, "(0,0)", (origin_pt[0] + 8, origin_pt[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), max(1, thickness), cv2.LINE_AA)
    if reprojected_points is not None:
        arm = radius + 3
        for x, y in np.asarray(reprojected_points, dtype=float).reshape(-1, 2):
            px = int(round(x))
            py = int(round(y))
            cv2.line(overlay, (px - arm, py), (px + arm, py), (0, 0, 255), thickness, cv2.LINE_AA)
            cv2.line(overlay, (px, py - arm), (px, py + arm), (0, 0, 255), thickness, cv2.LINE_AA)
    return overlay


def _calibration_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _rectification_matrices(result: dict[str, Any]) -> dict[str, np.ndarray] | None:
    rectification = result.get("stereo", {}).get("rectification")
    if not rectification:
        return None
    return {
        "K1": np.asarray(result["left"]["camera_matrix"], dtype=np.float64),
        "D1": np.asarray(result["left"]["distortion_coefficients"], dtype=np.float64).reshape(-1),
        "K2": np.asarray(result["right"]["camera_matrix"], dtype=np.float64),
        "D2": np.asarray(result["right"]["distortion_coefficients"], dtype=np.float64).reshape(-1),
        "R1": np.asarray(rectification["R1"], dtype=np.float64),
        "R2": np.asarray(rectification["R2"], dtype=np.float64),
        "P1": np.asarray(rectification["P1"], dtype=np.float64),
        "P2": np.asarray(rectification["P2"], dtype=np.float64),
        "Q": np.asarray(rectification["Q"], dtype=np.float64),
    }


def _rectify_pair(cv2, result: dict[str, Any], left_image, right_image):
    mats = _rectification_matrices(result)
    if mats is None:
        raise CalibrationError("缺少 stereoRectify 校正参数，无法生成极线校正图。")
    image_size = tuple(map(int, result["image_size"]))
    left_map1, left_map2 = cv2.initUndistortRectifyMap(
        mats["K1"], mats["D1"], mats["R1"], mats["P1"][:, :3], image_size, cv2.CV_16SC2
    )
    right_map1, right_map2 = cv2.initUndistortRectifyMap(
        mats["K2"], mats["D2"], mats["R2"], mats["P2"][:, :3], image_size, cv2.CV_16SC2
    )
    left_rectified = cv2.remap(left_image, left_map1, left_map2, cv2.INTER_LINEAR)
    right_rectified = cv2.remap(right_image, right_map1, right_map2, cv2.INTER_LINEAR)
    return left_rectified, right_rectified


class StereoRectifier:
    def __init__(self, calibration_result: dict[str, Any]):
        self.cv2 = _load_cv2()
        mats = _rectification_matrices(calibration_result)
        if mats is None:
            raise CalibrationError("缺少 stereoRectify 校正参数，无法进行极线校正。")
        self.image_size = tuple(map(int, calibration_result["image_size"]))
        self.left_map1, self.left_map2 = self.cv2.initUndistortRectifyMap(
            mats["K1"], mats["D1"], mats["R1"], mats["P1"][:, :3], self.image_size, self.cv2.CV_16SC2
        )
        self.right_map1, self.right_map2 = self.cv2.initUndistortRectifyMap(
            mats["K2"], mats["D2"], mats["R2"], mats["P2"][:, :3], self.image_size, self.cv2.CV_16SC2
        )

    def rectify(self, left_image, right_image):
        current_size = (int(left_image.shape[1]), int(left_image.shape[0]))
        if current_size != self.image_size:
            raise CalibrationError(f"输入图像尺寸 {current_size} 与标定尺寸 {self.image_size} 不一致。")
        left_rectified = self.cv2.remap(left_image, self.left_map1, self.left_map2, self.cv2.INTER_LINEAR)
        right_rectified = self.cv2.remap(right_image, self.right_map1, self.right_map2, self.cv2.INTER_LINEAR)
        return left_rectified, right_rectified


def _draw_epipolar_pair(cv2, left_rectified, right_rectified):
    left = left_rectified.copy()
    right = right_rectified.copy()
    height = min(left.shape[0], right.shape[0])
    step = max(70, height // 12)
    for y in range(step // 2, height, step):
        color = (0, 220, 255) if (y // step) % 2 == 0 else (0, 180, 80)
        cv2.line(left, (0, y), (left.shape[1] - 1, y), color, max(2, height // 900), cv2.LINE_AA)
        cv2.line(right, (0, y), (right.shape[1] - 1, y), color, max(2, height // 900), cv2.LINE_AA)
    return _compose_pair(cv2, left, right, "left rectified", "right rectified", max_width=2800)


def _rectified_point_disparities(cv2, result: dict[str, Any], pair: dict[str, Any]) -> np.ndarray:
    mats = _rectification_matrices(result)
    if mats is None:
        return np.array([], dtype=np.float32)
    left_points = np.asarray(pair.get("left_points", []), dtype=np.float32).reshape(-1, 1, 2)
    right_points = np.asarray(pair.get("right_points", []), dtype=np.float32).reshape(-1, 1, 2)
    if len(left_points) == 0 or len(right_points) == 0:
        return np.array([], dtype=np.float32)
    left_rect = cv2.undistortPoints(left_points, mats["K1"], mats["D1"], R=mats["R1"], P=mats["P1"])
    right_rect = cv2.undistortPoints(right_points, mats["K2"], mats["D2"], R=mats["R2"], P=mats["P2"])
    return (left_rect.reshape(-1, 2)[:, 0] - right_rect.reshape(-1, 2)[:, 0]).astype(np.float32)


def _rectified_board_mask(
    cv2,
    result: dict[str, Any],
    pair: dict[str, Any],
    shape: tuple[int, int],
    scale: float,
) -> np.ndarray | None:
    mats = _rectification_matrices(result)
    if mats is None:
        return None
    left_points = np.asarray(pair.get("left_points", []), dtype=np.float32).reshape(-1, 1, 2)
    if len(left_points) < 3:
        return None
    left_rect = cv2.undistortPoints(left_points, mats["K1"], mats["D1"], R=mats["R1"], P=mats["P1"]).reshape(-1, 2)
    left_rect *= float(scale)
    finite = np.isfinite(left_rect).all(axis=1)
    pts = left_rect[finite]
    if len(pts) < 3:
        return None
    mask = np.zeros(shape, dtype=np.uint8)
    hull = cv2.convexHull(np.round(pts).astype(np.int32).reshape(-1, 1, 2))
    cv2.fillConvexPoly(mask, hull, 1)
    pad = max(3, int(round(min(shape) * 0.003)))
    kernel = np.ones((pad * 2 + 1, pad * 2 + 1), dtype=np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask > 0


def _choose_disparity_range(disparities: np.ndarray, scale: float, width: int) -> tuple[int, int]:
    finite = disparities[np.isfinite(disparities)]
    finite = finite[finite > 0]
    if finite.size:
        low = float(np.percentile(finite, 5)) * scale
        high = float(np.percentile(finite, 98)) * scale
        min_disp = max(0, int(math.floor((low - 64) / 16.0) * 16))
        target_high = max(high + 96, min_disp + 64)
        num_disp = int(math.ceil((target_high - min_disp) / 16.0) * 16)
    else:
        min_disp = 0
        num_disp = min(256, max(64, int(math.ceil(width / 6 / 16.0) * 16)))
    max_num = max(16, int(math.floor((width - min_disp - 1) / 16.0) * 16))
    num_disp = max(16, min(num_disp, max_num))
    return int(min_disp), int(num_disp)


def _config_bool(config: dict[str, Any], key: str, default: bool) -> bool:
    value = config.get(key, default)
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on", "enabled"}:
        return True
    if text in {"0", "false", "no", "n", "off", "disabled"}:
        return False
    return default


def _config_float(config: dict[str, Any], key: str, default: float, minimum: float | None = None) -> float:
    try:
        value = float(config.get(key, default))
    except (TypeError, ValueError):
        value = float(default)
    if not np.isfinite(value):
        value = float(default)
    if minimum is not None:
        value = max(float(minimum), value)
    return value


def _config_int(config: dict[str, Any], key: str, default: int, minimum: int | None = None) -> int:
    try:
        value = int(float(config.get(key, default)))
    except (TypeError, ValueError):
        value = int(default)
    if minimum is not None:
        value = max(int(minimum), value)
    return value


def _available_system_memory_bytes() -> int | None:
    try:
        import psutil  # type: ignore

        return int(psutil.virtual_memory().available)
    except Exception:
        pass

    if os.name == "nt":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullAvailPhys)
        except Exception:
            return None

    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        return page_size * available_pages
    except Exception:
        return None


def _available_cuda_memory_bytes() -> int | None:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None
    try:
        import subprocess

        completed = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    values: list[int] = []
    for line in completed.stdout.splitlines():
        try:
            values.append(int(float(line.strip())) * 1024 * 1024)
        except ValueError:
            continue
    return max(values) if values else None


def _resolve_sam3_python(config: dict[str, Any]) -> str:
    configured = str(config.get("sam3_python", "") or "").strip()
    if configured:
        return configured
    sam3_root = Path(str(config.get("sam3_root", r"D:\SAM3") or r"D:\SAM3"))
    bundled = sam3_root / ".venv" / "Scripts" / "python.exe"
    if bundled.exists():
        return str(bundled)
    return sys.executable


def _find_sam3_cached_checkpoint(sam3_root: Path, filename: str = "sam3.pt") -> Path | None:
    cache_roots = [
        sam3_root / "hf_cache" / "hub",
        Path.home() / ".cache" / "huggingface" / "hub",
    ]
    for env_name in ("HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE"):
        value = os.environ.get(env_name)
        if value:
            cache_roots.insert(0, Path(value))
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        cache_roots.insert(0, Path(hf_home) / "hub")

    seen: set[str] = set()
    for cache_root in cache_roots:
        key = str(cache_root).casefold()
        if key in seen:
            continue
        seen.add(key)
        snapshots = cache_root / "models--facebook--sam3" / "snapshots"
        if not snapshots.is_dir():
            continue
        for snapshot in sorted(snapshots.iterdir(), key=lambda item: item.name):
            checkpoint = snapshot / filename
            if checkpoint.is_file() and checkpoint.stat().st_size > 0:
                return checkpoint
    return None


def _check_sam3_python(python_exe: str, sam3_root: Path) -> dict[str, Any]:
    if not python_exe:
        return {"ok": False, "error": "SAM3 Python executable is not configured."}
    path = Path(python_exe)
    if not path.exists():
        return {"ok": False, "path": python_exe, "error": "SAM3 Python executable does not exist."}

    try:
        import subprocess

        completed = subprocess.run(
            [
                python_exe,
                "-c",
                (
                    "import json, sys; "
                    f"sys.path.insert(0, {str(sam3_root)!r}); "
                    "import torch; import sam3; "
                    "print(json.dumps({"
                    "'executable': sys.executable, "
                    "'torch': getattr(torch, '__version__', ''), "
                    "'cuda': bool(torch.cuda.is_available()), "
                    "'sam3': getattr(sam3, '__version__', '')"
                    "}))"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except Exception as exc:
        return {"ok": False, "path": python_exe, "error": str(exc)}

    detail: dict[str, Any] = {
        "path": python_exe,
        "returncode": int(completed.returncode),
        "stderr": completed.stderr.strip(),
    }
    try:
        detail.update(json.loads(completed.stdout.strip().splitlines()[-1]))
    except Exception:
        detail["stdout"] = completed.stdout.strip()
    detail["ok"] = completed.returncode == 0
    if completed.returncode != 0 and not detail.get("error"):
        detail["error"] = completed.stderr.strip() or completed.stdout.strip()
    return detail


def _estimate_reconstruction_memory_bytes(width: int, height: int, config: dict[str, Any], method_requested: str) -> int:
    pixels = max(int(width) * int(height), 1)
    method = str(method_requested).lower()
    per_pixel = 112
    if method == "sgbm":
        per_pixel += 48
    if _config_bool(config, "use_wls_filter", True):
        per_pixel += 48
    if _config_bool(config, "confidence_filter", True):
        per_pixel += 24
    # Include point coordinates, colors, masks, and temporary OpenCV/NumPy buffers.
    return int(pixels * per_pixel + 512 * 1024 * 1024)


def _estimate_crestereo_cuda_memory_bytes(config: dict[str, Any], method_requested: str) -> int | None:
    wants_crestereo = method_requested in {"cres", "crestereo", "crestereo_onnx"} or (
        method_requested == "auto" and bool(str(config.get("crestereo_model_path", "") or "").strip())
    )
    if not wants_crestereo:
        return None
    providers = config.get("crestereo_providers")
    requested = list(providers or [])
    cuda_requested = "CUDAExecutionProvider" in requested or not requested
    if not cuda_requested:
        return None
    # CREStereo ONNX inputs are fixed-size, but ORT/CUDA still needs workspace.
    return 2 * 1024 * 1024 * 1024


def _resolve_reconstruction_scale(
    config: dict[str, Any],
    width: int,
    height: int,
    method_requested: str,
) -> tuple[float, dict[str, Any]]:
    default_safe_width = 2400
    requested_max_width = int(config.get("reconstruction_max_width", default_safe_width) or 0)
    target_width = int(width) if requested_max_width <= 0 else min(int(width), requested_max_width)
    requests_original = target_width >= int(width)
    resource_policy: dict[str, Any] = {
        "requested_max_width": int(requested_max_width),
        "default_safe_max_width": int(default_safe_width),
        "original_size": [int(width), int(height)],
        "requested_original_width": bool(requests_original),
        "fallback_applied": False,
        "fallback_reasons": [],
    }

    if requests_original:
        estimated_memory = _estimate_reconstruction_memory_bytes(width, height, config, method_requested)
        available_memory = _available_system_memory_bytes()
        required_memory = int(estimated_memory * 1.25 + 512 * 1024 * 1024)
        resource_policy.update(
            {
                "estimated_memory_bytes": int(estimated_memory),
                "required_memory_bytes": int(required_memory),
                "available_memory_bytes": int(available_memory) if available_memory is not None else None,
            }
        )
        if available_memory is not None and available_memory < required_memory:
            resource_policy["fallback_reasons"].append(
                f"available system memory {available_memory} B is below required {required_memory} B"
            )
        elif available_memory is None:
            resource_policy["memory_check"] = "unknown"
        else:
            resource_policy["memory_check"] = "ok"

        required_cuda_memory = _estimate_crestereo_cuda_memory_bytes(config, method_requested)
        if required_cuda_memory is not None:
            available_cuda_memory = _available_cuda_memory_bytes()
            resource_policy.update(
                {
                    "estimated_cuda_memory_bytes": int(required_cuda_memory),
                    "available_cuda_memory_bytes": int(available_cuda_memory) if available_cuda_memory is not None else None,
                }
            )
            if available_cuda_memory is not None and available_cuda_memory < required_cuda_memory:
                resource_policy["fallback_reasons"].append(
                    f"available CUDA memory {available_cuda_memory} B is below required {required_cuda_memory} B"
                )
            elif available_cuda_memory is None:
                resource_policy["cuda_memory_check"] = "unknown"
            else:
                resource_policy["cuda_memory_check"] = "ok"

        if resource_policy["fallback_reasons"]:
            target_width = min(int(width), default_safe_width)
            resource_policy["fallback_applied"] = True

    scale = min(1.0, float(max(target_width, 1)) / max(float(width), 1.0))
    resource_policy["effective_max_width"] = int(target_width)
    resource_policy["effective_size"] = [
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    ]
    resource_policy["scale"] = float(scale)
    return scale, resource_policy


def _normalize_reconstruction_config(config: dict[str, Any] | None) -> dict[str, Any]:
    normalized = {
        "reconstruction_method": "sgbm",
        "allow_sgbm_fallback": True,
        "prompt_before_sgbm_fallback": True,
        "force_sgbm_fallback": False,
        "sgbm_fallback_reason": "",
        "crestereo_model_path": "",
        "crestereo_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "use_wls_filter": True,
        "confidence_filter": True,
        "confidence_threshold": 0.35,
        "confidence_photometric_sigma": 0.15,
        "left_right_consistency_px": 2.0,
        "left_right_consistency_min_mean": 0.05,
        "left_right_consistency_min_pass_ratio": 0.01,
        "crestereo_validate_right_disparity": True,
        "crestereo_lr_fail_fallback": True,
        "wls_consistency_px": 2.0,
        "wls_lambda": 8000.0,
        "wls_sigma_color": 1.5,
        "reconstruction_max_width": 2400,
        "sam3_segmentation": True,
        "sam3_root": r"D:\SAM3",
        "sam3_python": r"D:\SAM3\.venv\Scripts\python.exe",
        "sam3_checkpoint": "",
        "sam3_prompt": "object",
        "sam3_confidence_threshold": 0.25,
        "sam3_top_k": 50,
        "sam3_resolution": 1008,
        "sam3_mask_selection": "union",
        "sam3_timeout_seconds": 600,
        "sam3_dilate_pixels": 0,
        "sam3_erode_pixels": 0,
        "sam3_filter_valid_depth": True,
        "sam3_required": False,
        "sam3_device": "auto",
        "depth_scale_validation_enabled": False,
        "depth_scale_reference_distance_mm": 0.0,
        "depth_scale_validation_tolerance_mm": 5.0,
        "depth_scale_validation_tolerance_percent": 0.5,
        "world_coordinate_enabled": False,
        "world_reference_prompt": "fixed target",
        "world_reference_required": False,
        "world_reference_min_points": 200,
    }
    if config:
        normalized.update(config)
    normalized["reconstruction_method"] = str(normalized.get("reconstruction_method", "sgbm")).strip().lower() or "sgbm"
    normalized["allow_sgbm_fallback"] = _config_bool(normalized, "allow_sgbm_fallback", True)
    normalized["prompt_before_sgbm_fallback"] = _config_bool(normalized, "prompt_before_sgbm_fallback", True)
    normalized["force_sgbm_fallback"] = _config_bool(normalized, "force_sgbm_fallback", False)
    normalized["sgbm_fallback_reason"] = str(normalized.get("sgbm_fallback_reason", "") or "").strip()
    normalized["use_wls_filter"] = _config_bool(normalized, "use_wls_filter", True)
    normalized["confidence_filter"] = _config_bool(normalized, "confidence_filter", True)
    normalized["confidence_threshold"] = _config_float(normalized, "confidence_threshold", 0.35, 0.0)
    normalized["confidence_photometric_sigma"] = _config_float(normalized, "confidence_photometric_sigma", 0.15, 1e-6)
    normalized["left_right_consistency_px"] = _config_float(normalized, "left_right_consistency_px", 2.0, 1e-6)
    normalized["left_right_consistency_min_mean"] = _config_float(normalized, "left_right_consistency_min_mean", 0.05, 0.0)
    normalized["left_right_consistency_min_pass_ratio"] = _config_float(normalized, "left_right_consistency_min_pass_ratio", 0.01, 0.0)
    normalized["crestereo_validate_right_disparity"] = _config_bool(normalized, "crestereo_validate_right_disparity", True)
    normalized["crestereo_lr_fail_fallback"] = _config_bool(normalized, "crestereo_lr_fail_fallback", True)
    normalized["wls_consistency_px"] = _config_float(normalized, "wls_consistency_px", 2.0, 1e-6)
    normalized["wls_lambda"] = _config_float(normalized, "wls_lambda", 8000.0, 0.0)
    normalized["wls_sigma_color"] = _config_float(normalized, "wls_sigma_color", 1.5, 0.0)
    normalized["sam3_segmentation"] = _config_bool(normalized, "sam3_segmentation", True)
    normalized["sam3_filter_valid_depth"] = _config_bool(normalized, "sam3_filter_valid_depth", True)
    normalized["sam3_required"] = _config_bool(normalized, "sam3_required", False)
    normalized["sam3_device"] = str(normalized.get("sam3_device", "auto") or "auto").strip().lower()
    if normalized["sam3_device"] not in {"auto", "cuda", "cpu"}:
        normalized["sam3_device"] = "auto"
    normalized["sam3_root"] = str(normalized.get("sam3_root", r"D:\SAM3") or "").strip()
    normalized["sam3_python"] = str(normalized.get("sam3_python", r"D:\SAM3\.venv\Scripts\python.exe") or "").strip()
    normalized["sam3_checkpoint"] = str(normalized.get("sam3_checkpoint", "") or "").strip()
    normalized["sam3_prompt"] = str(normalized.get("sam3_prompt", "object") or "object").strip() or "object"
    normalized["sam3_confidence_threshold"] = _config_float(normalized, "sam3_confidence_threshold", 0.25, 0.0)
    normalized["sam3_top_k"] = _config_int(normalized, "sam3_top_k", 50, 0)
    normalized["sam3_resolution"] = _config_int(normalized, "sam3_resolution", 1008, 224)
    normalized["sam3_timeout_seconds"] = _config_int(normalized, "sam3_timeout_seconds", 600, 30)
    normalized["sam3_dilate_pixels"] = _config_int(normalized, "sam3_dilate_pixels", 0, 0)
    normalized["sam3_erode_pixels"] = _config_int(normalized, "sam3_erode_pixels", 0, 0)
    mask_selection = str(normalized.get("sam3_mask_selection", "union") or "union").strip().lower()
    if mask_selection not in {"union", "best", "largest"}:
        mask_selection = "union"
    normalized["sam3_mask_selection"] = mask_selection
    normalized["depth_scale_validation_enabled"] = _config_bool(normalized, "depth_scale_validation_enabled", False)
    normalized["depth_scale_reference_distance_mm"] = _config_float(normalized, "depth_scale_reference_distance_mm", 0.0, 0.0)
    normalized["depth_scale_validation_tolerance_mm"] = _config_float(normalized, "depth_scale_validation_tolerance_mm", 5.0, 0.0)
    normalized["depth_scale_validation_tolerance_percent"] = _config_float(
        normalized,
        "depth_scale_validation_tolerance_percent",
        0.5,
        0.0,
    )
    normalized["world_coordinate_enabled"] = _config_bool(normalized, "world_coordinate_enabled", False)
    normalized["world_reference_prompt"] = str(normalized.get("world_reference_prompt", "fixed target") or "fixed target").strip() or "fixed target"
    normalized["world_reference_required"] = _config_bool(normalized, "world_reference_required", False)
    normalized["world_reference_min_points"] = _config_int(normalized, "world_reference_min_points", 200, 3)
    max_width = _config_int(normalized, "reconstruction_max_width", 2400)
    if max_width <= 0:
        normalized["reconstruction_max_width"] = 0
    else:
        normalized["reconstruction_max_width"] = max(320, max_width)
    providers = normalized.get("crestereo_providers")
    if isinstance(providers, str):
        normalized["crestereo_providers"] = [item.strip() for item in providers.split(",") if item.strip()]
    elif providers is None:
        normalized["crestereo_providers"] = None
    else:
        normalized["crestereo_providers"] = [str(item) for item in providers]
    return normalized


def check_reconstruction_environment(config: dict[str, Any] | None = None) -> dict[str, Any]:
    normalized = _normalize_reconstruction_config(config)
    method = str(normalized.get("reconstruction_method", "sgbm")).lower()
    wants_crestereo = method in {"cres", "crestereo", "crestereo_onnx"} or (
        method == "auto" and bool(str(normalized.get("crestereo_model_path", "") or "").strip())
    )
    checks: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    errors: list[str] = []

    model_path_text = str(normalized.get("crestereo_model_path", "") or "").strip()
    model_path = Path(model_path_text) if model_path_text else None
    model_exists = bool(model_path and model_path.exists() and model_path.is_file())
    checks["crestereo_model_file"] = {
        "required": bool(wants_crestereo),
        "ok": bool(model_exists) if wants_crestereo else (not model_path_text or bool(model_exists)),
        "path": model_path_text,
    }
    if wants_crestereo and not model_exists:
        message = f"CREStereo ONNX 模型文件不存在：{model_path_text or '(未填写)'}"
        if bool(normalized.get("allow_sgbm_fallback", True)):
            warnings.append(message + "；将尝试 SGBM fallback。")
        else:
            errors.append(message)

    ort_available = False
    ort_providers: list[str] = []
    ort_error = ""
    try:
        import onnxruntime as ort

        ort_available = True
        ort_providers = list(ort.get_available_providers())
    except Exception as exc:
        ort_error = str(exc)
    checks["onnxruntime"] = {
        "required": bool(wants_crestereo),
        "ok": bool(ort_available) if wants_crestereo else bool(ort_available),
        "available_providers": ort_providers,
        "error": ort_error,
    }
    if wants_crestereo and not ort_available:
        message = "onnxruntime 不可用，无法执行 CREStereo。"
        if bool(normalized.get("allow_sgbm_fallback", True)):
            warnings.append(message + " 将尝试 SGBM fallback。")
        else:
            errors.append(message)

    requested_providers = list(normalized.get("crestereo_providers") or [])
    cuda_requested = "CUDAExecutionProvider" in requested_providers or not requested_providers
    cuda_available = "CUDAExecutionProvider" in ort_providers
    checks["cuda_provider"] = {
        "required": bool(wants_crestereo and cuda_requested),
        "ok": bool(cuda_available),
        "requested": requested_providers,
        "available_providers": ort_providers,
    }
    if wants_crestereo and cuda_requested and ort_available and not cuda_available:
        warnings.append("onnxruntime 未检测到 CUDAExecutionProvider；CREStereo 将使用 CPU 或 fallback，速度会明显下降。")

    cv2_error = ""
    ximgproc_available = False
    wls_generic_available = False
    wls_sgbm_available = False
    try:
        cv2 = _load_cv2()
        ximgproc_available = hasattr(cv2, "ximgproc")
        wls_generic_available = ximgproc_available and hasattr(cv2.ximgproc, "createDisparityWLSFilterGeneric")
        wls_sgbm_available = ximgproc_available and hasattr(cv2.ximgproc, "createDisparityWLSFilter") and hasattr(cv2.ximgproc, "createRightMatcher")
    except Exception as exc:
        cv2_error = str(exc)
    checks["opencv_ximgproc"] = {
        "required": bool(normalized.get("use_wls_filter", True)),
        "ok": bool(ximgproc_available),
        "error": cv2_error,
    }
    checks["wls_interfaces"] = {
        "required": bool(normalized.get("use_wls_filter", True)),
        "ok": bool(wls_generic_available or wls_sgbm_available),
        "generic_wls": bool(wls_generic_available),
        "sgbm_wls": bool(wls_sgbm_available),
    }
    if bool(normalized.get("use_wls_filter", True)) and not (wls_generic_available or wls_sgbm_available):
        warnings.append("OpenCV ximgproc/WLS 接口不可用；重建会跳过 WLS 滤波。")

    wants_sam3 = bool(normalized.get("sam3_segmentation", True))
    sam3_root_text = str(normalized.get("sam3_root", "") or "").strip()
    sam3_root = Path(sam3_root_text) if sam3_root_text else Path()
    sam3_root_ok = bool(sam3_root_text and sam3_root.exists() and sam3_root.is_dir())
    checks["sam3_root"] = {
        "required": wants_sam3,
        "ok": sam3_root_ok if wants_sam3 else (not sam3_root_text or sam3_root_ok),
        "path": sam3_root_text,
    }
    if wants_sam3 and not sam3_root_ok:
        message = f"SAM3 path does not exist: {sam3_root_text or '(empty)'}"
        if bool(normalized.get("sam3_required", False)):
            errors.append(message)
        else:
            warnings.append(message + "; object_mask filtering will be skipped.")

    sam3_script_ok = bool(SAM3_MASK_SCRIPT.exists() and SAM3_MASK_SCRIPT.is_file())
    checks["sam3_adapter_script"] = {
        "required": wants_sam3,
        "ok": sam3_script_ok,
        "path": str(SAM3_MASK_SCRIPT),
    }
    if wants_sam3 and not sam3_script_ok:
        message = f"SAM3 adapter script is missing: {SAM3_MASK_SCRIPT}"
        if bool(normalized.get("sam3_required", False)):
            errors.append(message)
        else:
            warnings.append(message + "; object_mask filtering will be skipped.")

    checkpoint_text = str(normalized.get("sam3_checkpoint", "") or "").strip()
    checkpoint_path = Path(checkpoint_text) if checkpoint_text else (_find_sam3_cached_checkpoint(sam3_root) if sam3_root_ok else None)
    checkpoint_ok = bool(checkpoint_path and checkpoint_path.is_file() and checkpoint_path.stat().st_size > 0)
    checks["sam3_checkpoint"] = {
        "required": wants_sam3,
        "ok": checkpoint_ok if wants_sam3 else (not checkpoint_text or checkpoint_ok),
        "path": "" if checkpoint_path is None else str(checkpoint_path),
        "configured_path": checkpoint_text,
    }
    if wants_sam3 and not checkpoint_ok:
        message = "SAM3 checkpoint was not found; set sam3_checkpoint or keep sam3.pt in D:\\SAM3\\hf_cache."
        if bool(normalized.get("sam3_required", False)):
            errors.append(message)
        else:
            warnings.append(message + " SAM3 may auto-download it, or object_mask filtering may be skipped.")

    sam3_python = _resolve_sam3_python(normalized)
    sam3_python_check = _check_sam3_python(sam3_python, sam3_root) if wants_sam3 and sam3_root_ok else {
        "ok": False,
        "path": sam3_python,
        "error": "not checked",
    }
    sam3_python_check["required"] = wants_sam3
    checks["sam3_python"] = sam3_python_check
    if wants_sam3 and not sam3_python_check.get("ok"):
        message = f"SAM3 Python environment is not available: {sam3_python_check.get('error', sam3_python)}"
        if bool(normalized.get("sam3_required", False)):
            errors.append(message)
        else:
            warnings.append(message + "; object_mask filtering will be skipped.")

    ok = not errors
    return {
        "ok": bool(ok),
        "method": method,
        "allow_sgbm_fallback": bool(normalized.get("allow_sgbm_fallback", True)),
        "prompt_before_sgbm_fallback": bool(normalized.get("prompt_before_sgbm_fallback", True)),
        "checks": checks,
        "warnings": warnings,
        "errors": errors,
    }


def _make_sgbm_matcher(cv2, min_disp: int, num_disp: int, block_size: int = 5):
    return cv2.StereoSGBM_create(
        minDisparity=int(min_disp),
        numDisparities=int(num_disp),
        blockSize=int(block_size),
        P1=8 * block_size * block_size,
        P2=32 * block_size * block_size,
        disp12MaxDiff=1,
        uniquenessRatio=8,
        speckleWindowSize=80,
        speckleRange=2,
        preFilterCap=63,
        mode=getattr(cv2, "STEREO_SGBM_MODE_SGBM_3WAY", cv2.STEREO_SGBM_MODE_SGBM),
    )


def _wls_metadata(enabled: bool, status: str, config: dict[str, Any], note: str = "") -> dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "status": status,
        "lambda": float(config.get("wls_lambda", 8000.0)),
        "sigma_color": float(config.get("wls_sigma_color", 1.5)),
        "note": note,
    }


def _apply_generic_wls_filter(cv2, disparity: np.ndarray, guide_image, config: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    if not _config_bool(config, "use_wls_filter", True):
        return disparity.astype(np.float32), _wls_metadata(False, "disabled", config)
    if not hasattr(cv2, "ximgproc") or not hasattr(cv2.ximgproc, "createDisparityWLSFilterGeneric"):
        return disparity.astype(np.float32), _wls_metadata(False, "unavailable", config, "OpenCV ximgproc WLS is unavailable")

    finite = np.isfinite(disparity)
    scaled = np.where(finite, disparity, -1.0) * 16.0
    scaled = np.clip(np.round(scaled), -32768, 32767).astype(np.int16)
    try:
        wls_filter = cv2.ximgproc.createDisparityWLSFilterGeneric(False)
        wls_filter.setLambda(float(config.get("wls_lambda", 8000.0)))
        wls_filter.setSigmaColor(float(config.get("wls_sigma_color", 1.5)))
        filtered = wls_filter.filter(scaled, guide_image).astype(np.float32) / 16.0
        filtered[~finite] = np.nan
    except Exception as exc:
        return disparity.astype(np.float32), _wls_metadata(False, "failed", config, str(exc))
    return filtered.astype(np.float32), _wls_metadata(True, "applied", config)


def _compute_sgbm_disparity(
    cv2,
    gray_left: np.ndarray,
    gray_right: np.ndarray,
    left_guide,
    min_disp: int,
    num_disp: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    block_size = 5
    left_matcher = _make_sgbm_matcher(cv2, min_disp, num_disp, block_size)
    raw_left = left_matcher.compute(gray_left, gray_right)
    raw_disparity = raw_left.astype(np.float32) / 16.0
    filtered = raw_disparity
    right_disparity = None
    wls_confidence = None
    wls_info = _wls_metadata(False, "disabled", config)

    if _config_bool(config, "use_wls_filter", True):
        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "createDisparityWLSFilter") and hasattr(cv2.ximgproc, "createRightMatcher"):
            try:
                right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
                raw_right = right_matcher.compute(gray_right, gray_left)
                wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
                wls_filter.setLambda(float(config.get("wls_lambda", 8000.0)))
                wls_filter.setSigmaColor(float(config.get("wls_sigma_color", 1.5)))
                filtered = wls_filter.filter(raw_left, left_guide, None, raw_right).astype(np.float32) / 16.0
                right_disparity = raw_right.astype(np.float32) / 16.0
                wls_confidence = wls_filter.getConfidenceMap().astype(np.float32)
                wls_info = _wls_metadata(True, "applied", config)
            except Exception as exc:
                filtered, wls_info = _apply_generic_wls_filter(cv2, raw_disparity, left_guide, config)
                wls_info["note"] = f"SGBM WLS with right matcher failed; generic WLS used. {exc}"
        else:
            filtered, wls_info = _apply_generic_wls_filter(cv2, raw_disparity, left_guide, config)

    return {
        "method": "sgbm",
        "raw_disparity": raw_disparity.astype(np.float32),
        "disparity": filtered.astype(np.float32),
        "right_disparity": right_disparity,
        "wls_confidence": wls_confidence,
        "wls_filter": wls_info,
        "metadata": {
            "block_size": int(block_size),
            "min_disparity": int(min_disp),
            "num_disparities": int(num_disp),
        },
    }


def _left_right_error_metrics_for_candidate(
    cv2,
    left_disparity: np.ndarray,
    right_disparity: np.ndarray,
    threshold_px: float,
) -> dict[str, Any]:
    left = np.asarray(left_disparity, dtype=np.float32)
    right = np.asarray(right_disparity, dtype=np.float32)
    if left.shape != right.shape:
        return {
            "valid": False,
            "count": 0,
            "mean_abs_error_px": None,
            "median_abs_error_px": None,
            "p95_abs_error_px": None,
            "pass_ratio": 0.0,
        }

    height, width = left.shape[:2]
    xs, ys = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    valid = np.isfinite(left) & np.isfinite(right) & (left > 0.5)
    map_x = xs - left
    map_y = ys
    in_bounds = valid & np.isfinite(map_x) & (map_x >= 0.0) & (map_x <= float(width - 1))
    if not np.any(in_bounds):
        return {
            "valid": False,
            "count": 0,
            "mean_abs_error_px": None,
            "median_abs_error_px": None,
            "p95_abs_error_px": None,
            "pass_ratio": 0.0,
        }

    sampled = cv2.remap(
        right,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=np.nan,
    )
    error = np.abs(left + sampled)
    mask = in_bounds & np.isfinite(error)
    if not np.any(mask):
        return {
            "valid": False,
            "count": 0,
            "mean_abs_error_px": None,
            "median_abs_error_px": None,
            "p95_abs_error_px": None,
            "pass_ratio": 0.0,
        }

    values = error[mask].astype(np.float32, copy=False)
    return {
        "valid": True,
        "count": int(values.size),
        "mean_abs_error_px": float(np.mean(values)),
        "median_abs_error_px": float(np.percentile(values, 50)),
        "p95_abs_error_px": float(np.percentile(values, 95)),
        "pass_ratio": float(np.count_nonzero(values <= float(threshold_px)) / max(int(values.size), 1)),
    }


def _right_disparity_candidate_score(metrics: dict[str, Any]) -> tuple[float, float, float]:
    if not metrics.get("valid"):
        return (float("inf"), float("inf"), 1.0)
    return (
        float(metrics.get("median_abs_error_px") if metrics.get("median_abs_error_px") is not None else float("inf")),
        float(metrics.get("p95_abs_error_px") if metrics.get("p95_abs_error_px") is not None else float("inf")),
        -float(metrics.get("pass_ratio") or 0.0),
    )


def _format_right_disparity_policy(policy: dict[str, Any]) -> str:
    if not isinstance(policy, dict) or not policy:
        return ""
    selected = policy.get("selected", "")
    metrics = policy.get("candidate_metrics", {})
    selected_metrics = metrics.get(selected, {}) if isinstance(metrics, dict) else {}
    parts = [
        f"selected={selected or '--'}",
        f"status={policy.get('status', '--')}",
        f"threshold_px={policy.get('threshold_px', '--')}",
        f"min_pass_ratio={policy.get('min_pass_ratio', '--')}",
    ]
    if selected_metrics:
        parts.extend(
            [
                f"pass_ratio={selected_metrics.get('pass_ratio', '--')}",
                f"median_abs_error_px={selected_metrics.get('median_abs_error_px', '--')}",
                f"p95_abs_error_px={selected_metrics.get('p95_abs_error_px', '--')}",
            ]
        )
    return ", ".join(parts)


def _select_right_disparity_convention(
    cv2,
    left_disparity: np.ndarray,
    right_raw_disparity: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    right_raw = np.asarray(right_raw_disparity, dtype=np.float32)
    if not _config_bool(config, "crestereo_validate_right_disparity", True):
        return -np.abs(right_raw), {
            "status": "legacy_negative_abs",
            "selected": "negative_abs",
            "validated": False,
        }

    candidates: dict[str, np.ndarray] = {
        "raw": right_raw,
        "negative_raw": -right_raw,
        "negative_abs": -np.abs(right_raw),
        "positive_abs": np.abs(right_raw),
    }
    threshold_px = _config_float(config, "left_right_consistency_px", 2.0, 1e-6)
    metrics = {
        name: _left_right_error_metrics_for_candidate(cv2, left_disparity, candidate, threshold_px)
        for name, candidate in candidates.items()
    }
    selected = min(metrics, key=lambda name: _right_disparity_candidate_score(metrics[name]))
    selected_metrics = metrics[selected]
    min_pass_ratio = _config_float(config, "left_right_consistency_min_pass_ratio", 0.01, 0.0)
    status = "ok" if selected_metrics.get("valid") and float(selected_metrics.get("pass_ratio") or 0.0) >= min_pass_ratio else "invalid"
    return candidates[selected], {
        "status": status,
        "selected": selected,
        "validated": True,
        "threshold_px": float(threshold_px),
        "min_pass_ratio": float(min_pass_ratio),
        "candidate_metrics": metrics,
    }


def _compute_crestereo_disparity(cv2, left_image, right_image, config: dict[str, Any]) -> dict[str, Any]:
    model_path = str(config.get("crestereo_model_path", "") or "").strip()
    if not model_path:
        raise CalibrationError("已选择 CREStereo，但未配置 crestereo_model_path。")
    try:
        from crestereo_inference import CREStereoONNX
    except Exception as exc:
        raise CalibrationError(f"无法导入 CREStereo 推理模块：{exc}") from exc

    providers = config.get("crestereo_providers")
    provider_key = tuple(providers) if providers is not None else None
    cache_key = (str(Path(model_path)), provider_key)
    model = _CRESTEREO_MODEL_CACHE.get(cache_key)
    if model is None:
        model = CREStereoONNX(model_path, providers=providers)
        _CRESTEREO_MODEL_CACHE[cache_key] = model

    result = model.predict(cv2, left_image, right_image)
    right_result = model.predict(cv2, right_image, left_image)
    right_raw = np.asarray(right_result.disparity, dtype=np.float32)
    right_disparity, right_policy = _select_right_disparity_convention(
        cv2,
        np.asarray(result.disparity, dtype=np.float32),
        right_raw,
        config,
    )
    filtered, wls_info = _apply_generic_wls_filter(cv2, result.disparity, left_image, config)
    return {
        "method": "crestereo",
        "raw_disparity": result.disparity.astype(np.float32),
        "disparity": filtered.astype(np.float32),
        "right_disparity": right_disparity.astype(np.float32),
        "raw_right_disparity": right_raw.astype(np.float32),
        "wls_confidence": None,
        "wls_filter": wls_info,
        "metadata": {
            "model_path": result.model_path,
            "input_width": int(result.input_width),
            "input_height": int(result.input_height),
            "providers": result.providers,
            "input_names": result.input_names,
            "output_names": result.output_names,
            "right_inference": True,
            "right_disparity_policy": right_policy,
        },
    }


def _photometric_confidence(
    cv2,
    gray_left: np.ndarray,
    gray_right: np.ndarray,
    disparity: np.ndarray,
    valid: np.ndarray,
    sigma: float,
) -> np.ndarray:
    height, width = disparity.shape[:2]
    xs, ys = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    map_x = xs - disparity.astype(np.float32)
    map_y = ys
    in_bounds = valid & np.isfinite(map_x) & (map_x >= 0.0) & (map_x <= float(width - 1))
    warped_right = cv2.remap(
        gray_right,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    error = np.abs(gray_left.astype(np.float32) - warped_right.astype(np.float32)) / 255.0
    confidence = np.exp(-error / max(float(sigma), 1e-6)).astype(np.float32)
    confidence[~in_bounds] = 0.0
    return confidence


def _left_right_confidence(
    cv2,
    disparity: np.ndarray,
    right_disparity: np.ndarray | None,
    valid: np.ndarray,
    threshold_px: float,
) -> np.ndarray | None:
    if right_disparity is None:
        return None
    height, width = disparity.shape[:2]
    xs, ys = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    map_x = xs - disparity.astype(np.float32)
    map_y = ys
    in_bounds = valid & np.isfinite(map_x) & (map_x >= 0.0) & (map_x <= float(width - 1))
    sampled_right = cv2.remap(
        right_disparity.astype(np.float32),
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=1.0e6,
    )
    error = np.abs(disparity.astype(np.float32) + sampled_right)
    confidence = np.clip(1.0 - error / max(float(threshold_px), 1e-6), 0.0, 1.0).astype(np.float32)
    confidence[~in_bounds] = 0.0
    return confidence


def _normalize_wls_confidence(confidence: np.ndarray | None, valid: np.ndarray) -> np.ndarray | None:
    if confidence is None:
        return None
    normalized = np.asarray(confidence, dtype=np.float32)
    if normalized.shape != valid.shape:
        return None
    finite = np.isfinite(normalized)
    if not np.any(finite):
        return None
    max_value = float(np.nanmax(normalized[finite]))
    if max_value > 1.5:
        normalized = normalized / 255.0
    normalized = np.clip(normalized, 0.0, 1.0).astype(np.float32)
    normalized[~valid] = 0.0
    return normalized


def _wls_consistency_confidence(
    raw_disparity: np.ndarray | None,
    filtered_disparity: np.ndarray,
    valid: np.ndarray,
    threshold_px: float,
) -> np.ndarray | None:
    if raw_disparity is None:
        return None
    raw = np.asarray(raw_disparity, dtype=np.float32)
    filtered = np.asarray(filtered_disparity, dtype=np.float32)
    if raw.shape != filtered.shape or raw.shape != valid.shape:
        return None
    finite = valid & np.isfinite(raw) & np.isfinite(filtered)
    if not np.any(finite):
        return None
    delta = np.abs(raw - filtered)
    if float(np.nanmax(delta[finite])) <= 1e-4:
        return None
    confidence = np.exp(-delta / max(float(threshold_px), 1e-6)).astype(np.float32)
    confidence[~finite] = 0.0
    return confidence


def _confidence_source_metrics(confidence: np.ndarray, valid: np.ndarray, threshold: float) -> dict[str, Any]:
    values = np.asarray(confidence, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(values)
    if not np.any(mask):
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "p99": None,
            "max": None,
            "pass_ratio": 0.0,
            "pass_count": 0,
        }
    selected = values[mask]
    pass_mask = selected >= float(threshold)
    return {
        "count": int(selected.size),
        "mean": float(np.mean(selected)),
        "median": float(np.percentile(selected, 50)),
        "p95": float(np.percentile(selected, 95)),
        "p99": float(np.percentile(selected, 99)),
        "max": float(np.max(selected)),
        "pass_ratio": float(np.count_nonzero(pass_mask) / max(int(selected.size), 1)),
        "pass_count": int(np.count_nonzero(pass_mask)),
    }


def _compute_confidence_map(
    cv2,
    gray_left: np.ndarray,
    gray_right: np.ndarray,
    disparity: np.ndarray,
    valid: np.ndarray,
    config: dict[str, Any],
    raw_disparity: np.ndarray | None = None,
    right_disparity: np.ndarray | None = None,
    wls_confidence: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    maps: list[tuple[str, np.ndarray]] = []
    warnings: list[str] = []
    source_metrics: dict[str, Any] = {}
    photometric = _photometric_confidence(
        cv2,
        gray_left,
        gray_right,
        disparity,
        valid,
        _config_float(config, "confidence_photometric_sigma", 0.15, 1e-6),
    )
    maps.append(("photometric", photometric))
    source_metrics["photometric"] = _confidence_source_metrics(photometric, valid, float(config.get("confidence_threshold", 0.35)))
    lr_confidence = _left_right_confidence(
        cv2,
        disparity,
        right_disparity,
        valid,
        _config_float(config, "left_right_consistency_px", 2.0, 1e-6),
    )
    if lr_confidence is not None:
        lr_metrics = _confidence_source_metrics(lr_confidence, valid, float(config.get("confidence_threshold", 0.35)))
        source_metrics["left_right_consistency"] = lr_metrics
        min_mean = _config_float(config, "left_right_consistency_min_mean", 0.05, 0.0)
        min_pass_ratio = _config_float(config, "left_right_consistency_min_pass_ratio", 0.01, 0.0)
        if float(lr_metrics["mean"] or 0.0) >= min_mean and float(lr_metrics["pass_ratio"] or 0.0) >= min_pass_ratio:
            maps.append(("left_right_consistency", lr_confidence))
        else:
            warnings.append(
                "left-right consistency 置信度整体过低，已自动忽略该分量，避免把整张深度图过滤成黑图。"
                f" mean={float(lr_metrics['mean'] or 0.0):.6f}, pass_ratio={float(lr_metrics['pass_ratio'] or 0.0):.6f}"
            )
    normalized_wls = _normalize_wls_confidence(wls_confidence, valid)
    if normalized_wls is not None:
        maps.append(("wls_confidence", normalized_wls))
        source_metrics["wls_confidence"] = _confidence_source_metrics(normalized_wls, valid, float(config.get("confidence_threshold", 0.35)))
    wls_consistency = _wls_consistency_confidence(
        raw_disparity,
        disparity,
        valid,
        _config_float(config, "wls_consistency_px", 2.0, 1e-6),
    )
    if wls_consistency is not None:
        maps.append(("wls_consistency", wls_consistency))
        source_metrics["wls_consistency"] = _confidence_source_metrics(wls_consistency, valid, float(config.get("confidence_threshold", 0.35)))

    weights = {
        "photometric": 0.45,
        "left_right_consistency": 0.35,
        "wls_confidence": 0.20,
        "wls_consistency": 0.20,
    }
    stacked = np.stack([np.clip(item[1], 1e-6, 1.0) for item in maps], axis=0)
    active_weights = np.asarray([weights.get(item[0], 0.2) for item in maps], dtype=np.float32)
    active_weights /= max(float(active_weights.sum()), 1e-6)
    confidence = np.exp(np.sum(np.log(stacked) * active_weights[:, None, None], axis=0)).astype(np.float32)
    confidence[~valid] = 0.0
    return confidence, {
        "sources": [item[0] for item in maps],
        "ignored_sources": [name for name in source_metrics if name not in {item[0] for item in maps}],
        "warnings": warnings,
        "source_metrics": source_metrics,
        "fusion": "weighted_geometric_mean",
        "photometric_sigma": float(config.get("confidence_photometric_sigma", 0.15)),
        "left_right_consistency_px": float(config.get("left_right_consistency_px", 2.0)),
        "left_right_consistency_min_mean": float(config.get("left_right_consistency_min_mean", 0.05)),
        "left_right_consistency_min_pass_ratio": float(config.get("left_right_consistency_min_pass_ratio", 0.01)),
        "wls_consistency_px": float(config.get("wls_consistency_px", 2.0)),
    }


def _confidence_image(cv2, confidence: np.ndarray, valid: np.ndarray):
    color_map = getattr(cv2, "COLORMAP_VIRIDIS", getattr(cv2, "COLORMAP_JET"))
    clipped = np.clip(confidence, 0.0, 1.0)
    gray = (clipped * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(gray, color_map)
    output = np.zeros_like(colored)
    output[valid] = colored[valid]
    return output


def _mask_preview_image(cv2, image, mask: np.ndarray):
    valid_mask = np.asarray(mask, dtype=bool)
    overlay = np.asarray(image).copy()
    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
    if valid_mask.shape != overlay.shape[:2]:
        valid_mask = cv2.resize(
            valid_mask.astype(np.uint8),
            (overlay.shape[1], overlay.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ) > 0
    if not np.any(valid_mask):
        return overlay
    color = np.zeros_like(overlay)
    color[:, :] = (0, 190, 255)
    overlay[valid_mask] = np.clip(
        overlay[valid_mask].astype(np.float32) * 0.45 + color[valid_mask].astype(np.float32) * 0.55,
        0,
        255,
    ).astype(np.uint8)
    contours, _hierarchy = cv2.findContours(valid_mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), max(1, int(round(min(overlay.shape[:2]) / 600))))
    return overlay


def _colored_range_image(cv2, values: np.ndarray, valid: np.ndarray, colormap: int | None = None, invert: bool = False):
    color_map = colormap if colormap is not None else getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
    output = np.zeros((*values.shape, 3), dtype=np.uint8)
    if not np.any(valid):
        return output, None
    finite_values = values[valid]
    lo, hi = np.percentile(finite_values, [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(finite_values))
        hi = float(np.max(finite_values) + 1.0)
    normalized = np.clip((values - lo) / max(hi - lo, 1e-6), 0, 1)
    if invert:
        normalized = 1.0 - normalized
    gray = (normalized * 255).astype(np.uint8)
    colored = cv2.applyColorMap(gray, color_map)
    output[valid] = colored[valid]
    return output, (float(lo), float(hi))


def _save_ply(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    semantics: dict[str, np.ndarray] | None = None,
) -> None:
    semantic_label = None if semantics is None else np.asarray(semantics.get("semantic_label", []), dtype=np.int32)
    semantic_id = None if semantics is None else np.asarray(semantics.get("semantic_id", []), dtype=np.int32)
    instance_id = None if semantics is None else np.asarray(semantics.get("instance_id", []), dtype=np.int32)
    confidence = None if semantics is None else np.asarray(semantics.get("confidence", []), dtype=np.float32)
    has_semantics = (
        semantic_label is not None
        and semantic_id is not None
        and instance_id is not None
        and confidence is not None
        and len(semantic_label) == len(points)
        and len(semantic_id) == len(points)
        and len(instance_id) == len(points)
        and len(confidence) == len(points)
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as fh:
        fh.write("ply\n")
        fh.write("format ascii 1.0\n")
        fh.write(f"element vertex {len(points)}\n")
        fh.write("property float x\n")
        fh.write("property float y\n")
        fh.write("property float z\n")
        fh.write("property uchar red\n")
        fh.write("property uchar green\n")
        fh.write("property uchar blue\n")
        if has_semantics:
            fh.write("property int semantic_label\n")
            fh.write("property int semantic_id\n")
            fh.write("property int instance_id\n")
            fh.write("property float confidence\n")
        fh.write("end_header\n")
        for index, (point, color) in enumerate(zip(points, colors)):
            line = (
                f"{point[0]:.4f} {point[1]:.4f} {point[2]:.4f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}"
            )
            if has_semantics:
                line += (
                    f" {int(semantic_label[index])}"
                    f" {int(semantic_id[index])}"
                    f" {int(instance_id[index])}"
                    f" {float(confidence[index]):.6f}"
                )
            fh.write(line + "\n")


def _save_pcd_ascii(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    semantics: dict[str, np.ndarray] | None = None,
) -> None:
    semantic_label = None if semantics is None else np.asarray(semantics.get("semantic_label", []), dtype=np.int32)
    semantic_id = None if semantics is None else np.asarray(semantics.get("semantic_id", []), dtype=np.int32)
    instance_id = None if semantics is None else np.asarray(semantics.get("instance_id", []), dtype=np.int32)
    confidence = None if semantics is None else np.asarray(semantics.get("confidence", []), dtype=np.float32)
    has_semantics = (
        semantic_label is not None
        and semantic_id is not None
        and instance_id is not None
        and confidence is not None
        and len(semantic_label) == len(points)
        and len(semantic_id) == len(points)
        and len(instance_id) == len(points)
        and len(confidence) == len(points)
    )
    fields = ["x", "y", "z", "rgb"]
    sizes = ["4", "4", "4", "4"]
    types = ["F", "F", "F", "U"]
    counts = ["1", "1", "1", "1"]
    if has_semantics:
        fields.extend(["semantic_label", "semantic_id", "instance_id", "confidence"])
        sizes.extend(["4", "4", "4", "4"])
        types.extend(["I", "I", "I", "F"])
        counts.extend(["1", "1", "1", "1"])

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as fh:
        fh.write("# .PCD v0.7 - Point Cloud Data file format\n")
        fh.write("VERSION 0.7\n")
        fh.write(f"FIELDS {' '.join(fields)}\n")
        fh.write(f"SIZE {' '.join(sizes)}\n")
        fh.write(f"TYPE {' '.join(types)}\n")
        fh.write(f"COUNT {' '.join(counts)}\n")
        fh.write(f"WIDTH {len(points)}\n")
        fh.write("HEIGHT 1\n")
        fh.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        fh.write(f"POINTS {len(points)}\n")
        fh.write("DATA ascii\n")
        for index, (point, color) in enumerate(zip(points, colors)):
            rgb = (int(color[0]) << 16) | (int(color[1]) << 8) | int(color[2])
            line = f"{point[0]:.4f} {point[1]:.4f} {point[2]:.4f} {rgb}"
            if has_semantics:
                line += (
                    f" {int(semantic_label[index])}"
                    f" {int(semantic_id[index])}"
                    f" {int(instance_id[index])}"
                    f" {float(confidence[index]):.6f}"
                )
            fh.write(line + "\n")


def _resize_label_map(cv2, values: np.ndarray, shape: tuple[int, int], dtype) -> np.ndarray:
    if values.shape[:2] == shape:
        return values.astype(dtype, copy=False)
    resized = cv2.resize(values.astype(dtype, copy=False), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return resized.astype(dtype, copy=False)


def _semantic_payload_from_mask(
    cv2,
    object_mask_result: dict[str, Any],
    shape: tuple[int, int],
    ys: np.ndarray,
    xs: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, str]]:
    metadata = dict(object_mask_result.get("metadata", {}))
    paths = dict(object_mask_result.get("paths", {}))
    semantic_map = np.zeros(shape, dtype=np.int32)
    instance_map = np.zeros(shape, dtype=np.int32)
    confidence_map = np.zeros(shape, dtype=np.float32)

    try:
        if paths.get("semantic_map"):
            semantic_map = _resize_label_map(cv2, np.load(paths["semantic_map"]), shape, np.int32)
        elif (
            metadata.get("status") in {"ok", "empty"}
            and bool(metadata.get("enabled", False))
            and np.asarray(object_mask_result.get("mask", np.zeros(shape, dtype=bool)), dtype=bool).shape == shape
        ):
            semantic_map = np.asarray(object_mask_result["mask"], dtype=np.int32)
        if paths.get("instance_map"):
            instance_map = _resize_label_map(cv2, np.load(paths["instance_map"]), shape, np.int32)
        elif np.any(semantic_map):
            instance_map = semantic_map.astype(np.int32)
        if paths.get("confidence_map"):
            confidence_map = _resize_label_map(cv2, np.load(paths["confidence_map"]), shape, np.float32)
        elif np.any(semantic_map):
            confidence_map = np.where(semantic_map > 0, 1.0, 0.0).astype(np.float32)
    except Exception as exc:
        metadata["semantic_projection_warning"] = str(exc)
        semantic_map = np.zeros(shape, dtype=np.int32)
        instance_map = np.zeros(shape, dtype=np.int32)
        confidence_map = np.zeros(shape, dtype=np.float32)

    labels = metadata.get("semantic_labels")
    if not isinstance(labels, list):
        labels = [{"semantic_id": 0, "label": "background"}]
        prompt = str(metadata.get("prompt", "") or "").strip()
        if prompt and np.any(semantic_map > 0):
            labels.append({"semantic_id": 1, "label": prompt})
    labels_by_id: dict[int, str] = {}
    for item in labels:
        try:
            labels_by_id[int(item.get("semantic_id", 0))] = str(item.get("label", ""))
        except Exception:
            continue

    sampled_semantic_id = semantic_map[ys, xs].astype(np.int32, copy=False) if len(xs) else np.empty((0,), dtype=np.int32)
    sampled_instance_id = instance_map[ys, xs].astype(np.int32, copy=False) if len(xs) else np.empty((0,), dtype=np.int32)
    sampled_confidence = confidence_map[ys, xs].astype(np.float32, copy=False) if len(xs) else np.empty((0,), dtype=np.float32)
    # PLY/PCD fields are numeric. semantic_label is a numeric alias of semantic_id;
    # the text label mapping is written to semantic_labels.json.
    semantics = {
        "semantic_label": sampled_semantic_id,
        "semantic_id": sampled_semantic_id,
        "instance_id": sampled_instance_id,
        "confidence": sampled_confidence,
    }
    label_payload = {
        "labels": [
            {"semantic_id": int(key), "label": labels_by_id.get(key, "")}
            for key in sorted(labels_by_id)
        ],
        "instances": metadata.get("objects", []),
        "fields": {
            "semantic_label": "numeric semantic class id; see labels for text names",
            "semantic_id": "numeric semantic class id",
            "instance_id": "SAM3 instance index assigned per pixel after overlap resolution",
            "confidence": "SAM3 detection confidence sampled at the source pixel",
        },
    }
    path_payload: dict[str, str] = {}
    return semantics, label_payload, path_payload


def _validate_depth_scale_from_reference(
    depth_mm: np.ndarray,
    valid_depth: np.ndarray,
    reference_mask: np.ndarray | None,
    config: dict[str, Any],
    *,
    mask_source: str = "valid_depth",
) -> dict[str, Any]:
    enabled = _config_bool(config, "depth_scale_validation_enabled", False)
    reference = _config_float(config, "depth_scale_reference_distance_mm", 0.0, 0.0)
    tolerance_mm = _config_float(config, "depth_scale_validation_tolerance_mm", 5.0, 0.0)
    tolerance_percent = _config_float(config, "depth_scale_validation_tolerance_percent", 0.5, 0.0)
    payload: dict[str, Any] = {
        "enabled": bool(enabled),
        "reference_distance_mm": float(reference),
        "tolerance_mm": float(tolerance_mm),
        "tolerance_percent": float(tolerance_percent),
        "status": "disabled",
    }
    if not enabled:
        return payload
    if reference <= 0:
        payload.update({"status": "invalid_config", "error": "depth_scale_reference_distance_mm must be positive."})
        return payload

    mask = np.asarray(valid_depth, dtype=bool) & np.isfinite(depth_mm) & (np.asarray(depth_mm) > 0)
    if reference_mask is not None:
        ref = np.asarray(reference_mask, dtype=bool)
        if ref.shape == mask.shape:
            mask &= ref
            payload["mask_source"] = mask_source
        else:
            payload["mask_source"] = "valid_depth_shape_mismatch"
    else:
        payload["mask_source"] = "valid_depth"

    values = np.asarray(depth_mm, dtype=np.float32)[mask]
    if values.size == 0:
        payload.update({"status": "no_valid_samples", "sample_count": 0})
        return payload

    median_depth = float(np.percentile(values, 50))
    mean_depth = float(np.mean(values))
    error_mm = median_depth - float(reference)
    error_percent = 100.0 * error_mm / max(float(reference), 1e-6)
    abs_error = abs(error_mm)
    pass_mm = abs_error <= tolerance_mm
    pass_percent = abs(error_percent) <= tolerance_percent
    payload.update(
        {
            "status": "pass" if pass_mm and pass_percent else "fail",
            "sample_count": int(values.size),
            "mean_depth_mm": mean_depth,
            "median_depth_mm": median_depth,
            "p05_depth_mm": float(np.percentile(values, 5)),
            "p95_depth_mm": float(np.percentile(values, 95)),
            "error_mm": float(error_mm),
            "abs_error_mm": float(abs_error),
            "error_percent": float(error_percent),
            "pass_tolerance_mm": bool(pass_mm),
            "pass_tolerance_percent": bool(pass_percent),
        }
    )
    return payload


def _build_world_coordinate_payload(
    cv2,
    image,
    points: np.ndarray,
    depth_mm: np.ndarray,
    valid_depth: np.ndarray,
    p1: np.ndarray,
    focal: float,
    scale: float,
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    enabled = _config_bool(config, "world_coordinate_enabled", False)
    metadata: dict[str, Any] = {
        "enabled": bool(enabled),
        "status": "disabled",
        "coordinate_definition": {
            "source": "camera",
            "world_axes": "identity unless a fixed target is detected; units are millimeters",
        },
    }
    if not enabled:
        return {"points_world": None, "metadata": metadata, "metadata_path": ""}

    reference_result = _run_sam3_object_mask(
        cv2,
        image,
        output_dir,
        config,
        role="world_reference",
        prompt=str(config.get("world_reference_prompt", "fixed target")),
        required=_config_bool(config, "world_reference_required", False),
    )
    ref_mask = np.asarray(reference_result.get("mask", np.zeros(image.shape[:2], dtype=bool)), dtype=bool)
    ref_metadata = dict(reference_result.get("metadata", {}))
    min_points = _config_int(config, "world_reference_min_points", 200, 3)
    metadata["reference_target"] = ref_metadata
    if not bool(ref_metadata.get("enabled", False)):
        metadata.update({"status": "sam3_disabled", "reference_point_count": 0})
        return {"points_world": None, "metadata": metadata, "metadata_path": str(output_dir / "world_coordinate_system.json")}

    final_points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if final_points.size == 0:
        metadata.update({"status": "no_output_points", "reference_point_count": 0})
        return {"points_world": None, "metadata": metadata, "metadata_path": str(output_dir / "world_coordinate_system.json")}

    if bool(reference_result.get("usable", False)) and ref_mask.shape == image.shape[:2]:
        reference_valid = np.asarray(valid_depth, dtype=bool) & ref_mask & np.isfinite(depth_mm) & (np.asarray(depth_mm) > 0)
        ref_ys, ref_xs = np.nonzero(reference_valid)
    else:
        ref_ys = np.empty((0,), dtype=np.int64)
        ref_xs = np.empty((0,), dtype=np.int64)

    if len(ref_xs) < min_points:
        metadata.update(
            {
                "status": "reference_insufficient",
                "reference_point_count": int(len(ref_xs)),
                "min_points": int(min_points),
                "note": "SAM3 fixed-target detection did not overlap enough valid pre-object-mask depth pixels.",
            }
        )
        return {"points_world": None, "metadata": metadata, "metadata_path": str(output_dir / "world_coordinate_system.json")}

    z = np.asarray(depth_mm, dtype=np.float32)[ref_ys, ref_xs]
    cx = float(p1[0, 2]) * float(scale)
    cy = float(p1[1, 2]) * float(scale)
    ref_x = (ref_xs.astype(np.float32) - cx) * z / max(float(focal), 1e-6)
    ref_y = (ref_ys.astype(np.float32) - cy) * z / max(float(focal), 1e-6)
    ref_points = np.column_stack([ref_x, ref_y, z]).astype(np.float32)
    origin = np.median(ref_points, axis=0).astype(np.float32)
    points_world = (final_points - origin.reshape(1, 3)).astype(np.float32)
    metadata.update(
        {
            "status": "origin_from_sam3_fixed_target",
            "origin_camera_mm": [float(v) for v in origin.tolist()],
            "reference_point_count": int(len(ref_xs)),
            "transform_camera_to_world": {
                "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "t_mm": [float(-origin[0]), float(-origin[1]), float(-origin[2])],
            },
        }
    )
    return {"points_world": points_world, "metadata": metadata, "metadata_path": str(output_dir / "world_coordinate_system.json")}


def _run_sam3_object_mask(
    cv2,
    image,
    output_dir: Path,
    config: dict[str, Any],
    *,
    role: str = "object",
    prompt: str | None = None,
    required: bool | None = None,
) -> dict[str, Any]:
    role_name = re.sub(r"[^A-Za-z0-9_]+", "_", str(role or "object")).strip("_") or "object"
    prompt_text = str(prompt if prompt is not None else config.get("sam3_prompt", "object")).strip() or "object"
    is_required = bool(config.get("sam3_required", False) if required is None else required)
    metadata: dict[str, Any] = {
        "enabled": bool(config.get("sam3_segmentation", True)),
        "filter_valid_depth": bool(config.get("sam3_filter_valid_depth", True)),
        "status": "disabled",
        "role": role_name,
        "prompt": prompt_text,
        "mask_pixels": 0,
        "mask_ratio": 0.0,
    }
    input_path = output_dir / f"sam3_input_{role_name}_left_rectified.png"
    mask_path = output_dir / f"{role_name}_mask.png"
    mask_npy_path = output_dir / f"{role_name}_mask.npy"
    instance_map_path = output_dir / f"{role_name}_instance_map.npy"
    semantic_map_path = output_dir / f"{role_name}_semantic_map.npy"
    confidence_map_path = output_dir / f"{role_name}_semantic_confidence.npy"
    label_map_path = output_dir / f"{role_name}_semantic_labels.json"
    preview_path = output_dir / f"{role_name}_mask_preview.png"
    metadata_path = output_dir / f"{role_name}_mask_metadata.json"

    def write_failed_outputs(status: str, error: str = "") -> dict[str, Any]:
        metadata.update({"status": status})
        if error:
            metadata["error"] = error
        output_dir.mkdir(parents=True, exist_ok=True)
        empty_mask = np.zeros(image.shape[:2], dtype=bool)
        try:
            if not input_path.exists():
                _write_image(cv2, input_path, image)
        except Exception:
            pass
        np.save(mask_npy_path, empty_mask.astype(np.uint8))
        np.save(instance_map_path, np.zeros(image.shape[:2], dtype=np.int32))
        np.save(semantic_map_path, np.zeros(image.shape[:2], dtype=np.int32))
        np.save(confidence_map_path, np.zeros(image.shape[:2], dtype=np.float32))
        _write_image(cv2, mask_path, empty_mask.astype(np.uint8) * 255)
        _write_image(cv2, preview_path, _mask_preview_image(cv2, image, empty_mask))
        _write_json(
            label_map_path,
            {
                "labels": [{"semantic_id": 0, "label": "background"}],
                "instances": [],
            },
        )
        _write_json(metadata_path, metadata)
        return {
            "mask": empty_mask,
            "metadata": metadata,
            "paths": {
                "input": str(input_path),
                "mask": str(mask_path),
                "mask_npy": str(mask_npy_path),
                "instance_map": str(instance_map_path),
                "semantic_map": str(semantic_map_path),
                "confidence_map": str(confidence_map_path),
                "label_map": str(label_map_path),
                "preview": str(preview_path),
                "metadata": str(metadata_path),
            },
            "usable": False,
        }

    if not bool(config.get("sam3_segmentation", True)):
        return write_failed_outputs("disabled")

    sam3_root = Path(str(config.get("sam3_root", r"D:\SAM3") or r"D:\SAM3"))
    if not sam3_root.is_dir():
        metadata.update({"status": "skipped", "error": f"SAM3 root not found: {sam3_root}"})
        if is_required:
            raise CalibrationError(metadata["error"])
        return write_failed_outputs("skipped", metadata["error"])

    if not SAM3_MASK_SCRIPT.is_file():
        metadata.update({"status": "skipped", "error": f"SAM3 adapter script not found: {SAM3_MASK_SCRIPT}"})
        if is_required:
            raise CalibrationError(metadata["error"])
        return write_failed_outputs("skipped", metadata["error"])

    python_exe = _resolve_sam3_python(config)
    _write_image(cv2, input_path, image)

    base_command = [
        python_exe,
        str(SAM3_MASK_SCRIPT),
        "--image",
        str(input_path),
        "--output-mask",
        str(mask_path),
        "--output-json",
        str(metadata_path),
        "--output-instance-map",
        str(instance_map_path),
        "--output-semantic-map",
        str(semantic_map_path),
        "--output-confidence-map",
        str(confidence_map_path),
        "--output-label-map",
        str(label_map_path),
        "--sam3-root",
        str(sam3_root),
        "--prompt",
        prompt_text,
        "--threshold",
        str(float(config.get("sam3_confidence_threshold", 0.25))),
        "--top-k",
        str(int(config.get("sam3_top_k", 50))),
        "--resolution",
        str(int(config.get("sam3_resolution", 1008))),
        "--selection",
        str(config.get("sam3_mask_selection", "union")),
        "--device",
        str(config.get("sam3_device", "auto")),
    ]
    checkpoint = str(config.get("sam3_checkpoint", "") or "").strip()
    if checkpoint:
        base_command.extend(["--checkpoint", checkpoint])

    def run_command(command: list[str]):
        import subprocess

        return subprocess.run(
            command,
            cwd=str(sam3_root),
            capture_output=True,
            text=True,
            timeout=int(config.get("sam3_timeout_seconds", 600)),
            check=False,
        )

    try:
        completed = run_command(base_command)
        stderr_text = completed.stderr or ""
        stdout_text = completed.stdout or ""
        cuda_oom = (
            completed.returncode != 0
            and str(config.get("sam3_device", "auto")).lower() == "auto"
            and ("outofmemoryerror" in stderr_text.lower() or "cuda out of memory" in stderr_text.lower())
        )
        if cuda_oom:
            cpu_command = list(base_command)
            device_index = cpu_command.index("--device") + 1
            cpu_command[device_index] = "cpu"
            completed_cpu = run_command(cpu_command)
            metadata["cuda_retry"] = {
                "triggered": True,
                "first_returncode": int(completed.returncode),
                "first_stdout": stdout_text[-4000:],
                "first_stderr": stderr_text[-4000:],
                "retry_device": "cpu",
                "retry_returncode": int(completed_cpu.returncode),
            }
            completed = completed_cpu
    except Exception as exc:
        metadata.update({"status": "failed", "error": str(exc), "python": python_exe})
        if is_required:
            raise CalibrationError(f"SAM3 object mask failed: {exc}") from exc
        return write_failed_outputs("failed", str(exc))

    metadata.update(
        {
            "status": "ok" if completed.returncode == 0 else "failed",
            "returncode": int(completed.returncode),
            "python": python_exe,
            "sam3_root": str(sam3_root),
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
        }
    )
    if completed.returncode != 0:
        metadata["error"] = completed.stderr.strip() or completed.stdout.strip() or "SAM3 subprocess failed."
        if is_required:
            raise CalibrationError(f"SAM3 object mask failed: {metadata['error']}")
        return write_failed_outputs("failed", metadata["error"])

    if metadata_path.exists():
        try:
            with metadata_path.open("r", encoding="utf-8") as fh:
                metadata.update(json.load(fh))
        except Exception as exc:
            metadata["metadata_read_error"] = str(exc)

    if not mask_path.exists():
        metadata.update({"status": "failed", "error": f"SAM3 did not write mask: {mask_path}"})
        if is_required:
            raise CalibrationError(metadata["error"])
        return write_failed_outputs("failed", metadata["error"])

    mask_image = _read_gray(cv2, mask_path)
    if mask_image.shape[:2] != image.shape[:2]:
        mask_image = cv2.resize(mask_image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    object_mask = mask_image > 0

    erode_pixels = int(config.get("sam3_erode_pixels", 0))
    dilate_pixels = int(config.get("sam3_dilate_pixels", 0))
    if erode_pixels > 0 and np.any(object_mask):
        kernel = np.ones((erode_pixels * 2 + 1, erode_pixels * 2 + 1), dtype=np.uint8)
        object_mask = cv2.erode(object_mask.astype(np.uint8), kernel, iterations=1) > 0
    if dilate_pixels > 0 and np.any(object_mask):
        kernel = np.ones((dilate_pixels * 2 + 1, dilate_pixels * 2 + 1), dtype=np.uint8)
        object_mask = cv2.dilate(object_mask.astype(np.uint8), kernel, iterations=1) > 0

    mask_pixels = int(np.count_nonzero(object_mask))
    metadata.update(
        {
            "enabled": True,
            "filter_valid_depth": bool(config.get("sam3_filter_valid_depth", True)),
            "status": "ok" if mask_pixels > 0 else "empty",
            "mask_pixels": mask_pixels,
            "mask_ratio": float(mask_pixels / max(object_mask.size, 1)),
            "erode_pixels": erode_pixels,
            "dilate_pixels": dilate_pixels,
        }
    )
    np.save(mask_npy_path, object_mask.astype(np.uint8))
    _write_image(cv2, mask_path, (object_mask.astype(np.uint8) * 255))
    _write_image(cv2, preview_path, _mask_preview_image(cv2, image, object_mask))
    _write_json(metadata_path, metadata)
    return {
        "mask": object_mask,
        "metadata": metadata,
        "usable": bool(mask_pixels > 0),
        "paths": {
            "input": str(input_path),
            "mask": str(mask_path),
            "mask_npy": str(mask_npy_path),
            "instance_map": str(instance_map_path),
            "semantic_map": str(semantic_map_path),
            "confidence_map": str(confidence_map_path),
            "label_map": str(label_map_path),
            "preview": str(preview_path),
            "metadata": str(metadata_path),
        },
    }


def _basic_stats(values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p01": None,
            "p05": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    percentiles = np.percentile(finite, [1, 5, 25, 50, 75, 95, 99])
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "p01": float(percentiles[0]),
        "p05": float(percentiles[1]),
        "p25": float(percentiles[2]),
        "median": float(percentiles[3]),
        "p75": float(percentiles[4]),
        "p95": float(percentiles[5]),
        "p99": float(percentiles[6]),
        "max": float(np.max(finite)),
    }


def _point_cloud_outlier_metrics(points: np.ndarray) -> dict[str, Any]:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if len(pts) < 16:
        return {
            "method": "robust_distance_iqr",
            "point_count": int(len(pts)),
            "outlier_count": 0,
            "outlier_ratio": 0.0,
            "threshold_mm": None,
        }
    center = np.median(pts, axis=0)
    distance = np.linalg.norm(pts - center, axis=1)
    q1, q3 = np.percentile(distance, [25, 75])
    iqr = max(float(q3 - q1), 1e-6)
    threshold = float(q3 + 3.0 * iqr)
    outliers = distance > threshold
    return {
        "method": "robust_distance_iqr",
        "point_count": int(len(pts)),
        "outlier_count": int(np.count_nonzero(outliers)),
        "outlier_ratio": float(np.count_nonzero(outliers) / max(len(pts), 1)),
        "threshold_mm": threshold,
    }


def _calibration_depth_link_metrics(result: dict[str, Any]) -> dict[str, Any]:
    stereo = result.get("stereo", {})
    baseline = float(stereo.get("baseline_mm", 0.0) or 0.0)
    rectification = stereo.get("rectification", {})
    focal = None
    try:
        p1 = np.asarray(rectification.get("P1", []), dtype=np.float64)
        if p1.shape[0] >= 1 and p1.shape[1] >= 1:
            focal = abs(float(p1[0, 0]))
    except Exception:
        focal = None
    stereo_rms = float(stereo.get("rms_reprojection_error_px", 0.0) or 0.0)
    disparity_samples = []
    try:
        cv2 = _load_cv2()
        for pair in result.get("accepted_pairs", []):
            disparity = _rectified_point_disparities(cv2, result, pair)
            disparity = disparity[np.isfinite(disparity) & (disparity > 1e-6)]
            if disparity.size:
                disparity_samples.append(disparity)
    except Exception:
        disparity_samples = []
    if disparity_samples:
        disparities = np.concatenate(disparity_samples).astype(np.float64)
        estimated_depth = (float(focal) * baseline / disparities) if focal and baseline > 0 else np.array([], dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            depth_error_from_stereo_rms = (float(focal) * baseline / np.maximum(disparities - stereo_rms, 1e-6)) - (
                float(focal) * baseline / disparities
            ) if focal and baseline > 0 else np.array([], dtype=np.float64)
        return {
            "stereo_rms_px": stereo_rms,
            "baseline_mm": baseline,
            "focal_px": focal,
            "rectified_corner_disparity_px": _basic_stats(disparities),
            "estimated_corner_depth_mm": _basic_stats(estimated_depth),
            "estimated_depth_error_from_stereo_rms_mm": _basic_stats(np.abs(depth_error_from_stereo_rms)),
        }
    return {
        "stereo_rms_px": stereo_rms,
        "baseline_mm": baseline,
        "focal_px": focal,
        "rectified_corner_disparity_px": _basic_stats(np.array([], dtype=np.float32)),
        "estimated_corner_depth_mm": _basic_stats(np.array([], dtype=np.float32)),
        "estimated_depth_error_from_stereo_rms_mm": _basic_stats(np.array([], dtype=np.float32)),
    }


def _evaluate_reconstruction_quality(
    result: dict[str, Any],
    arrays: dict[str, Any],
    points: np.ndarray,
) -> dict[str, Any]:
    disparity = np.asarray(arrays["disparity"], dtype=np.float32)
    confidence = np.asarray(arrays["confidence"], dtype=np.float32)
    depth_mm = np.asarray(arrays["depth_mm"], dtype=np.float32)
    valid_disparity = np.asarray(arrays["valid_disparity"], dtype=bool)
    valid_depth = np.asarray(arrays["valid_depth"], dtype=bool)
    object_mask = np.asarray(arrays.get("object_mask", np.ones_like(valid_depth)), dtype=bool)
    if object_mask.shape != valid_depth.shape:
        object_mask = np.ones_like(valid_depth, dtype=bool)
    valid_depth_before_object_mask = np.asarray(arrays.get("valid_depth_before_object_mask", valid_depth), dtype=bool)
    if valid_depth_before_object_mask.shape != valid_depth.shape:
        valid_depth_before_object_mask = valid_depth
    total_pixels = int(disparity.size)
    finite_disparity = np.isfinite(disparity)
    finite_depth = np.isfinite(depth_mm) & (depth_mm > 0)
    confidence_valid = confidence[finite_disparity]
    quality = {
        "image_size": [int(disparity.shape[1]), int(disparity.shape[0])],
        "total_pixels": total_pixels,
        "valid_disparity_pixels": int(np.count_nonzero(valid_disparity)),
        "valid_disparity_ratio": float(np.count_nonzero(valid_disparity) / max(total_pixels, 1)),
        "valid_depth_pixels": int(np.count_nonzero(valid_depth)),
        "valid_depth_ratio": float(np.count_nonzero(valid_depth) / max(total_pixels, 1)),
        "valid_depth_pixels_before_object_mask": int(np.count_nonzero(valid_depth_before_object_mask)),
        "valid_depth_ratio_before_object_mask": float(np.count_nonzero(valid_depth_before_object_mask) / max(total_pixels, 1)),
        "object_mask_pixels": int(np.count_nonzero(object_mask)),
        "object_mask_ratio": float(np.count_nonzero(object_mask) / max(total_pixels, 1)),
        "object_mask_valid_depth_kept_ratio": float(
            np.count_nonzero(valid_depth) / max(int(np.count_nonzero(valid_depth_before_object_mask)), 1)
        ),
        "finite_disparity_ratio_before_filter": float(np.count_nonzero(finite_disparity) / max(total_pixels, 1)),
        "finite_depth_ratio_before_filter": float(np.count_nonzero(finite_depth) / max(total_pixels, 1)),
        "confidence": {
            "all_finite_disparity": _basic_stats(confidence_valid),
            "valid_depth": _basic_stats(confidence[valid_depth]),
            "below_threshold_ratio": float(
                np.count_nonzero(confidence_valid < float(arrays["confidence_threshold"])) / max(int(confidence_valid.size), 1)
            )
            if confidence_valid.size
            else None,
        },
        "disparity_px": {
            "valid": _basic_stats(disparity[valid_disparity]),
            "finite_before_filter": _basic_stats(disparity[finite_disparity]),
        },
        "depth_mm": {
            "valid": _basic_stats(depth_mm[valid_depth]),
            "finite_before_filter": _basic_stats(depth_mm[finite_depth]),
        },
        "point_cloud": _point_cloud_outlier_metrics(points),
        "calibration_depth_link": _calibration_depth_link_metrics(result),
    }
    return quality


def camera_points_to_display(points: np.ndarray) -> np.ndarray:
    """Map OpenCV camera coordinates to the MATLAB-style 3D display axes."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    return np.column_stack((pts[:, 0], pts[:, 2], -pts[:, 1]))


def _camera_points_to_plot(points: np.ndarray) -> np.ndarray:
    return camera_points_to_display(points)


def _set_axes_equal(ax, points: np.ndarray) -> None:
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    centers = (mins + maxs) / 2.0
    radius = max(float(np.max(maxs - mins)) / 2.0, 1.0)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def _plot_heatmap(path: Path, values: np.ndarray, title: str, xlabel: str, ylabel: str, cmap: str = "viridis") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.4, 5.8), dpi=220)
    masked = np.ma.masked_invalid(values)
    im = ax.imshow(masked, cmap=cmap, interpolation="nearest", origin="upper")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cbar = fig.colorbar(im, ax=ax, shrink=0.92)
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_error_hist(path: Path, values: np.ndarray, title: str, xlabel: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=220)
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    ax.hist(finite, bins=min(40, max(10, int(math.sqrt(max(len(finite), 1))))), color="#2b8cbe", edgecolor="white", linewidth=0.5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_depth_error_curve(path: Path, depth_mm: np.ndarray, disparity_error_px: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = np.isfinite(depth_mm) & np.isfinite(disparity_error_px) & (depth_mm > 0)
    depth = np.asarray(depth_mm, dtype=float)[valid]
    error = np.asarray(disparity_error_px, dtype=float)[valid]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.4, 5.2), dpi=220)
    if len(depth) == 0:
        ax.text(0.5, 0.5, "No valid depth samples", ha="center", va="center")
    else:
        bins = np.linspace(np.percentile(depth, 2), np.percentile(depth, 98), 18)
        bins = np.unique(bins)
        if len(bins) < 3:
            bins = np.linspace(depth.min(), depth.max() + 1e-6, 8)
        centers = []
        mean_error = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (depth >= lo) & (depth < hi)
            if np.any(mask):
                centers.append((lo + hi) / 2.0)
                mean_error.append(float(np.mean(np.abs(error[mask]))))
        if centers:
            ax.plot(centers, mean_error, marker="o", linewidth=1.8, color="#d95f0e")
            ax.set_xscale("log")
    ax.set_title("Depth error curve")
    ax.set_xlabel("Depth (mm, log scale)")
    ax.set_ylabel("Mean absolute disparity residual (px)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_board_coverage_heatmap(cv2, path: Path, result: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    height = int(result["image_size"][1])
    width = int(result["image_size"][0])
    left_density = np.zeros((height, width), dtype=np.float32)
    right_density = np.zeros((height, width), dtype=np.float32)
    for pair in result.get("accepted_pairs", []):
        for side, density in (("left", left_density), ("right", right_density)):
            points = np.asarray(pair.get(f"{side}_points", []), dtype=np.float32).reshape(-1, 2)
            if len(points) == 0:
                continue
            xs = np.clip(np.rint(points[:, 0]).astype(np.int32), 0, width - 1)
            ys = np.clip(np.rint(points[:, 1]).astype(np.int32), 0, height - 1)
            np.add.at(density, (ys, xs), 1.0)

    left_density = np.log1p(cv2.GaussianBlur(left_density, (0, 0), 12))
    right_density = np.log1p(cv2.GaussianBlur(right_density, (0, 0), 12))
    vmax = max(float(left_density.max()), float(right_density.max()), 1e-6)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.0), dpi=220, constrained_layout=True)
    for ax, density, title in (
        (axes[0], left_density, "Left camera board coverage"),
        (axes[1], right_density, "Right camera board coverage"),
    ):
        im = ax.imshow(density, cmap="magma", origin="upper", vmin=0, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("X (px)")
        ax.set_ylabel("Y (px)")
        fig.colorbar(im, ax=ax, shrink=0.88)
    fig.savefig(path)
    plt.close(fig)


def _save_reprojection_error_distribution(path: Path, result: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    left_errors: list[float] = []
    right_errors: list[float] = []
    for pair in result.get("accepted_pairs", []):
        left_points = np.asarray(pair.get("left_points", []), dtype=np.float32).reshape(-1, 2)
        left_reprojected = np.asarray(pair.get("left_reprojected_points", []), dtype=np.float32).reshape(-1, 2)
        right_points = np.asarray(pair.get("right_points", []), dtype=np.float32).reshape(-1, 2)
        right_reprojected = np.asarray(pair.get("right_reprojected_points", []), dtype=np.float32).reshape(-1, 2)
        if len(left_points) == len(left_reprojected) and len(left_points) > 0:
            left_errors.extend(np.linalg.norm(left_points - left_reprojected, axis=1).astype(float).tolist())
        if len(right_points) == len(right_reprojected) and len(right_points) > 0:
            right_errors.extend(np.linalg.norm(right_points - right_reprojected, axis=1).astype(float).tolist())

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), dpi=220, constrained_layout=True)
    for ax, values, title, color in (
        (axes[0], left_errors, "Left reprojection error distribution", "#1f77b4"),
        (axes[1], right_errors, "Right reprojection error distribution", "#d62728"),
    ):
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if len(finite) == 0:
            ax.text(0.5, 0.5, "No valid samples", ha="center", va="center")
        else:
            bins = min(60, max(20, int(math.sqrt(len(finite)))))
            ax.hist(finite, bins=bins, color=color, alpha=0.85, edgecolor="white", linewidth=0.4)
            ax.axvline(float(np.mean(finite)), color="black", linestyle="--", linewidth=1.2, label="Mean")
            ax.axvline(float(np.median(finite)), color="#444444", linestyle=":", linewidth=1.2, label="Median")
            ax.legend(loc="upper right", fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("Error (px)")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.2)
    fig.savefig(path)
    plt.close(fig)


def _save_depth_error_curve(path: Path, cv2, result: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rectification = result.get("stereo", {}).get("rectification")
    if not rectification:
        return
    K1 = np.asarray(result["left"]["camera_matrix"], dtype=np.float64)
    D1 = np.asarray(result["left"]["distortion_coefficients"], dtype=np.float64).reshape(-1)
    K2 = np.asarray(result["right"]["camera_matrix"], dtype=np.float64)
    D2 = np.asarray(result["right"]["distortion_coefficients"], dtype=np.float64).reshape(-1)
    R1 = np.asarray(rectification["R1"], dtype=np.float64)
    R2 = np.asarray(rectification["R2"], dtype=np.float64)
    P1 = np.asarray(rectification["P1"], dtype=np.float64)
    P2 = np.asarray(rectification["P2"], dtype=np.float64)
    fx = float(P1[0, 0])
    baseline = abs(float(P2[0, 3] / P2[0, 0])) if abs(float(P2[0, 0])) > 1e-12 else float(result["stereo"]["baseline_mm"])

    true_depths: list[np.ndarray] = []
    depth_errors: list[np.ndarray] = []
    for pair in result.get("accepted_pairs", []):
        pose = pair.get("board_pose_left_camera", {})
        rvec = np.asarray(pose.get("rotation_vector", []), dtype=np.float64).reshape(-1)
        tvec = np.asarray(pose.get("translation_vector_mm", []), dtype=np.float64).reshape(-1)
        obj = np.asarray(pair.get("object_points", []), dtype=np.float64).reshape(-1, 3)
        left_points = np.asarray(pair.get("left_points", []), dtype=np.float32).reshape(-1, 1, 2)
        right_points = np.asarray(pair.get("right_points", []), dtype=np.float32).reshape(-1, 1, 2)
        if len(obj) == 0 or len(left_points) == 0 or len(right_points) == 0 or len(rvec) != 3 or len(tvec) != 3:
            continue
        rotation, _ = cv2.Rodrigues(rvec.reshape(3, 1))
        cam_points = (rotation @ obj.T + tvec.reshape(3, 1)).T
        true_depth = cam_points[:, 2]
        left_rect = cv2.undistortPoints(left_points, K1, D1, R=R1, P=P1).reshape(-1, 2)
        right_rect = cv2.undistortPoints(right_points, K2, D2, R=R2, P=P2).reshape(-1, 2)
        disparity = left_rect[:, 0] - right_rect[:, 0]
        valid = np.isfinite(true_depth) & np.isfinite(disparity) & (true_depth > 0) & (disparity > 1e-6)
        if not np.any(valid):
            continue
        estimated_depth = fx * baseline / disparity[valid]
        true_depths.append(true_depth[valid])
        depth_errors.append(np.abs(estimated_depth - true_depth[valid]))

    fig, ax = plt.subplots(figsize=(8.0, 5.6), dpi=220)
    if not true_depths:
        ax.text(0.5, 0.5, "No valid depth samples", ha="center", va="center")
    else:
        depth = np.concatenate(true_depths)
        error = np.concatenate(depth_errors)
        finite = np.isfinite(depth) & np.isfinite(error)
        depth = depth[finite]
        error = error[finite]
        if len(depth) > 0:
            ax.scatter(depth, error, s=10, alpha=0.15, color="#9e9ac8", edgecolors="none")
            min_depth = max(float(np.percentile(depth, 2)), 1e-3)
            max_depth = max(float(np.percentile(depth, 98)), min_depth * 1.1)
            bins = np.geomspace(min_depth, max_depth, 14)
            centers = []
            means = []
            for lo, hi in zip(bins[:-1], bins[1:]):
                mask = (depth >= lo) & (depth < hi)
                if np.any(mask):
                    centers.append(math.sqrt(lo * hi))
                    means.append(float(np.mean(error[mask])))
            if centers:
                ax.plot(centers, means, color="#d95f0e", linewidth=2.0, marker="o", markersize=4, label="Mean abs error")
                ax.legend(loc="upper left")
            ax.set_xscale("log")
    ax.set_title("Depth error curve")
    ax.set_xlabel("True depth (mm, log scale)")
    ax.set_ylabel("Absolute depth error (mm)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_point_cloud_plot(path: Path, points: np.ndarray, colors: np.ndarray, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    if len(points) == 0:
        fig = plt.figure(figsize=(7, 5), dpi=140)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title(title)
        ax.text(0, 0, 0, "no valid points")
    else:
        count = min(len(points), 40000)
        indices = np.linspace(0, len(points) - 1, count, dtype=int)
        plot_points = _camera_points_to_plot(points[indices])
        plot_colors = np.asarray(colors[indices], dtype=float) / 255.0
        fig = plt.figure(figsize=(8.4, 6.2), dpi=220)
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(plot_points[:, 0], plot_points[:, 1], plot_points[:, 2], c=plot_colors, s=1.0, depthshade=False)
        ax.set_title(title)
        ax.set_xlabel("X right (mm)")
        ax.set_ylabel("Z forward (mm)")
        ax.set_zlabel("-Y up (mm)")
        _set_axes_equal(ax, plot_points)
        ax.view_init(elev=22, azim=-62)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_camera_pose_plot(path: Path, result: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rotation = np.asarray(result["stereo"]["rotation_matrix"], dtype=float)
    translation = np.asarray(result["stereo"]["translation_vector"], dtype=float).reshape(3)
    right_center = -rotation.T @ translation
    baseline = max(float(result["stereo"].get("baseline_mm", np.linalg.norm(right_center))), 1.0)
    axis_len = baseline * 0.22

    fig = plt.figure(figsize=(8.2, 6.0), dpi=220)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Camera pose")
    ax.set_xlabel("X right (mm)")
    ax.set_ylabel("Z forward (mm)")
    ax.set_zlabel("-Y up (mm)")

    def draw_frame(origin: np.ndarray, axes: np.ndarray, label: str) -> list[np.ndarray]:
        colors = ["red", "green", "blue"]
        labels = ["x", "y", "z"]
        plotted = [origin.reshape(1, 3)]
        origin_plot = _camera_points_to_plot(origin.reshape(1, 3))[0]
        ax.scatter([origin_plot[0]], [origin_plot[1]], [origin_plot[2]], s=50)
        ax.text(origin_plot[0], origin_plot[1], origin_plot[2], label)
        for col, axis_label, axis in zip(colors, labels, axes.T):
            endpoint = origin + axis * axis_len
            line = _camera_points_to_plot(np.vstack([origin, endpoint]))
            ax.plot(line[:, 0], line[:, 1], line[:, 2], color=col, linewidth=2)
            ax.text(line[-1, 0], line[-1, 1], line[-1, 2], f"{label}-{axis_label}", color=col)
            plotted.append(endpoint.reshape(1, 3))
        return plotted

    all_points = []
    all_points.extend(draw_frame(np.zeros(3), np.eye(3), "left"))
    all_points.extend(draw_frame(right_center, rotation.T, "right"))
    baseline_line = _camera_points_to_plot(np.vstack([np.zeros(3), right_center]))
    ax.plot(baseline_line[:, 0], baseline_line[:, 1], baseline_line[:, 2], color="black", linestyle="--", linewidth=1.4)

    board_centers = []
    for pair in result.get("accepted_pairs", []):
        pose = pair.get("board_pose_left_camera", {})
        t = np.asarray(pose.get("translation_vector_mm", []), dtype=float).reshape(-1)
        if t.size == 3:
            board_centers.append(t)
    if board_centers:
        centers = _camera_points_to_plot(np.asarray(board_centers))
        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c="gray", s=12, alpha=0.5)
        all_points.append(np.asarray(board_centers))

    _set_axes_equal(ax, _camera_points_to_plot(np.vstack(all_points)))
    ax.view_init(elev=22, azim=-58)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _save_board_pose_plot(path: Path, result: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8.4, 6.2), dpi=220)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Calibration board poses")
    ax.set_xlabel("X right (mm)")
    ax.set_ylabel("Z forward (mm)")
    ax.set_zlabel("-Y up (mm)")
    ax.scatter([0], [0], [0], c="blue", marker="o", s=55)
    ax.text(0, 0, 0, "left camera", color="blue")

    points_for_axes = [np.array([[0.0, 0.0, 0.0]])]
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(result.get("accepted_pairs", [])), 1)))
    for index, pair in enumerate(result.get("accepted_pairs", [])):
        pose = pair.get("board_pose_left_camera", {})
        rvec = np.asarray(pose.get("rotation_vector", [0, 0, 0]), dtype=float).reshape(3)
        t = np.asarray(pose.get("translation_vector_mm", [0, 0, 0]), dtype=float).reshape(3)
        obj = np.asarray(pair.get("object_points", []), dtype=float)
        if obj.size < 12:
            continue
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
        theta = float(np.linalg.norm(rvec))
        if theta > 1e-12:
            k = rvec / theta
            kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], dtype=float)
            rot = np.eye(3) + math.sin(theta) * kx + (1 - math.cos(theta)) * (kx @ kx)
            corners = corners @ rot.T
        corners = corners + t
        plot_corners = _camera_points_to_plot(corners)
        ax.plot(plot_corners[:, 0], plot_corners[:, 1], plot_corners[:, 2], color=colors[index], linewidth=1.3)
        center = _camera_points_to_plot(corners[:4].mean(axis=0).reshape(1, 3))[0]
        ax.text(center[0], center[1], center[2], str(index + 1), color=colors[index])
        points_for_axes.append(plot_corners)

    _set_axes_equal(ax, np.vstack(points_for_axes))
    ax.view_init(elev=24, azim=-58)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _make_reconstruction_montage(cv2, rectified_pair, disparity_image, depth_image, cloud_preview_path: Path, output_path: Path) -> None:
    cloud = _read_color(cv2, cloud_preview_path) if cloud_preview_path.exists() else np.zeros_like(depth_image)
    tiles = [
        _label_image(cv2, _resize_max(cv2, rectified_pair, 1200, 700), "rectified pair"),
        _label_image(cv2, _resize_max(cv2, disparity_image, 1200, 700), "disparity"),
        _label_image(cv2, _resize_max(cv2, depth_image, 1200, 700), "depth"),
        _label_image(cv2, _resize_max(cv2, cloud, 1200, 700), "point cloud"),
    ]
    tile_height = min(tile.shape[0] for tile in tiles)
    tile_width = min(tile.shape[1] for tile in tiles)
    normalized = [cv2.resize(tile, (tile_width, tile_height), interpolation=cv2.INTER_AREA) for tile in tiles]
    top = np.hstack(normalized[:2])
    bottom = np.hstack(normalized[2:])
    montage = np.vstack([top, bottom])
    _write_image(cv2, output_path, montage, quality=98)


def _compute_reconstruction_arrays(
    cv2,
    result: dict[str, Any],
    left_rectified,
    right_rectified,
    reconstruction_config: dict[str, Any] | None = None,
    point_disparities: np.ndarray | None = None,
) -> dict[str, Any]:
    config = _normalize_reconstruction_config(reconstruction_config)
    prompt_before_fallback = bool(config.get("prompt_before_sgbm_fallback", True))
    height, width = left_rectified.shape[:2]
    method_requested = str(config["reconstruction_method"]).lower()
    scale, resource_policy = _resolve_reconstruction_scale(config, width, height, method_requested)
    small_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    left_small = cv2.resize(left_rectified, small_size, interpolation=cv2.INTER_AREA) if scale < 1 else left_rectified.copy()
    right_small = cv2.resize(right_rectified, small_size, interpolation=cv2.INTER_AREA) if scale < 1 else right_rectified.copy()
    gray_left_raw = cv2.cvtColor(left_small, cv2.COLOR_BGR2GRAY)
    gray_right_raw = cv2.cvtColor(right_small, cv2.COLOR_BGR2GRAY)
    gray_left = cv2.equalizeHist(gray_left_raw)
    gray_right = cv2.equalizeHist(gray_right_raw)

    if point_disparities is None:
        point_disparities = np.array([], dtype=np.float32)
    min_disp, num_disp = _choose_disparity_range(point_disparities, scale, small_size[0])

    fallback_error = ""
    if bool(config.get("force_sgbm_fallback", False)):
        fallback_error = str(config.get("sgbm_fallback_reason", "") or "User approved SGBM fallback.")
        disparity_result = _compute_sgbm_disparity(cv2, gray_left, gray_right, left_small, min_disp, num_disp, config)
        disparity_result["metadata"]["fallback_from"] = "crestereo"
        disparity_result["metadata"]["fallback_reason"] = fallback_error
        disparity_result["metadata"]["fallback_user_approved"] = True
    elif method_requested in {"cres", "crestereo", "crestereo_onnx"}:
        try:
            disparity_result = _compute_crestereo_disparity(cv2, left_small, right_small, config)
        except Exception as exc:
            if not bool(config["allow_sgbm_fallback"]):
                raise
            if prompt_before_fallback:
                raise CREStereoFallbackRequired(
                    f"CREStereo failed before producing a usable disparity: {exc}",
                    stage="inference",
                ) from exc
            fallback_error = str(exc)
            disparity_result = _compute_sgbm_disparity(cv2, gray_left, gray_right, left_small, min_disp, num_disp, config)
            disparity_result["metadata"]["fallback_from"] = "crestereo"
            disparity_result["metadata"]["fallback_reason"] = fallback_error
    elif method_requested == "auto" and str(config.get("crestereo_model_path", "") or "").strip():
        try:
            disparity_result = _compute_crestereo_disparity(cv2, left_small, right_small, config)
        except Exception as exc:
            if not bool(config["allow_sgbm_fallback"]):
                raise
            if prompt_before_fallback:
                raise CREStereoFallbackRequired(
                    f"CREStereo failed before producing a usable disparity: {exc}",
                    stage="inference",
                ) from exc
            fallback_error = str(exc)
            disparity_result = _compute_sgbm_disparity(cv2, gray_left, gray_right, left_small, min_disp, num_disp, config)
            disparity_result["metadata"]["fallback_from"] = "crestereo"
            disparity_result["metadata"]["fallback_reason"] = fallback_error
    else:
        disparity_result = _compute_sgbm_disparity(cv2, gray_left, gray_right, left_small, min_disp, num_disp, config)

    raw_disparity = np.asarray(disparity_result["raw_disparity"], dtype=np.float32)
    disparity = np.asarray(disparity_result["disparity"], dtype=np.float32)
    valid = np.isfinite(disparity) & (disparity > max(0.5, float(min_disp) + 0.25))
    confidence, confidence_info = _compute_confidence_map(
        cv2,
        gray_left_raw,
        gray_right_raw,
        disparity,
        valid,
        config,
        raw_disparity=raw_disparity,
        right_disparity=disparity_result.get("right_disparity"),
        wls_confidence=disparity_result.get("wls_confidence"),
    )
    right_policy = disparity_result.get("metadata", {}).get("right_disparity_policy", {})
    lr_invalid = (
        disparity_result.get("method") == "crestereo"
        and isinstance(right_policy, dict)
        and right_policy.get("validated")
        and right_policy.get("status") != "ok"
    )
    if lr_invalid and _config_bool(config, "crestereo_lr_fail_fallback", True):
        fallback_error = (
            "CREStereo right disparity failed left-right validation. "
            + _format_right_disparity_policy(right_policy)
        ).strip()
        if bool(config.get("allow_sgbm_fallback", True)) and prompt_before_fallback:
            raise CREStereoFallbackRequired(
                fallback_error,
                stage="left_right_validation",
                right_disparity_policy=right_policy,
            )
        disparity_result = _compute_sgbm_disparity(cv2, gray_left, gray_right, left_small, min_disp, num_disp, config)
        disparity_result["metadata"]["fallback_from"] = "crestereo"
        disparity_result["metadata"]["fallback_reason"] = fallback_error
        disparity_result["metadata"]["crestereo_right_disparity_policy"] = right_policy
        raw_disparity = np.asarray(disparity_result["raw_disparity"], dtype=np.float32)
        disparity = np.asarray(disparity_result["disparity"], dtype=np.float32)
        valid = np.isfinite(disparity) & (disparity > max(0.5, float(min_disp) + 0.25))
        confidence, confidence_info = _compute_confidence_map(
            cv2,
            gray_left_raw,
            gray_right_raw,
            disparity,
            valid,
            config,
            raw_disparity=raw_disparity,
            right_disparity=disparity_result.get("right_disparity"),
            wls_confidence=disparity_result.get("wls_confidence"),
        )
    elif lr_invalid and not _config_bool(config, "crestereo_lr_fail_fallback", True):
        raise CalibrationError("CREStereo right disparity failed left-right validation; enable crestereo_lr_fail_fallback or fix the model convention.")
    confidence_threshold = float(config["confidence_threshold"])
    confidence_enabled = bool(config["confidence_filter"])
    if confidence_enabled:
        valid &= confidence >= confidence_threshold

    rectification = result["stereo"]["rectification"]
    p1 = np.asarray(rectification["P1"], dtype=np.float64)
    p2 = np.asarray(rectification["P2"], dtype=np.float64)
    focal = abs(float(p1[0, 0])) * scale
    baseline = abs(float(p2[0, 3] / p2[0, 0])) if abs(float(p2[0, 0])) > 1e-12 else float(result["stereo"]["baseline_mm"])
    with np.errstate(divide="ignore", invalid="ignore"):
        depth_mm = focal * baseline / disparity
    valid_depth = valid & np.isfinite(depth_mm) & (depth_mm > 0)
    if np.any(valid_depth):
        lo, hi = np.percentile(depth_mm[valid_depth], [1, 99])
        valid_depth &= (depth_mm >= lo) & (depth_mm <= hi)

    return {
        "config": config,
        "scale": float(scale),
        "left_small": left_small,
        "right_small": right_small,
        "raw_disparity": raw_disparity,
        "disparity": disparity,
        "confidence": confidence.astype(np.float32),
        "depth_mm": depth_mm.astype(np.float32),
        "valid_disparity": valid,
        "valid_depth": valid_depth,
        "method_requested": method_requested,
        "resource_policy": resource_policy,
        "fallback_error": fallback_error,
        "min_disparity": int(min_disp),
        "num_disparities": int(num_disp),
        "disparity_result": disparity_result,
        "confidence_info": confidence_info,
        "confidence_threshold": float(confidence_threshold),
        "confidence_enabled": bool(confidence_enabled),
        "focal_px": float(focal),
        "baseline_mm": float(baseline),
    }


def reconstruct_rectified_pair_preview(
    left_rectified,
    right_rectified,
    calibration_result: dict[str, Any],
    reconstruction_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cv2 = _load_cv2()
    arrays = _compute_reconstruction_arrays(
        cv2,
        calibration_result,
        left_rectified,
        right_rectified,
        reconstruction_config,
    )
    disparity_image, disparity_range = _colored_range_image(cv2, arrays["disparity"], arrays["valid_disparity"])
    depth_image, depth_range = _colored_range_image(cv2, arrays["depth_mm"], arrays["valid_depth"], invert=True)
    confidence_image = _confidence_image(cv2, arrays["confidence"], np.isfinite(arrays["disparity"]))
    return {
        **arrays,
        "disparity_image": disparity_image,
        "depth_image": depth_image,
        "confidence_image": confidence_image,
        "disparity_range_px": list(disparity_range) if disparity_range else None,
        "depth_range_mm": list(depth_range) if depth_range else None,
    }


def rectify_stereo_image_arrays(
    left_image,
    right_image,
    calibration_result: dict[str, Any],
):
    return StereoRectifier(calibration_result).rectify(left_image, right_image)


def _same_file_path(left: str | Path, right: str | Path) -> bool:
    try:
        return Path(left).resolve().samefile(Path(right).resolve())
    except Exception:
        return str(Path(left)).replace("\\", "/").casefold() == str(Path(right)).replace("\\", "/").casefold()


def _find_calibration_pair_for_sources(
    calibration_result: dict[str, Any],
    left_image_path: str | Path,
    right_image_path: str | Path,
) -> dict[str, Any] | None:
    for pair in calibration_result.get("accepted_pairs", []):
        if _same_file_path(pair.get("left", ""), left_image_path) and _same_file_path(pair.get("right", ""), right_image_path):
            return pair
    return None


def reconstruct_stereo_images(
    left_image_path: str | Path,
    right_image_path: str | Path,
    calibration_result_path: str | Path,
    output_dir: str | Path,
    reconstruction_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cv2 = _load_cv2()
    calibration_path = Path(calibration_result_path)
    with calibration_path.open("r", encoding="utf-8") as fh:
        calibration_result = json.load(fh)
    left_image = _read_color(cv2, left_image_path)
    right_image = _read_color(cv2, right_image_path)
    if left_image.shape[:2] != right_image.shape[:2]:
        raise CalibrationError(f"左右图像尺寸不同：{left_image.shape[1]}x{left_image.shape[0]} vs {right_image.shape[1]}x{right_image.shape[0]}")
    expected_size = tuple(map(int, calibration_result.get("image_size", [])))
    current_size = (int(left_image.shape[1]), int(left_image.shape[0]))
    if expected_size and current_size != expected_size:
        raise CalibrationError(f"图像尺寸 {current_size} 与标定尺寸 {expected_size} 不一致。")
    if not calibration_result.get("stereo", {}).get("rectification"):
        raise CalibrationError("标定结果缺少 stereoRectify 参数，无法进行独立深度重建。")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    left_rectified, right_rectified = _rectify_pair(cv2, calibration_result, left_image, right_image)
    rectified_pair_image = _draw_epipolar_pair(cv2, left_rectified, right_rectified)
    rectified_dir = output_path / "rectification"
    _write_image(cv2, rectified_dir / "left_rectified.png", _resize_max(cv2, left_rectified, 1800, 1200))
    _write_image(cv2, rectified_dir / "right_rectified.png", _resize_max(cv2, right_rectified, 1800, 1200))
    rectified_pair_path = rectified_dir / "rectified_pair.png"
    _write_image(cv2, rectified_pair_path, rectified_pair_image)

    matched_pair = _find_calibration_pair_for_sources(calibration_result, left_image_path, right_image_path)
    reconstruction = _generate_reconstruction_artifacts(
        cv2,
        calibration_result,
        matched_pair or {},
        left_rectified,
        right_rectified,
        rectified_pair_image,
        output_path,
        reconstruction_config,
    )
    reconstruction["left_source"] = str(Path(left_image_path))
    reconstruction["right_source"] = str(Path(right_image_path))
    reconstruction["calibration_result"] = str(calibration_path)
    reconstruction["rectified_pair"] = str(rectified_pair_path)
    result_path = output_path / "reconstruction_job.json"
    _write_json(result_path, reconstruction)
    reconstruction["result_json"] = str(result_path)
    return reconstruction


def _generate_reconstruction_artifacts(
    cv2,
    result: dict[str, Any],
    pair: dict[str, Any],
    left_rectified,
    right_rectified,
    rectified_pair_image,
    reconstruction_dir: Path,
    reconstruction_config: dict[str, Any] | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
    progress_base: float = 80.0,
    progress_span: float = 12.0,
) -> dict[str, Any]:
    started = time.perf_counter()
    timings: dict[str, float] = {}

    def mark_timing(name: str) -> None:
        timings[name] = round(time.perf_counter() - started, 3)

    def progress(offset_ratio: float, message: str) -> None:
        _invoke_progress(progress_callback, progress_base + progress_span * float(offset_ratio), message)

    progress(0.02, "正在计算诊断重建视差与深度...")
    point_disparities = _rectified_point_disparities(cv2, result, pair)
    arrays = _compute_reconstruction_arrays(
        cv2,
        result,
        left_rectified,
        right_rectified,
        reconstruction_config,
        point_disparities=point_disparities,
    )
    mark_timing("disparity_depth_seconds")
    config = arrays["config"]
    scale = float(arrays["scale"])
    left_small = arrays["left_small"]
    raw_disparity = arrays["raw_disparity"]
    disparity = arrays["disparity"]
    confidence = arrays["confidence"]
    valid = arrays["valid_disparity"]
    depth_mm = arrays["depth_mm"]
    valid_depth = arrays["valid_depth"]
    method_requested = arrays["method_requested"]
    fallback_error = arrays["fallback_error"]
    resource_policy = arrays["resource_policy"]
    min_disp = int(arrays["min_disparity"])
    num_disp = int(arrays["num_disparities"])
    disparity_result = arrays["disparity_result"]
    confidence_info = arrays["confidence_info"]
    confidence_threshold = float(arrays["confidence_threshold"])
    confidence_enabled = bool(arrays["confidence_enabled"])
    progress(0.28, "正在运行 SAM3 object_mask 分割...")
    object_mask_result = _run_sam3_object_mask(cv2, left_small, reconstruction_dir, config)
    mark_timing("sam3_object_mask_seconds")
    object_mask = np.asarray(object_mask_result["mask"], dtype=bool)
    object_mask_metadata = dict(object_mask_result.get("metadata", {}))
    object_mask_paths = dict(object_mask_result.get("paths", {}))
    valid_depth_before_object_mask = valid_depth.copy()
    object_mask_usable = bool(object_mask_result.get("usable", False))
    if bool(config.get("sam3_segmentation", True)) and bool(config.get("sam3_filter_valid_depth", True)) and object_mask_usable:
        if object_mask.shape[:2] != valid_depth.shape[:2]:
            object_mask = cv2.resize(object_mask.astype(np.uint8), (valid_depth.shape[1], valid_depth.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        valid_depth = valid_depth & object_mask
        arrays["valid_depth"] = valid_depth
    object_mask_metadata.update(
        {
            "valid_depth_pixels_before_object_mask": int(np.count_nonzero(valid_depth_before_object_mask)),
            "valid_depth_pixels_after_object_mask": int(np.count_nonzero(valid_depth)),
            "valid_depth_kept_ratio": float(
                np.count_nonzero(valid_depth) / max(int(np.count_nonzero(valid_depth_before_object_mask)), 1)
            ),
            "usable_for_filtering": bool(object_mask_usable),
            "semantic_point_cloud": bool(object_mask_usable),
        }
    )
    arrays["object_mask"] = object_mask
    arrays["object_mask_metadata"] = object_mask_metadata
    arrays["valid_depth_before_object_mask"] = valid_depth_before_object_mask
    if object_mask_paths.get("metadata"):
        _write_json(Path(object_mask_paths["metadata"]), object_mask_metadata)

    progress(0.42, "正在保存视差、置信度和深度矩阵...")
    disparity_image, disparity_range = _colored_range_image(cv2, disparity, valid)
    disparity_path = reconstruction_dir / "disparity_map.png"
    raw_disparity_path = reconstruction_dir / "raw_disparity.npy"
    filtered_disparity_path = reconstruction_dir / "disparity.npy"
    confidence_path = reconstruction_dir / "confidence_map.png"
    confidence_npy_path = reconstruction_dir / "confidence.npy"
    valid_disparity_npy_path = reconstruction_dir / "valid_disparity.npy"
    valid_depth_npy_path = reconstruction_dir / "valid_depth.npy"
    _write_image(cv2, disparity_path, disparity_image)
    _write_image(cv2, confidence_path, _confidence_image(cv2, confidence, np.isfinite(disparity)))
    np.save(raw_disparity_path, raw_disparity.astype(np.float32))
    np.save(filtered_disparity_path, disparity.astype(np.float32))
    np.save(confidence_npy_path, confidence.astype(np.float32))
    np.save(valid_disparity_npy_path, valid.astype(np.uint8))
    np.save(valid_depth_npy_path, valid_depth.astype(np.uint8))
    mark_timing("array_outputs_seconds")

    rectification = result["stereo"]["rectification"]
    p1 = np.asarray(rectification["P1"], dtype=np.float64)
    focal = float(arrays["focal_px"])
    depth_image, depth_range = _colored_range_image(cv2, depth_mm, valid_depth, invert=True)
    depth_path = reconstruction_dir / "depth_map.png"
    depth_npy_path = reconstruction_dir / "depth_mm.npy"
    _write_image(cv2, depth_path, depth_image)
    np.save(depth_npy_path, depth_mm.astype(np.float32))

    ys, xs = np.nonzero(valid_depth)
    if len(xs) > 0:
        z = depth_mm[ys, xs]
        cx = float(p1[0, 2]) * scale
        cy = float(p1[1, 2]) * scale
        x = (xs.astype(np.float32) - cx) * z / max(focal, 1e-6)
        y = (ys.astype(np.float32) - cy) * z / max(focal, 1e-6)
        points = np.column_stack([x, y, z]).astype(np.float32)
        colors = cv2.cvtColor(left_small, cv2.COLOR_BGR2RGB)[ys, xs]
    else:
        points = np.empty((0, 3), dtype=np.float32)
        colors = np.empty((0, 3), dtype=np.uint8)

    progress(0.58, f"正在生成点云文件（{len(points)} 点）...")
    semantic_payload, semantic_labels, _semantic_paths = _semantic_payload_from_mask(
        cv2,
        object_mask_result,
        valid_depth.shape,
        ys,
        xs,
    )
    world_payload = _build_world_coordinate_payload(
        cv2,
        left_small,
        points,
        depth_mm,
        valid_depth_before_object_mask,
        p1,
        focal,
        scale,
        config,
        reconstruction_dir,
    )
    board_mask = _rectified_board_mask(cv2, result, pair, valid_depth.shape, scale)
    if board_mask is not None:
        scale_mask = board_mask
        scale_mask_source = "rectified_calibration_board"
    elif object_mask_usable:
        scale_mask = object_mask
        scale_mask_source = "sam3_object_mask"
    else:
        scale_mask = None
        scale_mask_source = "valid_depth"
    scale_validation = _validate_depth_scale_from_reference(
        depth_mm,
        valid_depth,
        scale_mask,
        config,
        mask_source=scale_mask_source,
    )
    quality_metrics = _evaluate_reconstruction_quality(result, arrays, points)
    quality_metrics["depth_scale_validation"] = scale_validation
    quality_metrics["world_coordinate_system"] = world_payload["metadata"]

    point_cloud_path = reconstruction_dir / "point_cloud.ply"
    point_cloud_pcd_path = reconstruction_dir / "point_cloud.pcd"
    semantic_labels_path = reconstruction_dir / "semantic_labels.json"
    point_cloud_preview_path = reconstruction_dir / "point_cloud_preview.png"
    reconstruction_path = reconstruction_dir / "reconstruction_result.png"
    quality_metrics_path = reconstruction_dir / "quality_metrics.json"
    _save_ply(point_cloud_path, points, colors, semantic_payload)
    _save_pcd_ascii(point_cloud_pcd_path, points, colors, semantic_payload)
    if world_payload["points_world"] is not None:
        _save_ply(reconstruction_dir / "point_cloud_world.ply", world_payload["points_world"], colors, semantic_payload)
        _save_pcd_ascii(reconstruction_dir / "point_cloud_world.pcd", world_payload["points_world"], colors, semantic_payload)
    _write_json(semantic_labels_path, semantic_labels)
    if world_payload["metadata_path"]:
        _write_json(Path(world_payload["metadata_path"]), world_payload["metadata"])
    _write_json(quality_metrics_path, quality_metrics)
    mark_timing("point_cloud_outputs_seconds")
    progress(0.82, "正在绘制点云预览和重建拼图...")
    _save_point_cloud_plot(point_cloud_preview_path, points, colors, "Point cloud")
    _make_reconstruction_montage(
        cv2,
        rectified_pair_image,
        disparity_image,
        depth_image,
        point_cloud_preview_path,
        reconstruction_path,
    )
    mark_timing("preview_outputs_seconds")
    progress(0.96, "诊断重建结果写入完成。")
    return {
        "preview_scale": float(scale),
        "resource_policy": resource_policy,
        "method_requested": str(method_requested),
        "method_used": str(disparity_result["method"]),
        "fallback_error": fallback_error,
        "min_disparity": int(min_disp),
        "num_disparities": int(num_disp),
        "wls_filter": disparity_result.get("wls_filter", {}),
        "confidence_filter": {
            "enabled": bool(confidence_enabled),
            "threshold": float(confidence_threshold),
            **confidence_info,
        },
        "object_mask": object_mask_metadata,
        "disparity_range_px": list(disparity_range) if disparity_range else None,
        "depth_range_mm": list(depth_range) if depth_range else None,
        "valid_point_count": int(len(points)),
        "quality_metrics": quality_metrics,
        "timings": timings,
        "quality_metrics_json": str(quality_metrics_path),
        "disparity_map": str(disparity_path),
        "raw_disparity_npy": str(raw_disparity_path),
        "disparity_npy": str(filtered_disparity_path),
        "confidence_map": str(confidence_path),
        "confidence_npy": str(confidence_npy_path),
        "valid_disparity_npy": str(valid_disparity_npy_path),
        "valid_depth_npy": str(valid_depth_npy_path),
        "object_mask_png": object_mask_paths.get("mask", ""),
        "object_mask_npy": object_mask_paths.get("mask_npy", ""),
        "object_instance_map_npy": object_mask_paths.get("instance_map", ""),
        "object_semantic_map_npy": object_mask_paths.get("semantic_map", ""),
        "object_semantic_confidence_npy": object_mask_paths.get("confidence_map", ""),
        "object_mask_preview": object_mask_paths.get("preview", ""),
        "object_mask_metadata_json": object_mask_paths.get("metadata", ""),
        "semantic_labels_json": str(semantic_labels_path),
        "depth_scale_validation": scale_validation,
        "world_coordinate_system": world_payload["metadata"],
        "world_coordinate_json": world_payload["metadata_path"],
        "point_cloud_world_ply": str(reconstruction_dir / "point_cloud_world.ply") if world_payload["points_world"] is not None else "",
        "point_cloud_world_pcd": str(reconstruction_dir / "point_cloud_world.pcd") if world_payload["points_world"] is not None else "",
        "depth_map": str(depth_path),
        "depth_mm_npy": str(depth_npy_path),
        "point_cloud_ply": str(point_cloud_path),
        "point_cloud_pcd": str(point_cloud_pcd_path),
        "point_cloud_preview": str(point_cloud_preview_path),
        "reconstruction_result": str(reconstruction_path),
        "method_metadata": disparity_result.get("metadata", {}),
    }


def _select_diagnostic_pair(accepted_pairs: list[dict[str, Any]]) -> int:
    if not accepted_pairs:
        return 0
    scores = []
    for index, pair in enumerate(accepted_pairs):
        left_error = float(pair.get("left_reprojection_error_px", 0.0))
        right_error = float(pair.get("right_reprojection_error_px", 0.0))
        scores.append((left_error + right_error, index))
    return min(scores)[1]


def _generate_calibration_artifacts(
    cv2,
    output_path: Path,
    result: dict[str, Any],
    reconstruction_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    images_dir = output_path / "images"
    raw_dir = images_dir / "raw_pairs"
    corner_dir = images_dir / "corner_detection"
    undistort_dir = images_dir / "undistortion"
    rectified_dir = images_dir / "rectification"
    reconstruction_dir = output_path / "reconstruction"
    plots_dir = output_path / "plots"

    artifacts: dict[str, Any] = {
        "raw_calibration_images": {"directory": str(raw_dir), "count": 0},
        "corner_detection_images": {"directory": str(corner_dir), "count": 0},
    }

    for index, pair in enumerate(result.get("accepted_pairs", []), start=1):
        safe_key = f"{index:03d}_{_safe_name(pair.get('key', index))}"
        left_image = _read_color(cv2, pair["left"])
        right_image = _read_color(cv2, pair["right"])
        left_source = Path(pair["left"])
        right_source = Path(pair["right"])
        raw_left_path = raw_dir / f"{safe_key}_left{left_source.suffix.lower() or '.bmp'}"
        raw_right_path = raw_dir / f"{safe_key}_right{right_source.suffix.lower() or '.bmp'}"
        _link_or_copy_file(left_source, raw_left_path)
        _link_or_copy_file(right_source, raw_right_path)
        raw_pair = _compose_pair(cv2, left_image, right_image, "left original", "right original")
        raw_path = raw_dir / f"{safe_key}_raw_pair.png"
        _write_image(cv2, raw_path, raw_pair)

        left_overlay = _draw_detection_overlay(
            cv2,
            left_image,
            pair.get("left_points", []),
            pair.get("left_reprojected_points", []),
        )
        right_overlay = _draw_detection_overlay(
            cv2,
            right_image,
            pair.get("right_points", []),
            pair.get("right_reprojected_points", []),
        )
        corner_pair = _compose_pair(cv2, left_overlay, right_overlay, "left corners", "right corners")
        corner_path = corner_dir / f"{safe_key}_corner_detection.png"
        _write_image(cv2, corner_path, corner_pair)

        pair.setdefault("artifacts", {})
        pair["artifacts"].update(
            {
                "raw_left": str(raw_left_path),
                "raw_right": str(raw_right_path),
                "raw_pair_preview": str(raw_path),
                "corner_detection": str(corner_path),
            }
        )
        artifacts["raw_calibration_images"]["count"] += 1
        artifacts["corner_detection_images"]["count"] += 1

    if not result.get("accepted_pairs"):
        return artifacts

    plots_dir.mkdir(parents=True, exist_ok=True)
    _save_board_coverage_heatmap(cv2, plots_dir / "board_coverage_heatmap.png", result)
    _save_reprojection_error_distribution(plots_dir / "reprojection_error_distribution.png", result)

    diagnostic_index = _select_diagnostic_pair(result["accepted_pairs"])
    diagnostic_pair = result["accepted_pairs"][diagnostic_index]
    diagnostic_key = f"{diagnostic_index + 1:03d}_{_safe_name(diagnostic_pair.get('key', diagnostic_index + 1))}"
    left_image = _read_color(cv2, diagnostic_pair["left"])
    right_image = _read_color(cv2, diagnostic_pair["right"])

    left_undistorted = cv2.undistort(
        left_image,
        np.asarray(result["left"]["camera_matrix"], dtype=np.float64),
        np.asarray(result["left"]["distortion_coefficients"], dtype=np.float64).reshape(-1),
    )
    right_undistorted = cv2.undistort(
        right_image,
        np.asarray(result["right"]["camera_matrix"], dtype=np.float64),
        np.asarray(result["right"]["distortion_coefficients"], dtype=np.float64).reshape(-1),
    )
    left_undistort_path = undistort_dir / f"{diagnostic_key}_left_before_after.png"
    right_undistort_path = undistort_dir / f"{diagnostic_key}_right_before_after.png"
    _write_image(cv2, left_undistort_path, _compose_pair(cv2, left_image, left_undistorted, "left before", "left undistorted"))
    _write_image(cv2, right_undistort_path, _compose_pair(cv2, right_image, right_undistorted, "right before", "right undistorted"))

    artifacts["diagnostic_pair"] = {
        "index": int(diagnostic_index),
        "key": diagnostic_pair.get("key", ""),
        "left_source": diagnostic_pair["left"],
        "right_source": diagnostic_pair["right"],
    }
    artifacts["undistortion"] = {
        "left_before_after": str(left_undistort_path),
        "right_before_after": str(right_undistort_path),
    }

    if result.get("stereo", {}).get("rectification"):
        left_rectified, right_rectified = _rectify_pair(cv2, result, left_image, right_image)
        rectified_left_path = rectified_dir / f"{diagnostic_key}_left_rectified.png"
        rectified_right_path = rectified_dir / f"{diagnostic_key}_right_rectified.png"
        rectified_pair_path = rectified_dir / f"{diagnostic_key}_rectified_pair.png"
        rectified_pair_image = _draw_epipolar_pair(cv2, left_rectified, right_rectified)
        _write_image(cv2, rectified_left_path, _resize_max(cv2, left_rectified, 1800, 1200))
        _write_image(cv2, rectified_right_path, _resize_max(cv2, right_rectified, 1800, 1200))
        _write_image(cv2, rectified_pair_path, rectified_pair_image)
        artifacts["epipolar_rectification"] = {
            "left_rectified": str(rectified_left_path),
            "right_rectified": str(rectified_right_path),
            "rectified_pair": str(rectified_pair_path),
        }
        artifacts["reconstruction"] = _generate_reconstruction_artifacts(
            cv2,
            result,
            diagnostic_pair,
            left_rectified,
            right_rectified,
            rectified_pair_image,
            reconstruction_dir,
            reconstruction_config,
            progress_callback=progress_callback,
            progress_base=80.0,
            progress_span=12.0,
        )
        _save_depth_error_curve(plots_dir / "depth_error_curve.png", cv2, result)

    camera_pose_path = plots_dir / "camera_pose.png"
    board_pose_path = plots_dir / "calibration_board_poses.png"
    _save_camera_pose_plot(camera_pose_path, result)
    _save_board_pose_plot(board_pose_path, result)
    artifacts["camera_pose"] = {"image": str(camera_pose_path)}
    artifacts["calibration_board_poses"] = {"image": str(board_pose_path)}
    artifacts["board_coverage_heatmap"] = {"image": str(plots_dir / "board_coverage_heatmap.png")}
    artifacts["reprojection_error_distribution"] = {"image": str(plots_dir / "reprojection_error_distribution.png")}
    if (plots_dir / "depth_error_curve.png").exists():
        artifacts["depth_error_curve"] = {"image": str(plots_dir / "depth_error_curve.png")}
    return artifacts


def _matrix_summary(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _write_parameter_exports(output_path: Path, result: dict[str, Any]) -> dict[str, str]:
    artifacts = result.get("artifacts", {})
    rectification = result.get("stereo", {}).get("rectification", {})
    left_intr = result["left"]["matlab_like_intrinsics"]
    right_intr = result["right"]["matlab_like_intrinsics"]
    rows = [
        {"category": "内参", "content": "left K", "value": result["left"]["camera_matrix"], "files": ""},
        {"category": "内参", "content": "left D", "value": result["left"]["distortion_coefficients"], "files": ""},
        {"category": "内参", "content": "right K", "value": result["right"]["camera_matrix"], "files": ""},
        {"category": "内参", "content": "right D", "value": result["right"]["distortion_coefficients"], "files": ""},
        {"category": "外参", "content": "R", "value": result["stereo"]["rotation_matrix"], "files": ""},
        {"category": "外参", "content": "T", "value": result["stereo"]["translation_vector"], "files": ""},
        {"category": "校正参数", "content": "R1", "value": rectification.get("R1"), "files": ""},
        {"category": "校正参数", "content": "R2", "value": rectification.get("R2"), "files": ""},
        {"category": "校正参数", "content": "P1", "value": rectification.get("P1"), "files": ""},
        {"category": "校正参数", "content": "P2", "value": rectification.get("P2"), "files": ""},
        {"category": "校正参数", "content": "Q", "value": rectification.get("Q"), "files": ""},
        {
            "category": "精度",
            "content": "reprojection error",
            "value": {
                "left_rms_px": result["left"]["rms_reprojection_error_px"],
                "left_mean_px": result["left"]["mean_reprojection_error_px"],
                "right_rms_px": result["right"]["rms_reprojection_error_px"],
                "right_mean_px": result["right"]["mean_reprojection_error_px"],
                "stereo_rms_px": result["stereo"]["rms_reprojection_error_px"],
            },
            "files": result.get("files", {}).get("json", ""),
        },
        {
            "category": "标定图",
            "content": "原图+角点图",
            "value": {
                "raw_pair_preview_dir": artifacts.get("raw_calibration_images", {}).get("directory"),
                "corner_detection_dir": artifacts.get("corner_detection_images", {}).get("directory"),
                "raw_source_paths_saved_in": "accepted_pairs[].left/right",
            },
            "files": artifacts.get("corner_detection_images", {}).get("directory", ""),
        },
        {
            "category": "去畸变图",
            "content": "before/after",
            "value": artifacts.get("undistortion", {}),
            "files": "; ".join(str(v) for v in artifacts.get("undistortion", {}).values()),
        },
        {
            "category": "极线图",
            "content": "rectified pair",
            "value": artifacts.get("epipolar_rectification", {}),
            "files": artifacts.get("epipolar_rectification", {}).get("rectified_pair", ""),
        },
        {
            "category": "配置",
            "content": "分辨率、焦距、基线",
            "value": {
                "resolution": result["image_size"],
                "left_focal_length_px": left_intr["focal_length_px"],
                "right_focal_length_px": right_intr["focal_length_px"],
                "baseline_mm": result["stereo"]["baseline_mm"],
            },
            "files": "",
        },
        {"category": "时间", "content": "标定日期", "value": result["calibration_date"], "files": ""},
        {
            "category": "相机位姿图",
            "content": "camera pose",
            "value": artifacts.get("camera_pose", {}),
            "files": artifacts.get("camera_pose", {}).get("image", ""),
        },
        {
            "category": "三维重建",
            "content": "disparity/depth/point cloud/reconstruction",
            "value": artifacts.get("reconstruction", {}),
            "files": "; ".join(
                str(artifacts.get("reconstruction", {}).get(key, ""))
                for key in (
                    "disparity_map",
                    "depth_map",
                    "object_mask_png",
                    "object_mask_preview",
                    "point_cloud_ply",
                    "point_cloud_pcd",
                    "semantic_labels_json",
                    "point_cloud_preview",
                    "reconstruction_result",
                )
                if artifacts.get("reconstruction", {}).get(key)
            ),
        },
    ]
    table = {"columns": ["category", "content", "value", "files"], "rows": rows}
    json_path = output_path / "calibration_parameters_table.json"
    csv_path = output_path / "calibration_parameters_table.csv"
    _write_json(json_path, table)
    csv_rows = [
        {
            "类别": row["category"],
            "内容": row["content"],
            "值": _matrix_summary(row.get("value")),
            "文件": row.get("files", ""),
        }
        for row in rows
    ]
    _write_csv(csv_path, csv_rows, ["类别", "内容", "值", "文件"])
    result["parameter_table"] = table
    return {"json": str(json_path), "csv": str(csv_path)}


def calibrate_stereo_from_folders(
    left_dir: str | Path,
    right_dir: str | Path,
    output_dir: str | Path,
    *,
    pattern: str,
    columns: int,
    rows: int,
    square_size_mm: float,
    marker_size_mm: float | None = None,
    aruco_dictionary: str = "DICT_4X4_50",
    legacy_charuco: bool = False,
    min_pairs: int = 3,
    reconstruction_config: dict[str, Any] | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    cv2 = _load_cv2()
    reconstruction_config = _normalize_reconstruction_config(reconstruction_config)
    pattern = pattern.strip().lower()
    aruco_dictionary = normalize_aruco_dictionary_name(aruco_dictionary)
    if pattern not in {"chessboard", "charuco", "charuco_legacy", "circles", "acircles"}:
        raise CalibrationError(f"不支持的标定板类型：{pattern}")
    if pattern == "charuco_legacy":
        pattern = "charuco"
        legacy_charuco = True

    _invoke_progress(progress_callback, 2.0, "正在搜索左右图像...")
    pairs = find_stereo_pairs(left_dir, right_dir)
    if not pairs:
        raise CalibrationError("未找到可配对的左右图像。左右文件名需包含相同时间戳或序号。")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_size: tuple[int, int] | None = None
    object_points: list[np.ndarray] = []
    left_points: list[np.ndarray] = []
    right_points: list[np.ndarray] = []
    accepted_pairs: list[dict[str, Any]] = []
    rejected_pairs: list[dict[str, Any]] = []

    board = None
    dictionary = None
    charuco_corners = None
    if pattern == "charuco":
        marker = float(marker_size_mm if marker_size_mm is not None else square_size_mm * 0.75)
        board, dictionary = _create_charuco_board(
            cv2,
            columns,
            rows,
            float(square_size_mm),
            marker,
            aruco_dictionary,
            legacy_charuco,
        )
        charuco_corners = _charuco_board_corners(board)
    else:
        base_object_points = _grid_object_points(pattern, columns, rows, float(square_size_mm))

    total_pairs = max(len(pairs), 1)
    for pair_index, (key, left_path, right_path) in enumerate(pairs, start=1):
        _invoke_progress(progress_callback, 2.0 + 30.0 * (pair_index - 1) / total_pairs, f"正在检测角点：{pair_index}/{len(pairs)}")
        try:
            left_gray = _read_gray(cv2, left_path)
            right_gray = _read_gray(cv2, right_path)
        except CalibrationError as exc:
            rejected_pairs.append({"key": key, "left": str(left_path), "right": str(right_path), "reason": str(exc)})
            continue

        if left_gray.shape != right_gray.shape:
            rejected_pairs.append(
                {
                    "key": key,
                    "left": str(left_path),
                    "right": str(right_path),
                    "reason": f"左右图像尺寸不同：{left_gray.shape[::-1]} vs {right_gray.shape[::-1]}",
                }
            )
            continue

        current_size = (int(left_gray.shape[1]), int(left_gray.shape[0]))
        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            rejected_pairs.append(
                {
                    "key": key,
                    "left": str(left_path),
                    "right": str(right_path),
                    "reason": f"图像尺寸与首组不同：{current_size} vs {image_size}",
                }
            )
            continue

        if pattern == "charuco":
            assert board is not None and dictionary is not None and charuco_corners is not None
            left_detection = _detect_charuco(cv2, left_gray, board, dictionary)
            right_detection = _detect_charuco(cv2, right_gray, board, dictionary)
            if left_detection is None or right_detection is None:
                rejected_pairs.append(
                    {"key": key, "left": str(left_path), "right": str(right_path), "reason": "ChArUco 角点检测失败"}
                )
                continue
            left_ids, left_corners = left_detection
            right_ids, right_corners = right_detection
            left_map = {int(idx): point for idx, point in zip(left_ids, left_corners)}
            right_map = {int(idx): point for idx, point in zip(right_ids, right_corners)}
            common_ids = sorted(set(left_map) & set(right_map))
            if len(common_ids) < 4:
                rejected_pairs.append(
                    {"key": key, "left": str(left_path), "right": str(right_path), "reason": "左右共同 ChArUco 角点少于 4 个"}
                )
                continue
            obj = charuco_corners[common_ids].astype(np.float32)
            left_img = np.asarray([left_map[idx] for idx in common_ids], dtype=np.float32)
            right_img = np.asarray([right_map[idx] for idx in common_ids], dtype=np.float32)
            detected_count = len(common_ids)
        elif pattern == "chessboard":
            left_img = _detect_chessboard(cv2, left_gray, columns, rows)
            right_img = _detect_chessboard(cv2, right_gray, columns, rows)
            if left_img is None or right_img is None:
                rejected_pairs.append(
                    {"key": key, "left": str(left_path), "right": str(right_path), "reason": "棋盘格角点检测失败"}
                )
                continue
            obj = base_object_points.copy()
            detected_count = len(obj)
        elif pattern in {"circles", "acircles"}:
            asymmetric = pattern == "acircles"
            left_img = _detect_circles(cv2, left_gray, columns, rows, asymmetric)
            right_img = _detect_circles(cv2, right_gray, columns, rows, asymmetric)
            if left_img is None or right_img is None:
                rejected_pairs.append(
                    {"key": key, "left": str(left_path), "right": str(right_path), "reason": "圆点阵检测失败"}
                )
                continue
            obj = base_object_points.copy()
            detected_count = len(obj)
        else:
            raise CalibrationError(f"不支持的标定板类型：{pattern}")

        object_points.append(obj.reshape(-1, 3).astype(np.float32))
        left_points.append(left_img.reshape(-1, 1, 2).astype(np.float32))
        right_points.append(right_img.reshape(-1, 1, 2).astype(np.float32))
        accepted_pairs.append(
            {
                "key": key,
                "left": str(left_path),
                "right": str(right_path),
                "point_count": int(detected_count),
            }
        )

    rejected_path = output_path / "rejected_pairs.json"
    if image_size is None:
        _write_json(rejected_path, rejected_pairs)
        raise CalibrationError("未能读取有效图像。")
    if len(accepted_pairs) < min_pairs:
        _write_json(rejected_path, rejected_pairs)
        raise CalibrationError(
            f"有效标定图像只有 {len(accepted_pairs)} 对，至少需要 {min_pairs} 对。"
            f"已拒绝 {len(rejected_pairs)} 对，可查看输出目录中的 rejected_pairs.json。"
        )

    _invoke_progress(progress_callback, 35.0, "正在进行单目标定...")
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    left_rms, left_matrix, left_dist, left_rvecs, left_tvecs = cv2.calibrateCamera(
        object_points,
        left_points,
        image_size,
        None,
        None,
        criteria=criteria,
    )
    right_rms, right_matrix, right_dist, right_rvecs, right_tvecs = cv2.calibrateCamera(
        object_points,
        right_points,
        image_size,
        None,
        None,
        criteria=criteria,
    )
    stereo_rms, left_matrix, left_dist, right_matrix, right_dist, rotation, translation, essential, fundamental = (
        cv2.stereoCalibrate(
            object_points,
            left_points,
            right_points,
            left_matrix,
            left_dist,
            right_matrix,
            right_dist,
            image_size,
            criteria=criteria,
            flags=cv2.CALIB_FIX_INTRINSIC,
        )
    )

    _invoke_progress(progress_callback, 60.0, "正在计算重投影误差与位姿...")

    left_mean_error, left_view_errors = _mono_error(
        cv2, object_points, left_points, left_rvecs, left_tvecs, left_matrix, left_dist
    )
    right_mean_error, right_view_errors = _mono_error(
        cv2, object_points, right_points, right_rvecs, right_tvecs, right_matrix, right_dist
    )
    left_projected = _projected_points(cv2, object_points, left_rvecs, left_tvecs, left_matrix, left_dist)
    right_projected = _projected_points(cv2, object_points, right_rvecs, right_tvecs, right_matrix, right_dist)
    board_rvecs, board_tvecs = left_rvecs, left_tvecs

    for index, pair in enumerate(accepted_pairs):
        pair["left_points"] = _array(left_points[index].reshape(-1, 2))
        pair["right_points"] = _array(right_points[index].reshape(-1, 2))
        pair["left_reprojected_points"] = _array(left_projected[index])
        pair["right_reprojected_points"] = _array(right_projected[index])
        pair["left_reprojection_error_px"] = float(left_view_errors[index])
        pair["right_reprojection_error_px"] = float(right_view_errors[index])
        pair["object_points"] = _array(object_points[index].reshape(-1, 3))
        pair["board_pose_left_camera"] = {
            "rotation_vector": _array(board_rvecs[index].reshape(-1)),
            "translation_vector_mm": _array(board_tvecs[index].reshape(-1)),
        }

    stereo: dict[str, Any] = {
        "rms_reprojection_error_px": float(stereo_rms),
        "rotation_matrix": _array(rotation),
        "translation_vector": _array(translation.reshape(-1)),
        "baseline_mm": float(np.linalg.norm(translation)),
        "essential_matrix": _array(essential),
        "fundamental_matrix": _array(fundamental),
    }

    _invoke_progress(progress_callback, 70.0, "正在计算极线校正参数...")
    try:
        r1, r2, p1, p2, q, roi1, roi2 = cv2.stereoRectify(
            left_matrix,
            left_dist,
            right_matrix,
            right_dist,
            image_size,
            rotation,
            translation,
        )
        stereo["rectification"] = {
            "R1": _array(r1),
            "R2": _array(r2),
            "P1": _array(p1),
            "P2": _array(p2),
            "Q": _array(q),
            "roi1": list(map(int, roi1)),
            "roi2": list(map(int, roi2)),
        }
    except Exception:
        pass

    result: dict[str, Any] = {
        "calibration_date": _calibration_timestamp(),
        "pattern": pattern,
        "board": {
            "columns": int(columns),
            "rows": int(rows),
            "square_size_mm": float(square_size_mm),
            "marker_size_mm": float(marker_size_mm if marker_size_mm is not None else square_size_mm * 0.75)
            if pattern == "charuco"
            else None,
            "aruco_dictionary": aruco_dictionary if pattern == "charuco" else None,
            "legacy_charuco": bool(legacy_charuco) if pattern == "charuco" else False,
        },
        "image_size": list(image_size),
        "total_pairs": len(pairs),
        "accepted_pair_count": len(accepted_pairs),
        "rejected_pair_count": len(rejected_pairs),
        "accepted_pairs": accepted_pairs,
        "rejected_pairs": rejected_pairs,
        "configuration": {
            "resolution": list(image_size),
            "left_focal_length_px": _matlab_like_parameters(left_matrix, left_dist)["focal_length_px"],
            "right_focal_length_px": _matlab_like_parameters(right_matrix, right_dist)["focal_length_px"],
            "baseline_mm": float(np.linalg.norm(translation)),
            "left_source_dir": str(Path(left_dir)),
            "right_source_dir": str(Path(right_dir)),
            "output_dir": str(output_path),
            "reconstruction": reconstruction_config,
        },
        "left": {
            "rms_reprojection_error_px": float(left_rms),
            "mean_reprojection_error_px": float(left_mean_error),
            "per_view_reprojection_error_px": left_view_errors,
            "camera_matrix": _array(left_matrix),
            "distortion_coefficients": _array(left_dist.reshape(-1)),
            "matlab_like_intrinsics": _matlab_like_parameters(left_matrix, left_dist),
        },
        "right": {
            "rms_reprojection_error_px": float(right_rms),
            "mean_reprojection_error_px": float(right_mean_error),
            "per_view_reprojection_error_px": right_view_errors,
            "camera_matrix": _array(right_matrix),
            "distortion_coefficients": _array(right_dist.reshape(-1)),
            "matlab_like_intrinsics": _matlab_like_parameters(right_matrix, right_dist),
        },
        "stereo": stereo,
    }

    json_path = output_path / "calibration_result.json"
    yaml_path = output_path / "calibration_result.yaml"
    _write_json(json_path, result)
    _write_json(rejected_path, rejected_pairs)
    _write_opencv_yaml(cv2, yaml_path, result)

    _invoke_progress(progress_callback, 78.0, "正在生成诊断图、诊断重建与点云结果...")

    result["files"] = {
        "json": str(json_path),
        "yaml": str(yaml_path),
        "rejected_pairs": str(rejected_path),
        "output_dir": str(output_path),
    }
    result["artifacts"] = _generate_calibration_artifacts(cv2, output_path, result, reconstruction_config)
    result["files"]["parameter_table"] = _write_parameter_exports(output_path, result)
    _write_json(json_path, result)
    _invoke_progress(progress_callback, 100.0, "标定完成")
    return result


def summarize_result(result: dict[str, Any]) -> str:
    stereo = result["stereo"]
    return (
        f"标定完成：有效 {result['accepted_pair_count']}/{result['total_pairs']} 对；"
        f"左 RMS {result['left']['rms_reprojection_error_px']:.4f}px，"
        f"右 RMS {result['right']['rms_reprojection_error_px']:.4f}px，"
        f"双目 RMS {stereo['rms_reprojection_error_px']:.4f}px，"
        f"基线 {stereo['baseline_mm']:.3f} mm。"
    )

from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
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
    for x, y in np.asarray(detected_points, dtype=float).reshape(-1, 2):
        cv2.circle(overlay, (int(round(x)), int(round(y))), radius, (0, 210, 0), thickness, cv2.LINE_AA)
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


def _save_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
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
        fh.write("end_header\n")
        for point, color in zip(points, colors):
            fh.write(
                f"{point[0]:.4f} {point[1]:.4f} {point[2]:.4f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def _camera_points_to_plot(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    return np.column_stack((pts[:, 0], pts[:, 2], -pts[:, 1]))


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


def _generate_reconstruction_artifacts(
    cv2,
    result: dict[str, Any],
    pair: dict[str, Any],
    left_rectified,
    right_rectified,
    rectified_pair_image,
    reconstruction_dir: Path,
) -> dict[str, Any]:
    height, width = left_rectified.shape[:2]
    scale = min(1.0, 1400.0 / max(width, 1))
    small_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    left_small = cv2.resize(left_rectified, small_size, interpolation=cv2.INTER_AREA) if scale < 1 else left_rectified.copy()
    right_small = cv2.resize(right_rectified, small_size, interpolation=cv2.INTER_AREA) if scale < 1 else right_rectified.copy()
    gray_left = cv2.cvtColor(left_small, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_small, cv2.COLOR_BGR2GRAY)
    gray_left = cv2.equalizeHist(gray_left)
    gray_right = cv2.equalizeHist(gray_right)

    point_disparities = _rectified_point_disparities(cv2, result, pair)
    min_disp, num_disp = _choose_disparity_range(point_disparities, scale, small_size[0])
    block_size = 5
    matcher = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * block_size * block_size,
        P2=32 * block_size * block_size,
        disp12MaxDiff=1,
        uniquenessRatio=8,
        speckleWindowSize=80,
        speckleRange=2,
        preFilterCap=63,
        mode=getattr(cv2, "STEREO_SGBM_MODE_SGBM_3WAY", cv2.STEREO_SGBM_MODE_SGBM),
    )
    disparity = matcher.compute(gray_left, gray_right).astype(np.float32) / 16.0
    valid = disparity > (min_disp + 1)
    disparity_image, disparity_range = _colored_range_image(cv2, disparity, valid)
    disparity_path = reconstruction_dir / "disparity_map.png"
    _write_image(cv2, disparity_path, disparity_image)

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
    depth_image, depth_range = _colored_range_image(cv2, depth_mm, valid_depth, invert=True)
    depth_path = reconstruction_dir / "depth_map.png"
    depth_npy_path = reconstruction_dir / "depth_mm.npy"
    _write_image(cv2, depth_path, depth_image)
    np.save(depth_npy_path, depth_mm.astype(np.float32))

    ys, xs = np.nonzero(valid_depth)
    if len(xs) > 0:
        max_points = 200000
        if len(xs) > max_points:
            indices = np.linspace(0, len(xs) - 1, max_points, dtype=int)
            xs = xs[indices]
            ys = ys[indices]
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

    point_cloud_path = reconstruction_dir / "point_cloud.ply"
    point_cloud_preview_path = reconstruction_dir / "point_cloud_preview.png"
    reconstruction_path = reconstruction_dir / "reconstruction_result.png"
    _save_ply(point_cloud_path, points, colors)
    _save_point_cloud_plot(point_cloud_preview_path, points, colors, "Point cloud")
    _make_reconstruction_montage(
        cv2,
        rectified_pair_image,
        disparity_image,
        depth_image,
        point_cloud_preview_path,
        reconstruction_path,
    )
    return {
        "preview_scale": float(scale),
        "min_disparity": int(min_disp),
        "num_disparities": int(num_disp),
        "disparity_range_px": list(disparity_range) if disparity_range else None,
        "depth_range_mm": list(depth_range) if depth_range else None,
        "valid_point_count": int(len(points)),
        "disparity_map": str(disparity_path),
        "depth_map": str(depth_path),
        "depth_mm_npy": str(depth_npy_path),
        "point_cloud_ply": str(point_cloud_path),
        "point_cloud_preview": str(point_cloud_preview_path),
        "reconstruction_result": str(reconstruction_path),
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


def _generate_calibration_artifacts(cv2, output_path: Path, result: dict[str, Any]) -> dict[str, Any]:
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
                for key in ("disparity_map", "depth_map", "point_cloud_ply", "point_cloud_preview", "reconstruction_result")
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
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    cv2 = _load_cv2()
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

    _invoke_progress(progress_callback, 78.0, "正在生成诊断图与重建结果...")

    result["files"] = {
        "json": str(json_path),
        "yaml": str(yaml_path),
        "rejected_pairs": str(rejected_path),
        "output_dir": str(output_path),
    }
    result["artifacts"] = _generate_calibration_artifacts(cv2, output_path, result)
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

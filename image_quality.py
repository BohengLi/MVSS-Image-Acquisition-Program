from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
from PIL import Image


DEFAULT_FOCUS_ROI = {"x_frac": 0.30, "y_frac": 0.30, "w_frac": 0.40, "h_frac": 0.40}


def clamp_roi_frac(roi: dict[str, Any] | None) -> dict[str, float]:
    source = roi or DEFAULT_FOCUS_ROI
    x = float(source.get("x_frac", DEFAULT_FOCUS_ROI["x_frac"]))
    y = float(source.get("y_frac", DEFAULT_FOCUS_ROI["y_frac"]))
    w = float(source.get("w_frac", DEFAULT_FOCUS_ROI["w_frac"]))
    h = float(source.get("h_frac", DEFAULT_FOCUS_ROI["h_frac"]))
    w = min(max(w, 0.02), 1.0)
    h = min(max(h, 0.02), 1.0)
    x = min(max(x, 0.0), 1.0 - w)
    y = min(max(y, 0.0), 1.0 - h)
    return {"x_frac": x, "y_frac": y, "w_frac": w, "h_frac": h}


def roi_from_pixels(x: int, y: int, width: int, height: int, image_width: int, image_height: int) -> dict[str, float]:
    image_width = max(int(image_width), 1)
    image_height = max(int(image_height), 1)
    return clamp_roi_frac(
        {
            "x_frac": x / image_width,
            "y_frac": y / image_height,
            "w_frac": width / image_width,
            "h_frac": height / image_height,
        }
    )


def roi_to_pixels(roi: dict[str, Any] | None, image_width: int, image_height: int) -> tuple[int, int, int, int]:
    roi = clamp_roi_frac(roi)
    x = int(round(roi["x_frac"] * image_width))
    y = int(round(roi["y_frac"] * image_height))
    w = int(round(roi["w_frac"] * image_width))
    h = int(round(roi["h_frac"] * image_height))
    x = min(max(x, 0), max(image_width - 1, 0))
    y = min(max(y, 0), max(image_height - 1, 0))
    w = min(max(w, 1), max(image_width - x, 1))
    h = min(max(h, 1), max(image_height - y, 1))
    return x, y, w, h


def _pil_to_gray(image: Image.Image, max_side: int = 960) -> tuple[np.ndarray, float, float]:
    source_width, source_height = image.size
    if max(source_width, source_height) > max_side:
        scale = max_side / max(source_width, source_height)
        width = max(1, int(round(source_width * scale)))
        height = max(1, int(round(source_height * scale)))
        work = image.convert("L").resize((width, height), Image.Resampling.BILINEAR)
    else:
        width, height = source_width, source_height
        work = image.convert("L")
    gray = np.asarray(work, dtype=np.uint8)
    scale_x = width / max(source_width, 1)
    scale_y = height / max(source_height, 1)
    return gray, scale_x, scale_y


def focus_score(image: Image.Image, roi: dict[str, Any] | None, method: str = "laplacian") -> float:
    gray, _scale_x, _scale_y = _pil_to_gray(image, max_side=960)
    x, y, w, h = roi_to_pixels(roi, gray.shape[1], gray.shape[0])
    patch = gray[y : y + h, x : x + w]
    if patch.size == 0:
        return 0.0
    method = str(method or "laplacian").strip().lower()
    if method == "tenengrad":
        gx = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
        return float(np.mean(np.sqrt(gx * gx + gy * gy)))
    return float(cv2.Laplacian(patch, cv2.CV_64F).var())


def focus_pair_metrics(
    left: Image.Image | None,
    right: Image.Image | None,
    roi: dict[str, Any] | None,
    method: str = "laplacian",
) -> dict[str, Any]:
    left_score = focus_score(left, roi, method) if left is not None else None
    right_score = focus_score(right, roi, method) if right is not None else None
    scores = [score for score in (left_score, right_score) if score is not None]
    reference_score = float(np.mean(scores)) if scores else 0.0
    delta = None
    consistency_warning = False
    consistency_ratio = 0.0
    if left_score is not None and right_score is not None:
        delta = abs(left_score - right_score)
        denominator = max(left_score, right_score, 1e-9)
        consistency_ratio = delta / denominator
        consistency_warning = consistency_ratio > 0.30
    return {
        "left": left_score,
        "right": right_score,
        "score": reference_score,
        "delta": delta,
        "consistency_ratio": consistency_ratio,
        "consistency_warning": consistency_warning,
        "method": method,
    }


def make_focus_peaking_overlay(image: Image.Image, strength: float = 0.55, max_side: int = 960) -> Image.Image:
    gray, _scale_x, _scale_y = _pil_to_gray(image, max_side=max_side)
    blurred = cv2.GaussianBlur(gray, (0, 0), 1.2)
    high = cv2.absdiff(gray, blurred)
    gx = cv2.Sobel(high, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(high, cv2.CV_32F, 0, 1, ksize=3)
    edge = cv2.magnitude(gx, gy)
    if not np.any(edge):
        alpha = np.zeros_like(gray, dtype=np.uint8)
    else:
        cutoff = max(float(np.percentile(edge, 88)), 1.0)
        normalized = np.clip(edge / cutoff, 0.0, 1.0)
        alpha = (normalized * 255.0 * max(min(strength, 1.0), 0.0)).astype(np.uint8)
        alpha[normalized < 0.18] = 0
    red = np.full_like(gray, 255, dtype=np.uint8)
    green = np.where(alpha > 80, 54, 210).astype(np.uint8)
    blue = np.zeros_like(gray, dtype=np.uint8)
    rgba = np.dstack([red, green, blue, alpha])
    return Image.fromarray(rgba, "RGBA")


def exposure_metrics(image: Image.Image | None, include_histogram: bool = True) -> dict[str, Any] | None:
    if image is None:
        return None
    gray, _scale_x, _scale_y = _pil_to_gray(image, max_side=960)
    pixel_count = max(int(gray.size), 1)
    over_pct = float(np.count_nonzero(gray >= 255) * 100.0 / pixel_count)
    under_pct = float(np.count_nonzero(gray <= 1) * 100.0 / pixel_count)
    mean = float(np.mean(gray))
    snr_db = estimate_snr_db(gray)
    histogram = None
    if include_histogram:
        histogram = cv2.calcHist([gray], [0], None, [256], [0, 256]).reshape(-1).astype(float).tolist()
    return {
        "histogram": histogram,
        "mean": mean,
        "over_pct": over_pct,
        "under_pct": under_pct,
        "snr_db": snr_db,
        "advice": exposure_advice(mean, over_pct),
    }


def exposure_advice(mean: float, over_pct: float) -> str:
    if mean < 40.0 and over_pct < 0.5:
        return "曝光不足，建议增加曝光时间或增益"
    if over_pct > 5.0:
        return "过曝区域较大，建议减少曝光时间"
    if 80.0 <= mean <= 170.0 and over_pct < 1.0:
        return "曝光良好"
    return "曝光可用，请结合直方图微调"


def estimate_snr_db(gray: np.ndarray) -> float | None:
    height, width = gray.shape[:2]
    if height < 32 or width < 32:
        return None
    x0 = int(width * 0.20)
    x1 = int(width * 0.80)
    y0 = int(height * 0.20)
    y1 = int(height * 0.80)
    region = gray[y0:y1, x0:x1]
    if region.shape[0] < 32 or region.shape[1] < 32:
        region = gray
    best: tuple[float, float, float] | None = None
    step = 16
    block = 32
    for y in range(0, max(region.shape[0] - block + 1, 1), step):
        for x in range(0, max(region.shape[1] - block + 1, 1), step):
            patch = region[y : y + block, x : x + block].astype(np.float32)
            if patch.shape[0] != block or patch.shape[1] != block:
                continue
            mean = float(np.mean(patch))
            std = float(np.std(patch))
            var = std * std
            if best is None or var < best[0]:
                best = (var, mean, std)
    if best is None:
        return None
    var, mean, std = best
    if var > 220.0 or std <= 1e-6 or mean <= 1e-6:
        return None
    return float(20.0 * math.log10(mean / std))


def make_anaglyph(left: Image.Image, right: Image.Image) -> Image.Image:
    left_gray = left.convert("L")
    right_gray = right.convert("L").resize(left_gray.size, Image.Resampling.BILINEAR)
    red = np.asarray(left_gray, dtype=np.uint8)
    cyan = np.asarray(right_gray, dtype=np.uint8)
    rgb = np.dstack([red, cyan, cyan])
    return Image.fromarray(rgb, "RGB")


def calibration_board_coverage(image: Image.Image | None, config: dict[str, Any]) -> dict[str, Any] | None:
    if image is None:
        return None
    gray, scale_x, scale_y = _pil_to_gray(image, max_side=1280)
    rows = int(config.get("board_grid_rows", 3) or 3)
    cols = int(config.get("board_grid_cols", 3) or 3)
    pattern_cols = int(config.get("board_pattern_cols", 9) or 9)
    pattern_rows = int(config.get("board_pattern_rows", 6) or 6)
    found, corners = cv2.findChessboardCorners(gray, (pattern_cols, pattern_rows), None)
    if not found:
        return {
            "found": False,
            "area_frac": 0.0,
            "position": "未检测到",
            "grid": [[False for _ in range(cols)] for _ in range(rows)],
            "suggestion": "未检测到标定板，请检查棋盘格尺寸或画面清晰度",
        }
    points = corners.reshape(-1, 2)
    x, y, w, h = cv2.boundingRect(points.astype(np.float32))
    full_w = gray.shape[1]
    full_h = gray.shape[0]
    area_frac = float((w * h) / max(full_w * full_h, 1))
    center_x = (x + w / 2.0) / max(full_w, 1)
    center_y = (y + h / 2.0) / max(full_h, 1)
    position = _position_name(center_x, center_y)
    grid = [[False for _ in range(cols)] for _ in range(rows)]
    for row in range(rows):
        cell_y0 = row / rows * full_h
        cell_y1 = (row + 1) / rows * full_h
        for col in range(cols):
            cell_x0 = col / cols * full_w
            cell_x1 = (col + 1) / cols * full_w
            overlaps = not (x + w < cell_x0 or x > cell_x1 or y + h < cell_y0 or y > cell_y1)
            grid[row][col] = bool(overlaps)
    min_area = float(config.get("board_min_area_frac", 0.05) or 0.05)
    max_area = float(config.get("board_max_area_frac", 0.40) or 0.40)
    if area_frac < min_area:
        suggestion = "标定板过小，建议靠近"
    elif area_frac > max_area:
        suggestion = "标定板过大，建议拉远"
    else:
        missing = _missing_grid_names(grid)
        suggestion = "覆盖良好" if not missing else f"建议移动到：{'、'.join(missing[:4])}"
    return {
        "found": True,
        "area_frac": area_frac,
        "position": position,
        "grid": grid,
        "grid_icon": "\n".join("".join("■" if cell else "□" for cell in row) for row in grid),
        "suggestion": suggestion,
        "bbox_scaled": {
            "x": int(round(x / max(scale_x, 1e-9))),
            "y": int(round(y / max(scale_y, 1e-9))),
            "w": int(round(w / max(scale_x, 1e-9))),
            "h": int(round(h / max(scale_y, 1e-9))),
        },
    }


def _position_name(x: float, y: float) -> str:
    cols = ["左", "中", "右"]
    rows = ["上", "中", "下"]
    col = cols[min(max(int(x * 3), 0), 2)]
    row = rows[min(max(int(y * 3), 0), 2)]
    if row == "中" and col == "中":
        return "中"
    return row + col


def _missing_grid_names(grid: list[list[bool]]) -> list[str]:
    row_names = ["上", "中", "下"]
    col_names = ["左", "中", "右"]
    missing: list[str] = []
    for row_index, row in enumerate(grid):
        for col_index, covered in enumerate(row):
            if covered:
                continue
            row_name = row_names[row_index] if row_index < len(row_names) else f"{row_index + 1}行"
            col_name = col_names[col_index] if col_index < len(col_names) else f"{col_index + 1}列"
            missing.append("中" if row_name == "中" and col_name == "中" else row_name + col_name)
    return missing


def epipolar_alignment(left: Image.Image | None, right: Image.Image | None) -> dict[str, Any]:
    if left is None or right is None:
        return {"ok": False, "message": "左右图像不完整"}
    left_gray, _left_scale_x, left_scale_y = _pil_to_gray(left, max_side=1280)
    right_gray, _right_scale_x, right_scale_y = _pil_to_gray(right, max_side=1280)
    if left_gray.shape != right_gray.shape:
        right_gray = cv2.resize(right_gray, (left_gray.shape[1], left_gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    detector = cv2.ORB_create(nfeatures=900, fastThreshold=20)
    kp1, des1 = detector.detectAndCompute(left_gray, None)
    kp2, des2 = detector.detectAndCompute(right_gray, None)
    if des1 is None or des2 is None or len(kp1) < 12 or len(kp2) < 12:
        return {"ok": False, "message": "可匹配特征点不足，建议增加纹理或调整光照"}
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    if len(matches) < 12:
        return {"ok": False, "message": "匹配点不足，无法估计极线偏差"}
    matches = sorted(matches, key=lambda item: item.distance)[: min(len(matches), 120)]
    scale_y = max((left_scale_y + right_scale_y) / 2.0, 1e-9)
    y_diffs = [abs(kp1[m.queryIdx].pt[1] - kp2[m.trainIdx].pt[1]) / scale_y for m in matches]
    mean = float(np.mean(y_diffs))
    maximum = float(np.max(y_diffs))
    warning = mean > 3.0 or maximum > 10.0
    return {
        "ok": True,
        "match_count": len(matches),
        "mean_y_delta_px": mean,
        "max_y_delta_px": maximum,
        "warning": warning,
        "message": f"极线偏差: 均值 {mean:.1f}px 最大 {maximum:.1f}px",
    }

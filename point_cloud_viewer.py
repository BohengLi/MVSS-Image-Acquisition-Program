from __future__ import annotations

import argparse
import math
import sys
import traceback
from pathlib import Path


def _show_error(title: str, message: str) -> None:
    try:
        from tkinter import Tk, messagebox

        root = Tk()
        root.withdraw()
        messagebox.showerror(title, message)
        root.destroy()
    except Exception:
        print(f"{title}: {message}", file=sys.stderr)


def _load_open3d():
    try:
        import open3d as o3d
    except Exception as exc:
        raise RuntimeError(
            "当前 Python 环境无法导入 Open3D。\n"
            "请确认已执行：py -3.12 -m pip install -r requirements.txt\n\n"
            f"当前解释器：{sys.executable}\n"
            f"错误信息：{exc}"
        ) from exc
    return o3d


def _load_numpy():
    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError(
            "当前 Python 环境无法导入 NumPy。\n"
            "请确认点云查看器使用的是 Python 3.12，并已执行：py -3.12 -m pip install -r requirements.txt\n\n"
            f"当前解释器：{sys.executable}\n"
            f"错误信息：{exc}"
        ) from exc
    return np


def _load_text_tools():
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:
        raise RuntimeError(
            "当前 Python 环境无法导入 Pillow，无法生成中文三维标注。\n"
            "请执行：py -3.12 -m pip install -r requirements.txt\n\n"
            f"错误信息：{exc}"
        ) from exc

    font_candidates = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\msyhbd.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
    ]
    font = None
    for font_path in font_candidates:
        if font_path.exists():
            try:
                font = ImageFont.truetype(str(font_path), 36)
                break
            except Exception:
                continue
    if font is None:
        font = ImageFont.load_default()
    return Image, ImageDraw, font


def _expanded_bounds(cloud) -> tuple[list[float], list[float], list[float], float]:
    bbox = cloud.get_axis_aligned_bounding_box()
    try:
        min_bound = [float(value) for value in bbox.get_min_bound()]
        max_bound = [float(value) for value in bbox.get_max_bound()]
    except Exception:
        min_bound = [-0.5, -0.5, -0.5]
        max_bound = [0.5, 0.5, 0.5]

    if not all(math.isfinite(value) for value in [*min_bound, *max_bound]):
        raise RuntimeError("点云坐标包含非有限值，无法生成三维坐标框。")

    raw_extent = [max_bound[i] - min_bound[i] for i in range(3)]
    max_extent = max(max(raw_extent), 1e-6)
    padding = max_extent * 0.02
    for axis in range(3):
        if raw_extent[axis] <= max_extent * 1e-6:
            min_bound[axis] -= max_extent * 0.05
            max_bound[axis] += max_extent * 0.05
        else:
            min_bound[axis] -= padding
            max_bound[axis] += padding
    extent = [max_bound[i] - min_bound[i] for i in range(3)]
    return min_bound, max_bound, extent, max(extent)


def _create_coordinate_box(o3d, min_bound: list[float], max_bound: list[float]):
    xmin, ymin, zmin = min_bound
    xmax, ymax, zmax = max_bound
    points = [
        [xmin, ymin, zmin],
        [xmax, ymin, zmin],
        [xmax, ymax, zmin],
        [xmin, ymax, zmin],
        [xmin, ymin, zmax],
        [xmax, ymin, zmax],
        [xmax, ymax, zmax],
        [xmin, ymax, zmax],
    ]
    lines = [
        [0, 1],
        [3, 2],
        [4, 5],
        [7, 6],
        [0, 3],
        [1, 2],
        [4, 7],
        [5, 6],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    x_color = [0.95, 0.12, 0.12]
    y_color = [0.12, 0.75, 0.18]
    z_color = [0.18, 0.35, 1.0]
    colors = [x_color, x_color, x_color, x_color, y_color, y_color, y_color, y_color, z_color, z_color, z_color, z_color]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def _create_volume_grid(o3d, min_bound: list[float], max_bound: list[float], divisions: int):
    xmin, ymin, zmin = min_bound
    xmax, ymax, zmax = max_bound
    divisions = max(int(divisions), 1)
    xs = [xmin + (xmax - xmin) * index / divisions for index in range(divisions + 1)]
    ys = [ymin + (ymax - ymin) * index / divisions for index in range(divisions + 1)]
    zs = [zmin + (zmax - zmin) * index / divisions for index in range(divisions + 1)]

    points: list[list[float]] = []
    lines: list[list[int]] = []
    colors: list[list[float]] = []
    grid_color = [0.68, 0.68, 0.68]

    def add_line(start_point: list[float], end_point: list[float]) -> None:
        start = len(points)
        points.extend([start_point, end_point])
        lines.append([start, start + 1])
        colors.append(grid_color)

    for y in ys:
        for z in zs:
            add_line([xmin, y, z], [xmax, y, z])
    for x in xs:
        for z in zs:
            add_line([x, ymin, z], [x, ymax, z])
    for x in xs:
        for y in ys:
            add_line([x, y, zmin], [x, y, zmax])

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def _create_axis_ticks(o3d, min_bound: list[float], max_bound: list[float], tick_count: int):
    xmin, ymin, zmin = min_bound
    xmax, ymax, zmax = max_bound
    extent = [xmax - xmin, ymax - ymin, zmax - zmin]
    max_extent = max(max(extent), 1e-6)
    tick_length = max_extent * 0.025
    colors_by_axis = [
        [1.0, 0.22, 0.22],
        [0.18, 0.85, 0.24],
        [0.24, 0.45, 1.0],
    ]
    points: list[list[float]] = []
    lines: list[list[int]] = []
    colors: list[list[float]] = []

    tick_count = max(int(tick_count), 1)
    for axis in range(3):
        for index in range(tick_count + 1):
            t = index / tick_count
            if axis == 0:
                point = [xmin + extent[0] * t, ymin, zmin]
                tick_end = [point[0], ymin - tick_length, zmin]
            elif axis == 1:
                point = [xmin, ymin + extent[1] * t, zmin]
                tick_end = [xmin - tick_length, point[1], zmin]
            else:
                point = [xmin, ymin, zmin + extent[2] * t]
                tick_end = [xmin - tick_length, ymin, point[2]]
            start = len(points)
            points.extend([point, tick_end])
            lines.append([start, start + 1])
            colors.append(colors_by_axis[axis])

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def _create_camera_frame(o3d, max_extent: float):
    size = max(max_extent * 0.12, 1e-6)
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0.0, 0.0, 0.0])


def _create_camera_marker(o3d, max_extent: float):
    marker_radius = max(max_extent * 0.018, 1e-6)
    frustum_depth = max(max_extent * 0.16, 1e-6)
    half_width = frustum_depth * 0.45
    half_height = frustum_depth * 0.28

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=marker_radius, resolution=16)
    sphere.paint_uniform_color([1.0, 0.82, 0.1])
    sphere.translate([0.0, 0.0, 0.0])

    points = [
        [0.0, 0.0, 0.0],
        [-half_width, -half_height, frustum_depth],
        [half_width, -half_height, frustum_depth],
        [half_width, half_height, frustum_depth],
        [-half_width, half_height, frustum_depth],
        [0.0, 0.0, frustum_depth * 1.25],
    ]
    lines = [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 1],
        [0, 5],
    ]
    colors = [[1.0, 0.82, 0.1]] * len(lines)
    frustum = o3d.geometry.LineSet()
    frustum.points = o3d.utility.Vector3dVector(points)
    frustum.lines = o3d.utility.Vector2iVector(lines)
    frustum.colors = o3d.utility.Vector3dVector(colors)
    return sphere, frustum


def _format_length(value: float, unit: str) -> str:
    if abs(value) >= 100:
        return f"{value:.0f} {unit}"
    if abs(value) >= 10:
        return f"{value:.1f} {unit}"
    return f"{value:.2f} {unit}"


def _dimension_labels(np, min_bound: list[float], max_bound: list[float], extent: list[float], max_extent: float, unit: str, tick_count: int):
    xmin, ymin, zmin = min_bound
    xmax, ymax, zmax = max_bound
    offset = max(max_extent * 0.035, 1e-6)
    tick_count = max(int(tick_count), 1)
    tick_spacing = [value / tick_count for value in extent]
    labels = [
        (
            np.array([(xmin + xmax) * 0.5, ymin - offset, zmin - offset], dtype=np.float32),
            f"X 总长 = {_format_length(extent[0], unit)}",
        ),
        (
            np.array([xmin - offset, (ymin + ymax) * 0.5, zmin - offset], dtype=np.float32),
            f"Y 总长 = {_format_length(extent[1], unit)}",
        ),
        (
            np.array([xmin - offset, ymin - offset, (zmin + zmax) * 0.5], dtype=np.float32),
            f"Z 总长 = {_format_length(extent[2], unit)}",
        ),
        (
            np.array([(xmin + xmax) * 0.5, ymin - offset * 2.2, zmin - offset], dtype=np.float32),
            f"X 子刻度间距 = {_format_length(tick_spacing[0], unit)}",
        ),
        (
            np.array([xmin - offset * 2.2, (ymin + ymax) * 0.5, zmin - offset], dtype=np.float32),
            f"Y 子刻度间距 = {_format_length(tick_spacing[1], unit)}",
        ),
        (
            np.array([xmin - offset * 2.2, ymin - offset, (zmin + zmax) * 0.5], dtype=np.float32),
            f"Z 子刻度间距 = {_format_length(tick_spacing[2], unit)}",
        ),
        (
            np.array([max_extent * 0.04, max_extent * 0.04, max_extent * 0.04], dtype=np.float32),
            "相机点位 (0, 0, 0)",
        ),
    ]
    return labels


def _create_text_billboard(o3d, np, text: str, position, max_extent: float, color=(0.0, 0.0, 0.0)):
    Image, ImageDraw, font = _load_text_tools()
    probe = Image.new("L", (4, 4), 0)
    draw = ImageDraw.Draw(probe)
    bbox = draw.textbbox((0, 0), text, font=font)
    width = max(bbox[2] - bbox[0] + 8, 1)
    height = max(bbox[3] - bbox[1] + 8, 1)
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    draw.text((4 - bbox[0], 4 - bbox[1]), text, fill=255, font=font)

    mask = np.asarray(image, dtype=np.uint8)
    ys, xs = np.nonzero(mask > 64)
    if xs.size == 0:
        cloud = o3d.geometry.PointCloud()
        return cloud

    stride = max(1, int(max(width, height) / 180))
    xs = xs[::stride]
    ys = ys[::stride]
    scale = max_extent * 0.00055
    centered_x = (xs.astype(float) - width / 2.0) * scale
    centered_z = -(ys.astype(float) - height / 2.0) * scale
    base = np.asarray(position, dtype=float).reshape(3)
    points = np.column_stack(
        (
            base[0] + centered_x,
            np.full_like(centered_x, base[1]),
            base[2] + centered_z,
        )
    )
    colors = np.tile(np.asarray(color, dtype=float).reshape(1, 3), (points.shape[0], 1))
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud


def _create_text_geometries(o3d, np, labels: list[tuple[object, str]], max_extent: float):
    geometries = []
    for index, (position, text) in enumerate(labels, start=1):
        geometries.append((f"标注 {index}", _create_text_billboard(o3d, np, text, position, max_extent)))
    return geometries


def _combined_view_bounds(min_bound: list[float], max_bound: list[float], max_extent: float) -> tuple[list[float], float]:
    marker_margin = max_extent * 0.22
    view_min = [min(min_bound[index], -marker_margin) for index in range(3)]
    view_max = [max(max_bound[index], marker_margin) for index in range(3)]
    center = [(view_min[index] + view_max[index]) * 0.5 for index in range(3)]
    extent = max(view_max[index] - view_min[index] for index in range(3))
    return center, max(extent, max_extent)


def _choose_point_cloud_file() -> Path | None:
    try:
        from tkinter import Tk, filedialog

        root = Tk()
        root.withdraw()
        selected = filedialog.askopenfilename(
            title="打开点云文件",
            filetypes=[
                ("点云文件", "*.ply *.pcd *.xyz *.xyzn *.xyzrgb"),
                ("PLY 文件", "*.ply"),
                ("PCD 文件", "*.pcd"),
                ("所有文件", "*.*"),
            ],
        )
        root.destroy()
        return Path(selected).expanduser().resolve() if selected else None
    except Exception as exc:
        raise RuntimeError(f"无法打开文件选择窗口：{exc}") from exc


def _show_visualizer(o3d, np, geometries: list[tuple[str, object]], title: str, width: int, height: int, center: list[float], max_extent: float) -> None:
    try:
        app = o3d.visualization.gui.Application.instance
        app.initialize()
        visualizer = o3d.visualization.O3DVisualizer(title, width, height)
        visualizer.show_settings = False
        visualizer.show_axes = True
        for name, geometry in geometries:
            visualizer.add_geometry(name, geometry)
        eye = np.array(
            [
                center[0] + max_extent * 1.8,
                center[1] - max_extent * 1.8,
                center[2] + max_extent * 1.25,
            ],
            dtype=np.float32,
        )
        visualizer.setup_camera(60.0, np.asarray(center, dtype=np.float32), eye, np.array([0.0, 0.0, 1.0], dtype=np.float32))
        app.add_window(visualizer)
        app.run()
    except Exception:
        plain_geometries = [geometry for _name, geometry in geometries]
        o3d.visualization.draw_geometries(
            plain_geometries,
            window_name=title,
            width=width,
            height=height,
        )


def open_point_cloud(path: Path, *, width: int = 1280, height: int = 900, check_only: bool = False, tick_count: int = 5) -> None:
    if not path.exists():
        raise FileNotFoundError(f"未找到点云文件：{path}")
    if not path.is_file():
        raise FileNotFoundError(f"点云路径不是文件：{path}")

    o3d = _load_open3d()
    np = _load_numpy()
    cloud = o3d.io.read_point_cloud(str(path))
    if cloud.is_empty():
        raise RuntimeError(f"点云文件为空或无法读取：{path}")

    min_bound, max_bound, extent, max_extent = _expanded_bounds(cloud)
    tick_count = max(int(tick_count), 1)
    coordinate_box = _create_coordinate_box(o3d, min_bound, max_bound)
    volume_grid = _create_volume_grid(o3d, min_bound, max_bound, divisions=tick_count)
    axis_ticks = _create_axis_ticks(o3d, min_bound, max_bound, tick_count=tick_count)
    camera_frame = _create_camera_frame(o3d, max_extent)
    camera_origin, camera_frustum = _create_camera_marker(o3d, max_extent)
    center, view_extent = _combined_view_bounds(min_bound, max_bound, max_extent)
    labels = _dimension_labels(np, min_bound, max_bound, extent, max_extent, unit="毫米", tick_count=tick_count)
    text_geometries = _create_text_geometries(o3d, np, labels, max_extent)

    if check_only:
        print(f"检查通过：{path}")
        print(f"坐标框尺寸：X={extent[0]:.4f} 毫米，Y={extent[1]:.4f} 毫米，Z={extent[2]:.4f} 毫米")
        print(f"子刻度数量：{tick_count} 段")
        for _position, label in labels:
            print(f"标注：{label}")
        return

    geometries = [
        ("点云", cloud),
        ("三维坐标框", coordinate_box),
        ("框内网格线", volume_grid),
        ("子刻度", axis_ticks),
        ("相机点位", camera_origin),
        ("相机视锥", camera_frustum),
        ("相机坐标轴", camera_frame),
        *text_geometries,
    ]
    _show_visualizer(
        o3d,
        np,
        geometries,
        f"点云三维查看器 - {path.name}",
        width,
        height,
        center,
        view_extent,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="点云三维查看器", description="Open3D 交互式点云三维查看器")
    parser.add_argument("point_cloud", nargs="?", help="点云文件路径；不填写时会弹出文件选择窗口")
    parser.add_argument("--width", type=int, default=1280, help="查看窗口宽度")
    parser.add_argument("--height", type=int, default=900, help="查看窗口高度")
    parser.add_argument("--ticks", type=int, default=5, help="每条坐标边的子刻度分段数")
    parser.add_argument("--check", action="store_true", help="只检查 Open3D 和点云文件，不打开三维窗口")
    args = parser.parse_args(argv)

    try:
        point_cloud_path = Path(args.point_cloud).expanduser().resolve() if args.point_cloud else _choose_point_cloud_file()
        if point_cloud_path is None:
            return 0
        open_point_cloud(
            point_cloud_path,
            width=max(args.width, 320),
            height=max(args.height, 240),
            check_only=args.check,
            tick_count=max(args.ticks, 1),
        )
        return 0
    except Exception as exc:
        details = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        _show_error("点云查看器打开失败", details)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

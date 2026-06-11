# DIC 图像采集功能操作说明

本文档用于说明 MVSS Capture 中独立的 DIC 图像采集功能。DIC 采集按钮位于主界面上方“定时拍照”参数区内，在“复位”按钮右侧，按钮文字为“DIC采集”。

## 1. 功能用途

DIC 图像采集模式用于按固定参数快速进入数字图像相关法所需的双目图像采集流程。该模式启动时会自动覆盖当前界面中的部分采集参数，并将参数应用到已连接的海康相机。

DIC 模式会同时输出：

- 左右相机 JPEG 图像序列；
- 左右相机实时 MP4 视频；
- `frames.meta.json`、`meta.json`、采集报告和数据清单。

## 2. 软件默认设置

### 2.1 程序基础默认设置

下表为普通采集模式的仓库基线默认值，用于和 DIC 专用设置区分。软件会保存上一次界面状态；如果刚运行过 DIC 采集，`config.json` 顶层参数可能临时显示 DIC 参数，例如 `trigger_source=Software`、`record_fps=5.0`。DIC 功能的固定默认值以 `config.json` 的 `dic_capture` 段和程序内 `DIC_CAPTURE_CONFIG` 为准。

| 项目 | 默认值 | 说明 |
| --- | --- | --- |
| 数据根目录 | `captures` | 若启用项目管理，数据保存到 `captures/projects/<project_id>/` |
| 项目管理 | 启用 | 自动在项目目录下登记采集会话 |
| 左相机序列号 | `DB0371852` | 见 `config.json` |
| 右相机序列号 | `DB0371907` | 见 `config.json` |
| 相机绑定 | 启用 | 按配置中的左右序列号绑定相机 |
| 普通触发方式 | `Continuous` | 普通采集默认连续自由采流 |
| 普通像素格式 | `Mono16` | 相机原始输出按 Mono16 配置 |
| 普通图像格式 | `jpg` | 图像序列默认扩展名 |
| 普通预览帧率 | `15.0 fps` | DIC 模式会临时使用 DIC 专用帧率 |
| 普通录像帧率 | `19.2 fps` | DIC 模式会临时使用 DIC 专用帧率 |
| 普通录像采集优先模式 | 启用 | DIC 模式会关闭该策略，避免覆盖 DIC 输出设置 |
| 普通时间戳同步拒绝 | 启用 | 普通模式用于同步质量控制；DIC 默认关闭 |

### 2.2 DIC 采集默认设置

DIC 专用设置位于 `config.json` 的 `dic_capture` 段。当前默认值如下：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `trigger_source` | `Software` | 使用软件触发采集 |
| `pixel_format` | `Mono16` | 相机输出 Mono16 |
| `image_format` | `jpg` | 图像序列保存为 JPEG |
| `record_jpeg_quality` | `100` | JPEG 质量最高 |
| `exposure_auto` | `Off` | 关闭自动曝光 |
| `exposure_time_us` | `20000.0` | 曝光时间 20 ms |
| `gain_auto` | `Off` | 关闭自动增益 |
| `gain` | `0.0` | 增益为 0 |
| `roi_width` | `5472` | 满幅宽度 |
| `roi_height` | `3648` | 满幅高度 |
| `roi_offset_x` | `0` | ROI X 偏移 |
| `roi_offset_y` | `0` | ROI Y 偏移 |
| `record_fps` | `5.0` | 录像和采集目标帧率 |
| `interval_capture_seconds` | `0.5` | JPEG 图像序列写入间隔 |
| `interval_capture_count` | `null` | 不限制采集组数，直到手动停止 |
| `record_save_image_sequence` | `true` | 保存左右图像序列 |
| `record_realtime_mp4` | `true` | 同时实时保存 MP4 |
| `auto_make_mp4` | `false` | 不在采集结束后再由图像序列合成 MP4 |
| `preview_fps` | `5.0` | DIC 模式预览帧率 |
| `record_queue_max_items` | `32` | DIC 写入队列容量 |
| `record_queue_force_configured` | `true` | 队列容量按 32 使用，不自动扩展 |
| `chunk_data_enabled` | `true` | 启用相机 Chunk 元数据 |
| `timestamp_reject_enabled` | `false` | DIC 默认不做时间戳拒绝 |
| `record_preview_during_capture` | `false` | 采集时默认不持续刷新预览 |
| `preview_quality_analysis_enabled` | `false` | 关闭预览质量分析 |
| `record_force_image_format` | `true` | 即使相机为 Mono16，也按 JPEG 图像序列保存 |
| `capture_quality_gate.enabled` | `false` | 关闭采集质量闸门 |

> 注意：DIC 默认配置下相机以 `Mono16` 采集，但图像序列按 `jpg` 保存，JPEG/MP4 属于编码后的图像输出。若后续需要保留 16 位原始灰度数据，应另行启用原始数据保存策略，而不是只依赖 JPEG 文件。

## 3. 采集前准备

1. 确认两台海康相机已连接电脑，并能在 MVS 客户端中正常预览。
2. 确认两台相机序列号与 `config.json` 中的 `left_serial`、`right_serial` 一致。
3. 确认保存磁盘有足够空间。满幅双目 JPEG 序列加实时 MP4 会持续写入较多数据，建议使用 SSD。
4. 打开程序：

```powershell
python stereo_capture_only.py
```

也可以双击 `run_capture_only.bat` 启动。

## 4. 操作流程

### 4.1 连接相机

1. 点击“连接相机”。
2. 等待状态栏显示相机连接成功。
3. 如需确认画面，可点击“开始采集”查看预览。

### 4.2 启动 DIC 采集

1. 点击“DIC采集”按钮。
2. 程序会自动应用 DIC 专用参数：
   - 软件触发；
   - Mono16；
   - 曝光 20000 us；
   - 增益 0；
   - 满幅 ROI；
   - Chunk 元数据；
   - 关闭时间戳拒绝；
   - 关闭采集质量闸门。
3. 参数应用成功后，按钮文字变为“停止DIC”。
4. 状态栏显示“DIC 图像采集中”，并显示当前保存目录。

### 4.3 停止 DIC 采集

1. 点击“停止DIC”。
2. 程序停止采集，并等待写入线程整理文件。
3. 完成后状态栏显示“DIC 图像采集完成”。
4. 按钮恢复为“DIC采集”。

## 5. 输出目录和文件

默认情况下，DIC 数据保存到项目目录下的 `videos` 子目录，例如：

```text
captures/
  projects/
    <project_id>/
      videos/
        YYYYMMDD_HHMMSS/
          left/
            part_001/
              left_000001.jpg
              left_000002.jpg
              ...
          right/
            part_001/
              right_000001.jpg
              right_000002.jpg
              ...
          left.mp4
          right.mp4
          frames.meta.json
          meta.json
          record_report.csv
          record_report.html
          exports/
            file_manifest.csv
            capture_summary.json
```

其中：

- `left/`、`right/`：左右相机 JPEG 图像序列；
- `left.mp4`、`right.mp4`：左右相机实时 MP4；
- `frames.meta.json`：逐帧编号、时间戳、路径、温度等元数据；
- `meta.json`：本次 DIC 会话的总体元数据；
- `record_report.csv`、`record_report.html`：采集统计报告；
- `exports/file_manifest.csv`：文件清单；
- `exports/capture_summary.json`：采集摘要。

## 6. 采集过程检查

采集过程中建议关注：

| 检查项 | 正常现象 | 异常处理 |
| --- | --- | --- |
| 状态栏 | 显示 DIC 图像采集中 | 若报错，查看 `logs/capture.log` |
| 帧率 | 接近 5 fps | 若明显偏低，检查曝光、USB 带宽、磁盘速度 |
| 写入队列 | 不应频繁提示队列满 | 若队列满，降低帧率或更换更快 SSD |
| 输出目录 | JPEG 和 MP4 同步增长 | 若只生成部分文件，检查磁盘空间 |
| 相机温度 | 温度无持续超限告警 | 温度过高时暂停采集降温 |

## 7. 常见问题

### 7.1 为什么 DIC 使用 Software 触发？

DIC 默认使用软件触发，便于不接外部触发器时直接采集。若需要严格硬同步，应改为硬件触发并确认两台相机使用同一触发源。

### 7.2 为什么 DIC 默认关闭时间戳拒绝？

两台独立相机的内部时间戳不一定处于同一时间基准。未确认硬同步或统一时间基准前，启用时间戳拒绝可能导致有效帧被误判为不同步。

### 7.3 为什么 `auto_make_mp4=false` 仍然生成 MP4？

DIC 中 `auto_make_mp4=false` 表示不在采集结束后再从图像序列合成 MP4；`record_realtime_mp4=true` 表示采集过程中实时写入 MP4。因此 DIC 会同时保存 JPEG 序列和实时 MP4。

### 7.4 为什么相机是 Mono16，但保存的是 JPEG？

这是 DIC 默认配置要求：`pixel_format=Mono16`，`image_format=jpg`，`record_force_image_format=true`。程序会按 JPEG 图像格式保存图像序列。若实验需要完整 16 位灰度精度，应改用原始数据保存策略。

## 8. 建议的现场流程

1. 开机后先用 MVS 客户端确认相机工作正常。
2. 打开 MVSS Capture，点击“连接相机”。
3. 点击“开始采集”短暂查看画面，确认视场、光照和斑点质量。
4. 停止预览或保持当前状态均可，然后点击“DIC采集”。
5. 采集期间避免移动线缆和切换 USB 设备。
6. 采集完成后点击“停止DIC”。
7. 打开保存目录，检查左右 JPEG 数量、MP4 文件和 `record_report.html`。

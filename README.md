# 海康威视双目相机同步采集 GUI

本目录是采集专用版本，只保留双目相机连接、预览、同步拍照、定时拍照、录像和相机参数设置。相机标定、深度重建、SAM 分割、点云生成与点云查看功能已移除。

## 运行

1. 安装海康机器人 MVS，并确认相机能在 MVS 客户端中正常预览。
2. 安装 Python 依赖：

```powershell
python -m pip install -r requirements.txt
```

3. 双击 `run_capture_only.bat`，或在当前目录运行：

```powershell
python stereo_capture_only.py
```

## 保留功能

- 连接和刷新左右相机。
- 实时预览左右画面，支持鼠标滚轮缩放和一键还原画面。
- 同步拍照，保存左右原始图像和 `meta.json`。
- 定时拍照，可设置间隔秒数和可选张数。
- 连续录像，保存左右 BMP 帧序列，并可在安装 `ffmpeg` 时自动生成 MP4。
- 设置触发源、曝光、增益、白平衡、ROI、图像格式、保存路径和常用预设。
- 显示 FPS、丢帧计数、左右帧号差、磁盘写入状态等采集状态。

## 使用

1. 接入两台相机。
2. 启动程序，点击 `连接相机`。连接只打开相机并配置触发参数，不会自动显示画面。
3. 第一次使用时，如果左右相机顺序不对，退出程序，编辑 `config.json` 中的 `left_serial` 和 `right_serial`。
4. 点击 `开始采集` 开始实时预览；该模式只预览，不录制。
5. 实时采集中可点击 `同步拍照` 保存一组左右图。
6. 停止实时采集后，可使用 `定时拍照` 或 `开始录像`。
7. 按 `F11` 切换全屏，按 `Esc` 退出全屏。

## 配置

采集配置位于 `config.json`。常用字段包括：

```text
left_serial / right_serial         左右相机序列号
save_dir                           图片和录像输出目录
trigger_source                     Software 或 Line0
exposure_auto / exposure_time_us   自动曝光和手动曝光时间
gain_auto / gain                   自动增益和手动增益
roi_width / roi_height             ROI 宽高
roi_offset_x / roi_offset_y        ROI 起点
pixel_format                       相机像素格式，默认 Mono8
image_format                       保存图片格式，默认 bmp
preview_fps                        预览目标帧率
record_fps                         录像目标帧率
record_max_seconds                 录像最长秒数，0 表示不限时
```

`Line0` 模式需要外部硬件脉冲；此模式下程序等待外触发帧，不发送软件触发。

## 保存位置

默认保存到 `captures`：

```text
captures/
  photos/YYYYMMDD_HHMMSS_mmm/
    left.bmp
    right.bmp
    meta.json
  photos/left/
    YYYYMMDD_HHMMSS_mmm_left.bmp
  photos/right/
    YYYYMMDD_HHMMSS_mmm_right.bmp
  videos/YYYYMMDD_HHMMSS/
    left/
      left_000001.bmp
    right/
      right_000001.bmp
    meta.json
    left.mp4
    right.mp4
```

## 注意

`MV-CS200-10UM` 等高分辨率 USB3 相机满幅数据量较大。录像时建议降低 `record_fps`，必要时设置 ROI，并尽量把两台相机接到不同 USB3 控制器。

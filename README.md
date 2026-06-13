# 海康威视双目相机同步采集 GUI

本目录是双目采集专用版本，保留相机连接、实时预览、同步拍照、定时拍照、DIC 采集、录像、参数设置、标定向导和采集质量辅助。程序支持加载并应用已有标定参数；界面中的在线标定求解依赖相邻目录 `MVSS_Biaoding_CalibrationOnly` 中的标定程序接口，若该目录不存在或接口不匹配，仍可使用本程序完成采集和已有标定参数加载。

## 运行

1. 安装海康机器人 MVS，并确认相机能在 MVS 客户端中正常预览。
2. 安装 Python 依赖：

```powershell
python -m pip install -r requirements.txt
```

`requirements.txt` 使用 `opencv-contrib-python`，因为在线标定中的 ArUco/ChArUco 功能依赖 OpenCV contrib 模块。海康 MVS SDK 不在 PyPI 分发，需单独安装并确保 `MvImport` 与 MVS Runtime 路径可用。

3. 双击 `run_capture_only.bat`，或在当前目录运行：

```powershell
python stereo_capture_only.py
```

如需重新打包 Windows 可执行文件，使用 `MVSS_Capture.spec` 或 `MVSS_Capture_v2.spec`。两个 spec 均以 `stereo_capture_only.py` 为入口，并会尝试打包本机已安装的 MVS Runtime DLL。

## 主要功能

- 连接和刷新左右相机，记录相机型号、序列号和固件版本 `DeviceVersion`。
- 实时预览左右画面，支持缩放、平移、ROI 框选、峰值对焦、放大镜、斑马线和直方图。
- 加载已有双目标定文件，支持 K1/K2、D1/D2、R/T，并生成/缓存 stereo rectification maps。
- 预览模式支持 `校正叠加`，将校正后的左右图半透明叠加，并绘制水平参考线。
- 同步拍照、定时拍照、HDR 包围拍照和连续录像均写入 `meta.json`。
- 录像前检查剩余空间，并执行写入速度基准测试；带宽不足时弹窗警告。
- 周期读取相机传感器温度，超阈值告警，并记录到元数据。
- 按项目保存数据，项目级 `project.json` 记录采集会话索引。
- 每次采集生成 `exports/file_manifest.csv` 和 `exports/capture_summary.json`，便于论文数据清单整理。

## 配置

采集配置位于 `config.json`。常用字段包括：

```text
left_serial / right_serial              左右相机序列号
save_dir                                数据根目录
trigger_source                          Software 或 Continuous
exposure_auto / exposure_time_us        自动曝光和手动曝光时间
gain_auto / gain                        自动增益和手动增益
roi_width / roi_height                  ROI 宽高
pixel_format / image_format             相机像素格式和保存格式
preview_fps / record_fps                预览和录像目标帧率
record_disk_benchmark_*                 录像前写入测速设置
temperature_monitor                     温度轮询间隔和告警阈值
hdr_bracketing.ev_offsets               HDR 包围 EV 序列
project                                 项目保存设置
dic_capture                             DIC 专用采集设置
```

当前 GUI 主流程仅启用 `Software` 和 `Continuous` 两种触发方式。历史配置或旧文档中出现的 `Line0`、`Cascade`、`hardware_sync_enabled`、`require_hardware_trigger` 字段会被安全配置逻辑回退为软触发/非硬同步，不代表当前版本已经启用外部硬触发或级联硬同步。

## DIC 采集

当前 `config.json` 中的 DIC 专用配置位于 `dic_capture` 段，默认使用软触发 `Software`、`Mono8` 像素格式、`png` 图像格式、`record_save_image_sequence=true`、`record_realtime_mp4=true`、`auto_make_mp4=false`、`save_raw_frames=true` 和 `raw_frame_format=tiff16`。这表示 DIC 采集会保存左右图像序列并实时写入 MP4，同时保留原始帧输出策略。

当前版本不会按旧说明自动启用 `Cascade` 级联硬触发，也不会默认使用 `Mono16` 和 JPEG。若实验流程要求微秒级硬同步或 16 位原始灰度数据，应先在代码和配置中恢复并验证硬触发链路，再更新现场操作说明。

## 标定向导

程序提供“标定向导”入口，可采集左右相机标定样本，并尝试调用相邻目录 `../MVSS_Biaoding_CalibrationOnly` 中的 `calibration.py` 和 `biaoding_app.py` 完成在线标定与导出。该目录不是当前采集程序的内部模块；如果目录缺失、依赖未安装或接口发生变化，在线标定会报错。

不使用在线标定时，可直接把已有 `left.yaml`、`right.yaml`、`stereo.yaml` 放入 `calib/`，并在 `config.json` 中启用 `calibration.enabled` 后加载使用。

标定文件引用示例：

```json
"calibration": {
  "enabled": true,
  "left_intrinsics": "calib/left.yaml",
  "right_intrinsics": "calib/right.yaml",
  "stereo_params": "calib/stereo.yaml",
  "rectified_overlay_alpha": 0.5,
  "rectified_line_interval_px": 120
}
```

标定文件可使用 OpenCV YAML/XML 或 JSON。常用节点名包括 `K`/`camera_matrix`、`D`/`distortion_coefficients`、`K1`、`D1`、`K2`、`D2`、`R`、`T`、`R1`、`R2`、`P1`、`P2`、`Q`。

## 保存位置

默认保存到 `captures/projects/<project_id>/`：

```text
captures/
  projects/
    20260525_103012_123/
      project.json
      left/
        YYYYMMDD_HHMMSS_mmm_left.bmp
        YYYYMMDD_HHMMSS_mmm_hdr_ev_m2p0_left.bmp
        ...
      right/
        YYYYMMDD_HHMMSS_mmm_right.bmp
        YYYYMMDD_HHMMSS_mmm_hdr_ev_m2p0_right.bmp
        ...
      exports/
        captures/
          YYYYMMDD_HHMMSS_mmm/
            meta.json
            exports/
              file_manifest.csv
              capture_summary.json
          YYYYMMDD_HHMMSS_mmm_hdr/
            meta.json
            exports/
              file_manifest.csv
              capture_summary.json
      videos/
        YYYYMMDD_HHMMSS/
          left/
          right/
          frames.meta.json
          meta.json
          left.mp4
          right.mp4
          exports/
            file_manifest.csv
            capture_summary.json
logs/
  capture.log
```

每次启动程序都会创建新的项目文件夹。同步拍照、定时拍照和 HDR 包围拍照的图片直接保存到项目根目录下的 `left/`、`right/`，不再为每次拍照额外创建图片子文件夹；本次采集的 `meta.json` 和 manifest 保存到 `exports/captures/<capture_id>/`。如果 `project.enabled=false`，程序会退回到旧式 `captures/<mode>/` 目录。

## Configuration Safety

Default timestamp reject thresholds are `10000000` ns (10 ms) for both camera and host timestamps. If both thresholds are set to `0` while rejection is enabled, startup restores the 10 ms defaults to avoid silently accepting unsynchronized stereo pairs.

Calibration is disabled by default until real `calib/left.yaml`, `calib/right.yaml`, and `calib/stereo.yaml` files are supplied. Enable `calibration.enabled` only after placing valid calibration files in `calib/`.

## 注意

`Line0` 模式需要外部硬件脉冲；该模式下程序等待外触发帧，不发送软件触发。`timestamp_reject_enabled=true` 只表示启用同步校验开关；只有 `max_camera_timestamp_delta` 或 `max_host_timestamp_delta` 大于 0 时才会按阈值拒绝不同步帧。两台独立相机未做硬件同步时，建议保持 `max_camera_timestamp_delta=0`，避免比较不同设备的相机内部时间戳导致所有帧都被丢弃。

`MV-CS200-10UM` 等高分辨率 USB3 相机满幅数据量很大。5472 x 3648、双相机、5 fps、BMP 录制约需 200 MB/s 持续写入，建议使用 SSD，并尽量把两台相机接到不同 USB3 控制器。

## Git 与数据文件

仓库只跟踪程序源码、配置模板和文档。`captures/`、`logs/`、`build/`、`dist/`、`.idea/`、打包产物以及 `*.npz` 校正/采集数据默认忽略，不建议提交到 Git。若某个校正文件需要作为发布资产，请先确认体积和复现实验需求，再单独调整忽略规则。

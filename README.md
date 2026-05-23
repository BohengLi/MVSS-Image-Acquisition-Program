# 海康威视双目相机同步采集 GUI

适用目标：两台海康机器人 MVS 工业相机，例如 `MV-CS200-10UM`。程序使用 MVS SDK 做软触发同步，界面全屏显示，左侧为左相机，右侧为右相机，并提供同时拍照和同时录像按钮。

## 安装

1. 安装海康机器人 MVS，并确认相机能在 MVS 客户端中正常预览。
2. 安装 Python 依赖：

```powershell
python -m pip install -r requirements.txt
```

3. 双击 `run.bat`，或在当前目录运行：

```powershell
python app.py
```

## 使用

1. 接入两台相机。
2. 启动程序，点击 `连接相机`。连接只打开相机并配置触发参数，不会自动显示画面。
3. 第一次使用时，如果左右相机顺序不对，退出程序，编辑 `config.json` 中的 `left_serial` 和 `right_serial`，填入界面顶部显示的序列号。本机当前检测到的这台 `MV-CS200-10UM` 序列号为 `DB0371852`，已写入 `left_serial`；接上第二台相机后，程序会自动把另一台作为右相机。也可以手动填写 `right_serial` 固定右相机。
4. 点击 `开始采集`，屏幕开始实时显示左右画面；此模式只预览，不录制视频。
5. 实时采集中也可以点击 `同步拍照`，程序会短暂插入一次同步软触发并保存一组图片。默认采用 BMP 保存，尽量接近 MVS 的未压缩图像保存方式。
6. 停止实时采集后，在 `定时` 区域填写间隔秒数和可选张数，点击 `定时拍照` 后，程序会每 n 秒保存一组左右图；张数留空表示一直拍到手动停止。
7. 点击 `停止采集` 后，可点击 `开始录像`，程序按 `record_fps` 连续软触发并保存左右帧序列；再次点击 `停止录像` 后，如果系统安装了 `ffmpeg` 且 `auto_make_mp4=true`，会自动生成左右两个 MP4。
8. 按 `F11` 可切换全屏，按 `Esc` 退出全屏。
9. 鼠标滚轮可在左右画面上放大/缩小预览，便于对焦观察；该缩放只影响显示，不影响保存图像。点击 `还原画面` 可将左右预览一键恢复到正常缩放。

## 增益设置

顶部工具栏提供增益控制：

```text
GainAuto = Off / Once / Continuous
Gain = 手动增益值，仅 GainAuto=Off 时生效
自动下限 = AutoGainLowerLimit
自动上限 = AutoGainUpperLimit
```

修改参数后点击 `应用增益`，程序会同时写入左右相机。当前两台 `MV-CS200-10UM` 已验证 `Gain`、`GainAuto`、`AutoGainLowerLimit`、`AutoGainUpperLimit` 节点可写。

## 曝光、白平衡、ROI

顶部工具栏新增：

```text
ExposureAuto = Off / Once / Continuous
ExposureTime = 手动曝光时间，单位 us
自动曝光上下限 = 自动曝光范围，部分固件节点可能不可写
白平衡 = BalanceWhiteAuto 与 RGB Ratio，黑白相机通常不可用
ROI = Width Height OffsetX OffsetY
触发 = Software / Line0
```

`Line0` 模式需要外部硬件脉冲；此模式下程序不会发送软件触发，只等待外触发帧。

ROI 会同时写入左右相机。设置 ROI 后可降低 USB 带宽并提高实时帧率；留空表示使用相机当前/默认满幅设置。

## 保存路径和预设

点击 `保存路径` 可选择图片和录像输出目录。预设提供 `室内低光` 和 `室外强光`，也可以修改当前参数后点击 `保存` 覆盖当前预设。

录像前程序会估算 BMP 帧序列的磁盘占用。如果剩余空间不足，会阻止或询问是否继续。

## 运行状态

状态栏会显示实际 FPS、丢帧计数和左右帧号差。左右帧号差接近 0 说明两路帧号同步较好；若持续增大，应检查触发、USB 带宽和丢帧情况。

## 保存位置

默认保存在 `captures`：

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

单次拍照和定时拍照都会同时写入 `photos/YYYYMMDD_HHMMSS_mmm/` 组目录，以及 `photos/left/`、`photos/right/` 两个按相机分类的目录。后者便于直接作为标定输入。

## 相机标定

主界面点击 `相机标定` 会打开独立标定子页面。子页面提供 MATLAB Camera Calibrator 类似的输入项：左图目录、右图目录、输出目录、标定板类型、行列数、方格尺寸、ChArUco 标记尺寸和 ArUco 字典。支持：

```text
chessboard       棋盘格，列/行填写内角点数量，格mm填写方格边长
charuco          ChArUco，列/行填写棋盘方格数量，格mm填写方格边长，码mm填写标记边长
charuco_legacy   旧版 ChArUco 生成方式
circles          对称圆点阵，格mm填写圆心间距
acircles         非对称圆点阵，格mm填写圆心间距
```

点击 `生成标定板` 会打开 `E:\Desktop\SAM3\calibration-pattern-generator\index.html`，可生成棋盘格、ChArUco、ArUco、圆点阵等标定板。拍摄时建议使用 15 组以上不同角度和位置的左右同步图，保证标定板覆盖画面四角和中心。

点击 `导入标定板图片` 可读入标定板原图，程序会优先解析标定板生成器导出的文件名，例如 `chessboard_7x10_25mm`、`charuco_7x5_30mm_22mm_DICT_4X4_50`；如果文件名不含规格，则尝试从图片中识别棋盘格内角点、圆点阵或 ChArUco/ArUco 标记并自动填写行列数。自动识别出的格尺寸或码尺寸仍需按实际打印尺寸核对。

点击 `开始标定` 后，程序会配对左右目录中同一时间戳或序号的图片，检测角点并输出：

```text
captures/calibration/calibration_result.json
captures/calibration/calibration_result.yaml
captures/calibration/calibration_parameters_table.json
captures/calibration/calibration_parameters_table.csv
captures/calibration/rejected_pairs.json
captures/calibration/images/raw_pairs/
captures/calibration/images/corner_detection/
captures/calibration/images/undistortion/
captures/calibration/images/rectification/
captures/calibration/reconstruction/
captures/calibration/plots/
```

结果包含左右相机内参矩阵、焦距、主点、径向/切向畸变、每幅图重投影误差、双目旋转矩阵、平移向量、基线、Essential/Fundamental 矩阵和立体校正参数。`calibration_parameters_table.json/csv` 按“类别 / 内容 / 值 / 文件”保存交付表，覆盖内参 `K,D`、外参 `R,T`、校正参数 `R1,R2,P1,P2,Q`、精度 `reprojection error`、标定图、去畸变图、极线图、分辨率/焦距/基线和标定日期。ChArUco/ArUco 功能需要 `opencv-contrib-python`。

新增可视化输出说明：

```text
images/raw_pairs                  有效标定图原图左右文件 + 左右预览
images/corner_detection           角点检测图，绿色为检测点，红色为重投影点
images/undistortion               去畸变 before/after 图
images/rectification              极线校正图和 rectified pair
reconstruction/disparity_map.png  视差图
reconstruction/confidence_map.png  置信度图
reconstruction/depth_map.png      深度图
reconstruction/disparity.npy       WLS 后视差矩阵，单位 px
reconstruction/raw_disparity.npy   原始视差矩阵，单位 px
reconstruction/confidence.npy      置信度矩阵，范围 0~1
reconstruction/object_mask.png     SAM3 目标分割 mask
reconstruction/object_mask.npy     SAM3 目标分割 mask，0/1
reconstruction/object_mask_preview.png
reconstruction/object_mask_metadata.json
reconstruction/depth_mm.npy       深度矩阵，单位 mm
reconstruction/point_cloud.ply    点云文件
reconstruction/point_cloud_preview.png
reconstruction/reconstruction_result.png
reconstruction/quality_metrics.json 重建质量指标
plots/camera_pose.png             左右相机位姿图
plots/calibration_board_poses.png 标定板三维位姿图
```

深度重建配置位于 `config.json`：

```json
{
  "reconstruction_method": "crestereo",
  "allow_sgbm_fallback": true,
  "crestereo_model_path": "D:\\CREStereo\\models\\crestereo_combined_iter10_720x1280.onnx",
  "use_wls_filter": true,
  "wls_lambda": 8000.0,
  "wls_sigma_color": 1.5,
  "confidence_filter": true,
  "confidence_threshold": 0.35,
  "left_right_consistency_px": 2.0,
  "wls_consistency_px": 2.0
}
```

标定页现在提供 `重建参数` 面板，可直接切换 `auto` / `crestereo` / `sgbm`，选择 ONNX 模型，设置 WLS 开关、`lambda`、`sigma_color`、置信度过滤阈值、左右一致性阈值和 SGBM fallback。点击 `保存参数` 会写回 `config.json`，点击 `开始标定` 也会使用当前面板参数生成诊断重建。

程序启动后会自动做一次重建环境自检；标定页也可点击 `自检` 手动检查。自检项目包括 CREStereo 模型文件、`onnxruntime`、CUDA ExecutionProvider、OpenCV `ximgproc` 和 WLS 接口。标定、独立深度重建和实时深度开始前都会执行自检；如果 CREStereo 不可用但允许 SGBM fallback，会给出警告并继续，否则会阻止任务开始。

可启用 SAM3 单视角目标分割来清理点云。默认 SAM3 路径为 `D:\SAM3`，程序会调用 `D:\SAM3\.venv\Scripts\python.exe` 运行 `sam3_mask_inference.py`，基于左校正图生成 `object_mask`，再用 `object_mask & valid_depth` 过滤深度点，输出更干净的单视角目标点云。标定页可调整 SAM3 开关、prompt 和阈值；更多参数在 `config.json` 中设置，包括 `sam3_checkpoint`、`sam3_top_k`、`sam3_resolution`、`sam3_mask_selection`、`sam3_dilate_pixels` 和 `sam3_erode_pixels`。

当 `reconstruction_method` 为 `crestereo` 时，程序优先使用 CREStereo ONNX 输出视差，并继续执行 WLS 和置信度过滤；如果模型或 `onnxruntime` 不可用且 `allow_sgbm_fallback=true`，会回退到 SGBM + WLS。CREStereo 路径会额外执行左右互换推理生成 right disparity，置信度会融合光度一致性、left-right consistency 和 WLS 相关一致性；如果 left-right consistency 分量整体过低，程序会自动忽略该分量并在结果中写入提示，避免把整张深度图过滤成黑图。重建结果会在 `calibration_result.json` 的 `artifacts.reconstruction` 中记录实际使用的 `method_used`、`wls_filter` 和 `confidence_filter` 状态。

每次重建都会生成 `quality_metrics.json`，并把同一份指标写入结果 JSON。指标包含有效视差区域占比、有效深度点比例、置信度均值和分位数、深度范围统计、点云离群点比例，以及基于标定 RMS、焦距、基线和标定角点视差估算的深度误差联动分析。

标定页的 `独立深度重建` 可选择任意一对左右图、已有 `calibration_result.json` 和输出目录，直接输出视差图、深度图、置信度图、`depth_mm.npy`、`point_cloud.ply` 和 `reconstruction_result.png`，不需要重新标定。

标定结果摘要区和 `独立深度重建` 窗口提供 `打开点云` 按钮，会启动独立的 `point_cloud_viewer.py` 进程并用 Open3D 打开 `point_cloud.ply`，可交互式三维查看、旋转、缩放和平移。主界面还提供 `打开点云文件`，可不依赖当前标定结果，直接弹出中文文件选择窗口打开任意 `.ply/.pcd/.xyz` 点云。

点云查看器会根据点云包围盒自动生成匹配尺寸的三维坐标框、框内网格线和子刻度，X/Y/Z 方向边分别用红/绿/蓝标识，并在三维窗口中标注 X/Y/Z 线框总尺寸、子刻度间距和相机点位。中文尺寸标注由 PIL 先渲染成点阵几何，再作为点云加入 Open3D 场景，避免 Open3D 原生 3D label 中文字体显示为问号。这样即使主程序误用 Python 3.14 启动，点云查看也可通过 `config.json` 中的 `point_cloud_viewer_python` 指向 Python 3.12 解释器运行。该功能需要在 Python 3.12 环境安装 `open3d`；如果缺少依赖，查看器会给出中文提示。

主界面的 `实时深度` 按钮会使用当前 `calibration_output_dir/calibration_result.json` 和重建参数实时采集左右帧，做 rectification、视差/深度计算、WLS 和置信度过滤，并在独立窗口显示校正左图、深度图、视差图和置信度图。实时深度与普通采集、录像、定时拍照互斥。

标定完成后，子页面会显示中文标定摘要、左右相机识别图、绿色检测点、红色重投影点，以及标定板在三维空间中的位置分布。左右识别图可分别用滚轮或 `+` / `-` 按钮放大缩小。

标定界面显示项说明：

```text
单目重投影误差 = 左/右相机单目标定 RMS，越小越好
单目有效图像 = 成功识别标定板的左右配对图像数 / 总配对图像数
双目 RMS = stereoCalibrate 的双目标定 RMS
基线 = 左右相机平移向量长度，单位 mm
内参 / 畸变 = 左相机焦距 fx/fy 摘要；详细左右内参、主点和畸变在右侧参数栏显示
绿色圆点 = 实际检测到的角点
红色十字 = 使用标定参数重投影回图像的角点
三维位置图 = 标定板在左相机坐标系中的位姿，X 向右，Z 向前，-Y 向上
```

## 同步精度说明

本程序默认使用 SDK 软件触发：

```text
TriggerMode = On
TriggerSource = Software
TriggerSoftware
```

两台相机的触发命令由两个线程同时发出，适合一般双目静态/低速采集。若需要严格微秒级同步，建议接外部硬件脉冲到两台相机的 `Line0`，并把 `config.json` 的 `trigger_source` 改为 `Line0`；此时程序负责接收与保存左右帧，实际同步由硬件脉冲完成。

## 注意

`MV-CS200-10UM` 是 2000 万像素 USB3 相机，两台满幅采集数据量很大。预览目标帧率由 `preview_fps` 控制，默认 `15`；实际帧率还受曝光时间、USB 带宽、CPU 图像缩放速度影响。录像时建议降低 `record_fps`，或在 MVS 中设置 ROI，且两台相机尽量接到不同 USB3 控制器。

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from ctypes import POINTER, byref, c_ubyte, cast, create_string_buffer, memset, sizeof
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from config_utils import config_bool

_DLL_DIRECTORIES: list[Any] = []
LOGGER = logging.getLogger("mvss_capture")
_MVS_IMPORT_ATTEMPTS = 3
_MVS_IMPORT_RETRY_DELAY_SECONDS = 0.4
_SDK_LOCK = threading.Lock()


class MvsError(RuntimeError):
    pass


class FrameTimeoutError(MvsError):
    pass


class FrameSyncError(MvsError):
    pass


def _mvs_runtime_candidates() -> list[Path]:
    if sys.maxsize > 2**32:
        return [
            Path(r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64"),
            Path(r"C:\Program Files\MVS\Runtime\Win64_x64"),
            Path(r"C:\Program Files\MVS\Development\Samples\Python\MvImport"),
        ]
    return [
        Path(r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win32_i86"),
        Path(r"C:\Program Files\MVS\Runtime\Win32_i86"),
        Path(r"C:\Program Files\MVS\Development\Samples\Python\MvImport"),
    ]


def _add_mvs_runtime_path() -> None:
    for path in _mvs_runtime_candidates():
        if path.exists():
            try:
                _DLL_DIRECTORIES.append(os.add_dll_directory(str(path)))
            except (AttributeError, OSError):
                os.environ["PATH"] = str(path) + os.pathsep + os.environ.get("PATH", "")


def _mvs_python_candidates() -> list[Path]:
    candidates = [
        Path(r"C:\Program Files (x86)\MVS\Development\Samples\Python\MvImport"),
        Path(r"C:\Program Files\MVS\Development\Samples\Python\MvImport"),
        Path(r"C:\Program Files (x86)\MVS\Development\Samples\Python"),
        Path(r"C:\Program Files\MVS\Development\Samples\Python"),
    ]

    try:
        package_spec = find_spec("hikrobot")
        if package_spec and package_spec.submodule_search_locations:
            hikrobot_dir = Path(next(iter(package_spec.submodule_search_locations)))
            candidates.extend([hikrobot_dir, hikrobot_dir / "MvImport"])
    except Exception:
        pass

    return candidates


def _add_mvs_python_path() -> None:
    for path in _mvs_python_candidates():
        if path.exists():
            value = str(path)
            if value not in sys.path:
                sys.path.insert(0, value)


def _format_path_diagnostics(paths: list[Path]) -> str:
    return "; ".join(f"{path} ({'exists' if path.exists() else 'missing'})" for path in paths)


def _mvs_import_error_message(exc: BaseException) -> str:
    arch = "64-bit" if sys.maxsize > 2**32 else "32-bit"
    return (
        "无法加载海康 MVS Python SDK，已重试 "
        f"{_MVS_IMPORT_ATTEMPTS} 次仍失败。\n"
        f"Python: {sys.version.split()[0]} {arch}\n"
        "请确认：1) 已安装与 Python 位数匹配的 MVS；"
        "2) MVS Python 示例目录 MvImport 存在；"
        "3) 运行 `python -m pip install -r requirements.txt` 安装依赖；"
        "4) 如安装在非默认目录，请把 MvImport 目录加入 PYTHONPATH，"
        "并把 MVS Runtime 目录加入 PATH。\n"
        "已检查的 Python SDK 路径: "
        f"{_format_path_diagnostics(_mvs_python_candidates())}\n"
        "已检查的 Runtime 路径: "
        f"{_format_path_diagnostics(_mvs_runtime_candidates())}\n"
        f"原始错误: {type(exc).__name__}: {exc}"
    )


def _load_mvs_imports_once():
    _add_mvs_runtime_path()
    _add_mvs_python_path()
    try:
        from MvImport.CameraParams_const import MV_ACCESS_Exclusive, MV_GIGE_DEVICE, MV_USB_DEVICE
        from MvImport.CameraParams_header import (
            MV_CC_DEVICE_INFO,
            MV_CC_DEVICE_INFO_LIST,
            MV_CC_PIXEL_CONVERT_PARAM,
            MV_FRAME_OUT_INFO_EX,
            MVCC_INTVALUE,
        )
        try:
            from MvImport.CameraParams_header import MVCC_FLOATVALUE, MVCC_STRINGVALUE
        except Exception:
            MVCC_FLOATVALUE = None
            MVCC_STRINGVALUE = None
        from MvImport.MvCameraControl_class import MvCamera
        from MvImport.PixelType_header import PixelType_Gvsp_Mono8, PixelType_Gvsp_RGB8_Packed

        return {
            "MV_GIGE_DEVICE": MV_GIGE_DEVICE,
            "MV_USB_DEVICE": MV_USB_DEVICE,
            "MV_ACCESS_Exclusive": MV_ACCESS_Exclusive,
            "MV_CC_DEVICE_INFO": MV_CC_DEVICE_INFO,
            "MV_CC_DEVICE_INFO_LIST": MV_CC_DEVICE_INFO_LIST,
            "MV_CC_PIXEL_CONVERT_PARAM": MV_CC_PIXEL_CONVERT_PARAM,
            "MV_FRAME_OUT_INFO_EX": MV_FRAME_OUT_INFO_EX,
            "MVCC_INTVALUE": MVCC_INTVALUE,
            "MVCC_FLOATVALUE": MVCC_FLOATVALUE,
            "MVCC_STRINGVALUE": MVCC_STRINGVALUE,
            "MvCamera": MvCamera,
            "PixelType_Gvsp_Mono8": PixelType_Gvsp_Mono8,
            "PixelType_Gvsp_RGB8_Packed": PixelType_Gvsp_RGB8_Packed,
        }
    except Exception as exc:
        raise MvsError(
            "无法加载海康 MVS Python SDK。请先安装 MVS，并执行 "
            "`python -m pip install -r requirements.txt`。原始错误: "
            f"{exc}"
        ) from exc


def _load_mvs_imports():
    last_exc: BaseException | None = None
    for attempt in range(1, _MVS_IMPORT_ATTEMPTS + 1):
        try:
            return _load_mvs_imports_once()
        except Exception as exc:
            last_exc = exc
            if attempt < _MVS_IMPORT_ATTEMPTS:
                time.sleep(_MVS_IMPORT_RETRY_DELAY_SECONDS)
    if last_exc is None:
        last_exc = RuntimeError("unknown MVS SDK import failure")
    raise MvsError(_mvs_import_error_message(last_exc)) from last_exc


SDK: dict[str, Any] | None = None


def sdk() -> dict[str, Any]:
    global SDK
    with _SDK_LOCK:
        if SDK is None:
            SDK = _load_mvs_imports()
    return SDK


def _decode_c_ubyte_array(value: Any) -> str:
    data = bytes(bytearray(value))
    return data.split(b"\x00", 1)[0].decode("utf-8", errors="ignore").strip()


@dataclass(frozen=True)
class CameraInfo:
    index: int
    serial: str
    model: str
    user_name: str
    transport: str

    @property
    def label(self) -> str:
        name = self.user_name or self.model or "Camera"
        return f"{name} / {self.serial}"


@dataclass
class Frame:
    image: Image.Image
    frame_number: int
    width: int
    height: int
    host_timestamp: int
    camera_timestamp: int


@dataclass
class CameraParameters:
    trigger_source: str = "Software"
    exposure_time_us: float = 10000.0
    exposure_auto: str = "Off"
    auto_exposure_lower_limit: float | None = None
    auto_exposure_upper_limit: float | None = None
    gain: float = 0.0
    gain_auto: str = "Off"
    auto_gain_lower_limit: float | None = None
    auto_gain_upper_limit: float | None = None
    balance_white_auto: str = "Off"
    balance_ratio_red: float | None = None
    balance_ratio_green: float | None = None
    balance_ratio_blue: float | None = None
    roi_width: int | None = None
    roi_height: int | None = None
    roi_offset_x: int = 0
    roi_offset_y: int = 0


@dataclass(frozen=True)
class IntNodeInfo:
    current: int
    minimum: int
    maximum: int
    increment: int


@dataclass(frozen=True)
class RoiApplyResult:
    warnings: list[str]
    actual_roi: tuple[int, int, int, int] | None = None

    def __iter__(self):
        return iter(self.warnings)

    def __len__(self) -> int:
        return len(self.warnings)

    def __bool__(self) -> bool:
        return bool(self.warnings)


def enumerate_cameras() -> tuple[list[CameraInfo], Any]:
    s = sdk()
    dev_list = s["MV_CC_DEVICE_INFO_LIST"]()
    ret = s["MvCamera"].MV_CC_EnumDevices(s["MV_GIGE_DEVICE"] | s["MV_USB_DEVICE"], dev_list)
    if ret != 0:
        raise MvsError(f"枚举相机失败: 0x{ret:08x}")

    cameras: list[CameraInfo] = []
    for index in range(dev_list.nDeviceNum):
        info = cast(dev_list.pDeviceInfo[index], POINTER(s["MV_CC_DEVICE_INFO"])).contents
        if info.nTLayerType == s["MV_GIGE_DEVICE"]:
            detail = info.SpecialInfo.stGigEInfo
            transport = "GigE"
            serial = _decode_c_ubyte_array(detail.chSerialNumber)
            model = _decode_c_ubyte_array(detail.chModelName)
            user_name = _decode_c_ubyte_array(detail.chUserDefinedName)
        elif info.nTLayerType == s["MV_USB_DEVICE"]:
            detail = info.SpecialInfo.stUsb3VInfo
            transport = "USB3"
            serial = _decode_c_ubyte_array(detail.chSerialNumber)
            model = _decode_c_ubyte_array(detail.chModelName)
            user_name = _decode_c_ubyte_array(detail.chUserDefinedName)
        else:
            transport = f"TLayer-{info.nTLayerType}"
            serial = ""
            model = ""
            user_name = ""

        cameras.append(
            CameraInfo(
                index=index,
                serial=serial,
                model=model,
                user_name=user_name,
                transport=transport,
            )
        )
    return cameras, dev_list


def _select_stereo_devices_by_serial(
    cameras: list[CameraInfo],
    dev_list: Any,
    left_serial: str = "",
    right_serial: str = "",
) -> tuple[CameraInfo, CameraInfo, Any]:
    by_serial = {cam.serial: cam for cam in cameras if cam.serial}
    if not left_serial and not right_serial:
        left, right = cameras[0], cameras[1]
    else:
        available = ", ".join(cam.serial for cam in cameras)
        if left_serial:
            try:
                left = by_serial[left_serial]
            except KeyError as exc:
                raise MvsError(f"指定左相机序列号未找到: {left_serial}. 当前相机: {available}") from exc
            right = by_serial.get(right_serial) if right_serial else next(
                (cam for cam in cameras if cam.serial != left.serial), None
            )
            if right is None:
                raise MvsError(f"未找到可作为右相机的第二台设备。当前相机: {available}")
        else:
            try:
                right = by_serial[right_serial]
            except KeyError as exc:
                raise MvsError(f"指定右相机序列号未找到: {right_serial}. 当前相机: {available}") from exc
            left = next((cam for cam in cameras if cam.serial != right.serial), None)
            if left is None:
                raise MvsError(f"未找到可作为左相机的第二台设备。当前相机: {available}")

    if left.serial == right.serial:
        raise MvsError("左右相机序列号相同，请检查 config.json。")
    return left, right, dev_list


def select_stereo_devices(
    left_serial: str = "",
    right_serial: str = "",
    bind_serials: bool = False,
) -> tuple[CameraInfo, CameraInfo, Any]:
    cameras, dev_list = enumerate_cameras()
    if len(cameras) < 2:
        raise MvsError(f"至少需要两台相机，当前检测到 {len(cameras)} 台。")
    if not bind_serials:
        return cameras[0], cameras[1], dev_list
    return _select_stereo_devices_by_serial(cameras, dev_list, left_serial, right_serial)


def select_capture_devices(
    left_serial: str = "",
    right_serial: str = "",
    allow_single: bool = False,
    bind_serials: bool = False,
) -> tuple[CameraInfo | None, CameraInfo | None, Any]:
    cameras, dev_list = enumerate_cameras()
    if not cameras:
        raise MvsError("未检测到相机。")
    if len(cameras) >= 2:
        if not bind_serials:
            return cameras[0], cameras[1], dev_list
        return _select_stereo_devices_by_serial(cameras, dev_list, left_serial, right_serial)
    if not allow_single:
        raise MvsError(f"至少需要两台相机，当前检测到 {len(cameras)} 台。")

    camera = cameras[0]
    if bind_serials and right_serial and camera.serial == right_serial and camera.serial != left_serial:
        return None, camera, dev_list
    return camera, None, dev_list


class MvsCamera:
    def __init__(self, device_list: Any, info: CameraInfo):
        s = sdk()
        self.info = info
        self._device_info = cast(device_list.pDeviceInfo[info.index], POINTER(s["MV_CC_DEVICE_INFO"])).contents
        self._cam = s["MvCamera"]()
        self._payload_size = 0
        self._payload_lock = threading.Lock()
        self._grab_lock = threading.Lock()
        self._opened = False
        self._grabbing = False
        self._float_node_cache: dict[tuple[str, ...], str] = {}
        self._float_node_cache_lock = threading.Lock()

    def open(self) -> None:
        s = sdk()
        ret = self._cam.MV_CC_CreateHandle(self._device_info)
        if ret != 0:
            raise MvsError(f"{self.info.label} 创建句柄失败: 0x{ret:08x}")
        ret = self._cam.MV_CC_OpenDevice(s["MV_ACCESS_Exclusive"], 0)
        if ret != 0:
            self._cam.MV_CC_DestroyHandle()
            raise MvsError(f"{self.info.label} 打开失败，可能被 MVS 占用: 0x{ret:08x}")
        self._opened = True

    def configure(
        self,
        trigger_source: str,
        exposure_time_us: float | None = None,
        gain: float | None = None,
        pixel_format: str | None = "Mono8",
        gain_auto: str = "Off",
        auto_gain_lower_limit: float | None = None,
        auto_gain_upper_limit: float | None = None,
        exposure_auto: str = "Off",
        auto_exposure_lower_limit: float | None = None,
        auto_exposure_upper_limit: float | None = None,
        balance_white_auto: str = "Off",
        balance_ratio_red: float | None = None,
        balance_ratio_green: float | None = None,
        balance_ratio_blue: float | None = None,
        roi_width: int | None = None,
        roi_height: int | None = None,
        roi_offset_x: int = 0,
        roi_offset_y: int = 0,
    ) -> None:
        self._try_set_enum_by_string("AcquisitionMode", "Continuous")
        self._try_set_enum_by_string("TriggerSelector", "FrameStart")
        self._try_set_enum_by_string("TriggerMode", "On")
        self.apply_trigger_settings(trigger_source)

        if pixel_format:
            self._try_set_enum_by_string("PixelFormat", pixel_format)
        self.apply_roi_settings(roi_width, roi_height, roi_offset_x, roi_offset_y, restart_stream=False)
        self.apply_exposure_settings(
            exposure_auto,
            exposure_time_us,
            auto_exposure_lower_limit,
            auto_exposure_upper_limit,
        )
        self.apply_gain_settings(gain_auto, gain, auto_gain_lower_limit, auto_gain_upper_limit)
        self.apply_white_balance_settings(
            balance_white_auto,
            balance_ratio_red,
            balance_ratio_green,
            balance_ratio_blue,
        )

        self._set_payload_size(self._get_int("PayloadSize"))

    def apply_trigger_settings(self, trigger_source: str) -> list[str]:
        source = self._normalize_trigger_source(trigger_source)
        warnings: list[str] = []
        if source == "Software":
            if not self._try_set_enum_by_string("TriggerSource", "Software"):
                if not self._try_set_enum("TriggerSource", 7):
                    warnings.append(f"{self.info.label}: TriggerSource=Software 设置失败")
        elif source == "Line0":
            if not self._try_set_enum_by_string("TriggerSource", "Line0"):
                if not self._try_set_enum("TriggerSource", 0):
                    warnings.append(f"{self.info.label}: TriggerSource=Line0 设置失败")
            self._try_set_enum_by_string("TriggerActivation", "RisingEdge")
        else:
            warnings.append(f"{self.info.label}: 不支持的触发源 {trigger_source}")
        return warnings

    def _normalize_trigger_source(self, trigger_source: str) -> str:
        value = str(trigger_source).strip().lower()
        if value in {"line0", "line 0", "hardware", "硬件", "外触发"}:
            return "Line0"
        return "Software"

    def apply_exposure_settings(
        self,
        exposure_auto: str,
        exposure_time_us: float | None,
        auto_exposure_lower_limit: float | None = None,
        auto_exposure_upper_limit: float | None = None,
    ) -> list[str]:
        mode = self._normalize_auto_mode(exposure_auto)
        warnings: list[str] = []
        if auto_exposure_lower_limit is not None:
            if not self._try_set_float_any(
                (
                    "AutoExposureTimeLowerLimit",
                    "ExposureAutoLowerLimit",
                    "AutoExposureLowerLimit",
                    "AutoExposureTimeMin",
                    "ExposureTimeLowerLimit",
                ),
                auto_exposure_lower_limit,
            ):
                warnings.append(f"{self.info.label}: 自动曝光下限节点不可用")
        if auto_exposure_upper_limit is not None:
            if not self._try_set_float_any(
                (
                    "AutoExposureTimeUpperLimit",
                    "ExposureAutoUpperLimit",
                    "AutoExposureUpperLimit",
                    "AutoExposureTimeMax",
                    "ExposureTimeUpperLimit",
                ),
                auto_exposure_upper_limit,
            ):
                warnings.append(f"{self.info.label}: 自动曝光上限节点不可用")
        if not self._try_set_enum_by_string("ExposureAuto", mode):
            warnings.append(f"{self.info.label}: ExposureAuto={mode} 设置失败")
        if mode == "Off" and exposure_time_us is not None and exposure_time_us > 0:
            if not self._try_set_float("ExposureTime", exposure_time_us):
                warnings.append(f"{self.info.label}: ExposureTime={exposure_time_us} 设置失败")
        return warnings

    def apply_gain_settings(
        self,
        gain_auto: str,
        gain: float | None = None,
        auto_gain_lower_limit: float | None = None,
        auto_gain_upper_limit: float | None = None,
    ) -> list[str]:
        mode = self._normalize_auto_mode(gain_auto)
        warnings: list[str] = []

        if auto_gain_lower_limit is not None:
            if not self._try_set_float_any(("AutoGainLowerLimit", "GainAutoLowerLimit"), auto_gain_lower_limit):
                warnings.append(f"{self.info.label}: 自动增益下限节点不可用")
        if auto_gain_upper_limit is not None:
            if not self._try_set_float_any(("AutoGainUpperLimit", "GainAutoUpperLimit"), auto_gain_upper_limit):
                warnings.append(f"{self.info.label}: 自动增益上限节点不可用")

        if not self._try_set_enum_by_string("GainAuto", mode):
            warnings.append(f"{self.info.label}: GainAuto={mode} 设置失败")

        if mode == "Off" and gain is not None and gain >= 0:
            if not self._try_set_float("Gain", gain):
                warnings.append(f"{self.info.label}: 手动增益 Gain={gain} 设置失败")
        return warnings

    def apply_white_balance_settings(
        self,
        balance_white_auto: str,
        red: float | None = None,
        green: float | None = None,
        blue: float | None = None,
    ) -> list[str]:
        mode = self._normalize_auto_mode(balance_white_auto)
        warnings: list[str] = []
        if not self._try_set_enum_by_string("BalanceWhiteAuto", mode):
            warnings.append(f"{self.info.label}: BalanceWhiteAuto 节点不可用或设置失败")
            return warnings
        if mode == "Off":
            for selector, value in (("Red", red), ("Green", green), ("Blue", blue)):
                if value is None:
                    continue
                if not self._try_set_enum_by_string("BalanceRatioSelector", selector):
                    warnings.append(f"{self.info.label}: BalanceRatioSelector={selector} 设置失败")
                    continue
                if not self._try_set_float("BalanceRatio", value):
                    warnings.append(f"{self.info.label}: BalanceRatio {selector}={value} 设置失败")
        return warnings

    def apply_roi_settings(
        self,
        width: int | None,
        height: int | None,
        offset_x: int = 0,
        offset_y: int = 0,
        restart_stream: bool = True,
    ) -> RoiApplyResult:
        warnings: list[str] = []
        if width is None and height is None:
            return RoiApplyResult(warnings)
        was_grabbing = self._grabbing
        if restart_stream and was_grabbing:
            self.stop()
        roi_failed = False
        try:
            self._try_set_int("OffsetX", 0)
            self._try_set_int("OffsetY", 0)
            width, height, normalize_warnings = self._normalize_roi_size(width, height)
            warnings.extend(normalize_warnings)
            if width > 0 and not self._try_set_int("Width", width):
                warnings.append(f"{self.info.label}: Width={width} 设置失败")
            if height > 0 and not self._try_set_int("Height", height):
                warnings.append(f"{self.info.label}: Height={height} 设置失败")
            offset_x, offset_y, normalize_warnings = self._normalize_roi_offset(width, height, offset_x, offset_y)
            warnings.extend(normalize_warnings)
            if offset_x >= 0 and not self._try_set_int("OffsetX", offset_x):
                warnings.append(f"{self.info.label}: OffsetX={offset_x} 设置失败")
            if offset_y >= 0 and not self._try_set_int("OffsetY", offset_y):
                warnings.append(f"{self.info.label}: OffsetY={offset_y} 设置失败")
            try:
                self._set_payload_size(self._get_int("PayloadSize"))
            except MvsError as exc:
                LOGGER.debug("%s: failed to refresh PayloadSize after ROI update: %s", self.info.label, exc, exc_info=True)
        except BaseException:
            roi_failed = True
            raise
        finally:
            if restart_stream and was_grabbing:
                try:
                    self.start()
                except Exception:
                    LOGGER.exception("%s: failed to restart stream after applying ROI.", self.info.label)
                    if not roi_failed:
                        raise
        return RoiApplyResult(warnings, (width, height, offset_x, offset_y))

    def _normalize_roi_size(
        self,
        width: int | None,
        height: int | None,
    ) -> tuple[int, int, list[str]]:
        warnings: list[str] = []
        width_info = self._get_int_info("Width")
        height_info = self._get_int_info("Height")

        requested_width = width if width is not None and width > 0 else width_info.maximum
        requested_height = height if height is not None and height > 0 else height_info.maximum

        norm_width = self._align_to_increment(requested_width, width_info.minimum, width_info.maximum, width_info.increment, "down")
        norm_height = self._align_to_increment(
            requested_height,
            height_info.minimum,
            height_info.maximum,
            height_info.increment,
            "down",
        )

        if norm_width != requested_width or norm_height != requested_height:
            warnings.append(f"{self.info.label}: ROI 尺寸已按节点范围/步进修正为 W={norm_width}, H={norm_height}")
        return norm_width, norm_height, warnings

    def _normalize_roi_offset(
        self,
        width: int,
        height: int,
        offset_x: int,
        offset_y: int,
    ) -> tuple[int, int, list[str]]:
        warnings: list[str] = []
        offset_x_info = self._get_int_info("OffsetX")
        offset_y_info = self._get_int_info("OffsetY")
        requested_offset_x = max(offset_x, 0)
        requested_offset_y = max(offset_y, 0)

        norm_offset_x = self._align_to_increment(
            requested_offset_x,
            offset_x_info.minimum,
            offset_x_info.maximum,
            offset_x_info.increment,
            "down",
        )
        norm_offset_y = self._align_to_increment(
            requested_offset_y,
            offset_y_info.minimum,
            offset_y_info.maximum,
            offset_y_info.increment,
            "down",
        )

        if norm_offset_x != requested_offset_x or norm_offset_y != requested_offset_y:
            warnings.append(f"{self.info.label}: ROI 偏移已按节点范围/步进修正为 X={norm_offset_x}, Y={norm_offset_y}")
        return norm_offset_x, norm_offset_y, warnings

    def _normalize_auto_mode(self, gain_auto: str) -> str:
        value = str(gain_auto).strip().lower()
        mapping = {
            "off": "Off",
            "manual": "Off",
            "手动": "Off",
            "关闭": "Off",
            "once": "Once",
            "一次": "Once",
            "continuous": "Continuous",
            "continue": "Continuous",
            "连续": "Continuous",
            "自动": "Continuous",
            "auto": "Continuous",
        }
        return mapping.get(value, "Off")

    def start(self) -> None:
        if self._grabbing:
            return
        ret = self._cam.MV_CC_StartGrabbing()
        if ret != 0:
            raise MvsError(f"{self.info.label} 开始取流失败: 0x{ret:08x}")
        self._grabbing = True

    def stop(self) -> None:
        if not self._grabbing:
            return
        ret = self._cam.MV_CC_StopGrabbing()
        self._grabbing = False
        if ret != 0:
            raise MvsError(f"{self.info.label} 停止取流失败: 0x{ret:08x}")

    def close(self) -> None:
        try:
            if self._grabbing:
                self.stop()
        finally:
            if self._opened:
                self._cam.MV_CC_CloseDevice()
                self._cam.MV_CC_DestroyHandle()
                self._opened = False

    def trigger_software(self) -> None:
        ret = self._cam.MV_CC_SetCommandValue("TriggerSoftware")
        if ret != 0:
            raise MvsError(f"{self.info.label} 软触发失败: 0x{ret:08x}")

    def device_version(self) -> str | None:
        return self._try_get_string_any(("DeviceVersion", "DeviceFirmwareVersion", "FirmwareVersion"))

    def sensor_temperature(self) -> float | None:
        for selector_key in ("DeviceTemperatureSelector", "TemperatureSelector"):
            self._try_set_enum_by_string(selector_key, "Sensor")
        return self._try_get_float_any(
            (
                "DeviceTemperature",
                "SensorTemperature",
                "TemperatureAbs",
                "Temperature",
                "DeviceTemperatureSensor",
            )
        )

    def _set_payload_size(self, payload_size: int) -> None:
        with self._payload_lock:
            self._payload_size = int(payload_size)

    def _payload_size_snapshot(self) -> int:
        with self._payload_lock:
            payload_size = self._payload_size
            if payload_size <= 0:
                payload_size = self._get_int("PayloadSize")
                self._payload_size = payload_size
            return payload_size

    def grab_frame(self, timeout_ms: int) -> Frame:
        s = sdk()
        with self._grab_lock:
            payload_size = self._payload_size_snapshot()
            raw_buffer = (c_ubyte * payload_size)()
            frame_info = s["MV_FRAME_OUT_INFO_EX"]()
            memset(byref(frame_info), 0, sizeof(frame_info))
            ret = self._cam.MV_CC_GetOneFrameTimeout(raw_buffer, payload_size, frame_info, timeout_ms)
            if ret != 0:
                if ret == 0x80000007:
                    raise FrameTimeoutError(f"{self.info.label} 等待图像超时: 0x{ret:08x}")
                raise MvsError(f"{self.info.label} 获取图像失败: 0x{ret:08x}")

            image = self._frame_to_image(raw_buffer, frame_info)
            camera_timestamp = (int(frame_info.nDevTimeStampHigh) << 32) | int(frame_info.nDevTimeStampLow)
            return Frame(
                image=image,
                frame_number=int(frame_info.nFrameNum),
                width=int(frame_info.nWidth),
                height=int(frame_info.nHeight),
                host_timestamp=int(frame_info.nHostTimeStamp),
                camera_timestamp=camera_timestamp,
            )

    def _frame_to_image(self, raw_buffer: Any, frame_info: Any) -> Image.Image:
        s = sdk()
        width = int(frame_info.nWidth)
        height = int(frame_info.nHeight)
        frame_len = int(frame_info.nFrameLen)
        pixel_type = int(frame_info.enPixelType)

        if pixel_type == s["PixelType_Gvsp_Mono8"] and frame_len >= width * height:
            arr = np.ctypeslib.as_array(raw_buffer)[: width * height].reshape((height, width))
            return Image.fromarray(arr.copy(), mode="L")

        rgb_size = width * height * 3
        rgb_buffer = (c_ubyte * rgb_size)()
        convert = s["MV_CC_PIXEL_CONVERT_PARAM"]()
        memset(byref(convert), 0, sizeof(convert))
        convert.nWidth = width
        convert.nHeight = height
        convert.pSrcData = cast(raw_buffer, POINTER(c_ubyte))
        convert.nSrcDataLen = frame_len
        convert.enSrcPixelType = frame_info.enPixelType
        convert.enDstPixelType = s["PixelType_Gvsp_RGB8_Packed"]
        convert.pDstBuffer = cast(rgb_buffer, POINTER(c_ubyte))
        convert.nDstBufferSize = rgb_size

        ret = self._cam.MV_CC_ConvertPixelType(convert)
        if ret != 0:
            raise MvsError(f"{self.info.label} 像素格式转换失败: 0x{ret:08x}")

        converted_len = int(convert.nDstLen)
        if converted_len < rgb_size:
            raise MvsError(f"{self.info.label} 像素格式转换输出长度异常: {converted_len} < {rgb_size}")
        rgb = np.ctypeslib.as_array(rgb_buffer)[:rgb_size].reshape((height, width, 3))
        return Image.fromarray(rgb, mode="RGB").copy()

    def _get_int(self, key: str) -> int:
        s = sdk()
        value = s["MVCC_INTVALUE"]()
        memset(byref(value), 0, sizeof(value))
        ret = self._cam.MV_CC_GetIntValue(key, value)
        if ret != 0:
            raise MvsError(f"{self.info.label} 读取 {key} 失败: 0x{ret:08x}")
        return int(value.nCurValue)

    def _get_int_info(self, key: str) -> IntNodeInfo:
        s = sdk()
        value = s["MVCC_INTVALUE"]()
        memset(byref(value), 0, sizeof(value))
        ret = self._cam.MV_CC_GetIntValue(key, value)
        if ret != 0:
            raise MvsError(f"{self.info.label} 读取 {key} 范围失败: 0x{ret:08x}")
        increment = int(getattr(value, "nInc", 1) or 1)
        return IntNodeInfo(
            current=int(value.nCurValue),
            minimum=int(value.nMin),
            maximum=int(value.nMax),
            increment=max(increment, 1),
        )

    def _align_to_increment(self, value: int, minimum: int, maximum: int, increment: int, direction: str) -> int:
        if maximum < minimum:
            maximum = minimum
        value = min(max(int(value), minimum), maximum)
        increment = max(int(increment), 1)
        offset = value - minimum
        if direction == "up":
            steps = (offset + increment - 1) // increment
        else:
            steps = offset // increment
        aligned = minimum + steps * increment
        return min(max(aligned, minimum), maximum)

    def _set_enum(self, key: str, value: int) -> None:
        ret = self._cam.MV_CC_SetEnumValue(key, int(value))
        if ret != 0:
            raise MvsError(f"{self.info.label} 设置 {key}={value} 失败: 0x{ret:08x}")

    def _try_set_enum(self, key: str, value: int) -> bool:
        try:
            return self._cam.MV_CC_SetEnumValue(key, int(value)) == 0
        except Exception as exc:
            LOGGER.debug("%s: exception while probing enum node %s=%s: %s", self.info.label, key, value, exc, exc_info=True)
            return False

    def _try_set_enum_by_string(self, key: str, value: str) -> bool:
        try:
            ret = self._cam.MV_CC_SetEnumValueByString(key, value)
            return ret == 0
        except Exception as exc:
            LOGGER.debug("%s: exception while probing enum node %s=%s: %s", self.info.label, key, value, exc, exc_info=True)
            return False

    def _set_float(self, key: str, value: float) -> None:
        ret = self._cam.MV_CC_SetFloatValue(key, float(value))
        if ret != 0:
            raise MvsError(f"{self.info.label} 设置 {key}={value} 失败: 0x{ret:08x}")

    def _try_set_float(self, key: str, value: float) -> bool:
        try:
            return self._cam.MV_CC_SetFloatValue(key, float(value)) == 0
        except Exception as exc:
            LOGGER.debug("%s: exception while probing float node %s=%s: %s", self.info.label, key, value, exc, exc_info=True)
            return False

    def _try_set_float_any(self, keys: tuple[str, ...], value: float) -> bool:
        with self._float_node_cache_lock:
            cached_key = self._float_node_cache.get(keys)
        if cached_key is not None:
            if self._try_set_float(cached_key, value):
                return True
            with self._float_node_cache_lock:
                if self._float_node_cache.get(keys) == cached_key:
                    self._float_node_cache.pop(keys, None)

        for key in keys:
            if self._try_set_float(key, value):
                with self._float_node_cache_lock:
                    self._float_node_cache[keys] = key
                return True
        return False

    def _try_get_float(self, key: str) -> float | None:
        value_cls = sdk().get("MVCC_FLOATVALUE")
        if value_cls is not None:
            try:
                value = value_cls()
                memset(byref(value), 0, sizeof(value))
                ret = self._cam.MV_CC_GetFloatValue(key, value)
                if ret == 0 and hasattr(value, "fCurValue"):
                    return float(value.fCurValue)
            except Exception as exc:
                LOGGER.debug("%s: exception while reading float node %s using struct API: %s", self.info.label, key, exc, exc_info=True)
        try:
            value = self._cam.MV_CC_GetFloatValue(key)
            if isinstance(value, tuple) and value:
                for item in reversed(value):
                    if isinstance(item, (int, float)):
                        return float(item)
                    if hasattr(item, "fCurValue"):
                        return float(item.fCurValue)
            if hasattr(value, "fCurValue"):
                return float(value.fCurValue)
            if isinstance(value, (int, float)):
                return float(value)
        except Exception as exc:
            LOGGER.debug("%s: exception while reading float node %s using direct API: %s", self.info.label, key, exc, exc_info=True)
        try:
            return float(self._get_int(key))
        except Exception as exc:
            LOGGER.debug("%s: exception while reading float node %s as int fallback: %s", self.info.label, key, exc, exc_info=True)
            return None

    def _try_get_float_any(self, keys: tuple[str, ...]) -> float | None:
        for key in keys:
            value = self._try_get_float(key)
            if value is not None:
                return value
        return None

    def _try_get_string(self, key: str) -> str | None:
        value_cls = sdk().get("MVCC_STRINGVALUE")
        if value_cls is not None:
            try:
                value = value_cls()
                memset(byref(value), 0, sizeof(value))
                ret = self._cam.MV_CC_GetStringValue(key, value)
                if ret == 0:
                    text = self._string_from_sdk_value(value)
                    if text:
                        return text
            except Exception as exc:
                LOGGER.debug("%s: exception while reading string node %s using struct API: %s", self.info.label, key, exc, exc_info=True)
        try:
            value = self._cam.MV_CC_GetStringValue(key)
            if isinstance(value, tuple):
                for item in value:
                    text = self._string_from_sdk_value(item)
                    if text:
                        return text
            text = self._string_from_sdk_value(value)
            if text:
                return text
        except Exception as exc:
            LOGGER.debug("%s: exception while reading string node %s using direct API: %s", self.info.label, key, exc, exc_info=True)
        try:
            buffer = create_string_buffer(256)
            ret = self._cam.MV_CC_GetStringValue(key, buffer, sizeof(buffer))
            if ret == 0:
                return buffer.value.decode("utf-8", errors="ignore").strip()
        except Exception as exc:
            LOGGER.debug("%s: exception while reading string node %s using buffer API: %s", self.info.label, key, exc, exc_info=True)
            return None
        return None

    def _try_get_string_any(self, keys: tuple[str, ...]) -> str | None:
        for key in keys:
            value = self._try_get_string(key)
            if value:
                return value
        return None

    def _string_from_sdk_value(self, value: object) -> str | None:
        if value is None:
            return None
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore").strip("\x00").strip()
        if isinstance(value, str):
            return value.strip()
        for attr in ("chCurValue", "chCurString", "strCurValue"):
            if hasattr(value, attr):
                raw = getattr(value, attr)
                if isinstance(raw, bytes):
                    return raw.decode("utf-8", errors="ignore").strip("\x00").strip()
                if isinstance(raw, str):
                    return raw.strip()
                try:
                    return _decode_c_ubyte_array(raw)
                except Exception as exc:
                    LOGGER.debug("failed to decode SDK string value attribute %s: %s", attr, exc, exc_info=True)
                    return None
        return None

    @property
    def is_ready(self) -> bool:
        return self._opened and self._grabbing

    def _try_set_int(self, key: str, value: int) -> bool:
        try:
            return self._cam.MV_CC_SetIntValue(key, int(value)) == 0
        except Exception as exc:
            LOGGER.debug("%s: exception while probing int node %s=%s: %s", self.info.label, key, value, exc, exc_info=True)
            return False


class StereoCameraSystem:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.left: MvsCamera | None = None
        self.right: MvsCamera | None = None
        self.left_info: CameraInfo | None = None
        self.right_info: CameraInfo | None = None
        self.trigger_source = str(config.get("trigger_source", "Software"))
        self.timeout_ms = int(config.get("frame_timeout_ms", 3000))
        self.timestamp_reject_enabled = config_bool(config, "timestamp_reject_enabled", False)
        self.max_camera_timestamp_delta = int(config.get("max_camera_timestamp_delta", 0) or 0)
        self.max_host_timestamp_delta = int(config.get("max_host_timestamp_delta", 0) or 0)
        if self.max_camera_timestamp_delta <= 0 and self.max_host_timestamp_delta <= 0:
            self.timestamp_reject_enabled = False
        self.require_hardware_trigger = config_bool(config, "require_hardware_trigger", False)
        self._camera_timestamp_offset: int | None = None
        self._capture_lock = threading.Lock()

    def connect(self) -> tuple[CameraInfo | None, CameraInfo | None]:
        self._camera_timestamp_offset = None
        left_info, right_info, dev_list = select_capture_devices(
            str(self.config.get("left_serial", "")).strip(),
            str(self.config.get("right_serial", "")).strip(),
            config_bool(self.config, "allow_single_camera", False),
            config_bool(self.config, "bind_camera_serials", False),
        )
        left = MvsCamera(dev_list, left_info) if left_info is not None else None
        right = MvsCamera(dev_list, right_info) if right_info is not None else None
        opened: list[MvsCamera] = []
        try:
            for cam in (left, right):
                if cam is None:
                    continue
                cam.open()
                opened.append(cam)
                cam.configure(
                    trigger_source=self.trigger_source,
                    exposure_time_us=float(self.config.get("exposure_time_us", 0) or 0),
                    gain=float(self.config.get("gain", -1)),
                    pixel_format=str(self.config.get("pixel_format", "Mono8")),
                    gain_auto=str(self.config.get("gain_auto", self.config.get("gain_auto_mode", "Off"))),
                    auto_gain_lower_limit=self._optional_float("auto_gain_lower_limit"),
                    auto_gain_upper_limit=self._optional_float("auto_gain_upper_limit"),
                    exposure_auto=str(self.config.get("exposure_auto", "Off")),
                    auto_exposure_lower_limit=self._optional_float("auto_exposure_lower_limit"),
                    auto_exposure_upper_limit=self._optional_float("auto_exposure_upper_limit"),
                    balance_white_auto=str(self.config.get("balance_white_auto", "Off")),
                    balance_ratio_red=self._optional_float("balance_ratio_red"),
                    balance_ratio_green=self._optional_float("balance_ratio_green"),
                    balance_ratio_blue=self._optional_float("balance_ratio_blue"),
                    roi_width=self._optional_int("roi_width"),
                    roi_height=self._optional_int("roi_height"),
                    roi_offset_x=int(self.config.get("roi_offset_x", 0) or 0),
                    roi_offset_y=int(self.config.get("roi_offset_y", 0) or 0),
                )
                cam.start()
        except Exception:
            for cam in opened:
                try:
                    cam.close()
                except Exception as close_exc:
                    LOGGER.warning("Failed to close %s after connection failure: %s", cam.info.label, close_exc, exc_info=True)
            raise

        self.left = left
        self.right = right
        self.left_info = left_info
        self.right_info = right_info
        return left_info, right_info

    def _optional_float(self, key: str) -> float | None:
        value = self.config.get(key, None)
        if value in (None, ""):
            return None
        return float(value)

    def _optional_int(self, key: str) -> int | None:
        value = self.config.get(key, None)
        if value in (None, ""):
            return None
        return int(value)

    def close(self) -> None:
        errors: list[str] = []
        for cam in (self.left, self.right):
            if cam is None:
                continue
            try:
                cam.close()
            except Exception as exc:
                errors.append(str(exc))
        self.left = None
        self.right = None
        if errors:
            raise MvsError("; ".join(errors))

    def _connected_cameras(self) -> list[tuple[str, MvsCamera]]:
        cameras: list[tuple[str, MvsCamera]] = []
        if self.left is not None and self.left.is_ready:
            cameras.append(("left", self.left))
        if self.right is not None and self.right.is_ready:
            cameras.append(("right", self.right))
        return cameras

    def has_ready_camera(self) -> bool:
        return bool(self._connected_cameras())

    def device_versions(self) -> dict[str, str | None]:
        with self._capture_lock:
            return {name: cam.device_version() for name, cam in self._connected_cameras()}

    def sensor_temperatures(self) -> dict[str, float | None]:
        with self._capture_lock:
            return {name: cam.sensor_temperature() for name, cam in self._connected_cameras()}

    def apply_gain_settings(
        self,
        gain_auto: str,
        gain: float | None,
        auto_gain_lower_limit: float | None,
        auto_gain_upper_limit: float | None,
    ) -> list[str]:
        cameras = self._connected_cameras()
        if not cameras:
            raise MvsError("相机尚未连接。")
        with self._capture_lock:
            warnings: list[str] = []
            for _name, cam in cameras:
                warnings.extend(cam.apply_gain_settings(gain_auto, gain, auto_gain_lower_limit, auto_gain_upper_limit))
            return warnings

    def apply_exposure_settings(
        self,
        exposure_auto: str,
        exposure_time_us: float | None,
        auto_exposure_lower_limit: float | None,
        auto_exposure_upper_limit: float | None,
    ) -> list[str]:
        cameras = self._connected_cameras()
        if not cameras:
            raise MvsError("相机尚未连接。")
        with self._capture_lock:
            warnings: list[str] = []
            for _name, cam in cameras:
                warnings.extend(
                    cam.apply_exposure_settings(
                        exposure_auto,
                        exposure_time_us,
                        auto_exposure_lower_limit,
                        auto_exposure_upper_limit,
                    )
                )
            return warnings

    def apply_white_balance_settings(
        self,
        balance_white_auto: str,
        red: float | None,
        green: float | None,
        blue: float | None,
    ) -> list[str]:
        cameras = self._connected_cameras()
        if not cameras:
            raise MvsError("相机尚未连接。")
        with self._capture_lock:
            warnings: list[str] = []
            for _name, cam in cameras:
                warnings.extend(cam.apply_white_balance_settings(balance_white_auto, red, green, blue))
            return warnings

    def apply_roi_settings(
        self,
        width: int | None,
        height: int | None,
        offset_x: int,
        offset_y: int,
        restart_stream: bool = True,
    ) -> RoiApplyResult:
        cameras = self._connected_cameras()
        if not cameras:
            raise MvsError("相机尚未连接。")
        with self._capture_lock:
            warnings: list[str] = []
            results: dict[str, RoiApplyResult] = {}
            for name, cam in cameras:
                result = cam.apply_roi_settings(width, height, offset_x, offset_y, restart_stream=restart_stream)
                results[name] = result
                warnings.extend(result)
            left_result = results.get("left")
            right_result = results.get("right")
            actual_roi = next((result.actual_roi for result in results.values() if result.actual_roi), None)
            if (
                left_result is not None
                and right_result is not None
                and left_result.actual_roi
                and right_result.actual_roi
                and left_result.actual_roi != right_result.actual_roi
            ):
                warnings.append(
                    "左右相机实际 ROI 不一致："
                    f"左 W={left_result.actual_roi[0]}, H={left_result.actual_roi[1]}, X={left_result.actual_roi[2]}, Y={left_result.actual_roi[3]}；"
                    f"右 W={right_result.actual_roi[0]}, H={right_result.actual_roi[1]}, X={right_result.actual_roi[2]}, Y={right_result.actual_roi[3]}"
                )
            return RoiApplyResult(warnings, actual_roi)

    def apply_trigger_settings(self, trigger_source: str) -> list[str]:
        cameras = self._connected_cameras()
        if not cameras:
            raise MvsError("相机尚未连接。")
        with self._capture_lock:
            warnings: list[str] = []
            for _name, cam in cameras:
                warnings.extend(cam.apply_trigger_settings(trigger_source))
            self.trigger_source = trigger_source
            return warnings

    def capture_pair(self) -> tuple[Frame | None, Frame | None, float]:
        with self._capture_lock:
            return self._capture_pair_locked()

    def _capture_pair_locked(self) -> tuple[Frame | None, Frame | None, float]:
        cameras = self._connected_cameras()
        if not cameras:
            raise MvsError("相机尚未连接。")

        if self.require_hardware_trigger and self.trigger_source.lower() == "software":
            raise MvsError("Hardware trigger is required, but trigger_source is Software.")

        if self.trigger_source.lower() == "software":
            if len(cameras) == 1:
                trigger_time = time.time()
                cameras[0][1].trigger_software()
            else:
                barrier = threading.Barrier(len(cameras) + 1)
                errors: list[BaseException] = []

                def fire(cam: MvsCamera) -> None:
                    try:
                        barrier.wait()
                        cam.trigger_software()
                    except BaseException as exc:
                        errors.append(exc)

                trigger_threads = [threading.Thread(target=fire, args=(cam,), daemon=True) for _name, cam in cameras]
                for thread in trigger_threads:
                    thread.start()
                trigger_time = time.time()
                barrier.wait()
                for thread in trigger_threads:
                    thread.join()
                if errors:
                    message = "; ".join(str(exc) for exc in errors)
                    if all(isinstance(exc, FrameTimeoutError) for exc in errors):
                        raise FrameTimeoutError(message)
                    raise MvsError(message)
        else:
            trigger_time = time.time()

        frames: dict[str, Frame] = {}
        errors: list[BaseException] = []

        def grab(name: str, cam: MvsCamera) -> None:
            try:
                frames[name] = cam.grab_frame(self.timeout_ms)
            except BaseException as exc:
                errors.append(exc)

        grab_threads = [threading.Thread(target=grab, args=(name, cam), daemon=True) for name, cam in cameras]
        for thread in grab_threads:
            thread.start()
        for thread in grab_threads:
            thread.join()

        if errors:
            message = "; ".join(str(exc) for exc in errors)
            if all(isinstance(exc, FrameTimeoutError) for exc in errors):
                raise FrameTimeoutError(message)
            raise MvsError(message)
        left = frames.get("left")
        right = frames.get("right")
        if left is not None and right is not None:
            self._validate_frame_sync(left, right)
        return left, right, trigger_time

    def _validate_frame_sync(self, left: Frame, right: Frame) -> None:
        if not self.timestamp_reject_enabled:
            return
        issues: list[str] = []
        if self.max_camera_timestamp_delta > 0:
            raw_camera_delta = int(left.camera_timestamp) - int(right.camera_timestamp)
            if self._camera_timestamp_offset is None:
                self._camera_timestamp_offset = raw_camera_delta
            camera_delta = abs(raw_camera_delta - self._camera_timestamp_offset)
            if camera_delta > self.max_camera_timestamp_delta:
                issues.append(
                    "camera timestamp drift "
                    f"{camera_delta} exceeds {self.max_camera_timestamp_delta} "
                    f"(offset {self._camera_timestamp_offset})"
                )
        if self.max_host_timestamp_delta > 0:
            host_delta = abs(int(left.host_timestamp) - int(right.host_timestamp))
            if host_delta > self.max_host_timestamp_delta:
                issues.append(f"host timestamp delta {host_delta} exceeds {self.max_host_timestamp_delta}")
        if issues:
            raise FrameSyncError("Stereo frame rejected: " + "; ".join(issues))

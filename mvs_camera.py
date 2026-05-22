from __future__ import annotations

import os
import sys
import threading
import time
from ctypes import POINTER, byref, c_ubyte, cast, memset, sizeof, string_at
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

_DLL_DIRECTORIES: list[Any] = []


class MvsError(RuntimeError):
    pass


def _add_mvs_runtime_path() -> None:
    candidates: list[Path] = []
    if sys.maxsize > 2**32:
        candidates.extend(
            [
                Path(r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64"),
                Path(r"C:\Program Files\MVS\Runtime\Win64_x64"),
                Path(r"C:\Program Files\MVS\Development\Samples\Python\MvImport"),
            ]
        )
    else:
        candidates.extend(
            [
                Path(r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win32_i86"),
                Path(r"C:\Program Files\MVS\Runtime\Win32_i86"),
                Path(r"C:\Program Files\MVS\Development\Samples\Python\MvImport"),
            ]
        )

    for path in candidates:
        if path.exists():
            try:
                _DLL_DIRECTORIES.append(os.add_dll_directory(str(path)))
            except (AttributeError, OSError):
                os.environ["PATH"] = str(path) + os.pathsep + os.environ.get("PATH", "")


def _add_mvs_python_path() -> None:
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

    for path in candidates:
        if path.exists():
            value = str(path)
            if value not in sys.path:
                sys.path.insert(0, value)


def _load_mvs_imports():
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


SDK: dict[str, Any] | None = None


def sdk() -> dict[str, Any]:
    global SDK
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


def select_stereo_devices(left_serial: str = "", right_serial: str = "") -> tuple[CameraInfo, CameraInfo, Any]:
    cameras, dev_list = enumerate_cameras()
    if len(cameras) < 2:
        raise MvsError(f"至少需要两台相机，当前检测到 {len(cameras)} 台。")

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


class MvsCamera:
    def __init__(self, device_list: Any, info: CameraInfo):
        s = sdk()
        self.info = info
        self._device_info = cast(device_list.pDeviceInfo[info.index], POINTER(s["MV_CC_DEVICE_INFO"])).contents
        self._cam = s["MvCamera"]()
        self._payload_size = 0
        self._grab_lock = threading.Lock()
        self._opened = False
        self._grabbing = False

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

        self._payload_size = self._get_int("PayloadSize")

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
    ) -> list[str]:
        warnings: list[str] = []
        if width is None and height is None:
            return warnings
        was_grabbing = self._grabbing
        if restart_stream and was_grabbing:
            self.stop()
        try:
            self._try_set_int("OffsetX", 0)
            self._try_set_int("OffsetY", 0)
            if width is not None and width > 0 and not self._try_set_int("Width", width):
                warnings.append(f"{self.info.label}: Width={width} 设置失败")
            if height is not None and height > 0 and not self._try_set_int("Height", height):
                warnings.append(f"{self.info.label}: Height={height} 设置失败")
            if offset_x >= 0 and not self._try_set_int("OffsetX", offset_x):
                warnings.append(f"{self.info.label}: OffsetX={offset_x} 设置失败")
            if offset_y >= 0 and not self._try_set_int("OffsetY", offset_y):
                warnings.append(f"{self.info.label}: OffsetY={offset_y} 设置失败")
            try:
                self._payload_size = self._get_int("PayloadSize")
            except MvsError:
                pass
        finally:
            if restart_stream and was_grabbing:
                self.start()
        return warnings

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

    def grab_frame(self, timeout_ms: int) -> Frame:
        s = sdk()
        with self._grab_lock:
            if self._payload_size <= 0:
                self._payload_size = self._get_int("PayloadSize")
            raw_buffer = (c_ubyte * self._payload_size)()
            frame_info = s["MV_FRAME_OUT_INFO_EX"]()
            memset(byref(frame_info), 0, sizeof(frame_info))
            ret = self._cam.MV_CC_GetOneFrameTimeout(raw_buffer, self._payload_size, frame_info, timeout_ms)
            if ret != 0:
                raise MvsError(f"{self.info.label} 等待图像超时或失败: 0x{ret:08x}")

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

        rgb = string_at(convert.pDstBuffer, int(convert.nDstLen))
        return Image.frombytes("RGB", (width, height), rgb)

    def _get_int(self, key: str) -> int:
        s = sdk()
        value = s["MVCC_INTVALUE"]()
        memset(byref(value), 0, sizeof(value))
        ret = self._cam.MV_CC_GetIntValue(key, value)
        if ret != 0:
            raise MvsError(f"{self.info.label} 读取 {key} 失败: 0x{ret:08x}")
        return int(value.nCurValue)

    def _set_enum(self, key: str, value: int) -> None:
        ret = self._cam.MV_CC_SetEnumValue(key, int(value))
        if ret != 0:
            raise MvsError(f"{self.info.label} 设置 {key}={value} 失败: 0x{ret:08x}")

    def _try_set_enum(self, key: str, value: int) -> bool:
        try:
            return self._cam.MV_CC_SetEnumValue(key, int(value)) == 0
        except Exception:
            return False

    def _try_set_enum_by_string(self, key: str, value: str) -> bool:
        try:
            ret = self._cam.MV_CC_SetEnumValueByString(key, value)
            return ret == 0
        except Exception:
            return False

    def _set_float(self, key: str, value: float) -> None:
        ret = self._cam.MV_CC_SetFloatValue(key, float(value))
        if ret != 0:
            raise MvsError(f"{self.info.label} 设置 {key}={value} 失败: 0x{ret:08x}")

    def _try_set_float(self, key: str, value: float) -> bool:
        try:
            return self._cam.MV_CC_SetFloatValue(key, float(value)) == 0
        except Exception:
            return False

    def _try_set_float_any(self, keys: tuple[str, ...], value: float) -> bool:
        return any(self._try_set_float(key, value) for key in keys)

    def _try_set_int(self, key: str, value: int) -> bool:
        try:
            return self._cam.MV_CC_SetIntValue(key, int(value)) == 0
        except Exception:
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
        self._capture_lock = threading.Lock()

    def connect(self) -> tuple[CameraInfo, CameraInfo]:
        left_info, right_info, dev_list = select_stereo_devices(
            str(self.config.get("left_serial", "")).strip(),
            str(self.config.get("right_serial", "")).strip(),
        )
        left = MvsCamera(dev_list, left_info)
        right = MvsCamera(dev_list, right_info)
        try:
            left.open()
            right.open()
            for cam in (left, right):
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
            left.close()
            right.close()
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

    def apply_gain_settings(
        self,
        gain_auto: str,
        gain: float | None,
        auto_gain_lower_limit: float | None,
        auto_gain_upper_limit: float | None,
    ) -> list[str]:
        if self.left is None or self.right is None:
            raise MvsError("相机尚未连接。")
        with self._capture_lock:
            warnings: list[str] = []
            warnings.extend(
                self.left.apply_gain_settings(gain_auto, gain, auto_gain_lower_limit, auto_gain_upper_limit)
            )
            warnings.extend(
                self.right.apply_gain_settings(gain_auto, gain, auto_gain_lower_limit, auto_gain_upper_limit)
            )
            return warnings

    def apply_exposure_settings(
        self,
        exposure_auto: str,
        exposure_time_us: float | None,
        auto_exposure_lower_limit: float | None,
        auto_exposure_upper_limit: float | None,
    ) -> list[str]:
        if self.left is None or self.right is None:
            raise MvsError("相机尚未连接。")
        with self._capture_lock:
            warnings: list[str] = []
            warnings.extend(
                self.left.apply_exposure_settings(
                    exposure_auto,
                    exposure_time_us,
                    auto_exposure_lower_limit,
                    auto_exposure_upper_limit,
                )
            )
            warnings.extend(
                self.right.apply_exposure_settings(
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
        if self.left is None or self.right is None:
            raise MvsError("相机尚未连接。")
        with self._capture_lock:
            warnings: list[str] = []
            warnings.extend(self.left.apply_white_balance_settings(balance_white_auto, red, green, blue))
            warnings.extend(self.right.apply_white_balance_settings(balance_white_auto, red, green, blue))
            return warnings

    def apply_roi_settings(
        self,
        width: int | None,
        height: int | None,
        offset_x: int,
        offset_y: int,
    ) -> list[str]:
        if self.left is None or self.right is None:
            raise MvsError("相机尚未连接。")
        with self._capture_lock:
            warnings: list[str] = []
            warnings.extend(self.left.apply_roi_settings(width, height, offset_x, offset_y))
            warnings.extend(self.right.apply_roi_settings(width, height, offset_x, offset_y))
            return warnings

    def apply_trigger_settings(self, trigger_source: str) -> list[str]:
        if self.left is None or self.right is None:
            raise MvsError("相机尚未连接。")
        with self._capture_lock:
            warnings: list[str] = []
            warnings.extend(self.left.apply_trigger_settings(trigger_source))
            warnings.extend(self.right.apply_trigger_settings(trigger_source))
            self.trigger_source = trigger_source
            return warnings

    def capture_pair(self) -> tuple[Frame, Frame, float]:
        with self._capture_lock:
            return self._capture_pair_locked()

    def _capture_pair_locked(self) -> tuple[Frame, Frame, float]:
        if self.left is None or self.right is None:
            raise MvsError("相机尚未连接。")

        if self.trigger_source.lower() == "software":
            barrier = threading.Barrier(3)
            errors: list[BaseException] = []

            def fire(cam: MvsCamera) -> None:
                try:
                    barrier.wait()
                    cam.trigger_software()
                except BaseException as exc:
                    errors.append(exc)

            t_left = threading.Thread(target=fire, args=(self.left,), daemon=True)
            t_right = threading.Thread(target=fire, args=(self.right,), daemon=True)
            t_left.start()
            t_right.start()
            trigger_time = time.time()
            barrier.wait()
            t_left.join()
            t_right.join()
            if errors:
                raise MvsError("; ".join(str(exc) for exc in errors))
        else:
            trigger_time = time.time()

        frames: dict[str, Frame] = {}
        errors: list[BaseException] = []

        def grab(name: str, cam: MvsCamera) -> None:
            try:
                frames[name] = cam.grab_frame(self.timeout_ms)
            except BaseException as exc:
                errors.append(exc)

        g_left = threading.Thread(target=grab, args=("left", self.left), daemon=True)
        g_right = threading.Thread(target=grab, args=("right", self.right), daemon=True)
        g_left.start()
        g_right.start()
        g_left.join()
        g_right.join()

        if errors:
            raise MvsError("; ".join(str(exc) for exc in errors))
        return frames["left"], frames["right"], trigger_time

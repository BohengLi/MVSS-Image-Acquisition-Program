from __future__ import annotations

import logging
import os
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait
from ctypes import CFUNCTYPE, POINTER, byref, c_ubyte, c_void_p, cast, create_string_buffer, memset, memmove, sizeof, string_at
from dataclasses import asdict, dataclass, field
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from config_utils import config_bool, config_float, config_int

_DLL_DIRECTORIES: list[Any] = []
LOGGER = logging.getLogger("mvss_capture")
_MVS_IMPORT_ATTEMPTS = 3
_MVS_IMPORT_RETRY_DELAY_SECONDS = 0.4
_SDK_LOCK = threading.Lock()
DEFAULT_CHUNK_SELECTORS = ("Timestamp", "Framecounter", "FrameCounter", "ExposureTime", "Gain")
DEFAULT_HOST_TIMESTAMP_DELTA_NS = 10_000_000
DEFAULT_STREAM_BUFFER_SIZE = 128
DEFAULT_RAW_BUFFER_POOL_SIZE = 64
MVS_ERROR_CODES = {
    0x80000000: ("MV_E_HANDLE", "错误或无效的句柄"),
    0x80000001: ("MV_E_SUPPORT", "不支持的功能"),
    0x80000002: ("MV_E_BUFOVER", "缓存已满"),
    0x80000003: ("MV_E_CALLORDER", "接口调用顺序错误"),
    0x80000004: ("MV_E_PARAMETER", "参数错误"),
    0x80000006: ("MV_E_RESOURCE", "资源申请失败"),
    0x80000007: ("MV_E_NODATA", "无数据/等待图像超时"),
    0x80000008: ("MV_E_PRECONDITION", "前置条件错误或运行环境变化"),
    0x80000009: ("MV_E_VERSION", "版本不匹配"),
    0x8000000A: ("MV_E_NOENOUGH_BUF", "传入缓存空间不足"),
    0x8000000B: ("MV_E_ABNORMAL_IMAGE", "异常图像，可能丢包或数据不完整"),
    0x8000000C: ("MV_E_LOAD_LIBRARY", "动态库加载失败"),
    0x8000000D: ("MV_E_NOOUTBUF", "无可输出缓存"),
    0x800000FF: ("MV_E_UNKNOW", "未知错误"),
    0x80000100: ("MV_E_GC_GENERIC", "GenICam 通用错误"),
    0x80000101: ("MV_E_GC_ARGUMENT", "GenICam 参数错误"),
    0x80000102: ("MV_E_GC_RANGE", "GenICam 值超出范围"),
    0x80000103: ("MV_E_GC_PROPERTY", "GenICam 属性错误"),
    0x80000104: ("MV_E_GC_RUNTIME", "GenICam 运行时错误"),
    0x80000105: ("MV_E_GC_LOGICAL", "GenICam 逻辑错误"),
    0x80000106: ("MV_E_GC_ACCESS", "GenICam 节点访问失败"),
    0x80000107: ("MV_E_GC_TIMEOUT", "GenICam 超时"),
    0x80000108: ("MV_E_GC_DYNAMICCAST", "GenICam 类型转换失败"),
    0x80000200: ("MV_E_NOT_IMPLEMENTED", "命令不支持"),
    0x80000201: ("MV_E_INVALID_ADDRESS", "访问地址无效"),
    0x80000202: ("MV_E_WRITE_PROTECT", "写保护"),
    0x80000203: ("MV_E_ACCESS_DENIED", "访问权限不足"),
    0x80000204: ("MV_E_BUSY", "设备忙或网络忙"),
    0x80000205: ("MV_E_PACKET", "网络数据包错误"),
    0x80000206: ("MV_E_NETER", "网络相关错误"),
}


class MvsError(RuntimeError):
    pass


class FrameTimeoutError(MvsError):
    pass


class FrameSyncError(MvsError):
    pass


def _format_mvs_error(ret: int) -> str:
    code = int(ret) & 0xFFFFFFFF
    name, description = MVS_ERROR_CODES.get(code, ("UNKNOWN", "未收录的 SDK 错误码"))
    return f"0x{code:08x} ({name}: {description})"


def _mvs_runtime_candidates() -> list[Path]:
    candidates: list[Path] = []
    bundle_dir = getattr(sys, "_MEIPASS", None)
    if bundle_dir:
        bundle_path = Path(bundle_dir)
        candidates.extend([bundle_path, bundle_path / "ThirdParty"])
    exe_dir = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else None
    if exe_dir is not None and exe_dir != bundle_path:
        candidates.extend([exe_dir, exe_dir / "ThirdParty"])
    if sys.maxsize > 2**32:
        candidates.extend([
            Path(r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64"),
            Path(r"C:\Program Files\MVS\Runtime\Win64_x64"),
            Path(r"C:\Program Files\MVS\Development\Samples\Python\MvImport"),
        ])
    else:
        candidates.extend([
            Path(r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win32_i86"),
            Path(r"C:\Program Files\MVS\Runtime\Win32_i86"),
            Path(r"C:\Program Files\MVS\Development\Samples\Python\MvImport"),
        ])
    return candidates


def _add_mvs_runtime_path() -> None:
    for path in _mvs_runtime_candidates():
        if path.exists():
            add_dll_directory = getattr(os, "add_dll_directory", None)
            if callable(add_dll_directory):
                try:
                    _DLL_DIRECTORIES.append(add_dll_directory(str(path)))
                    continue
                except OSError:
                    LOGGER.debug("os.add_dll_directory failed for %s; falling back to PATH", path, exc_info=True)
            path_text = str(path)
            current_path = os.environ.get("PATH", "")
            if path_text not in current_path.split(os.pathsep):
                os.environ["PATH"] = path_text + os.pathsep + current_path


def _mvs_python_candidates() -> list[Path]:
    candidates = [
        Path(sys.executable).resolve().parent,
        Path(sys.executable).resolve().parent / "hikrobot",
        Path(r"C:\Program Files (x86)\MVS\Development\Samples\Python\MvImport"),
        Path(r"C:\Program Files\MVS\Development\Samples\Python\MvImport"),
        Path(r"C:\Program Files (x86)\MVS\Development\Samples\Python"),
        Path(r"C:\Program Files\MVS\Development\Samples\Python"),
    ]
    bundle_dir = getattr(sys, "_MEIPASS", None)
    if bundle_dir:
        bundle_path = Path(bundle_dir)
        candidates[:0] = [bundle_path, bundle_path / "hikrobot", bundle_path / "hikrobot" / "MvImport"]

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
        try:
            from MvImport.CameraParams_const import MV_Image_Bmp, MV_Image_Jpeg
        except Exception:
            MV_Image_Bmp = None
            MV_Image_Jpeg = None
        from MvImport.CameraParams_header import (
            MV_CC_DEVICE_INFO,
            MV_CC_DEVICE_INFO_LIST,
            MV_CC_PIXEL_CONVERT_PARAM,
            MV_FRAME_OUT_INFO_EX,
            MVCC_INTVALUE,
        )
        try:
            from MvImport.CameraParams_header import MV_FRAME_OUT
        except Exception:
            MV_FRAME_OUT = None
        try:
            from MvImport.CameraParams_header import MV_SAVE_IMAGE_PARAM_EX
        except Exception:
            MV_SAVE_IMAGE_PARAM_EX = None
        try:
            from MvImport.CameraParams_header import MVCC_FLOATVALUE, MVCC_STRINGVALUE
        except Exception:
            MVCC_FLOATVALUE = None
            MVCC_STRINGVALUE = None
        from MvImport.MvCameraControl_class import MvCamera
        from MvImport import PixelType_header as PixelTypeHeader
        from MvImport.PixelType_header import PixelType_Gvsp_Mono8, PixelType_Gvsp_RGB8_Packed
        pixel_type_names = {
            int(value): name
            for name, value in vars(PixelTypeHeader).items()
            if name.startswith("PixelType_") and isinstance(value, int)
        }

        return {
            "MV_GIGE_DEVICE": MV_GIGE_DEVICE,
            "MV_USB_DEVICE": MV_USB_DEVICE,
            "MV_ACCESS_Exclusive": MV_ACCESS_Exclusive,
            "MV_CC_DEVICE_INFO": MV_CC_DEVICE_INFO,
            "MV_CC_DEVICE_INFO_LIST": MV_CC_DEVICE_INFO_LIST,
            "MV_CC_PIXEL_CONVERT_PARAM": MV_CC_PIXEL_CONVERT_PARAM,
            "MV_FRAME_OUT": MV_FRAME_OUT,
            "MV_FRAME_OUT_INFO_EX": MV_FRAME_OUT_INFO_EX,
            "MV_SAVE_IMAGE_PARAM_EX": MV_SAVE_IMAGE_PARAM_EX,
            "MVCC_INTVALUE": MVCC_INTVALUE,
            "MVCC_FLOATVALUE": MVCC_FLOATVALUE,
            "MVCC_STRINGVALUE": MVCC_STRINGVALUE,
            "MvCamera": MvCamera,
            "MV_Image_Bmp": MV_Image_Bmp,
            "MV_Image_Jpeg": MV_Image_Jpeg,
            "PixelType_Gvsp_Mono8": PixelType_Gvsp_Mono8,
            "PixelType_Gvsp_RGB8_Packed": PixelType_Gvsp_RGB8_Packed,
            "PIXEL_TYPE_NAMES": pixel_type_names,
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
    image: Image.Image | None
    frame_number: int
    width: int
    height: int
    host_timestamp: int
    camera_timestamp: int
    raw_data: bytes | bytearray | memoryview | None = None
    raw_frame_len: int = 0
    pixel_type: int = 0
    pixel_type_name: str = ""
    raw_bit_depth: int = 8
    raw_array_shape: tuple[int, ...] | None = None
    _raw_release: Callable[[], None] | None = field(default=None, repr=False, compare=False)

    def release_raw_data(self) -> None:
        release = self._raw_release
        self._raw_release = None
        self.raw_data = None
        if release is not None:
            release()

    def __del__(self) -> None:
        self.release_raw_data()


@dataclass
class RawFramePacket:
    data: bytes | bytearray | memoryview
    frame_len: int
    width: int
    height: int
    pixel_type: int
    frame_number: int
    host_timestamp: int
    camera_timestamp: int
    _raw_release: Callable[[], None] | None = field(default=None, repr=False, compare=False)

    def take_release(self) -> Callable[[], None] | None:
        release = self._raw_release
        self._raw_release = None
        return release

    def release_raw_data(self) -> None:
        release = self._raw_release
        self._raw_release = None
        self.data = b""
        if release is not None:
            release()

    def __del__(self) -> None:
        self.release_raw_data()


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
    black_level: float | None = None
    digital_shift: float | None = None
    gamma: float | None = None
    roi_width: int | None = None
    roi_height: int | None = None
    roi_offset_x: int = 0
    roi_offset_y: int = 0
    chunk_data_enabled: bool = False


@dataclass(frozen=True)
class IntNodeInfo:
    current: int
    minimum: int
    maximum: int
    increment: int


@dataclass(frozen=True)
class StreamStats:
    buffered_frames: int
    dropped_frames: int
    callback_enabled: bool
    link_error_count: int | None = None
    resend_packet_count: int | None = None


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
        raise MvsError(f"枚举相机失败: {_format_mvs_error(ret)}")

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
        self._stream_callback_enabled = False
        self._stream_callback = None
        self._prefer_stream_callback = False
        self._stream_condition = threading.Condition()
        self._stream_frames: deque[RawFramePacket] = deque(maxlen=DEFAULT_STREAM_BUFFER_SIZE)
        self._stream_dropped_frames = 0
        self._raw_buffer_pool: deque[bytearray] = deque()
        self._raw_buffer_pool_limit = DEFAULT_RAW_BUFFER_POOL_SIZE
        self._raw_buffer_pool_lock = threading.Lock()
        self._float_node_cache: dict[tuple[str, ...], str] = {}
        self._float_node_cache_lock = threading.Lock()

    def configure_streaming(
        self,
        stream_buffer_size: int | None = None,
        raw_buffer_pool_size: int | None = None,
        prefer_callback: bool | None = None,
    ) -> None:
        if stream_buffer_size is not None:
            maxlen = max(int(stream_buffer_size), 1)
            with self._stream_condition:
                current = list(self._stream_frames)[-maxlen:]
                self._stream_frames.clear()
                self._stream_frames = deque(current, maxlen=maxlen)
        if raw_buffer_pool_size is not None:
            self._raw_buffer_pool_limit = max(int(raw_buffer_pool_size), 0)
            with self._raw_buffer_pool_lock:
                while len(self._raw_buffer_pool) > self._raw_buffer_pool_limit:
                    self._raw_buffer_pool.pop()
        if prefer_callback is not None:
            self._prefer_stream_callback = bool(prefer_callback)

    def open(self) -> None:
        s = sdk()
        ret = self._cam.MV_CC_CreateHandle(self._device_info)
        if ret != 0:
            raise MvsError(f"{self.info.label} 创建句柄失败: {_format_mvs_error(ret)}")
        ret = self._cam.MV_CC_OpenDevice(s["MV_ACCESS_Exclusive"], 0)
        if ret != 0:
            self._cam.MV_CC_DestroyHandle()
            raise MvsError(f"{self.info.label} 打开失败，可能被 MVS 占用: {_format_mvs_error(ret)}")
        self._opened = True
        self._optimize_gige_packet_size()

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
        chunk_data_enabled: bool = False,
        chunk_selectors: list[str] | tuple[str, ...] | None = None,
        acquisition_frame_rate: float | None = None,
        trigger_delay_us: float | None = None,
        line_debouncer_time_us: float | None = None,
        trigger_activation: str | None = None,
        black_level: float | None = None,
        digital_shift: float | None = None,
        gamma: float | None = None,
    ) -> None:
        self._try_set_enum_by_string("AcquisitionMode", "Continuous")
        self._try_set_enum_by_string("TriggerSelector", "FrameStart")
        self.apply_trigger_settings(trigger_source, trigger_activation=trigger_activation)
        for warning in self.apply_timing_settings(acquisition_frame_rate, trigger_delay_us, line_debouncer_time_us):
            LOGGER.warning(warning)

        if pixel_format:
            for warning in self.apply_pixel_format_settings(pixel_format, restart_stream=True):
                LOGGER.warning(warning)
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
        for warning in self.apply_image_correction_settings(black_level, digital_shift, gamma):
            LOGGER.warning(warning)
        for warning in self.apply_chunk_settings(chunk_data_enabled, chunk_selectors):
            LOGGER.warning(warning)

        self._set_payload_size(self._get_int("PayloadSize"))

    def apply_pixel_format_settings(self, pixel_format: str | None, restart_stream: bool = True) -> list[str]:
        warnings: list[str] = []
        value = str(pixel_format or "").strip()
        if not value:
            return warnings
        was_grabbing = self._grabbing
        if restart_stream and was_grabbing:
            self.stop()
        try:
            if not self._try_set_enum_by_string("PixelFormat", value):
                warnings.append(f"{self.info.label}: PixelFormat={value} 设置失败")
            try:
                self._set_payload_size(self._get_int("PayloadSize"))
            except MvsError as exc:
                LOGGER.debug("%s: failed to refresh PayloadSize after PixelFormat update: %s", self.info.label, exc, exc_info=True)
        finally:
            pending_exc_type = sys.exc_info()[0]
            if restart_stream and was_grabbing:
                try:
                    self.start()
                except Exception:
                    LOGGER.exception("%s: failed to restart stream after applying PixelFormat.", self.info.label)
                    if pending_exc_type is None:
                        raise
        return warnings

    def _optimize_gige_packet_size(self) -> None:
        if self.info.transport != "GigE":
            return
        getter = getattr(self._cam, "MV_CC_GetOptimalPacketSize", None)
        if getter is None:
            return
        try:
            packet_size = int(getter())
        except Exception as exc:
            LOGGER.debug("%s: MV_CC_GetOptimalPacketSize failed: %s", self.info.label, exc, exc_info=True)
            return
        if packet_size <= 0:
            LOGGER.info("%s: optimal GigE packet size unavailable: %s", self.info.label, packet_size)
            return
        if self._try_set_int("GevSCPSPacketSize", packet_size):
            LOGGER.info("%s: GigE packet size optimized to %s.", self.info.label, packet_size)
        else:
            LOGGER.info("%s: failed to set GevSCPSPacketSize=%s.", self.info.label, packet_size)

    def apply_trigger_settings(self, trigger_source: str, trigger_activation: str | None = None) -> list[str]:
        source = self._normalize_trigger_source(trigger_source)
        warnings: list[str] = []
        if source == "Continuous":
            if not self._try_set_enum_by_string("TriggerMode", "Off"):
                warnings.append(f"{self.info.label}: TriggerMode=Off setting failed")
            return warnings
        if not self._try_set_enum_by_string("TriggerMode", "On"):
            warnings.append(f"{self.info.label}: TriggerMode=On setting failed")
        if source == "Software":
            if not self._try_set_enum_by_string("TriggerSource", "Software"):
                if not self._try_set_enum("TriggerSource", 7):
                    warnings.append(f"{self.info.label}: TriggerSource=Software 设置失败")
        elif source == "Line0":
            if not self._try_set_enum_by_string("TriggerSource", "Line0"):
                if not self._try_set_enum("TriggerSource", 0):
                    warnings.append(f"{self.info.label}: TriggerSource=Line0 设置失败")
            activation = self._normalize_trigger_activation(trigger_activation)
            if not self._try_set_enum_by_string("TriggerActivation", activation):
                warnings.append(f"{self.info.label}: TriggerActivation={activation} 设置失败")
        else:
            warnings.append(f"{self.info.label}: 不支持的触发源 {trigger_source}")
        return warnings

    def apply_trigger_output_settings(self, line_selector: str, line_source: str) -> list[str]:
        warnings: list[str] = []
        line = self._normalize_line_selector(line_selector)
        source = self._normalize_line_source(line_source)
        if not self._try_set_enum_by_string("LineSelector", line):
            warnings.append(f"{self.info.label}: LineSelector={line} 设置失败")
        if not self._try_set_enum_by_string("LineMode", "Output"):
            warnings.append(f"{self.info.label}: LineMode=Output 设置失败")
        if not self._try_set_enum_by_string("LineSource", source):
            warnings.append(f"{self.info.label}: LineSource={source} 设置失败")
        return warnings

    def apply_trigger_input_settings(
        self,
        line_selector: str,
        trigger_activation: str | None = None,
    ) -> list[str]:
        warnings: list[str] = []
        line = self._normalize_line_selector(line_selector)
        if not self._try_set_enum_by_string("LineSelector", line):
            warnings.append(f"{self.info.label}: LineSelector={line} 设置失败")
        if not self._try_set_enum_by_string("LineMode", "Input"):
            warnings.append(f"{self.info.label}: LineMode=Input 设置失败")
        warnings.extend(self.apply_trigger_settings(line, trigger_activation=trigger_activation))
        return warnings

    def apply_hardware_cascade_settings(
        self,
        role: str,
        master_line: str = "Line2",
        master_line_source: str = "ExposureActive",
        slave_line: str = "Line0",
        slave_activation: str | None = None,
        master_trigger_source: str = "Software",
        master_trigger_activation: str | None = None,
    ) -> list[str]:
        normalized_role = str(role or "").strip().lower()
        if normalized_role == "master":
            warnings = self.apply_trigger_settings(master_trigger_source, trigger_activation=master_trigger_activation)
            warnings.extend(self.apply_trigger_output_settings(master_line, master_line_source))
            return warnings
        if normalized_role == "slave":
            return self.apply_trigger_input_settings(slave_line, trigger_activation=slave_activation)
        return [f"{self.info.label}: 不支持的硬触发级联角色 {role}"]

    def apply_timing_settings(
        self,
        acquisition_frame_rate: float | None = None,
        trigger_delay_us: float | None = None,
        line_debouncer_time_us: float | None = None,
    ) -> list[str]:
        warnings: list[str] = []
        if acquisition_frame_rate is not None and acquisition_frame_rate > 0:
            if not self._try_set_bool_any(("AcquisitionFrameRateEnable", "AcquisitionFrameRateEnabled"), True):
                LOGGER.debug("%s: AcquisitionFrameRateEnable is not available.", self.info.label)
            if not self._try_set_float_any(("AcquisitionFrameRate", "AcquisitionFrameRateAbs"), acquisition_frame_rate):
                warnings.append(f"{self.info.label}: AcquisitionFrameRate={acquisition_frame_rate} 设置失败")
        if trigger_delay_us is not None and trigger_delay_us >= 0:
            if not self._try_set_float_any(("TriggerDelay", "TriggerDelayAbs"), trigger_delay_us):
                warnings.append(f"{self.info.label}: TriggerDelay={trigger_delay_us} 设置失败")
        if line_debouncer_time_us is not None and line_debouncer_time_us >= 0:
            if not self._try_set_float_any(("LineDebouncerTime", "LineDebouncerTimeAbs"), line_debouncer_time_us):
                warnings.append(f"{self.info.label}: LineDebouncerTime={line_debouncer_time_us} 设置失败")
        return warnings

    def _normalize_trigger_source(self, trigger_source: str) -> str:
        value = str(trigger_source).strip().lower()
        if value in {"line0", "line 0", "hardware", "硬件", "外触发"}:
            return "Line0"
        if value in {"line1", "line 1"}:
            return "Line1"
        if value in {"line2", "line 2"}:
            return "Line2"
        if value in {"line3", "line 3"}:
            return "Line3"
        if value in {"continuous", "freerun", "free-run", "free run", "off", "none", "trigger off", "no trigger"}:
            return "Continuous"
        return "Software"

    def _normalize_line_selector(self, line: str | None) -> str:
        value = str(line or "Line0").strip().lower().replace("_", "").replace("-", "").replace(" ", "")
        mapping = {
            "0": "Line0",
            "line0": "Line0",
            "1": "Line1",
            "line1": "Line1",
            "2": "Line2",
            "line2": "Line2",
            "3": "Line3",
            "line3": "Line3",
        }
        return mapping.get(value, str(line or "Line0").strip() or "Line0")

    def _normalize_line_source(self, source: str | None) -> str:
        value = str(source or "ExposureActive").strip().lower().replace("_", "").replace("-", "").replace(" ", "")
        mapping = {
            "exposureactive": "ExposureActive",
            "exposing": "ExposureActive",
            "strobe": "Strobe",
            "triggerwait": "TriggerWait",
            "useroutput0": "UserOutput0",
            "useroutput1": "UserOutput1",
            "useroutput2": "UserOutput2",
        }
        return mapping.get(value, str(source or "ExposureActive").strip() or "ExposureActive")

    def _normalize_trigger_activation(self, trigger_activation: str | None) -> str:
        value = str(trigger_activation or "RisingEdge").strip().lower()
        mapping = {
            "risingedge": "RisingEdge",
            "rising": "RisingEdge",
            "上升沿": "RisingEdge",
            "fallingedge": "FallingEdge",
            "falling": "FallingEdge",
            "下降沿": "FallingEdge",
            "anyedge": "AnyEdge",
            "any": "AnyEdge",
            "任意沿": "AnyEdge",
            "levelhigh": "LevelHigh",
            "high": "LevelHigh",
            "高电平": "LevelHigh",
            "levellow": "LevelLow",
            "low": "LevelLow",
            "低电平": "LevelLow",
        }
        return mapping.get(value, "RisingEdge")

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

    def apply_image_correction_settings(
        self,
        black_level: float | None = None,
        digital_shift: float | None = None,
        gamma: float | None = None,
    ) -> list[str]:
        warnings: list[str] = []
        if black_level is not None:
            if not self._try_set_number_any(("BlackLevel", "BlackLevelRaw"), black_level):
                warnings.append(f"{self.info.label}: BlackLevel={black_level} 设置失败")
        if digital_shift is not None:
            if not self._try_set_number_any(("DigitalShift",), digital_shift):
                warnings.append(f"{self.info.label}: DigitalShift={digital_shift} 设置失败")
        if gamma is not None:
            self._try_set_bool("GammaEnable", True)
            if not self._try_set_float("Gamma", gamma):
                warnings.append(f"{self.info.label}: Gamma={gamma} 设置失败")
        return warnings

    def _try_set_number_any(self, keys: tuple[str, ...], value: float) -> bool:
        numeric = float(value)
        for key in keys:
            if self._try_set_float(key, numeric):
                return True
            if numeric.is_integer() and self._try_set_int(key, int(numeric)):
                return True
        return False

    def apply_chunk_settings(
        self,
        enabled: bool,
        selectors: list[str] | tuple[str, ...] | None = None,
    ) -> list[str]:
        warnings: list[str] = []
        enabled = bool(enabled)
        if not self._try_set_bool("ChunkModeActive", enabled):
            if enabled:
                warnings.append(f"{self.info.label}: ChunkModeActive=On is not supported or failed")
            return warnings
        if not enabled:
            return warnings

        raw_selectors: list[str] | tuple[str, ...]
        if isinstance(selectors, str):
            raw_selectors = tuple(selectors.replace(";", ",").split(","))
        else:
            raw_selectors = selectors or DEFAULT_CHUNK_SELECTORS
        requested = [str(selector).strip() for selector in raw_selectors if str(selector).strip()]
        enabled_count = 0
        for selector in requested:
            if not self._try_set_enum_by_string("ChunkSelector", selector):
                continue
            if self._try_set_bool("ChunkEnable", True):
                enabled_count += 1
        if requested and enabled_count == 0:
            warnings.append(f"{self.info.label}: ChunkModeActive enabled, but no ChunkSelector could be enabled")
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
        finally:
            pending_exc_type = sys.exc_info()[0]
            if restart_stream and was_grabbing:
                try:
                    self.start()
                except Exception:
                    LOGGER.exception("%s: failed to restart stream after applying ROI.", self.info.label)
                    if pending_exc_type is None:
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
        if self._prefer_stream_callback or not self._supports_image_buffer():
            self._register_stream_callback()
        ret = self._cam.MV_CC_StartGrabbing()
        if ret != 0:
            raise MvsError(f"{self.info.label} 开始取流失败: {_format_mvs_error(ret)}")
        self._grabbing = True

    def stop(self) -> None:
        if not self._grabbing:
            return
        ret = self._cam.MV_CC_StopGrabbing()
        if ret != 0:
            raise MvsError(f"{self.info.label} 停止取流失败: {_format_mvs_error(ret)}")
        self._grabbing = False
        self._clear_stream_frames()

    def close(self) -> None:
        try:
            if self._grabbing:
                self.stop()
        finally:
            if self._opened:
                self._cam.MV_CC_CloseDevice()
                self._cam.MV_CC_DestroyHandle()
                self._opened = False

    def _clear_stream_frames(self) -> None:
        with self._stream_condition:
            for packet in self._stream_frames:
                packet.release_raw_data()
            self._stream_frames.clear()
            self._stream_condition.notify_all()

    def _register_stream_callback(self) -> None:
        if self._stream_callback_enabled:
            return
        register = getattr(self._cam, "MV_CC_RegisterImageCallBackEx", None)
        if register is None:
            return
        s = sdk()
        frame_info_type = s["MV_FRAME_OUT_INFO_EX"]
        callback_type = CFUNCTYPE(None, POINTER(c_ubyte), POINTER(frame_info_type), c_void_p)

        def on_frame(data_ptr, frame_info_ptr, _user) -> None:
            try:
                if not data_ptr or not frame_info_ptr:
                    return
                packet = self._raw_packet_from_pointer(data_ptr, frame_info_ptr.contents)
                with self._stream_condition:
                    if len(self._stream_frames) == self._stream_frames.maxlen:
                        dropped = self._stream_frames.popleft()
                        dropped.release_raw_data()
                        self._stream_dropped_frames += 1
                    self._stream_frames.append(packet)
                    self._stream_condition.notify()
            except Exception:
                LOGGER.exception("%s: image stream callback failed.", self.info.label)

        callback = callback_type(on_frame)
        try:
            ret = register(callback, None)
        except Exception as exc:
            LOGGER.info("%s: stream callback registration failed; using polling grab: %s", self.info.label, exc)
            return
        if ret != 0:
            LOGGER.info("%s: stream callback registration returned %s; using polling grab.", self.info.label, _format_mvs_error(ret))
            return
        self._stream_callback = callback
        self._stream_callback_enabled = True
        LOGGER.info("%s: image stream callback enabled.", self.info.label)

    def _supports_image_buffer(self) -> bool:
        return (
            getattr(self._cam, "MV_CC_GetImageBuffer", None) is not None
            and getattr(self._cam, "MV_CC_FreeImageBuffer", None) is not None
            and sdk().get("MV_FRAME_OUT") is not None
        )

    def _raw_packet_from_pointer(self, data_ptr: Any, frame_info: Any) -> RawFramePacket:
        frame_len = int(getattr(frame_info, "nFrameLen", 0) or 0)
        data, release = self._copy_raw_buffer(data_ptr, frame_len)
        return RawFramePacket(
            data=data,
            frame_len=frame_len,
            width=int(getattr(frame_info, "nWidth", 0) or 0),
            height=int(getattr(frame_info, "nHeight", 0) or 0),
            pixel_type=int(getattr(frame_info, "enPixelType", 0) or 0),
            frame_number=self._frame_number_from_info(frame_info),
            host_timestamp=int(getattr(frame_info, "nHostTimeStamp", 0) or 0),
            camera_timestamp=self._camera_timestamp_from_info(frame_info),
            _raw_release=release,
        )

    def _copy_raw_buffer(self, data_ptr: Any, frame_len: int) -> tuple[bytearray, Callable[[], None] | None]:
        frame_len = max(int(frame_len), 0)
        if frame_len <= 0:
            return bytearray(), None
        buffer = self._acquire_raw_buffer(frame_len)
        memmove((c_ubyte * frame_len).from_buffer(buffer), data_ptr, frame_len)
        released = False

        def release() -> None:
            nonlocal released
            if released:
                return
            released = True
            self._release_raw_buffer(buffer)

        return buffer, release

    def _acquire_raw_buffer(self, min_size: int) -> bytearray:
        min_size = max(int(min_size), 0)
        with self._raw_buffer_pool_lock:
            for _ in range(len(self._raw_buffer_pool)):
                buffer = self._raw_buffer_pool.pop()
                if len(buffer) >= min_size:
                    return buffer
        return bytearray(min_size)

    def _release_raw_buffer(self, buffer: bytearray) -> None:
        if self._raw_buffer_pool_limit <= 0:
            return
        with self._raw_buffer_pool_lock:
            if len(self._raw_buffer_pool) < self._raw_buffer_pool_limit:
                self._raw_buffer_pool.append(buffer)

    def trigger_software(self) -> None:
        ret = self._cam.MV_CC_SetCommandValue("TriggerSoftware")
        if ret != 0:
            raise MvsError(f"{self.info.label} 软触发失败: {_format_mvs_error(ret)}")

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

    def current_throughput_mbps(self) -> float | None:
        bytes_per_second = self._try_get_float_any(
            (
                "DeviceLinkCurrentThroughput",
                "DeviceLinkThroughput",
                "GevSCDMT",
            )
        )
        if bytes_per_second is None:
            return None
        return max(bytes_per_second * 8.0 / 1_000_000.0, 0.0)

    def stream_stats(self) -> StreamStats:
        with self._stream_condition:
            buffered_frames = len(self._stream_frames)
            dropped_frames = self._stream_dropped_frames
            callback_enabled = self._stream_callback_enabled
        link_error_count = self._try_get_int_any(
            (
                "DeviceLinkErrorCount",
                "GevSCPSPacketErrorCount",
                "GevSCPSPacketLostCount",
                "GevStreamChannelPacketErrorCount",
                "GevStreamChannelPacketLostCount",
            )
        )
        resend_packet_count = self._try_get_int_any(
            (
                "GevStreamChannelResendPacketCount",
                "GevResendPacketCount",
                "GevSCPSPacketResendCount",
            )
        )
        return StreamStats(
            buffered_frames=buffered_frames,
            dropped_frames=dropped_frames,
            callback_enabled=callback_enabled,
            link_error_count=link_error_count,
            resend_packet_count=resend_packet_count,
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

    def grab_frame(self, timeout_ms: int, convert_image: bool = True) -> Frame:
        if self._stream_callback_enabled:
            return self._grab_frame_from_stream(timeout_ms, convert_image=convert_image)
        with self._grab_lock:
            frame = self._grab_frame_with_image_buffer(timeout_ms, convert_image=convert_image)
            if frame is not None:
                return frame
            return self._grab_frame_with_timeout(timeout_ms, convert_image=convert_image)

    def grab_raw_frame(self, timeout_ms: int) -> Frame:
        return self.grab_frame(timeout_ms, convert_image=False)

    def _grab_frame_with_image_buffer(self, timeout_ms: int, convert_image: bool = True) -> Frame | None:
        get_buffer = getattr(self._cam, "MV_CC_GetImageBuffer", None)
        free_buffer = getattr(self._cam, "MV_CC_FreeImageBuffer", None)
        frame_out_cls = sdk().get("MV_FRAME_OUT")
        if get_buffer is None or free_buffer is None or frame_out_cls is None:
            return None
        frame_out = frame_out_cls()
        memset(byref(frame_out), 0, sizeof(frame_out))
        ret = get_buffer(frame_out, timeout_ms)
        if ret != 0:
            if ret == 0x80000007:
                raise FrameTimeoutError(f"{self.info.label} 等待图像超时: {_format_mvs_error(ret)}")
            LOGGER.debug("%s: MV_CC_GetImageBuffer failed (%s); falling back to GetOneFrameTimeout.", self.info.label, _format_mvs_error(ret))
            return None
        try:
            frame_info = getattr(frame_out, "stFrameInfo", None)
            data_ptr = getattr(frame_out, "pBufAddr", None)
            if frame_info is None or not data_ptr:
                LOGGER.debug("%s: MV_CC_GetImageBuffer returned empty frame; falling back.", self.info.label)
                return None
            return self._frame_from_packet(self._raw_packet_from_pointer(data_ptr, frame_info), convert_image=convert_image)
        finally:
            try:
                free_buffer(frame_out)
            except Exception as exc:
                LOGGER.debug("%s: MV_CC_FreeImageBuffer failed: %s", self.info.label, exc, exc_info=True)

    def _grab_frame_with_timeout(self, timeout_ms: int, convert_image: bool = True) -> Frame:
        s = sdk()
        payload_size = self._payload_size_snapshot()
        raw_buffer = (c_ubyte * payload_size)()
        frame_info = s["MV_FRAME_OUT_INFO_EX"]()
        memset(byref(frame_info), 0, sizeof(frame_info))
        ret = self._cam.MV_CC_GetOneFrameTimeout(raw_buffer, payload_size, frame_info, timeout_ms)
        if ret != 0:
            if ret == 0x80000007:
                raise FrameTimeoutError(f"{self.info.label} 等待图像超时: {_format_mvs_error(ret)}")
            raise MvsError(f"{self.info.label} 获取图像失败: {_format_mvs_error(ret)}")

        return self._frame_from_packet(self._raw_packet_from_pointer(raw_buffer, frame_info), convert_image=convert_image)

    def _grab_frame_from_stream(self, timeout_ms: int, convert_image: bool = True) -> Frame:
        packets = self.pop_stream_packets(timeout_ms)
        if not packets:
            raise FrameTimeoutError(f"{self.info.label} waiting image stream timed out after {timeout_ms} ms")
        packet = packets[-1]
        for stale in packets[:-1]:
            stale.release_raw_data()
        return self._frame_from_packet(packet, convert_image=convert_image)

    def pop_stream_packets(self, timeout_ms: int) -> list[RawFramePacket]:
        deadline = time.perf_counter() + max(timeout_ms, 1) / 1000.0
        with self._stream_condition:
            while self._grabbing:
                if self._stream_frames:
                    packets = list(self._stream_frames)
                    self._stream_frames.clear()
                    return packets
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                self._stream_condition.wait(remaining)
        return []

    def _camera_timestamp_from_info(self, frame_info: Any) -> int:
        high = int(getattr(frame_info, "nDevTimeStampHigh", 0) or 0)
        low = int(getattr(frame_info, "nDevTimeStampLow", 0) or 0)
        return (high << 32) | low

    def _frame_number_from_info(self, frame_info: Any) -> int:
        try:
            return int(frame_info.nFrameNum)
        except AttributeError:
            return int(frame_info.stFrameInfo.nFrameNum)

    def _frame_from_packet(self, packet: RawFramePacket, convert_image: bool = True) -> Frame:
        pixel_type_name = self._pixel_type_name(packet.pixel_type)
        raw_bit_depth = self._raw_bit_depth(pixel_type_name, packet)
        return Frame(
            image=self._packet_to_image(packet) if convert_image else None,
            frame_number=packet.frame_number,
            width=packet.width,
            height=packet.height,
            host_timestamp=packet.host_timestamp,
            camera_timestamp=packet.camera_timestamp,
            raw_data=packet.data,
            raw_frame_len=packet.frame_len,
            pixel_type=packet.pixel_type,
            pixel_type_name=pixel_type_name,
            raw_bit_depth=raw_bit_depth,
            raw_array_shape=self._raw_array_shape(pixel_type_name, packet, raw_bit_depth),
            _raw_release=packet.take_release(),
        )

    def _pixel_type_name(self, pixel_type: int) -> str:
        names = sdk().get("PIXEL_TYPE_NAMES") or {}
        return str(names.get(int(pixel_type), f"PixelType_0x{int(pixel_type) & 0xFFFFFFFF:08x}"))

    def _raw_bit_depth(self, pixel_type_name: str, packet: RawFramePacket) -> int:
        name = pixel_type_name.lower()
        if "mono16" in name or ("bayer" in name and "16" in name):
            return 16
        if "mono12" in name or ("bayer" in name and "12" in name):
            return 12
        if "mono10" in name or ("bayer" in name and "10" in name):
            return 10
        if packet.width > 0 and packet.height > 0 and packet.frame_len >= packet.width * packet.height * 2:
            return 16
        return 8

    def _raw_array_shape(self, pixel_type_name: str, packet: RawFramePacket, raw_bit_depth: int) -> tuple[int, ...] | None:
        if packet.width <= 0 or packet.height <= 0:
            return None
        name = pixel_type_name.lower()
        if any(token in name for token in ("rgb", "bgr")) and packet.frame_len >= packet.width * packet.height * 3:
            return (packet.height, packet.width, 3)
        if raw_bit_depth > 8 or "mono" in name or "bayer" in name:
            return (packet.height, packet.width)
        return (packet.height, packet.width)

    def _packet_to_image(self, packet: RawFramePacket) -> Image.Image:
        s = sdk()
        expected_len = packet.width * packet.height
        if packet.pixel_type == s["PixelType_Gvsp_Mono8"] and packet.frame_len >= expected_len:
            payload = packet.data if len(packet.data) == expected_len else memoryview(packet.data)[:expected_len]
            return Image.frombytes("L", (packet.width, packet.height), payload)
        pixel_type_name = self._pixel_type_name(packet.pixel_type).lower()
        if (
            any(token in pixel_type_name for token in ("mono16", "mono12", "mono10"))
            and expected_len > 0
            and len(packet.data) >= expected_len * 2
        ):
            payload = packet.data if len(packet.data) == expected_len * 2 else memoryview(packet.data)[: expected_len * 2]
            raw = np.frombuffer(payload, dtype="<u2", count=expected_len).reshape((packet.height, packet.width))
            if "mono16" in pixel_type_name:
                display = (raw >> 8).astype(np.uint8)
            elif "mono12" in pixel_type_name:
                display = np.clip(raw >> 4, 0, 255).astype(np.uint8)
            else:
                display = np.clip(raw >> 2, 0, 255).astype(np.uint8)
            return Image.fromarray(display, "L")
        buffer_type = c_ubyte * len(packet.data)
        raw_buffer = buffer_type.from_buffer_copy(packet.data)

        class _FrameInfo:
            pass

        frame_info = _FrameInfo()
        frame_info.nWidth = packet.width
        frame_info.nHeight = packet.height
        frame_info.nFrameLen = packet.frame_len
        frame_info.enPixelType = packet.pixel_type
        return self._frame_to_image(raw_buffer, frame_info)

    def save_packet_with_sdk(
        self,
        packet: RawFramePacket,
        image_type: str = "bmp",
        jpeg_quality: int = 95,
    ) -> bytes | None:
        save_param_cls = sdk().get("MV_SAVE_IMAGE_PARAM_EX")
        save_func = getattr(self._cam, "MV_CC_SaveImageEx2", None) or getattr(self._cam, "MV_CC_SaveImageEx", None)
        if save_param_cls is None or save_func is None:
            return None
        s = sdk()
        image_type_value = s.get("MV_Image_Jpeg") if image_type.lower() in {"jpg", "jpeg"} else s.get("MV_Image_Bmp")
        if image_type_value is None:
            return None
        src_buffer_type = c_ubyte * len(packet.data)
        src_buffer = src_buffer_type.from_buffer_copy(packet.data)
        dst_size = max(packet.width * packet.height * 4 + 4096, len(packet.data) * 4 + 4096, 1024 * 1024)
        dst_buffer = (c_ubyte * dst_size)()
        save_param = save_param_cls()
        memset(byref(save_param), 0, sizeof(save_param))
        save_param.enImageType = image_type_value
        save_param.enPixelType = packet.pixel_type
        save_param.nWidth = packet.width
        save_param.nHeight = packet.height
        save_param.nDataLen = packet.frame_len
        save_param.pData = cast(src_buffer, POINTER(c_ubyte))
        save_param.pImageBuffer = cast(dst_buffer, POINTER(c_ubyte))
        save_param.nImageLen = dst_size
        save_param.nBufferSize = dst_size
        if hasattr(save_param, "nJpgQuality"):
            save_param.nJpgQuality = max(min(int(jpeg_quality), 100), 1)
        try:
            ret = save_func(save_param)
        except Exception as exc:
            LOGGER.debug("%s: SDK image save call failed: %s", self.info.label, exc, exc_info=True)
            return None
        if ret != 0:
            LOGGER.debug("%s: SDK image save returned %s.", self.info.label, _format_mvs_error(ret))
            return None
        output_len = int(getattr(save_param, "nImageLen", 0) or getattr(save_param, "nDstLen", 0) or 0)
        if output_len <= 0 or output_len > dst_size:
            return None
        return string_at(dst_buffer, output_len)

    def _frame_to_image(self, raw_buffer: Any, frame_info: Any) -> Image.Image:
        s = sdk()
        width = int(frame_info.nWidth)
        height = int(frame_info.nHeight)
        frame_len = int(frame_info.nFrameLen)
        pixel_type = int(frame_info.enPixelType)

        if pixel_type == s["PixelType_Gvsp_Mono8"] and frame_len >= width * height:
            return Image.frombytes("L", (width, height), string_at(raw_buffer, width * height))

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
            raise MvsError(f"{self.info.label} 像素格式转换失败: {_format_mvs_error(ret)}")

        converted_len = int(convert.nDstLen)
        if converted_len < rgb_size:
            raise MvsError(f"{self.info.label} 像素格式转换输出长度异常: {converted_len} < {rgb_size}")
        return Image.frombytes("RGB", (width, height), string_at(rgb_buffer, rgb_size))

    def _get_int(self, key: str) -> int:
        s = sdk()
        value = s["MVCC_INTVALUE"]()
        memset(byref(value), 0, sizeof(value))
        ret = self._cam.MV_CC_GetIntValue(key, value)
        if ret != 0:
            raise MvsError(f"{self.info.label} 读取 {key} 失败: {_format_mvs_error(ret)}")
        return int(value.nCurValue)

    def _get_int_info(self, key: str) -> IntNodeInfo:
        s = sdk()
        value = s["MVCC_INTVALUE"]()
        memset(byref(value), 0, sizeof(value))
        ret = self._cam.MV_CC_GetIntValue(key, value)
        if ret != 0:
            raise MvsError(f"{self.info.label} 读取 {key} 范围失败: {_format_mvs_error(ret)}")
        increment = int(getattr(value, "nInc", 1) or 1)
        return IntNodeInfo(
            current=int(value.nCurValue),
            minimum=int(value.nMin),
            maximum=int(value.nMax),
            increment=max(increment, 1),
        )

    def _try_get_int(self, key: str) -> int | None:
        try:
            return self._get_int(key)
        except Exception as exc:
            LOGGER.debug("%s: exception while reading int node %s: %s", self.info.label, key, exc, exc_info=True)
            return None

    def _try_get_int_any(self, keys: tuple[str, ...]) -> int | None:
        for key in keys:
            value = self._try_get_int(key)
            if value is not None:
                return value
        return None

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
            raise MvsError(f"{self.info.label} 设置 {key}={value} 失败: {_format_mvs_error(ret)}")

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

    def _try_set_bool(self, key: str, value: bool) -> bool:
        try:
            setter = getattr(self._cam, "MV_CC_SetBoolValue", None)
            if setter is not None:
                if setter(key, bool(value)) == 0:
                    return True
        except Exception as exc:
            LOGGER.debug("%s: exception while probing bool node %s=%s: %s", self.info.label, key, value, exc, exc_info=True)
        if self._try_set_enum_by_string(key, "On" if value else "Off"):
            return True
        return self._try_set_int(key, 1 if value else 0)

    def _try_set_bool_any(self, keys: tuple[str, ...], value: bool) -> bool:
        for key in keys:
            if self._try_set_bool(key, value):
                return True
        return False

    def _set_float(self, key: str, value: float) -> None:
        ret = self._cam.MV_CC_SetFloatValue(key, float(value))
        if ret != 0:
            raise MvsError(f"{self.info.label} 设置 {key}={value} 失败: {_format_mvs_error(ret)}")

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
                    parsed = self._float_from_sdk_value(item, key)
                    if parsed is not None:
                        return parsed
                LOGGER.debug("%s: direct float node %s returned tuple without numeric value: %r", self.info.label, key, value)
            else:
                parsed = self._float_from_sdk_value(value, key)
                if parsed is not None:
                    return parsed
                LOGGER.debug("%s: direct float node %s returned unsupported value: %r", self.info.label, key, value)
        except Exception as exc:
            LOGGER.debug("%s: exception while reading float node %s using direct API: %s", self.info.label, key, exc, exc_info=True)
        try:
            return float(self._get_int(key))
        except Exception as exc:
            LOGGER.debug("%s: exception while reading float node %s as int fallback: %s", self.info.label, key, exc, exc_info=True)
            return None

    def _float_from_sdk_value(self, value: object, key: str) -> float | None:
        if isinstance(value, (int, float)):
            return float(value)
        try:
            raw = getattr(value, "fCurValue")
        except AttributeError:
            return None
        except Exception as exc:
            LOGGER.debug("%s: exception while reading fCurValue from %s: %s", self.info.label, key, exc, exc_info=True)
            return None
        try:
            return float(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            LOGGER.debug("%s: invalid fCurValue for %s: %r (%s)", self.info.label, key, raw, exc, exc_info=True)
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
        self.timestamp_reject_enabled = config_bool(config, "timestamp_reject_enabled", True, False)
        self.max_camera_timestamp_delta = int(config.get("max_camera_timestamp_delta", 0) or 0)
        self.max_host_timestamp_delta = int(config.get("max_host_timestamp_delta", DEFAULT_HOST_TIMESTAMP_DELTA_NS) or 0)
        if self.max_camera_timestamp_delta <= 0 and self.max_host_timestamp_delta <= 0:
            self.timestamp_reject_enabled = False
        self.require_hardware_trigger = config_bool(config, "require_hardware_trigger", False, False)
        self.hardware_sync_enabled = config_bool(config, "hardware_sync_enabled", False, False)
        self.hardware_sync_master = str(config.get("hardware_sync_master", "left") or "left").strip().lower()
        self.hardware_sync_master_line = str(config.get("hardware_sync_master_line", "Line2") or "Line2")
        self.hardware_sync_master_line_source = str(
            config.get("hardware_sync_master_line_source", "ExposureActive") or "ExposureActive"
        )
        self.hardware_sync_slave_line = str(config.get("hardware_sync_slave_line", "Line0") or "Line0")
        self.hardware_sync_slave_activation = str(config.get("hardware_sync_slave_activation", "RisingEdge") or "RisingEdge")
        self.hardware_sync_master_trigger_source = str(
            config.get("hardware_sync_master_trigger_source", "Software") or "Software"
        )
        timeout_s = self.timeout_ms / 1000.0 + 1.0
        self.software_trigger_barrier_timeout_s = max(
            config_float(config, "software_trigger_barrier_timeout_seconds", timeout_s),
            0.1,
        )
        self.continuous_pair_buffer_size = max(config_int(config, "continuous_pair_buffer_size", 256), 1)
        self.continuous_pair_match_timeout_ms = max(
            config_int(config, "continuous_pair_match_timeout_ms", min(self.timeout_ms, 200)),
            1,
        )
        self.camera_timestamp_offset_samples = max(config_int(config, "camera_timestamp_offset_samples", 5), 1)
        self.camera_timestamp_offset_window = max(
            config_int(config, "camera_timestamp_offset_window", self.camera_timestamp_offset_samples * 4),
            self.camera_timestamp_offset_samples,
        )
        self._camera_timestamp_offset: int | None = None
        self._camera_timestamp_offset_samples: deque[int] = deque(maxlen=self.camera_timestamp_offset_window)
        self._capture_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="mvss-capture")

    def connect(self) -> tuple[CameraInfo | None, CameraInfo | None]:
        self._camera_timestamp_offset = None
        self._camera_timestamp_offset_samples = deque(maxlen=self.camera_timestamp_offset_window)
        left_info, right_info, dev_list = select_capture_devices(
            str(self.config.get("left_serial", "")).strip(),
            str(self.config.get("right_serial", "")).strip(),
            config_bool(self.config, "allow_single_camera", False, False),
            config_bool(self.config, "bind_camera_serials", False, False),
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
                cam.configure_streaming(
                    stream_buffer_size=config_int(self.config, "stream_buffer_size", DEFAULT_STREAM_BUFFER_SIZE),
                    raw_buffer_pool_size=config_int(self.config, "raw_buffer_pool_size", DEFAULT_RAW_BUFFER_POOL_SIZE),
                    prefer_callback=self._normalized_system_trigger_source() == "continuous",
                )
                side = "left" if cam is left else "right"
                camera_trigger_source = self._camera_trigger_source_for_connect(side)
                camera_trigger_activation = self._camera_trigger_activation_for_connect(side)
                cam.configure(
                    trigger_source=camera_trigger_source,
                    exposure_time_us=float(self.config.get("exposure_time_us", 0) or 0),
                    gain=float(self.config.get("gain", -1)),
                    pixel_format=str(self.config.get("pixel_format", "Mono8")),
                    gain_auto=str(self.config.get("gain_auto", "Off")),
                    auto_gain_lower_limit=self._optional_float("auto_gain_lower_limit"),
                    auto_gain_upper_limit=self._optional_float("auto_gain_upper_limit"),
                    exposure_auto=str(self.config.get("exposure_auto", "Off")),
                    auto_exposure_lower_limit=self._optional_float("auto_exposure_lower_limit"),
                    auto_exposure_upper_limit=self._optional_float("auto_exposure_upper_limit"),
                    balance_white_auto=str(self.config.get("balance_white_auto", "Off")),
                    balance_ratio_red=self._optional_float("balance_ratio_red"),
                    balance_ratio_green=self._optional_float("balance_ratio_green"),
                    balance_ratio_blue=self._optional_float("balance_ratio_blue"),
                    roi_width=self._roi_value(side, "width", "roi_width"),
                    roi_height=self._roi_value(side, "height", "roi_height"),
                    roi_offset_x=int(self._roi_value(side, "offset_x", "roi_offset_x", 0) or 0),
                    roi_offset_y=int(self._roi_value(side, "offset_y", "roi_offset_y", 0) or 0),
                    chunk_data_enabled=config_bool(self.config, "chunk_data_enabled", False, False),
                    chunk_selectors=self.config.get("chunk_selectors"),
                    acquisition_frame_rate=self._optional_float("acquisition_frame_rate"),
                    trigger_delay_us=self._optional_float("trigger_delay_us"),
                    line_debouncer_time_us=self._optional_float("line_debouncer_time_us"),
                    trigger_activation=camera_trigger_activation,
                    black_level=self._optional_float("black_level"),
                    digital_shift=self._optional_float("digital_shift"),
                    gamma=self._optional_float("gamma"),
                )
                for warning in self._apply_hardware_sync_role(cam, side):
                    LOGGER.warning(warning)
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

    def _roi_value(self, side: str, field: str, default_key: str, default: int | None = None) -> int | None:
        value = self.config.get(f"{side}_roi_{field}", self.config.get(default_key, default))
        if value in (None, ""):
            return default
        return int(value)

    def _hardware_sync_active(self) -> bool:
        return self.hardware_sync_enabled or self._normalized_system_trigger_source() == "cascade"

    def _hardware_sync_master_side(self) -> str:
        return "right" if self.hardware_sync_master == "right" else "left"

    def _hardware_sync_role(self, side: str) -> str:
        return "master" if side == self._hardware_sync_master_side() else "slave"

    def _camera_trigger_source_for_connect(self, side: str) -> str:
        if not self._hardware_sync_active():
            return self.trigger_source
        if self._hardware_sync_role(side) == "master":
            return self.hardware_sync_master_trigger_source
        return self.hardware_sync_slave_line

    def _camera_trigger_activation_for_connect(self, side: str) -> str:
        if self._hardware_sync_active() and self._hardware_sync_role(side) == "slave":
            return self.hardware_sync_slave_activation
        return str(self.config.get("trigger_activation", "RisingEdge"))

    def _apply_hardware_sync_role(self, cam: MvsCamera, side: str) -> list[str]:
        if not self._hardware_sync_active():
            return []
        return cam.apply_hardware_cascade_settings(
            self._hardware_sync_role(side),
            master_line=self.hardware_sync_master_line,
            master_line_source=self.hardware_sync_master_line_source,
            slave_line=self.hardware_sync_slave_line,
            slave_activation=self.hardware_sync_slave_activation,
            master_trigger_source=self.hardware_sync_master_trigger_source,
            master_trigger_activation=str(self.config.get("trigger_activation", "RisingEdge")),
        )

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
        executor = getattr(self, "_executor", None)
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
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

    def link_throughput_mbps(self) -> dict[str, float | None]:
        with self._capture_lock:
            return {name: cam.current_throughput_mbps() for name, cam in self._connected_cameras()}

    def stream_stats(self) -> dict[str, dict[str, int | bool]]:
        with self._capture_lock:
            return {name: asdict(cam.stream_stats()) for name, cam in self._connected_cameras()}

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

    def apply_image_correction_settings(
        self,
        black_level: float | None = None,
        digital_shift: float | None = None,
        gamma: float | None = None,
    ) -> list[str]:
        cameras = self._connected_cameras()
        if not cameras:
            raise MvsError("相机尚未连接。")
        with self._capture_lock:
            warnings: list[str] = []
            for _name, cam in cameras:
                warnings.extend(cam.apply_image_correction_settings(black_level, digital_shift, gamma))
            if black_level is not None:
                self.config["black_level"] = black_level
            if digital_shift is not None:
                self.config["digital_shift"] = digital_shift
            if gamma is not None:
                self.config["gamma"] = gamma
            return warnings

    def apply_pixel_format_settings(self, pixel_format: str) -> list[str]:
        cameras = self._connected_cameras()
        if not cameras:
            raise MvsError("相机尚未连接。")
        with self._capture_lock:
            warnings: list[str] = []
            for _name, cam in cameras:
                warnings.extend(cam.apply_pixel_format_settings(pixel_format))
            self.config["pixel_format"] = pixel_format
            return warnings

    def apply_chunk_settings(self, enabled: bool, selectors: list[str] | tuple[str, ...] | str | None = None) -> list[str]:
        cameras = self._connected_cameras()
        if not cameras:
            raise MvsError("相机尚未连接。")
        with self._capture_lock:
            warnings: list[str] = []
            for _name, cam in cameras:
                warnings.extend(cam.apply_chunk_settings(enabled, selectors))
            self.config["chunk_data_enabled"] = bool(enabled)
            if selectors is not None:
                self.config["chunk_selectors"] = selectors
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
            if restart_stream:
                warnings.extend(self._warm_up_after_roi_locked())
            return RoiApplyResult(warnings, actual_roi)

    def apply_side_roi_settings(
        self,
        rois: dict[str, tuple[int | None, int | None, int, int]],
        restart_stream: bool = True,
    ) -> tuple[dict[str, RoiApplyResult], list[str]]:
        cameras = self._connected_cameras()
        if not cameras:
            raise MvsError("相机尚未连接。")
        with self._capture_lock:
            warnings: list[str] = []
            results: dict[str, RoiApplyResult] = {}
            for name, cam in cameras:
                if name not in rois:
                    continue
                width, height, offset_x, offset_y = rois[name]
                result = cam.apply_roi_settings(width, height, offset_x, offset_y, restart_stream=restart_stream)
                results[name] = result
                warnings.extend(result)
            if restart_stream:
                warnings.extend(self._warm_up_after_roi_locked())
            return results, warnings

    def _warm_up_after_roi_locked(self) -> list[str]:
        warnings: list[str] = []
        settle_s = max(config_float(self.config, "roi_restart_settle_seconds", 0.20), 0.0)
        if settle_s > 0:
            time.sleep(settle_s)

        warmup_frames = max(config_int(self.config, "roi_warmup_frames", 2), 0)
        if warmup_frames <= 0 or self.trigger_source.lower() != "software":
            return warnings

        timeout_default = min(max(int(self.timeout_ms), 1), 800)
        timeout_ms = max(config_int(self.config, "roi_warmup_timeout_ms", timeout_default), 1)
        for index in range(warmup_frames):
            try:
                self._capture_pair_locked(timeout_ms=timeout_ms)
            except FrameTimeoutError as exc:
                message = f"ROI warm-up frame {index + 1} timed out: {exc}"
                LOGGER.warning(message)
                warnings.append(message)
            except Exception as exc:
                message = f"ROI warm-up frame {index + 1} failed: {exc}"
                LOGGER.warning(message)
                warnings.append(message)
                break
        return warnings

    def apply_trigger_settings(self, trigger_source: str) -> list[str]:
        cameras = self._connected_cameras()
        if not cameras:
            raise MvsError("相机尚未连接。")
        with self._capture_lock:
            warnings: list[str] = []
            self.trigger_source = trigger_source
            if self._normalized_system_trigger_source() == "cascade":
                self.hardware_sync_enabled = True
                self.config["hardware_sync_enabled"] = True
                for name, cam in cameras:
                    warnings.extend(self._apply_hardware_sync_role(cam, name))
            else:
                self.hardware_sync_enabled = False
                self.config["hardware_sync_enabled"] = False
                for _name, cam in cameras:
                    warnings.extend(cam.apply_trigger_settings(trigger_source))
            self.config["trigger_source"] = trigger_source
            self.trigger_source = trigger_source
            return warnings

    def capture_pair(
        self,
        timeout_ms: int | None = None,
        convert_image: bool = True,
    ) -> tuple[Frame | None, Frame | None, float]:
        with self._capture_lock:
            return self._capture_pair_locked(timeout_ms=timeout_ms, convert_image=convert_image)

    def _capture_pair_locked(
        self,
        timeout_ms: int | None = None,
        convert_image: bool = True,
    ) -> tuple[Frame | None, Frame | None, float]:
        cameras = self._connected_cameras()
        if not cameras:
            raise MvsError("相机尚未连接。")
        effective_timeout_ms = max(int(timeout_ms if timeout_ms is not None else self.timeout_ms), 1)
        if self._normalized_system_trigger_source() == "continuous":
            return self._capture_pair_continuous(cameras, effective_timeout_ms, convert_image=convert_image)
        return self._capture_pair_with_executor(cameras, effective_timeout_ms, convert_image=convert_image)

    def _capture_pair_with_executor(
        self,
        cameras: list[tuple[str, MvsCamera]],
        effective_timeout_ms: int,
        convert_image: bool = True,
    ) -> tuple[Frame | None, Frame | None, float]:
        trigger_mode = self._normalized_system_trigger_source()
        if self.require_hardware_trigger and trigger_mode in {"software", "continuous"}:
            raise MvsError("Hardware trigger is required, but trigger_source is Software/Continuous.")

        trigger_time = time.time()
        if trigger_mode == "cascade":
            frames = self._capture_pair_hardware_cascade(cameras, effective_timeout_ms, convert_image=convert_image)
            left = frames.get("left")
            right = frames.get("right")
            if left is not None and right is not None:
                self._validate_frame_sync(left, right)
            return left, right, trigger_time
        if trigger_mode == "software":
            if len(cameras) == 1:
                cameras[0][1].trigger_software()
            else:
                self._run_parallel(
                    [(f"trigger-{name}", cam.trigger_software) for name, cam in cameras],
                    self.software_trigger_barrier_timeout_s,
                )

        frames = dict(
            self._run_parallel(
                [
                    (name, lambda cam=cam: self._grab_camera_frame(cam, effective_timeout_ms, convert_image=convert_image))
                    for name, cam in cameras
                ],
                effective_timeout_ms / 1000.0 + 0.2,
            )
        )
        left = frames.get("left")
        right = frames.get("right")
        if left is not None and right is not None:
            self._validate_frame_sync(left, right)
        return left, right, trigger_time

    def _capture_pair_hardware_cascade(
        self,
        cameras: list[tuple[str, MvsCamera]],
        effective_timeout_ms: int,
        convert_image: bool = True,
    ) -> dict[str, Frame]:
        if len(cameras) < 2:
            raise MvsError("Hardware cascade trigger requires both stereo cameras.")
        master_side = self._hardware_sync_master_side()
        master = next((cam for name, cam in cameras if name == master_side), None)
        if master is None:
            raise MvsError(f"Hardware cascade master camera '{master_side}' is not connected.")
        executor = self._executor_snapshot()
        futures = {
            executor.submit(self._grab_camera_frame, cam, effective_timeout_ms, convert_image): name for name, cam in cameras
        }
        # Let both GetImageBuffer calls enter the SDK before the master emits its hardware output pulse.
        time.sleep(0.002)
        master.trigger_software()
        done, pending = wait(futures, timeout=effective_timeout_ms / 1000.0 + 0.2)
        errors: list[Exception] = []
        frames: dict[str, Frame] = {}
        for future in pending:
            errors.append(FrameTimeoutError(f"{futures[future]} timed out"))
            future.cancel()
        for future in done:
            name = futures[future]
            try:
                frames[name] = future.result()
            except Exception as exc:
                errors.append(exc)
        if errors:
            message = "; ".join(str(exc) for exc in errors)
            if all(isinstance(exc, FrameTimeoutError) for exc in errors):
                raise FrameTimeoutError(message)
            raise MvsError(message)
        return frames

    def _capture_pair_continuous(
        self,
        cameras: list[tuple[str, MvsCamera]],
        effective_timeout_ms: int,
        convert_image: bool = True,
    ) -> tuple[Frame | None, Frame | None, float]:
        if self.require_hardware_trigger:
            raise MvsError("Hardware trigger is required, but trigger_source is Continuous.")
        if any(not hasattr(cam, "_frame_from_packet") for _name, cam in cameras):
            return self._capture_pair_with_executor(cameras, effective_timeout_ms, convert_image=convert_image)
        trigger_time = time.time()
        if len(cameras) == 1:
            name, cam = cameras[0]
            frame = self._grab_camera_frame(cam, effective_timeout_ms, convert_image=convert_image)
            return (frame, None, trigger_time) if name == "left" else (None, frame, trigger_time)

        packet_batches = dict(
            self._run_parallel(
                [(name, lambda cam=cam: self._continuous_packets_from_camera(cam, effective_timeout_ms)) for name, cam in cameras],
                effective_timeout_ms / 1000.0 + 0.2,
            )
        )
        left_packets = packet_batches.get("left") or []
        right_packets = packet_batches.get("right") or []
        if not left_packets or not right_packets:
            self._release_packets(left_packets)
            self._release_packets(right_packets)
            raise FrameTimeoutError(f"continuous stream did not provide both stereo frames within {effective_timeout_ms} ms")

        left_packet, right_packet = self._select_best_continuous_packet_pair(left_packets, right_packets)
        for packet in left_packets:
            if packet is not left_packet:
                packet.release_raw_data()
        for packet in right_packets:
            if packet is not right_packet:
                packet.release_raw_data()
        left_cam = next(cam for name, cam in cameras if name == "left")
        right_cam = next(cam for name, cam in cameras if name == "right")
        left = left_cam._frame_from_packet(left_packet, convert_image=convert_image)
        right = right_cam._frame_from_packet(right_packet, convert_image=convert_image)
        self._validate_frame_sync(left, right)
        return left, right, trigger_time

    def _continuous_packets_from_camera(self, cam: MvsCamera, timeout_ms: int) -> list[RawFramePacket]:
        if getattr(cam, "_stream_callback_enabled", False):
            packets = cam.pop_stream_packets(min(timeout_ms, self.continuous_pair_match_timeout_ms))
            if packets:
                return packets[-self.continuous_pair_buffer_size :]
        frame = self._grab_camera_frame(cam, timeout_ms, convert_image=False)
        release = frame._raw_release
        frame._raw_release = None
        return [
            RawFramePacket(
                data=frame.raw_data or b"",
                frame_len=frame.raw_frame_len,
                width=frame.width,
                height=frame.height,
                pixel_type=frame.pixel_type,
                frame_number=frame.frame_number,
                host_timestamp=frame.host_timestamp,
                camera_timestamp=frame.camera_timestamp,
                _raw_release=release,
            )
        ]

    def _select_best_continuous_packet_pair(
        self,
        left_packets: list[RawFramePacket],
        right_packets: list[RawFramePacket],
    ) -> tuple[RawFramePacket, RawFramePacket]:
        best_left = left_packets[-1]
        best_right = right_packets[-1]
        best_delta: int | None = None
        use_camera_timestamp = self.max_camera_timestamp_delta > 0 and self._camera_timestamp_offset is not None
        for left in left_packets:
            for right in right_packets:
                if use_camera_timestamp:
                    delta = abs((int(left.camera_timestamp) - int(right.camera_timestamp)) - int(self._camera_timestamp_offset or 0))
                else:
                    delta = abs(int(left.host_timestamp) - int(right.host_timestamp))
                if best_delta is None or delta < best_delta:
                    best_left = left
                    best_right = right
                    best_delta = delta
        return best_left, best_right

    def _release_packets(self, packets: list[RawFramePacket]) -> None:
        for packet in packets:
            packet.release_raw_data()

    def _grab_camera_frame(self, cam: MvsCamera, timeout_ms: int, convert_image: bool = True) -> Frame:
        try:
            return cam.grab_frame(timeout_ms, convert_image=convert_image)
        except TypeError:
            return cam.grab_frame(timeout_ms)

    def _normalized_system_trigger_source(self) -> str:
        value = str(self.trigger_source).strip().lower()
        if value in {"continuous", "freerun", "free-run", "free run", "off", "none", "trigger off", "no trigger"}:
            return "continuous"
        if value in {"cascade", "hardwarecascade", "hardware-cascade", "hardwaresync", "hardware-sync", "级联", "硬触发级联"}:
            return "cascade"
        if value in {"line0", "line 0", "hardware"}:
            return "line0"
        return "software"

    def _executor_snapshot(self) -> ThreadPoolExecutor:
        executor = getattr(self, "_executor", None)
        if executor is None:
            executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="mvss-capture")
            self._executor = executor
        return executor

    def _run_parallel(self, tasks: list[tuple[str, Any]], timeout_s: float) -> list[tuple[str, Any]]:
        if len(tasks) == 1:
            name, func = tasks[0]
            return [(name, func())]
        executor = self._executor_snapshot()
        futures = {executor.submit(func): name for name, func in tasks}
        done, pending = wait(futures, timeout=max(timeout_s, 0.1))
        errors: list[Exception] = []
        results: list[tuple[str, Any]] = []
        for future in pending:
            errors.append(FrameTimeoutError(f"{futures[future]} timed out"))
            future.cancel()
        for future in done:
            name = futures[future]
            try:
                results.append((name, future.result()))
            except Exception as exc:
                errors.append(exc)
        if errors:
            message = "; ".join(str(exc) for exc in errors)
            if all(isinstance(exc, FrameTimeoutError) for exc in errors):
                raise FrameTimeoutError(message)
            raise MvsError(message)
        return results

    def _median_offset(self, values: deque[int] | list[int]) -> int:
        sorted_offsets = sorted(values)
        midpoint = len(sorted_offsets) // 2
        if len(sorted_offsets) % 2:
            return sorted_offsets[midpoint]
        return int(round((sorted_offsets[midpoint - 1] + sorted_offsets[midpoint]) / 2))

    def _observe_camera_timestamp_offset(self, raw_camera_delta: int) -> None:
        self._camera_timestamp_offset_samples.append(int(raw_camera_delta))
        if len(self._camera_timestamp_offset_samples) >= self.camera_timestamp_offset_samples:
            self._camera_timestamp_offset = self._median_offset(self._camera_timestamp_offset_samples)

    def camera_timestamp_offset(self) -> int | None:
        return self._camera_timestamp_offset

    def set_camera_timestamp_offset(self, offset: int | None, *, seed_samples: bool = True) -> None:
        self._camera_timestamp_offset = None if offset is None else int(offset)
        self._camera_timestamp_offset_samples.clear()
        if seed_samples and self._camera_timestamp_offset is not None:
            for _ in range(self.camera_timestamp_offset_samples):
                self._camera_timestamp_offset_samples.append(self._camera_timestamp_offset)

    def calibrate_camera_timestamp_offset(self, sample_count: int | None = None, timeout_ms: int | None = None) -> int:
        count = max(int(sample_count or self.camera_timestamp_offset_samples), 1)
        previous_enabled = self.timestamp_reject_enabled
        self.timestamp_reject_enabled = False
        offsets: list[int] = []
        try:
            for _ in range(count):
                left, right, _trigger_time = self.capture_pair(timeout_ms=timeout_ms)
                if left is None or right is None:
                    continue
                offsets.append(int(left.camera_timestamp) - int(right.camera_timestamp))
        finally:
            self.timestamp_reject_enabled = previous_enabled
        if not offsets:
            raise FrameTimeoutError("unable to calibrate camera timestamp offset: no stereo frame pairs captured")
        offset = self._median_offset(offsets)
        self.set_camera_timestamp_offset(offset)
        self.config["camera_timestamp_offset_fixed"] = offset
        return offset

    def _validate_frame_sync(self, left: Frame, right: Frame) -> None:
        if not self.timestamp_reject_enabled:
            return
        issues: list[str] = []
        raw_camera_delta: int | None = None
        offset_sample_used_for_warmup = False
        if self.max_camera_timestamp_delta > 0:
            raw_camera_delta = int(left.camera_timestamp) - int(right.camera_timestamp)
            if self._camera_timestamp_offset is None:
                self._observe_camera_timestamp_offset(raw_camera_delta)
                offset_sample_used_for_warmup = True
            if self._camera_timestamp_offset is not None:
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
        # Keep the sliding offset adaptive, but only learn from pairs accepted by the current baseline.
        if raw_camera_delta is not None and self._camera_timestamp_offset is not None and not offset_sample_used_for_warmup:
            self._observe_camera_timestamp_offset(raw_camera_delta)

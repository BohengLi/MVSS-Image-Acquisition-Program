from __future__ import annotations

import ctypes
import csv
import tempfile
import threading
import unittest
from pathlib import Path
from queue import Queue
from unittest.mock import patch

import importlib.util
import image_quality
import calibration_manager
import mvs_camera
import numpy as np
from PIL import Image
from mvs_camera import Frame, MvsCamera, MvsError, RawFramePacket, StereoCameraSystem
from project_manager import ProjectManager
import stereo_capture_only
from stereo_capture_only import StereoCaptureOnlyApp


class _Info:
    label = "test-camera"


class _BadFloatValue:
    @property
    def fCurValue(self):
        raise RuntimeError("SDK object is invalid")


class _PartialImage:
    mode = "L"

    def save(self, path: Path, *args, **kwargs) -> None:
        path.write_bytes(b"partial")
        raise OSError("simulated write failure")


class _FakeGrabCamera:
    def __init__(self, label: str):
        self.info = type("Info", (), {"label": label})()
        self.grab_timeouts: list[int] = []
        self.convert_image_values: list[object] = []
        self.trigger_count = 0

    def trigger_software(self) -> None:
        self.trigger_count += 1

    def grab_frame(self, timeout_ms: int, convert_image: bool = True) -> Frame:
        self.grab_timeouts.append(timeout_ms)
        self.convert_image_values.append(convert_image)
        frame_number = len(self.grab_timeouts)
        return Frame(
            image=object() if convert_image else None,
            frame_number=frame_number,
            width=1,
            height=1,
            host_timestamp=frame_number,
            camera_timestamp=frame_number,
        )


class _FakeRoiCamera:
    is_ready = True

    def __init__(self, actuals: list[tuple[int, int, int, int]]):
        self.actuals = list(actuals)
        self.calls: list[tuple[int | None, int | None, int, int, bool]] = []

    def apply_roi_settings(
        self,
        width: int | None,
        height: int | None,
        offset_x: int,
        offset_y: int,
        restart_stream: bool = True,
    ):
        self.calls.append((width, height, offset_x, offset_y, restart_stream))
        actual = self.actuals.pop(0)
        return mvs_camera.RoiApplyResult([], actual)


class _FakeNodeCamera:
    def __init__(self):
        self.enum_strings: list[tuple[str, str]] = []


class _FakeLineNodeCamera:
    def __init__(self):
        self.enum_strings: list[tuple[str, str]] = []

    def MV_CC_SetEnumValueByString(self, key: str, value: str) -> int:
        self.enum_strings.append((key, value))
        return 0


class _FakeChunkSdkCamera:
    def __init__(self):
        self.bool_values: list[tuple[str, bool]] = []
        self.enum_strings: list[tuple[str, str]] = []

    def MV_CC_SetBoolValue(self, key: str, value: bool) -> int:
        self.bool_values.append((key, bool(value)))
        return 0

    def MV_CC_SetEnumValueByString(self, key: str, value: str) -> int:
        self.enum_strings.append((key, value))
        return 0


class _FakeEnumBoolCamera:
    def __init__(self):
        self.enum_strings: list[tuple[str, str]] = []
        self.int_values: list[tuple[str, int]] = []

    def MV_CC_SetEnumValueByString(self, key: str, value: str) -> int:
        self.enum_strings.append((key, value))
        return 0

    def MV_CC_SetIntValue(self, key: str, value: int) -> int:
        self.int_values.append((key, value))
        return 0


class _FakeFloatCamera:
    def __init__(self):
        self.float_values: list[tuple[str, float]] = []
        self.bool_values: list[tuple[str, bool]] = []

    def MV_CC_SetFloatValue(self, key: str, value: float) -> int:
        self.float_values.append((key, float(value)))
        return 0

    def MV_CC_SetBoolValue(self, key: str, value: bool) -> int:
        self.bool_values.append((key, bool(value)))
        return 0


class _Var:
    def __init__(self):
        self.value = ""

    def set(self, value) -> None:
        self.value = str(value)

    def get(self) -> str:
        return self.value


class _FakeStatsSystem:
    def __init__(self):
        self.temperature_reads = 0
        self.stream_reads = 0

    def sensor_temperatures(self):
        self.temperature_reads += 1
        return {"left": 40.0}

    def link_throughput_mbps(self):
        return {"left": 100.0}

    def stream_stats(self):
        self.stream_reads += 1
        return {"left": {"buffered_frames": 0, "dropped_frames": 2, "callback_enabled": True}}


class _FakeCameraInfo:
    def __init__(self, index: int, serial: str, label: str, transport: str = "USB3"):
        self.index = index
        self.serial = serial
        self.label = label
        self.transport = transport


class _FakeCalibration:
    def meta(self):
        return {}


class ReliabilityFixTests(unittest.TestCase):
    def test_pyinstaller_spec_skips_hikrobot_hidden_import_when_package_missing(self) -> None:
        spec_path = Path(__file__).resolve().parents[1] / "MVSS_Capture.spec"
        calls: dict[str, object] = {}

        def fake_analysis(_scripts, **kwargs):
            calls["hiddenimports"] = kwargs.get("hiddenimports")
            calls["datas"] = kwargs.get("datas")
            return type("AnalysisResult", (), {"pure": [], "scripts": [], "binaries": [], "datas": []})()

        namespace = {
            "Analysis": fake_analysis,
            "PYZ": lambda _pure: object(),
            "EXE": lambda *_args, **_kwargs: object(),
            "__file__": str(spec_path),
        }
        original_find_spec = importlib.util.find_spec
        original_path_exists = Path.exists
        runtime_dir = str(Path(r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64"))

        def fake_path_exists(path: Path) -> bool:
            if str(path) == runtime_dir:
                return False
            return original_path_exists(path)

        importlib.util.find_spec = lambda name: None if name == "hikrobot" else original_find_spec(name)
        try:
            with patch.object(Path, "exists", fake_path_exists):
                exec(compile(spec_path.read_text(encoding="utf-8"), str(spec_path), "exec"), namespace)
        finally:
            importlib.util.find_spec = original_find_spec

        self.assertEqual(calls["hiddenimports"], [])
        self.assertEqual(calls["datas"], [("config.json", ".")])

    def test_float_from_sdk_value_handles_bad_sdk_object(self) -> None:
        camera = MvsCamera.__new__(MvsCamera)
        camera.info = _Info()

        self.assertIsNone(camera._float_from_sdk_value(object(), "Gain"))
        self.assertIsNone(camera._float_from_sdk_value(_BadFloatValue(), "Gain"))

    def test_camera_continuous_trigger_mode_turns_trigger_off(self) -> None:
        camera = MvsCamera.__new__(MvsCamera)
        camera.info = _Info()
        fake = _FakeNodeCamera()
        camera._try_set_enum_by_string = lambda key, value: fake.enum_strings.append((key, value)) or True

        warnings = camera.apply_trigger_settings("Continuous")

        self.assertEqual(warnings, [])
        self.assertIn(("TriggerMode", "Off"), fake.enum_strings)
        self.assertNotIn(("TriggerSource", "Software"), fake.enum_strings)

    def test_hardware_cascade_master_configures_output_line(self) -> None:
        camera = MvsCamera.__new__(MvsCamera)
        camera.info = _Info()
        camera._cam = _FakeLineNodeCamera()
        camera._try_set_enum = lambda *_args, **_kwargs: False

        warnings = camera.apply_hardware_cascade_settings(
            "master",
            master_line="Line2",
            master_line_source="ExposureActive",
            master_trigger_source="Software",
        )

        self.assertEqual(warnings, [])
        self.assertIn(("TriggerMode", "On"), camera._cam.enum_strings)
        self.assertIn(("TriggerSource", "Software"), camera._cam.enum_strings)
        self.assertIn(("LineSelector", "Line2"), camera._cam.enum_strings)
        self.assertIn(("LineMode", "Output"), camera._cam.enum_strings)
        self.assertIn(("LineSource", "ExposureActive"), camera._cam.enum_strings)

    def test_hardware_cascade_slave_configures_line_trigger_input(self) -> None:
        camera = MvsCamera.__new__(MvsCamera)
        camera.info = _Info()
        camera._cam = _FakeLineNodeCamera()
        camera._try_set_enum = lambda *_args, **_kwargs: False

        warnings = camera.apply_hardware_cascade_settings(
            "slave",
            slave_line="Line0",
            slave_activation="RisingEdge",
        )

        self.assertEqual(warnings, [])
        self.assertIn(("LineSelector", "Line0"), camera._cam.enum_strings)
        self.assertIn(("LineMode", "Input"), camera._cam.enum_strings)
        self.assertIn(("TriggerMode", "On"), camera._cam.enum_strings)
        self.assertIn(("TriggerSource", "Line0"), camera._cam.enum_strings)
        self.assertIn(("TriggerActivation", "RisingEdge"), camera._cam.enum_strings)

    def test_mono8_packet_to_image_uses_exact_payload_without_slice(self) -> None:
        camera = MvsCamera.__new__(MvsCamera)
        payload = b"123456"
        packet = RawFramePacket(payload, len(payload), 3, 2, 1, 1, 0, 0)
        seen: dict[str, object] = {}

        def fake_frombytes(mode, size, data):
            seen["mode"] = mode
            seen["size"] = size
            seen["data"] = data
            return object()

        with patch.object(mvs_camera, "sdk", return_value={"PixelType_Gvsp_Mono8": 1}), patch.object(
            mvs_camera.Image,
            "frombytes",
            side_effect=fake_frombytes,
        ):
            camera._packet_to_image(packet)

        self.assertEqual(seen["mode"], "L")
        self.assertEqual(seen["size"], (3, 2))
        self.assertIs(seen["data"], payload)

    def test_packet_to_image_accepts_non_contiguous_memoryview_payload(self) -> None:
        camera = MvsCamera.__new__(MvsCamera)
        source = np.arange(8, dtype=np.uint8).reshape(2, 4)[:, :3]
        packet = RawFramePacket(memoryview(source), source.nbytes, 3, 2, 1, 1, 0, 0)

        with patch.object(mvs_camera, "sdk", return_value={"PixelType_Gvsp_Mono8": 1}):
            image = camera._packet_to_image(packet)

        self.assertEqual(image.size, (3, 2))
        self.assertEqual(list(image.getdata()), [0, 1, 2, 4, 5, 6])

    def test_frame_from_packet_keeps_raw_payload_metadata(self) -> None:
        camera = MvsCamera.__new__(MvsCamera)
        camera._packet_to_image = lambda _packet: object()
        packet = RawFramePacket(b"123456", 6, 3, 2, 1, 9, 10, 11)

        with patch.object(
            mvs_camera,
            "sdk",
            return_value={"PIXEL_TYPE_NAMES": {1: "PixelType_Gvsp_Mono8"}},
        ):
            frame = camera._frame_from_packet(packet)

        self.assertEqual(frame.raw_data, b"123456")
        self.assertEqual(frame.raw_frame_len, 6)
        self.assertEqual(frame.pixel_type_name, "PixelType_Gvsp_Mono8")
        self.assertEqual(frame.raw_bit_depth, 8)
        self.assertEqual(frame.raw_array_shape, (2, 3))

    def test_frame_from_packet_can_skip_pil_conversion_for_raw_recording(self) -> None:
        camera = MvsCamera.__new__(MvsCamera)
        camera._packet_to_image = lambda _packet: self.fail("raw-only frame should not build PIL image")
        packet = RawFramePacket(b"123456", 6, 3, 2, 1, 9, 10, 11)

        with patch.object(
            mvs_camera,
            "sdk",
            return_value={"PIXEL_TYPE_NAMES": {1: "PixelType_Gvsp_Mono8"}},
        ):
            frame = camera._frame_from_packet(packet, convert_image=False)

        self.assertIsNone(frame.image)
        self.assertEqual(frame.raw_data, b"123456")

    def test_mono16_packet_to_image_uses_numpy_fast_path(self) -> None:
        camera = MvsCamera.__new__(MvsCamera)
        raw = np.array([[0, 65535], [32768, 16384]], dtype=np.uint16).tobytes()
        packet = RawFramePacket(raw, len(raw), 2, 2, 2, 1, 0, 0)

        with patch.object(
            mvs_camera,
            "sdk",
            return_value={"PixelType_Gvsp_Mono8": 1, "PIXEL_TYPE_NAMES": {2: "PixelType_Gvsp_Mono16"}},
        ):
            image = camera._packet_to_image(packet)

        self.assertEqual(image.mode, "L")
        self.assertEqual(image.size, (2, 2))
        self.assertEqual(np.asarray(image, dtype=np.uint8).tolist(), [[0, 255], [128, 64]])

    def test_frame_raw_release_clears_payload_after_metadata_is_available(self) -> None:
        released: list[bool] = []
        frame = Frame(
            image=None,
            frame_number=1,
            width=2,
            height=3,
            host_timestamp=4,
            camera_timestamp=5,
            raw_data=bytearray(b"123456"),
            raw_frame_len=6,
            _raw_release=lambda: released.append(True),
        )
        app = StereoCaptureOnlyApp.__new__(StereoCaptureOnlyApp)

        meta = app._frame_meta(frame)
        frame.release_raw_data()

        self.assertEqual(meta["raw_frame_len"], 6)
        self.assertIsNone(frame.raw_data)
        self.assertEqual(released, [True])

    def test_chunk_settings_enable_timestamp_metadata_nodes(self) -> None:
        camera = MvsCamera.__new__(MvsCamera)
        camera.info = _Info()
        camera._cam = _FakeChunkSdkCamera()

        warnings = camera.apply_chunk_settings(True, ["Timestamp", "ExposureTime"])

        self.assertEqual(warnings, [])
        self.assertIn(("ChunkModeActive", True), camera._cam.bool_values)
        self.assertIn(("ChunkSelector", "Timestamp"), camera._cam.enum_strings)
        self.assertIn(("ChunkSelector", "ExposureTime"), camera._cam.enum_strings)
        self.assertGreaterEqual(camera._cam.bool_values.count(("ChunkEnable", True)), 2)

    def test_bool_node_can_fall_back_to_on_off_enum_without_integer_write(self) -> None:
        camera = MvsCamera.__new__(MvsCamera)
        camera.info = _Info()
        camera._cam = _FakeEnumBoolCamera()

        self.assertTrue(camera._try_set_bool("ChunkEnable", True))

        self.assertEqual(camera._cam.enum_strings, [("ChunkEnable", "On")])
        self.assertEqual(camera._cam.int_values, [])

    def test_configure_restarts_stream_for_pixel_format_changes(self) -> None:
        camera = MvsCamera.__new__(MvsCamera)
        camera._try_set_enum_by_string = lambda *_args, **_kwargs: True
        camera.apply_trigger_settings = lambda *_args, **_kwargs: []
        camera.apply_timing_settings = lambda *_args, **_kwargs: []
        camera.apply_roi_settings = lambda *_args, **_kwargs: None
        camera.apply_exposure_settings = lambda *_args, **_kwargs: []
        camera.apply_gain_settings = lambda *_args, **_kwargs: []
        camera.apply_white_balance_settings = lambda *_args, **_kwargs: []
        camera.apply_image_correction_settings = lambda *_args, **_kwargs: []
        camera.apply_chunk_settings = lambda *_args, **_kwargs: []
        camera._set_payload_size = lambda *_args, **_kwargs: None
        camera._get_int = lambda *_args, **_kwargs: 1
        seen: list[bool] = []
        camera.apply_pixel_format_settings = lambda _pixel_format, restart_stream=True: seen.append(restart_stream) or []

        camera.configure("Software", pixel_format="Mono16")

        self.assertEqual(seen, [True])

    def test_grab_frame_with_timeout_uses_fallback_payload_when_payload_size_is_zero(self) -> None:
        camera = MvsCamera.__new__(MvsCamera)
        camera.info = _Info()
        camera._payload_lock = threading.Lock()
        camera._payload_size = 0
        camera._payload_size_snapshot = lambda: 0
        camera._try_get_int = lambda key: {"Width": 3, "Height": 2}.get(key, 0)
        camera._try_get_string = lambda key: "Mono16" if key == "PixelFormat" else None
        seen: dict[str, int] = {}

        class _FrameInfo(ctypes.Structure):
            _fields_ = [
                ("nWidth", ctypes.c_uint),
                ("nHeight", ctypes.c_uint),
                ("nFrameLen", ctypes.c_uint),
                ("enPixelType", ctypes.c_uint),
                ("nFrameNum", ctypes.c_uint),
                ("nDevTimeStampHigh", ctypes.c_uint),
                ("nDevTimeStampLow", ctypes.c_uint),
            ]

        class _Camera:
            def MV_CC_GetOneFrameTimeout(self, _buffer, payload_size, frame_info, _timeout_ms):
                seen["payload_size"] = payload_size
                frame_info.nWidth = 3
                frame_info.nHeight = 2
                frame_info.nFrameLen = 12
                frame_info.enPixelType = 1
                frame_info.nFrameNum = 1
                frame_info.nDevTimeStampHigh = 0
                frame_info.nDevTimeStampLow = 0
                return 0

        camera._cam = _Camera()
        camera._raw_packet_from_pointer = lambda _buffer, frame_info: RawFramePacket(
            b"\x00" * frame_info.nFrameLen,
            frame_info.nFrameLen,
            frame_info.nWidth,
            frame_info.nHeight,
            frame_info.enPixelType,
            frame_info.nFrameNum,
            0,
            0,
        )
        camera._frame_from_packet = lambda packet, convert_image=True: packet

        with patch.object(mvs_camera, "sdk", return_value={"MV_FRAME_OUT_INFO_EX": _FrameInfo}):
            packet = camera._grab_frame_with_timeout(100)

        self.assertEqual(seen["payload_size"], 12)
        self.assertEqual(packet.frame_len, 12)

    def test_image_correction_settings_apply_optional_float_nodes(self) -> None:
        camera = MvsCamera.__new__(MvsCamera)
        camera.info = _Info()
        camera._cam = _FakeFloatCamera()
        camera._float_node_cache = {}
        camera._float_node_cache_lock = threading.Lock()

        warnings = camera.apply_image_correction_settings(0.0, 2.0, 1.2)

        self.assertEqual(warnings, [])
        self.assertEqual(
            camera._cam.float_values,
            [("BlackLevel", 0.0), ("DigitalShift", 2.0), ("Gamma", 1.2)],
        )
        self.assertEqual(camera._cam.bool_values, [("GammaEnable", True)])

    def test_image_correction_settings_skip_empty_values(self) -> None:
        camera = MvsCamera.__new__(MvsCamera)
        camera.info = _Info()
        camera._cam = _FakeFloatCamera()
        camera._float_node_cache = {}
        camera._float_node_cache_lock = threading.Lock()

        warnings = camera.apply_image_correction_settings(None, None, None)

        self.assertEqual(warnings, [])
        self.assertEqual(camera._cam.float_values, [])

    def test_save_image_removes_partial_file_on_failure(self) -> None:
        app = StereoCaptureOnlyApp.__new__(StereoCaptureOnlyApp)
        app._config_snapshot = lambda: {"record_jpeg_quality": 95, "image_format": "png"}

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frame.png"
            previous_disabled = stereo_capture_only.LOGGER.disabled
            stereo_capture_only.LOGGER.disabled = True
            try:
                with self.assertRaises(Exception):
                    app._save_image(_PartialImage(), path)
            finally:
                stereo_capture_only.LOGGER.disabled = previous_disabled
            self.assertFalse(path.exists())

    def test_project_creation_prepares_flat_left_right_image_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = {"project": {"enabled": True, "projects_subdir": "projects"}}
            manager = ProjectManager(Path(tmp), config)

            project_dir = manager.create_project()

            self.assertTrue((project_dir / "left").is_dir())
            self.assertTrue((project_dir / "right").is_dir())
            self.assertTrue((project_dir / "videos").is_dir())
            self.assertFalse((project_dir / "photos").exists())

    def test_photo_pair_saves_images_to_project_left_right_without_capture_subdirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = {"image_format": "png", "project": {"enabled": True, "projects_subdir": "projects"}}
            app = StereoCaptureOnlyApp.__new__(StereoCaptureOnlyApp)
            app.config = config
            app.project_manager = ProjectManager(Path(tmp), config)
            app.project_manager.create_project()
            app.calibration = _FakeCalibration()
            app.camera_system = None
            app._latest_temperatures = {}
            app._latest_stream_stats = {}
            app._temperature_samples = []
            app._device_versions = {}
            app._quality_metrics_for_pair = lambda _left, _right: {
                "focus": {},
                "left_exposure": None,
                "right_exposure": None,
            }
            app._quality_report_from_metrics = lambda _metrics=None: {"ok": True, "results": []}
            app._capture_settings_snapshot = lambda _snapshot=None: {}
            app._checksum_algorithm = lambda _snapshot=None: "sha256"
            saved_paths: list[Path] = []

            def fake_save_frame(_frame, path, _snapshot=None):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"image")
                saved_paths.append(path)
                return path

            app._save_frame = fake_save_frame
            image = Image.fromarray(np.zeros((2, 2), dtype=np.uint8), "L")
            left = Frame(image, 1, 2, 2, 0, 0)
            right = Frame(image, 2, 2, 2, 0, 0)

            meta_dir = app._save_photo_pair(left, right, 1.0, mode="photo")
            project_dir = app.project_manager.active_project_dir

            self.assertEqual([path.parent.name for path in saved_paths], ["left", "right"])
            self.assertTrue(all(path.parent.parent == project_dir for path in saved_paths))
            self.assertTrue((meta_dir / "meta.json").exists())
            manifest_csv = meta_dir / "exports" / "file_manifest.csv"
            self.assertTrue(manifest_csv.exists())
            with manifest_csv.open("r", newline="", encoding="utf-8-sig") as fh:
                manifest_paths = {row["path"] for row in csv.DictReader(fh)}
            self.assertTrue(all(str(path) in manifest_paths for path in saved_paths))
            self.assertFalse((project_dir / "photos").exists())

    def test_guide_mode_key_supports_grid_and_cross_combinations(self) -> None:
        app = StereoCaptureOnlyApp.__new__(StereoCaptureOnlyApp)
        app.guide_mode_var = _Var()

        for text, expected in (
            ("关闭", "off"),
            ("仅十字线", "center"),
            ("中心十字", "center"),
            ("仅网格线", "grid"),
            ("十字+网格", "full"),
            ("全部网格线", "full"),
        ):
            app.guide_mode_var.set(text)
            self.assertEqual(app._guide_mode_key(), expected)

    def test_record_queue_full_drops_frame_without_blocking(self) -> None:
        app = StereoCaptureOnlyApp.__new__(StereoCaptureOnlyApp)
        app._state_lock = threading.RLock()
        app.recording = True
        app._clone_frame = lambda frame: frame
        app._raise_record_write_lag = lambda *_args: "queue full"
        app._notify_warning = lambda *_args, **_kwargs: None
        app.ui_queue = Queue()
        skipped: list[tuple[str, int]] = []
        app._record_skipped = lambda reason, index: skipped.append((reason, index))

        queue: Queue = Queue(maxsize=1)
        queue.put_nowait({"index": 1})

        ok = app._put_record_item(queue, {"index": 42, "left": None, "right": None})

        self.assertFalse(ok)
        self.assertEqual(skipped, [("record_queue_full", 42)])
        self.assertEqual(queue.qsize(), 1)

    def test_record_video_queue_can_skip_frame_clone(self) -> None:
        app = StereoCaptureOnlyApp.__new__(StereoCaptureOnlyApp)
        app._state_lock = threading.RLock()
        app.recording = True
        app._clone_frame = lambda _frame: self.fail("video queue should not clone full frames")

        queue: Queue = Queue(maxsize=1)
        left = object()
        right = object()

        ok = app._put_record_item(queue, {"index": 7, "left": left, "right": right}, clone_frames=False)

        self.assertTrue(ok)
        queued = queue.get_nowait()
        self.assertIs(queued["left"], left)
        self.assertIs(queued["right"], right)

    def test_raw_mono16_frame_converts_to_video_frame_without_pil_path(self) -> None:
        app = StereoCaptureOnlyApp.__new__(StereoCaptureOnlyApp)
        raw = np.array([[0, 65535], [32768, 16384]], dtype=np.uint16).tobytes()
        frame = Frame(
            image=object(),
            frame_number=1,
            width=2,
            height=2,
            host_timestamp=0,
            camera_timestamp=0,
            raw_data=raw,
            raw_frame_len=len(raw),
            pixel_type_name="PixelType_Gvsp_Mono16",
            raw_bit_depth=16,
            raw_array_shape=(2, 2),
        )

        video_frame = app._raw_frame_to_video_frame(frame)

        self.assertEqual(video_frame.shape, (2, 2, 3))
        self.assertEqual(int(video_frame[0, 0, 0]), 0)
        self.assertEqual(int(video_frame[0, 1, 0]), 255)

    def test_raw_frame_to_video_frame_accepts_non_contiguous_memoryview(self) -> None:
        app = StereoCaptureOnlyApp.__new__(StereoCaptureOnlyApp)
        source = np.arange(8, dtype=np.uint8).reshape(2, 4)[:, :3]
        frame = Frame(
            image=None,
            frame_number=1,
            width=3,
            height=2,
            host_timestamp=0,
            camera_timestamp=0,
            raw_data=memoryview(source),
            raw_frame_len=source.nbytes,
            pixel_type_name="PixelType_Gvsp_Mono8",
            raw_bit_depth=8,
            raw_array_shape=(2, 3),
        )

        video_frame = app._raw_frame_to_video_frame(frame)

        self.assertEqual(video_frame.shape, (2, 3, 3))
        self.assertEqual(int(video_frame[1, 2, 0]), 6)

    def test_record_preview_due_updates_every_half_second_at_two_fps(self) -> None:
        due, next_time = stereo_capture_only.record_preview_due(0.0, 0.0, 2.0)
        self.assertTrue(due)
        self.assertAlmostEqual(next_time, 0.5)

        due, next_time = stereo_capture_only.record_preview_due(0.49, next_time, 2.0)
        self.assertFalse(due)
        self.assertAlmostEqual(next_time, 0.5)

        due, next_time = stereo_capture_only.record_preview_due(0.5, next_time, 2.0)
        self.assertTrue(due)
        self.assertAlmostEqual(next_time, 1.0)

        due, next_time = stereo_capture_only.record_preview_due(2.2, next_time, 2.0)
        self.assertTrue(due)
        self.assertAlmostEqual(next_time, 2.5)

    def test_record_intervals_ignore_interval_capture_seconds_for_forced_images(self) -> None:
        interval, image_interval = stereo_capture_only.effective_record_intervals(
            {"record_force_image_format": True, "interval_capture_seconds": 5.0},
            10.0,
        )

        self.assertAlmostEqual(interval, 0.1)
        self.assertAlmostEqual(image_interval, 0.1)

    def test_thread_safe_config_setdefault_returns_wrapped_existing_dict(self) -> None:
        config = stereo_capture_only.ThreadSafeConfig({"nested": {"value": 1}})

        nested = config.setdefault("nested", {})

        self.assertIsInstance(nested, stereo_capture_only.ThreadSafeConfig)
        nested["added"] = 2
        self.assertEqual(config.snapshot()["nested"]["added"], 2)

    def test_calibration_file_storage_constructor_failure_is_propagated(self) -> None:
        with patch.object(calibration_manager.cv2, "FileStorage", side_effect=RuntimeError("open failed")):
            with self.assertRaises(RuntimeError):
                calibration_manager._load_calibration_file(Path("bad.yaml"))

    def test_record_frame_number_gap_is_counted_from_camera_sequence(self) -> None:
        app = StereoCaptureOnlyApp.__new__(StereoCaptureOnlyApp)
        app._record_stats_lock = threading.RLock()
        app._record_second_stats = {}
        app._record_first_trigger_time = None
        app.record_started_at = None
        app._record_last_camera_frame_numbers = {}
        app._record_frame_number_gap_count = 0
        app._notify_warning = lambda *_args, **_kwargs: None

        first = Frame(
            image=None,
            frame_number=10,
            width=1,
            height=1,
            host_timestamp=0.0,
            camera_timestamp=0,
        )
        second = Frame(
            image=None,
            frame_number=13,
            width=1,
            height=1,
            host_timestamp=0.0,
            camera_timestamp=0,
        )

        app._record_frame_numbers_observed(1, 100.0, first, None)
        app._record_frame_numbers_observed(2, 100.1, second, None)

        bucket = app._record_second_stats[0]
        self.assertEqual(app._record_frame_number_gap_count, 2)
        self.assertEqual(bucket["frame_number_gaps"], 2)
        self.assertEqual(bucket["drop_reasons"]["left_frame_number_gap"], 2)

    def test_high_bit_depth_frame_raw_helper_saves_npy(self) -> None:
        app = StereoCaptureOnlyApp.__new__(StereoCaptureOnlyApp)
        app._config_snapshot = lambda: {"save_raw_frames": False}
        raw = np.array([[0, 4095], [2048, 1024]], dtype=np.uint16).tobytes()
        frame = Frame(
            image=object(),
            frame_number=1,
            width=2,
            height=2,
            host_timestamp=0,
            camera_timestamp=0,
            raw_data=raw,
            raw_frame_len=len(raw),
            pixel_type_name="PixelType_Gvsp_Mono12",
            raw_bit_depth=12,
            raw_array_shape=(2, 2),
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = app._save_raw_frame(frame, Path(tmp) / "left_000001.bmp", {"raw_frame_format": "npy"})
            saved = np.load(path)

        self.assertEqual(path.suffix, ".npy")
        self.assertEqual(saved.dtype, np.uint16)
        self.assertEqual(saved.shape, (2, 2))

    def test_high_bit_depth_frames_can_be_saved_as_png16(self) -> None:
        app = StereoCaptureOnlyApp.__new__(StereoCaptureOnlyApp)
        raw = np.array([[0, 4095], [2048, 1024]], dtype=np.uint16).tobytes()
        frame = Frame(
            image=object(),
            frame_number=1,
            width=2,
            height=2,
            host_timestamp=0,
            camera_timestamp=0,
            raw_data=raw,
            raw_frame_len=len(raw),
            pixel_type_name="PixelType_Gvsp_Mono12",
            raw_bit_depth=12,
            raw_array_shape=(2, 2),
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = app._save_raw_frame(
                frame,
                Path(tmp) / "left_000001.bmp",
                {"raw_frame_format": "png16"},
            )
            saved = np.asarray(Image.open(path), dtype=np.uint16)

        self.assertEqual(path.suffix, ".png")
        self.assertEqual(saved.tolist(), [[0, 4095], [2048, 1024]])

    def test_high_bit_depth_frames_can_be_saved_as_tiff16(self) -> None:
        app = StereoCaptureOnlyApp.__new__(StereoCaptureOnlyApp)
        raw = np.array([[0, 65535], [32768, 1024]], dtype=np.uint16).tobytes()
        frame = Frame(
            image=object(),
            frame_number=1,
            width=2,
            height=2,
            host_timestamp=0,
            camera_timestamp=0,
            raw_data=raw,
            raw_frame_len=len(raw),
            pixel_type_name="PixelType_Gvsp_Mono16",
            raw_bit_depth=16,
            raw_array_shape=(2, 2),
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = app._save_raw_frame(
                frame,
                Path(tmp) / "left_000001.bmp",
                {"raw_frame_format": "tiff16"},
            )
            saved = np.asarray(Image.open(path), dtype=np.uint16)

        self.assertIn(path.suffix, {".tiff", ".tif"})
        self.assertEqual(saved.tolist(), [[0, 65535], [32768, 1024]])

    def test_high_bit_depth_frames_save_viewable_png_sidecar(self) -> None:
        app = StereoCaptureOnlyApp.__new__(StereoCaptureOnlyApp)
        raw = np.array([[100, 200], [300, 400]], dtype=np.uint16).tobytes()
        frame = Frame(
            image=object(),
            frame_number=1,
            width=2,
            height=2,
            host_timestamp=0,
            camera_timestamp=0,
            raw_data=raw,
            raw_frame_len=len(raw),
            pixel_type_name="PixelType_Gvsp_Mono16",
            raw_bit_depth=16,
            raw_array_shape=(2, 2),
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = app._save_raw_frame(
                frame,
                Path(tmp) / "left_000001.bmp",
                {
                    "raw_frame_format": "tiff16",
                    "image_format": "png",
                    "viewable_sidecar_enabled": True,
                    "viewable_sidecar_format": "png",
                },
            )
            view_path = path.with_suffix(".view.png")
            self.assertTrue(view_path.exists())
            with Image.open(view_path) as view:
                view_mode = view.mode
                view_size = view.size

        self.assertEqual(view_mode, "L")
        self.assertEqual(view_size, (2, 2))

    def test_packed_high_bit_depth_raw_is_not_miswritten_as_png16(self) -> None:
        app = StereoCaptureOnlyApp.__new__(StereoCaptureOnlyApp)
        frame = Frame(
            image=object(),
            frame_number=1,
            width=4,
            height=2,
            host_timestamp=0,
            camera_timestamp=0,
            raw_data=bytes(range(12)),
            raw_frame_len=12,
            pixel_type_name="PixelType_Gvsp_Mono12_Packed",
            raw_bit_depth=12,
        )

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(stereo_capture_only.LOGGER, "exception"), self.assertRaises(MvsError):
                app._save_raw_frame(
                    frame,
                    Path(tmp) / "left_000001.bmp",
                    {"raw_frame_format": "png16"},
                )

    def test_force_image_format_saves_high_bit_depth_frame_as_jpeg(self) -> None:
        app = StereoCaptureOnlyApp.__new__(StereoCaptureOnlyApp)
        frame = Frame(
            image=Image.new("L", (2, 2), 128),
            frame_number=1,
            width=2,
            height=2,
            host_timestamp=0,
            camera_timestamp=0,
            raw_data=np.array([[0, 4095], [2048, 1024]], dtype=np.uint16).tobytes(),
            raw_frame_len=8,
            pixel_type_name="PixelType_Gvsp_Mono16",
            raw_bit_depth=16,
            raw_array_shape=(2, 2),
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = app._save_frame(
                frame,
                Path(tmp) / "left_000001.jpg",
                {"record_force_image_format": True, "image_format": "jpg", "record_jpeg_quality": 100},
            )

        self.assertEqual(path.suffix, ".jpg")

    def test_capture_priority_config_uses_realtime_mp4_without_image_sequence(self) -> None:
        app = StereoCaptureOnlyApp.__new__(StereoCaptureOnlyApp)

        config = app._capture_priority_record_config(
            {
                "record_capture_priority_mode": True,
                "record_save_image_sequence": False,
                "auto_make_mp4": True,
                "record_realtime_mp4": True,
                "record_preview_during_capture": True,
                "record_clone_frames_for_writer": True,
                "record_checksum_during_capture": True,
                "record_queue_max_items": 8,
                "record_fps": 19.2,
                "preview_quality_analysis_enabled": True,
                "image_format": "jpg",
            }
        )

        self.assertFalse(config["record_save_image_sequence"])
        self.assertTrue(config["auto_make_mp4"])
        self.assertTrue(config["record_realtime_mp4"])
        self.assertTrue(config["record_preview_during_capture"])
        self.assertEqual(config["record_preview_fps"], 2.0)
        self.assertFalse(config["record_clone_frames_for_writer"])
        self.assertFalse(config["record_checksum_during_capture"])
        self.assertFalse(config["preview_quality_analysis_enabled"])
        self.assertGreaterEqual(config["record_queue_max_items"], 192)
        self.assertEqual(config["image_format"], "jpg")

    def test_capture_priority_config_can_be_disabled(self) -> None:
        app = StereoCaptureOnlyApp.__new__(StereoCaptureOnlyApp)
        original = {
            "record_capture_priority_mode": False,
            "record_save_image_sequence": False,
            "record_realtime_mp4": True,
            "record_preview_during_capture": True,
            "record_queue_max_items": 8,
            "image_format": "jpg",
        }

        expected = dict(original, pixel_format="Mono8", save_raw_frames=False, record_force_image_format=True)
        self.assertEqual(app._capture_priority_record_config(original), expected)

    def test_default_presets_include_dic_and_scientific_fields(self) -> None:
        presets = stereo_capture_only.default_presets()

        self.assertIn("DIC 标准", presets)
        self.assertEqual(presets["DIC 标准"]["trigger_source"], "Continuous")
        self.assertFalse(presets["DIC 标准"]["hardware_sync_enabled"])
        self.assertEqual(presets["DIC 标准"]["hardware_sync_master_line"], "Line2")
        self.assertEqual(presets["DIC 标准"]["hardware_sync_slave_line"], "Line0")
        self.assertEqual(presets["DIC 标准"]["pixel_format"], "Mono8")
        self.assertEqual(presets["DIC 标准"]["image_format"], "png")
        self.assertFalse(presets["DIC 标准"]["save_raw_frames"])
        self.assertTrue(presets["DIC 标准"]["record_force_image_format"])
        self.assertEqual(presets["DIC 标准"]["raw_frame_format"], "tiff16")
        self.assertTrue(presets["DIC 标准"]["chunk_data_enabled"])
        self.assertIn("black_level", presets["室内低光"])
        self.assertFalse(presets["室内低光"]["save_raw_frames"])
        self.assertEqual(presets["标定采集"]["pixel_format"], "Mono8")
        self.assertFalse(presets["标定采集"]["save_raw_frames"])

    def test_raw_frame_storage_estimate_uses_uncompressed_size(self) -> None:
        estimated = stereo_capture_only.estimate_frame_bytes(
            {"pixel_format": "Mono16", "save_raw_frames": True, "raw_frame_format": "tiff16", "image_format": "jpg"},
            10,
            10,
        )

        self.assertGreaterEqual(estimated, 150)

    def test_dic_capture_config_keeps_requested_outputs_and_camera_settings(self) -> None:
        app = StereoCaptureOnlyApp.__new__(StereoCaptureOnlyApp)
        app._config_snapshot = lambda: {"trigger_source": "Continuous", "record_capture_priority_mode": True}

        config = app._dic_capture_config()

        self.assertEqual(config["trigger_source"], "Continuous")
        self.assertFalse(config["require_hardware_trigger"])
        self.assertFalse(config["hardware_sync_enabled"])
        self.assertEqual(config["hardware_sync_master_line"], "Line2")
        self.assertEqual(config["hardware_sync_slave_line"], "Line0")
        self.assertEqual(config["pixel_format"], "Mono8")
        self.assertEqual(config["image_format"], "png")
        self.assertFalse(config["save_raw_frames"])
        self.assertEqual(config["raw_frame_format"], "tiff16")
        self.assertTrue(config["record_force_image_format"])
        self.assertEqual(config["record_jpeg_quality"], 100)
        self.assertEqual(config["record_fps"], 5.0)
        self.assertEqual(config["interval_capture_seconds"], 0.5)
        self.assertTrue(config["record_save_image_sequence"])
        self.assertTrue(config["record_realtime_mp4"])
        self.assertFalse(config["auto_make_mp4"])
        self.assertFalse(config["timestamp_reject_enabled"])
        self.assertFalse(config["capture_quality_gate"]["enabled"])
        self.assertFalse(config["record_capture_priority_mode"])

    def test_safe_trigger_config_clears_hardware_trigger_residue(self) -> None:
        config = stereo_capture_only.safe_trigger_config(
            {
                "trigger_source": "硬触发级联（无功能）",
                "require_hardware_trigger": True,
                "hardware_sync_enabled": True,
            }
        )

        self.assertEqual(config["trigger_source"], "Software")
        self.assertFalse(config["require_hardware_trigger"])
        self.assertFalse(config["hardware_sync_enabled"])

    def test_dic_capture_fps_entry_overrides_dic_record_fps_only(self) -> None:
        app = StereoCaptureOnlyApp.__new__(StereoCaptureOnlyApp)
        app.dic_record_fps_var = _Var()
        app.dic_record_fps_var.set("8")

        config = app._apply_dic_record_fps_to_config({"record_fps": 19.2, "dic_capture": {"record_fps": 5.0}})

        self.assertEqual(config["record_fps"], 8.0)
        self.assertEqual(config["dic_capture"]["record_fps"], 8.0)

    def test_dic_capture_always_uses_mono8_png_without_raw_outputs(self) -> None:
        app = StereoCaptureOnlyApp.__new__(StereoCaptureOnlyApp)
        app.dic_record_fps_var = _Var()
        app.dic_record_fps_var.set("5")

        config = app._apply_dic_ui_settings_to_config(
            {"pixel_format": "Mono16", "save_raw_frames": True, "dic_capture": {"record_fps": 5.0, "pixel_format": "Mono16"}}
        )

        self.assertEqual(config["pixel_format"], "Mono8")
        self.assertEqual(config["image_format"], "png")
        self.assertFalse(config["save_raw_frames"])
        self.assertTrue(config["record_force_image_format"])
        self.assertEqual(config["dic_capture"]["pixel_format"], "Mono8")
        self.assertFalse(config["dic_capture"]["save_raw_frames"])
        self.assertTrue(config["dic_capture"]["record_force_image_format"])
        self.assertEqual(config["viewable_sidecar_format"], "png")

    def test_dic_record_queue_uses_configured_capacity(self) -> None:
        self.assertEqual(
            stereo_capture_only.configured_record_queue_size(
                {"record_queue_max_items": 32, "record_fps": 5.0, "record_queue_force_configured": True}
            ),
            32,
        )

    def test_realtime_mp4_is_independent_from_post_sequence_mp4_flag(self) -> None:
        plan = stereo_capture_only.configured_record_outputs(
            {"record_save_image_sequence": True, "record_realtime_mp4": True, "auto_make_mp4": False},
            save_image_sequence=True,
        )

        self.assertFalse(plan["post_make_mp4"])
        self.assertTrue(plan["record_realtime_mp4"])
        self.assertFalse(plan["make_mp4_after"])
        self.assertTrue(plan["use_realtime_mp4"])
        self.assertEqual(plan["mp4_generation"], "opencv_realtime")

    def test_mp4_progress_total_counts_left_and_right_frame_paths(self) -> None:
        app = StereoCaptureOnlyApp.__new__(StereoCaptureOnlyApp)

        total = app._mp4_progress_total_units(
            [
                {"left_path": "left_1.bmp", "right_path": "right_1.bmp"},
                {"left_path": "left_2.bmp", "right_path": None},
                {"left_path": None, "right_path": "right_3.bmp"},
            ]
        )

        self.assertEqual(total, 4)

    def test_histogram_enabled_triggers_preview_analysis_without_quality_gate(self) -> None:
        app = StereoCaptureOnlyApp.__new__(StereoCaptureOnlyApp)
        app._histogram_enabled_setting = True
        app._focus_peaking_enabled_setting = False

        should_analyze = app._should_analyze_preview_frame(
            2,
            {
                "preview_quality_analysis_enabled": False,
                "preview_fps": 20.0,
                "preview_analysis_fps": 20.0,
            },
        )

        self.assertTrue(should_analyze)

    def test_preview_capture_timeout_uses_shorter_preview_value(self) -> None:
        app = StereoCaptureOnlyApp.__new__(StereoCaptureOnlyApp)

        timeout = app._preview_capture_timeout_ms(
            {
                "frame_timeout_ms": 3000,
                "preview_frame_timeout_ms": 500,
            }
        )

        self.assertEqual(timeout, 500)

    def test_preview_capture_timeout_does_not_exceed_frame_timeout(self) -> None:
        app = StereoCaptureOnlyApp.__new__(StereoCaptureOnlyApp)

        timeout = app._preview_capture_timeout_ms(
            {
                "frame_timeout_ms": 300,
                "preview_frame_timeout_ms": 500,
            }
        )

        self.assertEqual(timeout, 300)

    def test_capture_pair_timeout_override_reaches_grab_calls(self) -> None:
        system = StereoCameraSystem.__new__(StereoCameraSystem)
        system._capture_lock = threading.Lock()
        system.trigger_source = "Software"
        system.timeout_ms = 3000
        system.software_trigger_barrier_timeout_s = 1.0
        system.require_hardware_trigger = False
        system.timestamp_reject_enabled = False
        left = _FakeGrabCamera("left")
        right = _FakeGrabCamera("right")
        system._connected_cameras = lambda: [("left", left), ("right", right)]

        system.capture_pair(timeout_ms=500)

        self.assertEqual(left.grab_timeouts, [500])
        self.assertEqual(right.grab_timeouts, [500])
        self.assertEqual(left.trigger_count, 1)
        self.assertEqual(right.trigger_count, 1)

    def test_system_defaults_enable_host_timestamp_sync_threshold(self) -> None:
        system = StereoCameraSystem({})

        self.assertTrue(system.timestamp_reject_enabled)
        self.assertEqual(system.max_camera_timestamp_delta, 0)
        self.assertEqual(system.max_host_timestamp_delta, 10_000_000)
        system.close()

    def test_continuous_capture_does_not_fire_software_trigger(self) -> None:
        system = StereoCameraSystem.__new__(StereoCameraSystem)
        system._capture_lock = threading.Lock()
        system.trigger_source = "Continuous"
        system.timeout_ms = 800
        system.software_trigger_barrier_timeout_s = 1.0
        system.require_hardware_trigger = False
        system.timestamp_reject_enabled = False
        left = _FakeGrabCamera("left")
        right = _FakeGrabCamera("right")
        system._connected_cameras = lambda: [("left", left), ("right", right)]

        system.capture_pair(timeout_ms=250)

        self.assertEqual(left.grab_timeouts, [250])
        self.assertEqual(right.grab_timeouts, [250])
        self.assertEqual(left.trigger_count, 0)
        self.assertEqual(right.trigger_count, 0)

    def test_disabled_hardware_cascade_capture_falls_back_to_software_trigger(self) -> None:
        system = StereoCameraSystem.__new__(StereoCameraSystem)
        system._capture_lock = threading.Lock()
        system.trigger_source = "Cascade"
        system.timeout_ms = 800
        system.software_trigger_barrier_timeout_s = 1.0
        system.require_hardware_trigger = True
        system.hardware_sync_enabled = True
        system.hardware_sync_master = "left"
        system.timestamp_reject_enabled = False
        system._executor = None
        left = _FakeGrabCamera("left")
        right = _FakeGrabCamera("right")
        system._connected_cameras = lambda: [("left", left), ("right", right)]

        system.capture_pair(timeout_ms=250)

        self.assertEqual(left.grab_timeouts, [250])
        self.assertEqual(right.grab_timeouts, [250])
        self.assertEqual(left.trigger_count, 1)
        self.assertEqual(right.trigger_count, 1)
        self.assertFalse(system.require_hardware_trigger)
        self.assertFalse(system.hardware_sync_enabled)

    def test_system_apply_trigger_settings_disables_cascade_roles(self) -> None:
        system = StereoCameraSystem.__new__(StereoCameraSystem)
        system._capture_lock = threading.Lock()
        system.config = {
            "trigger_activation": "RisingEdge",
            "hardware_sync_enabled": False,
            "hardware_sync_master": "left",
            "hardware_sync_master_line": "Line2",
            "hardware_sync_master_line_source": "ExposureActive",
            "hardware_sync_slave_line": "Line0",
            "hardware_sync_slave_activation": "RisingEdge",
            "hardware_sync_master_trigger_source": "Software",
        }
        system.trigger_source = "Software"
        system.hardware_sync_enabled = False
        system.hardware_sync_master = "left"
        system.hardware_sync_master_line = "Line2"
        system.hardware_sync_master_line_source = "ExposureActive"
        system.hardware_sync_slave_line = "Line0"
        system.hardware_sync_slave_activation = "RisingEdge"
        system.hardware_sync_master_trigger_source = "Software"
        left = MvsCamera.__new__(MvsCamera)
        left.info = _Info()
        left._cam = _FakeLineNodeCamera()
        left._try_set_enum = lambda *_args, **_kwargs: False
        left._try_set_enum_by_string = lambda key, value: left._cam.enum_strings.append((key, value)) or True
        right = MvsCamera.__new__(MvsCamera)
        right.info = _Info()
        right._cam = _FakeLineNodeCamera()
        right._try_set_enum = lambda *_args, **_kwargs: False
        right._try_set_enum_by_string = lambda key, value: right._cam.enum_strings.append((key, value)) or True
        system._connected_cameras = lambda: [("left", left), ("right", right)]

        warnings = system.apply_trigger_settings("Cascade")

        self.assertEqual(warnings, [])
        self.assertEqual(system.config["trigger_source"], "Software")
        self.assertFalse(system.config["hardware_sync_enabled"])
        self.assertFalse(system.config["require_hardware_trigger"])
        self.assertIn(("TriggerSource", "Software"), left._cam.enum_strings)
        self.assertIn(("TriggerSource", "Software"), right._cam.enum_strings)
        self.assertNotIn(("TriggerSource", "Line0"), right._cam.enum_strings)

    def test_continuous_raw_capture_skips_pil_conversion(self) -> None:
        system = StereoCameraSystem.__new__(StereoCameraSystem)
        system._capture_lock = threading.Lock()
        system.trigger_source = "Continuous"
        system.timeout_ms = 800
        system.software_trigger_barrier_timeout_s = 1.0
        system.require_hardware_trigger = False
        system.timestamp_reject_enabled = False
        left = _FakeGrabCamera("left")
        right = _FakeGrabCamera("right")
        system._connected_cameras = lambda: [("left", left), ("right", right)]

        l_frame, r_frame, _trigger_time = system.capture_pair(timeout_ms=250, convert_image=False)

        self.assertIsNone(l_frame.image)
        self.assertIsNone(r_frame.image)
        self.assertEqual(left.convert_image_values, [False])
        self.assertEqual(right.convert_image_values, [False])

    def test_continuous_packet_pair_uses_closest_host_timestamp(self) -> None:
        system = StereoCameraSystem.__new__(StereoCameraSystem)
        system.max_camera_timestamp_delta = 0
        system._camera_timestamp_offset = None
        left_packets = [
            RawFramePacket(b"l1", 2, 1, 1, 1, 1, 100, 0),
            RawFramePacket(b"l2", 2, 1, 1, 1, 2, 200, 0),
        ]
        right_packets = [
            RawFramePacket(b"r1", 2, 1, 1, 1, 1, 130, 0),
            RawFramePacket(b"r2", 2, 1, 1, 1, 2, 205, 0),
        ]

        left, right = system._select_best_continuous_packet_pair(left_packets, right_packets)

        self.assertIs(left, left_packets[1])
        self.assertIs(right, right_packets[1])

    def test_continuous_frame_number_gap_is_logged(self) -> None:
        system = StereoCameraSystem.__new__(StereoCameraSystem)
        packets = [
            RawFramePacket(b"1", 1, 1, 1, 1, 10, 100, 0),
            RawFramePacket(b"2", 1, 1, 1, 1, 14, 101, 0),
        ]

        with self.assertLogs("mvss_capture", level="WARNING") as captured:
            system._warn_continuous_frame_number_gaps(packets, "left")

        self.assertIn("frame number gap", "\n".join(captured.output))
        self.assertIn("previous=10", "\n".join(captured.output))

    def test_selected_continuous_frame_number_gap_is_logged_across_batches(self) -> None:
        system = StereoCameraSystem.__new__(StereoCameraSystem)
        system._last_continuous_frame_numbers = {}
        first_left = RawFramePacket(b"l1", 1, 1, 1, 1, 10, 100, 0)
        first_right = RawFramePacket(b"r1", 1, 1, 1, 1, 11, 101, 0)
        second_left = RawFramePacket(b"l2", 1, 1, 1, 1, 15, 102, 0)
        second_right = RawFramePacket(b"r2", 1, 1, 1, 1, 16, 103, 0)

        system._warn_selected_continuous_frame_numbers(first_left, first_right)
        with self.assertLogs("mvss_capture", level="WARNING") as captured:
            system._warn_selected_continuous_frame_numbers(second_left, second_right)

        text = "\n".join(captured.output)
        self.assertIn("selected frame number gap", text)
        self.assertIn("previous=10", text)

    def test_selected_continuous_left_right_frame_number_delta_is_logged(self) -> None:
        system = StereoCameraSystem.__new__(StereoCameraSystem)
        system._last_continuous_frame_numbers = {}

        with self.assertLogs("mvss_capture", level="WARNING") as captured:
            system._warn_selected_continuous_frame_numbers(
                RawFramePacket(b"l", 1, 1, 1, 1, 10, 100, 0),
                RawFramePacket(b"r", 1, 1, 1, 1, 20, 101, 0),
            )

        self.assertIn("left/right frame number delta", "\n".join(captured.output))

    def test_preview_analysis_accepts_unlimited_preview_fps(self) -> None:
        app = StereoCaptureOnlyApp.__new__(StereoCaptureOnlyApp)
        app._histogram_enabled_setting = True
        app._focus_peaking_enabled_setting = False

        self.assertTrue(
            app._should_analyze_preview_frame(
                2,
                {
                    "preview_quality_analysis_enabled": False,
                    "preview_fps": 0,
                    "preview_analysis_fps": 20.0,
                },
            )
        )

    def test_unlimited_preview_analysis_is_time_gated(self) -> None:
        app = StereoCaptureOnlyApp.__new__(StereoCaptureOnlyApp)
        app._histogram_enabled_setting = True
        app._focus_peaking_enabled_setting = False
        config = {
            "preview_quality_analysis_enabled": False,
            "preview_fps": 0,
            "preview_analysis_fps": 20.0,
        }
        with patch.object(stereo_capture_only.time, "perf_counter", side_effect=[100.0, 100.2, 101.0]):
            self.assertTrue(app._should_analyze_preview_frame(2, config))
            self.assertFalse(app._should_analyze_preview_frame(3, config))
            self.assertTrue(app._should_analyze_preview_frame(4, config))

    def test_configured_preview_fps_clamps_zero_to_default(self) -> None:
        self.assertEqual(stereo_capture_only.configured_preview_fps({"preview_fps": 0}), 15.0)

    def test_record_queue_size_keeps_ten_seconds_of_target_fps(self) -> None:
        self.assertGreaterEqual(
            stereo_capture_only.configured_record_queue_size({"record_queue_max_items": 64, "record_fps": 19.2}),
            192,
        )

    def test_camera_timestamp_offset_uses_sliding_median_after_valid_pairs(self) -> None:
        system = StereoCameraSystem.__new__(StereoCameraSystem)
        system.timestamp_reject_enabled = True
        system.max_camera_timestamp_delta = 20
        system.max_host_timestamp_delta = 0
        system.camera_timestamp_offset_samples = 3
        system._camera_timestamp_offset_samples = mvs_camera.deque(maxlen=3)
        system._camera_timestamp_offset = None

        for delta in (100, 102, 104):
            system._validate_frame_sync(
                Frame(object(), 1, 1, 1, 0, 1000 + delta),
                Frame(object(), 1, 1, 1, 0, 1000),
            )
        self.assertEqual(system._camera_timestamp_offset, 102)

        for delta in (110, 112, 114):
            system._validate_frame_sync(
                Frame(object(), 1, 1, 1, 0, 1000 + delta),
                Frame(object(), 1, 1, 1, 0, 1000),
            )
        self.assertEqual(system._camera_timestamp_offset, 112)
        self.assertEqual(list(system._camera_timestamp_offset_samples), [110, 112, 114])

    def test_camera_timestamp_offset_rejects_outlier_without_learning_it(self) -> None:
        system = StereoCameraSystem.__new__(StereoCameraSystem)
        system.timestamp_reject_enabled = True
        system.max_camera_timestamp_delta = 5
        system.max_host_timestamp_delta = 0
        system.camera_timestamp_offset_samples = 3
        system._camera_timestamp_offset_samples = mvs_camera.deque([100, 101, 102], maxlen=3)
        system._camera_timestamp_offset = 101

        with self.assertRaises(mvs_camera.FrameSyncError):
            system._validate_frame_sync(
                Frame(object(), 1, 1, 1, 0, 2000),
                Frame(object(), 1, 1, 1, 0, 1000),
            )

        self.assertEqual(system._camera_timestamp_offset, 101)
        self.assertEqual(list(system._camera_timestamp_offset_samples), [100, 101, 102])

    def test_fixed_camera_timestamp_offset_seeds_samples(self) -> None:
        system = StereoCameraSystem.__new__(StereoCameraSystem)
        system.camera_timestamp_offset_samples = 3
        system._camera_timestamp_offset_samples = mvs_camera.deque(maxlen=8)
        system._camera_timestamp_offset = None

        system.set_camera_timestamp_offset(123)

        self.assertEqual(system.camera_timestamp_offset(), 123)
        self.assertEqual(list(system._camera_timestamp_offset_samples), [123, 123, 123])

    def test_camera_timestamp_offset_calibration_uses_median_and_restores_reject(self) -> None:
        system = StereoCameraSystem.__new__(StereoCameraSystem)
        system.camera_timestamp_offset_samples = 3
        system._camera_timestamp_offset_samples = mvs_camera.deque(maxlen=8)
        system._camera_timestamp_offset = None
        system.timestamp_reject_enabled = True
        system.config = {}
        deltas = iter([100, 104, 102])

        def fake_capture_pair(timeout_ms=None):
            delta = next(deltas)
            return (
                Frame(object(), 1, 1, 1, 0, 1000 + delta),
                Frame(object(), 1, 1, 1, 0, 1000),
                0.0,
            )

        system.capture_pair = fake_capture_pair

        offset = system.calibrate_camera_timestamp_offset(sample_count=3)

        self.assertEqual(offset, 102)
        self.assertTrue(system.timestamp_reject_enabled)
        self.assertEqual(system.config["camera_timestamp_offset_fixed"], 102)

    def test_record_status_text_accepts_unlimited_target_fps(self) -> None:
        app = StereoCaptureOnlyApp.__new__(StereoCaptureOnlyApp)
        app._record_elapsed_seconds = lambda: 1.0
        app._record_free_space_gb = lambda: 100.0
        app._record_write_state_snapshot = lambda: (0.0, "", 1, 1)
        app._record_counter_values = lambda: (10, 8)
        app._config_snapshot = lambda: {}

        text = app._record_status_text(None, 8.0, {})

        self.assertIn("max", text)

    def test_interval_status_text_includes_progress_and_eta_when_limited(self) -> None:
        app = StereoCaptureOnlyApp.__new__(StereoCaptureOnlyApp)
        app.interval_count = 3

        text = app._interval_status_text(0.5, 10, "capture_001", 1.5)

        self.assertIn("已保存 3/10 组", text)
        self.assertIn("剩余 7 组", text)
        self.assertIn("预计 00:03", text)

    def test_interval_status_text_keeps_continuous_mode_simple(self) -> None:
        app = StereoCaptureOnlyApp.__new__(StereoCaptureOnlyApp)
        app.interval_count = 3

        text = app._interval_status_text(0.5, None, "capture_001", 1.5)

        self.assertIn("已保存 3 组", text)
        self.assertNotIn("/", text)

    def test_apply_capture_config_preserves_zero_black_level(self) -> None:
        app = StereoCaptureOnlyApp.__new__(StereoCaptureOnlyApp)
        calls: list[tuple[float | None, float | None, float | None]] = []
        camera_system = type(
            "FakeCameraSystem",
            (),
            {
                "trigger_source": "Software",
                "timestamp_reject_enabled": False,
                "max_camera_timestamp_delta": 0,
                "max_host_timestamp_delta": 0,
                "apply_pixel_format_settings": lambda self, _pixel_format: [],
                "apply_trigger_settings": lambda self, _trigger_source: [],
                "apply_exposure_settings": lambda self, *_args: [],
                "apply_gain_settings": lambda self, *_args: [],
                "apply_image_correction_settings": lambda self, black, shift, gamma: calls.append(
                    (black, shift, gamma)
                )
                or [],
                "apply_side_roi_settings": lambda self, _rois, restart_stream=True: ({}, []),
            },
        )()
        camera_system.config = {}
        app._require_camera_system = lambda: camera_system

        app._apply_capture_config_to_camera(
            {
                "pixel_format": "Mono16",
                "trigger_source": "Line0",
                "require_hardware_trigger": True,
                "hardware_sync_enabled": True,
                "exposure_auto": "Off",
                "exposure_time_us": 1000.0,
                "gain_auto": "Off",
                "gain": 0.0,
                "black_level": 0.0,
                "digital_shift": "",
                "gamma": None,
            }
        )

        self.assertEqual(calls, [(0.0, None, None)])
        self.assertEqual(camera_system.trigger_source, "Software")
        self.assertFalse(camera_system.require_hardware_trigger)
        self.assertFalse(camera_system.hardware_sync_enabled)

    def test_field_correction_subtracts_dark_and_preserves_uint16_raw(self) -> None:
        app = StereoCaptureOnlyApp.__new__(StereoCaptureOnlyApp)
        app.config = {"field_correction": {"enabled": True}}
        app._field_correction_lock = threading.Lock()
        app._dark_frame_refs = {"left": np.array([[10, 10], [10, 10]], dtype=np.float32)}
        app._flat_field_refs = {}
        raw = np.array([[20, 30], [40, 50]], dtype=np.uint16).tobytes()
        frame = Frame(
            image=None,
            frame_number=1,
            width=2,
            height=2,
            host_timestamp=0,
            camera_timestamp=0,
            raw_data=raw,
            raw_frame_len=len(raw),
            pixel_type_name="PixelType_Gvsp_Mono16",
            raw_bit_depth=16,
        )

        corrected = app._correct_frame(frame, "left")

        self.assertIsNotNone(corrected)
        saved = np.frombuffer(corrected.raw_data, dtype=np.uint16).reshape((2, 2))
        self.assertEqual(saved.tolist(), [[10, 20], [30, 40]])

    def test_speckle_quality_scores_textured_image_above_flat(self) -> None:
        flat = Image.fromarray(np.full((64, 64), 128, dtype=np.uint8), "L")
        rng = np.random.default_rng(123)
        textured_array = rng.integers(64, 192, size=(64, 64), dtype=np.uint8)
        textured = Image.fromarray(textured_array, "L")

        flat_quality = image_quality.speckle_quality(flat)
        textured_quality = image_quality.speckle_quality(textured)

        self.assertLess(flat_quality["score"], textured_quality["score"])
        self.assertIn(textured_quality["rating"], {"usable", "good"})

    def test_displacement_overlay_composites_configured_output(self) -> None:
        app = StereoCaptureOnlyApp.__new__(StereoCaptureOnlyApp)
        app.status_var = _Var()
        frame = Frame(
            image=Image.fromarray(np.zeros((2, 2), dtype=np.uint8), "L"),
            frame_number=1,
            width=2,
            height=2,
            host_timestamp=0,
            camera_timestamp=0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            overlay_path = Path(tmp) / "overlay.png"
            overlay = np.zeros((2, 2, 4), dtype=np.uint8)
            overlay[..., 0] = 255
            overlay[..., 3] = 255
            Image.fromarray(overlay, "RGBA").save(overlay_path)
            app.config = {"dic_analysis": {"overlay_path": str(overlay_path)}}

            result = app._displacement_overlay_for_frame(frame)

        self.assertIsNotNone(result)
        self.assertEqual(result.getpixel((0, 0)), (255, 0, 0))

    def test_roi_warmup_uses_configured_short_timeout(self) -> None:
        system = StereoCameraSystem.__new__(StereoCameraSystem)
        system.config = {
            "roi_restart_settle_seconds": 0,
            "roi_warmup_frames": 2,
            "roi_warmup_timeout_ms": 400,
        }
        system.trigger_source = "Software"
        system.timeout_ms = 3000
        system.software_trigger_barrier_timeout_s = 1.0
        system.require_hardware_trigger = False
        system.timestamp_reject_enabled = False
        left = _FakeGrabCamera("left")
        right = _FakeGrabCamera("right")
        system._connected_cameras = lambda: [("left", left), ("right", right)]

        warnings = system._warm_up_after_roi_locked()

        self.assertEqual(warnings, [])
        self.assertEqual(left.grab_timeouts, [400, 400])
        self.assertEqual(right.grab_timeouts, [400, 400])

    def test_side_roi_adjusts_right_size_to_match_left_while_preserving_offset(self) -> None:
        system = StereoCameraSystem.__new__(StereoCameraSystem)
        system._capture_lock = threading.Lock()
        system.config = {"roi_restart_settle_seconds": 0, "roi_warmup_frames": 0}
        system.trigger_source = "Software"
        left = _FakeRoiCamera([(100, 80, 0, 0)])
        right = _FakeRoiCamera([(96, 80, 20, 10), (100, 80, 20, 10)])
        system._connected_cameras = lambda: [("left", left), ("right", right)]

        results, warnings = system.apply_side_roi_settings(
            {"left": (100, 80, 0, 0), "right": (96, 80, 20, 10)},
            restart_stream=False,
        )

        self.assertEqual(results["right"].actual_roi, (100, 80, 20, 10))
        self.assertEqual(right.calls[-1], (100, 80, 20, 10, False))
        self.assertTrue(any("Right camera ROI size adjusted" in warning for warning in warnings))

    def test_camera_assignment_maps_selected_devices_to_left_and_right_serials(self) -> None:
        app = StereoCaptureOnlyApp.__new__(StereoCaptureOnlyApp)
        app.config = {}
        app.left_camera_var = _Var()
        app.right_camera_var = _Var()
        app._camera_choice_serials = {stereo_capture_only.CAMERA_ASSIGNMENT_AUTO: ""}
        app._set_camera_assignment_menus = lambda _choices: None

        cameras = [
            _FakeCameraInfo(0, "LEFT123", "Camera A"),
            _FakeCameraInfo(1, "RIGHT456", "Camera B"),
        ]
        app._sync_camera_assignment_controls(cameras)
        labels = {serial: label for label, serial in app._camera_choice_serials.items() if serial}
        app.left_camera_var.set(labels["LEFT123"])
        app.right_camera_var.set(labels["RIGHT456"])

        values = app._selected_camera_assignment_config()

        self.assertEqual(values["left_serial"], "LEFT123")
        self.assertEqual(values["right_serial"], "RIGHT456")
        self.assertTrue(values["bind_camera_serials"])

    def test_camera_assignment_rejects_same_camera_for_both_views(self) -> None:
        app = StereoCaptureOnlyApp.__new__(StereoCaptureOnlyApp)
        app.config = {}
        app.left_camera_var = _Var()
        app.right_camera_var = _Var()
        app._camera_choice_serials = {stereo_capture_only.CAMERA_ASSIGNMENT_AUTO: ""}
        app._set_camera_assignment_menus = lambda _choices: None

        app._sync_camera_assignment_controls([_FakeCameraInfo(0, "CAM123", "Camera A")])
        label = next(label for label, serial in app._camera_choice_serials.items() if serial == "CAM123")
        app.left_camera_var.set(label)
        app.right_camera_var.set(label)

        with self.assertRaises(ValueError):
            app._selected_camera_assignment_config()

    def test_camera_assignment_preserves_saved_serials_before_refresh(self) -> None:
        app = StereoCaptureOnlyApp.__new__(StereoCaptureOnlyApp)
        app.config = {"left_serial": "SAVED_L", "right_serial": "SAVED_R", "bind_camera_serials": True}
        app.left_camera_var = _Var()
        app.right_camera_var = _Var()
        app._camera_choice_serials = {stereo_capture_only.CAMERA_ASSIGNMENT_AUTO: ""}
        app._available_cameras = []
        app._set_camera_assignment_menus = lambda _choices: None

        app._sync_camera_assignment_controls()
        values = app._selected_camera_assignment_config()

        self.assertEqual(values["left_serial"], "SAVED_L")
        self.assertEqual(values["right_serial"], "SAVED_R")
        self.assertTrue(values["bind_camera_serials"])

    def test_single_bound_right_serial_connects_single_camera_as_right_view(self) -> None:
        camera = _FakeCameraInfo(0, "RIGHT456", "Camera B")

        with patch("mvs_camera.enumerate_cameras", return_value=([camera], object())):
            left, right, _dev_list = mvs_camera.select_capture_devices(
                left_serial="",
                right_serial="RIGHT456",
                allow_single=True,
                bind_serials=True,
            )

        self.assertIsNone(left)
        self.assertIs(right, camera)

    def test_left_preview_roi_syncs_right_size_for_stereo_capture(self) -> None:
        app = StereoCaptureOnlyApp.__new__(StereoCaptureOnlyApp)
        app.focus_roi_editing = False
        app.camera_system = None
        app.status_var = _Var()
        app._set_roi_edit_mode = lambda enabled: setattr(app, "_roi_edit_mode", enabled)
        app.left_roi_width_var = _Var()
        app.left_roi_height_var = _Var()
        app.left_roi_offset_x_var = _Var()
        app.left_roi_offset_y_var = _Var()
        app.right_roi_width_var = _Var()
        app.right_roi_height_var = _Var()
        app.right_roi_offset_x_var = _Var()
        app.right_roi_offset_y_var = _Var()
        app.right_roi_width_var.set("999")
        app.right_roi_height_var.set("888")
        app.right_roi_offset_x_var.set("60")
        app.right_roi_offset_y_var.set("40")

        app.set_roi_from_preview((10, 20, 320, 240), side="left")

        self.assertEqual(app.left_roi_width_var.get(), "320")
        self.assertEqual(app.left_roi_height_var.get(), "240")
        self.assertEqual(app.left_roi_offset_x_var.get(), "10")
        self.assertEqual(app.left_roi_offset_y_var.get(), "20")
        self.assertEqual(app.right_roi_width_var.get(), "320")
        self.assertEqual(app.right_roi_height_var.get(), "240")
        self.assertEqual(app.right_roi_offset_x_var.get(), "60")
        self.assertEqual(app.right_roi_offset_y_var.get(), "40")

    def test_right_preview_roi_preserves_left_size_and_updates_right_position(self) -> None:
        app = StereoCaptureOnlyApp.__new__(StereoCaptureOnlyApp)
        app.focus_roi_editing = False
        app.camera_system = None
        app.status_var = _Var()
        app._set_roi_edit_mode = lambda enabled: setattr(app, "_roi_edit_mode", enabled)
        app.left_roi_width_var = _Var()
        app.left_roi_height_var = _Var()
        app.left_roi_offset_x_var = _Var()
        app.left_roi_offset_y_var = _Var()
        app.right_roi_width_var = _Var()
        app.right_roi_height_var = _Var()
        app.right_roi_offset_x_var = _Var()
        app.right_roi_offset_y_var = _Var()
        app.left_roi_width_var.set("320")
        app.left_roi_height_var.set("240")
        app.left_roi_offset_x_var.set("10")
        app.left_roi_offset_y_var.set("20")

        app.set_roi_from_preview((70, 55, 500, 400), side="right")

        self.assertEqual(app.left_roi_width_var.get(), "320")
        self.assertEqual(app.left_roi_height_var.get(), "240")
        self.assertEqual(app.left_roi_offset_x_var.get(), "10")
        self.assertEqual(app.left_roi_offset_y_var.get(), "20")
        self.assertEqual(app.right_roi_width_var.get(), "320")
        self.assertEqual(app.right_roi_height_var.get(), "240")
        self.assertEqual(app.right_roi_offset_x_var.get(), "70")
        self.assertEqual(app.right_roi_offset_y_var.get(), "55")

    def test_stream_stats_are_aggregated_for_ui(self) -> None:
        system = StereoCameraSystem.__new__(StereoCameraSystem)
        system._capture_lock = threading.Lock()
        left = MvsCamera.__new__(MvsCamera)
        left._stream_condition = threading.Condition()
        left._stream_frames = []
        left._stream_dropped_frames = 3
        left._stream_callback_enabled = True
        left._try_get_int_any = lambda keys: 7 if "DeviceLinkErrorCount" in keys else None
        right = MvsCamera.__new__(MvsCamera)
        right._stream_condition = threading.Condition()
        right._stream_frames = [object()]
        right._stream_dropped_frames = 0
        right._stream_callback_enabled = False
        right._try_get_int_any = lambda _keys: None
        system._connected_cameras = lambda: [("left", left), ("right", right)]

        stats = system.stream_stats()

        self.assertEqual(stats["left"]["dropped_frames"], 3)
        self.assertTrue(stats["left"]["callback_enabled"])
        self.assertEqual(stats["left"]["link_error_count"], 7)
        self.assertEqual(stats["right"]["buffered_frames"], 1)

    def test_temperature_display_shows_stream_drop_counter(self) -> None:
        app = StereoCaptureOnlyApp.__new__(StereoCaptureOnlyApp)
        app.temperature_status_var = _Var()
        app.config = {"temperature_monitor": {}}

        app._update_temperature_display(
            {
                "temperatures_c": {},
                "link_throughput_mbps": {"left": 750.0},
                "stream_stats": {"left": {"dropped_frames": 5}, "right": {"dropped_frames": 0}},
            }
        )

        self.assertIn("Link left:750Mbps", app.temperature_status_var.value)
        self.assertIn("StreamDrop left:5", app.temperature_status_var.value)

    def test_camera_health_text_reports_firmware_and_link_counters(self) -> None:
        app = StereoCaptureOnlyApp.__new__(StereoCaptureOnlyApp)
        app.camera_health_var = _Var()
        app._device_versions = {"left": "1.0", "right": "1.1"}
        app._temperature_samples = []
        app._update_temperature_trend_chart = lambda *_args, **_kwargs: None

        app._update_camera_health_display(
            {"left": 40.0},
            {"left": 750.0},
            {"left": {"link_error_count": 2, "resend_packet_count": 3}},
        )

        self.assertIn("版本不一致", app.camera_health_var.value)
        self.assertIn("left:err 2/resend 3", app.camera_health_var.value)

    def test_stream_stats_poll_runs_when_temperature_monitor_disabled(self) -> None:
        app = StereoCaptureOnlyApp.__new__(StereoCaptureOnlyApp)
        app.config = {"temperature_monitor": {"enabled": False, "stream_stats_interval_seconds": 1.0}}
        app.camera_system = _FakeStatsSystem()
        app._latest_temperatures = {}
        app._latest_link_throughput_mbps = {}
        app._latest_stream_stats = {}
        app._temperature_samples = []
        app._last_temperature_poll = 0.0
        app._last_stream_stats_poll = 0.0
        app.ui_queue = Queue()

        with patch.object(stereo_capture_only.time, "perf_counter", return_value=2.0):
            app._poll_camera_temperatures()

        self.assertEqual(app.camera_system.temperature_reads, 0)
        self.assertEqual(app.camera_system.stream_reads, 1)
        self.assertEqual(app._latest_stream_stats["left"]["dropped_frames"], 2)
        self.assertEqual(app._temperature_samples, [])
        kind, payload = app.ui_queue.get_nowait()
        self.assertEqual(kind, "temperature")
        self.assertEqual(payload["stream_stats"]["left"]["dropped_frames"], 2)

    def test_video_segment_size_estimate_uses_bitrate_and_fps(self) -> None:
        app = StereoCaptureOnlyApp.__new__(StereoCaptureOnlyApp)
        app._record_stats_lock = threading.RLock()
        app._record_split_index = 1
        app._record_segment_start_saved = 0

        estimated = app._estimate_video_segment_bytes(
            96,
            1,
            {"video_bitrate_kbps": 8000, "record_fps": 19.2},
        )

        self.assertEqual(estimated, 10_000_000)

    def test_background_thread_is_tracked_and_removed(self) -> None:
        app = StereoCaptureOnlyApp.__new__(StereoCaptureOnlyApp)
        app._closing = False
        app._background_threads_lock = threading.Lock()
        app._background_threads = []
        app.ui_queue = Queue()

        thread = app._start_background_thread(lambda: None, "unit-background")
        thread.join(timeout=1.0)

        self.assertFalse(thread.is_alive())
        self.assertEqual(app._background_threads_snapshot(), [])

    def test_estimate_snr_samples_large_regions(self) -> None:
        original_mean = image_quality.np.mean
        calls = 0

        def counting_mean(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original_mean(*args, **kwargs)

        try:
            image_quality.np.mean = counting_mean
            image_quality.estimate_snr_db(image_quality.np.full((960, 960), 128, dtype=image_quality.np.uint8))
        finally:
            image_quality.np.mean = original_mean

        self.assertLessEqual(calls, 96)


if __name__ == "__main__":
    unittest.main()

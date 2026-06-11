from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from queue import Queue
from unittest.mock import patch

import image_quality
import mvs_camera
from mvs_camera import Frame, MvsCamera, RawFramePacket, StereoCameraSystem
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
        self.trigger_count = 0

    def trigger_software(self) -> None:
        self.trigger_count += 1

    def grab_frame(self, timeout_ms: int) -> Frame:
        self.grab_timeouts.append(timeout_ms)
        frame_number = len(self.grab_timeouts)
        return Frame(
            image=object(),
            frame_number=frame_number,
            width=1,
            height=1,
            host_timestamp=frame_number,
            camera_timestamp=frame_number,
        )


class _FakeNodeCamera:
    def __init__(self):
        self.enum_strings: list[tuple[str, str]] = []


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


class _Var:
    def __init__(self):
        self.value = ""

    def set(self, value) -> None:
        self.value = str(value)


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


class ReliabilityFixTests(unittest.TestCase):
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

    def test_capture_priority_config_uses_sequence_and_disables_realtime_work(self) -> None:
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
                "preview_quality_analysis_enabled": True,
                "image_format": "jpg",
            }
        )

        self.assertTrue(config["record_save_image_sequence"])
        self.assertTrue(config["auto_make_mp4"])
        self.assertFalse(config["record_realtime_mp4"])
        self.assertFalse(config["record_preview_during_capture"])
        self.assertFalse(config["record_clone_frames_for_writer"])
        self.assertFalse(config["record_checksum_during_capture"])
        self.assertFalse(config["preview_quality_analysis_enabled"])
        self.assertGreaterEqual(config["record_queue_max_items"], 64)
        self.assertEqual(config["image_format"], "bmp")

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

        self.assertEqual(app._capture_priority_record_config(original), original)

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

    def test_record_status_text_accepts_unlimited_target_fps(self) -> None:
        app = StereoCaptureOnlyApp.__new__(StereoCaptureOnlyApp)
        app._record_elapsed_seconds = lambda: 1.0
        app._record_free_space_gb = lambda: 100.0
        app._record_write_state_snapshot = lambda: (0.0, "", 1, 1)
        app._record_counter_values = lambda: (10, 8)
        app._config_snapshot = lambda: {}

        text = app._record_status_text(None, 8.0, {})

        self.assertIn("max", text)

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

    def test_stream_stats_are_aggregated_for_ui(self) -> None:
        system = StereoCameraSystem.__new__(StereoCameraSystem)
        system._capture_lock = threading.Lock()
        left = MvsCamera.__new__(MvsCamera)
        left._stream_condition = threading.Condition()
        left._stream_frames = []
        left._stream_dropped_frames = 3
        left._stream_callback_enabled = True
        right = MvsCamera.__new__(MvsCamera)
        right._stream_condition = threading.Condition()
        right._stream_frames = [object()]
        right._stream_dropped_frames = 0
        right._stream_callback_enabled = False
        system._connected_cameras = lambda: [("left", left), ("right", right)]

        stats = system.stream_stats()

        self.assertEqual(stats["left"]["dropped_frames"], 3)
        self.assertTrue(stats["left"]["callback_enabled"])
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

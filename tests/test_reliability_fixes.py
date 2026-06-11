from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from queue import Queue

import image_quality
from mvs_camera import Frame, MvsCamera, StereoCameraSystem
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

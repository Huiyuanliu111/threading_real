from __future__ import annotations

import importlib.util
from pathlib import Path
import queue

import cv2
import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location(
    "video_recording", Path(__file__).resolve().parents[1]
    / "scripts" / "deployment" / "video_recording.py",
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class FakeRig:
    view_names = ("sideview", "frontview")
    color_intrinsics = [{"width": 32, "height": 24}] * 2

    def __init__(self, depth=False):
        self.enable_depth = depth
        self.frames = queue.Queue()
        self.closed = False

    def read(self, timeout_ms):
        frame = self.frames.get(timeout=0.2)
        if isinstance(frame, Exception):
            raise frame
        return frame

    read_rgbd = read

    def close(self):
        self.closed = True


@pytest.mark.parametrize("depth", [False, True])
def test_records_both_views_without_policy_reads_and_splits_episodes(tmp_path, depth):
    rig = FakeRig(depth)
    camera = module.RecordingCameraRig(rig)
    red = np.full((24, 32, 3), (255, 0, 0), dtype=np.uint8)
    blue = np.full((24, 32, 3), (0, 0, 255), dtype=np.uint8)
    frames = ((red, np.zeros((24, 32))), None, (blue, np.zeros((24, 32)))) if depth else (red, None, blue)
    try:
        for episode in (1, 2):
            camera.start_recording(tmp_path / str(episode))
            for _ in range(4):
                with camera._condition:
                    previous = camera._sequence
                    rig.frames.put(frames)
                    assert camera._condition.wait_for(lambda: camera._sequence > previous, timeout=1)
            observed = camera.read_rgbd() if depth else camera.read()
            assert observed is frames
            camera.stop_recording()
    finally:
        camera.close()
    assert rig.closed
    assert not camera._thread.is_alive()
    for episode in (1, 2):
        for filename, bgr in (("cam1.mp4", (0, 0, 255)), ("cam3.mp4", (255, 0, 0))):
            video = cv2.VideoCapture(str(tmp_path / str(episode) / filename))
            try:
                assert video.get(cv2.CAP_PROP_FRAME_COUNT) == 4
                assert video.get(cv2.CAP_PROP_FPS) == 30
                ok, image = video.read()
                assert ok
                np.testing.assert_allclose(image.mean(axis=(0, 1)), bgr, atol=8)
            finally:
                video.release()


def test_acquisition_failure_propagates_and_close_releases_video(tmp_path):
    rig = FakeRig()
    camera = module.RecordingCameraRig(rig)
    camera.start_recording(tmp_path / "episode")
    rig.frames.put(RuntimeError("camera disconnected"))
    with pytest.raises(RuntimeError, match="background camera"):
        camera.read()
    camera.close()
    assert not camera._writers
    assert rig.closed


def test_writer_open_failure_releases_all_writers(tmp_path, monkeypatch):
    class FailedWriter:
        released = False

        def isOpened(self):
            return False

        def release(self):
            self.released = True

    writer = FailedWriter()
    monkeypatch.setattr(module.cv2, "VideoWriter", lambda *args: writer)
    camera = module.RecordingCameraRig(FakeRig())
    try:
        with pytest.raises(RuntimeError, match="cannot open video writer"):
            camera.start_recording(tmp_path / "episode")
        assert writer.released
        assert not camera._writers
    finally:
        camera.close()

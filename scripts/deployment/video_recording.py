"""Continuous camera acquisition and per-episode RGB video recording."""
from __future__ import annotations

from pathlib import Path
import threading

import cv2


class RecordingCameraRig:
    """Own camera reads in one thread, sharing fresh frames with the policy."""

    def __init__(self, rig, *, timeout_ms: int = 1000, fps: int = 30):
        self.rig = rig
        self.timeout_ms = timeout_ms
        self.fps = fps
        self._condition = threading.Condition()
        self._stopping = threading.Event()
        self._frames = None
        self._sequence = self._consumed = 0
        self._error = None
        self._writers = {}
        self._thread = threading.Thread(target=self._capture, name="deployment-video", daemon=True)
        self._thread.start()

    def __getattr__(self, name):
        return getattr(self.rig, name)

    def _capture(self):
        try:
            while not self._stopping.is_set():
                frames = (self.rig.read_rgbd(self.timeout_ms) if self.rig.enable_depth
                          else self.rig.read(self.timeout_ms))
                with self._condition:
                    for index, writer in self._writers.items():
                        rgb = frames[index][0] if self.rig.enable_depth else frames[index]
                        if rgb is None:
                            raise RuntimeError(f"recording camera at index {index} has no RGB frame")
                        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    self._frames = frames
                    self._sequence += 1
                    self._condition.notify_all()
        except Exception as exc:
            with self._condition:
                self._error = exc
                self._condition.notify_all()

    def _read(self, timeout_ms):
        with self._condition:
            ready = self._condition.wait_for(
                lambda: self._error is not None or self._sequence > self._consumed
                or self._stopping.is_set(), timeout=timeout_ms / 1000,
            )
            if self._error is not None:
                raise RuntimeError("background camera acquisition/recording failed") from self._error
            if self._stopping.is_set():
                raise RuntimeError("camera rig is closed")
            if not ready:
                raise TimeoutError("timed out waiting for a fresh camera frame")
            self._consumed = self._sequence
            return self._frames

    def read(self, timeout_ms=1000):
        frames = self._read(timeout_ms)
        if self.rig.enable_depth:
            return tuple(None if frame is None else frame[0] for frame in frames)
        return frames

    def read_rgbd(self, timeout_ms=1000):
        if not self.rig.enable_depth:
            raise RuntimeError("camera rig was not opened with depth enabled")
        return self._read(timeout_ms)

    def start_recording(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=False)
        with self._condition:
            if self._writers:
                raise RuntimeError("an episode is already being recorded")
            try:
                for index, view, filename in ((0, "sideview", "cam1.mp4"),
                                               (2, "frontview", "cam3.mp4")):
                    intrinsic = self.rig.color_intrinsics[self.rig.view_names.index(view)]
                    path = directory / filename
                    writer = cv2.VideoWriter(
                        str(path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps,
                        (intrinsic["width"], intrinsic["height"]),
                    )
                    self._writers[index] = writer
                    if not writer.isOpened():
                        raise RuntimeError(f"cannot open video writer: {path}")
            except Exception:
                self._release_writers()
                raise
        print(f"[video] Recording cam1 and cam3 at {self.fps} FPS: {directory}", flush=True)

    def _release_writers(self):
        for writer in self._writers.values():
            writer.release()
        self._writers.clear()

    def stop_recording(self):
        with self._condition:
            self._release_writers()

    def close(self):
        self._stopping.set()
        self._thread.join()
        try:
            self.stop_recording()
        finally:
            self.rig.close()

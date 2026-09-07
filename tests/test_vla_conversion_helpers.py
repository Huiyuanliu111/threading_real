from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[2] / "convert_vla_to_lerobot_v3.py"
SPEC = importlib.util.spec_from_file_location("convert_vla_to_lerobot_v3", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
conversion = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(conversion)


def test_default_conversion_uses_two_fixed_cameras() -> None:
    assert conversion.DEFAULT_CAMERAS == (
        ("cam1.mp4", "observation.images.exterior_image_2_right"),
        ("cam3.mp4", "observation.images.exterior_image_1_left"),
    )


def test_read_follower_matrix(tmp_path: Path) -> None:
    matrix = np.arange(58, dtype=np.float64).reshape(2, 29)
    path = tmp_path / "DATA_follower.m"
    rows = "\n".join(" ".join(map(str, row)) for row in matrix)
    path.write_text(f"DATA_followerm=[{rows}];\n")

    np.testing.assert_array_equal(conversion.read_follower_matrix(path), matrix)


def test_read_timestamped_follower_matrix_and_column_offset(tmp_path: Path) -> None:
    matrix = np.arange(60, dtype=np.float64).reshape(2, 30)
    matrix[:, 0] = [8_000_000_000_000_001, 8_000_000_000_999_999]
    path = tmp_path / "DATA_follower.m"
    rows = "\n".join(" ".join(format(value, ".17g") for value in row) for row in matrix)
    path.write_text(f"DATA_followerm=[{rows}];\n")

    loaded = conversion.read_follower_matrix(path)
    np.testing.assert_array_equal(loaded, matrix)
    assert conversion.follower_column_offset(loaded) == 1


def test_longest_active_span() -> None:
    matrix = np.zeros((9, 29), dtype=np.float64)
    matrix[[1, 3, 4, 5, 7], 0] = 1.0

    assert conversion.longest_active_span(matrix) == (3, 6)


def test_resample_indices_include_both_ends() -> None:
    indices = conversion.resample_indices(1001, 31)

    assert len(indices) == 31
    assert indices[0] == 0
    assert indices[-1] == 1000
    assert np.all(np.diff(indices) >= 0)


def test_timestamp_alignment_matches_nearest_and_rejects_large_skew() -> None:
    robot = np.arange(0, 101_000_000, 1_000_000, dtype=np.int64)
    cameras = [
        np.array([10_000_000, 40_000_000, 70_000_000], dtype=np.int64),
        np.array([12_000_000, 43_000_000, 90_000_000], dtype=np.int64),
    ]
    camera_rows, robot_rows, camera_errors, robot_errors = conversion.timestamp_alignment(
        robot, cameras, max_camera_skew_ms=5.0, max_robot_skew_ms=2.0
    )

    np.testing.assert_array_equal(camera_rows, [[0, 0], [1, 1]])
    np.testing.assert_array_equal(robot_rows, [10, 40])
    np.testing.assert_array_equal(camera_errors[:, 1], [2_000_000, 3_000_000])
    np.testing.assert_array_equal(robot_errors, [0, 0])


def test_camera_timestamp_csv_ignores_uncommitted_video_tail(tmp_path: Path) -> None:
    path = tmp_path / "cam1_timestamps.csv"
    path.write_text(
        "frame_index,color_frame_number,depth_frame_number,"
        "color_sensor_timestamp_ms,depth_sensor_timestamp_ms,"
        "host_steady_timestamp_ns\n"
        "0,10,10,1,1,100\n"
        "1,11,11,2,2,200\n"
        "2,12,12,3,3,300\n"
    )
    np.testing.assert_array_equal(
        conversion.read_camera_timestamp_csv(path, video_frames=2), [100, 200]
    )

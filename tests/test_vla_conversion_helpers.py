from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[2] / "convert_vla_to_lerobot_v3.py"
SPEC = importlib.util.spec_from_file_location("convert_vla_to_lerobot_v3", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
conversion = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(conversion)


def test_read_follower_matrix(tmp_path: Path) -> None:
    matrix = np.arange(58, dtype=np.float64).reshape(2, 29)
    path = tmp_path / "DATA_follower.m"
    rows = "\n".join(" ".join(map(str, row)) for row in matrix)
    path.write_text(f"DATA_followerm=[{rows}];\n")

    np.testing.assert_array_equal(conversion.read_follower_matrix(path), matrix)


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

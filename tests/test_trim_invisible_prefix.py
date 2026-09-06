from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[2] / "trim_invisible_prefix.py"
SPEC = importlib.util.spec_from_file_location("trim_invisible_prefix", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
trim = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trim)


def test_first_persistent_hit_never_returns_tolerated_miss() -> None:
    scores = [0, 0, 500, 0, 500, 500, 500, 500, 500, 500, 500, 500]

    assert trim.first_persistent_hit(scores) == 2


def test_robot_start_row_uses_normalized_episode_progress() -> None:
    assert trim.robot_start_row(30, 301, 10_001) == 1000


def test_entry_score_ignores_dark_shadow_and_detects_bright_arm() -> None:
    background = np.full((240, 320), 100, dtype=np.uint8)
    shadow = background.copy()
    shadow[0:80, 120:200] = 50
    arm = background.copy()
    revealed_table = background.copy()
    revealed_table[30:80, 120:200] = 200
    arm[0:30, 145:175] = 200

    assert trim.entry_score(shadow, background) == 0
    assert trim.entry_score(revealed_table, background) == 0
    assert trim.entry_score(arm, background) >= trim.MINIMUM_COMPONENT_AREA

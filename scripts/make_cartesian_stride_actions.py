#!/usr/bin/env python3
"""Convert per-frame Cartesian deltas into future-K-frame cumulative deltas.

The output retains the same images and observations. At row t, action is the
base-frame TCP displacement from t to t+K, reducing one-frame control noise.
Deploy it at source_fps/K with --execute-steps 1.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent))
from convert_lerobot_v3_to_cartesian import (  # noqa: E402
    _episode_action_stats, _fixed_float_list, _fixed_list_to_numpy, _jsonable,
    _rewrite_episode_stats, _update_huggingface_metadata,
)


def cumulative_actions(
    actions: np.ndarray,
    episode_ids: np.ndarray,
    stride: int,
    smooth_window: int = 0,
    smooth_polyorder: int = 2,
) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float64); out = np.zeros_like(actions)
    for episode in np.unique(episode_ids):
        rows = np.flatnonzero(episode_ids == episode)
        # LeRobot rows are ordered, but make that invariant explicit.
        values = actions[rows]
        # Reconstruct the relative TCP path. Translation increments are in the
        # base frame, so their cumulative sum is a trajectory in one frame.
        positions = np.concatenate((np.zeros((1, 3)), np.cumsum(values[:, :3], axis=0)))
        if smooth_window:
            if smooth_window > len(values) + 1:
                raise ValueError(
                    f"smooth window {smooth_window} exceeds episode {episode} length {len(values)}"
                )
            filtered = savgol_filter(
                positions, window_length=smooth_window,
                polyorder=smooth_polyorder, axis=0, mode="interp"
            )
            # Anchor the reconstructed trajectory at its recorded start/end.
            filtered[0] = positions[0]
            filtered[-1] = positions[-1]
            positions = filtered
        for t in range(len(rows)):
            window = values[t:min(t + stride, len(rows))]
            end = min(t + stride, len(rows))
            out[rows[t], :3] = positions[end] - positions[t]
            rotation = np.eye(3)
            for delta in window:
                rotation = Rotation.from_rotvec(delta[3:6]).as_matrix() @ rotation
            out[rows[t], 3:6] = Rotation.from_matrix(rotation).as_rotvec()
            out[rows[t], 6] = window[:, 6].sum()
    return out.astype(np.float32)


def convert(source: Path, output: Path, stride: int, smooth_window: int, smooth_polyorder: int) -> dict:
    source, output = source.resolve(), output.resolve()
    building = output.with_name(output.name + ".building")
    if stride < 2: raise ValueError("stride must be at least 2")
    if smooth_window and (smooth_window < 3 or smooth_window % 2 == 0):
        raise ValueError("smooth-window must be 0 or an odd integer >= 3")
    if smooth_window and not 0 <= smooth_polyorder < smooth_window:
        raise ValueError("smooth-polyorder must be non-negative and smaller than smooth-window")
    if output.exists() or building.exists(): raise FileExistsError(f"refusing to overwrite {output} or {building}")
    info = json.loads((source / "meta" / "info.json").read_text())
    if info["features"]["action"]["shape"] != [7]: raise ValueError("source action must be 7D Cartesian delta")
    shutil.copytree(source, building, copy_function=shutil.copy2)
    try:
        all_actions, all_episodes = [], []
        for path in sorted((building / "data").rglob("*.parquet")):
            table = pq.read_table(path)
            actions = _fixed_list_to_numpy(table.column("action"))
            episodes = np.asarray(table.column("episode_index"), dtype=np.int64)
            result = cumulative_actions(actions, episodes, stride, smooth_window, smooth_polyorder)
            table = table.set_column(table.schema.get_field_index("action"), "action", _fixed_float_list(result))
            pq.write_table(_update_huggingface_metadata(table, result), path, compression="zstd")
            all_actions.append(result); all_episodes.append(episodes)
        actions, episodes = np.concatenate(all_actions), np.concatenate(all_episodes)
        episode_stats = _episode_action_stats(actions, episodes)
        _rewrite_episode_stats(building / "meta" / "episodes" / "chunk-000" / "file-000.parquet", episode_stats)
        from lerobot.datasets.compute_stats import aggregate_stats
        stats_path = building / "meta" / "stats.json"; stats = json.loads(stats_path.read_text())
        stats["action"] = _jsonable(aggregate_stats(episode_stats)["action"]); stats_path.write_text(json.dumps(stats, indent=2) + "\n")
        report = {"source": str(source), "output": str(output), "stride_frames": stride, "translation_filter": {"method": "savgol" if smooth_window else "none", "window_frames": smooth_window, "polyorder": smooth_polyorder if smooth_window else None}, "source_fps": float(info["fps"]), "recommended_policy_hz": float(info["fps"]) / stride, "total_frames": int(len(actions)), "max_translation_m": float(np.linalg.norm(actions[:, :3], axis=1).max())}
        (building / "meta" / "cartesian_stride_report.json").write_text(json.dumps(report, indent=2) + "\n")
        os.replace(building, output); return report
    except BaseException:
        print(f"partial output remains at {building}"); raise


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("source", type=Path); p.add_argument("output", type=Path); p.add_argument("--stride", type=int, default=5); p.add_argument("--smooth-window", type=int, default=0); p.add_argument("--smooth-polyorder", type=int, default=2)
    args = p.parse_args()
    print(json.dumps(convert(args.source, args.output, args.stride, args.smooth_window, args.smooth_polyorder), indent=2)); return 0

if __name__ == "__main__": raise SystemExit(main())

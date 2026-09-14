#!/usr/bin/env python3
"""Fit a spatial rule on training episodes and label both MVT dataset splits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from chunk_selector.mvt_data import read_trajectories
from chunk_selector.spatial_rule import SpatialRule, trajectory_progress


def create_labels(dataset, output, *, task, urdf=None, h=3, split_progress=None,
                  progress_mode="arc_length", transition_width_m=.02, val_ratio=.2, seed=42):
    dataset, output = Path(dataset).expanduser().resolve(), Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must lie strictly between zero and one")
    trajectories = read_trajectories(dataset, task=task, urdf=urdf)
    keys = sorted(trajectories)
    if len(keys) < 2:
        raise ValueError("at least two episodes are required for train/validation splits")
    n_val = min(max(1, round(len(keys) * val_ratio)), len(keys) - 1)
    val_keys = set(np.random.default_rng(seed).permutation(keys)[:n_val])
    train_keys = [key for key in keys if key not in val_keys]
    rule = SpatialRule.fit([trajectories[key] for key in train_keys], task=task, h=h,
                          split_progress=split_progress, progress_mode=progress_mode,
                          transition_width_m=transition_width_m)
    tables, nodes = [], []
    offset = 0
    for episode_index, key in enumerate(keys):
        xyz = trajectories[key]
        progress = trajectory_progress(xyz, progress_mode)
        probabilities, expected, distances = rule.label(xyz)
        node = int(np.argmin(abs(progress - rule.split_progress)))
        # A spatial sphere can be crossed multiple times; record every crossing.
        crossings = (np.flatnonzero(np.diff(distances <= rule.boundary_radius_m)) + 1).tolist()
        nodes.append(dict(episode_key=key, frame_index=node,
                          progress=float(progress[node]), tcp_xyz_m=xyz[node].tolist(),
                          boundary_crossing_frames=crossings,
                          fine_frame_fraction=float((probabilities[:, 0] >= .5).mean())))
        tables.append(pa.table({
            "index": np.arange(offset, offset + len(xyz)),
            "episode_index": np.full(len(xyz), episode_index),
            "episode_key": [key] * len(xyz), "frame_index": np.arange(len(xyz)),
            "split": ["validation" if key in val_keys else "train"] * len(xyz),
            "trajectory_progress": progress,
            "tcp_x_m": xyz[:, 0], "tcp_y_m": xyz[:, 1], "tcp_z_m": xyz[:, 2],
            "center_distance_m": distances,
            "chunk_size": np.where(probabilities[:, 0] >= .5, h, 10),
            "soft_chunk_size": expected,
            "execution_steps": np.floor(expected + .5).astype(np.int64),
            f"chunk_probability_{h}": probabilities[:, 0],
            "chunk_probability_10": probabilities[:, 1],
        }))
        offset += len(xyz)
    summary = dict(label_source="spatial_rule", rule=rule.to_dict(),
                   source_dataset=str(dataset), frame="panda_link0", tcp_frame="panda_hand_tcp",
                   urdf=str(urdf.expanduser().resolve()) if urdf is not None else None,
                   candidate_chunks=[h, 10], fit_episode_ids=train_keys,
                   validation_episode_ids=sorted(val_keys), seed=seed, val_ratio=val_ratio,
                   nodes=nodes, total_frames=offset)
    output.mkdir(parents=True)
    pq.write_table(pa.concat_tables(tables), output / "labels.parquet", compression="zstd")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=("threading", "maze"), required=True)
    parser.add_argument("--h", type=int, choices=(3, 4, 5), default=3)
    parser.add_argument("--split-progress", type=float, help="default: threading .8, maze .3")
    parser.add_argument("--progress-mode", choices=("arc_length", "frames"), default="arc_length")
    parser.add_argument("--transition-width-m", type=float, default=.02,
                        help="full width of linear probability transition around fitted boundary")
    parser.add_argument("--urdf", type=Path,
                        default=ROOT.parent / "remote_controller/src/remote_controller/assets/panda/panda_arm.urdf")
    parser.add_argument("--val-ratio", type=float, default=.2)
    parser.add_argument("--seed", type=int, default=42)
    args = vars(parser.parse_args())
    summary = create_labels(**args)
    print(json.dumps({"frames": summary["total_frames"], "rule": summary["rule"]}, indent=2))


if __name__ == "__main__":
    main()

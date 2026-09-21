#!/usr/bin/env python3
"""Run in the existing LeRobot v3 environment; export portable v2.1 image parquet."""

import argparse
import copy
import io
import json
from pathlib import Path
import shutil

import datasets
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from transforms import FRONT, SIDE, parse_image, validate_info


def statistics(values, image=False):
    values = np.asarray(values)
    if image:
        values = values.astype(np.float32) / 255
        axes, keepdims = (0, 1, 2), False
    else:
        axes, keepdims = 0, False
        if values.ndim == 1:
            values = values[:, None]
    result = {"min": values.min(axis=axes, keepdims=keepdims),
              "max": values.max(axis=axes, keepdims=keepdims),
              "mean": values.mean(axis=axes, keepdims=keepdims),
              "std": values.std(axis=axes, keepdims=keepdims)}
    if image:
        result = {key: value[:, None, None] for key, value in result.items()}
    return {**{key: value.tolist() for key, value in result.items()}, "count": [len(values)]}


def write_episode_parquet(rows, hf_features, path):
    table = datasets.Dataset.from_list(rows, features=datasets.Features(hf_features)).data.table
    metadata = dict(table.schema.metadata or {})
    hf_metadata = json.loads(metadata[b"huggingface"])
    # datasets >= 4 serializes Sequence as List, which the pinned v2 reader's
    # datasets 3.x cannot deserialize. The Arrow storage is identical here.
    for feature in hf_metadata["info"]["features"].values():
        if feature.get("_type") == "List":
            feature["_type"] = "Sequence"
    metadata[b"huggingface"] = json.dumps(hf_metadata).encode()
    pq.write_table(table.replace_schema_metadata(metadata), path)


def export(source, output, repo_id, max_episodes=None):
    source, output = source.resolve(), output.resolve()
    info = json.loads((source / "meta/info.json").read_text())
    validate_info(info)
    if max_episodes is not None and max_episodes < 1:
        raise ValueError("max_episodes must be positive")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    src = LeRobotDataset(repo_id=repo_id, root=source, video_backend="pyav")
    # Embedded PNGs preserve decoded pixels, avoid another lossy encode, and make
    # the dataset relocatable without absolute image paths or video offset rules.
    features = copy.deepcopy(info["features"])
    hf_features = {}
    for key, feature in features.items():
        if key in (FRONT, SIDE):
            feature["dtype"] = "image"
            feature.pop("info", None)
            hf_features[key] = datasets.Image()
        elif feature["shape"] == [1]:
            hf_features[key] = datasets.Value(feature["dtype"])
        else:
            hf_features[key] = datasets.Sequence(datasets.Value(feature["dtype"]), length=feature["shape"][0])
    output.mkdir(parents=True)
    meta = output / "meta"
    meta.mkdir()
    episodes, episode_stats, tasks = [], [], {}
    count = 0
    total = min(info["total_episodes"], max_episodes or info["total_episodes"])
    for ep in range(total):
        ep_meta = src.meta.episodes[ep]
        rows, values, episode_tasks = [], {key: [] for key in features}, set()
        for idx in range(int(ep_meta["dataset_from_index"]), int(ep_meta["dataset_to_index"])):
            sample = src[idx]
            row = {}
            task_index = int(sample["task_index"])
            tasks[task_index] = str(sample["task"])
            episode_tasks.add(tasks[task_index])
            for key, feature in features.items():
                value = sample[key].numpy() if hasattr(sample[key], "numpy") else np.asarray(sample[key])
                if key in (FRONT, SIDE):
                    value = parse_image(value)
                    buffer = io.BytesIO()
                    Image.fromarray(value).save(buffer, format="PNG")
                    row[key] = {"bytes": buffer.getvalue(), "path": None}
                else:
                    if key == "index":
                        value = np.asarray(count + len(rows))
                    row[key] = value.item() if feature["shape"] == [1] else value.tolist()
                values[key].append(value)
            rows.append(row)
        if not rows:
            raise ValueError(f"Empty episode {ep}")
        path = output / f"data/chunk-{ep // 1000:03d}/episode_{ep:06d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_episode_parquet(rows, hf_features, path)
        episodes.append({"episode_index": ep, "tasks": sorted(episode_tasks), "length": len(rows)})
        episode_stats.append({"episode_index": ep, "stats": {
            key: statistics(value, key in (FRONT, SIDE)) for key, value in values.items()
        }})
        count += len(rows)
        print(f"Exported episode {ep + 1}/{total}: {len(rows)} frames", flush=True)
    new_info = {
        "codebase_version": "v2.1", "robot_type": info["robot_type"], "fps": info["fps"],
        "total_episodes": total, "total_frames": count, "total_tasks": len(tasks),
        "total_videos": 0, "total_chunks": (total + 999) // 1000, "chunks_size": 1000,
        "splits": {"train": f"0:{total}"}, "features": features,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": None,
    }
    for name, records in [("episodes", episodes), ("episodes_stats", episode_stats),
                          ("tasks", [{"task_index": key, "task": value} for key, value in sorted(tasks.items())])]:
        (meta / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    for name in ("visual_preprocessing.json", "pi05_preparation_report.json", "state_representation.json"):
        if (source / "meta" / name).exists():
            shutil.copy2(source / "meta" / name, meta / name)
    (meta / "openpi_export.json").write_text(json.dumps({"source": str(source), "repo_id": repo_id,
        "frames": count, "episodes": total, "image_encoding": "lossless PNG embedded in parquet"}, indent=2))
    # A failed export has no info.json and cannot be mistaken for a complete dataset.
    (meta / "info.json").write_text(json.dumps(new_info, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo-id", default="threading_real/threading_tcp6_nosmooth_30hz")
    parser.add_argument("--max-episodes", type=int, help="Only for export/loader smoke tests")
    args = parser.parse_args()
    export(args.source, args.output, args.repo_id, args.max_episodes)

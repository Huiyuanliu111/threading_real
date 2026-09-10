#!/usr/bin/env python3
"""Convert an existing Cartesian-action pi0.5 dataset to a TCP state representation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from convert_lerobot_v3_to_cartesian import (  # noqa: E402
    _fixed_float_list,
    _fixed_list_to_numpy,
    _jsonable,
)
from threading_real.pi05.prepare_dataset import (  # noqa: E402
    STATE_FEATURES,
    UrdfForwardKinematics,
    convert_state,
)


def _copy_dataset_file(source: str, destination: str) -> str:
    """Hard-link immutable videos and copy mutable metadata/data files."""
    source_path = Path(source)
    if "videos" in source_path.parts:
        os.link(source, destination)
        return destination
    return shutil.copy2(source, destination)


def _update_huggingface_metadata(table: pa.Table, states: np.ndarray) -> pa.Table:
    metadata = dict(table.schema.metadata or {})
    if b"huggingface" in metadata:
        huggingface = json.loads(metadata[b"huggingface"].decode("utf-8"))
        state_feature = huggingface["info"]["features"]["observation.state"]
        state_feature["length"] = int(states.shape[1])
        huggingface["fingerprint"] = hashlib.sha256(states.tobytes()).hexdigest()[:16]
        metadata[b"huggingface"] = json.dumps(huggingface, separators=(",", ":")).encode()
    return table.replace_schema_metadata(metadata)


def _rewrite_episode_state_stats(path: Path, episode_stats: list[dict]) -> None:
    table = pq.read_table(path)
    if len(table) != len(episode_stats):
        raise ValueError("episode metadata count does not match data episode count")
    for stat_name in episode_stats[0]["observation.state"]:
        column_name = f"stats/observation.state/{stat_name}"
        column_index = table.schema.get_field_index(column_name)
        if column_index < 0:
            raise KeyError(f"missing episode metadata column {column_name!r}")
        field = table.schema.field(column_index)
        values = [
            _jsonable(stats["observation.state"][stat_name]) for stats in episode_stats
        ]
        table = table.set_column(column_index, field, pa.array(values, type=field.type))
    pq.write_table(table, path, compression="zstd")


def convert(
    source: Path,
    output: Path,
    representation: str,
    urdf: Path,
    repo_id: str | None = None,
) -> dict:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    building = output.with_name(output.name + ".building")
    if representation not in STATE_FEATURES or representation == "joint":
        raise ValueError("target state representation must be tcp_pose or tcp_pose_6d")
    if output.exists() or building.exists():
        raise FileExistsError(f"refusing to overwrite {output} or temporary {building}")
    info_path = source / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"not a LeRobot dataset: {source}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    source_feature = info["features"]["observation.state"]
    if source_feature.get("shape") != [8] or source_feature.get("names") != STATE_FEATURES["joint"]["names"]:
        raise ValueError("source observation.state must be [q1..q7, gripper_width]")

    fk = UrdfForwardKinematics(urdf.expanduser().resolve())
    state_spec = STATE_FEATURES[representation]
    shutil.copytree(source, building, copy_function=_copy_dataset_file)
    try:
        all_states: list[np.ndarray] = []
        all_episodes: list[np.ndarray] = []
        for path in sorted((building / "data").rglob("*.parquet")):
            table = pq.read_table(path)
            joint_states = _fixed_list_to_numpy(table.column("observation.state"))
            states = np.stack(
                [convert_state(state, representation, fk) for state in joint_states]
            )
            episodes = np.asarray(table.column("episode_index"), dtype=np.int64)
            column_index = table.schema.get_field_index("observation.state")
            table = table.set_column(
                column_index, "observation.state", _fixed_float_list(states)
            )
            pq.write_table(_update_huggingface_metadata(table, states), path, compression="zstd")
            all_states.append(states)
            all_episodes.append(episodes)

        states = np.concatenate(all_states)
        episodes = np.concatenate(all_episodes)
        feature = {
            "observation.state": {
                "dtype": "float32",
                "shape": state_spec["shape"],
                "names": state_spec["names"],
            }
        }
        from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats

        episode_stats = [
            compute_episode_stats(
                {"observation.state": states[episodes == episode]}, feature
            )
            for episode in np.unique(episodes)
        ]
        _rewrite_episode_state_stats(
            building / "meta/episodes/chunk-000/file-000.parquet", episode_stats
        )
        stats_path = building / "meta/stats.json"
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        stats["observation.state"] = _jsonable(
            aggregate_stats(episode_stats)["observation.state"]
        )
        stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

        output_info_path = building / "meta/info.json"
        info["features"]["observation.state"] = {
            "dtype": "float32",
            "shape": list(state_spec["shape"]),
            "names": state_spec["names"],
        }
        output_info_path.write_text(
            json.dumps(info, indent=4, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        preparation_path = building / "meta/pi05_preparation_report.json"
        if preparation_path.is_file():
            preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
            preparation.update({
                "source": str(source),
                "output": str(output),
                "repo_id": repo_id or preparation.get("repo_id"),
                "state_representation": representation,
                "state_names": state_spec["names"],
                "state_dim": int(state_spec["shape"][0]),
                "state_fk_urdf": str(urdf.expanduser().resolve()),
            })
            preparation_path.write_text(
                json.dumps(preparation, indent=2) + "\n", encoding="utf-8"
            )
        combined_path = building / "meta/combined_pipeline_report.json"
        if combined_path.is_file():
            combined = json.loads(combined_path.read_text(encoding="utf-8"))
            combined.update({
                "output": str(output),
                "repo_id": repo_id or combined.get("repo_id"),
                "state_representation": representation,
                "urdf": str(urdf.expanduser().resolve()),
            })
            combined_path.write_text(
                json.dumps(combined, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        report = {
            "source": str(source),
            "output": str(output),
            "repo_id": repo_id,
            "state_representation": representation,
            "state_names": state_spec["names"],
            "state_dim": int(state_spec["shape"][0]),
            "fk_urdf": str(urdf.expanduser().resolve()),
            "frames": int(len(states)),
            "episodes": int(len(np.unique(episodes))),
            "videos_hardlinked": True,
        }
        (building / "meta/pi05_state_conversion_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(building, output)
        return report
    except BaseException:
        print(f"partial output remains at {building}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--state-representation",
        choices=("tcp_pose", "tcp_pose_6d"),
        default="tcp_pose_6d",
    )
    parser.add_argument("--repo-id")
    parser.add_argument(
        "--urdf",
        type=Path,
        default=REPO_ROOT / "remote_controller/src/remote_controller/assets/panda/panda_arm.urdf",
    )
    args = parser.parse_args()
    print(json.dumps(convert(
        args.source, args.output, args.state_representation, args.urdf, args.repo_id
    ), indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render an auxiliary Threading camera with the official robosuite 1.4 stack."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

import mimicgen  # noqa: F401 - registers the MimicGen robosuite environments
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils


def sorted_demo_keys(data: h5py.Group) -> list[str]:
    return sorted(data.keys(), key=lambda key: int(key.rsplit("_", 1)[-1]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--camera-name", default="threading_closeup")
    parser.add_argument("--dataset-key", default=None)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    source = args.dataset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_file():
        parser.error(f"dataset does not exist: {source}")
    if output.exists() and not args.resume:
        parser.error(f"output already exists; pass --resume: {output}")
    dataset_key = args.dataset_key or f"{args.camera_name}_image"
    output.parent.mkdir(parents=True, exist_ok=True)

    ObsUtils.initialize_obs_utils_with_obs_specs(
        {"obs": {"low_dim": [], "rgb": [dataset_key]}}
    )
    env_meta = FileUtils.get_env_metadata_from_dataset(str(source))
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=[args.camera_name],
        camera_height=args.size,
        camera_width=args.size,
        reward_shaping=False,
    )

    mode = "a" if output.exists() else "w"
    with h5py.File(source, "r") as src, h5py.File(output, mode) as dst:
        dst_data = dst.require_group("data")
        dst_data.attrs["source_dataset"] = str(source)
        dst_data.attrs["env_args"] = json.dumps(env_meta)
        dst_data.attrs["camera_name"] = args.camera_name
        dst_data.attrs["dataset_key"] = dataset_key
        dst_data.attrs["camera_size"] = args.size

        demos = sorted_demo_keys(src["data"])
        for index, key in enumerate(demos, 1):
            source_demo = src["data"][key]
            states = source_demo["states"]
            if key in dst_data:
                existing = dst_data[key]
                images = existing.get(f"obs/{dataset_key}")
                if (
                    images is not None
                    and len(images) == len(states)
                    and bool(existing.attrs.get("complete", False))
                ):
                    print(f"[{index}/{len(demos)}] {key}: already complete")
                    continue
                del dst_data[key]

            destination = dst_data.create_group(key)
            destination.attrs["complete"] = False
            images = destination.create_dataset(
                f"obs/{dataset_key}",
                shape=(len(states), args.size, args.size, 3),
                dtype=np.uint8,
                chunks=(1, args.size, args.size, 3),
                compression="gzip",
                compression_opts=1,
            )
            env.reset()
            model = source_demo.attrs.get("model_file")
            for step, state in enumerate(states):
                reset_state = {"states": np.asarray(state)}
                if step == 0 and model is not None:
                    reset_state["model"] = model
                obs = env.reset_to(reset_state)
                images[step] = np.asarray(obs[dataset_key], dtype=np.uint8)
            destination.attrs["num_samples"] = len(states)
            destination.attrs["complete"] = True
            dst.flush()
            print(f"[{index}/{len(demos)}] {key}: {len(states)} frames", flush=True)

    print(f"saved auxiliary camera dataset: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

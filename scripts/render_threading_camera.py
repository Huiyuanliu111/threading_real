#!/usr/bin/env python3
"""Render an auxiliary Threading camera from the states in a core HDF5 file."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from threading_task.dataset import sorted_demo_keys
from threading_task.env import create_threading_env


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path, help="official Threading HDF5 with simulator states")
    parser.add_argument("output", type=Path, help="auxiliary camera-only HDF5")
    parser.add_argument("--camera-name", default="threading_closeup")
    parser.add_argument("--dataset-key", default=None)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--use-model-xml",
        action="store_true",
        help="load each demo's original XML (normally incompatible across robosuite versions)",
    )
    args = parser.parse_args()

    source = args.dataset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_file():
        parser.error(f"dataset does not exist: {source}")
    if output.exists() and not args.resume:
        parser.error(f"output already exists; pass --resume to continue: {output}")
    if args.size <= 0:
        parser.error("--size must be positive")
    dataset_key = args.dataset_key or f"{args.camera_name}_image"
    output.parent.mkdir(parents=True, exist_ok=True)

    env, env_meta = create_threading_env(
        source,
        camera_names=(args.camera_name,),
        camera_size=(args.size,),
    )
    mode = "a" if output.exists() else "w"
    try:
        with h5py.File(source, "r") as src, h5py.File(output, mode) as dst:
            dst_data = dst.require_group("data")
            dst_data.attrs["source_dataset"] = str(source)
            dst_data.attrs["env_args"] = json.dumps(env_meta)
            dst_data.attrs["camera_name"] = args.camera_name
            dst_data.attrs["dataset_key"] = dataset_key
            dst_data.attrs["camera_size"] = args.size

            demo_keys = sorted_demo_keys(src["data"])
            selected = demo_keys[args.start_episode :]
            if args.limit is not None:
                selected = selected[: args.limit]
            for selected_index, key in enumerate(selected, start=1):
                source_demo = src["data"][key]
                states = source_demo.get("states")
                if states is None:
                    raise KeyError(f"{source_demo.name}: missing states")
                if key in dst_data:
                    existing = dst_data[key]
                    existing_image = existing.get(f"obs/{dataset_key}")
                    if (
                        existing_image is not None
                        and len(existing_image) == len(states)
                        and bool(existing.attrs.get("complete", False))
                    ):
                        print(f"[{selected_index}/{len(selected)}] {key}: already complete")
                        continue
                    del dst_data[key]

                destination_demo = dst_data.create_group(key)
                destination_demo.attrs["complete"] = False
                obs_group = destination_demo.create_group("obs")
                image_dataset = obs_group.create_dataset(
                    dataset_key,
                    shape=(len(states), args.size, args.size, 3),
                    dtype=np.uint8,
                    chunks=(1, args.size, args.size, 3),
                    compression="gzip",
                    compression_opts=1,
                )
                model = source_demo.attrs.get("model_file")
                env.reset()
                for step, state in enumerate(states):
                    reset_state = {"states": np.asarray(state)}
                    if step == 0 and args.use_model_xml and model is not None:
                        reset_state["model"] = model
                    observation = env.reset_to(reset_state)
                    raw_image = np.asarray(observation[dataset_key])
                    image_dataset[step] = np.flipud(raw_image).astype(np.uint8)
                destination_demo.attrs["num_samples"] = len(states)
                destination_demo.attrs["complete"] = True
                dst.flush()
                print(f"[{selected_index}/{len(selected)}] {key}: {len(states)} frames")
    finally:
        env.close()

    print(f"saved auxiliary camera dataset: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

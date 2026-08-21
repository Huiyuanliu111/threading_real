#!/usr/bin/env python3
"""Render Threading joint and dual-camera observations from simulator states."""
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


OBS_KEYS = (
    "robot0_joint_pos",
    "robot0_gripper_qpos",
    "agentview_image",
    "robot0_eye_in_hand_image",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path, help="raw/state robomimic HDF5")
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--recompute-rewards", action="store_true")
    args = parser.parse_args()
    source = args.dataset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    env, env_meta = create_threading_env(source)
    total = 0
    try:
        with h5py.File(source, "r") as src, h5py.File(output, "w") as dst:
            dst_data = dst.create_group("data")
            demo_keys = sorted_demo_keys(src["data"])
            if args.limit is not None:
                demo_keys = demo_keys[: args.limit]
            for demo_index, key in enumerate(demo_keys):
                source_demo = src["data"][key]
                states = np.asarray(source_demo["states"])
                actions = np.asarray(source_demo["actions"], dtype=np.float32)
                if len(states) != len(actions):
                    raise ValueError(f"{key}: states/actions length mismatch")
                model = source_demo.attrs.get("model_file")
                observations = {obs_key: [] for obs_key in OBS_KEYS}
                inferred_rewards = []
                inferred_dones = []
                env.reset()
                for step, state in enumerate(states):
                    reset_state = {"states": state}
                    if step == 0 and model is not None:
                        reset_state["model"] = model
                    obs = env.reset_to(reset_state)
                    for obs_key in OBS_KEYS:
                        observations[obs_key].append(np.asarray(obs[obs_key]))
                    reward = env.get_reward()
                    inferred_rewards.append(reward)
                    inferred_dones.append(int(reward > 0 or step == len(states) - 1))

                dest_demo = dst_data.create_group(key)
                dest_demo.create_dataset("actions", data=actions)
                dest_demo.create_dataset("states", data=states)
                rewards = (
                    inferred_rewards
                    if args.recompute_rewards or "rewards" not in source_demo
                    else source_demo["rewards"][:]
                )
                dones = source_demo["dones"][:] if "dones" in source_demo else inferred_dones
                dest_demo.create_dataset("rewards", data=np.asarray(rewards))
                dest_demo.create_dataset("dones", data=np.asarray(dones))
                obs_group = dest_demo.create_group("obs")
                for obs_key, values in observations.items():
                    array = np.asarray(values)
                    options = {"compression": "gzip", "compression_opts": 1} if "image" in obs_key else {}
                    obs_group.create_dataset(obs_key, data=array, **options)
                if model is not None:
                    dest_demo.attrs["model_file"] = model
                dest_demo.attrs["num_samples"] = len(actions)
                total += len(actions)
                print(f"[{demo_index + 1}/{len(demo_keys)}] {key}: {len(actions)} frames")

            dst_data.attrs["total"] = total
            dst_data.attrs["env_args"] = json.dumps(env_meta)
            if "mask" in src:
                src.copy("mask", dst)
    finally:
        env.close()

    from scripts.validate_threading_dataset import validate

    report = validate(output)
    if not report["valid"]:
        print("Prepared file failed validation:", report["errors"], file=sys.stderr)
        return 1
    print(f"Prepared {report['episodes']} episodes / {report['total_frames']} frames at {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Visualize a Threading demonstration from stored observations or simulator states."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from threading_task.dataset import sorted_demo_keys
from threading_task.visualization import (
    VideoWriter,
    compose_camera_views,
    compose_views,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--replay-states", action="store_true")
    parser.add_argument(
        "--auxiliary-dataset",
        type=Path,
        help="optional camera-only HDF5 used to compose a three-view video",
    )
    parser.add_argument("--auxiliary-key", default="threading_closeup_image")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()
    if args.replay_states and args.auxiliary_dataset is not None:
        parser.error("--auxiliary-dataset cannot be combined with --replay-states")
    dataset = args.dataset.expanduser()
    output = args.output or Path("threading_replays") / f"episode_{args.episode:03d}.mp4"

    env = None
    with ExitStack() as stack:
        f = stack.enter_context(h5py.File(dataset, "r"))
        writer = stack.enter_context(VideoWriter(output, fps=args.fps))
        auxiliary = (
            stack.enter_context(h5py.File(args.auxiliary_dataset.expanduser(), "r"))
            if args.auxiliary_dataset is not None
            else None
        )
        keys = sorted_demo_keys(f["data"])
        if args.episode < 0 or args.episode >= len(keys):
            parser.error(f"episode index must be in [0, {len(keys) - 1}]")
        demo = f["data"][keys[args.episode]]
        auxiliary_obs = None
        if auxiliary is not None:
            demo_key = keys[args.episode]
            if demo_key not in auxiliary["data"]:
                raise KeyError(f"Auxiliary dataset is missing {demo_key}")
            auxiliary_demo = auxiliary["data"][demo_key]
            if not bool(auxiliary_demo.attrs.get("complete", False)):
                raise ValueError(f"Auxiliary rendering for {demo_key} is incomplete")
            auxiliary_obs = auxiliary_demo["obs"]
            if args.auxiliary_key not in auxiliary_obs:
                raise KeyError(
                    f"Auxiliary dataset is missing observation {args.auxiliary_key!r}"
                )
            if len(auxiliary_obs[args.auxiliary_key]) != len(demo["actions"]):
                raise ValueError("Primary and auxiliary camera trajectory lengths differ")
        rewards = np.asarray(demo.get("rewards", np.zeros(len(demo["actions"]))))
        if args.replay_states:
            from threading_task.env import create_threading_env

            env, _ = create_threading_env(dataset)
            states = np.asarray(demo["states"])
            model = demo.attrs.get("model_file")
            for step, state in enumerate(states):
                initial = {"states": state}
                if step == 0 and model is not None:
                    initial["model"] = model
                obs = env.reset_to(initial)
                writer.append(
                    compose_views(
                        obs["agentview_image"],
                        obs["robot0_eye_in_hand_image"],
                        [f"episode={args.episode} step={step}", f"reward={rewards[step]:.3f}"],
                        args.size,
                    )
                )
        else:
            obs = demo["obs"]
            for step in range(len(demo["actions"])):
                lines = [
                    f"episode={args.episode} step={step}",
                    f"reward={rewards[step]:.3f}",
                ]
                if auxiliary_obs is None:
                    frame = compose_views(
                        obs["agentview_image"][step],
                        obs["robot0_eye_in_hand_image"][step],
                        lines,
                        args.size,
                    )
                else:
                    frame = compose_camera_views(
                        [
                            obs["agentview_image"][step],
                            obs["robot0_eye_in_hand_image"][step],
                            auxiliary_obs[args.auxiliary_key][step],
                        ],
                        ["agentview", "robot0_eye_in_hand", "threading_closeup"],
                        lines,
                        args.size,
                    )
                writer.append(frame)
    if env is not None:
        env.close()
    print(f"saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

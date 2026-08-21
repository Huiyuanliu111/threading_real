#!/usr/bin/env python3
"""Validate a robomimic Threading HDF5 before ARP training."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from threading_task.dataset import ACTION_DIM, sorted_demo_keys


def validate(path: Path, check_env: bool = False, replay_actions: int = 0) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    lengths: list[int] = []
    required = (
        "robot0_joint_pos",
        "robot0_gripper_qpos",
        "agentview_image",
        "robot0_eye_in_hand_image",
    )
    with h5py.File(path, "r") as f:
        if "data" not in f:
            raise KeyError("missing /data group")
        env_args = f["data"].attrs.get("env_args")
        if env_args is None:
            errors.append("/data is missing env_args")
            env_meta = {}
        else:
            if isinstance(env_args, bytes):
                env_args = env_args.decode()
            env_meta = json.loads(env_args)
            if "Threading" not in str(env_meta.get("env_name")):
                errors.append(f"env_name is not Threading: {env_meta.get('env_name')!r}")

        demos = sorted_demo_keys(f["data"])
        if not demos:
            errors.append("dataset contains no demos")
        for demo_key in demos:
            demo = f["data"][demo_key]
            if "actions" not in demo or "obs" not in demo:
                errors.append(f"{demo_key}: missing actions or obs")
                continue
            actions = demo["actions"]
            n = len(actions)
            lengths.append(n)
            if actions.shape[1:] != (ACTION_DIM,):
                errors.append(f"{demo_key}: action shape {actions.shape}")
            obs = demo["obs"]
            missing = [key for key in required if key not in obs]
            if missing:
                errors.append(f"{demo_key}: missing {missing}")
                continue
            expected = {
                "robot0_joint_pos": (7,),
                "robot0_gripper_qpos": (2,),
            }
            for key in required:
                arr = obs[key]
                if len(arr) != n:
                    errors.append(f"{demo_key}/{key}: length {len(arr)} != {n}")
                if key in expected and arr.shape[1:] != expected[key]:
                    errors.append(f"{demo_key}/{key}: shape {arr.shape}")
                sample = np.asarray(arr[:: max(1, n // 20)])
                if not np.isfinite(sample).all():
                    errors.append(f"{demo_key}/{key}: NaN or Inf")
            if "rewards" in demo and np.max(demo["rewards"][:], initial=0) <= 0:
                warnings.append(f"{demo_key}: no positive reward found")

    replay_report = None
    if (check_env or replay_actions > 0) and not errors:
        from threading_task.env import create_threading_env, success_from_env

        env, _ = create_threading_env(path)
        try:
            obs = env.reset()
            for key in required:
                if key not in obs:
                    errors.append(f"live environment missing {key}")

            if replay_actions > 0 and not errors:
                results = []
                with h5py.File(path, "r") as f:
                    replay_keys = sorted_demo_keys(f["data"])[:replay_actions]
                    for demo_key in replay_keys:
                        demo = f["data"][demo_key]
                        if "states" not in demo:
                            errors.append(f"{demo_key}: action replay requires states")
                            continue
                        env.reset()
                        initial_state = np.asarray(demo["states"][0])
                        try:
                            env.reset_to({"states": initial_state})
                        except Exception as exc:
                            errors.append(f"{demo_key}: could not restore initial state: {exc}")
                            continue

                        success = False
                        success_step = None
                        for step, action in enumerate(demo["actions"], start=1):
                            env.step(np.asarray(action))
                            if success_from_env(env):
                                success = True
                                success_step = step
                                break
                        results.append(
                            {
                                "demo": demo_key,
                                "success": success,
                                "success_step": success_step,
                                "trajectory_steps": len(demo["actions"]),
                            }
                        )
                        if not success:
                            errors.append(
                                f"{demo_key}: expert actions failed in the live environment"
                            )
                replay_report = {
                    "requested": int(replay_actions),
                    "attempted": len(results),
                    "successes": sum(item["success"] for item in results),
                    "episodes": results,
                }
        finally:
            env.close()

    return {
        "dataset": str(path),
        "env_name": env_meta.get("env_name") if env_meta else None,
        "episodes": len(lengths),
        "total_frames": int(sum(lengths)),
        "episode_length_min": min(lengths, default=0),
        "episode_length_max": max(lengths, default=0),
        "episode_length_mean": float(np.mean(lengths)) if lengths else 0.0,
        "errors": errors,
        "warnings": warnings,
        "action_replay": replay_report,
        "valid": not errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument(
        "--replay-actions",
        type=int,
        default=0,
        metavar="N",
        help="replay expert actions for the first N demos in the live environment",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate(args.dataset.expanduser(), args.check_env, args.replay_actions)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

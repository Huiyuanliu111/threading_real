#!/usr/bin/env python3
"""Paired PushBox diagnostic for the causal value of short-chunk replanning.

Both environments receive identical actions through a shared prefix.  Once the
box reaches the requested Y band, one branch keeps executing the tail of the
old full plan while the other replans every ``small_chunk`` steps.  The two
branches receive the same number of post-prefix control-step opportunities.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from envs.pushbox_env import V2_FIXED_OBSTACLES
from scripts.eval_policy import load_policy
from scripts.eval_pushbox_spatial_chunks import (
    FULL_TASK_BOX_INIT_X_RANGE,
    POLICY_IMG_SIZE,
    _episode_obstacle_configs,
    _right_triangular_sampler,
    _seed_everything,
)
from scripts.eval_subtasks import capture_view, get_agent_state, get_box_pos


OBS_KEYS = ("top45", "sideview", "agent_pos", "box_pos")


@dataclass
class ObservationHistory:
    values: dict[str, list[np.ndarray]]

    @classmethod
    def empty(cls) -> "ObservationHistory":
        return cls({key: [] for key in OBS_KEYS})

    def append(self, env, obs: dict) -> None:
        current = {
            "top45": capture_view(env, "top45", POLICY_IMG_SIZE).astype(np.float32)
            / 255.0,
            "sideview": capture_view(env, "sideview", POLICY_IMG_SIZE).astype(np.float32)
            / 255.0,
            "agent_pos": get_agent_state(obs),
            "box_pos": get_box_pos(obs),
        }
        for key, value in current.items():
            if not self.values[key]:
                self.values[key] = [value, value]
            else:
                self.values[key] = (self.values[key] + [value])[-2:]

    def policy_input(self, device: str) -> dict[str, torch.Tensor]:
        return {
            "top45": torch.from_numpy(
                np.stack(self.values["top45"]).transpose(0, 3, 1, 2)
            )
            .unsqueeze(0)
            .to(device),
            "sideview": torch.from_numpy(
                np.stack(self.values["sideview"]).transpose(0, 3, 1, 2)
            )
            .unsqueeze(0)
            .to(device),
            "agent_pos": torch.from_numpy(np.stack(self.values["agent_pos"]))
            .unsqueeze(0)
            .to(device),
            "box_pos": torch.from_numpy(np.stack(self.values["box_pos"]))
            .unsqueeze(0)
            .to(device),
        }


def _make_env(max_steps: int, seed: int, obstacle_configs: list[dict] | None):
    from envs import PushBoxEnv

    return PushBoxEnv(
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        use_object_obs=True,
        reward_shaping=True,
        horizon=max_steps,
        hard_reset=False,
        camera_names="top45,sideview",
        camera_heights=256,
        camera_widths=256,
        obstacle_configs=(
            V2_FIXED_OBSTACLES if obstacle_configs is None else obstacle_configs
        ),
        box_init_x_range=FULL_TASK_BOX_INIT_X_RANGE,
        box_x_sampler=_right_triangular_sampler,
        seed=seed,
    )


def _predict(policy, history: ObservationHistory, device: str, chunk: int):
    policy.inference_chunk_size = int(chunk)
    start = time.perf_counter()
    with torch.no_grad():
        result = policy.predict_action(history.policy_input(device))
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    actions = result["action"][0].detach().cpu().numpy()
    expected_actions = (
        policy.execution_steps_for_chunk_label(int(chunk))
        if hasattr(policy, "execution_steps_for_chunk_label")
        else int(chunk)
    )
    if len(actions) != expected_actions:
        raise RuntimeError(
            f"requested chunk label {chunk} (execution={expected_actions}), "
            f"policy returned {len(actions)}"
        )
    return actions, elapsed


def _sim_state(env) -> np.ndarray:
    return np.asarray(env.sim.get_state().flatten(), dtype=np.float64)


def _assert_synchronised(env_a, env_b, tolerance: float, context: str) -> float:
    state_a = _sim_state(env_a)
    state_b = _sim_state(env_b)
    if state_a.shape != state_b.shape:
        raise RuntimeError(
            f"{context}: simulator state shapes differ: {state_a.shape} != {state_b.shape}"
        )
    maximum = float(np.max(np.abs(state_a - state_b)))
    if maximum > tolerance:
        raise RuntimeError(
            f"{context}: paired environments diverged before branching; "
            f"max state error={maximum:.3e} > {tolerance:.3e}"
        )
    python_fields = (
        "timestep",
        "done",
        "current_phase",
        "_consecutive_arm_collision",
        "_consecutive_oob",
    )
    for field in python_fields:
        if getattr(env_a, field) != getattr(env_b, field):
            raise RuntimeError(
                f"{context}: environment field {field!r} differs: "
                f"{getattr(env_a, field)!r} != {getattr(env_b, field)!r}"
            )
    return maximum


def _step_branch(env, actions: np.ndarray, remaining: int) -> tuple[dict, int, dict]:
    obs = None
    last_info: dict = {}
    raw_contact_steps = 0
    executed = 0
    for action in actions[:remaining]:
        obs, _reward, done, last_info = env.step(action)
        executed += 1
        raw_contact_steps += int(env._check_arm_collision())
        if done:
            break
    if obs is None:
        raise RuntimeError("branch was asked to execute zero actions")
    return obs, executed, {
        "done": bool(done),
        "info": last_info,
        "raw_contact_steps": raw_contact_steps,
    }


def _branch_summary(
    env,
    *,
    start_box: np.ndarray,
    executed: int,
    policy_calls: int,
    inference_time: float,
    raw_contact_steps: int,
    last_info: dict,
) -> dict:
    end_box = np.asarray(env.sim.data.body_xpos[env.box_body_id], dtype=float).copy()
    reason = str(last_info.get("phase_fail_reason", ""))
    return {
        "collision": bool(last_info.get("arm_collision", False)),
        "safe": not bool(last_info.get("arm_collision", False)),
        "done": bool(env.done),
        "failure_reason": reason,
        "executed_steps": int(executed),
        "policy_calls": int(policy_calls),
        "inference_time": float(inference_time),
        "raw_contact_steps": int(raw_contact_steps),
        "start_box_xy": [float(x) for x in start_box[:2]],
        "end_box_xy": [float(x) for x in end_box[:2]],
        "delta_box_y": float(end_box[1] - start_box[1]),
        "abs_end_box_x": float(abs(end_box[0])),
        "phase_success": [
            bool(x) for x in last_info.get("phase_success", env.phase_success)
        ],
    }


def _run_large_tail(env, tail: np.ndarray, start_box: np.ndarray) -> dict:
    obs, executed, step_result = _step_branch(env, tail, len(tail))
    del obs
    return _branch_summary(
        env,
        start_box=start_box,
        executed=executed,
        policy_calls=0,
        inference_time=0.0,
        raw_contact_steps=step_result["raw_contact_steps"],
        last_info=step_result["info"],
    )


def _run_small_replanning(
    env,
    obs: dict,
    history: ObservationHistory,
    policy,
    device: str,
    *,
    small_chunk: int,
    budget: int,
    start_box: np.ndarray,
    reference_tail: np.ndarray,
) -> dict:
    executed = 0
    policy_calls = 0
    inference_time = 0.0
    raw_contact_steps = 0
    last_info: dict = {}
    first_replan_action_l2 = None
    first_replan_prefix_l2_mean = None
    while executed < budget and not env.done:
        history.append(env, obs)
        actions, elapsed = _predict(policy, history, device, small_chunk)
        if policy_calls == 0:
            compared = min(len(actions), len(reference_tail), budget)
            differences = np.linalg.norm(
                actions[:compared] - reference_tail[:compared], axis=-1
            )
            first_replan_action_l2 = float(differences[0])
            first_replan_prefix_l2_mean = float(np.mean(differences))
        policy_calls += 1
        inference_time += elapsed
        obs, used, step_result = _step_branch(env, actions, budget - executed)
        executed += used
        raw_contact_steps += step_result["raw_contact_steps"]
        last_info = step_result["info"]
    result = _branch_summary(
        env,
        start_box=start_box,
        executed=executed,
        policy_calls=policy_calls,
        inference_time=inference_time,
        raw_contact_steps=raw_contact_steps,
        last_info=last_info,
    )
    result["first_replan_action_l2"] = first_replan_action_l2
    result["first_replan_prefix_l2_mean"] = first_replan_prefix_l2_mean
    return result


def run_paired_episode(
    policy,
    device: str,
    *,
    episode_seed: int,
    max_steps: int,
    obstacle_configs: list[dict] | None,
    collection_chunk: int,
    small_chunk: int,
    full_chunk: int,
    branch_y_min: float,
    branch_y_max: float,
    sync_tolerance: float,
) -> dict:
    envs = []
    try:
        for _ in range(2):
            _seed_everything(episode_seed)
            env = _make_env(max_steps, episode_seed, obstacle_configs)
            _seed_everything(episode_seed)
            obs, _grip_steps, _ = env.init_with_grip()
            envs.append((env, obs))
        large_env, large_obs = envs[0]
        small_env, small_obs = envs[1]
        max_sync_error = _assert_synchronised(
            large_env, small_env, sync_tolerance, "after initialization"
        )
        history = ObservationHistory.empty()
        collection_steps = 0
        collection_calls = 0
        collection_inference_time = 0.0
        last_info: dict = {}

        while collection_steps < max_steps:
            history.append(large_env, large_obs)
            box = np.asarray(large_env.sim.data.body_xpos[large_env.box_body_id]).copy()
            in_band = branch_y_min <= float(box[1]) <= branch_y_max
            requested = full_chunk if in_band else collection_chunk
            actions, elapsed = _predict(policy, history, device, requested)
            collection_calls += 1
            collection_inference_time += elapsed

            if in_band:
                branch_box = box.copy()
                common_actions = actions[:small_chunk]
                large_obs, large_used, large_common = _step_branch(
                    large_env, common_actions, small_chunk
                )
                small_obs, small_used, small_common = _step_branch(
                    small_env, common_actions, small_chunk
                )
                if large_used != small_used:
                    raise RuntimeError("paired environments executed different prefix lengths")
                max_sync_error = max(
                    max_sync_error,
                    _assert_synchronised(
                        large_env,
                        small_env,
                        sync_tolerance,
                        "after common prefix",
                    ),
                )
                common_info = large_common["info"]
                common_collision = bool(common_info.get("arm_collision", False))
                common_terminal = bool(large_common["done"])
                base = {
                    "seed": episode_seed,
                    "status": (
                        "common_prefix_failure"
                        if common_collision
                        else "common_prefix_terminal"
                        if common_terminal
                        else "eligible"
                    ),
                    "collection_steps": collection_steps,
                    "collection_calls": collection_calls,
                    "collection_inference_time": collection_inference_time,
                    "branch_box_xy": [float(x) for x in branch_box[:2]],
                    "post_prefix_box_xy": [
                        float(x)
                        for x in large_env.sim.data.body_xpos[large_env.box_body_id][:2]
                    ],
                    "common_prefix_steps": large_used,
                    "common_prefix_raw_contact_steps": large_common[
                        "raw_contact_steps"
                    ],
                    "post_prefix_collision_streak": int(
                        large_env._consecutive_arm_collision
                    ),
                    "common_prefix_failure_reason": str(
                        common_info.get("phase_fail_reason", "")
                    ),
                    "max_sync_error": max_sync_error,
                }
                if common_terminal:
                    return {**base, "large": None, "small": None}

                start_box = np.asarray(
                    large_env.sim.data.body_xpos[large_env.box_body_id], dtype=float
                ).copy()
                tail = actions[small_chunk:full_chunk]
                large = _run_large_tail(large_env, tail, start_box)
                small = _run_small_replanning(
                    small_env,
                    small_obs,
                    history,
                    policy,
                    device,
                    small_chunk=small_chunk,
                    budget=len(tail),
                    start_box=start_box,
                    reference_tail=tail,
                )
                return {**base, "large": large, "small": small}

            for action in actions:
                large_obs, _reward, large_done, large_info = large_env.step(action)
                small_obs, _reward, small_done, small_info = small_env.step(action)
                collection_steps += 1
                last_info = large_info
                max_sync_error = max(
                    max_sync_error,
                    _assert_synchronised(
                        large_env,
                        small_env,
                        sync_tolerance,
                        f"collection step {collection_steps}",
                    ),
                )
                if large_done != small_done:
                    raise RuntimeError("paired environments disagree on collection termination")
                if large_done or collection_steps >= max_steps:
                    return {
                        "seed": episode_seed,
                        "status": "not_reached",
                        "collection_steps": collection_steps,
                        "collection_calls": collection_calls,
                        "collection_inference_time": collection_inference_time,
                        "collection_failure_reason": str(
                            last_info.get("phase_fail_reason", "")
                        ),
                        "max_sync_error": max_sync_error,
                        "large": None,
                        "small": None,
                    }
        raise AssertionError("collection loop exited unexpectedly")
    finally:
        for env, _obs in envs:
            env.close()


def _mean(rows: list[dict], key: str) -> float:
    return float(np.mean([row[key] for row in rows])) if rows else 0.0


def _exact_mcnemar_pvalue(first_only: int, second_only: int) -> float:
    discordant = int(first_only) + int(second_only)
    if discordant == 0:
        return 1.0
    lower = min(int(first_only), int(second_only))
    lower_tail = sum(math.comb(discordant, value) for value in range(lower + 1))
    return min(1.0, 2.0 * lower_tail / (2.0**discordant))


def aggregate(episodes: list[dict]) -> dict:
    statuses: dict[str, int] = {}
    for episode in episodes:
        status = episode["status"]
        statuses[status] = statuses.get(status, 0) + 1
    eligible = [episode for episode in episodes if episode["status"] == "eligible"]
    large = [episode["large"] for episode in eligible]
    small = [episode["small"] for episode in eligible]
    pair_counts = {
        "both_safe": 0,
        "small_only_safe": 0,
        "large_only_safe": 0,
        "both_collision": 0,
    }
    for large_row, small_row in zip(large, small):
        if large_row["safe"] and small_row["safe"]:
            pair_counts["both_safe"] += 1
        elif small_row["safe"]:
            pair_counts["small_only_safe"] += 1
        elif large_row["safe"]:
            pair_counts["large_only_safe"] += 1
        else:
            pair_counts["both_collision"] += 1

    action_divergence = [
        row["first_replan_action_l2"]
        for row in small
        if row["first_replan_action_l2"] is not None
    ]
    prefix_divergence = [
        row["first_replan_prefix_l2_mean"]
        for row in small
        if row["first_replan_prefix_l2_mean"] is not None
    ]

    def branch_metrics(rows: list[dict]) -> dict:
        return {
            "collision_rate": (
                sum(row["collision"] for row in rows) / len(rows) if rows else 0.0
            ),
            "raw_contact_steps_mean": _mean(rows, "raw_contact_steps"),
            "delta_box_y_mean": _mean(rows, "delta_box_y"),
            "abs_end_box_x_mean": _mean(rows, "abs_end_box_x"),
            "executed_steps_mean": _mean(rows, "executed_steps"),
            "policy_calls_mean": _mean(rows, "policy_calls"),
            "inference_time_mean": _mean(rows, "inference_time"),
        }

    return {
        "episodes": len(episodes),
        "status_counts": statuses,
        "eligible_pairs": len(eligible),
        "common_prefix_failure_rate": (
            statuses.get("common_prefix_failure", 0) / len(episodes)
            if episodes
            else 0.0
        ),
        "common_prefix_raw_contact_rate": (
            sum(
                episode.get("common_prefix_raw_contact_steps", 0) > 0
                for episode in episodes
            )
            / len(episodes)
            if episodes
            else 0.0
        ),
        "paired_outcomes": pair_counts,
        "net_small_saves": pair_counts["small_only_safe"]
        - pair_counts["large_only_safe"],
        "mcnemar_exact_pvalue": _exact_mcnemar_pvalue(
            pair_counts["small_only_safe"], pair_counts["large_only_safe"]
        ),
        "large": branch_metrics(large),
        "small": branch_metrics(small),
        "paired_delta_small_minus_large": {
            "raw_contact_steps_mean": (
                float(
                    np.mean(
                        [
                            small_row["raw_contact_steps"]
                            - large_row["raw_contact_steps"]
                            for large_row, small_row in zip(large, small)
                        ]
                    )
                )
                if eligible
                else 0.0
            ),
            "delta_box_y_mean": (
                float(
                    np.mean(
                        [
                            small_row["delta_box_y"] - large_row["delta_box_y"]
                            for large_row, small_row in zip(large, small)
                        ]
                    )
                )
                if eligible
                else 0.0
            ),
        },
        "small_replan_divergence": {
            "first_action_l2_mean": (
                float(np.mean(action_divergence)) if action_divergence else 0.0
            ),
            "first_action_l2_median": (
                float(np.median(action_divergence)) if action_divergence else 0.0
            ),
            "first_action_l2_p90": (
                float(np.quantile(action_divergence, 0.9))
                if action_divergence
                else 0.0
            ),
            "first_prefix_l2_mean": (
                float(np.mean(prefix_divergence)) if prefix_divergence else 0.0
            ),
            "first_prefix_l2_median": (
                float(np.median(prefix_divergence)) if prefix_divergence else 0.0
            ),
            "first_prefix_l2_p90": (
                float(np.quantile(prefix_divergence, 0.9))
                if prefix_divergence
                else 0.0
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Paired common-prefix PushBox replanning diagnostic"
    )
    parser.add_argument("checkpoint")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--small-chunk", type=int, default=5)
    parser.add_argument("--full-chunk", type=int, default=19)
    parser.add_argument("--collection-chunk", type=int, default=5)
    parser.add_argument("--branch-y-min", type=float, default=-0.15)
    parser.add_argument("--branch-y-max", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=30000)
    parser.add_argument(
        "--obstacle-mode",
        choices=("fixed", "light", "medium", "hard"),
        default="hard",
    )
    parser.add_argument("--sync-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.episodes <= 0 or args.max_steps <= 0:
        parser.error("episodes and max-steps must be positive")
    if not 0 < args.small_chunk < args.full_chunk:
        parser.error("requires 0 < small-chunk < full-chunk")
    if args.collection_chunk <= 0:
        parser.error("collection-chunk must be positive")
    if args.branch_y_min > args.branch_y_max:
        parser.error("branch-y-min cannot exceed branch-y-max")
    if args.sync_tolerance < 0:
        parser.error("sync-tolerance cannot be negative")

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault(
        "__EGL_VENDOR_LIBRARY_FILENAMES",
        "/usr/share/glvnd/egl_vendor.d/10_nvidia.json",
    )
    os.environ.pop("DISPLAY", None)

    policy = load_policy(
        args.checkpoint,
        device=args.device,
        weights=args.weights,
        use_checkpoint_config=True,
    )
    maximum = int(
        getattr(
            policy,
            "max_selector_chunk_label",
            getattr(policy, "max_selector_chunk", policy.horizon),
        )
    )
    for name, chunk in (
        ("small", args.small_chunk),
        ("full", args.full_chunk),
        ("collection", args.collection_chunk),
    ):
        if chunk > maximum:
            parser.error(f"{name}-chunk={chunk} exceeds policy maximum={maximum}")
    policy.set_chunk_selector(None)

    episodes = []
    for index in range(args.episodes):
        episode_seed = args.seed + index
        obstacle_configs = _episode_obstacle_configs(args.obstacle_mode, episode_seed)
        print(f"[episode {index + 1}/{args.episodes}] seed={episode_seed}")
        episode = run_paired_episode(
            policy,
            args.device,
            episode_seed=episode_seed,
            max_steps=args.max_steps,
            obstacle_configs=obstacle_configs,
            collection_chunk=args.collection_chunk,
            small_chunk=args.small_chunk,
            full_chunk=args.full_chunk,
            branch_y_min=args.branch_y_min,
            branch_y_max=args.branch_y_max,
            sync_tolerance=args.sync_tolerance,
        )
        episodes.append(episode)
        if episode["status"] == "eligible":
            print(
                "  eligible: "
                f"large_collision={episode['large']['collision']} "
                f"small_collision={episode['small']['collision']} "
                f"dy=({episode['large']['delta_box_y']:.3f}, "
                f"{episode['small']['delta_box_y']:.3f})"
            )
        else:
            print(f"  {episode['status']}")

    payload = {
        "metadata": {
            "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
            "weights": args.weights,
            "seed": args.seed,
            "episode_seeds": [args.seed + index for index in range(args.episodes)],
            "small_chunk": args.small_chunk,
            "full_chunk": args.full_chunk,
            "post_prefix_budget": args.full_chunk - args.small_chunk,
            "collection_chunk": args.collection_chunk,
            "branch_y_band": [args.branch_y_min, args.branch_y_max],
            "obstacle_mode": args.obstacle_mode,
            "box_init_x_range": list(FULL_TASK_BOX_INIT_X_RANGE),
            "box_init_x_distribution": {
                "type": "triangular",
                "low": FULL_TASK_BOX_INIT_X_RANGE[0],
                "high": FULL_TASK_BOX_INIT_X_RANGE[1],
                "mode": FULL_TASK_BOX_INIT_X_RANGE[1],
            },
            "sync_tolerance": args.sync_tolerance,
            "prediction_horizon": int(policy.horizon),
        },
        "aggregate": aggregate(episodes),
        "episodes": episodes,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "branch_eval_stats.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["aggregate"], indent=2))
    print(f"results: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Paired full-task evaluation of fixed and spatial PushBox chunks."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from envs.pushbox_env import (
    OBS_HALF_Z,
    PHASE_Y_APPROACH,
    PHASE_Y_CROSS,
    V2_FIXED_OBSTACLES,
)
from chunk_selector.chunk_selector import ChunkSelector
from chunk_selector.execution import PREDICTION_MODES
from scripts.eval_policy import load_policy
from scripts.eval_subtasks import (
    _light_obstacle_configs,
    capture_view,
    get_agent_state,
    get_box_pos,
    save_video,
)


POLICY_IMG_SIZE = 96
RECORD_IMG_SIZE = 256
FULL_TASK_BOX_INIT_X_RANGE = (-0.01, 0.08)


def _right_triangular_sampler(rng, low: float, high: float) -> float:
    """Sample an increasing triangular density with its mode at ``high``."""
    return float(rng.triangular(low, high, high))


def spatial_chunk_for_box_y(
    box_y: float,
    phase_chunks: tuple[int, int, int],
    early_margin: float = 0.0,
    crossing_threshold: float | None = None,
    crossing_region: tuple[float, float] | None = None,
) -> tuple[int, int]:
    """Return the execution chunk and effective spatial phase.

    ``crossing_region`` supports an asymmetric SMALL-chunk interval. Otherwise
    ``crossing_threshold`` uses the symmetric interval
    ``[-crossing_threshold, crossing_threshold]``. ``early_margin`` remains
    available only to reproduce older evaluations.
    """
    if crossing_region is not None:
        crossing_start, crossing_end = crossing_region
        if crossing_start >= crossing_end:
            raise ValueError("crossing_region requires START < END")
        if box_y < crossing_start:
            phase = 0
        elif box_y <= crossing_end:
            phase = 1
        else:
            phase = 2
        return int(phase_chunks[phase]), phase

    if crossing_threshold is not None:
        if crossing_threshold <= 0:
            raise ValueError("crossing_threshold must be positive")
        if box_y < -crossing_threshold:
            phase = 0
        elif box_y <= crossing_threshold:
            phase = 1
        else:
            phase = 2
        return int(phase_chunks[phase]), phase

    if box_y > PHASE_Y_CROSS:
        phase = 2
    elif box_y > PHASE_Y_APPROACH:
        phase = 1
    else:
        phase = 0

    boundaries = (PHASE_Y_APPROACH, PHASE_Y_CROSS)
    if phase < 2 and phase_chunks[phase + 1] < phase_chunks[phase]:
        if box_y >= boundaries[phase] - early_margin:
            phase += 1
    return int(phase_chunks[phase]), phase


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_random_obstacle_configs(
    rng,
    *,
    y_range: tuple[float, float],
    half_len_range: tuple[float, float],
    asymmetric: bool,
    x_shift_range: tuple[float, float] | None = None,
) -> list[dict]:
    x_shift = 0.0
    if x_shift_range is not None:
        x_shift = float(rng.uniform(*x_shift_range))
    if asymmetric:
        y_offsets = [float(rng.uniform(*y_range)), float(rng.uniform(*y_range))]
        half_lengths = [
            float(rng.uniform(*half_len_range)),
            float(rng.uniform(*half_len_range)),
        ]
    else:
        y_offset = float(rng.uniform(*y_range))
        half_len = float(rng.uniform(*half_len_range))
        y_offsets = [y_offset, y_offset]
        half_lengths = [half_len, half_len]
    return [
        {
            "name": "fixed_obs_0",
            "pos": [0.215 + x_shift, y_offsets[0]],
            "half_size": [half_lengths[0], 0.04, OBS_HALF_Z],
            "euler": [0.0, 0.0, 0.0],
        },
        {
            "name": "fixed_obs_1",
            "pos": [-0.225 + x_shift, y_offsets[1]],
            "half_size": [half_lengths[1], 0.04, OBS_HALF_Z],
            "euler": [0.0, 0.0, 0.0],
        },
    ]


def _episode_obstacle_configs(mode: str, seed: int) -> list[dict] | None:
    rng = np.random.RandomState(seed)
    if mode == "fixed":
        return None
    if mode == "light":
        return _light_obstacle_configs(rng)
    if mode == "medium":
        return _make_random_obstacle_configs(
            rng,
            y_range=(-0.025, 0.025),
            half_len_range=(0.17, 0.18),
            asymmetric=False,
        )
    if mode == "hard":
        return _make_random_obstacle_configs(
            rng,
            y_range=(-0.10, 0.10),
            half_len_range=(0.175, 0.19),
            asymmetric=False,
            x_shift_range=(-0.03, 0.03),
        )
    if mode == "env_random":
        return None
    raise ValueError(f"unknown obstacle mode: {mode}")


def _make_env(
    max_steps: int,
    seed: int,
    *,
    env_random_obstacles: bool = False,
    box_init_x_range: tuple[float, float] | None = FULL_TASK_BOX_INIT_X_RANGE,
    box_init_y_range: tuple[float, float] | None = None,
):
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
        obstacle_configs=None if env_random_obstacles else V2_FIXED_OBSTACLES,
        box_init_x_range=box_init_x_range,
        box_x_sampler=(
            _right_triangular_sampler if box_init_x_range is not None else None
        ),
        box_init_y_range=box_init_y_range,
        seed=seed,
    )


def run_episode(
    env,
    policy,
    device: str,
    *,
    max_steps: int,
    fixed_chunk: int | None,
    phase_chunks: tuple[int, int, int] | None,
    early_margin: float,
    crossing_threshold: float | None,
    crossing_region: tuple[float, float] | None,
    obstacle_configs: list[dict] | None,
    record_video: bool,
    record_selector_trace: bool = False,
) -> dict:
    obs, _grip_steps, _ = env.init_with_grip()
    initial_box_pos = np.asarray(get_box_pos(obs), dtype=float).copy()
    if obstacle_configs is not None:
        env._place_obstacles(obstacle_configs)
    buffers: dict[str, list[np.ndarray]] = {
        "top45": [],
        "sideview": [],
        "agent_pos": [],
        "box_pos": [],
    }
    step = 0
    calls = 0
    inference_time = 0.0
    generated_action_tokens = 0
    chunks: list[int] = []
    prediction_lengths: list[int] = []
    effective_phases: list[int] = []
    frames: list[np.ndarray] = []
    last_info: dict = {}
    selector_trace: list[dict] = []

    wall_start = time.perf_counter()
    while step < max_steps:
        top45 = capture_view(env, "top45", POLICY_IMG_SIZE).astype(np.float32) / 255.0
        sideview = capture_view(env, "sideview", POLICY_IMG_SIZE).astype(np.float32) / 255.0
        values = {
            "top45": top45,
            "sideview": sideview,
            "agent_pos": get_agent_state(obs),
            "box_pos": get_box_pos(obs),
        }
        for key, value in values.items():
            if not buffers[key]:
                buffers[key] = [value, value]
            else:
                buffers[key] = (buffers[key] + [value])[-2:]

        box_y = float(values["box_pos"][1])
        if phase_chunks is not None:
            chunk, effective_phase = spatial_chunk_for_box_y(
                box_y,
                phase_chunks,
                early_margin=early_margin,
                crossing_threshold=crossing_threshold,
                crossing_region=crossing_region,
            )
        elif fixed_chunk is not None:
            chunk = fixed_chunk
            effective_phase = -1
        else:
            if policy.chunk_selector is None:
                raise RuntimeError(
                    "adaptive evaluation requires an attached chunk selector"
                )
            chunk = None
            effective_phase = -2
        if chunk is not None:
            policy.inference_chunk_size = int(chunk)

        policy_obs = {
            "top45": torch.from_numpy(np.stack(buffers["top45"]).transpose(0, 3, 1, 2))
            .unsqueeze(0)
            .to(device),
            "sideview": torch.from_numpy(
                np.stack(buffers["sideview"]).transpose(0, 3, 1, 2)
            )
            .unsqueeze(0)
            .to(device),
            "agent_pos": torch.from_numpy(np.stack(buffers["agent_pos"]))
            .unsqueeze(0)
            .to(device),
            "box_pos": torch.from_numpy(np.stack(buffers["box_pos"]))
            .unsqueeze(0)
            .to(device),
        }
        infer_start = time.perf_counter()
        with torch.no_grad():
            result = policy.predict_action(policy_obs)
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
        inference_time += time.perf_counter() - infer_start
        actions = result["action"][0].detach().cpu().numpy()
        prediction_length = int(result["action_pred"].shape[1])
        generated_action_tokens += prediction_length
        prediction_lengths.append(prediction_length)
        if chunk is None:
            selected = result.get("chunk_size")
            if selected is None:
                raise RuntimeError("selector result did not include chunk_size")
            chunk = int(selected.reshape(-1)[0].item())
        confidence = result.get("chunk_confidence")
        confidence_value = (
            float(confidence.reshape(-1)[0].item())
            if confidence is not None
            else float("nan")
        )
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
        calls += 1
        chunks.append(int(chunk))
        effective_phases.append(effective_phase)

        done = False
        decision_step = step
        for action in actions:
            obs, _reward, done, last_info = env.step(action)
            step += 1
            if record_video:
                top_hd = capture_view(env, "top45", RECORD_IMG_SIZE)
                side_hd = capture_view(env, "sideview", RECORD_IMG_SIZE)
                frames.append(np.concatenate([top_hd, side_hd], axis=1))
            if done or step >= max_steps:
                break
        if record_selector_trace:
            selector_trace.append(
                {
                    "policy_call": calls - 1,
                    "step": decision_step,
                    "box_x": float(values["box_pos"][0]),
                    "box_y": box_y,
                    "selected_chunk": int(chunk),
                    "confidence": confidence_value,
                    "prediction_length": prediction_length,
                    "executed_steps": step - decision_step,
                }
            )
        if done:
            break

    wall_time = time.perf_counter() - wall_start
    success = bool(last_info.get("success", False))
    failure_reason = ""
    if not success:
        failure_reason = str(last_info.get("phase_fail_reason", "")) or (
            "max_steps" if step >= max_steps else "done_without_success"
        )
    return {
        "success": success,
        "steps": step,
        "wall_time": wall_time,
        "inference_time": inference_time,
        "inference_calls": calls,
        "generated_action_tokens": generated_action_tokens,
        "prediction_length_counts": dict(Counter(prediction_lengths)),
        "chunk_counts": dict(Counter(chunks)),
        "effective_phase_counts": dict(Counter(effective_phases)),
        "initial_box_pos": [float(value) for value in initial_box_pos],
        "failure_reason": failure_reason,
        "selector_trace": selector_trace,
        "frames": frames,
    }


def _aggregate(episodes: list[dict]) -> dict:
    successes = [episode for episode in episodes if episode["success"]]

    def mean(items: list[dict], key: str) -> float:
        return float(np.mean([item[key] for item in items])) if items else 0.0

    return {
        "successes": len(successes),
        "episodes": len(episodes),
        "success_rate": len(successes) / len(episodes) if episodes else 0.0,
        "wall_time_mean_all": mean(episodes, "wall_time"),
        "wall_time_mean_success": mean(successes, "wall_time"),
        "steps_mean_all": mean(episodes, "steps"),
        "steps_mean_success": mean(successes, "steps"),
        "inference_calls_mean_all": mean(episodes, "inference_calls"),
        "inference_time_mean_all": mean(episodes, "inference_time"),
        "generated_action_tokens_mean_all": mean(
            episodes, "generated_action_tokens"
        ),
        "generated_action_tokens_per_call": (
            sum(item["generated_action_tokens"] for item in episodes)
            / sum(item["inference_calls"] for item in episodes)
            if sum(item["inference_calls"] for item in episodes)
            else 0.0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument(
        "--prediction-mode",
        choices=PREDICTION_MODES,
        default="full_then_truncate",
        help=(
            "full_then_truncate generates the checkpoint horizon before slicing; "
            "required_only generates only the stale alignment slot plus the "
            "actions that will be executed"
        ),
    )
    parser.add_argument("--full-chunk", type=int, default=None)
    parser.add_argument("--optimal-chunk", type=int, required=True)
    parser.add_argument("--phase-chunks", type=int, nargs=3, required=True)
    parser.add_argument(
        "--chunk-selector",
        type=Path,
        default=None,
        help="optional trained selector sidecar used by the learned_selector method",
    )
    parser.add_argument(
        "--crossing-threshold",
        type=float,
        default=0.15,
        help="symmetric SMALL-chunk region [-T, T] in metres (default: 0.15)",
    )
    parser.add_argument(
        "--crossing-region",
        type=float,
        nargs=2,
        metavar=("START", "END"),
        default=None,
        help=(
            "asymmetric SMALL-chunk region [START, END] in metres; overrides "
            "--crossing-threshold"
        ),
    )
    parser.add_argument(
        "--early-margin",
        type=float,
        default=None,
        help="legacy rule used only to reproduce older evaluations; overrides --crossing-threshold",
    )
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument(
        "--box-init-x-range",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=FULL_TASK_BOX_INIT_X_RANGE,
        help=(
            "full-task box initialization X range; sampled from an increasing "
            "triangular density whose mode is HIGH (default: -0.01 0.08)"
        ),
    )
    parser.add_argument(
        "--box-init-y-range",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=None,
        help="override the full-task box initialization Y range",
    )
    parser.add_argument(
        "--episode-seeds",
        type=int,
        nargs="+",
        default=None,
        help="explicit episode seeds; overrides --episodes and sequential --seed expansion",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("only_full", "only_optimal", "spatial_rule", "learned_selector"),
        default=None,
        help="optional subset of methods, primarily for exact-seed video replay",
    )
    parser.add_argument(
        "--obstacle-mode",
        choices=("fixed", "light", "env_random", "medium", "hard"),
        default="fixed",
        help=(
            "fixed uses V2_FIXED_OBSTACLES; light reuses eval_subtasks; "
            "env_random uses PushBoxEnv random_obstacle_configs; medium/hard "
            "apply wider paired randomisation after grip init"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--save-videos", choices=("none", "all", "failures"), default="none")
    parser.add_argument("--max-videos-per-method", type=int, default=5)
    parser.add_argument(
        "--save-selector-trace",
        action="store_true",
        help="write one CSV row per learned-selector decision",
    )
    args = parser.parse_args()

    if args.episodes <= 0 or args.max_steps <= 0:
        parser.error("episodes and max-steps must be positive")
    if args.crossing_threshold <= 0:
        parser.error("--crossing-threshold must be positive")
    if args.crossing_region is not None:
        if not args.crossing_region[0] < args.crossing_region[1]:
            parser.error("--crossing-region requires START < END")
        if args.early_margin is not None:
            parser.error("--crossing-region cannot be combined with --early-margin")
    if args.early_margin is not None and args.early_margin < 0:
        parser.error("--early-margin must be non-negative")
    for name, bounds in (
        ("box-init-x-range", args.box_init_x_range),
        ("box-init-y-range", args.box_init_y_range),
    ):
        if bounds is not None and not bounds[0] < bounds[1]:
            parser.error(f"--{name} requires LOW < HIGH")

    crossing_region = (
        None
        if args.crossing_region is None
        else (float(args.crossing_region[0]), float(args.crossing_region[1]))
    )
    crossing_threshold = (
        None
        if args.early_margin is not None or crossing_region is not None
        else float(args.crossing_threshold)
    )
    legacy_early_margin = float(args.early_margin or 0.0)

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
    if not hasattr(policy, "set_prediction_mode"):
        parser.error("checkpoint policy does not support explicit prediction modes")
    policy.set_prediction_mode(args.prediction_mode)
    print(f"[eval] prediction_mode={policy.prediction_mode}")
    selector = None
    if args.chunk_selector is not None:
        selector = ChunkSelector.from_pretrained(args.chunk_selector, device=args.device)
        selector.validate_for_policy(
            feature_dim=int(policy.policy.cfg.n_embd),
            max_chunk=int(
                getattr(
                    policy,
                    "max_selector_chunk_label",
                    getattr(policy, "max_selector_chunk", policy.horizon),
                )
            ),
        )
    maximum = int(
        getattr(
            policy,
            "max_selector_chunk_label",
            getattr(policy, "max_selector_chunk", policy.horizon),
        )
    )
    full_chunk = maximum if args.full_chunk is None else args.full_chunk
    candidates = [full_chunk, args.optimal_chunk, *args.phase_chunks]
    if any(chunk < 1 or chunk > maximum for chunk in candidates):
        parser.error(f"all chunks must lie in [1, {maximum}]")
    if selector is not None and selector.candidate_chunks != (
        int(args.optimal_chunk),
        int(full_chunk),
    ):
        parser.error(
            "selector candidates must match (optimal_chunk, full_chunk): "
            f"got {selector.candidate_chunks}, expected "
            f"{(int(args.optimal_chunk), int(full_chunk))}"
        )
    if args.methods is not None and "learned_selector" in args.methods:
        if selector is None:
            parser.error("--methods learned_selector requires --chunk-selector")

    phase_chunks = tuple(int(x) for x in args.phase_chunks)
    methods = {
        "only_full": {"fixed_chunk": full_chunk, "phase_chunks": None},
        "only_optimal": {"fixed_chunk": args.optimal_chunk, "phase_chunks": None},
        "spatial_rule": {"fixed_chunk": None, "phase_chunks": phase_chunks},
    }
    if selector is not None:
        methods["learned_selector"] = {"fixed_chunk": None, "phase_chunks": None}
    if args.methods is not None:
        selected = set(args.methods)
        methods = {name: config for name, config in methods.items() if name in selected}
    if not methods:
        parser.error("no evaluation methods selected")
    results: dict[str, list[dict]] = {name: [] for name in methods}
    selector_trace_rows: list[dict] = []
    videos_kept = Counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    episode_seeds = (
        list(args.episode_seeds)
        if args.episode_seeds is not None
        else [args.seed + index for index in range(args.episodes)]
    )
    for episode_index, episode_seed in enumerate(episode_seeds):
        names = list(methods)
        rotation = episode_index % len(names)
        names = names[rotation:] + names[:rotation]
        print(
            f"[episode {episode_index + 1}/{len(episode_seeds)}] "
            f"seed={episode_seed} order={names}"
        )
        obstacle_configs = _episode_obstacle_configs(args.obstacle_mode, episode_seed)
        for name in names:
            _seed_everything(episode_seed)
            env = _make_env(
                args.max_steps,
                episode_seed,
                env_random_obstacles=args.obstacle_mode == "env_random",
                box_init_x_range=(
                    None
                    if args.box_init_x_range is None
                    else tuple(args.box_init_x_range)
                ),
                box_init_y_range=(
                    None
                    if args.box_init_y_range is None
                    else tuple(args.box_init_y_range)
                ),
            )
            config = methods[name]
            policy.set_chunk_selector(selector if name == "learned_selector" else None)
            record = args.save_videos != "none" and videos_kept[name] < args.max_videos_per_method
            episode = run_episode(
                env,
                policy,
                args.device,
                max_steps=args.max_steps,
                fixed_chunk=config["fixed_chunk"],
                phase_chunks=config["phase_chunks"],
                early_margin=legacy_early_margin,
                crossing_threshold=crossing_threshold,
                crossing_region=crossing_region,
                obstacle_configs=obstacle_configs,
                record_video=record,
                record_selector_trace=(
                    args.save_selector_trace and name == "learned_selector"
                ),
            )
            env.close()
            frames = episode.pop("frames")
            trace = episode.pop("selector_trace")
            keep = record and (
                args.save_videos == "all"
                or (args.save_videos == "failures" and not episode["success"])
            )
            if keep and frames:
                videos_kept[name] += 1
                outcome = "success" if episode["success"] else "failure"
                video_path = args.output_dir / name / f"seed_{episode_seed}_{outcome}.mp4"
                save_video(frames, str(video_path), fps=20)
                episode["video"] = str(video_path)
            else:
                episode["video"] = None
            episode["seed"] = episode_seed
            for row in trace:
                box_y = row["box_y"]
                if crossing_region is not None:
                    if box_y < crossing_region[0]:
                        spatial_region = "before_crossing"
                    elif box_y <= crossing_region[1]:
                        spatial_region = "crossing_precision"
                    else:
                        spatial_region = "after_crossing"
                else:
                    threshold = float(crossing_threshold or 0.15)
                    if box_y < -threshold:
                        spatial_region = "before_crossing"
                    elif box_y <= threshold:
                        spatial_region = "crossing_precision"
                    else:
                        spatial_region = "after_crossing"
                selector_trace_rows.append(
                    {
                        "episode": episode_index,
                        "seed": episode_seed,
                        **row,
                        "episode_steps": episode["steps"],
                        "normalized_progress": (
                            row["step"] / episode["steps"]
                            if episode["steps"]
                            else 0.0
                        ),
                        "budget_progress": row["step"] / args.max_steps,
                        "spatial_region": spatial_region,
                        "success": episode["success"],
                        "failure_reason": episode["failure_reason"],
                    }
                )
            results[name].append(episode)
            print(
                f"  {name}: success={episode['success']} steps={episode['steps']} "
                f"wall={episode['wall_time']:.2f}s calls={episode['inference_calls']}"
            )

    payload = {
        "metadata": {
            "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
            "weights": args.weights,
            "seed": args.seed,
            "episode_seeds": episode_seeds,
            "methods": list(methods),
            "max_steps": args.max_steps,
            "full_chunk": full_chunk,
            "optimal_chunk": args.optimal_chunk,
            "phase_chunks": list(phase_chunks),
            "chunk_selector": (
                None
                if args.chunk_selector is None
                else str(args.chunk_selector.expanduser().resolve())
            ),
            "spatial_rule": (
                {
                    "type": "asymmetric_crossing_region",
                    "crossing_start": crossing_region[0],
                    "crossing_end": crossing_region[1],
                    "small_region": list(crossing_region),
                }
                if crossing_region is not None
                else {
                    "type": "symmetric_crossing_region",
                    "crossing_threshold": crossing_threshold,
                    "small_region": [-crossing_threshold, crossing_threshold],
                }
                if crossing_threshold is not None
                else {
                    "type": "legacy_early_margin",
                    "early_margin": legacy_early_margin,
                }
            ),
            "obstacle_mode": args.obstacle_mode,
            "fixed_obstacles": args.obstacle_mode == "fixed",
            "paired_obstacle_configs": args.obstacle_mode
            in ("light", "medium", "hard"),
            "box_init_x_range": args.box_init_x_range,
            "box_init_x_distribution": {
                "type": "triangular",
                "low": float(args.box_init_x_range[0]),
                "high": float(args.box_init_x_range[1]),
                "mode": float(args.box_init_x_range[1]),
            },
            "box_init_y_range": args.box_init_y_range,
            "prediction_horizon": int(policy.horizon),
            "prediction_mode": args.prediction_mode,
        },
        "aggregate": {name: _aggregate(items) for name, items in results.items()},
        "episodes": results,
    }
    output_path = args.output_dir / "eval_stats.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.save_selector_trace:
        if not selector_trace_rows:
            parser.error(
                "--save-selector-trace requires the learned_selector method"
            )
        trace_path = args.output_dir / "selector_trace.csv"
        with trace_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=list(selector_trace_rows[0]),
            )
            writer.writeheader()
            writer.writerows(selector_trace_rows)
        print(f"selector trace: {trace_path}")
    print(json.dumps(payload["aggregate"], indent=2))
    print(f"results: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Evaluate a PushBox-compatible ARP checkpoint on MimicGen Threading."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from threading_task.env_runner import ThreadingImageRunner
from threading_task.metrics import EpisodeMetrics, write_metrics
from chunk_selector.chunk_selector import ChunkSelector
from chunk_selector.execution import PREDICTION_MODES
from scripts.eval_policy import load_policy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--env-name", default=None)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--state-mode",
        choices=("auto", "joint", "eef"),
        default="auto",
        help="policy state layout; auto infers 8D EEF or 9D joint state",
    )
    parser.add_argument(
        "--camera-size",
        type=int,
        default=None,
        help="policy RGB input size; defaults to the checkpoint policy shape",
    )
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=None,
        help="override how many predicted actions are executed before replanning",
    )
    parser.add_argument(
        "--prediction-mode",
        choices=PREDICTION_MODES,
        default="full_then_truncate",
        help=(
            "full_then_truncate predicts the checkpoint horizon and executes a prefix; "
            "required_only predicts exactly the actions that will be executed"
        ),
    )
    parser.add_argument(
        "--translation-scale",
        type=float,
        default=0.25,
        help="scale OSC translation commands before env.step (default: 0.25)",
    )
    parser.add_argument(
        "--chunk-selector",
        type=Path,
        default=None,
        help="optional adaptive_chunk directory containing a trained selector sidecar",
    )
    parser.add_argument(
        "--spatial-rule-chunks",
        type=int,
        nargs=2,
        metavar=("SMALL", "FULL"),
        default=None,
        help="bypass the learned selector and choose SMALL/FULL from online spatial stages",
    )
    parser.add_argument("--grasp-distance", type=float, default=0.10)
    parser.add_argument("--lift-threshold", type=float, default=0.05)
    parser.add_argument("--insert-approach-distance", type=float, default=0.20)
    parser.add_argument(
        "--gmm-eval-mode",
        choices=("checkpoint", "sample", "mean", "map"),
        default="checkpoint",
        help="GMM inference rule; map selects the highest-probability component",
    )
    parser.add_argument(
        "--weights",
        choices=("ema", "model"),
        default="ema",
        help="checkpoint weights to evaluate (default: training-time EMA policy)",
    )
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--output-dir", type=Path, default=Path("threading_eval"))
    parser.add_argument("--save-videos", choices=("none", "all", "successes", "failures"), default="failures")
    parser.add_argument("--max-videos", type=int, default=20)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument(
        "--save-chunk-trace",
        action="store_true",
        help="record every selector decision for temporal chunk analysis",
    )
    args = parser.parse_args()

    if not 0.0 < args.translation_scale <= 1.0:
        parser.error("--translation-scale must be in (0, 1]")

    selection_modes = sum(
        value is not None
        for value in (args.n_action_steps, args.chunk_selector, args.spatial_rule_chunks)
    )
    if selection_modes > 1:
        parser.error(
            "--n-action-steps, --chunk-selector, and --spatial-rule-chunks "
            "are mutually exclusive"
        )

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
    if args.chunk_selector is not None:
        if not hasattr(policy, "set_chunk_selector"):
            parser.error("checkpoint policy does not support adaptive chunk selection")
        selector = ChunkSelector.from_pretrained(args.chunk_selector, device=args.device)
        policy.set_chunk_selector(selector)
        print(f"[eval] adaptive chunk selector={args.chunk_selector.expanduser().resolve()}")
    if args.n_action_steps is not None:
        if args.n_action_steps <= 0:
            parser.error("--n-action-steps must be positive")
        policy.n_action_steps = int(args.n_action_steps)
        print(f"[eval] override n_action_steps={policy.n_action_steps}")
    spatial_rule_config = None
    if args.spatial_rule_chunks is not None:
        small_chunk, full_chunk = args.spatial_rule_chunks
        if not 0 < small_chunk < full_chunk:
            parser.error("--spatial-rule-chunks requires 0 < SMALL < FULL")
        if full_chunk > int(policy.horizon):
            parser.error(
                f"FULL={full_chunk} exceeds checkpoint horizon={policy.horizon}"
            )
        if min(
            args.grasp_distance,
            args.lift_threshold,
            args.insert_approach_distance,
        ) <= 0:
            parser.error("spatial rule thresholds must be positive")
        spatial_rule_config = {
            "small_chunk": small_chunk,
            "full_chunk": full_chunk,
            "grasp_distance": args.grasp_distance,
            "lift_threshold": args.lift_threshold,
            "insert_approach_distance": args.insert_approach_distance,
        }
        print(f"[eval] deterministic spatial chunk rule={spatial_rule_config}")
    if args.gmm_eval_mode != "checkpoint":
        if args.gmm_eval_mode == "map":
            from threading_task.policy import enable_map_gmm_inference

            enable_map_gmm_inference(policy)
        policy.use_sample = {
            "sample": True,
            "mean": False,
            "map": "map",
        }[args.gmm_eval_mode]
        print(f"[eval] override GMM inference={args.gmm_eval_mode}")
    agent_state_dim = int(getattr(policy, "agent_state_dim", 9))
    if args.state_mode == "auto":
        state_mode = "eef" if agent_state_dim == 8 else "joint"
    else:
        state_mode = args.state_mode
    expected_state_dim = 8 if state_mode == "eef" else 9
    if agent_state_dim != expected_state_dim:
        parser.error(
            f"--state-mode={state_mode} produces {expected_state_dim}D state, "
            f"but checkpoint policy expects {agent_state_dim}D"
        )
    camera_output_keys = tuple(getattr(policy, "rgb_keys", ("top45", "sideview")))
    camera_keys = getattr(policy, "camera_obs_keys", None)
    if camera_keys is None:
        camera_keys = ("agentview_image", "robot0_eye_in_hand_image")
        camera_output_keys = ("top45", "sideview")
    camera_keys = tuple(camera_keys)
    image_shapes = getattr(policy, "image_shapes", None)
    if args.camera_size is not None:
        camera_sizes = (int(args.camera_size),) * len(camera_keys)
    elif image_shapes is not None:
        camera_sizes = tuple(int(image_shapes[key][-1]) for key in camera_output_keys)
    else:
        image_shape = getattr(policy, "image_shape", (3, 96, 96))
        camera_sizes = (int(image_shape[-1]),) * len(camera_keys)
    print(
        f"[eval] observations: state_mode={state_mode}, "
        f"cameras={dict(zip(camera_output_keys, zip(camera_keys, camera_sizes)))}"
    )
    save_candidates = 0 if args.save_videos == "none" else args.episodes
    runner = ThreadingImageRunner(
        output_dir=str(args.output_dir),
        dataset_path=args.dataset,
        env_name=args.env_name,
        n_eval_episodes=args.episodes,
        max_steps=args.max_steps,
        test_start_seed=args.seed,
        n_video_episodes=save_candidates,
        video_fps=args.video_fps,
        state_mode=state_mode,
        camera_keys=camera_keys,
        camera_output_keys=camera_output_keys,
        camera_sizes=camera_sizes,
        translation_scale=args.translation_scale,
        spatial_chunk_rule=spatial_rule_config,
        save_chunk_trace=args.save_chunk_trace,
        chunk_trace_spatial_thresholds={
            "grasp_distance": args.grasp_distance,
            "lift_threshold": args.lift_threshold,
            "insert_approach_distance": args.insert_approach_distance,
        },
    )
    runner.run(policy)

    run_dir = args.output_dir / "threading_rollouts" / "run_0000"
    payload = json.loads((run_dir / "eval_stats.json").read_text())
    episodes = [EpisodeMetrics(**item) for item in payload["episodes"]]
    runner_metadata = dict(payload.get("metadata", {}))
    kept = 0
    filtered: list[EpisodeMetrics] = []
    for ep in episodes:
        keep_kind = (
            args.save_videos == "all"
            or (args.save_videos == "successes" and ep.success)
            or (args.save_videos == "failures" and not ep.success)
        )
        keep = bool(ep.video) and keep_kind and kept < args.max_videos
        if keep:
            kept += 1
            filtered.append(ep)
        else:
            if ep.video:
                Path(ep.video).unlink(missing_ok=True)
            filtered.append(replace(ep, video=None))
    final = write_metrics(
        run_dir,
        filtered,
        metadata={
            **runner_metadata,
            "checkpoint": str(Path(args.checkpoint).expanduser()),
            "weights": args.weights,
            "prediction_mode": args.prediction_mode,
            "execution_mode": (
                "spatial_rule"
                if spatial_rule_config is not None
                else "adaptive_selector"
                if args.chunk_selector is not None
                else "fixed"
            ),
            "n_action_steps": (
                int(policy.n_action_steps)
                if spatial_rule_config is None and args.chunk_selector is None
                else None
            ),
            "translation_scale": args.translation_scale,
            "chunk_selector": (
                None
                if args.chunk_selector is None
                else str(args.chunk_selector.expanduser().resolve())
            ),
            "spatial_chunk_rule": runner_metadata.get("spatial_chunk_rule"),
            "gmm_eval_mode": args.gmm_eval_mode,
            "state_mode": state_mode,
            "camera_keys": list(camera_keys),
            "camera_output_keys": list(camera_output_keys),
            "camera_sizes": list(camera_sizes),
            "dataset": str(Path(args.dataset).expanduser()),
            "env_name": args.env_name,
            "save_videos": args.save_videos,
            "save_chunk_trace": args.save_chunk_trace,
        },
    )
    print(json.dumps(final["aggregate"], indent=2))
    print(f"results: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

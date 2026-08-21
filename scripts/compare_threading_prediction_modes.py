#!/usr/bin/env python3
"""Run and summarize the paired Threading prediction-mode experiment matrix."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chunk_selector.execution import PREDICTION_MODES


@dataclass(frozen=True)
class Experiment:
    prediction_mode: str
    strategy: str
    kind: str
    chunks: tuple[int, ...]

    @property
    def key(self) -> str:
        return f"{self.prediction_mode}/{self.strategy}"


def experiment_matrix() -> list[Experiment]:
    strategies = [
        *((f"fixed_{chunk}", "fixed", (chunk,)) for chunk in (2, 4, 6, 8, 10)),
        ("selector_4_10", "selector", (4, 10)),
        ("spatial_4_10", "spatial", (4, 10)),
    ]
    # Keep the two prediction modes adjacent for every execution strategy.
    return [
        Experiment(mode, name, kind, chunks)
        for name, kind, chunks in strategies
        for mode in PREDICTION_MODES
    ]


def result_file(root: Path, experiment: Experiment) -> Path:
    return (
        root
        / experiment.prediction_mode
        / experiment.strategy
        / "threading_rollouts"
        / "run_0000"
        / "eval_stats.json"
    )


def completed_result(
    root: Path,
    experiment: Experiment,
    episodes: int,
) -> dict | None:
    path = result_file(root, experiment)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    aggregate = payload.get("aggregate", {})
    metadata = payload.get("metadata", {})
    if (
        aggregate.get("num_episodes") == episodes
        and metadata.get("prediction_mode") == experiment.prediction_mode
    ):
        return payload
    return None


def command_for(
    args: argparse.Namespace,
    experiment: Experiment,
    device: str,
) -> list[str]:
    output_dir = args.output_root / experiment.prediction_mode / experiment.strategy
    command = [
        args.python,
        "scripts/eval_threading.py",
        str(args.checkpoint),
        "--dataset",
        str(args.dataset),
        "--env-name",
        args.env_name,
        "--episodes",
        str(args.episodes),
        "--max-steps",
        str(args.max_steps),
        "--device",
        device,
        "--weights",
        args.weights,
        "--prediction-mode",
        experiment.prediction_mode,
        "--translation-scale",
        str(args.translation_scale),
        "--seed",
        str(args.seed),
        "--output-dir",
        str(output_dir),
        "--save-videos",
        "none",
    ]
    if experiment.kind == "fixed":
        command.extend(("--n-action-steps", str(experiment.chunks[0])))
    elif experiment.kind == "selector":
        command.extend(("--chunk-selector", str(args.selector)))
    elif experiment.kind == "spatial":
        command.extend(
            (
                "--spatial-rule-chunks",
                str(experiment.chunks[0]),
                str(experiment.chunks[1]),
                "--grasp-distance",
                str(args.grasp_distance),
                "--lift-threshold",
                str(args.lift_threshold),
                "--insert-approach-distance",
                str(args.insert_approach_distance),
            )
        )
    else:
        raise ValueError(f"unsupported strategy kind {experiment.kind!r}")
    return command


def write_summaries(root: Path, experiments: list[Experiment]) -> None:
    rows: list[dict] = []
    for experiment in experiments:
        path = result_file(root, experiment)
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        aggregate = payload["aggregate"]
        rows.append(
            {
                "prediction_mode": experiment.prediction_mode,
                "strategy": experiment.strategy,
                "num_episodes": aggregate["num_episodes"],
                "successes": aggregate["successes"],
                "success_rate": aggregate["success_rate"],
                "ci95_low": aggregate["success_rate_ci95"][0],
                "ci95_high": aggregate["success_rate_ci95"][1],
                "timeout_rate": aggregate["timeout_rate"],
                "avg_steps_all": aggregate["avg_steps_all"],
                "avg_policy_calls": aggregate["avg_policy_calls"],
                "avg_wall_time_sec_all": aggregate["avg_wall_time_sec_all"],
                "inference_time_mean_ms": aggregate["inference_time_mean_ms"],
                "result_dir": str(path.parent),
            }
        )

    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    if not rows:
        return
    with (root / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    indexed = {(row["prediction_mode"], row["strategy"]): row for row in rows}
    lines = [
        "# Threading prediction-mode comparison",
        "",
        "| Strategy | Full success | Required-only success | Delta | Full calls | Required-only calls | Full inference ms | Required-only inference ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    strategies = list(dict.fromkeys(experiment.strategy for experiment in experiments))
    for strategy in strategies:
        full = indexed.get(("full_then_truncate", strategy))
        required = indexed.get(("required_only", strategy))
        if full is None or required is None:
            continue
        delta = required["success_rate"] - full["success_rate"]
        lines.append(
            f"| {strategy} | {full['successes']}/{full['num_episodes']} "
            f"({full['success_rate']:.1%}) | {required['successes']}/{required['num_episodes']} "
            f"({required['success_rate']:.1%}) | {delta:+.1%} | "
            f"{full['avg_policy_calls']:.2f} | {required['avg_policy_calls']:.2f} | "
            f"{full['inference_time_mean_ms']:.2f} | {required['inference_time_mean_ms']:.2f} |"
        )
    lines.extend(("", "All result directories are recorded in `results.csv`."))
    (root / "comparison.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--selector",
        type=Path,
        default=Path("outputs/chunk_selector/threading_4_10"),
    )
    parser.add_argument("--env-name", default="Threading_D0")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--translation-scale", type=float, default=0.25)
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--grasp-distance", type=float, default=0.10)
    parser.add_argument("--lift-threshold", type=float, default=0.05)
    parser.add_argument("--insert-approach-distance", type=float, default=0.20)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/threading_prediction_mode_comparison_200"),
    )
    parser.add_argument("--devices", nargs="+", default=("cuda:0",))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.episodes <= 0 or args.max_steps <= 0:
        parser.error("--episodes and --max-steps must be positive")
    if len(set(args.devices)) != len(args.devices):
        parser.error("--devices must not contain duplicates")
    if not args.checkpoint.exists():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if not args.dataset.exists():
        parser.error(f"dataset does not exist: {args.dataset}")
    if not args.selector.exists():
        parser.error(f"selector does not exist: {args.selector}")

    experiments = experiment_matrix()
    pending = [
        experiment
        for experiment in experiments
        if args.force
        or completed_result(args.output_root, experiment, args.episodes) is None
    ]
    print(
        f"[matrix] experiments={len(experiments)} pending={len(pending)} "
        f"devices={list(args.devices)}"
    )
    for index, experiment in enumerate(pending):
        device = args.devices[index % len(args.devices)]
        print(f"[matrix] {experiment.key} -> {device}")
        if args.dry_run:
            print(" ".join(command_for(args, experiment, device)))
    if args.dry_run:
        return 0

    args.output_root.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    output_lock = Lock()
    failures: list[tuple[str, int]] = []

    # One worker per device; jobs assigned to a device remain sequential so
    # inference timing is never contaminated by same-GPU concurrency.
    assignments = [pending[index:: len(args.devices)] for index in range(len(args.devices))]

    def device_worker(device: str, jobs: list[Experiment]) -> None:
        env = os.environ.copy()
        env.update({"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"})
        for experiment in jobs:
            command = command_for(args, experiment, device)
            log_path = log_dir / f"{experiment.prediction_mode}__{experiment.strategy}.log"
            with output_lock:
                print(f"[start] {experiment.key} on {device}; log={log_path}", flush=True)
            with log_path.open("w") as log:
                process = subprocess.run(
                    command,
                    cwd=Path(__file__).resolve().parent.parent,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            with output_lock:
                if process.returncode:
                    failures.append((experiment.key, process.returncode))
                    print(f"[failed] {experiment.key}: exit={process.returncode}", flush=True)
                else:
                    print(f"[done] {experiment.key}", flush=True)
                    write_summaries(args.output_root, experiments)

    if pending:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=len(args.devices)) as executor:
            futures = [
                executor.submit(device_worker, device, jobs)
                for device, jobs in zip(args.devices, assignments)
            ]
            for future in futures:
                future.result()

    write_summaries(args.output_root, experiments)
    if failures:
        print(f"[matrix] failures={failures}")
        return 1
    print(f"[matrix] complete; summary={args.output_root / 'comparison.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the ordered 22-experiment PushBox prediction-mode comparison matrix."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PREDICTION_MODES = ("full_then_truncate", "required_only")
FIXED_CHUNKS = (2, 4, 5, 6, 8, 10, 19)
SELECTOR_CHUNKS = (5, 10)
SPATIAL_CHUNKS = (4, 10)


@dataclass(frozen=True)
class Experiment:
    prediction_mode: str
    strategy: str
    chunk: int
    selector: Path | None = None

    @property
    def name(self) -> str:
        return f"{self.strategy}_{self.chunk}"

    @property
    def evaluator_method(self) -> str:
        return {
            "fixed": "only_optimal",
            "selector": "learned_selector",
            "spatial": "spatial_rule",
        }[self.strategy]


def build_experiments(
    selector_5: Path,
    selector_10: Path,
    prediction_modes: tuple[str, ...] = PREDICTION_MODES,
) -> list[Experiment]:
    selectors = {5: selector_5, 10: selector_10}
    experiments: list[Experiment] = []
    for mode in prediction_modes:
        experiments.extend(Experiment(mode, "fixed", chunk) for chunk in FIXED_CHUNKS)
        experiments.extend(
            Experiment(mode, "selector", chunk, selectors[chunk])
            for chunk in SELECTOR_CHUNKS
        )
        experiments.extend(
            Experiment(mode, "spatial", chunk) for chunk in SPATIAL_CHUNKS
        )
    return experiments


def _result_path(output_dir: Path, experiment: Experiment) -> Path:
    return output_dir / experiment.prediction_mode / experiment.name / "eval_stats.json"


def _command(
    args: argparse.Namespace,
    experiment: Experiment,
    evaluator: Path,
) -> list[str]:
    output_dir = args.output_dir / experiment.prediction_mode / experiment.name
    command = [
        args.python,
        str(evaluator),
        str(args.checkpoint),
        "--prediction-mode",
        experiment.prediction_mode,
        "--episodes",
        str(args.episodes),
        "--max-steps",
        str(args.max_steps),
        "--device",
        args.device,
        "--weights",
        args.weights,
        "--full-chunk",
        str(args.full_chunk),
        "--optimal-chunk",
        str(experiment.chunk),
        "--phase-chunks",
        str(args.full_chunk),
        str(experiment.chunk),
        str(args.full_chunk),
        "--methods",
        experiment.evaluator_method,
        "--seed",
        str(args.seed),
        "--obstacle-mode",
        args.obstacle_mode,
        "--output-dir",
        str(output_dir),
        "--save-videos",
        args.save_videos,
        "--max-videos-per-method",
        str(args.max_videos_per_method),
    ]
    if experiment.selector is not None:
        command.extend(("--chunk-selector", str(experiment.selector)))
    if args.early_margin is None:
        command.extend(("--crossing-threshold", str(args.crossing_threshold)))
    else:
        command.extend(("--early-margin", str(args.early_margin)))
    if args.box_init_x_range is not None:
        command.extend(
            ("--box-init-x-range", *(str(value) for value in args.box_init_x_range))
        )
    if args.box_init_y_range is not None:
        command.extend(
            ("--box-init-y-range", *(str(value) for value in args.box_init_y_range))
        )
    return command


def _load_record(output_dir: Path, experiment: Experiment) -> dict | None:
    result_path = _result_path(output_dir, experiment)
    if not result_path.is_file():
        return None
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    aggregate = payload["aggregate"][experiment.evaluator_method]
    episodes = payload["episodes"][experiment.evaluator_method]
    return {
        "prediction_mode": experiment.prediction_mode,
        "strategy": experiment.strategy,
        "chunk": experiment.chunk,
        "result_path": str(result_path.resolve()),
        **aggregate,
        "episode_success": {
            str(item["seed"]): bool(item["success"]) for item in episodes
        },
    }


def write_summary(output_dir: Path, experiments: list[Experiment]) -> dict:
    records = [
        record
        for experiment in experiments
        if (record := _load_record(output_dir, experiment)) is not None
    ]
    by_key = {
        (record["prediction_mode"], record["strategy"], record["chunk"]): record
        for record in records
    }
    comparisons = []
    for strategy in ("fixed", "selector", "spatial"):
        chunks = {
            "fixed": FIXED_CHUNKS,
            "selector": SELECTOR_CHUNKS,
            "spatial": SPATIAL_CHUNKS,
        }[strategy]
        for chunk in chunks:
            full = by_key.get(("full_then_truncate", strategy, chunk))
            required = by_key.get(("required_only", strategy, chunk))
            if full is None or required is None:
                continue
            shared_seeds = sorted(
                set(full["episode_success"]) & set(required["episode_success"]),
                key=int,
            )
            full_only = sum(
                full["episode_success"][seed]
                and not required["episode_success"][seed]
                for seed in shared_seeds
            )
            required_only = sum(
                required["episode_success"][seed]
                and not full["episode_success"][seed]
                for seed in shared_seeds
            )
            comparisons.append(
                {
                    "strategy": strategy,
                    "chunk": chunk,
                    "paired_episodes": len(shared_seeds),
                    "success_rate_delta_required_minus_full": (
                        required["success_rate"] - full["success_rate"]
                    ),
                    "required_only_successes": required_only,
                    "full_only_successes": full_only,
                    "wall_time_ratio_required_over_full": (
                        required["wall_time_mean_all"] / full["wall_time_mean_all"]
                        if full["wall_time_mean_all"]
                        else None
                    ),
                    "inference_time_ratio_required_over_full": (
                        required["inference_time_mean_all"]
                        / full["inference_time_mean_all"]
                        if full["inference_time_mean_all"]
                        else None
                    ),
                    "generated_token_ratio_required_over_full": (
                        required["generated_action_tokens_mean_all"]
                        / full["generated_action_tokens_mean_all"]
                        if full["generated_action_tokens_mean_all"]
                        else None
                    ),
                }
            )

    serializable_records = []
    for record in records:
        record = dict(record)
        record.pop("episode_success")
        serializable_records.append(record)
    payload = {
        "completed": len(records),
        "expected": len(experiments),
        "experiments": serializable_records,
        "paired_mode_comparisons": comparisons,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "matrix_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    csv_path = output_dir / "matrix_summary.csv"
    columns = [
        "prediction_mode",
        "strategy",
        "chunk",
        "successes",
        "episodes",
        "success_rate",
        "wall_time_mean_all",
        "inference_time_mean_all",
        "inference_calls_mean_all",
        "generated_action_tokens_mean_all",
        "generated_action_tokens_per_call",
        "result_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(serializable_records)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--selector-5", type=Path, required=True)
    parser.add_argument("--selector-10", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--full-chunk", type=int, default=19)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument(
        "--prediction-modes",
        nargs="+",
        choices=PREDICTION_MODES,
        default=list(PREDICTION_MODES),
    )
    parser.add_argument(
        "--obstacle-mode",
        choices=("fixed", "light", "env_random", "medium", "hard"),
        default="fixed",
    )
    parser.add_argument("--crossing-threshold", type=float, default=0.15)
    parser.add_argument("--early-margin", type=float, default=None)
    parser.add_argument(
        "--box-init-x-range", type=float, nargs=2, default=(-0.01, 0.08)
    )
    parser.add_argument("--box-init-y-range", type=float, nargs=2, default=None)
    parser.add_argument(
        "--save-videos", choices=("none", "all", "failures"), default="none"
    )
    parser.add_argument("--max-videos-per-method", type=int, default=5)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.episodes <= 0 or args.max_steps <= 0:
        parser.error("--episodes and --max-steps must be positive")
    if args.full_chunk <= max(FIXED_CHUNKS):
        parser.error(f"--full-chunk must be greater than {max(FIXED_CHUNKS)}")
    if args.crossing_threshold <= 0:
        parser.error("--crossing-threshold must be positive")
    if args.early_margin is not None and args.early_margin < 0:
        parser.error("--early-margin must be non-negative")
    for label, path in (("selector-5", args.selector_5), ("selector-10", args.selector_10)):
        if not path.exists():
            parser.error(f"--{label} does not exist: {path}")

    modes = tuple(dict.fromkeys(args.prediction_modes))
    experiments = build_experiments(args.selector_5, args.selector_10, modes)
    evaluator = Path(__file__).with_name("eval_pushbox_spatial_chunks.py")
    if not args.resume:
        existing = [
            _result_path(args.output_dir, experiment)
            for experiment in experiments
            if _result_path(args.output_dir, experiment).exists()
        ]
        if existing:
            parser.error(
                f"{len(existing)} result files already exist; pass --resume to skip them"
            )

    for index, experiment in enumerate(experiments, start=1):
        result_path = _result_path(args.output_dir, experiment)
        label = f"{experiment.prediction_mode}/{experiment.name}"
        if args.resume and result_path.is_file():
            print(f"[{index}/{len(experiments)}] skip completed {label}")
            continue
        command = _command(args, experiment, evaluator)
        print(f"[{index}/{len(experiments)}] run {label}")
        if args.dry_run:
            print("  " + " ".join(command))
            continue
        subprocess.run(command, check=True)
        summary = write_summary(args.output_dir, experiments)
        print(f"  matrix progress: {summary['completed']}/{summary['expected']}")

    if not args.dry_run:
        summary = write_summary(args.output_dir, experiments)
        print(json.dumps({key: summary[key] for key in ("completed", "expected")}, indent=2))
        print(f"summary: {args.output_dir / 'matrix_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

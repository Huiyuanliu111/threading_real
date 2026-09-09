#!/usr/bin/env python3
"""Benchmark required-only against full-then-truncate pi0.5 inference."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from pi05.selector_inference import PI05SelectorInference


MODES = ("required_only", "full_then_truncate")


def _aggregate(records: list[dict]) -> dict[str, float | int]:
    result: dict[str, float | int] = {"samples": len(records)}
    for key in ("vision_seconds", "selector_seconds", "action_seconds", "total_seconds"):
        values = [float(record[key]) for record in records]
        result[f"{key}_mean"] = statistics.fmean(values)
        result[f"{key}_median"] = statistics.median(values)
        result[f"{key}_p95"] = float(np.quantile(values, 0.95))
    result["predicted_actions_mean"] = statistics.fmean(
        int(record["predicted_chunk"]) for record in records
    )
    result["executed_actions_mean"] = statistics.fmean(
        int(record["execution_chunk"]) for record in records
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/threading_combined_pi05_15hz_sg5_nozero"),
    )
    parser.add_argument(
        "--repo-id", default="threading_real/threading_combined_pi05_15hz_sg5_nozero"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.samples <= 0 or args.warmup < 0:
        parser.error("samples must be positive and warmup cannot be negative")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        parser.error(f"refusing to overwrite existing output: {output_dir}")
    output_dir.mkdir(parents=True)

    engine = PI05SelectorInference(
        args.checkpoint,
        args.selector,
        device=args.device,
    )
    dataset = LeRobotDataset(
        args.repo_id,
        root=args.dataset_root.expanduser().resolve(),
        video_backend="pyav",
    )
    indices = np.linspace(0, len(dataset) - 1, args.samples, dtype=np.int64)
    warmup_index = int(indices[0])
    for _ in range(args.warmup):
        batch = engine.prepare(dict(dataset[warmup_index]))
        for mode in MODES:
            engine.predict(batch, mode=mode)

    records: list[dict] = []
    for sample_number, index in enumerate(indices):
        batch = engine.prepare(dict(dataset[int(index)]))
        mode_order = MODES if sample_number % 2 == 0 else tuple(reversed(MODES))
        for mode in mode_order:
            prediction = engine.predict(batch, mode=mode)
            record = {
                "sample": sample_number,
                "dataset_index": int(index),
                "mode": mode,
                "execution_chunk": prediction.execution_chunk,
                "predicted_chunk": prediction.predicted_chunk,
                "continuous_chunk": prediction.continuous_chunk,
                "probability_4": prediction.probabilities[0],
                "probability_10": prediction.probabilities[1],
                "vision_seconds": prediction.vision_seconds,
                "selector_seconds": prediction.selector_seconds,
                "action_seconds": prediction.action_seconds,
                "total_seconds": prediction.total_seconds,
            }
            records.append(record)
        print(f"benchmarked {sample_number + 1}/{len(indices)} observations", flush=True)

    with (output_dir / "timings.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    by_mode = {
        mode: _aggregate([record for record in records if record["mode"] == mode])
        for mode in MODES
    }
    required = by_mode["required_only"]
    full = by_mode["full_then_truncate"]
    summary = {
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "selector": str(args.selector.expanduser().resolve()),
        "dataset_root": str(args.dataset_root.expanduser().resolve()),
        "warmup": args.warmup,
        "modes": by_mode,
        "comparison": {
            "action_time_ratio_required_over_full": (
                required["action_seconds_mean"] / full["action_seconds_mean"]
            ),
            "total_time_ratio_required_over_full": (
                required["total_seconds_mean"] / full["total_seconds_mean"]
            ),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

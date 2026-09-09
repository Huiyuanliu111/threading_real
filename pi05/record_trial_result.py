#!/usr/bin/env python3
"""Attach a manual task outcome to one adaptive pi0.5 timing trace."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import numpy as np


TIMING_KEYS = ("vision_seconds", "selector_seconds", "action_seconds", "total_seconds")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--success", choices=("yes", "no"), required=True)
    parser.add_argument(
        "--episode",
        type=int,
        help="only summarize this episode from a multi-episode trace",
    )
    args = parser.parse_args()

    records = [json.loads(line) for line in args.trace.read_text().splitlines() if line.strip()]
    if args.episode is not None:
        records = [record for record in records if int(record.get("episode", 1)) == args.episode]
    if not records:
        suffix = "" if args.episode is None else f" for episode {args.episode}"
        parser.error(f"trace contains no inference cycles{suffix}: {args.trace}")
    modes = {record.get("prediction_mode") for record in records}
    if len(modes) != 1 or None in modes:
        parser.error(f"trace must contain exactly one adaptive prediction mode, got {modes}")

    timing: dict[str, float] = {}
    for key in TIMING_KEYS:
        values = [float(record[key]) for record in records]
        timing[f"{key}_mean"] = statistics.fmean(values)
        timing[f"{key}_median"] = statistics.median(values)
        timing[f"{key}_p95"] = float(np.quantile(values, 0.95))
    total_mean = timing["total_seconds_mean"]
    result = {
        "mode": modes.pop(),
        "episode": args.episode,
        "success": args.success == "yes",
        "cycles": len(records),
        "execution_chunk_mean": statistics.fmean(
            int(record["execution_chunk"]) for record in records
        ),
        "predicted_chunk_mean": statistics.fmean(
            int(record["predicted_chunk"]) for record in records
        ),
        "inference_hz": 1.0 / total_mean,
        "timing": timing,
        "trace": str(args.trace.expanduser().resolve()),
    }
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error(f"refusing to overwrite existing result: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

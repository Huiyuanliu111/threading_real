#!/usr/bin/env python3
"""Compare success rate and inference timing across recorded pi0.5 trials."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


MODES = ("required_only", "full_then_truncate")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    paths = sorted(args.results_dir.expanduser().resolve().glob("**/trial.json"))
    trials = [json.loads(path.read_text()) for path in paths]
    if not trials:
        parser.error(f"no trial.json files found under {args.results_dir}")

    summary: dict[str, object] = {"trial_count": len(trials), "modes": {}}
    for mode in MODES:
        selected = [trial for trial in trials if trial["mode"] == mode]
        if not selected:
            summary["modes"][mode] = {"trials": 0}
            continue
        successes = sum(bool(trial["success"]) for trial in selected)
        summary["modes"][mode] = {
            "trials": len(selected),
            "successes": successes,
            "success_rate": successes / len(selected),
            "total_seconds_mean": statistics.fmean(
                trial["timing"]["total_seconds_mean"] for trial in selected
            ),
            "action_seconds_mean": statistics.fmean(
                trial["timing"]["action_seconds_mean"] for trial in selected
            ),
            "inference_hz_mean": statistics.fmean(trial["inference_hz"] for trial in selected),
            "execution_chunk_mean": statistics.fmean(
                trial["execution_chunk_mean"] for trial in selected
            ),
            "predicted_chunk_mean": statistics.fmean(
                trial["predicted_chunk_mean"] for trial in selected
            ),
        }
    rendered = json.dumps(summary, indent=2) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

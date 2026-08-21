"""Episode-level metrics and stable JSON / CSV output for Threading."""
from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np


@dataclass
class EpisodeMetrics:
    episode: int
    seed: int
    success: bool
    steps: int
    episode_return: float
    max_reward: float
    timeout: bool
    num_policy_calls: int
    wall_time_sec: float
    inference_time_mean_ms: float
    inference_time_p95_ms: float
    video: str | None = None


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return [max(0.0, center - margin), min(1.0, center + margin)]


def aggregate_episodes(episodes: list[EpisodeMetrics]) -> dict[str, Any]:
    total = len(episodes)
    successes = sum(int(ep.success) for ep in episodes)
    successful_steps = [ep.steps for ep in episodes if ep.success]
    successful_wall_times = [ep.wall_time_sec for ep in episodes if ep.success]
    return {
        "num_episodes": total,
        "successes": successes,
        "success_rate": successes / total if total else 0.0,
        "success_rate_ci95": wilson_interval(successes, total),
        "timeout_rate": sum(int(ep.timeout) for ep in episodes) / total if total else 0.0,
        "avg_steps_all": float(np.mean([ep.steps for ep in episodes])) if total else 0.0,
        "avg_steps_success": float(np.mean(successful_steps)) if successful_steps else None,
        "avg_return": float(np.mean([ep.episode_return for ep in episodes])) if total else 0.0,
        "avg_policy_calls": float(np.mean([ep.num_policy_calls for ep in episodes])) if total else 0.0,
        "avg_wall_time_sec_all": (
            float(np.mean([ep.wall_time_sec for ep in episodes])) if total else 0.0
        ),
        "avg_wall_time_sec_success": (
            float(np.mean(successful_wall_times)) if successful_wall_times else None
        ),
        "inference_time_mean_ms": (
            float(np.mean([ep.inference_time_mean_ms for ep in episodes])) if total else 0.0
        ),
    }


def write_metrics(
    output_dir: str | Path,
    episodes: list[EpisodeMetrics],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    aggregate = aggregate_episodes(episodes)
    payload = {
        "metadata": metadata or {},
        "aggregate": aggregate,
        "episodes": [asdict(ep) for ep in episodes],
    }
    (output / "eval_stats.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    rows = [asdict(ep) for ep in episodes]
    if rows:
        with (output / "episodes.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = "\n".join(f"{key}: {value}" for key, value in aggregate.items()) + "\n"
    (output / "summary.txt").write_text(summary)
    return payload

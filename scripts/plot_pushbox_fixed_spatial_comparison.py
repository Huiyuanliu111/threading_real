#!/usr/bin/env python3
"""Summarize and plot the paired PushBox fixed/spatial chunk comparison."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHODS = (
    ("fixed 5", "fixed_5/eval_stats.json", "only_optimal"),
    (
        "fixed 10",
        "fixed10_vs_spatial_19_5_19_m020_p015/eval_stats.json",
        "only_optimal",
    ),
    ("fixed 15", "fixed_15/eval_stats.json", "only_optimal"),
    ("fixed 19", "fixed_19/eval_stats.json", "only_optimal"),
    (
        "spatial 19/5/19",
        "fixed10_vs_spatial_19_5_19_m020_p015/eval_stats.json",
        "spatial_rule",
    ),
)


def wilson(successes: int, episodes: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = successes / episodes
    denominator = 1 + z * z / episodes
    center = (p + z * z / (2 * episodes)) / denominator
    radius = z * math.sqrt(p * (1 - p) / episodes + z * z / (4 * episodes**2)) / denominator
    return center - radius, center + radius


def exact_mcnemar(a_only: int, b_only: int) -> float:
    discordant = a_only + b_only
    if not discordant:
        return 1.0
    lower = min(a_only, b_only)
    tail = sum(math.comb(discordant, k) for k in range(lower + 1)) / (2**discordant)
    return min(1.0, 2 * tail)


def load_results(root: Path) -> tuple[list[dict], dict, dict[str, list[dict]]]:
    rows = []
    metadata = None
    episodes_by_method = {}
    reference_initialization = None

    for name, relative_path, episode_key in METHODS:
        path = root / relative_path
        data = json.loads(path.read_text())
        current_metadata = data["metadata"]
        episodes = data["episodes"][episode_key]
        initialization = [(e["seed"], e["initial_box_pos"]) for e in episodes]
        if reference_initialization is None:
            reference_initialization = initialization
            metadata = current_metadata
        elif initialization != reference_initialization:
            raise ValueError(f"{name} does not use the same seeds and initial positions")

        aggregate = data["aggregate"][episode_key]
        successes = aggregate["successes"]
        episode_count = aggregate["episodes"]
        ci_low, ci_high = wilson(successes, episode_count)
        failures = Counter(e["failure_reason"] for e in episodes if not e["success"])
        rows.append(
            {
                "method": name,
                "successes": successes,
                "episodes": episode_count,
                "success_rate": aggregate["success_rate"],
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "steps_mean_all": aggregate["steps_mean_all"],
                "policy_calls_mean": aggregate["inference_calls_mean_all"],
                "generated_action_tokens_mean": aggregate["generated_action_tokens_mean_all"],
                "inference_time_mean": aggregate["inference_time_mean_all"],
                "failure_counts": dict(sorted(failures.items())),
                "source": str(path),
            }
        )
        episodes_by_method[name] = episodes

    assert metadata is not None
    retained_metadata = {
        "checkpoint": metadata["checkpoint"],
        "weights": metadata["weights"],
        "prediction_mode": metadata["prediction_mode"],
        "prediction_horizon": metadata["prediction_horizon"],
        "max_steps": metadata["max_steps"],
        "seed_start": reference_initialization[0][0],
        "seed_end": reference_initialization[-1][0],
        "box_init_x_range": metadata["box_init_x_range"],
        "box_init_x_distribution": metadata["box_init_x_distribution"],
        "box_init_y_range": metadata["box_init_y_range"],
        "obstacle_mode": metadata["obstacle_mode"],
        "paired_initial_states_verified": True,
    }
    return rows, retained_metadata, episodes_by_method


def paired_comparisons(episodes_by_method: dict[str, list[dict]]) -> list[dict]:
    names = list(episodes_by_method)
    comparisons = []
    for index, a_name in enumerate(names):
        for b_name in names[index + 1 :]:
            pairs = zip(episodes_by_method[a_name], episodes_by_method[b_name])
            both = a_only = b_only = neither = 0
            for a_episode, b_episode in pairs:
                a_success, b_success = a_episode["success"], b_episode["success"]
                both += a_success and b_success
                a_only += a_success and not b_success
                b_only += b_success and not a_success
                neither += not a_success and not b_success
            comparisons.append(
                {
                    "method_a": a_name,
                    "method_b": b_name,
                    "both_success": both,
                    "a_only_success": a_only,
                    "b_only_success": b_only,
                    "neither_success": neither,
                    "exact_mcnemar_p": exact_mcnemar(a_only, b_only),
                }
            )
    return comparisons


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "method",
        "successes",
        "episodes",
        "success_rate",
        "ci95_low",
        "ci95_high",
        "steps_mean_all",
        "policy_calls_mean",
        "generated_action_tokens_mean",
        "inference_time_mean",
        "failure_counts",
        "source",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["failure_counts"] = json.dumps(row["failure_counts"], sort_keys=True)
            writer.writerow(csv_row)


def write_markdown(path: Path, metadata: dict, rows: list[dict], comparisons: list[dict]) -> None:
    lines = [
        "# PushBox fixed chunk 与 spatial rule 对比（200 episodes）",
        "",
        "## 实验设置",
        "",
        f"- checkpoint：`{metadata['checkpoint']}`；",
        f"- `{metadata['prediction_mode']}`，EMA 权重，最大 {metadata['max_steps']} 环境步；",
        f"- seed {metadata['seed_start']}–{metadata['seed_end']}，各方法共享完全相同的 200 个初始状态；",
        "- Box X：三角分布 `[0.00, 0.05]`，众数 `0.05`；",
        "- Box Y：均匀分布 `[-0.25, -0.20]`；固定障碍物；",
        "- spatial rule：Box Y 在 `[-0.20, +0.15]` 时 chunk 5，否则 chunk 19。",
        "",
        "## 结果",
        "",
        "| 方法 | 成功数 | 成功率（Wilson 95% CI） | 平均环境步 | 平均 policy calls | 平均生成 action tokens | 失败类型 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        failures = ", ".join(f"{key}={value}" for key, value in row["failure_counts"].items()) or "无"
        lines.append(
            f"| {row['method']} | {row['successes']}/{row['episodes']} | "
            f"{row['success_rate']:.1%}（{row['ci95_low']:.1%}–{row['ci95_high']:.1%}） | "
            f"{row['steps_mean_all']:.2f} | {row['policy_calls_mean']:.2f} | "
            f"{row['generated_action_tokens_mean']:.2f} | {failures} |"
        )

    spatial = next(row for row in rows if row["method"] == "spatial 19/5/19")
    lines.extend(["", "## Spatial rule 的配对比较", ""])
    for comparison in comparisons:
        if "spatial 19/5/19" not in (comparison["method_a"], comparison["method_b"]):
            continue
        other = comparison["method_b"] if comparison["method_a"] == "spatial 19/5/19" else comparison["method_a"]
        other_row = next(row for row in rows if row["method"] == other)
        lines.append(
            f"- 相比 {other}，成功率变化为 "
            f"{(spatial['success_rate'] - other_row['success_rate']):+.1%}；"
            f"精确 McNemar 检验 `p={comparison['exact_mcnemar_p']:.4g}`。"
        )
    path.write_text("\n".join(lines) + "\n")


def plot(path: Path, rows: list[dict]) -> None:
    names = [row["method"].replace("spatial ", "spatial\n") for row in rows]
    x = np.arange(len(rows))
    colors = ["#4C78A8"] * 4 + ["#F58518"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    rates = np.array([row["success_rate"] * 100 for row in rows])
    lows = rates - np.array([row["ci95_low"] * 100 for row in rows])
    highs = np.array([row["ci95_high"] * 100 for row in rows]) - rates
    axes[0, 0].bar(x, rates, color=colors, yerr=np.vstack([lows, highs]), capsize=5)
    axes[0, 0].set_ylabel("Success rate (%)")
    axes[0, 0].set_ylim(0, 100)
    axes[0, 0].set_title("Success rate with Wilson 95% CI")
    for index, rate in enumerate(rates):
        axes[0, 0].text(index, rate + highs[index] + 2, f"{rate:.1f}%", ha="center")

    calls = [row["policy_calls_mean"] for row in rows]
    axes[0, 1].bar(x, calls, color=colors)
    axes[0, 1].set_ylabel("Calls / episode")
    axes[0, 1].set_title("Mean policy calls")
    for index, value in enumerate(calls):
        axes[0, 1].text(index, value + 0.7, f"{value:.1f}", ha="center")

    tokens = [row["generated_action_tokens_mean"] for row in rows]
    axes[1, 0].bar(x, tokens, color=colors)
    axes[1, 0].set_ylabel("Tokens / episode")
    axes[1, 0].set_title("Mean generated action tokens")
    for index, value in enumerate(tokens):
        axes[1, 0].text(index, value + 4, f"{value:.1f}", ha="center")

    failure_types = sorted({key for row in rows for key in row["failure_counts"]})
    bottoms = np.zeros(len(rows))
    failure_colors = plt.get_cmap("Set2").colors
    for index, failure_type in enumerate(failure_types):
        counts = np.array([row["failure_counts"].get(failure_type, 0) for row in rows])
        axes[1, 1].bar(x, counts, bottom=bottoms, label=failure_type, color=failure_colors[index])
        bottoms += counts
    axes[1, 1].set_ylabel("Episodes")
    axes[1, 1].set_title("Failure reasons")
    axes[1, 1].legend(frameon=False)

    for axis in axes.flat:
        axis.set_xticks(x, names)
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("PushBox paired evaluation: X triangular [0.00, 0.05], Y [-0.25, -0.20]", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    rows, metadata, episodes_by_method = load_results(args.root)
    comparisons = paired_comparisons(episodes_by_method)
    summary = {"metadata": metadata, "methods": rows, "paired_comparisons": comparisons}
    (args.root / "comparison_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_csv(args.root / "comparison_summary.csv", rows)
    write_markdown(args.root / "comparison_table.md", metadata, rows, comparisons)
    plot(args.root / "comparison.png", rows)
    print(json.dumps({row["method"]: row["success_rate"] for row in rows}, indent=2))


if __name__ == "__main__":
    main()

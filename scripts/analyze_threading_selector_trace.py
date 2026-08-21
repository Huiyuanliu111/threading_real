#!/usr/bin/env python3
"""Summarize and plot selector chunk choices over rollout time."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_rows(path: Path) -> list[dict]:
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in (
            "episode",
            "seed",
            "policy_call",
            "step",
            "selected_chunk",
            "executed_steps",
            "episode_steps",
        ):
            row[key] = int(row[key])
        for key in (
            "confidence",
            "normalized_progress",
            "budget_progress",
            "handle_distance",
            "lift_height",
            "insert_distance",
        ):
            row[key] = float(row[key])
        row["success"] = row["success"].lower() == "true"
        row["timeout"] = row["timeout"].lower() == "true"
    return rows


def _bin_rows(rows: list[dict], key: str, edges: np.ndarray) -> list[dict]:
    result = []
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    bins = np.clip(np.digitize(values, edges, right=False) - 1, 0, len(edges) - 2)
    for index in range(len(edges) - 1):
        selected = [row for row, bin_index in zip(rows, bins) if bin_index == index]
        decisions = len(selected)
        steps = sum(row["executed_steps"] for row in selected)
        chunk10_decisions = sum(row["selected_chunk"] == 10 for row in selected)
        chunk10_steps = sum(
            row["executed_steps"]
            for row in selected
            if row["selected_chunk"] == 10
        )
        result.append(
            {
                "bin_start": float(edges[index]),
                "bin_end": float(edges[index + 1]),
                "bin_center": float((edges[index] + edges[index + 1]) / 2),
                "decisions": decisions,
                "environment_steps": steps,
                "chunk10_decision_share": (
                    chunk10_decisions / decisions if decisions else None
                ),
                "chunk10_time_share": chunk10_steps / steps if steps else None,
                "mean_confidence": (
                    float(np.mean([row["confidence"] for row in selected]))
                    if selected
                    else None
                ),
            }
        )
    return result


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_time_series(
    rows: list[dict],
    *,
    x_label: str,
    output: Path,
) -> None:
    centers = np.asarray([row["bin_center"] for row in rows])
    decision_share = np.asarray(
        [row["chunk10_decision_share"] for row in rows], dtype=float
    )
    time_share = np.asarray([row["chunk10_time_share"] for row in rows], dtype=float)
    confidence = np.asarray([row["mean_confidence"] for row in rows], dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(centers, decision_share, marker="o", label="decision share")
    axes[0].plot(centers, time_share, marker="s", label="environment-time share")
    axes[0].set_ylabel("P(chunk=10)")
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(centers, confidence, marker="o", color="tab:green")
    axes[1].set_ylabel("mean confidence")
    axes[1].set_xlabel(x_label)
    axes[1].set_ylim(0.45, 1.01)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _sequence_summary(rows: list[dict]) -> dict:
    by_episode: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_episode[row["episode"]].append(row)
    transition_counts: Counter[str] = Counter()
    compressed_sequences: Counter[str] = Counter()
    switches: list[int] = []
    for episode_rows in by_episode.values():
        episode_rows.sort(key=lambda row: row["policy_call"])
        chunks = [row["selected_chunk"] for row in episode_rows]
        transition_counts.update(
            f"{left}->{right}" for left, right in zip(chunks, chunks[1:])
        )
        compressed = [chunks[0]] if chunks else []
        for chunk in chunks[1:]:
            if chunk != compressed[-1]:
                compressed.append(chunk)
        switches.append(max(0, len(compressed) - 1))
        compressed_sequences["-".join(map(str, compressed))] += 1
    return {
        "transition_counts": dict(sorted(transition_counts.items())),
        "mean_switches_per_episode": float(np.mean(switches)) if switches else 0.0,
        "median_switches_per_episode": float(np.median(switches)) if switches else 0.0,
        "most_common_compressed_sequences": compressed_sequences.most_common(10),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--normalized-bins", type=int, default=20)
    parser.add_argument("--absolute-bin-size", type=int, default=50)
    args = parser.parse_args()
    output = args.output_dir or args.trace.parent / "selector_temporal_analysis"
    output.mkdir(parents=True, exist_ok=True)
    rows = _read_rows(args.trace)
    if not rows:
        parser.error("trace contains no decisions")

    normalized = _bin_rows(
        rows,
        "normalized_progress",
        np.linspace(0.0, 1.0, args.normalized_bins + 1),
    )
    max_step = max(row["step"] for row in rows)
    absolute_edges = np.arange(
        0,
        max_step + args.absolute_bin_size + 1,
        args.absolute_bin_size,
        dtype=np.float64,
    )
    absolute = _bin_rows(rows, "step", absolute_edges)
    _write_csv(output / "normalized_time_bins.csv", normalized)
    _write_csv(output / "absolute_step_bins.csv", absolute)
    outcome_bins = {
        outcome: _bin_rows(
            [row for row in rows if row["success"] == is_success],
            "normalized_progress",
            np.linspace(0.0, 1.0, args.normalized_bins + 1),
        )
        for outcome, is_success in (("success", True), ("failure", False))
    }
    for outcome, bins in outcome_bins.items():
        _write_csv(output / f"normalized_time_bins_{outcome}.csv", bins)

    chunks = Counter(row["selected_chunk"] for row in rows)
    weighted_chunks = Counter()
    region_chunks: dict[str, Counter[int]] = defaultdict(Counter)
    region_steps: dict[str, Counter[int]] = defaultdict(Counter)
    success_chunks: dict[str, Counter[int]] = defaultdict(Counter)
    success_steps: dict[str, Counter[int]] = defaultdict(Counter)
    for row in rows:
        chunk = row["selected_chunk"]
        outcome = "success" if row["success"] else "failure"
        weighted_chunks[chunk] += row["executed_steps"]
        region_chunks[row["spatial_region"]][chunk] += 1
        region_steps[row["spatial_region"]][chunk] += row["executed_steps"]
        success_chunks[outcome][chunk] += 1
        success_steps[outcome][chunk] += row["executed_steps"]

    summary = {
        "trace": str(args.trace),
        "episodes": len({row["episode"] for row in rows}),
        "decisions": len(rows),
        "decision_counts": dict(sorted(chunks.items())),
        "decision_shares": {
            str(chunk): count / len(rows) for chunk, count in sorted(chunks.items())
        },
        "environment_step_counts": dict(sorted(weighted_chunks.items())),
        "environment_time_shares": {
            str(chunk): count / sum(weighted_chunks.values())
            for chunk, count in sorted(weighted_chunks.items())
        },
        "mean_confidence": float(np.mean([row["confidence"] for row in rows])),
        "region_decision_counts": {
            region: dict(sorted(counts.items()))
            for region, counts in sorted(region_chunks.items())
        },
        "region_environment_step_counts": {
            region: dict(sorted(counts.items()))
            for region, counts in sorted(region_steps.items())
        },
        "outcome_decision_counts": {
            outcome: dict(sorted(counts.items()))
            for outcome, counts in sorted(success_chunks.items())
        },
        "outcome_environment_step_counts": {
            outcome: dict(sorted(counts.items()))
            for outcome, counts in sorted(success_steps.items())
        },
        **_sequence_summary(rows),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    _plot_time_series(
        normalized,
        x_label="normalized episode progress",
        output=output / "chunk_choice_over_normalized_time.png",
    )
    _plot_time_series(
        absolute,
        x_label="environment step",
        output=output / "chunk_choice_over_environment_steps.png",
    )

    fig, ax = plt.subplots(figsize=(9, 4.5))
    centers = np.asarray([row["bin_center"] for row in normalized])
    for outcome, color in (("success", "tab:blue"), ("failure", "tab:red")):
        shares = np.asarray(
            [row["chunk10_time_share"] for row in outcome_bins[outcome]], dtype=float
        )
        ax.plot(centers, shares, marker="o", color=color, label=outcome)
    ax.set_xlabel("normalized episode progress")
    ax.set_ylabel("environment-time share of chunk 10")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "chunk_choice_success_vs_failure.png", dpi=180)
    plt.close(fig)

    regions = [
        "free_approach",
        "pick_precision",
        "free_transport",
        "insert_precision",
    ]
    region_chunk10 = []
    for region in regions:
        counts = region_steps[region]
        total = sum(counts.values())
        region_chunk10.append(counts[10] / total if total else 0.0)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(regions, region_chunk10, color="tab:blue")
    ax.set_ylim(0, 1)
    ax.set_ylabel("environment-time share of chunk 10")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "chunk_choice_by_spatial_region.png", dpi=180)
    plt.close(fig)

    lines = [
        "# Threading selector temporal analysis",
        "",
        f"- Episodes: {summary['episodes']}",
        f"- Decisions: {summary['decisions']}",
        f"- Mean confidence: {summary['mean_confidence']:.3f}",
        f"- Mean switches per episode: {summary['mean_switches_per_episode']:.2f}",
        "",
        "| Spatial region | chunk 4 decisions | chunk 10 decisions | chunk 10 time share |",
        "|---|---:|---:|---:|",
    ]
    for region in regions:
        decisions = region_chunks[region]
        steps = region_steps[region]
        total_steps = sum(steps.values())
        lines.append(
            f"| {region} | {decisions[4]} | {decisions[10]} | "
            f"{(steps[10] / total_steps if total_steps else 0.0):.1%} |"
        )
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Summarize and plot PushBox selector chunk choices over space and rollout time."""
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
            "prediction_length",
            "executed_steps",
            "episode_steps",
        ):
            row[key] = int(row[key])
        for key in (
            "box_x",
            "box_y",
            "confidence",
            "normalized_progress",
            "budget_progress",
        ):
            row[key] = float(row[key])
        row["success"] = row["success"].lower() == "true"
    return rows


def _bin_rows(
    rows: list[dict], key: str, edges: np.ndarray, long_chunk: int
) -> list[dict]:
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    bins = np.clip(np.digitize(values, edges, right=False) - 1, 0, len(edges) - 2)
    result = []
    for index in range(len(edges) - 1):
        selected = [row for row, bin_index in zip(rows, bins) if bin_index == index]
        decisions = len(selected)
        steps = sum(row["executed_steps"] for row in selected)
        long_decisions = sum(row["selected_chunk"] == long_chunk for row in selected)
        long_steps = sum(
            row["executed_steps"]
            for row in selected
            if row["selected_chunk"] == long_chunk
        )
        result.append(
            {
                "bin_start": float(edges[index]),
                "bin_end": float(edges[index + 1]),
                "bin_center": float((edges[index] + edges[index + 1]) / 2),
                "decisions": decisions,
                "environment_steps": steps,
                "long_chunk_decision_share": long_decisions / decisions if decisions else None,
                "long_chunk_time_share": long_steps / steps if steps else None,
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


def _sequence_summary(rows: list[dict]) -> dict:
    by_episode: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_episode[row["episode"]].append(row)
    transitions: Counter[str] = Counter()
    compressed_sequences: Counter[str] = Counter()
    switches = []
    for episode_rows in by_episode.values():
        episode_rows.sort(key=lambda row: row["policy_call"])
        chunks = [row["selected_chunk"] for row in episode_rows]
        transitions.update(f"{a}->{b}" for a, b in zip(chunks, chunks[1:]))
        compressed = [chunks[0]] if chunks else []
        for chunk in chunks[1:]:
            if chunk != compressed[-1]:
                compressed.append(chunk)
        switches.append(max(0, len(compressed) - 1))
        compressed_sequences["-".join(map(str, compressed))] += 1
    return {
        "transition_counts": dict(sorted(transitions.items())),
        "mean_switches_per_episode": float(np.mean(switches)) if switches else 0.0,
        "median_switches_per_episode": float(np.median(switches)) if switches else 0.0,
        "most_common_compressed_sequences": compressed_sequences.most_common(10),
    }


def _plot_series(rows: list[dict], x_label: str, long_chunk: int, output: Path) -> None:
    centers = np.asarray([row["bin_center"] for row in rows])
    decision_share = np.asarray(
        [row["long_chunk_decision_share"] for row in rows], dtype=float
    )
    time_share = np.asarray([row["long_chunk_time_share"] for row in rows], dtype=float)
    confidence = np.asarray([row["mean_confidence"] for row in rows], dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(centers, decision_share, marker="o", label="decision share")
    axes[0].plot(centers, time_share, marker="s", label="environment-time share")
    axes[0].set_ylabel(f"P(chunk={long_chunk})")
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--normalized-bins", type=int, default=20)
    parser.add_argument("--absolute-bin-size", type=int, default=25)
    parser.add_argument("--box-y-bin-size", type=float, default=0.025)
    parser.add_argument("--short-chunk", type=int, default=5)
    parser.add_argument("--long-chunk", type=int, default=19)
    args = parser.parse_args()
    if args.normalized_bins <= 0 or args.absolute_bin_size <= 0 or args.box_y_bin_size <= 0:
        parser.error("bin counts and sizes must be positive")

    rows = _read_rows(args.trace)
    if not rows:
        parser.error("trace contains no decisions")
    expected = {args.short_chunk, args.long_chunk}
    actual = {row["selected_chunk"] for row in rows}
    if not actual <= expected:
        parser.error(f"trace chunks {sorted(actual)} are not a subset of {sorted(expected)}")

    output = args.output_dir or args.trace.parent / "selector_chunk_analysis"
    output.mkdir(parents=True, exist_ok=True)
    normalized_edges = np.linspace(0.0, 1.0, args.normalized_bins + 1)
    absolute_edges = np.arange(
        0,
        max(row["step"] for row in rows) + args.absolute_bin_size + 1,
        args.absolute_bin_size,
        dtype=float,
    )
    y_min = np.floor(min(row["box_y"] for row in rows) / args.box_y_bin_size) * args.box_y_bin_size
    y_max = np.ceil(max(row["box_y"] for row in rows) / args.box_y_bin_size) * args.box_y_bin_size
    box_y_edges = np.arange(y_min, y_max + args.box_y_bin_size * 1.01, args.box_y_bin_size)
    normalized = _bin_rows(rows, "normalized_progress", normalized_edges, args.long_chunk)
    absolute = _bin_rows(rows, "step", absolute_edges, args.long_chunk)
    box_y = _bin_rows(rows, "box_y", box_y_edges, args.long_chunk)
    _write_csv(output / "normalized_time_bins.csv", normalized)
    _write_csv(output / "absolute_step_bins.csv", absolute)
    _write_csv(output / "box_y_bins.csv", box_y)

    outcome_bins = {
        outcome: _bin_rows(
            [row for row in rows if row["success"] == success],
            "normalized_progress",
            normalized_edges,
            args.long_chunk,
        )
        for outcome, success in (("success", True), ("failure", False))
    }
    for outcome, bins in outcome_bins.items():
        _write_csv(output / f"normalized_time_bins_{outcome}.csv", bins)

    decisions = Counter(row["selected_chunk"] for row in rows)
    weighted = Counter()
    region_decisions: dict[str, Counter[int]] = defaultdict(Counter)
    region_steps: dict[str, Counter[int]] = defaultdict(Counter)
    outcome_decisions: dict[str, Counter[int]] = defaultdict(Counter)
    for row in rows:
        chunk = row["selected_chunk"]
        steps = row["executed_steps"]
        outcome = "success" if row["success"] else "failure"
        weighted[chunk] += steps
        region_decisions[row["spatial_region"]][chunk] += 1
        region_steps[row["spatial_region"]][chunk] += steps
        outcome_decisions[outcome][chunk] += 1

    summary = {
        "trace": str(args.trace),
        "episodes": len({row["episode"] for row in rows}),
        "successful_episodes": len({row["episode"] for row in rows if row["success"]}),
        "decisions": len(rows),
        "candidate_chunks": [args.short_chunk, args.long_chunk],
        "decision_counts": dict(sorted(decisions.items())),
        "decision_shares": {str(k): v / len(rows) for k, v in sorted(decisions.items())},
        "environment_step_counts": dict(sorted(weighted.items())),
        "environment_time_shares": {
            str(k): v / sum(weighted.values()) for k, v in sorted(weighted.items())
        },
        "mean_confidence": float(np.mean([row["confidence"] for row in rows])),
        "region_decision_counts": {
            region: dict(sorted(counts.items())) for region, counts in sorted(region_decisions.items())
        },
        "region_environment_step_counts": {
            region: dict(sorted(counts.items())) for region, counts in sorted(region_steps.items())
        },
        "outcome_decision_counts": {
            outcome: dict(sorted(counts.items())) for outcome, counts in sorted(outcome_decisions.items())
        },
        **_sequence_summary(rows),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    _plot_series(normalized, "normalized episode progress", args.long_chunk, output / "chunk_choice_over_normalized_time.png")
    _plot_series(absolute, "environment step", args.long_chunk, output / "chunk_choice_over_environment_steps.png")
    _plot_series(box_y, "Box Y (m)", args.long_chunk, output / "chunk_choice_over_box_y.png")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    centers = np.asarray([row["bin_center"] for row in normalized])
    for outcome, color in (("success", "tab:blue"), ("failure", "tab:red")):
        shares = np.asarray(
            [row["long_chunk_time_share"] for row in outcome_bins[outcome]], dtype=float
        )
        ax.plot(centers, shares, marker="o", color=color, label=outcome)
    ax.set_xlabel("normalized episode progress")
    ax.set_ylabel(f"environment-time share of chunk {args.long_chunk}")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "chunk_choice_success_vs_failure.png", dpi=180)
    plt.close(fig)

    regions = ["before_crossing", "crossing_precision", "after_crossing"]
    long_time_shares = []
    for region in regions:
        counts = region_steps[region]
        total = sum(counts.values())
        long_time_shares.append(counts[args.long_chunk] / total if total else 0.0)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(regions, long_time_shares, color="tab:blue")
    ax.set_ylim(0, 1)
    ax.set_ylabel(f"environment-time share of chunk {args.long_chunk}")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "chunk_choice_by_spatial_region.png", dpi=180)
    plt.close(fig)

    lines = [
        "# PushBox selector chunk analysis",
        "",
        f"- Episodes: {summary['episodes']}",
        f"- Successful episodes: {summary['successful_episodes']}",
        f"- Decisions: {summary['decisions']}",
        f"- Mean confidence: {summary['mean_confidence']:.3f}",
        f"- Mean switches per episode: {summary['mean_switches_per_episode']:.2f}",
        "",
        f"| Spatial region | chunk {args.short_chunk} decisions | chunk {args.long_chunk} decisions | chunk {args.long_chunk} time share |",
        "|---|---:|---:|---:|",
    ]
    for region in regions:
        counts = region_decisions[region]
        steps = region_steps[region]
        total = sum(steps.values())
        lines.append(
            f"| {region} | {counts[args.short_chunk]} | {counts[args.long_chunk]} | "
            f"{(steps[args.long_chunk] / total if total else 0.0):.1%} |"
        )
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

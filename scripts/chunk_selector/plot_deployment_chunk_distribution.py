#!/usr/bin/env python3
"""Plot selector execution-chunk frequencies over normalized deployment time."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="Selector JSONL trace file")
    parser.add_argument("--output", type=Path, required=True, help="Output PNG path")
    parser.add_argument("--csv-output", type=Path, required=True, help="Output frequency CSV path")
    parser.add_argument("--bins", type=int, default=10, help="Normalized-time bins (default: 10)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bins < 1:
        raise ValueError("--bins must be positive")

    records = [json.loads(line) for line in args.trace.read_text().splitlines() if line.strip()]
    # Trace output is append-only.  The runner restarts its episode counter at one
    # on every invocation, so the JSONL ``episode`` field is not globally unique.
    # A new cycle-one record is the reliable boundary between trace sequences.
    trace_sequences: list[list[dict]] = []
    for record in records:
        if not trace_sequences or record["cycle"] == 1:
            trace_sequences.append([])
        trace_sequences[-1].append(record)

    normalized: list[tuple[float, int]] = []
    for sequence_records in trace_sequences:
        start, end = sequence_records[0]["timestamp"], sequence_records[-1]["timestamp"]
        duration = end - start
        for record in sequence_records:
            # A one-decision episode is placed at the beginning of its normalized run.
            progress = (record["timestamp"] - start) / duration if duration else 0.0
            normalized.append((min(progress, 1.0), int(record["execution_chunk"])))

    chunks = sorted({chunk for _, chunk in normalized})
    counts = {bin_index: Counter() for bin_index in range(args.bins)}
    totals = Counter()
    for progress, chunk in normalized:
        bin_index = min(int(progress * args.bins), args.bins - 1)
        counts[bin_index][chunk] += 1
        totals[bin_index] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "normalized_time_start", "normalized_time_end", "decision_count",
            "execution_chunk", "count", "frequency",
        ])
        for bin_index in range(args.bins):
            for chunk in chunks:
                count = counts[bin_index][chunk]
                writer.writerow([
                    bin_index / args.bins, (bin_index + 1) / args.bins, totals[bin_index],
                    chunk, count, count / totals[bin_index] if totals[bin_index] else 0.0,
                ])

    centers = np.array([(index + 0.5) / args.bins for index in range(args.bins)])
    fig, axis = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
    for chunk in chunks:
        frequencies = [counts[index][chunk] / totals[index] if totals[index] else 0.0 for index in range(args.bins)]
        axis.plot(centers, frequencies, marker="o", linewidth=1.8, label=f"{chunk} steps")
    axis.set(
        xlabel="Normalized episode time",
        ylabel="Chunk frequency",
        xlim=(0, 1),
        ylim=(0, 1.05),
        xticks=np.linspace(0, 1, 6),
        yticks=np.linspace(0, 1, 6),
        title=(f"Threading selector chunks over normalized time ({len(trace_sequences)} trace sequences, "
               f"{len(normalized)} decisions)"),
    )
    axis.set_xticklabels([f"{value:.0%}" for value in axis.get_xticks()])
    axis.set_yticklabels([f"{value:.0%}" for value in axis.get_yticks()])
    axis.grid(axis="y", alpha=0.3)
    axis.legend(title="Executed chunk", ncol=2, fontsize=9)
    fig.savefig(args.output, dpi=200)


if __name__ == "__main__":
    main()

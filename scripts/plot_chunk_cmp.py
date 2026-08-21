#!/usr/bin/env python3
"""Plot chunk-size comparison chart from chunk_comparison.json."""

import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_comparison(json_path: str, output_dir: Optional[str] = None) -> None:
    with open(json_path) as f:
        data = json.load(f)

    chunk_data: dict[str, dict] = data["chunk_sizes"]
    chunk_sizes = sorted(chunk_data.keys(), key=int)  # ["10","20","30","40","50"]
    subtasks = sorted(
        chunk_data[chunk_sizes[0]]["subtasks"].keys(),
        key=int,  # ["1","2","3"]
    )

    n_chunks = len(chunk_sizes)
    n_subtasks = len(subtasks)

    cmap = plt.cm.viridis  # auto-generate enough distinct colors
    colors = [cmap(i / max(n_chunks - 1, 1)) for i in range(n_chunks)]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    x = np.arange(n_subtasks)
    width = 0.8 / n_chunks  # auto-fit bar width

    # ── Top: Success Rate ──────────────────────────────────────────
    for i, cs in enumerate(chunk_sizes):
        rates = []
        for st in subtasks:
            r = chunk_data[cs]["subtasks"].get(st, {})
            rates.append(r.get("success_rate", 0) * 100)
        offset = (i - (n_chunks - 1) / 2) * width
        bars = ax1.bar(x + offset, rates, width, label=f"chunk={cs}", color=colors[i])
        for bar, rate in zip(bars, rates):
            if rate > 0:
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.5,
                    f"{rate:.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax1.set_ylabel("Success Rate (%)")
    ax1.set_title(f"Chunk Size Comparison — {Path(json_path).parent.name}", fontsize=12)
    ax1.set_ylim(0, 115)
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    # ── Middle: Avg Wall Time ───────────────────────────────────────
    for i, cs in enumerate(chunk_sizes):
        times = []
        for st in subtasks:
            r = chunk_data[cs]["subtasks"].get(st, {})
            t = r.get("wall_time", {}).get("mean", 0)
            times.append(t)
        offset = (i - (n_chunks - 1) / 2) * width
        bars = ax2.bar(x + offset, times, width, label=f"chunk={cs}", color=colors[i])
        for bar, t in zip(bars, times):
            if t > 0:
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{t:.2f}s",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    ax2.set_ylabel("Avg Wall Time (s)")
    ax2.set_title("Wall Time")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    # ── Bottom: Avg Inference Passes ────────────────────────────────
    for i, cs in enumerate(chunk_sizes):
        passes = []
        for st in subtasks:
            r = chunk_data[cs]["subtasks"].get(st, {})
            p = r.get("inference_passes", {}).get("mean", 0)
            passes.append(p)
        offset = (i - (n_chunks - 1) / 2) * width
        bars = ax3.bar(x + offset, passes, width, label=f"chunk={cs}", color=colors[i])
        max_pass = max(passes) if passes else 0
        for bar, p in zip(bars, passes):
            if p > 0:
                ax3.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max_pass * 0.03,
                    f"{p:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax3.set_ylabel("Avg Inference Passes")
    ax3.set_xlabel("Subtask")
    ax3.set_title("Inference Passes")
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"Subtask {st}" for st in subtasks])
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    out = Path(output_dir) if output_dir else Path(json_path).parent
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / "chunk_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_chunk_cmp.py <chunk_comparison.json> [output_dir]")
        sys.exit(1)

    json_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else None
    plot_comparison(json_path, out_dir)

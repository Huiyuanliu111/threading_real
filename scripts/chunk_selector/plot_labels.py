#!/usr/bin/env python3
"""Plot TCP-derived adaptive-chunk pseudo-labels for one LeRobot episode."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


COLORS = {1: "#d73027", 2: "#fc8d59", 4: "#fee08b", 8: "#91bfdb", 20: "#4575b4"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path,
                        default=Path("data/threading_lerobot_v3_cartesian_tcp_chunk_labels/labels.parquet"))
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--output", type=Path,
                        default=Path("data/threading_lerobot_v3_cartesian_tcp_chunk_labels/episode_000_chunk_distribution.png"))
    args = parser.parse_args()

    labels = pd.read_parquet(args.labels)
    episode = labels.loc[labels["episode_index"] == args.episode].sort_values("frame_index")
    if episode.empty:
        raise ValueError(f"Episode {args.episode} is not present in {args.labels}")

    normalized_time = (episode["frame_index"].to_numpy() - episode["frame_index"].iloc[0])
    normalized_time = normalized_time / max(normalized_time[-1], 1)
    chunks = episode["chunk_size"].to_numpy()

    fig, ax = plt.subplots(figsize=(12, 3.8), constrained_layout=True)
    ax.step(normalized_time, chunks, where="post", color="#333333", linewidth=0.8, alpha=0.55)
    for chunk_size, color in COLORS.items():
        mask = chunks == chunk_size
        ax.scatter(normalized_time[mask], chunks[mask], s=12, color=color,
                   label=f"chunk {chunk_size}", zorder=3)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 21)
    ax.set_yticks(sorted(COLORS))
    ax.set_xlabel("Normalized episode time")
    ax.set_ylabel("Pseudo-label chunk size")
    ax.set_title(f"TCP-derived chunk labels — episode {args.episode:03d} ({len(episode)} frames)")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.23), frameon=False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()

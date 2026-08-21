#!/usr/bin/env python3
"""Plot the PushBox full-plan versus required-only experiment matrix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODES = ("full_then_truncate", "required_only")
MODE_LABELS = {
    "full_then_truncate": "Full then truncate",
    "required_only": "Required only",
}
MODE_COLORS = {
    "full_then_truncate": "#4C78A8",
    "required_only": "#F58518",
}
CONFIGS = (
    ("fixed", 2),
    ("fixed", 4),
    ("fixed", 5),
    ("fixed", 6),
    ("fixed", 8),
    ("fixed", 10),
    ("fixed", 19),
    ("selector", 5),
    ("selector", 10),
    ("spatial", 4),
    ("spatial", 10),
)
GROUP_BOUNDARIES = (6.5, 8.5)


def _load_records(path: Path) -> dict[tuple[str, str, int], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("completed") != payload.get("expected"):
        raise ValueError(
            f"matrix is incomplete: {payload.get('completed')}/{payload.get('expected')}"
        )
    return {
        (item["prediction_mode"], item["strategy"], int(item["chunk"])): item
        for item in payload["experiments"]
    }


def _decorate_axis(axis, *, ylabel: str, title: str) -> None:
    axis.set_title(title, loc="left", fontweight="bold")
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", color="#D8D8D8", linewidth=0.8, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    for boundary in GROUP_BOUNDARIES:
        axis.axvline(boundary, color="#A0A0A0", linewidth=1.0, linestyle="--")


def plot_matrix(summary_path: Path, output_prefix: Path) -> tuple[Path, Path]:
    records = _load_records(summary_path)
    labels = [f"{strategy.title()} {chunk}" for strategy, chunk in CONFIGS]
    positions = np.arange(len(CONFIGS), dtype=float)
    width = 0.38
    panels = (
        ("success_rate", "Success rate (%)", "A  Closed-loop success", 100.0, ".0f"),
        ("wall_time_mean_all", "Seconds / episode", "B  End-to-end wall time", 1.0, ".2f"),
        (
            "inference_time_mean_all",
            "Seconds / episode",
            "C  Policy inference time",
            1.0,
            ".2f",
        ),
        (
            "generated_action_tokens_mean_all",
            "Tokens / episode",
            "D  Generated action tokens",
            1.0,
            ".0f",
        ),
    )

    figure, axes = plt.subplots(2, 2, figsize=(18, 10), constrained_layout=True)
    for axis, (key, ylabel, title, scale, number_format) in zip(axes.flat, panels):
        maximum = 0.0
        for mode_index, mode in enumerate(MODES):
            values = [
                float(records[(mode, strategy, chunk)][key]) * scale
                for strategy, chunk in CONFIGS
            ]
            maximum = max(maximum, max(values))
            bars = axis.bar(
                positions + (mode_index - 0.5) * width,
                values,
                width=width,
                color=MODE_COLORS[mode],
                edgecolor="white",
                linewidth=0.6,
                label=MODE_LABELS[mode],
            )
            axis.bar_label(
                bars,
                labels=[format(value, number_format) for value in values],
                padding=2,
                fontsize=7.5,
                rotation=90 if key == "generated_action_tokens_mean_all" else 0,
            )
        _decorate_axis(axis, ylabel=ylabel, title=title)
        axis.set_xticks(positions, labels, rotation=28, ha="right")
        axis.set_ylim(0, maximum * (1.22 if key != "generated_action_tokens_mean_all" else 1.32))
        if key == "success_rate":
            axis.set_ylim(0, 100)
            axis.set_yticks(np.arange(0, 101, 20))

    figure.suptitle(
        "PushBox prediction-mode matrix (50 paired episodes per configuration)",
        fontsize=16,
        fontweight="bold",
    )
    axes[0, 1].legend(loc="upper right", frameon=False)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    figure.savefig(png_path, dpi=220, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return png_path, pdf_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="output path without extension; defaults beside matrix_summary.json",
    )
    args = parser.parse_args()
    output_prefix = args.output_prefix or args.summary.with_name(
        "prediction_mode_matrix_comparison"
    )
    png_path, pdf_path = plot_matrix(args.summary, output_prefix)
    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Plot box position density heatmap from LeRobot datagen dataset.

Usage:
    python scripts/plot_data_density.py [--data data/datagen] [--out density.png] [--bins 100]
    python scripts/plot_data_density.py --weights [--temp 1.0] [--sigma 1.5]
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pushbox.paths import DEFAULT_DEMO_DIR

# ── desk layout constants ──────────────────────────────────────────────────
TABLE_HALF = 0.4          # half of 0.8m table
BOX_HALF = 0.025
TABLE_XY_LIMIT = TABLE_HALF - BOX_HALF - 0.03  # 0.345

GOAL_POS = (0.0, 0.25)
GOAL_RADIUS = 0.05
BOX_INIT_X = (0.01, 0.05)
BOX_INIT_Y = (-0.25, -0.20)
TARGET_X = (GOAL_POS[0] - GOAL_RADIUS, GOAL_POS[0] + GOAL_RADIUS)
TARGET_Y = (GOAL_POS[1] - GOAL_RADIUS, GOAL_POS[1] + GOAL_RADIUS)


def load_box_positions(data_dir: Path) -> np.ndarray:
    """Load all box_pos from parquet files, return (N, 2) float32 array."""
    data_files = sorted((data_dir / "data").rglob("*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No parquet files found under {data_dir / 'data'}")
    tables = [pq.read_table(str(f)) for f in data_files]
    table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
    box = np.array(table.column("observation.box_pos").to_pylist(), dtype=np.float32)
    print(f"Loaded {len(box):,} box positions from {len(data_files)} parquet file(s)")
    print(f"  x in [{box[:,0].min():.4f}, {box[:,0].max():.4f}]")
    print(f"  y in [{box[:,1].min():.4f}, {box[:,1].max():.4f}]")
    return box


def plot_density(box: np.ndarray, out_path: str, bins: int = 100,
                 view_range: tuple = None):
    """Plot 2D histogram heatmap with desk layout annotations on a single axes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Rectangle, Circle

    if view_range is None:
        view_range = (-TABLE_XY_LIMIT, TABLE_XY_LIMIT, -TABLE_XY_LIMIT, TABLE_XY_LIMIT)

    xmin, xmax, ymin, ymax = view_range

    # Compute 2D histogram
    H, xe, ye = np.histogram2d(box[:, 0], box[:, 1], bins=bins,
                                range=[[xmin, xmax], [ymin, ymax]])
    # Transpose so x->columns, y->rows (matches imshow default)
    H = H.T

    fig, ax = plt.subplots(figsize=(10, 10))

    # Log-scale density heatmap
    im = ax.imshow(H, extent=[xmin, xmax, ymin, ymax], origin="lower",
                   cmap="inferno", norm=LogNorm(vmin=max(H.max() * 1e-4, 1)),
                   aspect="equal")

    # ── desk boundaries ──
    ax.add_patch(Rectangle((-TABLE_HALF, -TABLE_HALF), TABLE_HALF * 2, TABLE_HALF * 2,
                            fill=False, edgecolor="white", linewidth=2, linestyle="--",
                            label="table edge"))
    ax.add_patch(Rectangle((-TABLE_XY_LIMIT, -TABLE_XY_LIMIT),
                            TABLE_XY_LIMIT * 2, TABLE_XY_LIMIT * 2,
                            fill=False, edgecolor="cyan", linewidth=1,
                            label="box limit"))

    # ── data bounding box ──
    ax.add_patch(Rectangle((box[:, 0].min(), box[:, 1].min()),
                            box[:, 0].max() - box[:, 0].min(),
                            box[:, 1].max() - box[:, 1].min(),
                            fill=False, edgecolor="yellow", linewidth=1.5, linestyle=":",
                            label="data range"))

    # ── goal region ──
    goal = Circle(GOAL_POS, GOAL_RADIUS, fill=False, edgecolor="lime",
                  linewidth=2, label="goal")
    ax.add_patch(goal)
    ax.plot(GOAL_POS[0], GOAL_POS[1], "lime", marker="x", markersize=10,
            markeredgewidth=2)

    # ── init region ──
    ax.add_patch(Rectangle((BOX_INIT_X[0], BOX_INIT_Y[0]),
                            BOX_INIT_X[1] - BOX_INIT_X[0],
                            BOX_INIT_Y[1] - BOX_INIT_Y[0],
                            fill=False, edgecolor="red", linewidth=1.5,
                            linestyle="-", label="init region"))

    # ── colorbar ──
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("frame count (log scale)")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"Box Position Density ({len(box):,} frames, {bins}x{bins} bins)")

    ax.legend(loc="upper right", fontsize=8, framealpha=0.7)

    # ── stats text ──
    stats = (
        f"x in [{box[:,0].min():.3f}, {box[:,0].max():.3f}]  "
        f"|  y in [{box[:,1].min():.3f}, {box[:,1].max():.3f}]\n"
        f"x span: {box[:,0].max() - box[:,0].min():.3f}m  "
        f"y span: {box[:,1].max() - box[:,1].min():.3f}m\n"
        f"desk xy limit: +/-{TABLE_XY_LIMIT:.3f}m"
    )
    ax.text(0.02, 0.02, stats, transform=ax.transAxes, fontsize=8,
            family="monospace", verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.6, edgecolor="gray"))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot box position density heatmap")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DEMO_DIR),
                        help="LeRobot dataset directory")
    parser.add_argument("--out", type=str, default="density.png",
                        help="output image path")
    parser.add_argument("--bins", type=int, default=100,
                        help="number of histogram bins per axis")
    parser.add_argument("--zoom", action="store_true",
                        help="zoom to data range instead of full table")
    parser.add_argument("--weights", action="store_true",
                        help="plot inverse-density sampling weights (instead of raw density)")
    parser.add_argument("--sigma", type=float, default=1.5,
                        help="Gaussian smoothing sigma for weight map")
    parser.add_argument("--temp", type=float, default=1.0,
                        help="temperature: >1 sharpens, <1 flattens weights")
    args = parser.parse_args()

    data_dir = Path(args.data).expanduser()
    box = load_box_positions(data_dir)

    view_range = None
    if args.zoom:
        margin = 0.02
        view_range = (box[:, 0].min() - margin, box[:, 0].max() + margin,
                      box[:, 1].min() - margin, box[:, 1].max() + margin)

    if args.weights:
        plot_weights(box, args.out, bins=min(args.bins, 80),
                     smooth_sigma=args.sigma, temp=args.temp,
                     view_range=view_range)
    else:
        plot_density(box, args.out, bins=args.bins, view_range=view_range)


def plot_weights(box: np.ndarray, out_path: str, bins: int = 50,
                 smooth_sigma: float = 1.5, temp: float = 1.0,
                 view_range: tuple = None):
    """Plot inverse-density sampling weights as a heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle

    if view_range is None:
        view_range = (-TABLE_XY_LIMIT, TABLE_XY_LIMIT, -TABLE_XY_LIMIT, TABLE_XY_LIMIT)

    xmin, xmax, ymin, ymax = view_range

    # Compute 2D histogram
    H, xe, ye = np.histogram2d(box[:, 0], box[:, 1], bins=bins,
                                range=[[xmin, xmax], [ymin, ymax]])
    H = H.T.astype(np.float32)
    H[H == 0] = 1
    if smooth_sigma > 0:
        H = gaussian_filter(H, sigma=smooth_sigma)

    inv_density = 1.0 / (H + 1e-6)
    inv_density = np.power(inv_density, temp)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    from matplotlib.colors import LogNorm
    # Left: frame density
    ax = axes[0]
    im0 = ax.imshow(H, extent=[xmin, xmax, ymin, ymax], origin="lower",
                    cmap="inferno", norm=LogNorm(vmin=max(H.max() * 1e-4, 1)),
                    aspect="equal")
    ax.add_patch(Rectangle((-TABLE_HALF, -TABLE_HALF), TABLE_HALF * 2, TABLE_HALF * 2,
                            fill=False, edgecolor="white", linewidth=2, linestyle="--",
                            label="table edge"))
    ax.add_patch(Rectangle((-TABLE_XY_LIMIT, -TABLE_XY_LIMIT),
                            TABLE_XY_LIMIT * 2, TABLE_XY_LIMIT * 2,
                            fill=False, edgecolor="cyan", linewidth=1,
                            label="box limit"))
    goal = Circle(GOAL_POS, GOAL_RADIUS, fill=False, edgecolor="lime", linewidth=2, label="goal")
    ax.add_patch(goal)
    ax.plot(GOAL_POS[0], GOAL_POS[1], "lime", marker="x", markersize=10, markeredgewidth=2)
    ax.add_patch(Rectangle((BOX_INIT_X[0], BOX_INIT_Y[0]),
                            BOX_INIT_X[1] - BOX_INIT_X[0],
                            BOX_INIT_Y[1] - BOX_INIT_Y[0],
                            fill=False, edgecolor="red", linewidth=1.5, label="init"))
    ax.set_title("Frame Density (log scale)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.7)
    plt.colorbar(im0, ax=ax, fraction=0.046, pad=0.04, label="frame count")

    # Right: sampling weights (high weight → rare region)
    ax = axes[1]
    im1 = ax.imshow(inv_density, extent=[xmin, xmax, ymin, ymax], origin="lower",
                    cmap="RdYlGn_r", aspect="equal")
    ax.add_patch(Rectangle((-TABLE_HALF, -TABLE_HALF), TABLE_HALF * 2, TABLE_HALF * 2,
                            fill=False, edgecolor="white", linewidth=2, linestyle="--",
                            label="table edge"))
    ax.add_patch(Rectangle((-TABLE_XY_LIMIT, -TABLE_XY_LIMIT),
                            TABLE_XY_LIMIT * 2, TABLE_XY_LIMIT * 2,
                            fill=False, edgecolor="cyan", linewidth=1,
                            label="box limit"))
    goal2 = Circle(GOAL_POS, GOAL_RADIUS, fill=False, edgecolor="lime", linewidth=2, label="goal")
    ax.add_patch(goal2)
    ax.plot(GOAL_POS[0], GOAL_POS[1], "lime", marker="x", markersize=10, markeredgewidth=2)
    ax.add_patch(Rectangle((BOX_INIT_X[0], BOX_INIT_Y[0]),
                            BOX_INIT_X[1] - BOX_INIT_X[0],
                            BOX_INIT_Y[1] - BOX_INIT_Y[0],
                            fill=False, edgecolor="red", linewidth=1.5, label="init"))
    ax.set_title(f"Sampling Weights (sigma={smooth_sigma}, temp={temp})\n"
                 f"min={inv_density.min():.2f}  max={inv_density.max():.2f}  "
                 f"ratio={inv_density.max()/inv_density.min():.1f}:1")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.7)
    plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04, label="weight")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()

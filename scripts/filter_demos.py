"""Downsample stationary (zero-action) segments in push-box demo trajectories.

Usage:
    python scripts/filter_demos.py --in data/no_rotate --out data/no_rotate_filtered --dry-run
    python scripts/filter_demos.py --in data/no_rotate --out data/no_rotate_filtered

The script keeps all non-stationary frames in full, and downsamples long runs of
stationary (near-zero-action, gripper-unchanged) frames by keeping the boundary
frames plus a sparse subset in between.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def filter_zero_actions(
    actions: np.ndarray,
    pos_epsilon: float = 1e-3,
    rot_epsilon: float = 1e-3,
    keep_every_n: int = 10,
) -> np.ndarray:
    """Return a boolean mask (same length as actions) indicating which frames to keep.

    Stationary frame definition (ALL conditions must hold):
      - |translation delta|  < pos_epsilon  on the current action
      - |rotation delta|     < rot_epsilon  on the current action
      - gripper command unchanged from previous frame

    Downsampling rule for each contiguous stationary segment [start, end):
      - always keep start (first stationary frame)
      - always keep end-1 (last stationary frame)
      - interior frames [start+1, end-1): keep every keep_every_n-th frame
    """
    n = len(actions)
    if n <= 2:
        return np.ones(n, dtype=bool)

    T, R, G = slice(0, 3), slice(3, 6), 6

    # ---------- 1. classify every frame ----------
    stationary = np.zeros(n, dtype=bool)
    # frame 0 can't reference a previous gripper → non-stationary
    for i in range(1, n):
        trans_still = np.all(np.abs(actions[i, T]) < pos_epsilon)
        rot_still = np.all(np.abs(actions[i, R]) < rot_epsilon)
        grip_unchanged = (actions[i, G] == actions[i - 1, G])
        stationary[i] = trans_still and rot_still and grip_unchanged

    # ---------- 2. build keep mask ----------
    keep = np.ones(n, dtype=bool)

    i = 0
    while i < n:
        if not stationary[i]:
            i += 1
            continue

        start = i
        while i < n and stationary[i]:
            i += 1
        end = i  # exclusive: stationary[start:end] are all True

        run_len = end - start
        if run_len <= 2:
            # too short — keep all (already True)
            continue

        # always keep the last stationary frame
        keep[end - 1] = True

        # interior: downsample
        for j in range(start + 1, end - 1):
            offset = j - start  # 1-indexed within the run
            keep[j] = (offset % keep_every_n == 0)

    return keep


def filter_episode(src: Path, dst: Path, **kwargs) -> dict:
    """Load one npz episode, filter, save. Returns stats dict."""
    data = dict(np.load(src, allow_pickle=True))
    actions = data["actions"]
    original_len = len(actions)

    mask = filter_zero_actions(actions, **kwargs)
    filtered_len = int(mask.sum())

    filtered_data = {}
    for key, arr in data.items():
        if isinstance(arr, np.ndarray) and len(arr) == original_len:
            filtered_data[key] = arr[mask]
        else:
            filtered_data[key] = arr

    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, **filtered_data)

    return {
        "episode": src.name,
        "original": original_len,
        "filtered": filtered_len,
        "removed": original_len - filtered_len,
        "kept_pct": 100.0 * filtered_len / original_len,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Downsample stationary frames in push-box demo data"
    )
    parser.add_argument("--in", dest="in_dir", required=True,
                        help="input directory with ep_*.npz")
    parser.add_argument("--out", dest="out_dir", required=True,
                        help="output directory")
    parser.add_argument("--pos-epsilon", type=float, default=1e-3,
                        help="translation deadband")
    parser.add_argument("--rot-epsilon", type=float, default=1e-3,
                        help="rotation deadband")
    parser.add_argument("--keep-every-n", type=int, default=10,
                        help="keep 1 per N interior stationary frames")
    parser.add_argument("--dry-run", action="store_true",
                        help="print stats without writing files")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    paths = sorted(in_dir.glob("ep_*.npz"))
    if not paths:
        print(f"No ep_*.npz files found in {in_dir}")
        return

    filter_kwargs = {
        "pos_epsilon": args.pos_epsilon,
        "rot_epsilon": args.rot_epsilon,
        "keep_every_n": args.keep_every_n,
    }

    total_orig, total_filt = 0, 0
    for p in paths:
        dst = out_dir / p.name
        stats = filter_episode(p, dst, **filter_kwargs)
        total_orig += stats["original"]
        total_filt += stats["filtered"]
        print(
            f"  {stats['episode']}: {stats['original']:4d} -> {stats['filtered']:4d}  "
            f"({stats['removed']:4d} removed, {stats['kept_pct']:5.1f}% kept)"
        )

    print(f"\nTotal: {total_orig} -> {total_filt} frames "
          f"({100.0 * total_filt / total_orig:.1f}% kept)")
    print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    main()

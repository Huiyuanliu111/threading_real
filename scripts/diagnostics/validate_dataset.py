"""Validate PushBox LeRobot v3 dataset for ARP training compatibility.

Usage:
    python scripts/validate_demos.py                               # check default data/demos
    python scripts/validate_demos.py --dir data/demos              # check directory
    python scripts/validate_demos.py --dir path/to/dataset         # check LeRobot dataset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Ensure project root is on sys.path so pushbox can be imported
# whether or not the package is installed via pip.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from pushbox.lerobot_io import load_lerobot_episodes

    HAS_LEROBOT = True
except ImportError:
    HAS_LEROBOT = False

# --- expected spec ---
IMG_H, IMG_W = 96, 96
AGENT_STATE_DIM = 9   # 7 joint pos + 2 gripper qpos
ACTION_DIM = 7
BOX_POS_DIM = 2
EXPECTED_FEATURES = {
    "observation.images.top45",
    "observation.images.sideview",
    "observation.state",
    "action",
}

# Panda joint limits — standard Franka Emika Panda (radians)
JOINT_MINS = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], dtype=np.float32)
JOINT_MAXS = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973], dtype=np.float32)
GRIPPER_MIN, GRIPPER_MAX = -0.05, 0.05

LIMIT_MARGIN = 0.05


def _find_dataset_root(start: Path) -> Path | None:
    for p in [start] + list(start.parents):
        if (p / "meta" / "info.json").exists():
            return p
    return None


def check_lerobot_dataset(dataset_dir: Path, verbose: bool = True) -> dict:
    """Validate a LeRobot v3 dataset directory."""
    msgs = []
    results = {"ok": 0, "warn": 0, "fail": 0}

    def ok(msg):
        msgs.append(f"  OK {msg}")

    def warn(msg):
        msgs.append(f"  WARN {msg}")

    def fail(msg):
        msgs.append(f"  FAIL {msg}")

    # Find dataset root
    root = _find_dataset_root(dataset_dir)
    if root is None:
        return {
            "status": "fail",
            "messages": [f"  FAIL No LeRobot dataset found: missing meta/info.json in {dataset_dir}"],
            "stats": {},
        }
    ok(f"LeRobot dataset found at: {root}")

    # Load info.json
    import json

    info_path = root / "meta" / "info.json"
    try:
        info = json.loads(info_path.read_text())
    except Exception as e:
        return {"status": "fail", "messages": [f"  FAIL Cannot read info.json: {e}"], "stats": {}}

    fps = info.get("fps", "unknown")
    features = info.get("features", {})
    ok(f"info.json: fps={fps}, features={sorted(features.keys())}")

    # Check expected features
    actual_features = set(features.keys())
    missing = EXPECTED_FEATURES - actual_features
    if missing:
        for feat in missing:
            fail(f"Missing feature: {feat}")
    else:
        ok("All expected features present")

    # Check feature shapes/dtypes
    feature_checks = {
        "observation.state": (AGENT_STATE_DIM, "float32"),
        "action": (ACTION_DIM, "float32"),
    }
    for feat_name, (expected_shape, expected_dtype) in feature_checks.items():
        if feat_name in features:
            feat = features[feat_name]
            shape = feat.get("shape")
            dtype = feat.get("dtype")
            if shape is not None:
                actual = shape[-1] if isinstance(shape, list) else shape
                expected = expected_shape
                if actual == expected:
                    ok(f"{feat_name}: shape={shape}, dtype={dtype}")
                else:
                    fail(f"{feat_name}: shape={shape}, expected dim={expected}")
            if dtype and dtype != expected_dtype:
                warn(f"{feat_name}: dtype={dtype}, expected {expected_dtype}")

    # Video feature checks
    for video_feat in ["observation.images.top45", "observation.images.sideview"]:
        if video_feat in features:
            feat = features[video_feat]
            if feat.get("dtype") == "video":
                ok(f"{video_feat}: video feature OK")
            else:
                warn(f"{video_feat}: expected dtype=video, got {feat.get('dtype')}")
        else:
            fail(f"Missing video feature: {video_feat}")

    # Check per-episode data
    try:
        episodes = load_lerobot_episodes(str(root))
    except Exception as e:
        return {"status": "fail", "messages": msgs + [f"  FAIL Cannot load dataset: {e}"], "stats": {}}

    total_episodes = len(episodes)
    ok(f"Total episodes: {total_episodes}")

    for ep_idx, ep in enumerate(episodes):
        episode_ok = True

        # Check state
        state = ep["state"]
        if state.shape[-1] != AGENT_STATE_DIM:
            fail(f"Episode {ep_idx}: state dim={state.shape[-1]}, expected {AGENT_STATE_DIM}")
            episode_ok = False
        else:
            joints = state[:, :7]
            gripper = state[:, 7:]
            jmin, jmax = joints.min(axis=0), joints.max(axis=0)
            gmin, gmax = gripper.min(axis=0), gripper.max(axis=0)
            joint_violations = []
            for i in range(7):
                if jmin[i] < JOINT_MINS[i] - LIMIT_MARGIN:
                    joint_violations.append(f"joint{i+1} min={jmin[i]:.3f}")
                if jmax[i] > JOINT_MAXS[i] + LIMIT_MARGIN:
                    joint_violations.append(f"joint{i+1} max={jmax[i]:.3f}")
            if joint_violations:
                warn(f"Episode {ep_idx}: joint out of range: {', '.join(joint_violations)}")
                episode_ok = False

            if gmin[0] < GRIPPER_MIN or gmax[0] > GRIPPER_MAX:
                warn(f"Episode {ep_idx}: gripper qpos out of range [{gmin[0]:.4f}, {gmax[0]:.4f}]")
                episode_ok = False

        # Check action
        action = ep["action"]
        if action.shape[-1] != ACTION_DIM:
            fail(f"Episode {ep_idx}: action dim={action.shape[-1]}, expected {ACTION_DIM}")
            episode_ok = False
        else:
            abs_max = np.abs(action[:, :6]).max()
            if abs_max > 1.05:
                warn(f"Episode {ep_idx}: action |max|={abs_max:.3f} (first 6 dims)")
                episode_ok = False

        # Check images
        top45 = ep["top45"]
        sideview = ep["sideview"]
        if top45.ndim not in (3, 4):
            fail(f"Episode {ep_idx}: top45 shape={top45.shape}")
            episode_ok = False
        elif top45.size > 0 and (top45.min() < 0 or top45.max() > 1.01):
            warn(f"Episode {ep_idx}: top45 value range [{top45.min():.3f}, {top45.max():.3f}]")
            episode_ok = False

        if sideview.size > 0 and (sideview.min() < 0 or sideview.max() > 1.01):
            warn(f"Episode {ep_idx}: sideview value range [{sideview.min():.3f}, {sideview.max():.3f}]")

        # Check length consistency
        lengths = {k: ep[k].shape[0] for k in ["state", "action"]}
        if len(set(lengths.values())) > 1:
            warn(f"Episode {ep_idx}: length mismatch: {lengths}")
            episode_ok = False

        if episode_ok:
            results["ok"] += 1
        else:
            # Check if there were actual failures in recent messages
            recent = msgs[-6:]
            has_fail = any(m.startswith("  FAIL") for m in recent)
            results["fail" if has_fail else "warn"] += 1

        if verbose:
            status_str = "PASS" if episode_ok else "WARN"
            n_frames = ep["action"].shape[0]
            print(f"  Episode {ep_idx}: {status_str}  frames={n_frames}")

    status = "fail" if results["fail"] > 0 else ("warn" if results["warn"] > 0 else "ok")
    return {
        "status": status,
        "messages": msgs,
        "stats": {
            "total_episodes": total_episodes,
            "ok": results["ok"],
            "warn": results["warn"],
            "fail": results["fail"],
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Validate PushBox LeRobot v3 dataset")
    parser.add_argument("--dir", type=str, default="data/demos", help="LeRobot dataset directory")
    parser.add_argument("--quiet", action="store_true", help="only print failures")
    args = parser.parse_args()

    if not HAS_LEROBOT:
        print("[ERROR] lerobot not installed. Please run: pip install lerobot", file=sys.stderr)
        sys.exit(1)

    dataset_dir = Path(args.dir)
    if not dataset_dir.is_dir():
        print(f"[ERROR] directory not found: {dataset_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Validating LeRobot dataset at {dataset_dir}...\n")
    result = check_lerobot_dataset(dataset_dir, verbose=not args.quiet)

    for m in result["messages"]:
        print(m)

    stats = result["stats"]
    print(f"\n--- Summary ---")
    if stats:
        print(
            f"  PASS: {stats['ok']}  WARN: {stats['warn']}  "
            f"FAIL: {stats['fail']}  TOTAL: {stats['total_episodes']}"
        )
    print(f"  Status: {result['status'].upper()}")

    if result["status"] == "fail":
        sys.exit(1)


if __name__ == "__main__":
    main()

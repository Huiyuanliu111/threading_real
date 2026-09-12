#!/usr/bin/env python3
"""Extract a TCP endpoint reference from final measured states in MVT episodes."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import h5py
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from threading_task.kinematics import PandaForwardKinematics
from scripts.deployment.endpoint_schedule import EndpointExecutionSchedule


def build_schedule(dataset: Path, urdf: Path, fine_radius_m=0.06, coarse_steps=10, fine_steps=3):
    fk = PandaForwardKinematics(urdf)
    with h5py.File(dataset, "r") as source:
        if source.attrs.get("format") != "threading-mvt-pointcloud-v1":
            raise ValueError("expected a threading-mvt-pointcloud-v1 dataset")
        keys = sorted(source.keys())
        if not keys:
            raise ValueError("dataset contains no episodes")
        endpoints = np.stack([fk.pose(source[key]["observation_state"][-1, :7])[:3, 3]
                              for key in keys])
        fps = float(source.attrs["fps"])
    center = np.median(endpoints, axis=0)
    EndpointExecutionSchedule(center, fine_radius_m, coarse_steps, fine_steps)
    spread = np.linalg.norm(endpoints - center, axis=1)
    return {
        "frame": "panda_link0", "tcp_frame": "panda_hand_tcp",
        "endpoint_xyz_m": center.tolist(), "fine_radius_m": fine_radius_m,
        "coarse_steps": coarse_steps, "fine_steps": fine_steps,
        "source": {"dataset": str(dataset.resolve()), "urdf": str(urdf.resolve()),
                   "episodes": len(keys), "fps": fps,
                   "method": "coordinatewise median of FK(final measured joint state) across all episodes",
                   "endpoint_spread_m": {"median": float(np.median(spread)),
                                         "p95": float(np.quantile(spread, .95)),
                                         "max": float(spread.max())}},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, default=PROJECT_ROOT.parent /
                        "remote_controller/src/remote_controller/assets/panda/panda_arm.urdf")
    parser.add_argument("--fine-radius-m", type=float, default=0.06)
    parser.add_argument("--coarse-steps", type=int, default=10)
    parser.add_argument("--fine-steps", type=int, default=3)
    args = parser.parse_args()
    config = build_schedule(args.dataset.expanduser(), args.urdf.expanduser(),
                            args.fine_radius_m, args.coarse_steps, args.fine_steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(config, sort_keys=False))
    print(yaml.safe_dump(config, sort_keys=False))


if __name__ == "__main__":
    main()

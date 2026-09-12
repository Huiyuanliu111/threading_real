#!/usr/bin/env python3
"""Compare one MVT ARP dataset point cloud with the current RealSense streams."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from build_threading_mvt_dataset import deterministic_surface_points  # noqa: E402
from build_vla_pointcloud_dataset import depth_to_base_points  # noqa: E402
from scripts.deployment.cartesian import _live_rays  # noqa: E402
from scripts.deployment.joint import (  # noqa: E402
    DEFAULT_FRONTVIEW_SERIAL,
    DEFAULT_SIDEVIEW_SERIAL,
    DEFAULT_WRIST_SERIAL,
    RealSenseRig,
)


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Write a compact binary PLY without an Open3D dependency."""
    vertices = np.empty(
        len(points),
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
               ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    vertices["x"], vertices["y"], vertices["z"] = points.T
    vertices["red"], vertices["green"], vertices["blue"] = colors.T
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as stream:
        stream.write(header)
        vertices.tofile(stream)


def load_dataset_frame(path: Path, episode: int, frame: int) -> tuple[np.ndarray, np.ndarray, dict]:
    with h5py.File(path, "r") as source:
        group_name = f"episode_{episode:06d}"
        if group_name not in source:
            raise KeyError(f"{path} has no {group_name}")
        group = source[group_name]
        frame = frame if frame >= 0 else len(group["points"]) + frame
        if not 0 <= frame < len(group["points"]):
            raise IndexError(f"frame {frame} outside [0, {len(group['points'])})")
        valid = int(group["valid_points"][frame])
        points = np.asarray(group["points"][frame, :valid], dtype=np.float32)
        colors = np.asarray(group["colors"][frame, :valid], dtype=np.uint8)
        bounds = np.asarray(source.attrs["bounds_m"], dtype=np.float32)
        metadata = {
            "source": str(path.resolve()), "episode": episode, "frame": frame,
            "valid_points": valid, "bounds_m": bounds.tolist(),
        }
    return points, colors, metadata


def capture_live(
    calibration_path: Path,
    bounds: np.ndarray,
    max_points: int,
    voxel_m: float,
    sideview_serial: str,
    frontview_serial: str,
    views: tuple[str, ...] = ("sideview", "frontview"),
    capture_frames: int = 1,
) -> tuple[np.ndarray, np.ndarray, dict, dict[str, np.ndarray]]:
    calibration = json.loads(calibration_path.read_text())
    serial_by_view = {"sideview": sideview_serial, "frontview": frontview_serial}
    expected = {view: serial_by_view[view] for view in views}
    for view, serial in expected.items():
        calibrated = str(calibration["cameras"][view].get("serial", ""))
        if calibrated and calibrated != serial:
            raise ValueError(f"{view} calibration serial {calibrated} != requested {serial}")

    rig = RealSenseRig(
        sideview_serial, DEFAULT_WRIST_SERIAL, frontview_serial,
        width=640, height=480, fps=30, enable_depth=True,
        enabled_views=views,
    )
    try:
        depth_counts = {view: [] for view in views}
        rgbd = None
        for _ in range(capture_frames):
            rgbd = rig.read_rgbd(timeout_ms=3000)
            frame_by_view = dict(zip(("sideview", "wrist", "frontview"), rgbd, strict=True))
            for view in views:
                depth_counts[view].append(int(np.count_nonzero(frame_by_view[view][1])))
        assert rgbd is not None
        frame_by_view = dict(zip(("sideview", "wrist", "frontview"), rgbd, strict=True))
        points, colors, per_camera = [], [], {}
        raw_depth_valid = {}
        for index, view in enumerate(rig.view_names):
            rgb, depth = frame_by_view[view]
            xyz, valid = depth_to_base_points(
                depth,
                _live_rays(rig.color_intrinsics[index]),
                rig.depth_scales[index],
                np.asarray(calibration["cameras"][view]["camera_from_base"]),
            )
            points.append(xyz)
            colors.append(rgb[valid])
            per_camera[f"{view}_rgb"] = rgb
            per_camera[f"{view}_depth"] = depth
            raw_depth_valid[view] = int(valid.sum())
        xyz, rgb, valid_count, cropped_count = deterministic_surface_points(
            points, colors, bounds, max_points, voxel_m,
        )
        metadata = {
            "calibration": str(calibration_path.resolve()),
            "serials": expected,
            "raw_depth_valid": raw_depth_valid,
            "stream_depth_valid": {
                view: {
                    "min": int(np.min(counts)),
                    "median": int(np.median(counts)),
                    "max": int(np.max(counts)),
                    "last": int(counts[-1]),
                }
                for view, counts in depth_counts.items()
            },
            "cropped_before_voxel": int(cropped_count),
            "valid_points": int(valid_count),
            "bounds_m": bounds.tolist(),
            "voxel_m": voxel_m,
        }
        return xyz[:valid_count], rgb[:valid_count], metadata, per_camera
    finally:
        rig.close()


def point_size(count: int) -> float:
    return max(0.15, min(1.2, 18000.0 / max(count, 1)))


def plot_cloud(ax, points: np.ndarray, colors: np.ndarray, bounds: np.ndarray,
               title: str, projection: str) -> None:
    stride = max(1, len(points) // 35000)
    pts, rgb = points[::stride], colors[::stride].astype(np.float32) / 255.0
    if projection == "top":
        ax.scatter(pts[:, 0], pts[:, 1], c=rgb, s=point_size(len(pts)), linewidths=0)
        ax.set(xlabel="base X (m)", ylabel="base Y (m)",
               xlim=(bounds[0], bounds[3]), ylim=(bounds[1], bounds[4]))
    elif projection == "front":
        ax.scatter(pts[:, 1], pts[:, 2], c=rgb, s=point_size(len(pts)), linewidths=0)
        ax.set(xlabel="base Y (m)", ylabel="base Z (m)",
               xlim=(bounds[1], bounds[4]), ylim=(bounds[2], bounds[5]))
    else:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=rgb,
                   s=point_size(len(pts)), linewidths=0, depthshade=False)
        ax.set(xlabel="X", ylabel="Y", zlabel="Z",
               xlim=(bounds[0], bounds[3]), ylim=(bounds[1], bounds[4]),
               zlim=(bounds[2], bounds[5]))
        ax.view_init(elev=24, azim=-125)
        try:
            ax.set_box_aspect((bounds[3] - bounds[0], bounds[4] - bounds[1], bounds[5] - bounds[2]))
        except AttributeError:
            pass
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.set_aspect("equal", adjustable="box") if projection != "3d" else None


def save_cloud_figure(path: Path, points: np.ndarray, colors: np.ndarray,
                      bounds: np.ndarray, title: str) -> None:
    fig = plt.figure(figsize=(15, 4.8), constrained_layout=True)
    for index, (projection, label) in enumerate((("top", "Top: X-Y"), ("front", "Front: Y-Z"), ("3d", "3D")), 1):
        ax = fig.add_subplot(1, 3, index, projection="3d" if projection == "3d" else None)
        plot_cloud(ax, points, colors, bounds, f"{label} | {len(points):,} points", projection)
    fig.suptitle(title, fontsize=14)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_comparison(path: Path, dataset: tuple[np.ndarray, np.ndarray],
                    live: tuple[np.ndarray, np.ndarray], bounds: np.ndarray) -> None:
    fig = plt.figure(figsize=(15, 9), constrained_layout=True)
    rows = (("ARP dataset", *dataset), ("Live RealSense", *live))
    for row, (name, points, colors) in enumerate(rows):
        for column, (projection, label) in enumerate((("top", "Top X-Y"), ("front", "Front Y-Z"), ("3d", "3D"))):
            ax = fig.add_subplot(2, 3, row * 3 + column + 1,
                                 projection="3d" if projection == "3d" else None)
            plot_cloud(ax, points, colors, bounds,
                       f"{name} — {label} ({len(points):,})", projection)
    fig.suptitle("ARP input point clouds — identical base-frame bounds and axes", fontsize=15)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=REPO_ROOT / "data/threading_combined_80_mvt_7p5hz.h5")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--calibration", type=Path, default=PROJECT_ROOT / "calibration/block_grasp_spatial.json")
    parser.add_argument("--sideview-serial", default=DEFAULT_SIDEVIEW_SERIAL)
    parser.add_argument("--frontview-serial", default=DEFAULT_FRONTVIEW_SERIAL)
    parser.add_argument(
        "--live-views", nargs="+", choices=("sideview", "frontview"),
        default=("sideview", "frontview"),
        help="RealSense views to fuse; pass one view for single-camera diagnostics",
    )
    parser.add_argument("--capture-frames", type=int, default=1)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts/pointcloud_comparison")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    dataset_points, dataset_colors, dataset_meta = load_dataset_frame(args.dataset, args.episode, args.frame)
    bounds = np.asarray(dataset_meta["bounds_m"], dtype=np.float32)
    live_points, live_colors, live_meta, camera_frames = capture_live(
        args.calibration, bounds, max_points=65536, voxel_m=0.001,
        sideview_serial=args.sideview_serial, frontview_serial=args.frontview_serial,
        views=tuple(args.live_views),
        capture_frames=args.capture_frames,
    )

    write_ply(args.output / "dataset_pointcloud.ply", dataset_points, dataset_colors)
    write_ply(args.output / "live_pointcloud.ply", live_points, live_colors)
    save_cloud_figure(args.output / "dataset_pointcloud.png", dataset_points, dataset_colors,
                      bounds, f"ARP dataset | episode {args.episode}, frame {dataset_meta['frame']}")
    save_cloud_figure(args.output / "live_pointcloud.png", live_points, live_colors,
                      bounds, f"Current RealSense stream | {' + '.join(args.live_views)}")
    save_comparison(args.output / "comparison.png",
                    (dataset_points, dataset_colors), (live_points, live_colors), bounds)
    for view in args.live_views:
        plt.imsave(args.output / f"live_{view}_rgb.png", camera_frames[f"{view}_rgb"])
        depth = camera_frames[f"{view}_depth"].astype(np.float32)
        plt.imsave(args.output / f"live_{view}_depth.png", depth, cmap="turbo", vmin=0, vmax=1500)
    summary = {"dataset": dataset_meta, "live": live_meta}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(args.output.resolve())


if __name__ == "__main__":
    main()

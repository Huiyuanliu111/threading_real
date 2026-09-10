#!/usr/bin/env python3
"""Calibrate fixed cameras by clicking the Panda TCP in synchronized frames.

The robot joint state supplies the 3D base-frame TCP coordinate. A user click
supplies its 2D pixel coordinate. At least six well-spread correspondences are
used to fit a full 3x4 camera projection matrix with normalized DLT.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from threading_task.kinematics import PandaForwardKinematics  # noqa: E402


CAMERA_FEATURES = {
    "sideview": "observation.images.exterior_image_2_right",
    "frontview": "observation.images.exterior_image_1_left",
}


def _normalize_points(points: np.ndarray, target_distance: float) -> tuple[np.ndarray, np.ndarray]:
    dimension = points.shape[1]
    center = points.mean(axis=0)
    centered = points - center
    mean_distance = np.linalg.norm(centered, axis=1).mean()
    if mean_distance < 1e-9:
        raise ValueError("calibration points do not span space")
    scale = target_distance / mean_distance
    transform = np.eye(dimension + 1)
    transform[:dimension, :dimension] *= scale
    transform[:dimension, dimension] = -scale * center
    homogeneous = np.concatenate((points, np.ones((len(points), 1))), axis=1)
    normalized = (transform @ homogeneous.T).T[:, :dimension]
    return normalized, transform


def fit_projection_matrix(points_xyz: np.ndarray, pixels_xy: np.ndarray) -> np.ndarray:
    """Fit a camera matrix using Hartley-normalized direct linear transform."""
    xyz = np.asarray(points_xyz, dtype=np.float64)
    xy = np.asarray(pixels_xy, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or xy.shape != (len(xyz), 2):
        raise ValueError(f"expected Nx3/Nx2 correspondences, got {xyz.shape}/{xy.shape}")
    if len(xyz) < 6:
        raise ValueError("at least six correspondences are required")
    xyz_n, world_transform = _normalize_points(xyz, np.sqrt(3.0))
    xy_n, image_transform = _normalize_points(xy, np.sqrt(2.0))
    world_h = np.concatenate((xyz_n, np.ones((len(xyz_n), 1))), axis=1)
    rows = []
    zeros = np.zeros(4)
    for point, pixel in zip(world_h, xy_n):
        u, v = pixel
        rows.append(np.concatenate((point, zeros, -u * point)))
        rows.append(np.concatenate((zeros, point, -v * point)))
    _, _, vh = np.linalg.svd(np.asarray(rows), full_matrices=False)
    normalized_projection = vh[-1].reshape(3, 4)
    projection = np.linalg.inv(image_transform) @ normalized_projection @ world_transform
    scale = np.linalg.norm(projection[2, :3])
    if scale < 1e-12:
        raise ValueError("degenerate camera calibration")
    projection = projection / scale
    homogeneous = np.concatenate((xyz, np.ones((len(xyz), 1))), axis=1)
    if np.median((projection @ homogeneous.T).T[:, 2]) < 0:
        projection = -projection
    return projection


def reprojection_errors(matrix: np.ndarray, xyz: np.ndarray, xy: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate((xyz, np.ones((len(xyz), 1))), axis=1)
    projected = (matrix @ homogeneous.T).T
    projected = projected[:, :2] / projected[:, 2:3]
    return np.linalg.norm(projected - xy, axis=1)


def farthest_point_rows(points: np.ndarray, count: int) -> np.ndarray:
    """Choose spatially diverse candidate rows instead of adjacent trajectory frames."""
    points = np.asarray(points, dtype=np.float64)
    if count <= 0:
        raise ValueError("candidate count must be positive")
    scale = np.ptp(points, axis=0)
    normalized = (points - points.mean(axis=0)) / np.where(scale > 1e-6, scale, 1.0)
    chosen = [int(np.linalg.norm(normalized, axis=1).argmax())]
    nearest = np.linalg.norm(normalized - normalized[chosen[0]], axis=1)
    for _ in range(1, min(count, len(points))):
        index = int(nearest.argmax())
        chosen.append(index)
        nearest = np.minimum(nearest, np.linalg.norm(normalized - normalized[index], axis=1))
    return np.asarray(chosen, dtype=np.int64)


def _as_rgb_uint8(value: object) -> np.ndarray:
    import torch

    tensor = value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    array = tensor.numpy()
    if array.ndim != 3:
        raise ValueError(f"expected a decoded image, got {array.shape}")
    if array.shape[0] in (1, 3, 4):
        array = np.moveaxis(array[:3], 0, -1)
    else:
        array = array[..., :3]
    if array.dtype != np.uint8:
        if array.max() <= 1.5:
            array = array * 255
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def _opencv_has_gui() -> bool:
    return "GUI:                           NONE" not in cv2.getBuildInformation()


def _collect_clicks_opencv(
    dataset: object,
    rows: np.ndarray,
    feature: str,
    camera: str,
) -> list[dict]:
    annotations: list[dict] = []
    window = f"calibrate {camera}: click TCP | n skip | q finish"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    try:
        for sample_index in rows:
            rgb = _as_rgb_uint8(dataset[int(sample_index)][feature])
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            selected: list[tuple[float, float]] = []

            def callback(event, x, y, _flags, _param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    selected[:] = [(float(x), float(y))]

            cv2.setMouseCallback(window, callback)
            while True:
                display = bgr.copy()
                if selected:
                    cv2.drawMarker(
                        display,
                        tuple(map(int, selected[0])),
                        (0, 0, 255),
                        cv2.MARKER_CROSS,
                        16,
                        2,
                    )
                cv2.putText(
                    display,
                    f"row {sample_index} accepted {len(annotations)}",
                    (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    1,
                )
                cv2.imshow(window, display)
                key = cv2.waitKey(30) & 0xFF
                if selected and key in (13, 32):
                    annotations.append({"index": int(sample_index), "pixel": list(selected[0])})
                    break
                if key == ord("n"):
                    break
                if key == ord("q"):
                    return annotations
    finally:
        cv2.destroyWindow(window)
    return annotations


def _collect_clicks_matplotlib(
    dataset: object,
    rows: np.ndarray,
    feature: str,
    camera: str,
) -> list[dict]:
    import matplotlib.pyplot as plt

    annotations: list[dict] = []
    figure, axis = plt.subplots(num=f"calibrate {camera}")
    try:
        for sample_index in rows:
            rgb = _as_rgb_uint8(dataset[int(sample_index)][feature])
            selected: list[tuple[float, float]] = []
            action = {"value": None}
            axis.clear()
            axis.imshow(rgb)
            axis.set_title(
                f"{camera} row={sample_index} accepted={len(annotations)}\n"
                "left click TCP; Enter/Space accept; n skip; q finish"
            )
            axis.set_axis_off()

            def on_click(event):
                if event.inaxes is not axis or event.xdata is None or event.ydata is None:
                    return
                selected[:] = [(float(event.xdata), float(event.ydata))]
                for artist in list(axis.lines):
                    artist.remove()
                axis.plot(event.xdata, event.ydata, marker="+", color="red", markersize=16)
                figure.canvas.draw_idle()

            def on_key(event):
                if event.key in ("enter", " ") and selected:
                    action["value"] = "accept"
                elif event.key == "n":
                    action["value"] = "skip"
                elif event.key == "q":
                    action["value"] = "quit"

            click_id = figure.canvas.mpl_connect("button_press_event", on_click)
            key_id = figure.canvas.mpl_connect("key_press_event", on_key)
            figure.canvas.draw_idle()
            figure.show()
            while action["value"] is None and plt.fignum_exists(figure.number):
                plt.pause(0.05)
            figure.canvas.mpl_disconnect(click_id)
            figure.canvas.mpl_disconnect(key_id)
            if not plt.fignum_exists(figure.number) or action["value"] == "quit":
                return annotations
            if action["value"] == "accept":
                annotations.append({"index": int(sample_index), "pixel": list(selected[0])})
    finally:
        plt.close(figure)
    return annotations


def collect_clicks(
    dataset: object,
    rows: np.ndarray,
    feature: str,
    camera: str,
    ui: str = "auto",
) -> list[dict]:
    if ui not in {"auto", "opencv", "matplotlib"}:
        raise ValueError(f"unknown UI {ui!r}")
    if ui == "auto":
        ui = "opencv" if _opencv_has_gui() else "matplotlib"
        print(f"[ui] selected {ui}")
    if ui == "opencv":
        if not _opencv_has_gui():
            raise RuntimeError(
                "OpenCV was built without GUI support; use --ui matplotlib"
            )
        return _collect_clicks_opencv(dataset, rows, feature, camera)
    return _collect_clicks_matplotlib(dataset, rows, feature, camera)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cameras", nargs="+", choices=tuple(CAMERA_FEATURES), default=list(CAMERA_FEATURES))
    parser.add_argument("--candidates", type=int, default=60)
    parser.add_argument(
        "--ui",
        choices=("auto", "opencv", "matplotlib"),
        default="auto",
        help="interactive click UI; auto falls back to Matplotlib for headless OpenCV",
    )
    args = parser.parse_args()

    import pyarrow.parquet as pq
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    files = sorted((args.dataset / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files below {args.dataset / 'data'}")
    tables = [pq.read_table(path, columns=["index", "observation.state"]) for path in files]
    indices = np.concatenate(
        [np.asarray(table.column("index").to_pylist(), dtype=np.int64) for table in tables]
    )
    states = np.concatenate(
        [
            np.asarray(table.column("observation.state").to_pylist(), dtype=np.float64)
            for table in tables
        ]
    )
    order = np.argsort(indices)
    indices, states = indices[order], states[order]
    if not np.array_equal(indices, np.arange(len(indices))):
        raise ValueError("dataset global indices must be contiguous")
    tcp = PandaForwardKinematics(args.urdf).positions(states[:, :7])
    candidate_rows = farthest_point_rows(tcp, args.candidates)
    dataset = LeRobotDataset(
        repo_id="local/threading_real_spatial_calibration",
        root=args.dataset,
        download_videos=False,
        video_backend="pyav",
    )

    payload = {"format": "threading-spatial-projection-v1", "cameras": {}}
    for camera in args.cameras:
        feature = CAMERA_FEATURES[camera]
        clicks = collect_clicks(dataset, candidate_rows, feature, camera, ui=args.ui)
        if len(clicks) < 6:
            raise RuntimeError(f"{camera}: only {len(clicks)} clicks; at least six are required")
        rows = np.asarray([item["index"] for item in clicks], dtype=np.int64)
        pixels = np.asarray([item["pixel"] for item in clicks], dtype=np.float64)
        matrix = fit_projection_matrix(tcp[rows], pixels)
        errors = reprojection_errors(matrix, tcp[rows], pixels)
        image = _as_rgb_uint8(dataset[int(rows[0])][feature])
        payload["cameras"][camera] = {
            "feature": feature,
            "image_width": int(image.shape[1]),
            "image_height": int(image.shape[0]),
            "projection_matrix": matrix.tolist(),
            "num_points": len(clicks),
            "reprojection_rmse_px": float(np.sqrt(np.mean(errors**2))),
            "reprojection_median_px": float(np.median(errors)),
            "annotations": clicks,
        }
        print(
            f"[{camera}] points={len(clicks)} "
            f"rmse={np.sqrt(np.mean(errors**2)):.2f}px "
            f"median={np.median(errors):.2f}px"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

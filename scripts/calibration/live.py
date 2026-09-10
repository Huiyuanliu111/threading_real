#!/usr/bin/env python3
"""Live, stationary-pose calibration of fixed RealSense cameras to Panda base.

This tool is observation-only: it starts robot state streaming and camera
streams but never sends a robot motion command. Move the robot with the normal
safe operator interface, wait until the UI reports STABLE, press ``c`` to
freeze a synchronized sample, click the TCP in both views, and press Enter.
"""
from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO_ROOT / "remote_controller" / "src"))

from threading_task.kinematics import PandaForwardKinematics  # noqa: E402


DEFAULT_SERIALS = {
    "sideview": "233722072293",
    "frontview": "233522077069",
}


def save_calibration_progress(
    path: str | Path,
    records: list[dict[str, Any]],
    sideview_serial: str,
    frontview_serial: str,
) -> None:
    """Atomically persist accepted correspondences for crash recovery."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "threading-spatial-live-progress-v1",
        "sideview_serial": sideview_serial,
        "frontview_serial": frontview_serial,
        "records": records,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def load_calibration_progress(
    path: str | Path,
    sideview_serial: str,
    frontview_serial: str,
) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no calibration progress found: {path}")
    progress = json.loads(path.read_text())
    if progress.get("sideview_serial") != sideview_serial:
        raise ValueError("progress sideview serial does not match current camera")
    if progress.get("frontview_serial") != frontview_serial:
        raise ValueError("progress frontview serial does not match current camera")
    records = progress.get("records")
    if not isinstance(records, list):
        raise ValueError(f"{path}: records must be a list")
    return records


class RealSenseColorCamera:
    def __init__(self, serial: str, width: int, height: int, fps: int) -> None:
        import pyrealsense2 as rs

        self.rs = rs
        self.serial = serial
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        profile = self.pipeline.start(config)
        stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsics = stream.get_intrinsics()
        self.width = int(intrinsics.width)
        self.height = int(intrinsics.height)
        self.intrinsic_matrix = np.array(
            [
                [intrinsics.fx, 0.0, intrinsics.ppx],
                [0.0, intrinsics.fy, intrinsics.ppy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.distortion = np.asarray(intrinsics.coeffs, dtype=np.float64)
        self.distortion_model = str(intrinsics.model)

    def read(self, timeout_ms: int = 1000) -> np.ndarray:
        frames = self.pipeline.wait_for_frames(timeout_ms)
        color = frames.get_color_frame()
        if not color:
            raise RuntimeError(f"RealSense {self.serial} returned no color frame")
        return np.asanyarray(color.get_data()).copy()

    def close(self) -> None:
        self.pipeline.stop()


def fit_camera_extrinsics(
    points_base: np.ndarray,
    pixels: np.ndarray,
    intrinsic_matrix: np.ndarray,
    distortion: np.ndarray,
    reprojection_threshold: float = 4.0,
) -> dict[str, Any]:
    """Estimate base-to-camera pose with PnP-RANSAC and LM refinement."""
    points = np.asarray(points_base, dtype=np.float64)
    image_points = np.asarray(pixels, dtype=np.float64)
    intrinsic = np.asarray(intrinsic_matrix, dtype=np.float64)
    distortion = np.asarray(distortion, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or image_points.shape != (len(points), 2):
        raise ValueError(f"expected Nx3/Nx2 points, got {points.shape}/{image_points.shape}")
    if len(points) < 6:
        raise ValueError("at least six live poses are required")
    success, rotation_vector, translation, inliers = cv2.solvePnPRansac(
        points,
        image_points,
        intrinsic,
        distortion,
        iterationsCount=1000,
        reprojectionError=float(reprojection_threshold),
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not success or inliers is None or len(inliers) < 6:
        count = 0 if inliers is None else len(inliers)
        raise RuntimeError(f"PnP-RANSAC failed: only {count}/{len(points)} inliers")
    inlier_rows = inliers[:, 0]
    rotation_vector, translation = cv2.solvePnPRefineLM(
        points[inlier_rows],
        image_points[inlier_rows],
        intrinsic,
        distortion,
        rotation_vector,
        translation,
    )
    rotation = cv2.Rodrigues(rotation_vector)[0]
    projected = cv2.projectPoints(
        points,
        rotation_vector,
        translation,
        intrinsic,
        distortion,
    )[0][:, 0]
    errors = np.linalg.norm(projected - image_points, axis=1)
    camera_points = (rotation @ points.T).T + translation.reshape(1, 3)
    if np.any(camera_points[inlier_rows, 2] <= 0):
        raise RuntimeError("estimated camera pose places calibration points behind the camera")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation[:, 0]
    projection = intrinsic @ transform[:3]
    return {
        "camera_from_base": transform,
        "projection_matrix": projection,
        "errors": errors,
        "inliers": inlier_rows,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("calibration/block_grasp_spatial.json"))
    parser.add_argument(
        "--urdf",
        type=Path,
        default=REPO_ROOT / "remote_controller/src/remote_controller/assets/panda/panda_arm.urdf",
    )
    parser.add_argument("--server-url", default="http://localhost:8008/RPC2")
    parser.add_argument("--udp-ip", default="127.0.0.1")
    parser.add_argument("--udp-port", type=int, default=9000)
    parser.add_argument("--udp-frequency", type=int, default=500)
    parser.add_argument("--sideview-serial", default=DEFAULT_SERIALS["sideview"])
    parser.add_argument("--frontview-serial", default=DEFAULT_SERIALS["frontview"])
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--min-points", type=int, default=10)
    parser.add_argument("--stability-frames", type=int, default=12)
    parser.add_argument("--stability-mm", type=float, default=0.5)
    parser.add_argument("--min-pose-separation-mm", type=float, default=8.0)
    parser.add_argument("--max-rmse-px", type=float, default=4.0)
    progress_mode = parser.add_mutually_exclusive_group()
    progress_mode.add_argument(
        "--resume-progress",
        action="store_true",
        help="resume accepted TCP/pixel pairs from OUTPUT with .progress.json suffix",
    )
    progress_mode.add_argument(
        "--fresh",
        action="store_true",
        help="explicitly overwrite an existing progress file and start from zero",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.sideview_serial == args.frontview_serial:
        raise ValueError("sideview and frontview must be different RealSense devices")
    if args.min_points < 6 or args.stability_frames < 2:
        raise ValueError("min-points must be >=6 and stability-frames must be >=2")

    import matplotlib.pyplot as plt
    from remote_controller.RemoteControllerClient import RemoteControllerClient

    client = RemoteControllerClient(
        args.server_url,
        capacity=max(32, args.stability_frames + 2),
        horizon_prev=2,
        sensor_size=29,
    )
    cameras: dict[str, RealSenseColorCamera] = {}
    figure = None
    progress_path = args.output.with_suffix(".progress.json")
    records: list[dict[str, Any]] = []

    def save_progress() -> None:
        save_calibration_progress(
            progress_path,
            records,
            args.sideview_serial,
            args.frontview_serial,
        )

    try:
        if progress_path.exists() and not args.resume_progress and not args.fresh:
            raise RuntimeError(
                f"existing progress would be overwritten: {progress_path}; use "
                "--resume-progress to continue or --fresh to start over"
            )
        if args.resume_progress:
            if progress_path.is_file():
                records = load_calibration_progress(
                    progress_path,
                    args.sideview_serial,
                    args.frontview_serial,
                )
                print(
                    f"[progress] resumed {len(records)} accepted poses "
                    f"from {progress_path}"
                )
            else:
                print(
                    f"[progress] no existing progress at {progress_path}; "
                    "starting from zero"
                )
        result = client.init(
            udp_ip=args.udp_ip,
            udp_port=args.udp_port,
            udp_frequency_hz=args.udp_frequency,
            recover_before_init=False,
        )
        if result != 0:
            raise RuntimeError(f"remote controller init failed: {result}")
        client.wait_for_first_udp(timeout=3.0)
        cameras = {
            "sideview": RealSenseColorCamera(
                args.sideview_serial, args.width, args.height, args.fps
            ),
            "frontview": RealSenseColorCamera(
                args.frontview_serial, args.width, args.height, args.fps
            ),
        }
        for _ in range(10):
            for camera in cameras.values():
                camera.read(timeout_ms=2000)

        fk = PandaForwardKinematics(args.urdf)
        history: deque[np.ndarray] = deque(maxlen=args.stability_frames)
        state: dict[str, Any] = {
            "mode": "live",
            "quit": False,
            "fatal_error": None,
            "stable": False,
            "latest_tcp": None,
            "latest_frames": None,
            "snapshot_tcp": None,
            "selected": {},
            "markers": {},
        }

        figure, axes = plt.subplots(1, 2, num="Threading live spatial calibration")
        image_artists = {}
        for axis, key in zip(axes, ("sideview", "frontview")):
            image_artists[key] = axis.imshow(
                np.zeros((args.height, args.width, 3), dtype=np.uint8)
            )
            axis.set_axis_off()
        figure.suptitle(
            "Move robot safely; wait for STABLE; c=capture; click TCP in both views; "
            "Enter=accept; r=retry; q=finish"
        )

        def update_titles() -> None:
            status = "STABLE" if state["stable"] else "MOVING"
            mode = str(state["mode"]).upper()
            for axis, key in zip(axes, ("sideview", "frontview")):
                clicked = " clicked" if key in state["selected"] else ""
                axis.set_title(f"{key} | {status} | {mode}{clicked}")
            figure.canvas.draw_idle()

        def clear_markers() -> None:
            for marker in state["markers"].values():
                marker.remove()
            state["markers"] = {}

        def on_click(event) -> None:
            if state["mode"] != "annotate" or event.xdata is None or event.ydata is None:
                return
            for axis, key in zip(axes, ("sideview", "frontview")):
                if event.inaxes is not axis:
                    continue
                old = state["markers"].pop(key, None)
                if old is not None:
                    old.remove()
                state["selected"][key] = [float(event.xdata), float(event.ydata)]
                state["markers"][key] = axis.plot(
                    event.xdata,
                    event.ydata,
                    marker="+",
                    color="red",
                    markersize=16,
                )[0]
                update_titles()

        def on_key(event) -> None:
            if event.key == "q":
                state["quit"] = True
                return
            if event.key == "c" and state["mode"] == "live":
                if not state["stable"]:
                    print("[capture] rejected: robot is not stable")
                    return
                if records:
                    separation = np.linalg.norm(
                        state["latest_tcp"] - np.asarray(records[-1]["tcp_base"])
                    )
                    if separation * 1000 < args.min_pose_separation_mm:
                        print(
                            f"[capture] rejected: only {separation*1000:.2f}mm "
                            "from previous pose"
                        )
                        return
                state["snapshot_tcp"] = state["latest_tcp"].copy()
                state["snapshot_frames"] = {
                    key: value.copy() for key, value in state["latest_frames"].items()
                }
                state["selected"] = {}
                clear_markers()
                state["mode"] = "annotate"
                print(f"[capture] frozen tcp_base={state['snapshot_tcp'].round(5).tolist()}")
                update_titles()
                return
            if event.key == "r" and state["mode"] == "annotate":
                state["selected"] = {}
                clear_markers()
                state["mode"] = "live"
                history.clear()
                update_titles()
                return
            if event.key in ("enter", " ") and state["mode"] == "annotate":
                if set(state["selected"]) != {"sideview", "frontview"}:
                    print("[capture] click TCP once in both views before accepting")
                    return
                records.append(
                    {
                        "tcp_base": state["snapshot_tcp"].tolist(),
                        "pixels": dict(state["selected"]),
                    }
                )
                save_progress()
                print(f"[capture] accepted pose {len(records)}/{args.min_points}")
                state["selected"] = {}
                clear_markers()
                state["mode"] = "live"
                history.clear()
                update_titles()

        def guard_callback(callback):
            """Make GUI callback failures visible to the main recovery path."""

            def guarded(event) -> None:
                try:
                    callback(event)
                except BaseException as error:
                    state["fatal_error"] = error
                    state["quit"] = True
                    print(
                        f"[error] UI callback failed: {type(error).__name__}: {error}",
                        file=sys.stderr,
                    )

            return guarded

        figure.canvas.mpl_connect("button_press_event", guard_callback(on_click))
        figure.canvas.mpl_connect("key_press_event", guard_callback(on_key))
        figure.show()

        while not state["quit"] and plt.fignum_exists(figure.number):
            if state["mode"] == "live":
                frames = {key: camera.read() for key, camera in cameras.items()}
                robot_state, info = client.get_latest_state(allow_stale=False)
                if robot_state is None:
                    print(f"[state] waiting for fresh robot state: {info}")
                    plt.pause(0.03)
                    continue
                tcp = fk.pose(np.asarray(robot_state["q"], dtype=np.float64))[:3, 3]
                history.append(tcp)
                stable = False
                if len(history) == history.maxlen:
                    samples = np.stack(history)
                    radius = np.linalg.norm(samples - samples.mean(axis=0), axis=1).max()
                    stable = radius * 1000 <= args.stability_mm
                state["stable"] = stable
                state["latest_tcp"] = tcp
                state["latest_frames"] = frames
                for key in ("sideview", "frontview"):
                    image_artists[key].set_data(frames[key])
                update_titles()
            plt.pause(0.03)

        if state["fatal_error"] is not None:
            raise state["fatal_error"]
        if len(records) < args.min_points:
            raise RuntimeError(
                f"only {len(records)} accepted poses; need at least {args.min_points}"
            )
        points = np.asarray([record["tcp_base"] for record in records])
        payload: dict[str, Any] = {
            "format": "threading-spatial-projection-v1",
            "method": "live-stationary-pnp-ransac",
            "tcp_frame": "panda_hand_tcp",
            "cameras": {},
        }
        failures = []
        for key, camera in cameras.items():
            pixels = np.asarray([record["pixels"][key] for record in records])
            fitted = fit_camera_extrinsics(
                points,
                pixels,
                camera.intrinsic_matrix,
                camera.distortion,
                reprojection_threshold=args.max_rmse_px,
            )
            errors = fitted["errors"]
            inliers = fitted["inliers"]
            rmse = float(np.sqrt(np.mean(errors[inliers] ** 2)))
            median = float(np.median(errors[inliers]))
            print(
                f"[{key}] inliers={len(inliers)}/{len(points)} "
                f"rmse={rmse:.2f}px median={median:.2f}px"
            )
            if rmse > args.max_rmse_px:
                failures.append(f"{key} RMSE {rmse:.2f}px > {args.max_rmse_px:.2f}px")
            payload["cameras"][key] = {
                "serial": camera.serial,
                "image_width": camera.width,
                "image_height": camera.height,
                "intrinsic_matrix": camera.intrinsic_matrix.tolist(),
                "distortion_model": camera.distortion_model,
                "distortion_coefficients": camera.distortion.tolist(),
                "camera_from_base": fitted["camera_from_base"].tolist(),
                "projection_matrix": fitted["projection_matrix"].tolist(),
                "num_points": len(points),
                "num_inliers": len(inliers),
                "reprojection_rmse_px": rmse,
                "reprojection_median_px": median,
                "inlier_indices": inliers.tolist(),
            }
        payload["records"] = records
        if failures:
            failed_path = args.output.with_suffix(".failed.json")
            failed_path.parent.mkdir(parents=True, exist_ok=True)
            failed_path.write_text(json.dumps(payload, indent=2) + "\n")
            raise RuntimeError(
                "; ".join(failures) + f"; diagnostics saved to {failed_path}"
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"saved valid calibration to {args.output}")
        print(f"accepted-point backup retained at {progress_path}")
        return 0
    except BaseException:
        if records:
            save_progress()
            print(
                f"[progress] preserved {len(records)} accepted poses at {progress_path}; "
                "restart with --resume-progress",
                file=sys.stderr,
            )
        raise
    finally:
        if figure is not None:
            import matplotlib.pyplot as plt

            plt.close(figure)
        for camera in cameras.values():
            camera.close()
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())

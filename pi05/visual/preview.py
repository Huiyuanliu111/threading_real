#!/usr/bin/env python3
"""Select fixed ROIs on raw cam1/cam3 videos and export annotated previews."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from threading_real.pi05.visual.crop import CAMERAS, load, save, transform, validate


def read_frame(path, index):
    import cv2
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open {path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"Cannot decode {path} frame {index}")
        return frame
    finally:
        capture.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True, help="raw episode containing cam1.mp4 and cam3.mp4")
    parser.add_argument("--frames", type=int, nargs="+", default=[0], help="raw frame indices to preview; select on first")
    parser.add_argument("--config", type=Path, help="preview existing config")
    parser.add_argument("--save-config", type=Path, help="new config destination; refuses overwrite")
    parser.add_argument("--select", action="store_true", help="interactive OpenCV ROI selection (requires desktop OpenCV)")
    parser.add_argument("--cam1-roi", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--cam3-roi", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--output", type=Path, required=True, help="new preview directory")
    args = parser.parse_args()
    if any(index < 0 for index in args.frames):
        parser.error("frame indices must be nonnegative")
    if args.config and (args.select or args.cam1_roi or args.cam3_roi or args.save_config):
        parser.error("--config is preview-only; cannot combine with ROI creation options")
    if args.select and (args.cam1_roi or args.cam3_roi):
        parser.error("choose either interactive selection or explicit ROIs")
    if not args.config and not args.select and not (args.cam1_roi and args.cam3_roi):
        parser.error("provide --config, --select, or both --cam1-roi and --cam3-roi")
    if not args.config and not args.save_config:
        parser.error("ROI creation requires --save-config")
    if args.save_config and args.save_config.exists():
        raise FileExistsError(args.save_config)
    if args.output.exists():
        raise FileExistsError(args.output)
    import cv2
    config = load(args.config) if args.config else {"version": 1, "output_size": 224, "resize": "letterbox", "cameras": {}}
    if not args.config:
        for camera in CAMERAS:
            frame = read_frame(args.episode / f"{camera}.mp4", args.frames[0])
            roi = getattr(args, f"{camera}_roi")
            if args.select:
                # Display native pixels: ROI coordinates are never taken from a 224px thumbnail.
                title = f"{camera}: include needle and block approach; Enter confirms"
                x, y, w, h = cv2.selectROI(title, frame, showCrosshair=True, fromCenter=False)
                cv2.destroyAllWindows()
                roi = [int(x), int(y), int(x + w), int(y + h)]
            config["cameras"][camera] = {"source_size": [frame.shape[1], frame.shape[0]], "roi": roi}
        validate(config)
    # Validate every selected frame before writing config or previews.
    for index in args.frames:
        for camera in CAMERAS:
            transform(read_frame(args.episode / f"{camera}.mp4", index), camera, config)
    args.output.mkdir(parents=True)
    for index in args.frames:
        for camera in CAMERAS:
            frame = read_frame(args.episode / f"{camera}.mp4", index)
            cropped = transform(frame, camera, config)
            full_config = copy.deepcopy(config)
            full_config["cameras"][camera]["roi"] = [0, 0, frame.shape[1], frame.shape[0]]
            full = transform(frame, camera, full_config)
            x0, y0, x1, y1 = config["cameras"][camera]["roi"]
            outlined = frame.copy()
            cv2.rectangle(outlined, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 0), 2)
            for suffix, image in (("raw", frame), ("roi", outlined), ("full224", full), ("input224", cropped)):
                path = args.output / f"{camera}_{index:06d}_{suffix}.png"
                if not cv2.imwrite(str(path), image):
                    raise OSError(f"Cannot write {path}")
    if args.save_config:
        args.save_config.parent.mkdir(parents=True, exist_ok=True)
        save(config, args.save_config)
    save(config, args.output / "visual_preprocessing.json")
    print(f"Previews: {args.output}")


if __name__ == "__main__":
    main()

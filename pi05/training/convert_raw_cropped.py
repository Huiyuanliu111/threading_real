#!/usr/bin/env python3
"""Use the shared raw converter with pi0.5-only pre-resize cropping."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import convert_vla_to_lerobot_v3 as converter
from threading_real.pi05.visual.crop import load, make_reader, save, METADATA_FILE


def main():
    parser = converter.build_parser()
    parser.add_argument("--visual-config", type=Path, required=True)
    args = parser.parse_args()
    config = load(args.visual_config)
    if args.fps <= 0 or args.max_camera_skew_ms <= 0 or args.max_robot_skew_ms <= 0:
        raise ValueError("fps and timestamp skew limits must be positive")
    if args.image_size != config["output_size"] or not args.skip_depth or args.camera:
        raise ValueError("Cropped conversion requires --image-size 224 --skip-depth and default cam1/cam3 mapping")
    original = converter.SequentialVideoReader
    converter.SequentialVideoReader = make_reader(original, config)
    try:
        report = converter.convert(args)
    finally:
        converter.SequentialVideoReader = original
    save(config, Path(report["output"]) / "meta" / METADATA_FILE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

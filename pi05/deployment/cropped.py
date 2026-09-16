#!/usr/bin/env python3
"""Deploy a cropped-image pi0.5 checkpoint using its saved visual metadata."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from threading_real.pi05.visual.crop import load, make_observation_adapter, METADATA_FILE


def main():
    from threading_real.scripts.deployment import cartesian

    args = cartesian.build_parser().parse_args()
    config = load(args.checkpoint / METADATA_FILE)
    model = json.loads((args.checkpoint / "config.json").read_text())
    if model.get("type") != "pi05" or args.policy_kind not in ("auto", "pi05"):
        raise ValueError("This entry point only supports pi05 checkpoints")
    if args.pre_resize_image_size is not None or args.image_size not in (None, 224):
        raise ValueError("Crop deployment requires 224px input without pre-resize")
    if args.chunk_selector is not None:
        raise ValueError("First crop experiment uses fixed chunks; the old selector was trained on different images")
    sizes = {tuple(camera["source_size"]) for camera in config["cameras"].values()}
    if len(sizes) != 1:
        raise ValueError("The existing camera rig requires cam1 and cam3 to share a raw resolution")
    width, height = sizes.pop()
    original = cartesian.make_observation
    original_rig = cartesian.RealSenseRig
    def configured_rig(*positional, **kwargs):
        kwargs.update(width=width, height=height)
        return original_rig(*positional, **kwargs)
    cartesian.make_observation = make_observation_adapter(original, config)
    cartesian.RealSenseRig = configured_rig
    print(f"[crop] checkpoint ROIs: {config['cameras']}; letterbox to 224px", flush=True)
    try:
        return cartesian.main()
    finally:
        cartesian.make_observation = original
        cartesian.RealSenseRig = original_rig


if __name__ == "__main__":
    raise SystemExit(main())

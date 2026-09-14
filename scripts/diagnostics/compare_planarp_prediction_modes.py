#!/usr/bin/env python3
"""Offline paired PlanARP decode timing; never connects to a robot.

Full-call measurements include rendering/vision and decoding. Cached-visual
measurements exclude the shared visual encoding, but include all plan generation.
This is a held-out observation benchmark, not a live selector or success trial.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.arp.policy_loader import load_policy  # noqa: E402

MODES = ("full_then_truncate", "required_only")


def summarize(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "max": float(values.max())}


def errors(full, required, steps):
    """Compare relative Cartesian deltas with a proper SO(3) rotation metric."""
    def array(result, key):
        return result[key].detach().cpu().double().numpy()

    a, b = (array(result, "action_pred")[:, :steps] for result in (full, required))
    ra = Rotation.from_rotvec(a[..., 3:6].reshape(-1, 3))
    rb = Rotation.from_rotvec(b[..., 3:6].reshape(-1, 3))
    ca, cb = (array(result, "target_control_points")[:, :steps]
              for result in (full, required))
    result = {
        "translation_delta_error_m": summarize(np.linalg.norm(a[..., :3] - b[..., :3], axis=-1)),
        "rotation_delta_geodesic_error_rad": summarize((ra.inv() * rb).magnitude()),
        "control_point_error_m": summarize(np.linalg.norm(ca - cb, axis=-1)),
        "gripper_delta_abs_error": summarize(np.abs(a[..., 6] - b[..., 6])),
    }
    if "plan_control_points" in full:
        result["plan_control_point_error_m"] = summarize(np.linalg.norm(
            array(full, "plan_control_points") - array(required, "plan_control_points"), axis=-1))
    return result


def timed_call(call, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        with torch.cuda.device(device):
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            wall_start = time.perf_counter()
            start.record()
            result = call()
            end.record()
            end.synchronize()
            wall_ms = (time.perf_counter() - wall_start) * 1000
            elapsed_ms = start.elapsed_time(end)
        return result, {"cuda_event_ms": elapsed_ms, "synchronized_wall_ms": wall_ms}
    start = time.perf_counter()
    result = call()
    return result, {"wall_ms": (time.perf_counter() - start) * 1000}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, help="Override dataset path stored in checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="JSON output file")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--requested-steps", nargs="+", type=int, default=[3, 10])
    parser.add_argument("--warmup", type=int, default=1, help="Unmeasured calls per mode/length/boundary")
    args = parser.parse_args()
    if args.samples < 1 or args.warmup < 0:
        parser.error("samples must be positive and warmup nonnegative")
    if not args.checkpoint.is_file():
        parser.error("checkpoint must be an existing file")
    return args


@torch.inference_mode()
def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(42)
    np.random.seed(42)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = payload["cfg"]
    dataset_config = OmegaConf.create(OmegaConf.to_container(config.task.dataset, resolve=True))
    del payload, config
    if args.dataset:
        dataset_config.dataset_path = str(args.dataset.expanduser().resolve())
    dataset = hydra.utils.instantiate(dataset_config).get_validation_dataset()
    if len(dataset) < args.samples:
        raise ValueError(f"requested {args.samples} observations but validation split has {len(dataset)}")
    policy = load_policy(args.checkpoint, device=str(device), weights="model")
    if any(not 1 <= h <= policy.horizon for h in args.requested_steps):
        raise ValueError(f"requested steps must lie in [1, {policy.horizon}]")
    indices = np.linspace(0, len(dataset) - 1, args.samples, dtype=int).tolist()
    rows = []
    pair_index = 0
    for sample_number, index in enumerate(indices):
        sample = dataset[index]
        obs = {key: value.unsqueeze(0).to(device) for key, value in sample["obs"].items()}
        visual = policy._visual(obs)
        episode, frame = dataset.sample_indices[index]
        for h in args.requested_steps:
            for boundary in ("complete_policy", "cached_visual_plan_and_action"):
                kwargs = {} if boundary == "complete_policy" else {"visual_features": visual}
                # Warm up each measured shape and both modes before the first sample.
                if sample_number == 0:
                    for _ in range(args.warmup):
                        for mode in MODES:
                            policy.predict_action(obs, prediction_mode=mode, requested_steps=h, **kwargs)
                order = MODES if pair_index % 2 == 0 else MODES[::-1]
                predictions, timing, diagnostics = {}, {}, {}
                for mode in order:
                    prediction, measurement = timed_call(
                        lambda mode=mode: policy.predict_action(
                            obs, prediction_mode=mode, requested_steps=h, **kwargs), device)
                    # Move results out of the GPU outside the timing boundary.
                    predictions[mode] = {
                        key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                        for key, value in prediction.items()
                    }
                    timing[mode] = measurement
                    diagnostics[mode] = prediction["prediction_diagnostics"]
                    del prediction
                row = {
                    "pair_index": pair_index, "validation_index": index,
                    "episode": dataset.episode_keys[episode], "frame": frame,
                    "requested_steps": h, "timing_boundary": boundary,
                    "order": list(order), "timing": timing, "diagnostics": diagnostics,
                    "prefix_errors": errors(predictions[MODES[0]], predictions[MODES[1]], h),
                }
                rows.append(row)
                pair_index += 1
                print(json.dumps(row), flush=True)
        del visual, obs
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset": dataset.dataset_path, "device": str(device),
        "torch_version": torch.__version__, "sample_indices": indices,
        "warmup_calls_per_mode_length_boundary": args.warmup,
        "scope": "Offline held-out observations; no selector, robot execution, safety checks or success evaluation.",
        "timing_notes": "Complete policy includes vision; cached visual excludes its common encoding and includes plans. CUDA events explicitly synchronized; CPU uses perf_counter. Output copies and comparison are excluded.",
        "pairs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved {len(rows)} paired measurements to {args.output}")


if __name__ == "__main__":
    main()

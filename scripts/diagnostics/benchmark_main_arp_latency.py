#!/usr/bin/env python3
"""Offline generate-only benchmark for the deployed Threading H20 / Maze H10.

Short lengths are experimental partial action groups, not deployment API modes.
Capture the production generate inputs once; keep all plan tokens and shorten
only the action suffix and its heatmap feature context. No robot connection.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "maze_real"))
sys.path.insert(0, str(ROOT / "threading_real"))
from scripts.arp.policy_loader import load_policy as load_threading
from maze_policy.checkpoint import load_policy as load_maze
from maze_policy.dataset import MazeMVTDataset

CHECKPOINTS = {
    "threading": "training_runs/planarp_chunks_20260914_165545/threading_planarp_chunk20/checkpoints/epoch=0008-val_loss=10.981.ckpt",
    "maze": "maze_real/outputs/maze_planarp_train49_7p5hz/checkpoints/epoch_0209.pt",
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=CHECKPOINTS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--batches", type=int, default=12)
    parser.add_argument("--calls", type=int, default=50)
    parser.add_argument("--lengths", type=int, nargs="+")
    parser.add_argument("--dataset", type=Path, help="Override a relocated dataset path")
    args = parser.parse_args()
    if min(args.batches, args.calls) < 1 or args.warmup < 0:
        parser.error("batches/calls must be positive; warmup must be nonnegative")
    torch.manual_seed(42)
    np.random.seed(42)
    checkpoint = ROOT / CHECKPOINTS[args.task]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if args.task == "threading":
        dataset_cfg = OmegaConf.to_container(payload["cfg"].task.dataset, resolve=True)
        if args.dataset:
            dataset_cfg["dataset_path"] = str(args.dataset.resolve())
        dataset = hydra.utils.instantiate(dataset_cfg)
        policy = load_threading(checkpoint, device="cuda:0", weights="model")
        tokens_per_step = 6
    else:
        launch = json.loads((checkpoint.parent.parent / "launch.json").read_text())
        dataset_cfg = dict(dataset_path=payload["dataset"]["path"],
                           horizon=payload["policy_config"]["horizon"],
                           val_ratio=launch["arguments"]["val_ratio"],
                           seed=launch["arguments"]["seed"])
        if args.dataset:
            dataset_cfg["dataset_path"] = str(args.dataset.resolve())
        dataset = MazeMVTDataset(**dataset_cfg)
        policy = load_maze(checkpoint, device="cuda:0", weights="model")
        tokens_per_step = 2
    del payload
    assert policy.action_chunk_size == policy.horizon
    lengths = sorted(set(args.lengths or ([8, 20] if args.task == "threading" else [4, 10])))
    if any(not 1 <= k <= policy.horizon for k in lengths):
        parser.error("lengths must lie within the saved horizon")
    obs = {k: v.unsqueeze(0).cuda() for k, v in dataset[0]["obs"].items()}
    visual = policy._visual(obs)
    original_generate = policy.policy.generate
    captured = {}

    def capture(prompt, future, **kwargs):
        captured.update(prompt=prompt, future=future, kwargs=kwargs)
        return original_generate(prompt, future, **kwargs)

    policy.policy.generate = capture
    try:
        baseline = policy.predict_action(obs, visual_features=visual, requested_steps=policy.horizon)
    finally:
        policy.policy.generate = original_generate
    if args.task == "threading":
        native = policy.predict_action(obs, visual_features=visual,
                                       requested_steps=min(lengths), prediction_mode="required_only")
        native_behavior = native["prediction_diagnostics"]
        del native
    else:
        native_behavior = "predict_action accepts only full_then_truncate"
    del baseline
    plan_tokens = policy.plan_steps * tokens_per_step
    inputs, checks = {}, {}
    for k in lengths:
        future = captured["future"][:plan_tokens + k * tokens_per_step]
        contexts = dict(captured["kwargs"]["contexts"])
        action_chunk = str(future[-1]["chk_id"])
        contexts["action-predict-featmap"] = {
            action_chunk: captured["kwargs"]["contexts"]["action-predict-featmap"][action_chunk][:k * tokens_per_step]
        }
        inputs[k] = (future, {**captured["kwargs"], "contexts": contexts})

    def generate(k):
        future, kwargs = inputs[k]
        return original_generate(captured["prompt"], future, **kwargs)

    full = original_generate(captured["prompt"], captured["future"], **captured["kwargs"])
    for k in lengths:
        result = generate(k)
        assert result.shape[1] == tokens_per_step + plan_tokens + k * tokens_per_step
        assert torch.isfinite(result).all()
        prefix = full[:, :result.shape[1]]
        checks[k] = {"shape": list(result.shape),
                     "prefix_max_abs_token_difference": float((prefix - result).abs().max())}
        if k == policy.horizon:
            torch.testing.assert_close(result, full, rtol=0, atol=0)
        for _ in range(args.warmup):
            generate(k)
    torch.cuda.synchronize()
    rows = {k: [] for k in lengths}
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    for batch in range(args.batches):
        order = lengths if batch % 2 == 0 else lengths[::-1]
        for k in order:
            torch.cuda.synchronize()
            wall_start = time.perf_counter()
            start.record()
            for _ in range(args.calls):
                generate(k)
            end.record()
            end.synchronize()
            row = {"batch": batch, "cuda_ms_per_call": start.elapsed_time(end) / args.calls,
                   "synchronized_wall_ms_per_call": (time.perf_counter() - wall_start) * 1000 / args.calls}
            rows[k].append(row)
            print(json.dumps({"task": args.task, "steps": k, **row}), flush=True)
    ep, frame = dataset.sample_indices[0]
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "task": args.task,
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "script_sha256": sha256(__file__), "weights": "model",
        "dataset_config": dataset_cfg, "split": "train", "dataset_index": 0,
        "episode": dataset.episode_keys[ep], "frame": frame,
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "dtype": str(next(policy.parameters()).dtype),
        "torch_threads": torch.get_num_threads(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "horizon": policy.horizon, "action_chunk_size": policy.action_chunk_size,
        "plan_tokens": plan_tokens, "action_tokens_per_step": tokens_per_step,
        "native_required_only_behavior": native_behavior,
        "batch_size": 1, "warmup_per_length": args.warmup,
        "batches": args.batches, "calls_per_batch": args.calls,
        "scope": "Only policy.policy.generate; all coarse plans included; visual encoding, input/context preparation and action postprocessing excluded. Short action groups are experimental and bypass deployment length rules.",
        "checks": checks, "measurements": rows,
        "summary": {k: {"mean_ms": float(np.mean([r["cuda_ms_per_call"] for r in values])),
                        "batch_sd_ms": float(np.std([r["cuda_ms_per_call"] for r in values], ddof=1))}
                    for k, values in rows.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()

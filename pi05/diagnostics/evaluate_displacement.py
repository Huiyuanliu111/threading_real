#!/usr/bin/env python3
"""Compare a π0.5 checkpoint's Cartesian displacement predictions with labels.

The diagnostic evaluates every complete action window in the selected training
or validation episodes. It reads only the offline LeRobot dataset and never
connects to the robot.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import resource
import statistics
import time

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05 import PI05Policy
from torch.utils.data import DataLoader, Subset


def describe(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values)
    if not values.size:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def displacement_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    horizon: int,
) -> dict[str, object]:
    prediction_xyz = np.asarray(prediction[:, :horizon, :3], dtype=np.float64) * 1000
    target_xyz = np.asarray(target[:, :horizon, :3], dtype=np.float64) * 1000
    if not len(prediction_xyz):
        return {"windows": 0}

    predicted_vector = prediction_xyz.sum(axis=1)
    target_vector = target_xyz.sum(axis=1)
    predicted_norm = np.linalg.norm(predicted_vector, axis=-1)
    target_norm = np.linalg.norm(target_vector, axis=-1)
    active_threshold = 0.1 * horizon
    active = target_norm > active_threshold
    stationary = ~active
    result: dict[str, object] = {
        "windows": int(len(prediction_xyz)),
        "predicted_net_mm": describe(predicted_norm),
        "target_net_mm": describe(target_norm),
        "signed_magnitude_difference_mm": describe(predicted_norm - target_norm),
        "endpoint_vector_error_mm": describe(
            np.linalg.norm(predicted_vector - target_vector, axis=-1)
        ),
        "predicted_path_mm": describe(np.linalg.norm(prediction_xyz, axis=-1).sum(1)),
        "target_path_mm": describe(np.linalg.norm(target_xyz, axis=-1).sum(1)),
        "signed_xyz_bias_mm": (predicted_vector - target_vector).mean(0).tolist(),
        "xyz_mae_mm": np.abs(predicted_vector - target_vector).mean(0).tolist(),
        "ratio_of_mean_magnitudes": (
            float(predicted_norm.mean() / target_norm.mean())
            if target_norm.mean() > 0
            else None
        ),
        "active_threshold_mm": active_threshold,
        "active_windows": int(active.sum()),
        "stationary_windows": int(stationary.sum()),
    }
    if active.any():
        cosine = np.clip(
            (predicted_vector[active] * target_vector[active]).sum(-1)
            / np.maximum(predicted_norm[active] * target_norm[active], 1e-12),
            -1,
            1,
        )
        result.update(
            active_magnitude_ratio=describe(predicted_norm[active] / target_norm[active]),
            active_direction_error_deg=describe(np.degrees(np.arccos(cosine))),
            active_projection_ratio=describe(
                (predicted_vector[active] * target_vector[active]).sum(-1)
                / target_norm[active] ** 2
            ),
            active_under_half_fraction=float(
                (predicted_norm[active] < 0.5 * target_norm[active]).mean()
            ),
            active_opposite_direction_fraction=float((cosine < 0).mean()),
        )
    if stationary.any():
        result.update(
            stationary_predicted_net_mm=describe(predicted_norm[stationary]),
            stationary_false_active_fraction=float(
                (predicted_norm[stationary] > active_threshold).mean()
            ),
        )
    return result


def split_episodes(dataset: LeRobotDataset, eval_split: float) -> tuple[list[int], list[int]]:
    if not 0 < eval_split < 1:
        raise ValueError("eval_split must be between zero and one")
    episode_tasks = dataset.meta.episodes["tasks"]
    task_to_episodes: dict[str, list[int]] = {}
    for episode in range(dataset.meta.total_episodes):
        task = episode_tasks[episode][0] if episode_tasks[episode] else ""
        task_to_episodes.setdefault(task, []).append(episode)
    train, validation = [], []
    for episodes in task_to_episodes.values():
        held_out = math.ceil(len(episodes) * eval_split)
        train.extend(episodes[:-held_out])
        validation.extend(episodes[-held_out:])
    return sorted(train), sorted(validation)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_worker_file_descriptors(num_workers: int) -> tuple[int, int]:
    """Avoid exhausting the low SSH-session fd limit during long DataLoader runs."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if num_workers > 0:
        target = min(max(soft, 65_536), hard)
        if target > soft:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
            soft = target
        torch.multiprocessing.set_sharing_strategy("file_system")
    return int(soft), int(hard)


def make_plot(
    report: dict[str, object],
    prediction: np.ndarray,
    target: np.ndarray,
    phases: np.ndarray,
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    horizon = int(report["action_horizon"])
    predicted = np.linalg.norm(prediction[:, :horizon, :3].sum(1), axis=-1) * 1000
    demonstrated = np.linalg.norm(target[:, :horizon, :3].sum(1), axis=-1) * 1000
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].scatter(demonstrated, predicted, c=phases, cmap="viridis", s=7, alpha=0.3)
    upper = max(float(predicted.max()), float(demonstrated.max()), 1.0)
    axes[0, 0].plot([0, upper], [0, upper], "k--", lw=1)
    axes[0, 0].set(
        xlabel=f"Demonstrated {horizon}-step net displacement (mm)",
        ylabel="Predicted (mm)",
        title=f"Every evaluated {report['split']} window",
    )

    positions = np.arange(5)
    phases_report = report["phases"]

    def phase_mean(phase: int, metric: str) -> float:
        value = phases_report[str(phase)][str(horizon)].get(metric, {}).get("mean")
        return float("nan") if value is None else float(value)

    for delta, key, label in (
        (-0.18, "target_net_mm", "Demonstration"),
        (0.18, "predicted_net_mm", "π0.5"),
    ):
        axes[0, 1].bar(
            positions + delta,
            [phase_mean(int(index), key) for index in positions],
            0.36,
            label=label,
        )
    phase_labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    axes[0, 1].set(
        xticks=positions,
        xticklabels=phase_labels,
        ylabel=f"Mean {horizon}-step net displacement (mm)",
        title="Episode progress",
    )
    axes[0, 1].legend()
    axes[1, 0].bar(
        positions,
        [phase_mean(int(index), "endpoint_vector_error_mm") for index in positions],
    )
    axes[1, 0].set(
        xticks=positions,
        xticklabels=phase_labels,
        ylabel="Mean endpoint vector error (mm)",
        title="Displacement and direction error",
    )
    for key, label in (
        ("predicted_mm", "Predicted magnitude"),
        ("target_mm", "Demonstrated magnitude"),
        ("error_mm", "Prediction error"),
    ):
        axes[1, 1].plot(
            [item["step"] for item in report["per_future_step"]],
            [item[key]["mean"] for item in report["per_future_step"]],
            label=label,
        )
    axes[1, 1].set(
        xlabel="Future action index",
        ylabel="Mean per-step displacement / error (mm)",
        title="Complete windows only",
    )
    axes[1, 1].legend()
    figure.suptitle(f"π0.5 checkpoint {Path(report['checkpoint']).parent.name} — {report['split']}")
    figure.tight_layout()
    figure.savefig(output / "displacement.png", dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--repo-id")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--split", choices=["validation", "train"], default="validation")
    parser.add_argument("--eval-split", type=float)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument(
        "--max-windows", type=int, default=0,
        help="evaluate an evenly spaced subset; zero evaluates every complete window",
    )
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0 or args.max_windows < 0:
        parser.error("batch-size must be positive; num-workers and max-windows cannot be negative")

    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error(f"refusing to overwrite existing output: {output}")
    if not (checkpoint / "model.safetensors").is_file():
        parser.error(f"checkpoint must be a pretrained_model directory: {checkpoint}")
    train_config_path = checkpoint / "train_config.json"
    if not train_config_path.is_file():
        parser.error(f"missing training configuration: {train_config_path}")
    train_config = json.loads(train_config_path.read_text(encoding="utf-8"))
    dataset_config = train_config["dataset"]
    dataset_root = (
        args.dataset_root.expanduser().resolve()
        if args.dataset_root is not None
        else Path(dataset_config["root"]).expanduser().resolve()
    )
    repo_id = args.repo_id or dataset_config["repo_id"]
    eval_split = args.eval_split if args.eval_split is not None else float(dataset_config["eval_split"])

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(4)
    nofile_soft, nofile_hard = prepare_worker_file_descriptors(args.num_workers)
    full_dataset = LeRobotDataset(repo_id, root=dataset_root, video_backend="pyav")
    train_episodes, validation_episodes = split_episodes(full_dataset, eval_split)
    selected_episodes = validation_episodes if args.split == "validation" else train_episodes

    policy = PI05Policy.from_pretrained(checkpoint).to(args.device).eval()
    horizon = int(policy.config.chunk_size)
    if horizon < 1:
        raise ValueError("checkpoint action horizon must be positive")
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    delta_timestamps = {"action": [step / full_dataset.fps for step in range(horizon)]}
    evaluated = LeRobotDataset(
        repo_id,
        root=dataset_root,
        episodes=selected_episodes,
        delta_timestamps=delta_timestamps,
        video_backend="pyav",
        return_uint8=True,
    )
    table = evaluated.hf_dataset.select_columns(["episode_index", "frame_index"]).with_format("numpy")[:]
    row_episodes = np.asarray(table["episode_index"], dtype=np.int64)
    row_frames = np.asarray(table["frame_index"], dtype=np.int64)
    episode_lengths = {
        int(episode["episode_index"]): int(episode["length"])
        for episode in full_dataset.meta.episodes
    }
    complete_indices = np.flatnonzero(
        row_frames + horizon <= np.asarray([episode_lengths[int(ep)] for ep in row_episodes])
    )
    complete_windows_available = int(len(complete_indices))
    if args.max_windows and len(complete_indices) > args.max_windows:
        selection = np.linspace(0, len(complete_indices) - 1, args.max_windows, dtype=np.int64)
        complete_indices = complete_indices[selection]
    loader = DataLoader(
        Subset(evaluated, complete_indices.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )

    predictions, targets, episodes, frames = [], [], [], []
    inference_times = []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if "action_is_pad" in batch and bool(batch["action_is_pad"].any()):
                raise RuntimeError("complete-window selection unexpectedly produced padded actions")
            targets.append(batch["action"].numpy())
            episodes.append(batch["episode_index"].numpy())
            frames.append(batch["frame_index"].numpy())
            processed = preprocess(batch)
            inference_started = time.perf_counter()
            prediction = postprocess(policy.predict_action_chunk(processed))
            if str(args.device).startswith("cuda"):
                torch.cuda.synchronize(args.device)
            inference_times.append(time.perf_counter() - inference_started)
            predictions.append(prediction.detach().float().cpu().numpy())
            completed = min((batch_index + 1) * args.batch_size, len(complete_indices))
            if batch_index % 20 == 0 or completed == len(complete_indices):
                print(f"evaluated {completed}/{len(complete_indices)} complete windows", flush=True)

    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    episodes_array = np.concatenate(episodes).astype(np.int64)
    frames_array = np.concatenate(frames).astype(np.int64)
    if len(prediction) != len(complete_indices):
        raise RuntimeError("prediction count does not match selected complete windows")
    if prediction.shape != target.shape or prediction.shape[1:] != (horizon, 7):
        raise RuntimeError(f"unexpected prediction/target shapes: {prediction.shape}, {target.shape}")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise RuntimeError("predictions or targets contain NaN/Inf")

    progress = frames_array / np.maximum(
        np.asarray([episode_lengths[int(ep)] for ep in episodes_array]) - 1,
        1,
    )
    phases = np.minimum((progress * 5).astype(np.int64), 4)
    train_actions = LeRobotDataset(
        repo_id, root=dataset_root, episodes=train_episodes, video_backend="pyav"
    ).hf_dataset.select_columns("action").with_format("numpy")[:]["action"]
    mean_action = np.asarray(train_actions, dtype=np.float32).mean(axis=0)
    mean_prediction = np.broadcast_to(mean_action, target.shape)
    horizons = sorted(set((1, min(4, horizon), horizon)))
    model_file = checkpoint / "model.safetensors"
    report: dict[str, object] = {
        "checkpoint": str(checkpoint),
        "checkpoint_model_sha256": sha256_file(model_file),
        "dataset": str(dataset_root),
        "repo_id": repo_id,
        "split": args.split,
        "eval_split": eval_split,
        "evaluated_episodes": selected_episodes,
        "action_horizon": horizon,
        "complete_windows_available": complete_windows_available,
        "evaluated_windows": int(len(prediction)),
        "exhaustive": int(len(prediction)) == complete_windows_available,
        "seed": args.seed,
        "sampling": "one seeded Gaussian flow sample per observation",
        "runtime": {
            "device": args.device,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "nofile_soft_limit": nofile_soft,
            "nofile_hard_limit": nofile_hard,
            "wall_seconds": time.perf_counter() - started,
            "mean_model_batch_seconds": statistics.fmean(inference_times),
        },
        "global_metrics": {
            str(value): displacement_metrics(prediction, target, value) for value in horizons
        },
        "mean_baseline": {
            str(value): displacement_metrics(mean_prediction, target, value) for value in horizons
        },
        "phases": {},
        "episodes": {},
    }
    for phase in range(5):
        selected = phases == phase
        report["phases"][str(phase)] = {
            str(value): displacement_metrics(prediction[selected], target[selected], value)
            for value in horizons
        }
    for episode in np.unique(episodes_array):
        selected = episodes_array == episode
        report["episodes"][str(int(episode))] = {
            str(value): displacement_metrics(prediction[selected], target[selected], value)
            for value in horizons
        }
    first_target_norm = np.linalg.norm(target[:, 0, :3], axis=-1) * 1000
    report["first_target_magnitude_bins"] = {}
    for low, high in ((0, 0.1), (0.1, 0.5), (0.5, 1), (1, 2), (2, 1000)):
        selected = (first_target_norm >= low) & (first_target_norm < high)
        report["first_target_magnitude_bins"][f"{low}..{high}mm"] = displacement_metrics(
            prediction[selected], target[selected], 1
        )
    report["per_future_step"] = [
        {
            "step": index + 1,
            "predicted_mm": describe(np.linalg.norm(prediction[:, index, :3], axis=-1) * 1000),
            "target_mm": describe(np.linalg.norm(target[:, index, :3], axis=-1) * 1000),
            "error_mm": describe(
                np.linalg.norm(prediction[:, index, :3] - target[:, index, :3], axis=-1) * 1000
            ),
        }
        for index in range(horizon)
    ]

    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        output / "predictions.npz",
        prediction=prediction,
        target=target,
        episodes=episodes_array,
        frames=frames_array,
        progress=progress,
        phases=phases,
    )
    with (output / "windows.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "episode", "frame", "progress", "pred_net_mm", "target_net_mm",
                "magnitude_difference_mm", "vector_error_mm", "pred_dx_mm", "pred_dy_mm",
                "pred_dz_mm", "target_dx_mm", "target_dy_mm", "target_dz_mm",
            ]
        )
        for index in range(len(prediction)):
            predicted_vector = prediction[index, :, :3].sum(0) * 1000
            target_vector = target[index, :, :3].sum(0) * 1000
            writer.writerow(
                [
                    episodes_array[index], frames_array[index], progress[index],
                    np.linalg.norm(predicted_vector), np.linalg.norm(target_vector),
                    np.linalg.norm(predicted_vector) - np.linalg.norm(target_vector),
                    np.linalg.norm(predicted_vector - target_vector),
                    *predicted_vector, *target_vector,
                ]
            )
    make_plot(report, prediction, target, phases, output)
    print(json.dumps(report["global_metrics"], indent=2), flush=True)
    print(f"wrote diagnostic artifacts to {output}", flush=True)


if __name__ == "__main__":
    main()

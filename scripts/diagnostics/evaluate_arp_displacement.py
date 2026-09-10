#!/usr/bin/env python3
"""Compare an ARP checkpoint's Cartesian displacement predictions with labels.

This offline diagnostic evaluates every complete action window in the requested
training or validation split.  It never connects to the robot or changes the
dataset.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.arp.policy_loader import load_policy  # noqa: E402
from threading_task.policy import enable_map_gmm_inference  # noqa: E402


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
    prediction: np.ndarray, target: np.ndarray, valid: np.ndarray, horizon: int
) -> dict[str, object]:
    complete = valid[:, :horizon].all(axis=1)
    prediction_xyz = prediction[complete, :horizon, :3].astype(np.float64) * 1000
    target_xyz = target[complete, :horizon, :3].astype(np.float64) * 1000
    if not len(prediction_xyz):
        return {"windows": 0}

    predicted_vector = prediction_xyz.sum(axis=1)
    target_vector = target_xyz.sum(axis=1)
    predicted_norm = np.linalg.norm(predicted_vector, axis=-1)
    target_norm = np.linalg.norm(target_vector, axis=-1)
    active = target_norm > 0.1 * horizon
    cosine = np.clip(
        (predicted_vector[active] * target_vector[active]).sum(-1)
        / np.maximum(predicted_norm[active] * target_norm[active], 1e-12),
        -1,
        1,
    )
    result: dict[str, object] = {
        "windows": len(prediction_xyz),
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
        "active_threshold_mm": 0.1 * horizon,
        "active_windows": int(active.sum()),
    }
    if active.any():
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
    return result


def make_plot(
    report: dict[str, object],
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    phases: np.ndarray,
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    complete = valid[:, :10].all(1)
    predicted = np.linalg.norm(prediction[complete, :10, :3].sum(1), axis=-1) * 1000
    demonstrated = np.linalg.norm(target[complete, :10, :3].sum(1), axis=-1) * 1000
    axes[0, 0].scatter(demonstrated, predicted, c=phases[complete], cmap="viridis", s=7, alpha=0.3)
    upper = max(predicted.max(), demonstrated.max())
    axes[0, 0].plot([0, upper], [0, upper], "k--", lw=1)
    axes[0, 0].set(
        xlabel="Demonstrated 10-step net displacement (mm)",
        ylabel="Predicted (mm)",
        title=f"Every complete 10-step {report['split']} window",
    )

    positions = np.arange(5)
    phases_report = report["phases"]
    for delta, key, label in (
        (-0.18, "target_net_mm", "Demonstration"),
        (0.18, "predicted_net_mm", "ARP"),
    ):
        axes[0, 1].bar(
            positions + delta,
            [phases_report[str(index)]["10"][key]["mean"] for index in positions],
            0.36,
            label=label,
        )
    phase_labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    axes[0, 1].set(
        xticks=positions,
        xticklabels=phase_labels,
        ylabel="Mean 10-step net displacement (mm)",
        title="Episode progress (not task-stage labels)",
    )
    axes[0, 1].legend()
    axes[1, 0].bar(
        positions,
        [phases_report[str(index)]["10"]["endpoint_vector_error_mm"]["mean"] for index in positions],
    )
    axes[1, 0].set(
        xticks=positions,
        xticklabels=phase_labels,
        ylabel="Mean endpoint vector error (mm)",
        title="Includes displacement and direction error",
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
        title="Padding excluded",
    )
    axes[1, 1].legend()
    figure.suptitle(
        f"ARP {Path(report['checkpoint']).stem} ({report['weights']}) — {report['split']}"
    )
    figure.tight_layout()
    figure.savefig(output / "displacement.png", dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", choices=["ema", "model"], default="ema")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--split", choices=["validation", "train"], default="validation")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    dataset = hydra.utils.instantiate(payload["cfg"].task.dataset)
    if dataset.action_mode != "cartesian_delta" or dataset.temporal_stride != 1:
        raise ValueError("This evaluation requires consecutive Cartesian delta actions")
    dataset.max_validation_sequences = None
    evaluated = dataset.get_validation_dataset() if args.split == "validation" else dataset
    policy = load_policy(str(args.checkpoint), args.device, args.weights, use_checkpoint_config=True)
    enable_map_gmm_inference(policy)
    policy.use_sample = "map"
    if policy.horizon < 20:
        raise ValueError("This evaluation reports 20-step predictions")

    start = policy.n_obs_steps - 1
    predictions, targets, masks = [], [], []
    with torch.inference_mode():
        for batch_index, batch in enumerate(
            DataLoader(evaluated, batch_size=args.batch_size, shuffle=False, num_workers=0)
        ):
            observations = {key: value.to(args.device) for key, value in batch["obs"].items()}
            predictions.append(policy.predict_action(observations)["action_pred"].cpu().numpy())
            targets.append(batch["action"][:, start : start + policy.horizon].numpy())
            masks.append((~batch["action_is_pad"][:, start : start + policy.horizon]).numpy())
            if batch_index % 10 == 0:
                print(
                    f"evaluated {min((batch_index + 1) * args.batch_size, len(evaluated))}/{len(evaluated)} windows",
                    flush=True,
                )

    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    valid = np.concatenate(masks)
    episodes = evaluated.sample_indices[:, 0].astype(int)
    frames = evaluated.sample_indices[:, 1].astype(int) + start
    progress = frames / np.maximum(dataset.episode_lengths[episodes] - 1, 1)
    phases = np.minimum((progress * 5).astype(int), 4)
    expected_mask = dataset.val_mask if args.split == "validation" else dataset.train_mask
    assert len(prediction) == len(evaluated) and set(episodes) == set(np.flatnonzero(expected_mask))
    assert np.isfinite(prediction).all() and np.isfinite(target).all()
    mean = np.concatenate([dataset.actions[index] for index in np.flatnonzero(dataset.train_mask)]).mean(0)
    report: dict[str, object] = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "weights": args.weights,
        "dataset": dataset.dataset_path,
        "split": args.split,
        "observation_steps": policy.n_obs_steps,
        "evaluated_episodes": np.unique(episodes).tolist(),
        "observations": len(prediction),
        "valid_actions": int(valid.sum()),
        "global_metrics": {
            str(horizon): displacement_metrics(prediction, target, valid, horizon)
            for horizon in (1, 10, 20)
        },
        "mean_baseline": {
            str(horizon): displacement_metrics(
                np.broadcast_to(mean, target.shape), target, valid, horizon
            )
            for horizon in (1, 10, 20)
        },
        "phases": {},
        "episodes": {},
    }
    for phase in range(5):
        selected = phases == phase
        report["phases"][str(phase)] = {
            str(horizon): displacement_metrics(prediction[selected], target[selected], valid[selected], horizon)
            for horizon in (1, 10, 20)
        }
    for episode in np.unique(episodes):
        selected = episodes == episode
        report["episodes"][str(episode)] = {
            str(horizon): displacement_metrics(prediction[selected], target[selected], valid[selected], horizon)
            for horizon in (1, 10, 20)
        }
    late_active = (
        (phases == 4)
        & valid[:, :10].all(1)
        & (np.linalg.norm(target[:, :10, :3].sum(1), axis=-1) * 1000 > 1)
    )
    report["late_active_10step"] = displacement_metrics(
        prediction[late_active], target[late_active], valid[late_active], 10
    )
    first_target_norm = np.linalg.norm(target[:, 0, :3], axis=-1) * 1000
    report["first_target_magnitude_bins"] = {}
    for low, high in ((0, 0.1), (0.1, 0.5), (0.5, 1), (1, 2), (2, 1000)):
        selected = (first_target_norm >= low) & (first_target_norm < high)
        report["first_target_magnitude_bins"][f"{low}..{high}mm"] = displacement_metrics(
            prediction[selected], target[selected], valid[selected], 1
        )
    report["per_future_step"] = [
        {
            "step": index + 1,
            "predicted_mm": describe(np.linalg.norm(prediction[valid[:, index], index, :3], axis=-1) * 1000),
            "target_mm": describe(np.linalg.norm(target[valid[:, index], index, :3], axis=-1) * 1000),
            "error_mm": describe(
                np.linalg.norm(
                    prediction[valid[:, index], index, :3] - target[valid[:, index], index, :3], axis=-1
                )
                * 1000
            ),
        }
        for index in range(policy.horizon)
    ]
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(
        args.output / "predictions.npz",
        prediction=prediction,
        target=target,
        valid=valid,
        episodes=episodes,
        frames=frames,
        phases=phases,
    )
    with (args.output / "windows_10step.csv").open("w") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "episode", "frame", "progress", "pred_net_mm", "target_net_mm",
                "magnitude_difference_mm", "vector_error_mm", "pred_dx_mm", "pred_dy_mm",
                "pred_dz_mm", "target_dx_mm", "target_dy_mm", "target_dz_mm",
            ]
        )
        for index in np.flatnonzero(valid[:, :10].all(1)):
            predicted_vector = prediction[index, :10, :3].sum(0) * 1000
            target_vector = target[index, :10, :3].sum(0) * 1000
            writer.writerow(
                [
                    episodes[index], frames[index], progress[index],
                    np.linalg.norm(predicted_vector), np.linalg.norm(target_vector),
                    np.linalg.norm(predicted_vector) - np.linalg.norm(target_vector),
                    np.linalg.norm(predicted_vector - target_vector),
                    *predicted_vector, *target_vector,
                ]
            )
    make_plot(report, prediction, target, valid, phases, args.output)
    print(json.dumps(report["global_metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()

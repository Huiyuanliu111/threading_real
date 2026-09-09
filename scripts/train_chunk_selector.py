#!/usr/bin/env python3
"""Train a ChunkSelector from cached visual features."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from chunk_selector.chunk_dataset import ChunkFeatureDataset, episode_split_indices
from chunk_selector.chunk_selector import ChunkSelector, ChunkSelectorConfig


def _embedding_sizes(path: Path) -> dict[str, int]:
    result = {}
    with h5py.File(path, "r") as h5:
        for dataset_name, config_name in (
            ("camera_ids", "num_cameras"),
            ("time_ids", "max_time_steps"),
            ("spatial_ids", "max_spatial_positions"),
        ):
            values = h5[dataset_name][:]
            maximum = int(values.max()) if values.size else -1
            minimum = int(values.min()) if values.size else -1
            if maximum >= 0 and minimum < 0:
                raise ValueError(
                    f"{dataset_name} mixes missing (-1) and valid ids; normalize the dataset"
                )
            result[config_name] = maximum + 1
    return result


def _selector_inputs(
    batch: dict[str, Any],
    config: ChunkSelectorConfig,
    device: torch.device,
) -> dict[str, torch.Tensor | None]:
    result: dict[str, torch.Tensor | None] = {}
    for batch_name, config_name, output_name in (
        ("camera_ids", "num_cameras", "camera_ids"),
        ("time_ids", "max_time_steps", "time_ids"),
        ("spatial_ids", "max_spatial_positions", "spatial_ids"),
    ):
        result[output_name] = (
            batch[batch_name].to(device, non_blocking=True)
            if getattr(config, config_name)
            else None
        )
    return result


def _macro_f1(predictions: np.ndarray, targets: np.ndarray, num_classes: int) -> float:
    scores = []
    for class_id in range(num_classes):
        true_positive = int(((predictions == class_id) & (targets == class_id)).sum())
        false_positive = int(((predictions == class_id) & (targets != class_id)).sum())
        false_negative = int(((predictions != class_id) & (targets == class_id)).sum())
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        scores.append(
            0.0
            if precision + recall == 0
            else 2.0 * precision * recall / (precision + recall)
        )
    return float(np.mean(scores))


def _run_epoch(
    model: ChunkSelector,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    class_weights: torch.Tensor | None,
    soft_target_temperature: float | None,
    use_target_probabilities: bool,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    chunk_absolute_errors: list[np.ndarray] = []

    context = torch.enable_grad if training else torch.inference_mode
    with context():
        for batch in loader:
            features = batch["features"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            logits = model(
                features,
                **_selector_inputs(batch, model.config, device),
            )
            hard_losses = F.cross_entropy(
                logits,
                labels,
                weight=class_weights,
                reduction="none",
            )
            losses_per_sample = hard_losses
            if use_target_probabilities:
                soft_targets = batch["target_probabilities"].to(device, non_blocking=True)
                has_soft_targets = batch["has_target_probabilities"].to(
                    device, non_blocking=True
                )
                safe_targets = torch.where(
                    torch.isfinite(soft_targets),
                    soft_targets,
                    torch.zeros_like(soft_targets),
                )
                soft_losses = -(safe_targets * F.log_softmax(logits, dim=-1)).sum(dim=-1)
                losses_per_sample = torch.where(
                    has_soft_targets,
                    soft_losses,
                    hard_losses,
                )
                chunk_values = torch.as_tensor(
                    model.candidate_chunks, device=device, dtype=logits.dtype
                )
                predicted_chunks = torch.softmax(logits, dim=-1) @ chunk_values
                target_chunks = safe_targets @ chunk_values
                chunk_absolute_errors.append(
                    (predicted_chunks - target_chunks).abs().detach().cpu().numpy()
                )
            if soft_target_temperature is not None:
                utilities = batch["utilities"].to(device, non_blocking=True)
                has_utilities = batch["has_utilities"].to(device, non_blocking=True)
                safe_utilities = torch.where(
                    torch.isfinite(utilities),
                    utilities,
                    torch.zeros_like(utilities),
                )
                soft_targets = torch.softmax(
                    safe_utilities / soft_target_temperature,
                    dim=-1,
                )
                soft_losses = -(soft_targets * F.log_softmax(logits, dim=-1)).sum(dim=-1)
                losses_per_sample = torch.where(
                    has_utilities,
                    soft_losses,
                    hard_losses,
                )
            sample_weights = batch["sample_weight"].to(
                device,
                non_blocking=True,
            )
            loss = (losses_per_sample * sample_weights).sum() / sample_weights.sum()
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            losses.append(float(loss.detach().cpu()))
            predictions.append(logits.argmax(dim=-1).detach().cpu().numpy())
            targets.append(labels.detach().cpu().numpy())

    all_predictions = np.concatenate(predictions)
    all_targets = np.concatenate(targets)
    metrics = {
        "loss": float(np.mean(losses)),
        "accuracy": float((all_predictions == all_targets).mean()),
        "macro_f1": _macro_f1(
            all_predictions,
            all_targets,
            len(model.candidate_chunks),
        ),
    }
    if chunk_absolute_errors:
        metrics["chunk_mae"] = float(np.concatenate(chunk_absolute_errors).mean())
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path, help="ChunkFeatureWriter HDF5 dataset")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--confidence-threshold", type=float, default=None)
    parser.add_argument("--safe-chunk", type=int, default=None)
    parser.add_argument(
        "--soft-target-temperature",
        type=float,
        default=None,
        help="use utility-derived soft labels where available",
    )
    parser.add_argument(
        "--use-target-probabilities",
        action="store_true",
        help="train from explicit target_probabilities stored in the feature dataset",
    )
    parser.add_argument(
        "--selection-mode",
        choices=("argmax", "expected"),
        default="argmax",
        help="map predicted probabilities to a candidate or their expected chunk",
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="disable inverse-frequency class weighting",
    )
    args = parser.parse_args()

    if args.epochs <= 0 or args.batch_size <= 0 or args.patience <= 0:
        parser.error("epochs, batch-size, and patience must be positive")
    if args.soft_target_temperature is not None and args.soft_target_temperature <= 0:
        parser.error("--soft-target-temperature must be positive")
    if args.use_target_probabilities and args.soft_target_temperature is not None:
        parser.error(
            "--use-target-probabilities and --soft-target-temperature are mutually exclusive"
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset_path = args.dataset.expanduser().resolve()
    dataset = ChunkFeatureDataset(dataset_path)
    if args.use_target_probabilities:
        with h5py.File(dataset_path, "r") as h5:
            if "target_probabilities" not in h5:
                parser.error("dataset does not contain target_probabilities")
            targets = np.asarray(h5["target_probabilities"])
            if not (
                np.isfinite(targets).all()
                and np.all(targets >= 0.0)
                and np.allclose(targets.sum(axis=1), 1.0, atol=1e-5)
            ):
                parser.error("dataset target_probabilities are missing or invalid")
    train_indices, val_indices = episode_split_indices(
        dataset_path,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    train_dataset = ChunkFeatureDataset(dataset_path, train_indices)
    val_dataset = ChunkFeatureDataset(dataset_path, val_indices)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    token_count, feature_dim = dataset.feature_shape
    embedding_sizes = _embedding_sizes(dataset_path)
    config = ChunkSelectorConfig(
        input_dim=feature_dim,
        candidate_chunks=dataset.candidate_chunks,
        d_model=args.d_model,
        num_layers=args.num_layers,
        n_heads=args.n_heads,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_tokens=token_count,
        safe_chunk=args.safe_chunk,
        confidence_threshold=args.confidence_threshold,
        selection_mode=args.selection_mode,
        metadata={
            **dataset.metadata,
            "training_dataset": str(dataset_path),
            "training_seed": args.seed,
            "training_targets": (
                "explicit_probabilities" if args.use_target_probabilities else "hard_labels"
            ),
        },
        **embedding_sizes,
    )
    device = torch.device(args.device)
    model = ChunkSelector(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    class_weights = None
    if not args.no_class_weights:
        with h5py.File(dataset_path, "r") as h5:
            train_labels = np.asarray(h5["labels"])[train_indices]
        counts = np.bincount(train_labels, minlength=len(dataset.candidate_chunks))
        weights = len(train_labels) / np.maximum(counts, 1)
        weights = weights / weights.mean()
        class_weights = torch.as_tensor(weights, device=device, dtype=torch.float32)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            class_weights=class_weights,
            soft_target_temperature=args.soft_target_temperature,
            use_target_probabilities=args.use_target_probabilities,
        )
        val_metrics = _run_epoch(
            model,
            val_loader,
            device=device,
            optimizer=None,
            class_weights=class_weights,
            soft_target_temperature=args.soft_target_temperature,
            use_target_probabilities=args.use_target_probabilities,
        )
        record = {"epoch": epoch, "train": train_metrics, "validation": val_metrics}
        history.append(record)
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.3f} "
            f"val_macro_f1={val_metrics['macro_f1']:.3f}"
            + (
                f" val_chunk_mae={val_metrics['chunk_mae']:.3f}"
                if "chunk_mae" in val_metrics
                else ""
            )
        )
        score = (
            -val_metrics["loss"]
            if args.use_target_probabilities
            else val_metrics["macro_f1"]
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            epochs_without_improvement = 0
            model.eval()
            model.save_pretrained(
                args.output_dir,
                metadata={
                    "best_epoch": str(epoch),
                    "best_validation_loss": str(val_metrics["loss"]),
                    "best_validation_macro_f1": str(val_metrics["macro_f1"]),
                },
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"early stopping after {epoch} epochs")
                break

    (args.output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"best epoch={best_epoch}")
    print(f"selector saved to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

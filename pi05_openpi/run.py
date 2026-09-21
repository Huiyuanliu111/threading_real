#!/usr/bin/env python3
"""Launch official OpenPI norm computation / JAX training with local config."""

import argparse
import importlib
import json
import os
from pathlib import Path
import sys

from transforms import validate_info

HERE = Path(__file__).resolve().parent


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["check", "norm", "train"])
    p.add_argument("--openpi-root", type=Path, default=Path(os.environ.get("OPENPI_ROOT", HERE / "vendor/openpi")))
    p.add_argument("--dataset-root", type=Path, default=HERE / "data/threading_tcp6_nosmooth_30hz")
    p.add_argument("--repo-id", default="threading_real/threading_tcp6_nosmooth_30hz")
    p.add_argument("--exp-name", default="threading_tcp6_30hz_h50_vision_lora_action_full_v6")
    p.add_argument("--mode", choices=["full", "lora", "vision_lora_action_full"], default="vision_lora_action_full")
    p.add_argument("--vision-lora-rank", type=int, default=16)
    p.add_argument("--image-profile", choices=["224", "native640"], default="224",
                   help="native640 preserves 640x480 pixels and only pads to 644x490")
    p.add_argument("--camera-views", choices=["both", "cam1"], default="both")
    p.add_argument("--freeze-vision", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--freeze-language", action=argparse.BooleanOptionalAction, default=True,
                   help="Freeze language backbone including its LoRA; focus updates on action expert")
    p.add_argument("--base-checkpoint", default="gs://openpi-assets/checkpoints/pi05_base")
    p.add_argument("--horizon", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=12, help="Global batch across all visible GPUs")
    p.add_argument("--fsdp-devices", type=int, default=1)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--warmup-steps", type=int, default=250)
    p.add_argument("--learning-rate", type=float, default=2.5e-5)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--save-interval", type=int, default=2000)
    p.add_argument("--eval-interval", type=int, default=1000)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--assets-dir", type=Path, default=HERE / "assets")
    p.add_argument("--checkpoint-dir", type=Path, default=HERE / "checkpoints")
    p.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--print-config", action="store_true", help="No dependency imports, downloads or GPU access")
    return p


def setup(settings):
    root = Path(settings["openpi_root"])
    if not (root / "scripts/train.py").is_file():
        raise FileNotFoundError(f"Missing OpenPI checkout: {root}; run bootstrap.sh")
    os.environ.setdefault("HF_LEROBOT_HOME", str(HERE / ".cache/lerobot"))
    os.environ.setdefault("OPENPI_DATA_HOME", str(HERE / ".cache/openpi"))
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "packages/openpi-client/src"))
    # Import the official scripts as real modules: spawn workers must be able
    # to import compute_norm_stats.RemoveStrings when unpickling the dataset.
    sys.path.insert(0, str(root / "scripts"))
    from config import make_config
    return make_config(settings)


def bind_dataset(settings):
    source = Path(settings["dataset_root"])
    info = json.loads((source / "meta/info.json").read_text())
    validate_info(info)
    if settings.get("image_profile", "224") == "native640":
        from transforms import FRONT, SIDE
        if any(info['features'][key]['shape'] != [480, 640, 3] for key in (FRONT, SIDE)):
            raise ValueError("native640 requires the rebuilt original-resolution dataset, not 224 images")
    if info["codebase_version"] != "v2.1":
        raise ValueError("Export v3 data with export_dataset.py before using the pinned OpenPI reader")
    from lerobot.common.constants import HF_LEROBOT_HOME
    target = Path(HF_LEROBOT_HOME) / settings["repo_id"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        if target.resolve() != source.resolve():
            raise FileExistsError(f"Dataset cache points elsewhere: {target}; choose a different repo-id")
    else:
        target.symlink_to(source, target_is_directory=True)
    return info


def main():
    args = parser().parse_args()
    settings = {key: str(value.expanduser().resolve()) if isinstance(value, Path) else value
                for key, value in vars(args).items() if key not in ("command", "print_config")}
    for key in ("repo_id", "exp_name"):
        value = settings[key]
        if not value or Path(value).is_absolute() or any(part in (".", "..") for part in value.split("/")):
            raise ValueError(f"Invalid {key}")
    if "/" in settings["exp_name"]:
        raise ValueError("exp-name must be one directory name")
    for key in ("horizon", "batch_size", "fsdp_devices", "steps", "save_interval", "eval_interval", "vision_lora_rank"):
        if settings[key] < 1:
            raise ValueError(f"{key} must be positive")
    if not 0 <= settings["warmup_steps"] < settings["steps"]:
        raise ValueError("warmup-steps must be >= 0 and < steps")
    if settings["num_workers"] < 0 or settings["learning_rate"] <= 0:
        raise ValueError("Invalid workers or learning rate")
    if not 0 < settings["val_fraction"] < 1:
        raise ValueError("val-fraction must lie between 0 and 1")
    if settings["steps"] % settings["save_interval"]:
        raise ValueError("steps must be divisible by save-interval")
    if settings["save_interval"] % settings["eval_interval"]:
        raise ValueError("save-interval must be divisible by eval-interval so checkpoints have validation metrics")
    # Stats are isolated by experiment so another dataset/run cannot overwrite them.
    settings["assets_dir"] = str(Path(settings["assets_dir"]) / settings["exp_name"])
    settings["fps"] = 30
    settings["gripper_removed"] = True
    settings["physical_action_dim"] = 6
    settings["physical_state_dim"] = 9
    settings["training_image_augmentation"] = "OpenPI default: front 95% random crop and +/-5deg rotation; both cameras color jitter (brightness=.3, contrast=.4, saturation=.5)"
    if settings["image_profile"] == "native640":
        settings["training_image_augmentation"] = "None: original 640x480 pixels; pad L/R=2, T/B=5 to 644x490; no resize/crop"
    print(json.dumps(settings, indent=2), flush=True)
    if args.print_config:
        return
    config = setup(settings)
    info = bind_dataset(settings)
    from validation import episode_split, compute_norm, validate_norm
    split, _ = episode_split(settings["dataset_root"], settings["val_fraction"], settings["seed"])
    settings["validation_split"] = split
    print(json.dumps({"validation_split": split}, indent=2), flush=True)
    import openpi.training.config as registry
    registry._CONFIGS_DICT[config.name] = config
    import openpi.training.data_loader as loader
    data_config = config.data.create(config.assets_dirs, config.model)
    if args.command == "check":
        dataset = loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
        transformed = loader.transform_dataset(dataset, data_config, skip_norm_stats=True)
        sample = transformed[0]
        print({"frames": len(dataset), "episodes": info["total_episodes"], "fps": info["fps"],
               "chunk_duration_seconds": config.model.action_horizon / info["fps"],
               "state_shape": sample["state"].shape, "action_shape": sample["actions"].shape,
               "image_masks": sample["image_mask"]})
        return
    manifest = config.checkpoint_dir.parent / f"{config.exp_name}.json"
    visual = Path(settings["dataset_root"]) / "meta/visual_preprocessing.json"
    settings["visual_preprocessing"] = json.loads(visual.read_text()) if visual.exists() else None
    processing = Path(settings["dataset_root"]) / "meta/raw_processing.json"
    settings["data_processing"] = (
        {key: value for key, value in json.loads(processing.read_text()).items() if key != "per_episode"}
        if processing.exists() else None
    )
    # Before training starts, machine paths / batch / sharding may change;
    # validate_norm separately enforces the split and action representation.
    if manifest.exists() and config.checkpoint_dir.exists():
        previous = json.loads(manifest.read_text())
        if {k: v for k, v in previous.items() if k != "resume"} != {k: v for k, v in settings.items() if k != "resume"}:
            raise ValueError(f"Run settings differ from {manifest}; use a new exp-name")
    if args.command == "norm":
        if config.checkpoint_dir.exists():
            raise FileExistsError("Do not recompute normalization after training starts; use a new exp-name")
        if info["total_frames"] < config.batch_size:
            raise ValueError("Dataset must contain at least one full batch for official norm computation")
        compute_norm(config, settings)
    else:
        if data_config.norm_stats is None:
            raise FileNotFoundError("Run the norm command with the same settings before training")
        validate_norm(config, settings)
        import jax
        devices = jax.device_count()
        if jax.default_backend() != "gpu":
            raise RuntimeError("Training requires the OpenPI CUDA/JAX environment")
        if devices % config.fsdp_devices or config.batch_size % devices:
            raise ValueError("GPU count must divide global batch; fsdp-devices must divide GPU count")
        if not config.resume and config.checkpoint_dir.exists():
            raise FileExistsError(f"Checkpoint directory already exists: {config.checkpoint_dir}")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(settings, indent=2) + "\n")
    if args.command == "train":
        importlib.import_module("train_validated").main(config, settings)


if __name__ == "__main__":
    main()

"""Load an ARP checkpoint with the policy architecture it was trained with."""
from __future__ import annotations

from pathlib import Path
import sys

import hydra
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pushbox.diffusion_policy.model.common.normalizer import LinearNormalizer


def load_policy(
    checkpoint_dir: str | Path,
    device: str = "cuda:0",
    weights: str = "model",
    use_checkpoint_config: bool = True,
):
    """Return an evaluation-mode ARP policy from a real-robot checkpoint.

    Real Threading checkpoints must include their Hydra policy configuration.
    This avoids reconstructing a policy from obsolete PushBox defaults.
    """
    checkpoint_dir = Path(checkpoint_dir).expanduser()
    if checkpoint_dir.is_file():
        checkpoint_path = checkpoint_dir
    else:
        checkpoints = sorted(checkpoint_dir.glob("*.ckpt"))
        if not checkpoints:
            raise FileNotFoundError(f"No .ckpt files found in {checkpoint_dir}")
        checkpoint_path = checkpoints[0]

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "state_dicts" in checkpoint:
        state_dicts = checkpoint["state_dicts"]
        if weights == "ema" and "ema_model" in state_dicts:
            state_key = "ema_model"
        elif "model" in state_dicts:
            state_key = "model"
        else:
            raise KeyError(f"Checkpoint has no model weights: {sorted(state_dicts)}")
        state_dict = state_dicts[state_key]
    elif "model" in checkpoint:
        state_key = "model"
        state_dict = checkpoint["model"]
    else:
        raise KeyError("Checkpoint has neither state_dicts nor model weights")

    if not use_checkpoint_config or checkpoint.get("cfg") is None:
        raise ValueError(
            "Real Threading ARP checkpoints must contain cfg.policy; "
            "legacy PushBox checkpoint reconstruction was removed."
        )
    config = checkpoint["cfg"]
    if "policy" not in config:
        raise KeyError("Checkpoint cfg has no policy section")
    policy = hydra.utils.instantiate(config.policy)

    cleaned = {}
    for name, value in state_dict.items():
        if name.startswith("ema_model."):
            name = name[len("ema_model.") :]
        elif name.startswith("model."):
            name = name[len("model.") :]
        cleaned[name] = value

    normalizer_state = {
        name[len("normalizer.") :]: value
        for name, value in cleaned.items()
        if name.startswith("normalizer.")
    }
    normalizer = LinearNormalizer()
    normalizer.load_state_dict(normalizer_state)
    policy.set_normalizer(normalizer)
    policy.load_state_dict(cleaned, strict=True)
    policy.to(device).eval()
    print(
        f"[arp] loaded {checkpoint_path} ({state_key}, "
        f"{len(cleaned)}/{len(policy.state_dict())} parameter keys)"
    )
    return policy

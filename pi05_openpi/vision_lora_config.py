"""Pi0 config with checkpoint-compatible SigLIP LoRA and a full action expert."""
import dataclasses

from flax import nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from openpi.models import gemma, pi0_config
from openpi.shared import nnx_utils
from openpi.shared import array_typing as at

import siglip_lora


@dataclasses.dataclass(frozen=True)
class VisionLoRAConfig(pi0_config.Pi0Config):
    vision_lora_rank: int = 16
    vision_lora_alpha: float = 16.0
    image_profile: str = "224"
    camera_views: str = "both"

    def inputs_spec(self, *, batch_size=1):
        observation, actions = super().inputs_spec(batch_size=batch_size)
        if self.image_profile == "native640":
            from native_images import MODEL_HW
            keys = ("left_wrist_0_rgb",) if self.camera_views == "cam1" else tuple(observation.images)
            with at.disable_typechecking():
                observation = dataclasses.replace(observation, images={
                    k: jax.ShapeDtypeStruct((batch_size, *MODEL_HW, 3), jnp.float32)
                    for k in keys
                }, image_masks={k: observation.image_masks[k] for k in keys})
        return observation, actions

    def create(self, rng):
        if self.image_profile == "native640":
            from pi0_native import NativePi0
            model = NativePi0(self, rngs=nnx.Rngs(rng))
        else:
            model = super().create(rng)
        img = nnx_bridge.ToNNX(siglip_lora.Module(
            num_classes=gemma.get_config(self.paligemma_variant).width,
            variant="So400m/14", pool_type="none", scan=True, dtype_mm=self.dtype,
            lora_rank=self.vision_lora_rank, lora_alpha=self.vision_lora_alpha,
            pretrained_grid=(16, 16) if self.image_profile == "native640" else None,
        ))
        img.lazy_init(next(iter(self.fake_obs().images.values())), train=False,
                      rngs=nnx.Rngs(jax.random.fold_in(rng, 601)))
        model.PaliGemma.img = img
        return model

    def get_freeze_filter(self):
        return nnx.Any(
            nnx.All(nnx_utils.PathRegex(r"PaliGemma/img/.*"),
                    nnx.Not(nnx_utils.PathRegex(r".*lora_[ab].*"))),
            nnx.All(nnx_utils.PathRegex(r"PaliGemma/llm/.*"),
                    nnx.Not(nnx_utils.PathRegex(r".*_1.*"))),
        )

"""Extend openpi.training.config without editing the upstream checkout."""

import dataclasses
from flax import nnx

from openpi import transforms as T
from openpi.models import pi0_config
from openpi.shared import nnx_utils
from openpi.training import config as C
from openpi.training import optimizer, weight_loaders

from transforms import FRONT, SIDE, ThreadingInputs, ThreadingOutputs


@dataclasses.dataclass(frozen=True)
class ThreadingDataConfig(C.DataConfigFactory):
    def create(self, assets_dirs, model_config):
        camera_views = getattr(model_config, "camera_views", "both")
        mapping = {"side": SIDE, "state": "observation.state", "actions": "action", "prompt": "prompt"}
        if camera_views == "both":
            mapping["front"] = FRONT
        model_transforms = C.ModelTransformFactory()(model_config)
        if getattr(model_config, "image_profile", "224") == "native640":
            from native_images import PadNativeImages
            model_transforms = dataclasses.replace(model_transforms, inputs=[
                PadNativeImages() if isinstance(t, T.ResizeImages) else t
                for t in model_transforms.inputs
            ])
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=T.Group(inputs=[T.RepackTransform(mapping)]),
            data_transforms=T.Group(
                inputs=[ThreadingInputs(camera_views=camera_views)],
                outputs=[ThreadingOutputs()],
            ),
            model_transforms=model_transforms,
            action_sequence_keys=("action",),
            prompt_from_task=True,
        )


def make_config(settings):
    if settings.get("camera_views", "both") not in ("both", "cam1"):
        raise ValueError("camera_views must be both or cam1")
    if settings.get("camera_views", "both") == "cam1" and settings.get("image_profile", "224") != "native640":
        raise ValueError("cam1-only mode requires native640 image profile")
    lora = settings["mode"] == "lora"
    vision_lora = settings["mode"] == "vision_lora_action_full"
    model_class = pi0_config.Pi0Config
    extra = {}
    if vision_lora:
        from vision_lora_config import VisionLoRAConfig
        model_class = VisionLoRAConfig
        rank = settings.get("vision_lora_rank", 16)
        if rank < 1:
            raise ValueError("vision_lora_rank must be positive")
        if not settings.get("freeze_vision", True) or not settings.get("freeze_language", True):
            raise ValueError("vision_lora_action_full requires frozen vision base and language weights")
        extra = {"vision_lora_rank": rank, "vision_lora_alpha": float(rank),
                 "image_profile": settings.get("image_profile", "224"),
                 "camera_views": settings.get("camera_views", "both")}
    elif settings.get("image_profile", "224") != "224":
        raise ValueError("native640 requires vision_lora_action_full mode")
    model = model_class(
        pi05=True, action_dim=32, action_horizon=settings["horizon"],
        discrete_state_input=True,
        paligemma_variant="gemma_2b_lora" if lora else "gemma_2b",
        action_expert_variant="gemma_300m_lora" if lora else "gemma_300m",
        **extra,
    )
    # Missing flags preserve the architecture and filters of historical manifests.
    frozen = [model.get_freeze_filter()]
    if not vision_lora and settings.get("freeze_vision", False):
        frozen.append(nnx_utils.PathRegex(r"PaliGemma/img/.*"))
    if not vision_lora and settings.get("freeze_language", False):
        frozen.append(nnx.All(
            nnx_utils.PathRegex(r"PaliGemma/llm/.*"),
            nnx.Not(nnx_utils.PathRegex(r".*_1.*")),
        ))
    return C.TrainConfig(
        name=f"pi05_threading_{settings['mode']}", exp_name=settings["exp_name"],
        project_name="threading_pi05_openpi", model=model,
        data=ThreadingDataConfig(repo_id=settings["repo_id"]),
        weight_loader=weight_loaders.CheckpointWeightLoader(settings["base_checkpoint"].rstrip("/") + "/params"),
        freeze_filter=nnx.Any(*frozen), ema_decay=None if lora or vision_lora else 0.99,
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=settings["warmup_steps"], peak_lr=settings["learning_rate"],
            decay_steps=settings["steps"], decay_lr=settings["learning_rate"] / 10,
        ),
        assets_base_dir=settings["assets_dir"], checkpoint_base_dir=settings["checkpoint_dir"],
        batch_size=settings["batch_size"], num_workers=settings["num_workers"],
        num_train_steps=settings["steps"], save_interval=settings["save_interval"],
        keep_period=None, log_interval=20, fsdp_devices=settings["fsdp_devices"], seed=settings["seed"],
        resume=settings["resume"], wandb_enabled=settings["wandb"],
        policy_metadata={"state_representation": "tcp_pose_6d", "fps": 30, "physical_action_dim": 6, "physical_state_dim": 9, "gripper_removed": True,
                         "camera_views": settings.get("camera_views", "both"),
                         "action_names": ["dx", "dy", "dz", "drotvec_x", "drotvec_y", "drotvec_z"]},
    )

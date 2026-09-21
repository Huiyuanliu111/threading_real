"""Extend openpi.training.config without editing the upstream checkout."""

import dataclasses

from openpi import transforms as T
from openpi.models import pi0_config
from openpi.training import config as C
from openpi.training import optimizer, weight_loaders

from transforms import FRONT, SIDE, ThreadingInputs, ThreadingOutputs


@dataclasses.dataclass(frozen=True)
class ThreadingDataConfig(C.DataConfigFactory):
    def create(self, assets_dirs, model_config):
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=T.Group(inputs=[T.RepackTransform({
                "front": FRONT, "side": SIDE, "state": "observation.state",
                "actions": "action", "prompt": "prompt",
            })]),
            data_transforms=T.Group(
                inputs=[ThreadingInputs()],
                outputs=[ThreadingOutputs()],
            ),
            model_transforms=C.ModelTransformFactory()(model_config),
            action_sequence_keys=("action",),
            prompt_from_task=True,
        )


def make_config(settings):
    lora = settings["mode"] == "lora"
    model = pi0_config.Pi0Config(
        pi05=True, action_dim=32, action_horizon=settings["horizon"],
        discrete_state_input=True,
        paligemma_variant="gemma_2b_lora" if lora else "gemma_2b",
        action_expert_variant="gemma_300m_lora" if lora else "gemma_300m",
    )
    return C.TrainConfig(
        name=f"pi05_threading_{settings['mode']}", exp_name=settings["exp_name"],
        project_name="threading_pi05_openpi", model=model,
        data=ThreadingDataConfig(repo_id=settings["repo_id"]),
        weight_loader=weight_loaders.CheckpointWeightLoader(settings["base_checkpoint"].rstrip("/") + "/params"),
        freeze_filter=model.get_freeze_filter(), ema_decay=None if lora else 0.99,
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
                         "action_names": ["dx", "dy", "dz", "drotvec_x", "drotvec_y", "drotvec_z"]},
    )

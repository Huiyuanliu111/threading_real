"""PushBox ARP policy – joint-space (9D robot state + 7D action)."""
from typing import Dict, Tuple, List, Union
from copy import deepcopy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, reduce

from pushbox.diffusion_policy.model.common.normalizer import LinearNormalizer
from pushbox.diffusion_policy.policy.base_image_policy import BaseImagePolicy
from pushbox.diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
from pushbox.diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
from chunk_selector.chunk_selector import ChunkSelection, ChunkSelector
from chunk_selector.execution import (
    PREDICTION_MODES,
    PredictionMode,
    execution_steps_from_chunk_label,
    prediction_execution_lengths,
)
from pushbox import arp

AGENT_STATE_DIM = 9   # 7 joint pos + 2 gripper
ACTION_DIM = 7        # 7D OSC_POSE


def segmented_range_list(start, end, segment_size):
    orig_end = end
    if (end - start) % segment_size != 0:
        end = start + int(math.ceil((end - start) / segment_size)) * segment_size
    multiple = (end - start) // segment_size
    lst = [start + i for i in range(multiple) for j in range(segment_size)]
    return lst[:orig_end - start]


class PushBoxARPolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            horizon,
            n_action_steps,
            n_obs_steps,

            crop_shape=(76, 76),
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,
            pretrained=False,
            freeze_obs_encoder=False,
            
            arp_cfg={},
            tokens: list = None,
    ):
        super().__init__()

        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {'rgb': [], 'depth': [], 'scan': []}
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            if attr['type'] == 'rgb':
                shape = attr['shape']
                obs_key_shapes[key] = list(shape)
                obs_config['rgb'].append(key)

        config = get_robomimic_config(
            algo_name='bc_rnn', hdf5_type='image',
            task_name='square', dataset_type='ph')
        
        with config.unlocked():
            config.observation.modalities.obs = obs_config
            # pretrained ResNet-18 backbone
            config.observation.encoder.rgb.core_kwargs.backbone_kwargs.pretrained = pretrained
            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        ObsUtils.initialize_obs_utils_with_config(config)

        policy: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes=obs_key_shapes,
                ac_dim=action_dim,
                device='cpu',
            )

        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
        
        if obs_encoder_group_norm:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16, 
                    num_channels=x.num_features)
            )
        
        # NOTE: rmbn.CropRandomizer is already monkey-patched to pushbox's
        # CropRandomizer in robomimic_config_util.py. eval_fixed_crop=True
        # means we use the deterministic version (pushbox's CropRandomizer),
        # which is already what robomimic uses due to the monkey-patch.
        # No explicit replacement needed.

        obs_feature_dim = obs_encoder.output_shape()[0]

        self.obs_encoder = obs_encoder
        self.freeze_obs_encoder = freeze_obs_encoder
        if freeze_obs_encoder:
            for p in self.obs_encoder.parameters():
                p.requires_grad = False
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps

        #region arp ########################################################
        self.use_sample = arp_cfg.get('sample', True)
        self.plan_steps = arp_cfg.get('plan_steps', 4)
        self.num_latents = arp_cfg.get('num_latents', 1)
        self.low_var_eval = arp_cfg.get('low_var_eval', True)
        self.plan_chunk_size = arp_cfg.get('plan_chunk_size', 1)
        self.action_chunk_size = arp_cfg.get('action_chunk_size', horizon)

        # Pick first rgb key's shape for wandb visualization
        _rgb_keys = [k for k, v in obs_shape_meta.items() if v.get('type') == 'rgb']
        self.image_shape = obs_shape_meta[_rgb_keys[0]]['shape'] if _rgb_keys else [3, 96, 96]

        self.policy = arp.AutoRegressivePolicy(arp.ModelConfig(
            n_embd=arp_cfg['n_embd'],
            embd_pdrop=arp_cfg['embd_pdrop'],
            layer_norm_every_block=arp_cfg.get('layer_norm_every_block', True),
            max_chunk_size=horizon if self.plan_chunk_size > 0 else (horizon + self.plan_steps),
            max_seq_len=(self.n_obs_steps + self.plan_steps + self.horizon) * 2,
            layers=[arp.LayerType.make(
                    **arp_cfg['layer_cfg'],
                    condition_on='visual-token'
                )] * arp_cfg['num_layers'],
            tokens=tokens if tokens is not None else [
                # Control token: 9D robot state (joints + gripper)
                arp.TokenType.make(name='pos', dim=AGENT_STATE_DIM, is_continuous=True,
                                   embedding='linear', is_control=True),

                # Plan tokens: 9D waypoints in state space, GMM predictor
                arp.TokenType.make(name='coarse-plan', is_continuous=True, dim=AGENT_STATE_DIM,
                                   embedding='linear', predictor='gmm',
                                   predictor_kwargs={'num_latents': self.num_latents,
                                                     'low_var_eval': self.low_var_eval}),

                # Action tokens: 7D OSC_POSE, GMM predictor
                arp.TokenType.make(name='fine-action', dim=ACTION_DIM, is_continuous=True,
                                   embedding='linear', predictor='gmm',
                                   predictor_kwargs={'num_latents': self.num_latents,
                                                     'low_var_eval': self.low_var_eval}),
            ]
        ))

        self.obs_feat_linear = nn.Linear(obs_feature_dim, arp_cfg['n_embd'])
        self.inference_chunk_size = None
        self.prediction_mode: PredictionMode = "full_then_truncate"
        # Loaded separately from the action checkpoint so old checkpoints remain
        # strict-load compatible.
        object.__setattr__(self, "chunk_selector", None)
        self.last_chunk_features: torch.Tensor | None = None
        #endregion ################################################################

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer = normalizer

    @property
    def max_selector_chunk(self) -> int:
        """Maximum usable actions after dropping the stale alignment prediction."""
        start = self.n_obs_steps - 1
        return min(self.horizon, self.policy.cfg.max_chunk_size) - start

    @property
    def max_selector_chunk_label(self) -> int:
        """Maximum public chunk label retained in reports and selector classes."""
        return min(self.horizon, self.policy.cfg.max_chunk_size)

    def execution_steps_for_chunk_label(self, chunk_label: int) -> int:
        """Resolve the public full label (20) to its usable prefix (19)."""
        return execution_steps_from_chunk_label(
            chunk_label,
            max_execution_steps=self.max_selector_chunk,
            full_chunk_label=self.max_selector_chunk_label,
        )

    def set_chunk_selector(self, selector: ChunkSelector | None) -> None:
        if selector is not None:
            selector.validate_for_policy(
                feature_dim=self.policy.cfg.n_embd,
                max_chunk=self.max_selector_chunk_label,
            )
            selector.to(device=self.device)
            selector.eval()
            selector.requires_grad_(False)
        # Keep selector weights out of this action policy's state_dict. They are
        # persisted in their own sidecar checkpoint.
        self._modules.pop("chunk_selector", None)
        object.__setattr__(self, "chunk_selector", selector)

    def set_prediction_mode(self, mode: PredictionMode) -> None:
        """Select full-plan truncation or required-only action generation."""
        if mode not in PREDICTION_MODES:
            raise ValueError(
                f"unknown prediction mode {mode!r}; expected one of {PREDICTION_MODES}"
            )
        self.prediction_mode = mode

    def get_optimizer(
            self, 
            transformer_weight_decay: float, 
            obs_encoder_weight_decay: float,
            lr: float, 
            betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:
        optim_groups = [{'params': self.policy.parameters(), 'weight_decay': transformer_weight_decay}]
        if not self.freeze_obs_encoder:
            optim_groups.append({
                "params": self.obs_encoder.parameters(),
                "weight_decay": obs_encoder_weight_decay
            })
        optim_groups.append({
            "params": self.obs_feat_linear.parameters(),
            "weight_decay": obs_encoder_weight_decay
        })
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas)
        return optimizer

    def compute_loss(self, batch):
        return self.predict_action(batch, training=True)
        
    def predict_action(self, batch_or_obs_dict, training=False) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        nobs_ = batch_or_obs_dict['obs'] if 'obs' in batch_or_obs_dict else batch_or_obs_dict
        # Detect which rgb keys are present (backward compat: "image" or "top45"+"sideview")
        _rgb_keys = [k for k in nobs_ if k in ('top45', 'sideview', 'image')]
        if not _rgb_keys:
            raise KeyError(f"No image keys found in obs, got {list(nobs_.keys())}")
        batch_size = len(nobs_[_rgb_keys[0]])
        dev = nobs_[_rgb_keys[0]].device

        # Normalize and get labels
        if training:
            label_actions = batch_or_obs_dict['action'].clone()
        else:
            label_actions = None

        #region normalize images to ImageNet stats (required for pretrained ResNet)
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 1, 3, 1, 1)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 1, 3, 1, 1)
        nobs_imgs = {}
        for k in _rgb_keys:
            img = nobs_[k][:, :self.n_obs_steps].clone()
            assert img.max() <= 1.0 and img.min() >= 0.0
            nobs_imgs[k] = (img - imagenet_mean) / imagenet_std
        #endregion

        #region normalize agent_pos using normalizer (full horizon needed for plan targets)
        agent_pos_full = nobs_['agent_pos'].clone()  # (B, H, AGENT_STATE_DIM)
        nagents_flat = self.normalizer['agent_pos'].normalize(agent_pos_full.flatten(0, 1))
        agent_pos_norm = nagents_flat.reshape(batch_size, -1, AGENT_STATE_DIM)  # (B, H, 9)
        agent_pos_ctx = agent_pos_norm[:, :self.n_obs_steps]  # (B, n_obs, 9)

        nactions_flat = None
        if training:
            nactions_flat = self.normalizer['action'].normalize(label_actions.flatten(0, 1))
            label_actions_norm = nactions_flat.reshape(batch_size, -1, ACTION_DIM)[:, :self.horizon]
        #endregion

        # region: feature extraction — multi-camera
        _flat_imgs = {}
        for k, img in nobs_imgs.items():
            _flat_imgs[k] = img[:, :self.n_obs_steps].reshape(-1, *img.shape[2:])
        nobs_features = self.obs_encoder(_flat_imgs)
        nobs_features = self.obs_feat_linear(nobs_features)
        nobs_features = nobs_features.reshape(batch_size, self.n_obs_steps, self.policy.cfg.n_embd)
        if not training:
            self.last_chunk_features = nobs_features.detach()
        # endregion

        chunk_selection: ChunkSelection | None = None
        requested_chunk = self.inference_chunk_size
        if not training and self.chunk_selector is not None:
            chunk_selection = self.chunk_selector.select(nobs_features.detach())
            selected_chunks = chunk_selection.chunk_sizes
            if not torch.equal(selected_chunks, selected_chunks[:1].expand_as(selected_chunks)):
                raise ValueError(
                    "PushBox adaptive chunk inference currently requires one shared chunk per batch"
                )
            requested_chunk = int(selected_chunks[0].item())

        requested_execution_steps = (
            None
            if requested_chunk is None
            else self.execution_steps_for_chunk_label(int(requested_chunk))
        )

        # PushBox drops the stale alignment prediction at index zero.  In
        # required-only mode that slot is generated in addition to the actions
        # that will actually be executed.
        prediction_mode: PredictionMode = (
            "full_then_truncate" if training else self.prediction_mode
        )
        prediction_offset = self.n_obs_steps - 1
        _inference_horizon, _n_action_steps = prediction_execution_lengths(
            mode=prediction_mode,
            prediction_horizon=self.horizon,
            default_execution_steps=self.n_action_steps,
            requested_execution_steps=(
                requested_execution_steps if not training else None
            ),
            max_execution_steps=self.max_selector_chunk,
            dropped_prediction_steps=prediction_offset,
        )

        future_tk_types = ['coarse-plan'] * self.plan_steps + ['fine-action'] * _inference_horizon

        if self.plan_chunk_size > 0 and self.plan_steps > 0:
            plan_chk_ids = segmented_range_list(self.n_obs_steps, self.n_obs_steps + self.plan_steps, self.plan_chunk_size)
            action_chk_ids = segmented_range_list(max(plan_chk_ids) + 1, max(plan_chk_ids) + 1 + _inference_horizon, self.action_chunk_size)
            future_chk_ids = plan_chk_ids + action_chk_ids
        else:
            future_chk_ids = [self.n_obs_steps] * (self.plan_steps + _inference_horizon)
        future_tk_chk_ids = [{'tk_id': self.policy.token_name_2_ids[tk_type], 'chk_id': chk_id}
                             for tk_type, chk_id in zip(future_tk_types, future_chk_ids)]

        if training:
            # Coarse-plan tokens: downsample future agent_pos trajectory to plan_steps
            if self.plan_steps > 0:
                future_states = agent_pos_norm[:, self.n_obs_steps:]
                coarse_plans_norm = F.interpolate(
                    future_states.permute(0, 2, 1),
                    size=self.plan_steps, mode='linear',
                    align_corners=self.plan_steps >= 3
                ).permute(0, 2, 1)  # (B, plan_steps, AGENT_STATE_DIM)

            # Pad action tokens to AGENT_STATE_DIM for uniform sequence dim
            label_actions_padded = F.pad(label_actions_norm, (0, AGENT_STATE_DIM - ACTION_DIM))

            if self.plan_steps > 0:
                tk_vals = torch.cat([agent_pos_ctx, coarse_plans_norm, label_actions_padded], dim=1)
            else:
                tk_vals = torch.cat([agent_pos_ctx, label_actions_padded], dim=1)
            tk_names = ['pos'] * self.n_obs_steps + future_tk_types
            tk_types = torch.as_tensor([self.policy.token_name_2_ids[tname] for tname in tk_names]).reshape(1, -1, 1).repeat(batch_size, 1, 1).to(dev)
            seq = torch.cat([tk_vals, tk_types], dim=-1)

            loss_dict = self.policy.compute_loss(
                seq,
                chk_ids=torch.as_tensor(list(range(self.n_obs_steps)) + future_chk_ids).to(dev),
                contexts={'visual-token': nobs_features}
            )
            return loss_dict
        else:
            tk_names = ['pos'] * self.n_obs_steps
            tk_types = torch.as_tensor([self.policy.token_name_2_ids[tname] for tname in tk_names]).reshape(1, -1, 1).repeat(batch_size, 1, 1).to(dev)
            seq = torch.cat([agent_pos_ctx, tk_types], dim=-1)

            action_pred_norm = self.policy.generate(
                seq, future_tk_chk_ids,
                contexts={'visual-token': nobs_features},
                sample=self.use_sample
            )
            # Extract action tokens (after plan tokens), trim to ACTION_DIM
            action_pred_norm = action_pred_norm[:, agent_pos_ctx.size(1) + self.plan_steps:, :ACTION_DIM]

            # Denormalize
            action_pred = self.normalizer['action'].unnormalize(
                action_pred_norm.reshape(-1, ACTION_DIM)
            ).reshape(batch_size, -1, ACTION_DIM)

            start = prediction_offset
            end = min(start + _n_action_steps, action_pred.shape[1])
            action = action_pred[:, start:end]
            result = {
                'action_pred': action_pred,
                'action': action
            }
            if chunk_selection is not None:
                result.update({
                    'chunk_logits': chunk_selection.logits,
                    'chunk_probabilities': chunk_selection.probabilities,
                    'chunk_size': chunk_selection.chunk_sizes,
                    'chunk_confidence': chunk_selection.confidences,
                })
            return result
        # endregion ########################################

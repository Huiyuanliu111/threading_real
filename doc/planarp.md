# Point-cloud PlanARP

PlanARP retains the MVT point-cloud encoder and the three control points per
target pose (origin, +X, +Y). It uses PushT's temporal resampling and decoding
order, without task-stage labels.

The `threading_new_1_mvt_planarp` configuration predicts four sparse poses in
one ARP chunk, followed by ten dense poses in another chunk.
Each sparse pose has six spatial tokens: three anchors in two virtual views.
Dense poses retain the existing spatial heatmap heads; unlike
PushT, dense actions are not replaced with a new continuous action head.

During training, valid target control-point trajectories are linearly resampled
to four points with `align_corners=True`, projected, rounded and clamped to the
image. As in PushT, `reverse_plan=true` puts the farthest guide first. This is
coordinate interpolation of geometric guides, not rigid-pose interpolation.
The plan and action heatmap losses are summed by the trainer.
The action chunk can attend to the teacher-forced plan; plan predictions cannot
read the action labels in later chunks.

At inference, only the current observation is supplied. The model generates
the plan before the actions, so actions condition on predicted plan tokens.
`action` and `action_pred` retain their deployment meanings. The additional
`plan_control_points` result contains reconstructed sparse guides in chronological
order, even when reverse decoding is enabled. These guides are for inspection;
the deployment runner executes the dense action prefix.

PlanARP defaults to `predict_gripper=false` for insertion with an already grasped
box: there are no gripper tokens in training or generation, no gripper loss, and
the seventh action coordinate is exactly zero (hold the current gripper state).
The optional GMM module remains registered for compatibility; it is unused and
receives no gradients in this mode. The class default remains true for older
checkpoint configs that do not specify this option. Gripper labels in this MVT
path are raw widths, not normalized values.

Both levels cover the existing ten-step target window. This change tests
coarse-to-fine supervision, not a longer planning horizon. Future observations
are never inputs, and no new dataset conversion is required.

From the `threading_real` directory:

```bash
python scripts/arp/train.py --config-name=threading_new_1_mvt_planarp
```

For the combined 80-episode, 7.5 Hz dataset (65,536 points per frame), use:

```bash
python scripts/arp/train.py --config-name=threading_combined_80_mvt_planarp
```

PlanARP configs default to W&B online logging. The combined config retains the
80/20 episode split and 266 epochs from the base training setup.
Validation runs halfway through and at the end of every epoch, after a completed
gradient-accumulation group. Checkpoints are saved at the end of every epoch,
starting with the first. `latest.ckpt` and the best three validation checkpoints
are retained. Epoch-end checkpoints store the next epoch index for resumption.

Use a fresh training run: the plan head and larger chunk embedding add parameters.
The original config and old checkpoints still use `plan_steps=0` and
`action_chunk_size=1`. To compare with timestep-by-timestep dense decoding while
keeping the plan, override `policy.action_chunk_size=1`.

## V2: autoregressive dense actions

`pushbox/configs/threading_combined_80_mvt_planarp_v2.yaml` is a standalone
experiment configuration. Relative to the five-epoch v1 run, the policy change
is `action_chunk_size: 2`: generate the four sparse guides as one chunk, then
decode ten dense poses in five autoregressive groups of two. Dataset, split, optimizer and
learning-rate schedule match v1. Training stops after five completed epochs,
validates every half epoch and saves every epoch, with W&B online logging.

```bash
python scripts/arp/train.py --config-name=threading_combined_80_mvt_planarp_v2
```

Run outputs are grouped under `outputs/threading_combined_80_mvt_planarp_v2/`.

## Deployment

Use `scripts/deployment/cartesian.py` with `--weights model --policy-hz 7.5`
for the combined dataset checkpoint. The runner defaults to the remote
controller RPC URL `http://10.157.175.22:8008/RPC2`, command destination
`10.157.175.22`, and local UDP state receiver `10.157.175.211`, matching the
documented lab network. These can be overridden with `--server-url`,
`--server-ip`, and `--udp-ip`. Start the remote controller server first.

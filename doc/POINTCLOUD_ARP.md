# Threading MVT ARP

This model uses the original ARP real-robot visual structure: calibrated RGB-D
point clouds, top and left orthographic MVT images at 420x420, a patch-14
eight-layer ViT, and a four-layer ARP decoder. The patchifier and ViT are fully
trainable. Spatial tokens use the paper's 1.5-pixel Gaussian heatmap targets,
and optimization uses LAMB with warmup and cosine decay. Three spatial control points encode translation and full 3-D
orientation; gripper width is a continuous autoregressive token.

The source images remain at their recorded 640x480 resolution. The builder does
not crop or resize either physical camera. It rejects invalid depth, transforms
both fixed cameras into the Panda base frame, and applies the fixed scene cube
required by MVT. A deterministic 1 mm voxel pass removes duplicate samples; it
does not use the old random 8192-point sampling.

Build the filtered 6 Hz dataset:

```bash
cd /home/huiyuan/teleoperation
conda run -n pushbox python build_threading_mvt_dataset.py
```

Train for approximately the paper's 50,000 optimizer updates:

```bash
cd /home/huiyuan/teleoperation/threading_real
conda run -n pushbox python pushbox/train.py --config-name=threading_new_1_mvt_arp
```

After training, manually move the robot to the grasp pose and deploy without
`--move-to-training-start`. The runner closes the gripper before its first
observation and uses only the calibrated side/front RGB-D cameras:

The Cartesian runner defaults to the follower controller at
`http://10.157.175.22:8008/RPC2`, TrackC destination `10.157.175.22`, and local
state receiver `10.157.175.211`. Start the controller server on the follower
before deployment. Override `--server-url`, `--server-ip`, and `--udp-ip` when
using a different network setup.

```bash
conda run -n pushbox python scripts/deployment/cartesian.py \
  /path/to/checkpoint.ckpt --weights model --policy-hz 6 --execute-steps 10 \
  --grasp-before-inference --initial-grasp-width 0.02 \
  --execute --confirm-real-robot
```

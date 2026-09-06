# Threading spatial camera calibration

`block_grasp_spatial.json` is deliberately not committed with invented values.
Generate it from the synchronized LeRobot recording by clicking the physical
TCP center in well-spread frames from each fixed camera:

```bash
conda run -n pushbox python scripts/calibrate_threading_spatial.py \
  --dataset /home/huiyuan/teleoperation/data/block_grasp_minimal_lerobot_v3_cartesian_stride5_224 \
  --urdf /home/huiyuan/teleoperation/remote_controller/src/remote_controller/assets/panda/panda_arm.urdf \
  --output calibration/block_grasp_spatial.json \
  --cameras sideview frontview \
  --candidates 80 \
  --ui matplotlib
```

Click the `panda_hand_tcp` point, then press Space or Enter. Press `n` when the
TCP is occluded and `q` after at least 12--20 diverse points. Use frames that
span x, y and z; points from one nearly straight trajectory are insufficient.

The `pushbox` environment contains a headless OpenCV build through LeRobot, so
the Matplotlib UI is the reliable default on this machine. `--ui auto` detects
this condition and selects Matplotlib automatically.

Do not train if the reported reprojection RMSE is above roughly 4 pixels at
224x224. Re-run the clicks or use more spatially diverse calibration poses.

## Recommended live calibration

The offline recorder does not preserve a shared camera/robot clock, so moving
frames can associate a pixel with the wrong joint state. The live tool captures
both fixed cameras and the follower state only after the TCP is stationary:

```bash
conda run -n pushbox python scripts/calibrate_threading_spatial_live.py \
  --server-url http://10.157.175.22:8008/RPC2 \
  --udp-ip <THIS_COMPUTER_IP_REACHABLE_FROM_FOLLOWER> \
  --output calibration/block_grasp_spatial.json
```

The tool never sends a motion command. In the live window, move the robot with
the normal safe operator interface. After both titles show `STABLE`, press `c`
to freeze the current RGB/state pair, click the same TCP center in sideview and
frontview, and press Enter. Press `r` to discard the frozen pose and `q` after
at least 10 accepted, spatially diverse poses. Every accepted pair is backed up
to `block_grasp_spatial.progress.json`; after an interruption, add
`--resume-progress` to continue from that file.

Any exception and Ctrl+C force one final atomic progress save. If a progress
file already exists, the tool refuses to overwrite it by default: choose
`--resume-progress`, or explicitly choose `--fresh` to discard it and start a
new calibration. For convenience, `--resume-progress` starts from zero when no
progress file exists yet.

It reads factory color intrinsics from each RealSense and estimates only the
base-to-camera extrinsics with PnP-RANSAC. A calibration above 4 px RMSE is not
written as the requested output, and the training loader also refuses an
existing JSON whose recorded RMSE exceeds 4 px.

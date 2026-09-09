# π0.5：两视角 Threading 微调

本目录使用 Hugging Face LeRobot 的 π0.5 实现和 `lerobot/pi05_base` 权重，
训练任务为 `insert the grasped block through the needle`。

## 数据

最终数据集默认位于：

```text
data/threading_combined_pi05_15hz_sg5_nozero
```

数据由 `data/threading_new_1` 和 `data/threading_new_2` 动态发现并合并，当前应为
80 个 episode。处理流程为：

1. 以主机单调时钟对齐机器人状态与 cam1/cam3；
2. 图像编码为两个 224×224 RGB 视角；
3. 关节目标通过 Panda FK 转为基座坐标系 TCP delta；
4. 对重建的平移轨迹应用 Savitzky–Golay `(window=5, polyorder=2)` 滤波；
5. 计算 `t -> t+2` 动作并从 30 Hz 降采样到 15 Hz；
6. 删除平移 `<1 mm`、旋转 `<0.01 rad` 且夹爪变化 `<0.5 mm` 的动作。

生成数据：

```bash
cd /home/huiyuan/teleoperation
.venv-smolvla/bin/python threading_real/pi05/build_combined_dataset.py \
  --raw-root data/threading_new_1 \
  --raw-root data/threading_new_2 \
  --expected-episodes 80
```

流水线成功后会删除中间 LeRobot 数据；增加 `--keep-intermediates` 可保留中间结果。
所有输出目录均拒绝覆盖已有内容。

单独检查最终数据：

```bash
.venv-smolvla/bin/python threading_real/pi05/preflight.py \
  --dataset-root data/threading_combined_pi05_15hz_sg5_nozero \
  --expected-episodes 80
```

## 训练

训练服务器使用 6 张空闲的 48 GB A40，并通过六卡 DDP 训练。默认预测并执行
完整的 10 帧 action chunk（15 Hz 下约 0.67 秒）；按 episode 留出 20% 数据计算
validation loss。

条件敏感度实验显示旧 π0.5 checkpoint 对图像变化的响应仅约为随机 noise 变化的
9.4%。任务文本和场景固定，因此新的默认 profile 冻结 PaliGemma 语言骨干与
词嵌入，训练 SigLIP vision tower、multimodal projector 和 action expert。
vision tower 使用 action expert 的 0.1 倍学习率；当前训练使用完整的 state
prompt，不执行状态丢弃。
状态和动作使用 π0.5 的 quantile normalization。这个任务不需要夹爪开合，因此
训练 wrapper 会在 flow input 和 target 建立前，把记录的第 7 维 gripper delta
替换为物理零动作对应的归一化常数；部署代码在反归一化后再次强制
`dgripper=0`。模型只从数据学习前 6 维 Cartesian 轨迹，同时保留受监督的稳定
no-op gripper 通道。
它使用 2 次梯度累积，将有效 optimizer batch
设为 12，并运行 20,000 个 microstep；六卡每卡 batch 1 时共读取 120,000 个样本，
相当于当前 11,390 帧训练 split 的约 10.5 次遍历。

```bash
cd /home/huiyuan/teleoperation
source .venv-pi05/bin/activate
GPU_IDS=0,1,3,4,5,6 NUM_PROCESSES=6 BATCH_SIZE=1 \
  GRADIENT_ACCUMULATION=2 STEPS=20000 SAVE_FREQ=10000 EVAL_FREQ=1000 \
  PROPRIOCEPTION_DROPOUT=0 IGNORE_GRIPPER_ACTION=true \
  FINETUNE_MODE=visual_expert VISION_LR_SCALE=0.1 \
  bash threading_real/pi05/train_full.sh
```

loss 默认同步到 W&B 项目 `threading_pi05`。关闭同步可设置
`WANDB_ENABLE=false`。新结果写入
`pi05/outputs/threading_combined_pi05_conditioned_v7_visual_expert_no_gripper`，不会覆盖旧 checkpoint。
当前 `*_nozero` 数据和零动作过滤保持不变。

训练脚本固定：

- `chunk_size=10`
- `n_action_steps=10`
- `eval_split=0.2`
- 每 1,000 microstep 在固定的 512 个验证样本上计算 loss
- 每 10,000 microstep 保存 checkpoint（20,000 steps 保存中点和最终模型）
- 每 20 microstep 汇总一次 loss；六卡归约后每个点包含 120 个样本
- 训练、validation 和 inference 均保留完整状态提示
- 忽略数据中的 gripper delta，并将部署动作的第 7 维固定为零
- 冻结 PaliGemma 语言骨干，vision tower 使用 0.1 倍学习率微调
- bfloat16、gradient checkpointing、六卡 DDP

首次在训练服务器配置环境：

```bash
cd ~/pi05
curl -LsSf https://astral.sh/uv/install.sh | sh  # 仅在尚未安装 uv 时执行
source "$HOME/.local/bin/env"
bash pi05/bootstrap_remote.sh
source .venv-pi05/bin/activate
read -r -s -p "HF token: " HF_TOKEN; echo
export HF_TOKEN
```

环境固定创建在 `~/pi05/.venv-pi05`。安装使用 `uv --no-cache`，避免在空间紧张的
根分区额外保留一份 wheel 缓存。

训练脚本把 Hugging Face 和 W&B 缓存固定在项目的 `.cache/` 下，避免同一用户的
多个缓存目录重复下载权重。不要把 `SAVE_FREQ` 改成很小的值；默认只生成 step
10,000 的中点 checkpoint 和 step 20,000 的最终 checkpoint。训练前后可用
`df -h "$HOME"` 和 `du -sh ~/pi05/* ~/pi05/.cache/*` 检查占用。

GPU 被其他任务占用时，可在远端后台等待 GPU 4、5。脚本要求连续三次检查（默认
每 30 秒一次）均有至少 40000 MiB 空闲显存且利用率不超过 5%，并在启动前再次
检查磁盘至少剩余 45 GiB：

```bash
mkdir -p ~/pi05/logs
read -r -s -p "HF token: " HF_TOKEN; echo
export HF_TOKEN
nohup setsid bash ~/pi05/pi05/wait_for_gpus_and_train.sh \
  > ~/pi05/logs/wait-and-train.log 2>&1 &
echo $! > ~/pi05/wait-and-train.pid
unset HF_TOKEN
tail -f ~/pi05/logs/wait-and-train.log
```

停止尚未启动训练的等待任务：

```bash
kill -- "-$(cat ~/pi05/wait-and-train.pid)"
```

π0.5 需要访问 gated 的 `google/paligemma-3b-pt-224`。训练前必须在 Hugging Face
接受许可并登录。

## Checkpoint 推理检查

```bash
python threading_real/pi05/infer_one.py \
  --checkpoint threading_real/pi05/outputs/threading_combined_pi05/checkpoints/last/pretrained_model \
  --dataset-root data/threading_combined_pi05_15hz_sg5_nozero \
  --frame 0
```

输出为反归一化后的 `10×7` Cartesian action chunk。

## 本地实机部署

部署加载整个 `pretrained_model` 目录。推理机环境应与训练使用相同的 LeRobot
revision，并安装本仓库的控制客户端、Pinocchio 和 RealSense Python bindings：

```bash
cd /home/huiyuan/teleoperation
source .venv-pi05/bin/activate
uv pip install --python .venv-pi05/bin/python -e remote_controller pyrealsense2 h5py
```

π0.5 使用两路相机：`sideview`/cam1 对应
`observation.images.exterior_image_2_right`，`frontview`/cam3 对应
`observation.images.exterior_image_1_left`。状态是 7 个 Panda 关节角加夹爪宽度，
动作是基座坐标系下的 `[dxyz, drotvec, dgripper]`。策略频率必须保持为训练数据的
15 Hz。

在 follower 控制机启动服务：

```bash
cd /home/truphysics/teleoperation/remote_controller
./run_server.sh
```

在 GPU 推理机先运行 dry-run；它读取真实相机和机器人状态，但不会启动 TrackC 或
发送夹爪命令：

```bash
cd /home/huiyuan/teleoperation
source .venv-pi05/bin/activate
python threading_real/scripts/deploy_threading_real_cartesian.py \
  threading_real/pi05/outputs/threading_combined_pi05/checkpoints/last/pretrained_model \
  --task "insert the grasped block through the needle" \
  --server-url http://10.157.175.22:8008/RPC2 \
  --server-ip 10.157.175.22 \
  --udp-ip 10.157.175.211 \
  --policy-hz 15 \
  --execute-steps 1 \
  --max-cycles 20
```

runner 默认使用同步模式：完成当前 action chunk 后才采集下一帧并开始下一次推理，
因此不会用新 chunk 覆盖尚未执行的旧 chunk。`--no-synchronous` 会启用当前的
replace 异步模式；该模式没有 π0.5 RTC 延迟补偿，不用于实机部署。

清空工作空间并确认急停可用后，先执行一个同步单步周期：

```bash
python threading_real/scripts/deploy_threading_real_cartesian.py \
  threading_real/pi05/outputs/threading_combined_pi05/checkpoints/last/pretrained_model \
  --task "insert the grasped block through the needle" \
  --server-url http://10.157.175.22:8008/RPC2 \
  --server-ip 10.157.175.22 \
  --udp-ip 10.157.175.211 \
  --policy-hz 15 \
  --execute-steps 1 \
  --max-cycles 1 \
  --workspace-min 0.410 -0.243 -0.101 \
  --workspace-max 0.538 0.199 0.107 \
  --grasp-before-inference \
  --initial-grasp-width 0.02 \
  --execute \
  --confirm-real-robot
```

上述 workspace 是对全部 14,054 个训练帧执行 Panda FK 后，在各轴最小/最大 TCP
位置外扩 3 cm 得到的。实机布局或 TCP 定义改变后必须重新计算。

确认单步的运动方向、相机对应关系和动作幅度正确后，用一个常驻进程运行 10 个
episode；模型、processor、相机和控制客户端只加载一次：

```bash
python threading_real/scripts/deploy_threading_real_cartesian.py \
  threading_real/pi05/outputs/threading_combined_pi05/checkpoints/last/pretrained_model \
  --task "insert the grasped block through the needle" \
  --server-url http://10.157.175.22:8008/RPC2 \
  --server-ip 10.157.175.22 \
  --udp-ip 10.157.175.211 \
  --policy-hz 15 \
  --execute-steps 1 \
  --episodes 10 \
  --workspace-min 0.410 -0.243 -0.101 \
  --workspace-max 0.538 0.199 0.107 \
  --grasp-before-inference \
  --initial-grasp-width 0.02 \
  --execute \
  --confirm-real-robot
```

加载完成后按 Enter 开始 episode 1。episode 运行期间按 Enter 请求结束；runner 会
先完成当前同步动作、停止 TrackC、等待机械臂进入 `IDLE`，并保留已加载的模型和
相机。停止 TrackC 只会释放本程序的轨迹控制，不会自动开启 freedrive；使用 Franka
本体的 hand-guiding 按钮或 Desk 引导模式手动复位。复位完成后再次按 Enter 开始
下一个 episode。重复到 10 个 episode 完成，或随时按 Ctrl+C 退出。
`--max-cycles` 在此模式下表示每个 episode 的最大推理轮数；保持默认 0 时只由
Enter 结束 episode。任务假定方块已位于夹爪中，`--grasp-before-inference` 会在
每个 episode 开始时、当前手动放置的起始位姿闭合夹爪。

## Soft Chunk Selector

Selector 与 π0.5 分开训练。先从真实机器人 Cartesian 动作生成 `{4,10}` 的概率标签：

```bash
python threading_real/scripts/label_lerobot_tcp_chunks.py \
  --dataset data/threading_combined_pi05_15hz_sg5_nozero \
  --output data/threading_combined_pi05_15hz_sg5_nozero_tcp_chunk_soft_labels_4_10_smoothed \
  --candidate-chunks 4 10 \
  --smoothing-window 5 \
  --label-smoothing-window 3
```

使用训练完成的 π0.5 checkpoint 提取冻结的两相机视觉 token。默认把每路 SigLIP
token 池化为 `4×4`，每帧共缓存 32 个 token：

```bash
python threading_real/pi05/extract_selector_features.py \
  --checkpoint threading_real/pi05/outputs/threading_combined_pi05/checkpoints/last/pretrained_model \
  --dataset-root data/threading_combined_pi05_15hz_sg5_nozero \
  --labels data/threading_combined_pi05_15hz_sg5_nozero_tcp_chunk_soft_labels_4_10_smoothed/labels.parquet \
  --output data/threading_combined_pi05_selector_soft_4_10.hdf5 \
  --batch-size 4 \
  --num-workers 2
```

最后只训练 selector sidecar。模型学习 `p(chunk=4)` 和 `p(chunk=10)`，推理时使用
`4*p4 + 10*p10` 得到连续 chunk，再四舍五入为 4 到 10 的执行步数：

```bash
python threading_real/scripts/train_chunk_selector.py \
  data/threading_combined_pi05_selector_soft_4_10.hdf5 \
  --output-dir threading_real/pi05/outputs/selector_soft_4_10 \
  --use-target-probabilities \
  --selection-mode expected \
  --no-class-weights \
  --batch-size 256 \
  --epochs 100 \
  --patience 15 \
  --device cuda:0
```

这一步只训练约几百万参数的 selector，π0.5 全部冻结，不需要联合训练。特征提取会
加载约 8.8 GB 的 π0.5 权重；`--batch-size 4` 适用于本机 16 GB 4060 Ti，如果显存
仍受其他进程占用则降为 2 或 1。缓存采用 float16，14,054 帧、每帧 `32×2048`
大约占 1.8 GB。

## 两种 chunk 推理路径对比

两条路径使用同一个 selector 输出的 4～10 步执行长度，并且复用同一次视觉编码：

- `required_only`：π0.5 的动作后缀长度直接设为 selector 给出的 `k`，只生成 `k` 步。
- `full_then_truncate`：π0.5 始终生成 10 步，只执行前 `k` 步。

先用离线数据比较纯推理速度。脚本交替两种模式的运行顺序，分别报告视觉编码、
selector、动作生成和总耗时，避免固定先后顺序造成 warmup 偏差：

```bash
cd /home/huiyuan/teleoperation
source .venv-pi05/bin/activate
python threading_real/pi05/benchmark_prediction_modes.py \
  --checkpoint threading_real/pi05/outputs/threading_combined_pi05/checkpoints/last/pretrained_model \
  --selector threading_real/pi05/outputs/selector_soft_4_10 \
  --dataset-root data/threading_combined_pi05_15hz_sg5_nozero \
  --samples 100 \
  --warmup 3 \
  --output-dir threading_real/pi05/outputs/offline_mode_benchmark
```

实机 A/B 测试分别运行下面两条脚本。两者接收原部署 runner 的全部参数。每个有推理
记录的 episode 结束后，脚本分别询问 `Episode N success? [y/n]`，并把人工成功标记和逐周期推理耗时写入
`pi05/outputs/adaptive_comparison/`。两组实验应交替执行，并使用相同的初始物体位置、
任务终止条件和最大 cycle 数。

```bash
# 路径 A：只生成 selector 所需的 k 步
bash threading_real/pi05/run_required_only.sh \
  --task "insert the grasped block through the needle" \
  --server-url http://10.157.175.22:8008/RPC2 \
  --server-ip 10.157.175.22 \
  --udp-ip 10.157.175.211 \
  --policy-hz 15 \
  --max-cycles 30 \
  --workspace-min 0.410 -0.243 -0.101 \
  --workspace-max 0.538 0.199 0.107 \
  --execute --confirm-real-robot

# 路径 B：始终生成 10 步，再执行 selector 所需的前 k 步
bash threading_real/pi05/run_full_then_truncate.sh \
  --task "insert the grasped block through the needle" \
  --server-url http://10.157.175.22:8008/RPC2 \
  --server-ip 10.157.175.22 \
  --udp-ip 10.157.175.211 \
  --policy-hz 15 \
  --max-cycles 30 \
  --workspace-min 0.410 -0.243 -0.101 \
  --workspace-max 0.538 0.199 0.107 \
  --execute --confirm-real-robot
```

汇总全部实机 trial 的成功率和平均推理速度：

```bash
python threading_real/pi05/summarize_real_trials.py \
  threading_real/pi05/outputs/adaptive_comparison \
  --output threading_real/pi05/outputs/adaptive_comparison/summary.json
```

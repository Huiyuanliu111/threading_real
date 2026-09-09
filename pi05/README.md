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

训练服务器默认使用 GPU 4、5 两张 48 GB A40，并通过两卡 DDP 训练。默认预测并执行
完整的 10 帧 action chunk（15 Hz 下约 0.67 秒）；按 episode 留出 20% 数据计算
validation loss。小数据集
默认设置 `TRAIN_EXPERT_ONLY=true`，冻结 VLM 并训练 action expert，以减轻过拟合。

```bash
cd /home/huiyuan/teleoperation
source .venv-pi05/bin/activate
GPU_IDS=4,5 BATCH_SIZE=1 STEPS=15000 SAVE_FREQ=5000 EVAL_FREQ=500 \
  bash threading_real/pi05/train_full.sh
```

loss 默认同步到 W&B 项目 `threading_pi05`。关闭同步可设置
`WANDB_ENABLE=false`。做全参数微调可设置 `TRAIN_EXPERT_ONLY=false`，但应与默认的
expert-only 训练分别保存并比较验证集和实机成功率。

训练脚本固定：

- `chunk_size=10`
- `n_action_steps=10`
- `eval_split=0.2`
- 每 500 step 在固定的 512 个验证样本上计算 loss
- 每 5000 step 保存 checkpoint（15000 steps 共保存 3 份）
- bfloat16、gradient checkpointing、两卡 DDP

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
多个缓存目录重复下载权重。目标服务器只剩约 62 GiB 时，不要把 `SAVE_FREQ` 改回
500；默认 5000 只生成 step 5000、10000、15000 三份 checkpoint。训练前后可用
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
  --output data/threading_combined_pi05_selector_soft_4_10.hdf5
```

最后只训练 selector sidecar。模型学习 `p(chunk=4)` 和 `p(chunk=10)`，推理时使用
`4*p4 + 10*p10` 得到连续 chunk，再四舍五入为 4 到 10 的执行步数：

```bash
python threading_real/scripts/train_chunk_selector.py \
  data/threading_combined_pi05_selector_soft_4_10.hdf5 \
  --output-dir threading_real/pi05/outputs/selector_soft_4_10 \
  --use-target-probabilities \
  --selection-mode expected \
  --no-class-weights
```

# SmolVLA 轻量微调：最小积木抓取

这套脚本在单张 16 GB RTX 4060 Ti 上微调 `lerobot/smolvla_base`。默认冻结视觉编码器
和 VLM，只训练 action expert 与状态投影；这比 π0.5 全量微调节省大量显存和磁盘。

训练环境固定到 LeRobot 提交
`3f2c29ef7e44b1ddccbcda3b6a63939e53639e9e`，并固定 PyTorch 2.7.1 CUDA 11.8，
以同时兼容本机 4060 Ti 和驱动较旧的 A40 服务器。

## 数据与控制定义

- 复用 `data/block_grasp_minimal_pi05_6hz`，无需重新转换。
- 输入：三个 224×224 RGB 视角和 8 维状态。
- 输出：7 维 Cartesian 增量。
- 控制频率：6 Hz。
- action chunk：10 步；每次执行 2 步后重规划。
- 提示词：`pick up the block`。

SmolVLA 会在模型内部将图像 resize/pad 到预训练分辨率。状态和动作会自动填充到模型的
32 维上限，输出时再裁回 7 维。

预训练模型使用 `camera1/2/3` 键名。训练脚本显式采用以下固定映射，并把映射保存在
checkpoint 的预处理器中：

- `exterior_image_1_left` → `camera1`
- `exterior_image_2_right` → `camera2`
- `wrist_image_left` → `camera3`

## 1. 创建本机环境

脚本要求已安装 `uv`。它不会覆盖已有环境：

```bash
cd /home/huiyuan/teleoperation
bash threading_real/smolVLA/bootstrap_local.sh
source .venv-smolvla/bin/activate
```

如果环境已存在且确实要重建，手动删除 `/home/huiyuan/teleoperation/.venv-smolvla` 后再执行。
SmolVLA 权重公开，不需要 PaliGemma gated model 许可。

## 2. 数据预检

```bash
python threading_real/smolVLA/preflight.py \
  --dataset-root data/block_grasp_minimal_pi05_6hz
```

输出必须包含 `"errors": []`。

## 3. 两步显存测试

先关闭本机其他 GPU 程序，然后运行：

```bash
GPU_ID=0 BATCH_SIZE=1 STEPS=2 SAVE_FREQ=2 \
  OUTPUT_DIR="$PWD/threading_real/smolVLA/outputs/smoke" \
  bash threading_real/smolVLA/train_expert.sh
```

另开终端观察：

```bash
watch -n 1 'nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv'
```

测试成功后记录显存和 checkpoint 大小，再删除 smoke 输出。

## 4. 正式训练

默认 batch size 1、10000 step、每 1000 step 保存一次：

```bash
GPU_ID=0 BATCH_SIZE=1 STEPS=10000 SAVE_FREQ=1000 \
  bash threading_real/smolVLA/train_expert.sh
```

20 条示范很少，建议实机比较多个 checkpoint。若 smoke test 后仍有充足显存，可以尝试
`BATCH_SIZE=2`；不要在未测试时直接增大 batch。

断点续训使用 checkpoint 保存的配置：

```bash
CUDA_VISIBLE_DEVICES=0 lerobot-train \
  --config_path=threading_real/smolVLA/outputs/block_grasp_minimal_expert/checkpoints/last/pretrained_model/train_config.json \
  --resume=true
```

## 5. 单帧推理

```bash
python threading_real/smolVLA/infer_one.py \
  --checkpoint threading_real/smolVLA/outputs/block_grasp_minimal_expert/checkpoints/last/pretrained_model \
  --dataset-root data/block_grasp_minimal_pi05_6hz \
  --frame 0
```

输出应为反归一化后的 10×7 action chunk，并包含单次推理延迟和峰值显存。首次调用包含
冷启动开销；评估实时性时应预热后重复测量。接入实机前继续使用现有平移/旋转限幅与急停。

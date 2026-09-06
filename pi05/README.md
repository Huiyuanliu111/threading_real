# π0.5 全量微调：最小积木抓取

这套脚本使用 Hugging Face LeRobot 的官方 π0.5 PyTorch 实现和
`lerobot/pi05_base` 权重。选择该实现是因为现有数据已经是 LeRobot v3；当前
`openpi` 主分支仍固定使用 LeRobot v2.1，不能直接读取本项目的数据。

训练环境固定到 LeRobot 官方提交
`3f2c29ef7e44b1ddccbcda3b6a63939e53639e9e`。PyPI 最新稳定版 0.4.4 虽能训练
π0.5，但它尚没有完整的 FSDP checkpoint/resume 路径；本服务器的两张空闲 A40
各为 48 GB，全量微调必须使用官方主线新增的 FSDP2 支持。

## 数据定义

- 输入：三个 224×224 RGB 视角和 8 维状态（7 个关节角 + 夹爪宽度）。
- 输出：7 维 Cartesian 增量（`dx,dy,dz,drotvec_x,drotvec_y,drotvec_z,dgripper`）。
- 控制频率：6 Hz。
- action chunk：10 步（1.67 秒），推理时每次先执行 2 步再重规划。
- 提示词：`pick up the block`。

原 `cartesian_stride5_224` 数据中的动作已经是 `t -> t+5`，但视频和状态仍为
30 Hz。必须先真正降采样到 6 Hz，否则 π0.5 的相邻 chunk 动作会重叠。

## 1. 传到远端

训练服务器：`huiyuan@10.157.174.249`。远端项目目录默认使用
`/home/huiyuan/teleoperation`。在本机执行：

```bash
ssh huiyuan@10.157.174.249 'mkdir -p /home/huiyuan/teleoperation/threading_real/pi05 /home/huiyuan/teleoperation/data/block_grasp_minimal_pi05_6hz'

rsync -av --progress threading_real/pi05/ \
  huiyuan@10.157.174.249:/home/huiyuan/teleoperation/threading_real/pi05/
rsync -av --progress data/block_grasp_minimal_pi05_6hz/ \
  huiyuan@10.157.174.249:/home/huiyuan/teleoperation/data/block_grasp_minimal_pi05_6hz/
```

这里不自动登录或修改远端服务器。

如果远端实际采用当前终端里的 `~/pi05/pi05`（脚本）和 `~/pi05/data`（数据）布局，
训练脚本也会自动识别，不必搬到 `/home/huiyuan/teleoperation`。

## 2. 安装环境

登录并在远端项目根目录手动执行：

```bash
ssh huiyuan@10.157.174.249
cd /home/huiyuan/teleoperation
PYTHON_BIN=python3.12 bash threading_real/pi05/bootstrap_remote.sh
source .venv-pi05/bin/activate
hf auth login
```

固定版本要求 Python 3.12。若系统没有 `python3.12`，可先用 Conda 创建环境，再让
脚本使用该解释器；不要退回安装 `lerobot==0.4.4`：

```bash
conda create -n threading_pi05 python=3.12 -y
PYTHON_BIN="$(conda run -n threading_pi05 which python)" \
  bash threading_real/pi05/bootstrap_remote.sh
```

π0.5 使用 gated 的 `google/paligemma-3b-pt-224` tokenizer。训练前需要在
Hugging Face 页面接受许可，并登录有权限的账号。

全量微调没有冻结视觉编码器或 VLM。服务器上 GPU 4、5 是两张空闲的 48 GB
NVIDIA A40，脚本默认只暴露这两张卡并用 FSDP2 对参数、梯度和优化器状态做全分片。
两卡之间为同一 NUMA 节点的 `NODE` PCIe 路径，没有 NVLink，因此能训练但通信速度
会慢于 A100/H100 NVLink 机器。默认每卡 batch size 为 1，有效 batch size 为 2。

## 3. 生成或检查 6 Hz 数据

本机已经生成 `data/block_grasp_minimal_pi05_6hz`，按上面的命令传输后只需执行
`preflight.py`。如果需要在远端从 30 Hz 数据重新生成，再执行转换命令：

```bash
python threading_real/pi05/prepare_dataset.py \
  --source data/block_grasp_minimal_lerobot_v3_cartesian_stride5_224 \
  --output data/block_grasp_minimal_pi05_6hz

python threading_real/pi05/preflight.py \
  --dataset-root data/block_grasp_minimal_pi05_6hz
```

转换器不会覆盖已有输出。确实需要重建时显式增加 `--overwrite`。

## 4. 全量微调

```bash
source .venv-pi05/bin/activate
GPU_IDS=4,5 BATCH_SIZE=1 STEPS=3000 SAVE_FREQ=500 \
  bash threading_real/pi05/train_full.sh
```

默认关闭 W&B 和 Hub 上传。每 500 step 保存可直接推理的 safetensors；FSDP
优化器状态仍以 DCP 分片保存并支持续训。没有额外再保存一份模型 DCP，以减少磁盘
占用。20 条示范很少，建议比较 500、1000、1500、2000、2500、3000 step 的实机
成功率，不要只选训练 loss 最低的检查点。

先做 2 step 启动测试，确认日志出现 `dp_shard=2`，并观察两卡显存：

```bash
GPU_IDS=4,5 STEPS=2 SAVE_FREQ=2 \
  OUTPUT_DIR="$PWD/threading_real/pi05/outputs/fsdp_smoke" \
  bash threading_real/pi05/train_full.sh
```

另开终端观察：

```bash
watch -n 1 'nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv'
```

不要使用 GPU 0、1、2、3、6，也不要终止占用这些卡的 PID `1941144`。

断点续训使用 LeRobot 保存的配置（将路径改成实际 run）：

```bash
CUDA_VISIBLE_DEVICES=4,5 torchrun --standalone --nproc-per-node=2 "$(which lerobot-train)" \
  --config_path=threading_real/pi05/outputs/block_grasp_minimal_full/checkpoints/last/pretrained_model/train_config.json \
  --resume=true
```

## 5. 单帧推理冒烟测试

```bash
python threading_real/pi05/infer_one.py \
  --checkpoint threading_real/pi05/outputs/block_grasp_minimal_full/checkpoints/last/pretrained_model \
  --dataset-root data/block_grasp_minimal_pi05_6hz \
  --frame 0
```

输出是已经反归一化的 10×7 Cartesian action chunk。接入实机前应继续沿用现有
部署脚本的平移/旋转限幅和急停逻辑，并先在不使能机器人时检查动作分布。

## 版本

脚本固定到上述 LeRobot Git commit，并使用其原生 FSDP2 checkpoint 路径。若升级
LeRobot，请先重新运行 `preflight.py` 和 `infer_one.py`；训练 CLI 与 checkpoint
processor 格式可能变化。
训练脚本显式使用 `pyav` 解码视频，以免远端 PyTorch、TorchCodec 与系统 FFmpeg
的二进制版本不匹配。

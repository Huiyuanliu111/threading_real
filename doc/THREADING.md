# Threading 任务集成与运行指南

推荐配置 `threading_arp_v3` 使用独立的
`threading_task.policy.ThreadingARPolicy`，不会修改或覆盖 PushBox 共用策略。
模型输入和原始 MimicGen 观测的对应关系如下：

| 模型输入 | Threading HDF5 字段 |
| --- | --- |
| `top45` | `agentview_image` |
| `wrist` | `robot0_eye_in_hand_image` |
| `sideview` | 补渲染的 `threading_closeup_image`（128x128） |
| `agent_pos` | `robot0_eef_pos` + `robot0_eef_quat` + 第一维 `robot0_gripper_qpos`（8D） |
| action | 7维 `OSC_POSE` action |

v3 使用 `agentview 84 + wrist 84 + threading_closeup 128` 三路图像。近景相机从环的
背面正对孔口，D0 插入阶段孔口约占20像素。三路图像共享 ImageNet 预训练
ResNet-34，并通过 EEF 状态条件化的 BFA 风格权重动态融合空间 token。ARP 主干为
384维、12层、12头 Transformer，共约6093万参数，每执行1步就重新规划。旧配置
`threading_arp`、`threading_arp_v2` 及其 checkpoint 仍可评估，但不能续训 v3。

仓库内置兼容 robosuite 1.5 的 `Threading_D0`、`Threading_D1` 和
`Threading_D2` 环境。直接使用官方 core 数据训练 ARP 不需要安装 MimicGen；
只有重新生成演示数据时才需要官方 MimicGen 工具和独立环境。

注意：robosuite 1.5 端口适合本仓库的日常训练和在线 rollout，但与生成 core 数据的
robosuite 1.4.1 不是逐位动力学等价。若要复现官方 benchmark 或发表严格可比的成功率，
仍应另建隔离环境，使用 robosuite 1.4.1 和官方 MimicGen Threading 环境评估。

## 1. 进入项目和激活环境

所有命令均从仓库根目录执行：

```bash
cd /home/huiyuan/pushbox

source /home/huiyuan/miniconda3/etc/profile.d/conda.sh
conda activate pushbox

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```


## 2. 验证、统计和训练数据视频
下载数据

https://huggingface.co/datasets/amandlek/mimicgen_datasets/resolve/main/core/threading_d0.hdf5


```bash
mkdir -p artifacts/threading/train_videos
mkdir -p artifacts/threading/dataset_analysis

python scripts/validate_threading_dataset.py \
  data/threading/threading_d0.hdf5 \
  --check-env \
  --replay-actions 1 \
  --output artifacts/threading/full_validation.json

python scripts/analyze_threading_dataset.py \
  data/threading/threading_d0.hdf5 \
  --output-dir artifacts/threading/dataset_analysis

python scripts/replay_threading_demo.py \
  data/threading/threading_d0.hdf5 \
  --auxiliary-dataset data/threading/threading_d0_sideview_128.hdf5 \
  --episode 0 \
  --output artifacts/threading/train_videos/episode_000_three_view.mp4 \
  --fps 20 \
  --size 256
```

查看结果：

```bash
cat artifacts/threading/full_validation.json
cat artifacts/threading/dataset_analysis/dataset_stats.json
ls -lh artifacts/threading/train_videos/episode_000_three_view.mp4
```


`--replay-actions N` 不只是检查 HDF5 schema，还会把前 N 条官方专家动作放进
在线环境闭环重放。至少先用 `N=1` 验证控制器；需要审计 1.4→1.5 的剩余动力学
差异时可增大 N。

### 2.1 生成第三路近景相机数据

官方 HDF5 已含每帧 simulator state，因此只补渲染相机，不改变专家动作。完整输出
包含1000条 episode、224508帧，约3.3 GiB。命令支持中断后继续：

```bash
python scripts/render_threading_camera.py \
  data/threading/threading_d0.hdf5 \
  data/threading/threading_d0_sideview_128.hdf5 \
  --camera-name threading_closeup \
  --size 128 \
  --resume
```

验证生成结果：

```bash
ls -lh data/threading/threading_d0_sideview_128.hdf5

python - <<'PY'
import h5py

path = "data/threading/threading_d0_sideview_128.hdf5"
with h5py.File(path, "r") as f:
    data = f["data"]
    episodes = len(data)
    complete = sum(bool(data[key].attrs.get("complete", False)) for key in data)
    frames = sum(len(data[key]["obs/threading_closeup_image"]) for key in data)
print({"episodes": episodes, "complete": complete, "frames": frames})
PY
```

## 3. 从头训练（W&B 实时同步）


```bash
cd /home/huiyuan/pushbox
conda activate pushbox

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

RUN_DIR="/home/huiyuan/pushbox/outputs/threading_d0_v3/$(date +%Y%m%d_%H%M%S)"
WANDB_RUN_ID=$(python -c 'import wandb; print(wandb.util.generate_id())')

mkdir -p "$RUN_DIR"
mkdir -p outputs/threading_d0_v3
printf '%s\n' "$RUN_DIR" > outputs/threading_d0_v3/latest_run.txt
printf '%s\n' "$WANDB_RUN_ID" > "$RUN_DIR/wandb_run_id.txt"

python pushbox/train.py \
  --config-name=threading_arp_v3 \
  task.dataset.dataset_path=/home/huiyuan/pushbox/data/threading/threading_d0.hdf5 \
  training.device=cuda:0 \
  logging.mode=online \
  logging.id="$WANDB_RUN_ID" \
  +logging.resume=never \
  "hydra.run.dir=$RUN_DIR"
```

启动日志必须出现下面一行，否则 Threading rollout workspace 没有生效：

```text
[train] workspace: threading_task.workspace.ThreadingARPWorkspace
```

默认每个 epoch 会在 `$RUN_DIR/threading_rollouts/` 写视频和统计，并把成功率实时
同步到 W&B。`val_loss` 是 GMM 负对数似然，可能为负数；选择 checkpoint 时必须同时
检查 `test_success_rate`，不能只选择数值最小的 `val_loss`。

配置默认训练8个完整 epoch，每个 epoch 约18300个 batch，batch size 为8。
ResNet-34 学习率为 `5e-6`，新视觉投影、视角融合层和 ARP 主干学习率为 `8e-5`。
W&B rollout 额外记录 `test_view_weight_top45`、`test_view_weight_wrist` 和
`test_view_weight_sideview`。

## 4. 中断和断点续训

前台训练可使用 `Ctrl-C` 停止。也可以从另一个终端发送中断信号：

```bash
pgrep -af 'pushbox/train.py.*threading_arp_v3'
kill -INT <训练主进程PID>
```

从同一 checkpoint 和同一个 W&B run 继续：

```bash
cd /home/huiyuan/pushbox
source /home/huiyuan/miniconda3/etc/profile.d/conda.sh
conda activate pushbox

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

RUN_DIR=$(cat outputs/threading_d0_v3/latest_run.txt)
CKPT="$RUN_DIR/checkpoints/latest.ckpt"
WANDB_RUN_ID=$(cat "$RUN_DIR/wandb_run_id.txt")

python pushbox/train.py \
  --config-name=threading_arp_v3 \
  task.dataset.dataset_path=/home/huiyuan/pushbox/data/threading/threading_d0.hdf5 \
  training.device=cuda:0 \
  training.resume=true \
  training.resume_path="$CKPT" \
  logging.mode=online \
  logging.id="$WANDB_RUN_ID" \
  +logging.resume=must \
  "hydra.run.dir=$RUN_DIR"
```

如果之前使用 `logging.mode=offline`，先上传旧记录：

```bash
OLD_RUN=/path/to/offline/run
wandb sync "$OLD_RUN"/wandb/offline-run-*
```

离线进程不能在运行过程中直接切换成实时同步；需要在 checkpoint 保存后停止，
再使用上面的在线续训命令启动新进程。

## 5. 最终评估、视频和统计

### 5.1 平移动作缩放

在线评估默认将 `OSC_POSE` action 的 XYZ 平移分量乘以 `0.25`，旋转和夹爪分量
保持不变。缩放发生在 `env.step` 前，不需要重新训练 checkpoint；可通过
`--translation-scale` 修改。

平移速度降为四分之一后，应同步增加环境步数预算。完整任务继续使用
`--max-steps 1000`；子任务评估建议将 approach、pick、insert 的上限分别设为
`400`、`480`、`1200`。沿用原始 `100/120/300` 会造成超时偏差，不能用于选择
最佳 action chunk。

```bash
cd /home/huiyuan/pushbox
source /home/huiyuan/miniconda3/etc/profile.d/conda.sh
conda activate pushbox

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

RUN_DIR=$(cat outputs/threading_d0_v3/latest_run.txt)
CKPT="$RUN_DIR/checkpoints/latest.ckpt"

python scripts/eval_threading.py "$CKPT" \
  --dataset /home/huiyuan/pushbox/data/threading/threading_d0.hdf5 \
  --env-name Threading_D0 \
  --episodes 50 \
  --max-steps 1000 \
  --device cuda:0 \
  --weights ema \
  --output-dir "$RUN_DIR/final_eval" \
  --save-videos all \
  --max-videos 50 \
  --video-fps 20
```

查看评估结果：

```bash
RESULT_DIR="$RUN_DIR/final_eval/threading_rollouts/run_0000"

cat "$RESULT_DIR/summary.txt"
cat "$RESULT_DIR/eval_stats.json"
ls -lh "$RESULT_DIR/videos"
```

评估目录包含：

- `eval_stats.json`：聚合指标和逐条 episode 指标；
- `episodes.csv`：逐 episode 统计；
- `summary.txt`：成功率、置信区间、平均步数和推理延迟摘要；
- `videos/*.mp4`：三相机评估视频，并显示当前动态视角权重。

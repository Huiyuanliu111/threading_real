# pushbox狭窄通道任务

基于 [robosuite](https://robosuite.ai/) 的自定义推箱环境，机械臂为 **Franka Emika Panda**。

## 环境要求

- Linux
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 或 Anaconda
- Python ≥ 3.10（conda 环境 `pushbox`）
- 可选：NVIDIA GPU（`torch` 会安装 CUDA 相关依赖）

## Conda 环境

```bash
conda create -n pushbox python=3.10 -y
conda activate pushbox
```

进入本项目目录时，Cursor / VS Code 终端会自动激活 `pushbox`（见 `.vscode/settings.json`）。手动激活：

```bash
conda activate pushbox
```

## 安装

依赖定义在 `pyproject.toml`。在项目根目录执行：

```bash
conda activate pushbox
pip install -e .
```

```
pushbox/
├── pyproject.toml
├── README.md
├── envs/              # PushBoxEnv、maze_layouts
├── scripts/
│   ├── teleop.py      # 遥操作录制（LeRobot v3 格式）
│   └── find_mazes.py  # 迷宫搜索 / 确认 / 校验
├── data/
│   ├── mazes/         # selected.json、候选预览
│   └── demos/         # LeRobot v3 数据集（meta/, data/, videos/）
└── pushbox/           # 核心逻辑

arp/pushbox/           # ARP 训练（独立，见上文）
```

## 任务定义

- 使用OSC_POSE action space
- Robot model: franka emika panda
- Arena: 一个正方形积木、两块障碍物、规定的成功区域，均位于平整桌面
- 任务性质：机械臂推动积木进行二维的移动，避开障碍物，达到成功区域。限制z的大小为常数。
- 任务起始：机械臂已经抓握到了积木，积木还处于桌面上。
- 模型输入：低维输入，直接将机器人关节角度、末端执行器（End-effector）的位姿以及目标物体的精确三维坐标输入给模型，不需要视觉
- 成功条件:积木达到成功区域。
- 失败条件:机械臂碰撞障碍物、末端出界、积木滑落桌面、或达到最大步数（Time Out）。

## 场景详情


积木初始位置（`BOX_INIT_X_RANGE` / `BOX_INIT_Y_RANGE`）：**x ∈ [1, 5] cm, y ∈ [-25, -20] cm**。目标 **(0, 25) cm**，半径 **5 cm**。

两块障碍物：棱柱，高 10 cm，y 方向固定 4 cm 半宽，不可穿过。基准位置为 x 方向通道宽 8 cm。每次 reset 随机变化，以训练模型的避障泛化能力：

| 参数 | 范围 | 说明 |
|------|------|------|
| 通道宽度 | 8 ~ 10 cm | 0.44 − (L0 + L1)，保证 ≥ 8 cm（积木 5 cm 可通过） |
| y 偏移 | ±2.5 cm | 整体平移 |
| 障碍物 x 半长 | 15 ~ 19 cm | 两块长度一样 |

基准配置（`V2_FIXED_OBSTACLES`）：
- 障碍物1: xy 坐标 [3, -4] ~ [40, 4] cm（中心 21.5 cm，0 cm）
- 障碍物2: xy 坐标 [-40, -4] ~ [-5, 4] cm（中心 −22.5 cm，0 cm）



## ARP 模仿学习（waypoint 建模）

训练代码在独立目录 [`arp/`](arp/) 内。

PushBox 是 **3 维任务**（机械臂在三维空间运动），不能像 PushT 一样简化为 2D 像素坐标。因此采用 **关节空间建模**：


| Token         | 维度 | 含义                                                           |
| --------------- | ------ | ---------------------------------------------------------------- |
| `pos`         | 9    | 当前机器人状态（7 关节位置 + 2 夹爪 qpos），控制 token，不预测 |
| `coarse-plan` | 9    | 未来关节状态 waypoint（从完整状态序列降采样），GMM 头预测      |
| `fine-action` | 7    | 细粒度 OSC_POSE 动作（`[dx,dy,dz,dr,dp,dy,grip]`），GMM 头预测 |

训练时使用 `LinearNormalizer` 对 joint state（9D）、action（7D）、box_to_goal（2D）、goal_radius（1D）做归一化/反归一化。与 PushT 的关键区别：

- PushT：2D 像素坐标 + `upsample_from_2d_attn`（从 birdview 特征图定位 waypoint）
- PushBox：9D 关节状态 + GMM 预测（直接回归连续值，不依赖 2D 空间定位）


### 视觉输入

需要使用 **top45（正前方向下俯瞰45度图）** 以及 **侧视图** 作为视觉输入（两张 96×96 RGB）。PushBox 是 3 维任务，需要两个视角以及本体感知作为输入

与其他 ARP 任务对比：


| 任务                | 视觉输入                             | 说明                        |
| --------------------- | -------------------------------------- | ----------------------------- |
| PushT               | 1× 俯视 96×96                      | 2D 平面推块，单视角足够     |
| ALOHA               | 1× 顶部 480×640                    | 双机械臂桌面操作            |
| RLB                 | 3× 多视角（top/left/front）128×128 | 3D 空间抓取，需要多视角     |

### 数据生成
生成程序是 [datagen.py](~/pushbox/scripts/datagen.py)

预定义任务阶段和路径点
→ 样条曲线插值
→ 转换为机器人控制动作
→ 在 PushBox 环境执行
→ 记录双相机图像、机器人状态、box 位置和动作

### 数据划分

- 共 **101** 条演示：`~80` 训练 / `~21` 验证（`val_ratio: 0.2`）
- 数据路径：`data/datagen`（LeRobot v3 格式）
- 配置见 `pushbox/configs/arp.yaml`

```bash
# 回放第 0 个 episode（默认）
python scripts/replay_demo.py data/datagen
# 回放指定 episode（帧流式写临时文件 → 子进程 ffmpeg 转 mp4，避免 GPU 崩溃）
python scripts/replay_demo.py data/datagen --episode 20 --size 128
# 仅物理回放（不渲染，最快最安全）
python scripts/replay_demo.py data/datagen --episode 3 --no-render
```

### 训练依赖

在 `pushbox` conda 环境中额外安装 ARP 训练依赖：

```bash
pip install -r arp/requirements.txt
```

### 训练命令

```bash
cd pushbox
python train.py --config-name=arp
```

检查点与日志输出在 `outputs/`。最佳 checkpoint 按验证集 `val_loss` 保存。

### 评估命令
评估时关掉障碍物的随机化
```bash
# 传目录 → 取第一个 .ckpt
cd /home/huiyuan/pushbox
python scripts/eval_policy.py outputs/2026-06-16/13-58-55/checkpoints --save-video

# 传具体文件
python scripts/eval_policy.py pushbox/outputs/2026-06-26/23-08-49/checkpoints/epoch=0120-val_loss=-30.466.ckpt --save-video
```

### teleop 与 GPU 渲染（EGL / GLX）

MuJoCo 离屏渲染默认走 **EGL**（NVIDIA GPU），pygame 窗口在 X11 上默认走 **GLX**。两者在同一 `DISPLAY` 上不能同时 `MakeCurrent`，常见报错：

- `EGL_BAD_ACCESS`（`eglMakeCurrent`）
- `X_GLXMakeCurrent` / `BadAccess`

`scripts/teleop.py` 已内置处理：创建环境前临时去掉 `DISPLAY`（纯 EGL 初始化）、`SDL_VIDEO_X11_FORCE_EGL=1`（让 SDL 也用 EGL）、`hard_reset=False`（避免 reset 时重建 EGL 与窗口冲突）。一般直接运行即可；若仍异常，用 `--debug` 查看各阶段环境变量。

与 `scripts/demo.py --render` 的区别：demo 不开离屏渲染器，不触发 EGL；teleop 需要离屏相机画面，才会碰到上述问题。

### 三子任务定义

按箱子 **y 坐标** 将推箱任务拆为三个阶段（顺序不可逆）：

| 阶段 | 名称 | 判定条件 |
|------|------|----------|
| 0 | 接近障碍物 | `box_y < -0.1` |
| 1 | 穿过障碍物 | `-0.1 ≤ box_y ≤ 0.1` |
| 2 | 到达目的区域 | `box_y > 0.1`，最终以进入 goal 区域为成功 |

子任务独立评估 (`eval_subtasks.py`) 时，每个子任务有独立的箱子初始化范围，互不依赖：

| 子任务 | 阶段 | box x 范围 | box y 范围 | 成功条件 |
|--------|------|-----------|-----------|----------|
| 1 | 0 | `[0.01, 0.05]` | `[-0.25, -0.12]` | `box_y > -0.10` |
| 2 | 1 | `[-0.01, 0.06]` | `[-0.2, -0.15]` | `box_y > 0.1` |
| 3 | 2 | `[0.01, 0.05]` | `[0.10, 0.12]` | 推到目标 `(0.0, 0.25)` |

**障碍物布局**：两个固定的长条形障碍物在 `y = 0` 处形成一个狭窄通道：

| 障碍物 | 中心位置 (x, y) | 半尺寸 (x, y) | x 范围 |
|--------|----------------|--------------|--------|
| `fixed_obs_0`（右侧） | `(0.215, 0.0)` | `(0.185, 0.04)` | `[0.03, 0.40]` |
| `fixed_obs_1`（左侧） | `(-0.225, 0.0)` | `(0.175, 0.04)` | `[-0.40, -0.05]` |

两个障碍物之间的**开口**位于 `x ∈ [-0.05, 0.03]`，宽度约 **8 cm**，箱子必须从这个开口穿过 y=0 通道。

实现位置：
- [`envs/pushbox_env.py`](envs/pushbox_env.py) — 阶段检测、分阶段 reward/termination、`info` 输出（`current_phase`、`phase_steps`、`phase_success`）
- [`scripts/eval_policy.py`](scripts/eval_policy.py) — 分阶段成功率/步数统计，写入 `eval_stats.json`
- [`scripts/verify_phases.py`](scripts/verify_phases.py) — 阶段逻辑 smoke test

ARP 模型与 dataset **不变**；阶段信息仅用于环境 reward/终止与评估统计。

```bash
# 验证阶段逻辑
python scripts/verify_phases.py

# 整体评估（默认，保留原评估方式）
python scripts/eval_policy.py /path/to/checkpoint.ckpt --episodes 20 --device cuda:0

# 分阶段评估（额外输出每个阶段成功率和步数）
# 默认输出到 <run_dir>/subtasks/epoch=XXXX/eval_stats.json
python scripts/eval_subtasks.py /home/huiyuan/pushbox/pushbox/outputs/2026-06-26/23-08-49/checkpoints/epoch=0140-val_loss=-30.500.ckpt --episodes 20 --device cuda:0 --tasks approach --save-all-videos
```

Persistent collision

## 待办事项

## 常用命令
```bash
python scripts/eval_policy.py "/home/huiyuan/pushbox/pushbox/outputs/2026-06-26/10-16-32/checkpoints/epoch=0140-val_loss=-35.273.ckpt" --device cuda:0 --episodes 5 --save-video

python scripts/replay_demo.py data/old_data/datagen --episode 10 --size 128
```

# Threading H20 / Maze H10：ARP decoding latency

本机离线实测，日期为 2026-09-22。模型严格对应
[selector 部署文档](../../doc/deploy_selector.md) 的主实验：Threading H20 与 Maze H10，
使用训练集真实观测，只计时 ARP 解码，不包含视觉编码、selector 或机器人执行。

## 模型与比较口径

| 任务 | checkpoint | 权重 | H / action group | coarse plan | 对比生成步数 |
|---|---|---|---|---|---|
| Threading | epoch 8，val_loss=10.981 | model | 20 / 20 | 4 个 pose，24 个空间 token | 20 与 8 |
| Maze | epoch 209 | model | 10 / 10 | 4 个 XY waypoint，8 个空间 token | 10 与 4 |

checkpoint 路径（相对于 `/home/huiyuan/teleoperation`）：

```text
training_runs/planarp_chunks_20260914_165545/threading_planarp_chunk20/checkpoints/epoch=0008-val_loss=10.981.ckpt
maze_real/outputs/maze_planarp_train49_7p5hz/checkpoints/epoch_0209.pt
```

这里必须区分**现有部署行为**和**实验性短序列解码**：

- 主实验均为 `full_then_truncate`，selector 只改变执行前缀长度。
- Threading 当前 `predict_action(required_only, requested_steps=8)` 会按完整 action group
  补齐，实际仍生成 20 步；Maze 当前 `predict_action` 只接受 `full_then_truncate`。
- 本 benchmark 的短序列条件直接调用底层 `policy.policy.generate`，保留完整 coarse plan、
  原始 prompt 和 chunk ID，只把末尾 action group 缩短到 8 / 4 步，并匹配对应的热图特征数量。
  模型权重、训练 horizon 和部署代码均未修改。因此它测量的是实验性 `generate k` 的成本，
  不能解释为当前部署参数已经实现这些收益。

两个模型始终解码一个 coarse-plan group 和一个 action group。缩短序列减少 action token
和热图预测数量，不减少 group 数。Threading 每步有 6 个空间 token，Maze 每步有 2 个；
两者没有 PushBox 的首位 stale action，不需要额外生成 `k+1` 步。

## 测量方法

1. 按训练时的 split（seed 42，validation ratio 0.2）读取训练集第 0 个样本；
   同一任务的长短序列使用同一观测，不使用未来观测或训练 action label 作为解码输入。
2. 在计时前完成点云渲染、视觉编码、prompt 与上下文构造，并缓存视觉 token 和空间特征。
3. 仅运行底层 `policy.policy.generate`：包含 coarse-plan 与 action 解码及空间热图预测，
   不包含输出反投影、Cartesian delta 计算、selector、数据加载或 GPU 数据搬运。
4. `eval()`、`torch.inference_mode()`、`sample=False`、batch size 1、FP32，无 autocast。
5. 每种生成长度预热 30 次，再测 12 批，每批连续调用 50 次；每批交替长短条件的顺序。
   使用 CUDA events，每批结束显式同步，以批耗时除以 50 得到单次 latency。
   Batch SD 是 12 个批均值的样本标准差，不是逐调用标准差或跨观测标准差。
6. 核验输出长度与有限值；完整长度的底层输出与生产推理路径捕获的原始调用逐元素一致。

使用 NVIDIA GeForce RTX 4060 Ti（16 GB），驱动 560.35.05。
PyTorch 为 `2.10.0+cu128`，运行时 CUDA 为 `12.8`。
GPU 同时承担桌面显示；两个任务顺序测量，不并行运行 benchmark。
这些数值来自固定观测的重复测量，不代表训练集全部观测上的 latency 分布。

## 实测结果

| 任务 | 生成条件 | action steps | action tokens | plan + action tokens | ARP latency/call | Batch SD |
|---|---|---:|---:|---:|---:|---:|
| Threading H20 | 完整生成 | 20 | 120 | 144 | **99.326 ms** | 0.404 ms |
| Threading H20 | 实验性 required-only | 8 | 48 | 72 | **50.870 ms** | 0.284 ms |
| Maze H10 | 完整生成 | 10 | 20 | 28 | **22.345 ms** | 0.266 ms |
| Maze H10 | 实验性 required-only | 4 | 8 | 16 | **15.079 ms** | 0.141 ms |

token 数不包含固定 prompt（Threading 6 个、Maze 2 个空间 token）。

- Threading：20 → 8 步，latency 减少 **48.456 ms（48.78%）**；action token 减少 60%，含 coarse plan 的生成 token 总量减少 50.00%。
- Maze：10 → 4 步，latency 减少 **7.266 ms（32.52%）**；action token 减少 60%，含 coarse plan 的生成 token 总量减少 42.86%。

两者仍需生成完整 coarse plan，且每次都有固定解码开销，因此 token 减幅不等于 latency 减幅。

训练输入定位：

| 任务 | 训练集文件（相对于仓库根目录） | episode / frame |
|---|---|---|
| Threading | `data/datasets/threading_combined_80_mvt_cam1_7p5hz.h5` | `episode_000001` / 0 |
| Maze | `data/datasets/maze_train49_mvt_7p5hz.h5` | `episode_000000` / 0 |

Maze checkpoint 保存的是搬移前路径 `data/maze_train49_mvt_7p5hz.h5`；本次通过 `--dataset` 指向现存的 `data/datasets/maze_train49_mvt_7p5hz.h5`，split 参数沿用原训练 `launch.json`。

输出核验还发现：实验性短序列并不保证与完整预测前缀相同。
Threading 在本观测上的短序列与完整前缀，空间 token 像素坐标最大绝对差为 3 像素。
Maze 在本观测上的短序列与完整前缀，空间 token 像素坐标最大绝对差为 0 像素。
这不是物理动作误差或任务质量评估；本次未测量其闭环影响。

## 复现与原始记录

脚本：[benchmark_main_arp_latency.py](../scripts/diagnostics/benchmark_main_arp_latency.py)。
从仓库根目录顺序运行：

```bash
cd /home/huiyuan/teleoperation
LD_LIBRARY_PATH=/home/huiyuan/miniconda3/envs/pushbox/lib \
/home/huiyuan/miniconda3/envs/pushbox/bin/python \
  threading_real/scripts/diagnostics/benchmark_main_arp_latency.py \
  --task threading --warmup 30 --batches 12 --calls 50 \
  --output data/analysis/arp_decode_latency_20260922/threading.json

LD_LIBRARY_PATH=/home/huiyuan/miniconda3/envs/pushbox/lib \
/home/huiyuan/miniconda3/envs/pushbox/bin/python \
  threading_real/scripts/diagnostics/benchmark_main_arp_latency.py \
  --task maze --warmup 30 --batches 12 --calls 50 \
  --dataset data/datasets/maze_train49_mvt_7p5hz.h5 \
  --output data/analysis/arp_decode_latency_20260922/maze.json
```

原始记录：[Threading JSON](../../data/analysis/arp_decode_latency_20260922/threading.json)、
[Maze JSON](../../data/analysis/arp_decode_latency_20260922/maze.json)。
包含 checkpoint / 脚本 SHA-256、数据与样本定位、PyTorch / CUDA 版本、输出核验以及各批原始计时。

本次只测 decoding latency，不测闭环成功率。部署文档中的 fixed / selector 成功率对比
均来自完整生成，不能用作 full / required 的成功率 A/B。
下节补测实际 selector 分布中的中间长度，估算固定调用分布下的解码节省；不据此推算端到端加速比例。

## 按实际 selector chunk 分布加权

读取部署文档主实验的原始 trace，按 `cycle=1` 或 episode 编号变化划分 trace。
Threading 使用新 h8 progress selector，共 20 段、168 次决策；Maze 使用历史 h4 selector，
共 20 段、344 次决策。所有记录均为 `executed=true`，未混入旧 Threading selector 或 AAC。

对实际出现的每种长度 k，使用同一训练观测、同一 checkpoint，按上述 30 次预热、12×50 次调用
补测其 decoding latency；端点沿用上一节实测值。各长度均为实测值，不做线性插值。
设长度 k 的次数为 n_k，单次解码延迟为 L_k，完整长度为 H，总次数 N = Σ n_k：

```text
full 平均解码时间/call     = L_H
required 平均解码时间/call = Σ(n_k × L_k) / N
总节省时间                 = N × L_H − Σ(n_k × L_k)
节省比例                   = 1 − Σ(n_k × L_k) / (N × L_H)
平均每段 trace 节省        = 总节省时间 / 20
```

这是“保持已有 selector 决策与调用次数不变，只缩短每次生成长度”的估算。
当前部署仍完整生成 H，因此当前实际实现的这项解码节省为 0；下面是采用实验性短序列解码的预计收益。
重新闭环执行后，动作与观测可能改变，selector 分布和调用次数也可能改变。

| 任务 | chunk k | 次数 | 占比 | 实测 ARP latency/call | Batch SD |
|---|---:|---:|---:|---:|---:|
| Threading H20 | 8 | 106 | 63.10% | 50.870 ms | 0.284 ms |
| Threading H20 | 10 | 2 | 1.19% | 59.093 ms | 0.597 ms |
| Threading H20 | 11 | 2 | 1.19% | 62.779 ms | 1.318 ms |
| Threading H20 | 13 | 1 | 0.60% | 71.062 ms | 1.437 ms |
| Threading H20 | 15 | 1 | 0.60% | 79.204 ms | 0.894 ms |
| Threading H20 | 18 | 3 | 1.79% | 91.191 ms | 0.880 ms |
| Threading H20 | 19 | 5 | 2.98% | 94.927 ms | 0.348 ms |
| Threading H20 | 20 | 48 | 28.57% | 99.326 ms | 0.404 ms |
| Maze H10 | 4 | 180 | 52.33% | 15.079 ms | 0.141 ms |
| Maze H10 | 5 | 12 | 3.49% | 15.100 ms | 0.118 ms |
| Maze H10 | 6 | 9 | 2.62% | 16.094 ms | 0.131 ms |
| Maze H10 | 7 | 2 | 0.58% | 17.281 ms | 0.146 ms |
| Maze H10 | 8 | 5 | 1.45% | 18.488 ms | 0.251 ms |
| Maze H10 | 9 | 6 | 1.74% | 19.621 ms | 0.168 ms |
| Maze H10 | 10 | 130 | 37.79% | 22.345 ms | 0.266 ms |

**按实际分布加权的解码时间：**

| 任务 | full / call | required / call | 节省 / call | 解码时间减少 |
|---|---:|---:|---:|---:|
| Threading H20 | 99.326 ms | 67.274 ms | **32.052 ms** | **32.27%** |
| Maze H10 | 22.345 ms | 17.994 ms | **4.351 ms** | **19.47%** |

**固定现有调用次数时的累计估算：**

| 任务 | 决策数 / trace 数 | full 合计 | required 合计 | 合计节省 | 平均每段 full → required | 平均每段节省 |
|---|---:|---:|---:|---:|---:|---:|
| Threading H20 | 168 / 20 | 16.687 s | 11.302 s | **5.385 s** | 0.834 → 0.565 s | **0.269 s** |
| Maze H10 | 344 / 20 | 7.687 s | 6.190 s | **1.497 s** | 0.384 → 0.309 s | **0.075 s** |

因此，在保持当前 selector 调用分布的假设下，实验性 required-only 可使 Threading 的 ARP 解码时间减少约 **32.3%**，Maze 减少约 **19.5%**。
完整长度调用仍分别占 28.6% 和 37.8%，所以分布加权节省低于仅比较最短长度与完整长度的 48.8% / 32.5%。

原始分布：[Threading h8 trace](../../data/analysis/selector_eval_matched_20260916/logs/threading20_selector_h8_progress.jsonl)、
[Maze h4 trace](../../data/analysis/selector_eval_20260914_142324/logs/maze10_selector_h4.jsonl)。
补测记录：[Threading 中间长度](../../data/analysis/arp_decode_latency_20260922/threading_intermediate.json)、
[Maze 中间长度](../../data/analysis/arp_decode_latency_20260922/maze_intermediate.json)。
加权计算和逐 trace 估算保存在 [selector_weighted.json](../../data/analysis/arp_decode_latency_20260922/selector_weighted.json)，
包含输入 trace 的 SHA-256、截止时间戳和逐长度频数。

在仓库根目录复现补测和统计（两项 GPU 测量顺序执行）：

```bash
LD_LIBRARY_PATH=/home/huiyuan/miniconda3/envs/pushbox/lib \
/home/huiyuan/miniconda3/envs/pushbox/bin/python \
  threading_real/scripts/diagnostics/benchmark_main_arp_latency.py \
  --task threading --lengths 10 11 13 15 18 19 \
  --output data/analysis/arp_decode_latency_20260922/threading_intermediate.json

LD_LIBRARY_PATH=/home/huiyuan/miniconda3/envs/pushbox/lib \
/home/huiyuan/miniconda3/envs/pushbox/bin/python \
  threading_real/scripts/diagnostics/benchmark_main_arp_latency.py \
  --task maze --lengths 5 6 7 8 9 \
  --dataset data/datasets/maze_train49_mvt_7p5hz.h5 \
  --output data/analysis/arp_decode_latency_20260922/maze_intermediate.json

python threading_real/scripts/diagnostics/summarize_selector_decode_latency.py
```

这里的每段 trace 平均值覆盖全部 20 段，不是只统计成功 episodes。
它只包含 coarse-plan 与 action 解码，不包含视觉、selector、通信、机器人运动或等待；
不能把解码节省百分比当作整个任务耗时的下降比例。

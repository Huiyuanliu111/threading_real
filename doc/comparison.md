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

## ARP 解码占完整推理的比例

为避免把不同计时范围直接相除，另在同一次完整 selector 推理内部记录嵌套 CUDA events。
完整推理路径使用主实验的 selector checkpoint 和 `full_then_truncate`：

```text
已在 GPU 的训练观测 → 点云渲染 / 视觉编码 → selector → 完整 ARP 解码 → 动作后处理
```

仅预先加载观测；每次重新运行视觉编码，不复用视觉缓存。使用与上文相同的训练样本、model 权重、
FP32、batch size 1 和确定性解码。每项任务预热 30 次，再测 12×10 次完整调用。
每次在最外层结束后同步；解码边界只记录 events，不额外同步或拆开推理链。
下表的比例为同一批调用的平均 ARP latency / 平均完整推理 latency。

| 任务 | 完整推理 / call | 其中 ARP 解码 | 点云渲染 + 视觉编码 | 其余 selector / 上下文准备 / 后处理 | ARP 占比 |
|---|---:|---:|---:|---:|---:|
| Threading H20 | **132.106 ms** | 99.187 ms | 25.511 ms | 7.408 ms | **75.08%** |
| Maze H10 | **52.767 ms** | 22.531 ms | 25.658 ms | 4.578 ms | **42.70%** |

“其余”是完整时间减去解码和视觉时间的残差，不是单独测得的 selector 延迟。
完整推理的 batch SD 分别为 0.415 ms 和 0.910 ms。
同步 wall time 均值分别为 132.123 ms 和 52.780 ms，与 CUDA-event 总时间接近。

这里的完整推理不含相机读取、观测构建和 CPU→GPU 搬运、部署动作检查、网络通信或机器人执行。
仍然是各任务单个训练观测的离线测量，不是实机各阶段观测的平均值。
嵌入完整推理时的 ARP 延迟与之前缓存特征的隔离测量略有差异；占比采用本节同时测得的值。

若仅用上节加权节省除以本节完整推理时间，并假设视觉、selector、后处理与调用分布均不变，
则完整**模型推理**时间的预计降幅为：

- Threading：32.052 / 132.106 ≈ **24.26%**。
- Maze：4.351 / 52.767 ≈ **8.25%**。

以上降幅为跨两组 microbenchmark 组合的估算，不是完整 required-only 推理或实机任务耗时的直接实测。

原始记录：[Threading 完整推理](../../data/analysis/arp_decode_latency_20260922/threading_full_inference.json)、
[Maze 完整推理](../../data/analysis/arp_decode_latency_20260922/maze_full_inference.json)。
脚本：[benchmark_selector_inference_latency.py](../scripts/diagnostics/benchmark_selector_inference_latency.py)。
原始 JSON 保存逐调用分项计时、selector 文件 SHA-256、模型和观测定位。

从仓库根目录顺序复现：

```bash
LD_LIBRARY_PATH=/home/huiyuan/miniconda3/envs/pushbox/lib \
/home/huiyuan/miniconda3/envs/pushbox/bin/python \
  threading_real/scripts/diagnostics/benchmark_selector_inference_latency.py \
  --task threading --warmup 30 --batches 12 --calls 10 \
  --output data/analysis/arp_decode_latency_20260922/threading_full_inference.json

LD_LIBRARY_PATH=/home/huiyuan/miniconda3/envs/pushbox/lib \
/home/huiyuan/miniconda3/envs/pushbox/bin/python \
  threading_real/scripts/diagnostics/benchmark_selector_inference_latency.py \
  --task maze --warmup 30 --batches 12 --calls 10 \
  --output data/analysis/arp_decode_latency_20260922/maze_full_inference.json

```

## 实机循环还有哪些耗时：已有记录核查

模型推理约 0.1 秒不等于一次机器人决策循环约 0.1 秒。现有 runner 顺序经历：
相机读取 → 状态 / FK / 点云构建与搬运 → 模型推理 → 动作检查与轨迹安装 →
发送并执行 chunk → 同步等待 → 下一次观测。

从主实验原始 trace 重新统计，保留首次调用和所有有效记录：

| 项目 | Threading H20 selector | Maze H10 selector |
|---|---:|---:|
| 实机推理时间 | 168 次，均值 129.192 ms，中位 126.547 ms，P95 133.763 ms，最大 460.317 ms | trace 未记录独立推理耗时 |
| 同一段内相邻决策记录间隔 | 148 个间隔，均值 1.808 s，中位 1.387 s，P95 3.099 s | 324 个间隔，均值 1.019 s，中位 0.707 s，P95 1.551 s |
| 7.5 Hz 下的名义动作时长 | 8 步 1.067 s；20 步 2.667 s | 4 步 0.533 s；10 步 1.333 s |

推理字段 `total_seconds` 的代码计时范围包含模型前向和结果取回 CPU，不包含前面的观测构建。
决策时间戳写在动作等待之后；相邻记录只在 episode 相同且 cycle 连续时相减，不跨手动复位间隔。
它们是完成记录之间的 wall-time 间隔，并非独立网络 RTT，也不是纯 GPU 时间。
Maze 个别间隔小于名义动作时长（最小 0.197 s），所以不能直接用名义时长反推出每次完整执行的耗时；
现有 trace 缺少发送 / 等待完成标记，未据此推断网络时间。

Threading 对相同的 148 个间隔，逐条扣除下一次记录对应的 `k/7.5` 后，平均剩余 342.012 ms；
再扣除该次实测 `total_seconds`，平均剩余 **214.902 ms**。
这是以名义动作时长为基准的混合残差，含相机读取、状态读取、FK / 点云构建、搬运、RPC、
轨迹安装、轮询和实际发送时长相对名义时长的偏差，不能全部归为通信。
Threading 默认每 20 ms 轮询一次，连续 3 次满足完成条件才返回；Maze 默认每 20 ms 轮询一次。
默认 500 Hz UDP 发送周期为 2 ms，发送周期同样不是网络延迟。

### 能否确定通信时间

ARP 主实验在本机推理，不需要把图像发给远程 GPU。机器人侧主要是 UDP 动作发送 / 状态回传，
另有 XML-RPC 调用，例如 Threading 每次读取夹爪宽度的 `getGripperWidth()`。
当前主实验 trace 没有网络 RTT、控制端接收确认或跨主机对时数据，因此**机器人通信延迟尚无可靠独立实测值**。

另找到一条 [OpenPI 远程部署诊断](../../data/analysis/pi05_openpi_remote_eval/logs/sync_diagnostics.jsonl)，
属于其他策略、单个最终超时的循环，只能作为计时字段的实例，不能代表 ARP：

| 该单次记录的阶段 | 耗时 |
|---|---:|
| 相机 read 调用 | 0.704 ms |
| 相机 read 结束 → observation 构建结束 | 22.947 ms |
| 远程推理调用 | 145.896 ms |
| 推理结束 → 提交轨迹 | 4.501 ms |
| 提交轨迹 → 安装完成 | 30.032 ms |
| 安装后等待至超时 | 5.003 s |

相机 read 耗时不等于图像曝光到使用时的年龄；轨迹安装耗时也不等于网络传输。
该记录的 83 个状态样本，缓存年龄均值 0.970 ms、P95 1.947 ms。
源码在本机收到 UDP 数据并写入缓存时打时间戳，因此这是**接收后缓存年龄，不是单向网络延迟或 RTT**。

[OpenPI 历史延迟文档](../../doc/deploy_pi05_openpi.md) 另记载：2026-09-21 的 H50 / A40 测量，
远程往返平均 163 ms，服务器内部推理 157 ms，传输、序列化等合计约 6 ms。
这是本机 ↔ GPU 服务器链路，不是本机 ↔ 机器人链路；本次未找到该文档链接的原始
`latency_comparison.json/md`，故只保留为历史文档记录，不作为当前网络测量结论。

本次原始统计保存在 [deployment_timing_evidence.json](../../data/analysis/arp_decode_latency_20260922/deployment_timing_evidence.json)。
要独立测机器人通信，需要增加只读 RPC 往返计时或带序号的发送 / 控制端接收确认；
单向网络延迟还需要跨主机时钟同步。本次只核查已有文件，未连接或操作机器人。

## Selector 独立耗时（2026-09-23 补测）

在完整 `predict_with_selector` 调用中，为 `select_from_visual` 单独记录 CUDA events。
范围包含共享视觉 token 的整理 / 池化、selector 网络前向、概率计算和 chunk 长度选择；
不含共享视觉编码器，不含后续 `chunk_sizes.max().item()` 读取 Python 整数，也不含 ARP 解码。
仍使用主实验 checkpoint、训练集第 0 个观测、batch size 1、FP32、RTX 4060 Ti；
预热 30 次，测量 12×10 次完整调用，最外层同步，分项内部不额外同步。

| 任务 | Selector latency/call | Batch SD | 本轮完整推理/call | Selector 占比 |
|---|---:|---:|---:|---:|
| Threading H20 / h8 selector | **2.763 ms** | 0.089 ms | 128.820 ms | **2.14%** |
| Maze H10 / h4 selector | **2.947 ms** | 0.129 ms | 53.334 ms | **5.53%** |

两组 selector 约需 **3 ms/次**。这不是“从原始图像开始”的 selector 全流程时间：
其视觉输入复用动作模型本来就需要的编码结果。本节直接测量 selector 调用，
此前的 7.408 / 4.578 ms 是 selector、上下文准备和动作后处理的合计残差，不能当作 selector 本身耗时。
占比使用本轮同时测得的完整推理时间；不同轮次的 GPU 时间有波动，保留前一天结果不覆盖。

原始结果：[Threading](../../data/analysis/arp_decode_latency_20260923/threading_selector_timing.json)、
[Maze](../../data/analysis/arp_decode_latency_20260923/maze_selector_timing.json)。
复用并扩展 [完整推理计时脚本](../scripts/diagnostics/benchmark_selector_inference_latency.py)，在仓库根目录顺序执行：

```bash
LD_LIBRARY_PATH=/home/huiyuan/miniconda3/envs/pushbox/lib \
/home/huiyuan/miniconda3/envs/pushbox/bin/python \
  threading_real/scripts/diagnostics/benchmark_selector_inference_latency.py \
  --task threading --warmup 30 --batches 12 --calls 10 \
  --output data/analysis/arp_decode_latency_20260923/threading_selector_timing.json

LD_LIBRARY_PATH=/home/huiyuan/miniconda3/envs/pushbox/lib \
/home/huiyuan/miniconda3/envs/pushbox/bin/python \
  threading_real/scripts/diagnostics/benchmark_selector_inference_latency.py \
  --task maze --warmup 30 --batches 12 --calls 10 \
  --output data/analysis/arp_decode_latency_20260923/maze_selector_timing.json

```

# Threading `required_only` 闭环评估结果

本文只记录 `required_only` 推理结果：每次仅生成本次准备执行的 `k` 个 action
tokens，执行完后重新观测和规划。不包含“生成完整 horizon 再截断”的实验。

## 实验设置

- checkpoint：`outputs/2026-08-07/13-33-44/checkpoints/latest.ckpt`；
- EMA 权重，prediction horizon 为 10；
- `Threading_D0`，robosuite 1.4.1；
- D0 的木桩底座与圆环（`tripod`）初始位姿固定：相对桌面参考点
  `x=0.0 m`、`y=-0.15 m`、绕 Z 轴旋转 `π/2`，不随 episode 随机化；needle
  的初始位姿仍然随机。这里的“固定”指固定初始放置，并非将底座焊接到桌面；源码写在 [envs/threading_env.py]
- 数据集：`data/threading/threading_d0.hdf5`；
- 配置：v2_gmm_noplan.yaml
- 平移 action scale：`0.25`；
- EEF state、`top45 + wrist` 两路 84×84 图像；
- seed 10000–10199，每组 200 episodes。

## 完整任务：200 episodes

每个 episode 最多 1000 个环境步。`policy calls` 是每个 episode 的平均策略推理
次数；相对变化以 fixed 4 为基准。

| 策略 | 成功数 | 成功率（95% CI） | 超时率 | 平均环境步数 | 平均 policy calls | calls 变化 |
|---|---:|---:|---:|---:|---:|---:|
| fixed 2 | 129/200 | 64.5%（57.7%–70.8%） | 35.5% | 723.83 | 362.05 | +133.9% |
| fixed 4 | 178/200 | 89.0%（83.9%–92.6%） | 11.0% | 617.98 | 154.81 | 基准 |
| fixed 6 | 151/200 | 75.5%（69.1%–80.9%） | 24.5% | 680.11 | 113.72 | -26.5% |
| fixed 8 | 112/200 | 56.0%（49.1%–62.7%） | 44.0% | 770.17 | 96.47 | -37.7% |
| fixed 10 | 46/200 | 23.0%（17.7%–29.3%） | 77.0% | 915.03 | 91.61 | -40.8% |
| **selector 4/10** | **184/200** | **92.0%（87.4%–95.0%）** | **8.0%** | 642.29 | **127.82** | **-17.4%** |
| spatial rule 4/10 | 181/200 | 90.5%（85.6%–93.8%） | 9.5% | 643.92 | 130.56 | -15.7% |

### 结论

- learned selector 4/10 的成功率最高，为 92.0%；相比 fixed 4 高 3 个百分点，
  policy calls 减少 17.4%。
- spatial rule 4/10 达到 90.5%，policy calls 减少 15.7%。
- fixed chunk 的最佳值是 4；过短的 chunk 2 和过长的 chunk 6–10 都降低成功率。
- selector、spatial rule 与 fixed 4 的置信区间重叠，200 episodes 尚不足以断言成功率
  存在统计显著差异；可确认的收益是保持约 90%–92% 成功率并减少策略调用。

完整任务结果根目录：

`outputs/2026-08-07/13-33-44/prediction_mode_comparison_200/required_only`

各策略结果位于该目录下的：

```text
fixed_2/threading_rollouts/run_0000
fixed_4/threading_rollouts/run_0000
fixed_6/threading_rollouts/run_0000
fixed_8/threading_rollouts/run_0000
fixed_10/threading_rollouts/run_0000
selector_4_10/threading_rollouts/run_0000
spatial_4_10/threading_rollouts/run_0000
```

## 子任务：200 episodes

子任务同样使用 `required_only`。由于平移 scale 为 0.25，环境步数预算同步调整为：

- Subtask 1 Approach：400；
- Subtask 2 Pick：480；
- Subtask 3 Insert：1200。

| 子任务 | chunk 2 | chunk 4 | chunk 6 | chunk 8 | chunk 10 |
|---|---:|---:|---:|---:|---:|
| Approach | 186/200（93.0%） | 198/200（99.0%） | **200/200（100%）** | **200/200（100%）** | **200/200（100%）** |
| Pick | 189/200（94.5%） | **200/200（100%）** | **200/200（100%）** | **200/200（100%）** | **200/200（100%）** |
| Insert | **197/200（98.5%）** | 196/200（98.0%） | 193/200（96.5%） | 186/200（93.0%） | 177/200（88.5%） |

子任务结果表明：自由接近和抓取阶段可以使用较长 chunk；插入阶段随 chunk 增大而
持续退化。因此适合的阶段策略是 Approach=10、Pick=10、Insert=2。

子任务结果保存在：

`outputs/2026-08-07/13-33-44/subtasks_required_only_budget4x_200`

其中：

- `chunk_comparison.json`：聚合指标；
- `evaluation_details.json`：metadata、起始状态与逐 episode 结果；
- `episodes.csv`：逐 episode 指标；
- `chunk_comparison.png`：对比图。

## Selector 随时间的 chunk 选择

使用与主实验相同的 `required_only`、translation scale 0.25 和 seed 10000 起始配置，
额外运行 50 episodes 并记录每次 selector 决策。结果为 44/50 成功（88.0%），共
记录 6445 次决策。chunk 10 占决策次数的 17.2%；按每次决策实际执行的环境步数
加权后，chunk 10 占 34.2%。后一个数字更能反映长 chunk 实际覆盖的轨迹时间。

因此，两种口径应明确区分：

| 统计口径 | chunk 4 | chunk 10 | 含义 |
|---|---:|---:|---|
| selector 决策次数 | 82.8%（5338/6445） | **17.2%（1107/6445）** | selector 每次被调用时输出哪个 chunk |
| 实际执行环境步数 | 65.8%（21282/32352） | **34.2%（11070/32352）** | 整条 rollout 轨迹由哪个 chunk 覆盖 |

换言之，如果问题是“selector 有多少次选择 chunk 10”，答案约为 **17%**；如果问题是
“整个 episode 中有多少动作时间由 chunk 10 执行”，答案约为 **34%**。

| 归一化 episode 进度 | 主要阶段 | chunk 10 环境时间占比 |
|---|---|---:|
| 0%–20% | 自由接近 | 92.3% |
| 20%–50% | 抓取精调 | 11.5% |
| 50%–70% | 抬起与运输 | 53.7% |
| 70%–100% | 插入精调 | 3.1% |

selector 的典型压缩序列是 `10→4→10→4`：50 条 episode 中有 45 条采用该序列，
每条 episode 平均切换 3.28 次。用 spatial rule 仅作事后阶段标注（不参与 selector
推理）时，chunk 10 在 `free_approach`、`free_transport`、`pick_precision` 和
`insert_precision` 中的环境时间占比分别为 96.5%、93.0%、10.0% 和 1.7%。这说明
selector 的选择与任务阶段一致：自由运动使用长 chunk，抓取和插入精调使用短 chunk。

成功 episode 中 chunk 10 的环境时间占比为 36.6%，失败 episode 为 23.8%。这里只能
视为相关性，不能据此认为更多选择 chunk 10 会导致成功，因为失败 episode 通常在插入
阶段停留至超时。

原始结果和分析保存在：

`outputs/2026-08-07/13-33-44/selector_temporal_required_only_50/threading_rollouts/run_0000`

- `chunk_trace.csv` / `chunk_trace.json`：逐 policy call 的 step、chunk、置信度、概率、
  spatial phase 和实际执行步数；
- `selector_temporal_analysis/normalized_time_bins.csv`：归一化时间分箱；
- `selector_temporal_analysis/absolute_step_bins.csv`：绝对环境步分箱；
- `selector_temporal_analysis/chunk_choice_over_normalized_time.png`：随 episode 进度变化；
- `selector_temporal_analysis/chunk_choice_over_environment_steps.png`：随绝对环境步变化；
- `selector_temporal_analysis/chunk_choice_success_vs_failure.png`：成功/失败轨迹对照；
- `selector_temporal_analysis/chunk_choice_by_spatial_region.png`：按 spatial phase 对照。

## Selector 数据与模型

| 内容 | 保存位置 |
|---|---|
| 训练数据 | `data/chunk_selector/threading_arp_spatial_v2_chunk_4_10.hdf5` |
| 数据摘要 | `data/chunk_selector/threading_arp_spatial_v2_chunk_4_10.summary.json` |
| selector 权重 | `outputs/chunk_selector/threading_4_10/chunk_selector.safetensors` |
| selector 配置 | `outputs/chunk_selector/threading_4_10/chunk_selector_config.json` |
| 训练历史 | `outputs/chunk_selector/threading_4_10/training_history.json` |

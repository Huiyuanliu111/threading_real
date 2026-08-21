# PushBox crossing `required_only` 闭环评估结果

本文记录 PushBox crossing 完整任务在 `required_only` 模式下的闭环评估结果：每次
只生成本轮准备执行的 action tokens，执行完对应 chunk 后重新观测和规划。不包含
“生成完整 prediction horizon 再截断”的实验。

## 实验设置

- 动作策略 checkpoint：`/home/huiyuan/实验结果/pushbox/23-08-49/checkpoints/epoch=0140-val_loss=-30.500.ckpt`；
- EMA 权重，prediction horizon 为 20；
- PushBox 完整任务，robosuite 1.5.2；
- 配置：`pushbox/configs/arp.yaml`；
- 原始训练数据集：`data/datagen`；
- agent state、box position、`top45 + sideview` 两路 96×96 图像；
- 使用 `V2_FIXED_OBSTACLES`，所有 episode 的障碍物位置固定；
- Box X 在 `[0.00, 0.05] m` 上按三角分布采样，众数为 `0.05 m`，概率密度
  从 `0.00 m` 向 `0.05 m` 线性增加；
- Box Y 在 `[-0.25, -0.20] m` 上均匀采样；
- seed 10000–10999，每组 1000 episodes；
- 所有方法逐 episode 使用完全相同的 seed、Box 初始位置和障碍物配置；
- 每个 episode 最多 500 个环境步。

## Spatial rule

spatial rule 以箱子当前的 Y 坐标决定执行 chunk，规则为 **19/5/19**：

| Box Y 区域 | chunk |
|---|---:|
| `Y < -0.20 m` | 19 |
| `-0.20 m ≤ Y ≤ +0.15 m` | 5 |
| `Y > +0.15 m` | 19 |

两个边界 `-0.20 m` 和 `+0.15 m` 均属于 chunk 5 区域。该规则的目的，是在进入
crossing 精细操作区后提高闭环重规划频率，在 crossing 前后的自由运动区使用较长
chunk。

## 完整任务：1000 episodes

`policy calls` 是每个 episode 的平均策略推理次数。由于失败 episode 通常会因机械臂
碰撞、箱子掉落或越界而提前结束，平均环境步数不能单独作为执行效率指标。95% CI
使用 Wilson score interval。

| 策略 | 成功数 | 成功率（95% CI） | 平均环境步数 | 平均 policy calls | 平均生成 action tokens | 失败类型 |
|---|---:|---:|---:|---:|---:|---|
| fixed 5 | 683/1000 | 68.3%（65.4%–71.1%） | 184.00 | 37.20 | 223.21 | arm collision=75，box fell=106，out of bounds=136 |
| fixed 10 | 713/1000 | 71.3%（68.4%–74.0%） | 140.80 | 14.52 | 159.72 | arm collision=287 |
| fixed 15 | 561/1000 | 56.1%（53.0%–59.1%） | 120.66 | 8.52 | 136.32 | arm collision=439 |
| fixed 19 | 504/1000 | 50.4%（47.3%–53.5%） | 113.76 | 6.45 | 128.96 | arm collision=496 |
| **spatial rule 19/5/19** | **795/1000** | **79.5%（76.9%–81.9%）** | 164.94 | **26.39** | 198.96 | arm collision=146，box fell=28，out of bounds=31 |
| **selector 5/19（epoch 37）** | **783/1000** | **78.3%（75.6%–80.7%）** | 165.81 | 27.33 | **198.78** | arm collision=141，box fell=30，out of bounds=46 |

### Fixed chunk 与 spatial rule

- fixed chunk 的成功率排序为 fixed 10（71.3%）、fixed 5（68.3%）、
  fixed 15（56.1%）、fixed 19（50.4%）。与 200-episode 结果不同，1000 episodes
  下 fixed 10 高于 fixed 5。长 chunk 的主要失败原因仍是机械臂碰撞；fixed 19 的 496 个
  失败 episode 全部为 `arm_collision`。
- spatial rule 19/5/19 成功率最高，为 79.5%；相比 fixed 5 高 11.2 个百分点，
  同时平均 policy calls 从 37.20 降到 26.39，减少 29.1%。两者的配对精确
  McNemar 检验 `p=2.13e-10`。
- spatial rule 相比 fixed 10 高 8.2 个百分点，配对精确 McNemar 检验
  `p=8.28e-7`；相比 fixed 15 和 fixed 19 分别高 23.4 和 29.1 个百分点，
  配对精确 McNemar 检验分别为 `p=4.89e-35` 和 `p=4.03e-49`。

### Learned selector 与 spatial rule

selector 使用 spatial rule 产生的标签训练，并在相同的 1000 个初始状态上评估：

| 配对结果 | episodes |
|---|---:|
| selector 与 spatial rule 均成功 | 733 |
| 仅 selector 成功 | 50 |
| 仅 spatial rule 成功 | 62 |
| 两者均失败 | 155 |

- selector 成功率为 78.3%，比 spatial rule 的 79.5% 低 1.2 个百分点；配对精确
  McNemar 检验 `p=0.299`，没有可检测的成功率差异。
- selector 平均 policy calls 为 27.33，spatial rule 为 26.39；平均生成 action
  tokens 分别为 198.78 和 198.96，执行成本接近。
- selector 共作出 27,332 次 chunk 决策，其中 chunk 5 为 24,847 次，chunk 19 为
  2,485 次；chunk 19 占全部 policy calls 的 9.1%。
- 结果表明 learned selector 基本复现了 19/5/19 spatial rule 的完整任务表现。

## 评估结果目录

本轮 1000-episode 实验根目录：

`outputs/pushbox_triangular_x_000_005_y_m025_m020_n1000_seed10000/required_only`

各方法的原始 JSON 位于：

```text
fixed5_vs_spatial_19_5_19_m020_p015/eval_stats.json  # fixed 5 与 spatial rule
fixed_10/eval_stats.json
fixed_15/eval_stats.json
fixed_19/eval_stats.json
selector_5_19_epoch37/eval_stats.json
```

1000-episode 结果目前直接来自各方法的 `eval_stats.json`；尚未重新生成
`comparison_summary.json/csv` 和 `comparison.png`。

## Selector 数据与模型

selector 的候选 chunk 为 `{5, 19}`。训练标签直接来自上述非对称 spatial rule：
`Y < -0.20` 和 `Y > +0.15` 标为 chunk 19，中间闭区间标为 chunk 5。

特征数据从 `data/datagen` 的 100 个 episode 中提取，共 23,067 个样本：

| 标签或区域 | 样本数 |
|---|---:|
| chunk 5 / crossing precision | 20,003 |
| chunk 19 / free before crossing | 652 |
| chunk 19 / free after crossing | 2,412 |
| chunk 19 合计 | 3,064 |

训练使用 inverse-frequency class weights，并按 episode 划分训练集和验证集。最多训练
100 epochs，early-stopping patience 为 15，以验证集 macro-F1 选择最佳权重。训练在
epoch 52 停止，保存的最佳权重来自 **epoch 37**：验证 accuracy 为 99.1%，验证
macro-F1 为 0.9740，验证 loss 为 0.0259。

| 内容 | 保存位置 |
|---|---|
| 原始 PushBox 数据 | `data/datagen` |
| selector 特征数据 | `data/chunk_selector/pushbox_arp_spatial_v2_region_m0p20_p0p15_chunk_5_19.hdf5` |
| selector 数据摘要 | `data/chunk_selector/pushbox_arp_spatial_v2_region_m0p20_p0p15_chunk_5_19.summary.json` |
| selector checkpoint 目录 | `outputs/chunk_selector/pushbox_5_19_region_m0p20_p0p15` |
| selector 权重（最佳 epoch 37） | `outputs/chunk_selector/pushbox_5_19_region_m0p20_p0p15/chunk_selector.safetensors` |
| selector 配置 | `outputs/chunk_selector/pushbox_5_19_region_m0p20_p0p15/chunk_selector_config.json` |
| selector 训练历史 | `outputs/chunk_selector/pushbox_5_19_region_m0p20_p0p15/training_history.json` |

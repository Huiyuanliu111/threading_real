# PushBox Spatial Rule 评估

## 实验设置

- 模型：PushBox ARP，使用 EMA 权重。
- 所有方法始终预测完整 horizon，只执行所选 chunk 的前缀，然后重新观测和规划。
- 固定障碍，200 个配对测试 seed：`30000-30199`。
- `only_full`：固定 chunk 19。
- `only_optimal`：固定 chunk 5。
- `spatial_rule`：三个阶段使用 `[19, 5, 19]`；进入走廊前 0.05 m 提前切换到 chunk 5。
- wall time 不包含自动抓取初始化，包含控制、观测渲染和同步策略推理。

## 旧 Spatial Rule 结果

| 方法 | 成功率 | 成功 episode 平均 wall time | 全部 episode 平均推理次数 |
|---|---:|---:|---:|
| `only_full` | 130/200（65.0%） | 1.15 s | 7.49 |
| `only_optimal` | 129/200（64.5%） | 1.51 s | 26.93 |
| `spatial_rule` | 130/200（65.0%） | 1.36 s | 17.46 |

`only_full` 与 `spatial_rule` 的配对结果：两者都成功 124 个、都失败 64 个、仅 full
成功 6 个、仅 spatial rule 成功 6 个。精确 McNemar 检验 `p=1.0`，两者没有可辨别的
成功率差异。三种方法的 95% Wilson 成功率区间高度重叠。

当前 spatial rule 相比固定 chunk 5 更快，但比 full 慢约 0.21 s；成功率与 full
完全相同。因此 spatial rule 尚未带来收益，only full 仍是当前更好的配置。子任务标定
按“成功率优先、再比较 wall time”选择时，三个阶段实际也都选择 chunk 19；
`[19, 5, 19]` 是对“走廊使用短 chunk”假设的主动验证，并不是标定得到的最优阶段规则。

完整结果位于：

```text
outputs/pushbox_spatial_rule/formal_200_seed30000_margin005/eval_stats.json
```

每种方法另外重放并渲染了 10 个成功和 10 个失败 episode。视频位于：

```text
outputs/pushbox_spatial_rule/formal_200_seed30000_margin005/videos/
```

视频重放不参与 wall time 统计；性能表来自无视频的 200-episode 主评估。

## 当前穿越阈值

当前标签和评估使用对称的 0.15 m 穿越区域：

```text
box_y < -0.15 m             -> chunk 19
-0.15 m <= box_y <= 0.15 m -> chunk 5
box_y > 0.15 m              -> chunk 19
```

上面的旧 200-episode 结果对应规则 `[进入=-0.15, 离开=0.10]`；当前规则的进入阈值
相同，但离开阈值改为 `0.15 m`。

## 对称阈值标签与 Selector 训练

使用 100 条 PushBox 演示生成新数据集。每条 episode 最多取 500 帧，与 PushBox ARP
训练数据加载方式一致；原始 LeRobot 数据不修改。标签定义为：

```text
free_before_crossing: box_y < -0.15 m          -> chunk 19
crossing_precision:  -0.15 <= box_y <= 0.15 m -> chunk 5
free_after_crossing: box_y > 0.15 m           -> chunk 19
```

数据集共有 23,067 个共享视觉特征样本，特征形状为 `[2, 64]`：

| 标签 | 样本数 | 比例 |
|---|---:|---:|
| chunk 5 | 16,890 | 73.2% |
| chunk 19 | 6,177 | 26.8% |

数据集路径：

```text
data/chunk_selector/pushbox_arp_spatial_v1_threshold_0p15_chunk_5_19.hdf5
```

Selector 使用 2 层 Transformer，在完整 episode 级别做 80/20 划分，训练 20 epochs；
最佳模型来自 epoch 18，验证集 accuracy 为 97.17%，macro-F1 为 0.961。模型路径：

```text
outputs/pushbox_arp_selector_spatial_v1_threshold_0p15_chunk_5_19_epoch20/
```

## 对称阈值与 Learned Selector 闭环结果

使用相同的 200 个测试 seed `30000-30199` 重新配对运行四种方法。方法顺序按 episode
轮换；不保存视频，避免视频捕获和编码污染 wall time。当前 spatial rule 使用对称
`[-0.15, 0.15] m` 区域，不再是旧的离开阈值 `0.10 m`。

| 方法 | 成功率 | 全部 episode 平均 wall time | 成功 episode 平均 wall time | 全部 episode 平均推理次数 |
|---|---:|---:|---:|---:|
| `only_full`，chunk 19 | 130/200（65.0%） | 0.816 s | 1.120 s | 7.49 |
| `only_optimal`，chunk 5 | 129/200（64.5%） | 1.089 s | 1.500 s | 26.93 |
| `spatial_rule`，`[19, 5, 19]` | 130/200（65.0%） | 0.992 s | 1.384 s | 19.74 |
| `learned_selector`，`{5, 19}` | 127/200（63.5%） | 0.969 s | 1.364 s | 18.09 |

Learned selector 相对 only full 的配对结果为：selector 独有成功 0 个、only full 独有
成功 3 个、相同 197 个。精确 McNemar 检验 `p=0.25`；1.5 个百分点的差异没有统计
显著性，但方向不支持 selector 带来成功率提升。三个 only-full 成功而 selector 失败的
seed 为 `30030`、`30142` 和 `30186`，前两个是 `arm_collision`，最后一个是
`phase_timeout`。

效率方面：

- learned selector 比 only full 慢 0.153 s/episode，约 18.7%；在双方都成功的配对
  episode 上慢 0.249 s，约 22.2%。
- learned selector 比固定 chunk 5 快 0.120 s/episode，约 11.0%。
- learned selector 比 spatial rule 快 0.023 s/episode，约 2.3%，但配对 wall-time
  差异的 95% bootstrap 区间包含 0，不能认为存在稳定优势。
- learned selector 的 3,617 次决策中，2,942 次选择 chunk 5、675 次选择 chunk 19；
  短 chunk 占策略调用的 81.3%，按名义执行步数计算占 53.4%。它没有退化为固定短
  chunk，但仍大量使用短 chunk。

因此，当前 learned selector 虽然成功学会了离线空间标签，也能在闭环中切换长短
chunk，但没有达到“相对最佳固定 chunk 成功率非劣，同时改善效率”的要求。当前完整
任务配置仍应优先使用 only full；下一步应重点分析三个 selector 独有失败 seed 的切换
时刻，而不是继续提高离线分类准确率。

完整结果位于：

```text
outputs/pushbox_selector_eval/formal_200_seed30000_threshold015_paired4/eval_stats.json
```

## 脚本改动

- `chunk_selector/execution.py`：增加统一的 prediction/execution 长度计算；执行 chunk
  不再改变预测 horizon。
- `pushbox/policy.py`、`threading_task/policy.py`：ARP 始终生成完整计划，selector 或
  spatial rule 仅截取要执行的动作前缀。
- `envs/pushbox_env.py`：将 `UniformRandomSampler` 绑定到环境的 `np_random`，保证相同
  seed 产生相同初态。修复前的 full-task 结果不是真正的配对比较。
- `scripts/eval_subtasks.py`：校正三个子任务的阶段范围；默认严格加载 EMA；同时记录全部
  episode 和成功 episode 的 steps、wall time、推理次数；按 episode 固定策略随机种子。
- `scripts/build_pushbox_arp_chunk_dataset.py`：从 PushBox 演示计算对称空间区域标签，缓存
  ARP 共享视觉特征并生成 selector HDF5 数据集。
- `scripts/eval_pushbox_spatial_chunks.py`：完整任务配对评估支持 fixed full、fixed
  optimal、spatial rule 和 learned selector；非 selector 方法显式卸载 sidecar，轮换
  方法运行顺序并输出成功率、wall time、推理耗时、推理次数和失败原因；支持显式
  episode seed 和单方法重放。
- `tests/test_chunk_selector.py`：增加“改变执行 chunk 不改变预测 horizon”的回归测试。

相关回归测试结果：`10 passed`。

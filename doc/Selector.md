# Adaptive Chunk Selector 设计方案（当前版）

> 当前路线不使用模拟器候选分支、反事实 utility 或反事实 oracle，也不把 DAgger
> 当作标签生成方法。第一版标签只来自专家演示中的空间距离事件。
>
> 公式统一使用纯文本，避免 Markdown 数学公式渲染问题。

## 1. 目标

为 PushBox ARP 和项目内 Threading ARP 增加 Chunk Selector。
动作模型仍然预测 checkpoint 定义的完整 action chunk，Selector 只决定执行其前
多少步：

```text
完整预测 plan[t] -> 执行前 k 步 -> 重新观察和规划
```

需要区分：

- **prediction horizon**：动作模型一次最多预测多少步。
- **execution chunk / `n_action_steps`**：本次实际执行多少步。
- **replanning**：所选动作执行完后重新观察并调用动作模型。

### 1.1 同步执行

当前两种 ARP 策略都是同步执行：

```text
停止 -> 获取观测 -> 同步推理 -> 执行 k 步 -> 停止 -> 再推理
```

不存在边执行边推理。

因此：

- 推理耗时增加停止等待时间和任务总时间；
- 推理期间不会继续执行旧动作；
- 安全边界不需要补偿推理延迟；
- 大 chunk 的效率收益是减少停止和策略调用次数。

### 1.2 候选集

候选是任意有序集合：

```text
candidate_chunks = [k1, k2, ..., kM]
k1 < k2 < ... < kM
1 <= ki <= 最大 prediction horizon
```

不同动作模型使用不同候选和不同 Selector 权重。候选不要求全部被使用：没有独立价值
的候选应删除，最终只剩 `{small, full}` 也可以是正确结果。

## 2. 决策原则

Selector 要回答：

> 当前处于自由运动区，还是需要频繁重规划的精细操作区？

原则如下：

- 夹爪尚未接近物体时使用 full chunk；
- 夹爪接近物体、但物体尚未抬升时使用 pick 的最佳小 chunk；
- 物体完成抬升并在自由空间运输时恢复 full chunk；
- 物体接近插孔时使用 insert 的最佳小 chunk；
- 不追求 chunk 平滑递减。

可能的在线序列包括：

```text
full -> pick_small -> full -> insert_small
```

## 3. 标签设计

### 3.1 粗标签

粗标签来自预定义子任务的固定 chunk 闭环评估：

```text
subtask -> 该子任务中表现最好的固定 chunk
```

子任务继续使用物体距离、抓取状态、抬升高度和插入条件定义。所有候选使用相同 demo
起点和随机种子。

“最好”的选择顺序：

```text
1. 排除成功率或安全性明显更差的候选
2. 在非劣候选中选择总时间更短者
3. 再选择策略调用次数更少者
4. 仍相同时保留较大的 chunk
```

其中：

```text
total_time = control_execution_time + synchronous_inference_time
```

粗标签用于建立基线和阶段先验，不是细标签的硬上限。一个子任务内部仍可能同时包含
自由运动区和精细操作区。

### 3.2 三个空间事件

Threading 第一版只检测三个单调事件：

```text
事件 1：夹爪进入物体抓取区域
    eef_handle_distance <= 0.10 m

事件 2：物体完成抬升
    needle_z - initial_needle_z >= 0.05 m

事件 3：物体进入插孔精细区域
    needle_ring_distance <= 0.20 m
```

每条演示分别取三个条件首次成立的时刻，并要求：

```text
grasp_entry < lift_entry < insert_entry
```

这样不会因为距离在阈值附近抖动而反复切换阶段。

### 3.2.1 与子任务评估的阈值和语义对照

Label 生成与子任务评估共用 `0.10 m` 的 pick 空间入口和 `0.05 m` 的
抬升阈值，但两者的阶段语义并不完全等价：

| 事件 | Label 生成 | 子任务评估 | 是否一致 |
|---|---|---|---|
| 接近针柄 | EEF 到 handle 距离不大于 `0.10 m` 时进入 `pick_precision` | approach 成功阈值同为 `0.10 m` | 数值一致 |
| pick 起点 | 取第一次进入 `0.10 m` 的专家演示帧 | 以第一次进入 `0.10 m` 为搜索锚点，再从稳定抓取位置反推并合成高 `0.085 m` 的对齐 pregrasp | 语义不一致 |
| 抬升完成 | 第一次满足 lift height 不小于 `0.05 m` 时进入 `free_transport` | pick 成功要求仍抓住针，并且 lift height 不小于 `0.05 m` | 阈值一致，条件不同 |
| transport 起点 | lift height 不小于 `0.05 m` | insert 子任务也从 lift height 不小于 `0.05 m` 开始 | 一致 |
| 精细插入起点 | needle 到 ring 距离不大于 `0.20 m` 时进入 `insert_precision` | 当前没有独立的 `0.20 m` 子任务边界 | 不一致 |
| insert 成功 | `0.20 m` 仅用于区域切换，不代表任务成功 | 使用环境的真实插入成功条件，即 needle 几何中心进入 ring 的严格判定 | 不一致 |

因此，当前三子任务评估中的 `insert` 同时包含 `free_transport` 和
`insert_precision`。若要直接验证“运输阶段用 full、精细插入用 small”，需要把它拆成
独立的 transport 和 insert 两段，并分别进行固定 chunk 闭环评估。

### 3.3 空间区域与标签

三个事件将轨迹分成四个区域：

| 区域 | 时间范围 | 标签 |
|---|---|---|
| `free_approach` | 开始至事件 1 | `full_chunk` |
| `pick_precision` | 事件 1 至事件 2 | `pick_small_chunk` |
| `free_transport` | 事件 2 至事件 3 | `full_chunk` |
| `insert_precision` | 事件 3 至结束 | `insert_small_chunk` |

各 ARP 模型的 `full_chunk`、`pick_small_chunk`、`insert_small_chunk` 和
`candidate_chunks` 必须来自各自的闭环子任务评估。如果后续评估得到不同最优值，
只修改配置并重新生成，不手工修改已有标签。

### 3.4 距离定义

```text
eef_handle_distance
    = EEF 到 needle handle 中心的欧氏距离

lift_height
    = 当前 needle root 高度 - 初始 needle root 高度

needle_ring_distance
    = needle 几何中心到 ring 几何中心的欧氏距离
```

所有距离均直接从 HDF5 中记录的 EEF、needle 和 tripod 位姿计算，不恢复模拟器状态。

## 4. 中间候选的意义与裁剪

中间 chunk 不是为了让输出平滑递减。它必须同时比 full 更早停在关键边界前，又比
`min_chunk` 减少同步策略调用。例如候选为 `[4, 8, 12, 16]`、事件还有 10 步且
`boundary_buffer=0` 时，`8` 能停在边界前，因而有独立价值。

如果另一个候选成功率不低、安全性不差，且总时间或策略调用次数更少，则当前候选被
支配，应删除。最终只剩 `[best_small_chunk, full_chunk]` 不属于类别坍塌，而是任务
只有两个可区分的时间尺度。不要用 class weight 放大没有独立支撑的中间类别。

## 5. Selector 模型与共享特征

```text
共享 visual features [B, L, D]
        -> detach
        -> LayerNorm + Linear
        -> 2 层 TransformerEncoder
        -> CLS
        -> MLP
        -> K 个 logits
```

初始超参数：

```yaml
d_model: 256
num_layers: 2
n_heads: 4
dim_feedforward: 1024
dropout: 0.1
activation: gelu
```

同时保留 `visual tokens -> mean pooling -> MLP -> logits` 基线。如果 Transformer
没有明显优于 MLP，优先检查标签、候选可分性和输入信息。第一版只使用动作模型已经
计算的共享视觉 token：

- PushBox ARP：复用 `nobs_features`，注意原有 `+1/-1` 动作对齐；
- Threading ARP：复用 `_visual_tokens()` 返回的多相机空间 token。

若视觉特征无法预测动作变化，再消融加入 robot state 或短时 state/action delta。

## 6 Cross Obstacle 结果

### 6.1 实验设置

- PushBox ARP 使用 EMA 权重，prediction horizon 为 20。
- `only_full` 固定执行 chunk 19，`only_optimal` 固定执行 chunk 5，
  `learned_selector` 在 chunk 5 和 19 之间选择。
- 100 个配对 episode，seed 为 `33000-33099`；每个 seed 的三种方法使用相同的
  box 初始位置和 `hard` 障碍物配置。
- box 初始化范围为 `X=(-0.04, 0.02) m`、`Y=(-0.30, -0.20) m`。
- wall time 包含控制、观测渲染和同步策略推理。

### 6.2 结果

| 方法 | 成功率 | 全部 episode 平均 wall time | 成功 episode 平均 wall time | 平均推理次数 |
|---|---:|---:|---:|---:|
| `only_full` | 50/100（50%） | 1.144 s | 1.377 s | 9.85 |
| `only_optimal` | 48/100（48%） | 1.483 s | 1.845 s | 35.19 |
| `learned_selector` | 51/100（51%） | 1.374 s | 1.685 s | 25.22 |

配对成功结果如下：

- selector 对 full：仅 selector 成功 3 个，仅 full 成功 2 个，精确 McNemar
  检验 `p=1.0`。
- selector 对 small：仅 selector 成功 3 个，仅 small 成功 0 个，`p=0.25`。
- full 对 small：仅 full 成功 4 个，仅 small 成功 2 个，`p=0.6875`。

失败原因计数：

| 方法 | Arm collision | Box fell | Out of bounds |
|---|---:|---:|---:|
| `only_full` | 32 | 9 | 9 |
| `only_optimal` | 35 | 8 | 9 |
| `learned_selector` | 32 | 9 | 8 |

![Cross Obstacle selector comparison](pushbox_cross_obstacle_selector_comparison.png)

这组初始化范围没有显著拉开成功率：三种方法只相差 1-3 个百分点，失败仍主要由共同的
arm collision 主导。selector 在本组样本中成功率略高于两个固定基线，并且比固定
small 更快，但相对 full 的成功率提升不显著，不能据此认定 selector 已带来可靠的闭环
收益。完整数据位于：

```text
outputs/pushbox_selector_eval/confirm_init_xm004_002_ym030_m020_n100_seed33000/eval_stats.json
```

## 7. 数据与验收

### 7.1 数据生成

```text
1. 缓存每个专家时刻的共享视觉特征
2. 计算三个空间距离、事件索引、区域和标签
```

```text
索引：episode_id, decision_step, subtask
标签：label, execution_chunk, spatial_region
距离：eef_handle_distance, lift_height, needle_ring_distance
事件：grasp_entry, lift_entry, insert_entry
版本：label_rule_version, checkpoint_hash
```

要求：

- 最后不足最大候选长度的样本不能自动标成短 chunk；
- 按完整 episode 划分训练、校准和测试集；
- 阈值不能使用最终闭环测试集调节。

### 7.2 四层验收

| 层级 | 评价对象 | 作用 | 能否决定上线 |
|---|---|---|---|
| 数据层 | 自动标签 | 排除错误、泄漏和无意义候选 | 不能 |
| 离线层 | Selector | 检查学习和校准 | 不能 |
| 闭环层 | Selector + 策略 | 验证控制收益 | 主要依据 |
| 实机层 | 完整同步系统 | 验证安全和泛化 | 最终依据 |

数据层重点检查：

- action/observation 对齐；
- 三个事件是否存在且顺序正确；
- 每个空间区域的样本数和独立 episode 数；
- 距离阈值附近的标签是否符合预期。

离线层除 accuracy 和 macro-F1 外，还要报告：

- action-step MAE；
- over-selection rate 和平均超出步数；
- under-selection rate；
- NLL、Brier score、ECE；
- 按空间区域和距离分桶的混淆矩阵。

闭环基线包括：

```text
每个固定候选
固定 full
固定最佳 small
粗标签 Selector
细标签 Selector
```

所有方法使用相同起点、demo 和随机种子。主要指标：

- 完整任务和子任务成功率；
- 碰撞、掉落、错误释放、插入失败；
- 控制步数和同步策略调用次数；
- 推理停止等待总时间；
- `control_execution_time + synchronous_inference_time`；
- 关键事件附近的 over-selection。

上线要求：

```text
成功率相对最佳固定 chunk 非劣
安全事件不增加
并且至少改善策略调用次数、停止等待时间或任务总时间之一
```

中间候选只有同时满足以下条件才保留：

```text
相比 full 更安全
相比 small 更高效
```

## 8. 第一版实施顺序

```text
1. 固定 chunk 闭环评估，得到粗标签和初始候选
2. 从演示位姿计算三个空间事件
3. 生成四个空间区域的共享特征和 chunk 标签
4. 检查事件顺序、区域分布和 episode 支撑
5. 训练 MLP 与 Transformer 基线
6. 完成配对闭环评估
7. 决定保留三级标签还是收敛为 {small, full}
```

最终目标：

> 自由空间尽可能使用 full chunk；需要新观测和精细反馈时及时停止并重规划；只保留
> 能带来可验证闭环收益的 execution chunk。

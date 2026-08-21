# Threading epoch20 ARP：执行 chunk 评估

## 评估设置

- Checkpoint：`epoch20_arp.ckpt`
- 权重：EMA
- 数据集：`data/threading/threading_d0.hdf5`
- 状态：EEF；相机：`top45`、`wrist`，分辨率 84×84
- 每个 episode 最多 1000 步；seed 从 10000 连续递增
- checkpoint 的 prediction horizon 为 20

## 当前执行机制

当前评估中，模型每次都预测完整 20 步 action plan，随后只执行前 `k` 步，再获取新观测并重新推理。`fixed chunk`、selector 和 spatial rule 改变的是执行长度，不改变 prediction horizon。

```text
预测 20 步 -> 执行前 k 步 -> 重新观测和预测
```

因此，小 chunk 不减少单次模型计算量，只会改变重规划频率。

## 初始 50-episode 对比

以下结果位于 `outputs/threading_epoch20_comparison_rs141/`，生成于完整预测改动之前；其中 fixed 配置始终预测 20 步，adaptive selector 和 spatial 配置则按所选 chunk 改变 prediction horizon：

| 配置 | 成功数 | 成功率 | 平均 policy calls |
|---|---:|---:|---:|
| fixed chunk=2 | 30/50 | 60% | 268.62 |
| spatial 2/4 early-short（0.15/0.25） | 27/50 | 54% | 295.54 |
| spatial 2/4 | 21/50 | 42% | 335.08 |
| spatial 2/8 | 20/50 | 40% | 339.40 |
| spatial 2/10 | 19/50 | 38% | 348.14 |
| fixed chunk=3 | 18/50 | 36% | 242.06 |
| fixed chunk=4 | 13/50 | 26% | 200.52 |
| selector v2（2/10） | 12/50 | 24% | 287.64 |
| fixed chunk=10 | 1/50 | 2% | 98.48 |
| selector v1（4/10） | 0/50 | 0% | 188.36 |

固定 chunk 从 2 增至 3、4、10 时，成功率依次由 60% 降至 36%、26%、2%。early-short spatial 2/4 比默认 spatial 2/4 高 12 个百分点，但与 fixed chunk=2 的逐 seed 差异不显著（exact McNemar `p=0.678`）。

## 当前代码下的 200-episode 配对评估

结果位于 `outputs/threading_epoch20_200ep_rs141/`。三组均使用 seed 10000–10199，其他设置相同。

| 配置 | 成功数 | 成功率（95% CI） | 总 policy calls | 平均 calls/episode |
|---|---:|---:|---:|---:|
| fixed chunk=2 | 116/200 | 58.0%（51.1%–64.6%） | 55,723 | 278.615 |
| spatial 2/6 | 48/200 | 24.0%（18.6%–30.4%） | 77,949 | 389.745 |
| spatial 2/10 | 25/200 | 12.5%（8.6%–17.8%） | 85,140 | 425.700 |

spatial 2/10 比 fixed chunk=2：

- 成功率低 45.5 个百分点；
- 多调用模型 29,417 次，即增加 52.8%。

逐 seed 配对结果：

| 配对结果 | seed 数 |
|---|---:|
| 两者都成功 | 14 |
| 仅 fixed chunk=2 成功 | 102 |
| 仅 spatial 2/10 成功 | 11 |
| 两者都失败 | 73 |

Exact McNemar 检验为 `p=1.25e-19`。在这次 200-episode 评估中，成功率差异不能用普通采样波动解释。

spatial 2/6 与 fixed chunk=2 的配对结果为：两者都成功 26，只有 fixed 成功 90，只有 spatial 2/6 成功 22，两者都失败 62；exact McNemar `p=5.97e-11`。spatial 2/6 比 fixed 少成功 68 个 episode，并多调用模型 22,226 次（+39.9%）。

spatial 2/6 与 spatial 2/10 的配对结果为：两者都成功 9，只有 2/6 成功 39，只有 2/10 成功 16，两者都失败 136；exact McNemar `p=0.00267`。2/6 比 2/10 多成功 23 个 episode，并少调用模型 7,191 次（-8.4%）。

spatial 2/10 的实际选择次数为：

| 执行 chunk | 次数 | 占比 |
|---|---:|---:|
| 2 | 83,710 | 98.32% |
| 10 | 1,430 | 1.68% |

尽管 chunk=10 只占 1.68%，spatial 2/10 的失败和超时更多，最终总推理次数也高于 fixed chunk=2。

spatial 2/6 的实际选择次数为：

| 执行 chunk | 次数 | 占比 |
|---|---:|---:|
| 2 | 75,852 | 97.31% |
| 6 | 2,097 | 2.69% |

spatial 2/6 的四个连续 50-seed 分块成功数分别为 14、11、12、11。

## 版本差异

旧的 50-episode adaptive selector 和 spatial 结果生成于完整预测改动之前。当时所选执行 chunk 同时改变 prediction horizon；2026-08-03 的代码改为始终预测完整 20 步、只截取执行前缀。因此旧 spatial 2/10 的 19/50 与当前 200-episode 结果不是同一推理逻辑，不能直接用于判断当前实现的随机波动。

复现检查：fixed chunk=2 在新评估前 50 个 seed 上仍为 30/50，成功标签与旧结果完全一致；spatial 2/10 在新评估前 50 个 seed 上为 8/50，旧结果为 19/50。

## 当前事实结论

在当前“固定预测 20 步、截断执行”的代码下，三组结果按成功率排序为 fixed chunk=2（58.0%）、spatial 2/6（24.0%）、spatial 2/10（12.5%）；总推理次数排序相反，分别为 55,723、77,949、85,140。fixed chunk=2 的成功率和总推理次数均最优。

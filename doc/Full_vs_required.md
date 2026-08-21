# Full-then-truncate vs Required-only

本文定义两种 adaptive action chunk 解码方式，并分别记录当前能在输出目录中找到的
PushBox、Threading、ALOHA 结果。

## 定义

设 policy 训练时的完整预测长度为 `H`，selector 或 fixed baseline 在一次 replan 时
选择执行 chunk `k`，其中 `k <= H`。

### full_then_truncate

`full_then_truncate` 指每次都生成完整长度 `H` 的 action plan：

```text
generate: a_1, a_2, ..., a_H
execute : a_1, ..., a_k
discard : a_{k+1}, ..., a_H
```

因此 `k` 只改变执行前缀长度和重新规划频率，不改变单次生成长度。它更接近训练时的
完整 horizon，但短 chunk 不会降低单次自回归生成成本。

### required_only

`required_only` 指当前选择 `k`，就只生成本次需要执行的 action：

```text
generate: a_1, ..., a_k
execute : a_1, ..., a_k
```

因此 `k` 同时改变生成长度、执行长度和重新规划频率。短 chunk 可以减少生成 token，
但也可能因为预测 horizon 变短而引入和训练 full horizon 不同的分布。

一句话：

```text
full_then_truncate = always generate H, execute first k
required_only      = generate k, execute k
```

## PushBox

PushBox 里有最明确的 full/required 对比实验。

主结果路径：

```text
/home/huiyuan/pushbox/outputs/pushbox_prediction_mode_matrix
```

该目录包含 `matrix_summary.csv/json`，覆盖：

- prediction mode：`full_then_truncate`、`required_only`
- strategy：fixed、selector、spatial
- fixed chunk：`2, 4, 5, 6, 8, 10, 19`
- adaptive chunk：selector/spatial 的 `5` 或 `10` 配置
- 每组 50 episodes

### Fixed chunk 对比

| mode | chunk | success | wall time | inference time | calls | generated tokens | tokens/call |
|---|---:|---:|---:|---:|---:|---:|---:|
| full_then_truncate | 2 | 30/50, 60% | 1.634 | 0.848 | 63.64 | 1272.80 | 20.00 |
| required_only | 2 | 0/50, 0% | 2.123 | 0.876 | 100.26 | 300.78 | 3.00 |
| full_then_truncate | 4 | 33/50, 66% | 1.265 | 0.469 | 34.64 | 692.80 | 20.00 |
| required_only | 4 | 10/50, 20% | 1.844 | 0.523 | 55.50 | 277.50 | 5.00 |
| full_then_truncate | 5 | 32/50, 64% | 1.059 | 0.321 | 27.28 | 545.60 | 20.00 |
| required_only | 5 | 28/50, 56% | 1.502 | 0.376 | 38.86 | 233.16 | 6.00 |
| full_then_truncate | 6 | 33/50, 66% | 1.099 | 0.319 | 23.20 | 464.00 | 20.00 |
| required_only | 6 | 31/50, 62% | 1.237 | 0.285 | 28.32 | 198.24 | 7.00 |
| full_then_truncate | 8 | 29/50, 58% | 0.904 | 0.218 | 15.84 | 316.80 | 20.00 |
| required_only | 8 | 39/50, 78% | 1.097 | 0.212 | 19.56 | 176.04 | 9.00 |
| full_then_truncate | 10 | 32/50, 64% | 0.925 | 0.189 | 13.66 | 273.20 | 20.00 |
| required_only | 10 | 38/50, 76% | 1.029 | 0.172 | 15.30 | 168.30 | 11.00 |
| full_then_truncate | 19 | 30/50, 60% | 0.791 | 0.101 | 7.24 | 144.80 | 20.00 |
| required_only | 19 | 30/50, 60% | 0.827 | 0.105 | 7.24 | 144.80 | 20.00 |

观察：

- required-only 明显降低 generated tokens/call：例如 chunk 10 从 20 降到 11。
- required-only 对很短 chunk 伤害很大：chunk 2 从 60% 降到 0%，chunk 4 从 66% 降到
  20%。
- required-only 对中长 chunk 反而更好：chunk 8 从 58% 到 78%，chunk 10 从 64% 到
  76%。
- full chunk label 19 在两种模式下等价，tokens/call 都是 20，成功率也同为 60%。

### Adaptive selector / spatial 对比

| mode | strategy | config | success | wall time | inference time | calls | generated tokens | tokens/call |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| full_then_truncate | selector | 5 | 30/50, 60% | 0.996 | 0.250 | 17.08 | 341.60 | 20.00 |
| required_only | selector | 5 | 32/50, 64% | 0.959 | 0.190 | 16.26 | 156.92 | 9.65 |
| full_then_truncate | selector | 10 | 30/50, 60% | 0.858 | 0.155 | 10.64 | 212.80 | 20.00 |
| required_only | selector | 10 | 31/50, 62% | 0.859 | 0.138 | 10.54 | 149.42 | 14.18 |
| full_then_truncate | spatial | 4 | 31/50, 62% | 1.111 | 0.323 | 23.24 | 464.80 | 20.00 |
| required_only | spatial | 4 | 29/50, 58% | 1.118 | 0.254 | 24.72 | 182.10 | 7.37 |
| full_then_truncate | spatial | 10 | 30/50, 60% | 0.882 | 0.153 | 10.92 | 218.40 | 20.00 |
| required_only | spatial | 10 | 34/50, 68% | 0.921 | 0.146 | 11.86 | 159.26 | 13.43 |

观察：

- adaptive required-only 的 token 数减少很明显；
- selector 5/10 的成功率略高于 full_then_truncate；
- spatial 10 的 required-only 成功率从 60% 到 68%；
- spatial 4 的 required-only 成功率略低，但 token 数大幅下降。

### PushBox 200-episode required-only 正式结果

另一个正式 required-only 结果路径：

```text
/home/huiyuan/pushbox/outputs/pushbox_triangular_x_000_005_y_m025_m020_n200_seed10000/required_only
```

| method | success | steps | policy calls | generated tokens | inference time |
|---|---:|---:|---:|---:|---:|
| fixed 5 | 153/200, 76.5% | 178.20 | 36.04 | 216.21 | 0.314 |
| fixed 10 | 134/200, 67.0% | 133.82 | 13.86 | 152.41 | 0.148 |
| fixed 15 | 107/200, 53.5% | 117.60 | 8.34 | 133.44 | 0.098 |
| fixed 19 | 86/200, 43.0% | 103.35 | 5.86 | 117.20 | 0.077 |
| spatial 19/5/19 | 162/200, 81.0% | 161.88 | 26.05 | 196.06 | 0.252 |

该结果只包含 required-only，不是 full/required A/B；但它说明 spatial rule 在
required-only 下可以超过 fixed baselines。

## Threading



## ALOHA

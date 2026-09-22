# AutoHorizon 的实机 Threading 适配

严格移植 `/home/huiyuan/pushbox/scripts/autohorizon`。
`official.py`、`core.py` 原样复制参考实现，保留上游 `c7504f1` 的三个
soft-pointer 函数、校验、诊断量、版本标识和 `LICENSE.upstream`。

默认 `bidirectional`，`hold_thr=0.3`、`max_entropy_q=0.9`、`run_len=1`。
所有层和头的动作 self-attention 先平均，再交给官方函数归一化。
保留累计最大值、零基准首增量、熵分位数 `min(q, 0.999)`、无平台时选择末行、
反向指针映射，以及长度条件和 `join_row` 同时满足才执行完整范围的规则。
不修改上游反向坐标，不添加平滑、启发式长度边界或离散候选长度。
`forward` 消融模式只支持 `run_len=1`。

## Python 接口

在 `threading_real` 目录运行，或安装本项目后导入：

```python
from autohorizon import AutoHorizonConfig, select_horizon, predict_action_with_autohorizon

# 独立张量算法，仅依赖 PyTorch；不接受 batch 轴。
# attention: [H,H]、[heads,H,H] 或 [layers,heads,H,H]
decision = select_horizon(attention, AutoHorizonConfig())

# policy 为已加载的 threading_task.policy.ThreadingARPolicy。
policy.eval()
result = predict_action_with_autohorizon(policy, observation)
actions_to_execute = result.actions  # [h_star, action_dim]，保留原动作单位和坐标系
trace = result.metrics()             # 可直接 JSON 序列化
```

适配器只调用一次 `predict_action`，临时设置 `full_then_truncate` 和完整执行上限，
关闭附加的 learned selector；从同次生成的 softmax 后、dropout 前提取 self-attention，
排除观测、plan token 和视觉 cross-attention。采样和 `low_var_eval` 保持原配置，
不重训练、不二次生成。成功或异常后均移除 hooks 并恢复策略配置。

与参考适配器的必要差异：本地 `ThreadingARPolicy` 没有 `max_selector_chunk_label`，
因此使用 `horizon` 取得完整输出；默认预测偏移为 0，执行
`action_pred[0, :raw_horizon]`，不扣除 PushBox 的历史位置，也不套用仿真中的 9 步上限。
`prediction_offset=` 保留为显式适配入口，并验证输出确实对应同一预测切片。

## 向量 ARP 支持范围

与参考实现相同，要求 `eval()`、batch size 1、`H >= 2`、
`action_chunk_size >= horizon`，同一策略实例不能并发调用。
双向模式要求 `run_len < H`。不支持多组动作生成。

向量策略入口支持每个动作对应一个 token 的 `ThreadingARPolicy`；
MVT/PlanARP 使用下面单独的动作步聚合适配。pi0.5 不使用这里的 ARP hooks。
ARP 没有 flow 去噪步骤，采集完整动作生成的 attention，沿用参考目录的适配方式。

已提供 Python 推理入口及 MVT 实机部署 CLI 接入，未运行机器人或宣称实机收益。
参考目录的 `summarize_threading.py` 依赖仿真 runner 的日志格式，因此未移植。

## 单相机点云 PlanARP

已适配当前单相机 checkpoint：
`outputs/threading_combined_80_mvt_cam1_planarp_v2/20260912_171342/checkpoints/epoch=0004-val_loss=10.049.ckpt`。
其配置是 `pointcloud_views=[sideview]`、`H=10`、`plan_steps=4`、
`action_chunk_size=2`、`predict_gripper=false`。单物理相机仍产生两个 MVT 虚拟视图，
每个动作含 3 个控制点 × 2 个视图 = 6 个 token。

```python
from autohorizon import predict_mvt_with_autohorizon

policy.eval()
prediction, result = predict_mvt_with_autohorizon(policy, observation)
full_actions = prediction["action_pred"]  # [1,H,7]，供部署检查完整预测
selected_actions = prediction["action"]  # [1,h_star,7]
trace = prediction["prediction_diagnostics"]
# 通用 predict_action_with_autohorizon 也会自动分派到 MVT，返回 result。
```

只进行一次视觉编码和一次 `predict_action`，完整生成原策略的所有 plan 和动作组。
默认 `sample=False` 与现有 MVT 部署一致；专用接口可显式传 `sample=True`。
不修改 `action_chunk_size`、权重、采样器、相机选择、控制点解码或动作值。

必要的模型适配（这些操作不属于上游 soft-pointer 函数）：

1. token 序列开头为 6 个 current-point 和 `6 * plan_steps` 个 plan token，
   均不作为动作 attention 的行或列。
2. 当前 checkpoint 分 5 组生成动作。逐层采集每组生成时新增的动作 query 行，
   保留其对当前及过去动作 key 的 attention，未来 key 补因果零。
   后续前向重新编码的历史动作行不会覆盖生成时的记录。
   这扩展了参考适配器仅支持单组生成的限制，同时保持原模型的分组生成语义。
3. 对每个动作步，query token 取平均，key token 求和：
   `A_step[i,j] = sum_q sum_k A_token[i*K+q,j*K+k] / K`。
   保留动作 key 的 attention 质量，不在此处归一化；若模型预测夹爪，则 `K=7`。
4. 对层、头取平均得到 `H×H`，交给未修改的官方算法。返回的长度就是动作步数，
   偏移为 0，可取 `1..H`，不要求是生成组大小的整数倍。

trace 记录聚合方法、采集方式、相机、生成 token 数、生成步数、原始/执行长度和
前后向指针。MVT 版本标识为 `official_c7504f1_mvt_step_attention_v1`，
与参考的一步一 token 适配明确区分。该聚合和分组采集是模型适配，不能视为
上游 π0.5 实验的直接复现。

### 实机部署入口

沿用现有单相机 PlanARP 部署命令，添加：

```text
--autohorizon --prediction-mode full_then_truncate
--autohorizon-hold-thr 0.3 --autohorizon-entropy-q 0.9
--autohorizon-run-len 1 --autohorizon-method bidirectional
```

例如在 `threading_real` 目录先运行现有只推理模式（仍需要相机和机器人状态连接）：

```bash
python scripts/deployment/cartesian.py \
  'outputs/threading_combined_80_mvt_cam1_planarp_v2/20260912_171342/checkpoints/epoch=0004-val_loss=10.049.ckpt' \
  --weights model --device cuda:0 \
  --pointcloud-calibration calibration/block_grasp_spatial.json \
  --policy-hz 7.5 --autohorizon --prediction-mode full_then_truncate \
  --synchronous --sync-timeout 5 --max-cycles 1 \
  --trace-output artifacts/autohorizon_cam1_trace.jsonl
```

AutoHorizon 与 `--aac`、`--chunk-selector`、`--execution-schedule` 互斥，
不支持 `required_only`。`--execute-steps` 不再截断所选长度；同步超时按完整
`H / policy_hz` 校验。保留原部署对完整预测的异常幅度检查、积分限幅和执行流程，
通过检查后执行选定前缀。实际执行仍使用现有 `--execute --confirm-real-robot` 机制。
CSV condition 为 `autohorizon`，JSONL trace 包含算法诊断和实际执行步数。

## 验证

```bash
cd /home/huiyuan/teleoperation/threading_real
/home/huiyuan/miniconda3/envs/arp/bin/python -m pytest autohorizon -q
```

默认从参考目录独立提取函数，对照 CPU/GPU、随机/均匀/对角/尖锐 attention、
多个 horizon 和参数的输出及全部诊断量，并验证函数 AST 一致。
可以显式对照官方克隆：

```bash
AUTOHORIZON_REFERENCE=/home/huiyuan/pushbox/AutoHorizon/src/openpi/models_pytorch/pi0_pytorch.py \
/home/huiyuan/miniconda3/envs/arp/bin/python -m pytest autohorizon -q
```

参考文件不可用时对照项会明确跳过。其余测试覆盖真实 ARP Attention hooks、
本地 ThreadingARPolicy 的有/无 plan 完整前向、默认零偏移、动作前缀一致性、
单次生成、非法输入和异常清理；不依赖机器人、相机或预训练权重下载。

### 本次单相机适配验证

- `autohorizon`、Cartesian 部署、MVT 和预测模式回归共 81 项测试通过。
- 使用上面的实际 epoch 4 checkpoint，在已有单相机数据的一个留出 episode
  上选取前、中、后三个观测，完整动作预测与原策略逐元素一致，所选前缀也逐元素一致。
- 三次选择长度均为 1；尚未运行实机，不能据此宣称性能提升。
- 离线记录：`artifacts/autohorizon_cam1_smoke/report.json`。

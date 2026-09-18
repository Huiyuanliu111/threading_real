# Adaptive Action Chunking (AAC)

实现来源：[论文](https://arxiv.org/html/2604.04161v1) §3.2、§4、附录 A。
已对照[作者实现](https://github.com/Adaptive-Action-Chunking/libero/blob/main/action_optimization/action_entropy_v2.py)
及本机 `/home/huiyuan/arp/aloha/adaptive/aac.py`。只依赖 PyTorch，无训练、额外编码器或 selector checkpoint。

## 论文规则

1. 同一观测随机采样 `N` 条完整动作序列 `[N,H,7]`，默认 `N=20`。
2. 平移、旋转各计算三维高斯微分熵（完整样本协方差），夹爪计算二值离散熵，三项相加。
3. `E_bar = cumsum(E) / arange(1,H+1)`；`h_entropy = argmax(diff(E_bar)) + 1`。
4. 对实际要执行的轨迹计算前缀运动量：累计平移的欧氏范数、依次复合旋转后的轴角范数、前缀内是否发生夹爪切换（0 或 1）。三项直接相加。
5. `xi` 是第一个满足 `m(l) > alpha` 的长度，默认 `alpha=3`；`h_star=max(h_entropy,xi)`。
6. 执行这条轨迹的前 `h_star` 步，再用新观测重新采样。

默认遵循论文公式，不沿用参考代码的 `gripper_weight=0.2` 或最短长度 2，且不把结果限制为离散候选长度。
论文未明确指定执行哪条候选及无阈值交点的处理；这里默认使用候选 0，可用 `execution_candidate_index` 指定其他候选，无交点返回 `H`，与作者运动量计算及 fallback 一致。
协方差使用无偏估计及 `1e-6 I` 正则（数值实现细节）。`H=1` 时返回 1，熵差分并列时取最早位置。
旋转采用作者的 `delta_q * q_total` 顺序；式 (9) 按复合后轴角的范数解释，单位四元数本身的范数恒为 1，不能作为运动量。
夹爪切换只比较预测前缀中的相邻状态，不额外比较当前观测夹爪状态。

## 调用

从 `threading_real` 目录运行，或安装本项目后导入：

```python
from AAC import AACConfig, select_chunk_size

# candidates: torch.Tensor [N,H,7]
# [dx,dy,dz,rx,ry,rz,binary_gripper]，旋转是弧度轴角增量。
result = select_chunk_size(candidates, config=AACConfig())
actions_to_execute = candidates[0, :result.h_star]  # 默认 execution_candidate_index=0
diagnostics = result.metrics()  # 可直接写入 JSON
```

`nominal=` 可显式传入 `[H,7]` 执行轨迹，调用方必须执行同一轨迹。
`entropy_candidates=` 可传归一化动作（与候选同形），仅前六维参与连续熵；
夹爪熵和运动量始终使用物理动作。核心函数允许任意 `N>=2`；配置的 `num_samples` 供采样入口使用。

提供 `AACInference.predict(sample_chunks, to_aac=..., to_entropy=...)` 作为模型无关推理入口：
`sample_chunks(N)` 必须用同一观测、独立噪声，一次批量生成完整 `[N,H,7]` 可执行动作。
返回对象的 `actions` 是指定候选的执行前缀，`decision` 包含算法诊断量。该入口拒绝完全相同的候选。
可以在 sampler 闭包中复用原策略已计算的视觉条件；AAC 不调用视觉编码器。
不能用确定性预测复制 N 份估计不确定性，也不能先预测短 chunk 再用 AAC 选择。

### 对齐 ALOHA v7

本地版本标识为 `paper_candidate0_v7_threading_execution`，对齐
ALOHA 的 `paper_candidate0_v7_aloha_execution`：

- `AACConfig(execution_candidate_index=2)` 可指定候选；默认仍为 0。
  同一候选同时用于运动量约束和实际执行，不再生成额外 nominal 或执行预测。
- `xi` 从长度 1 开始搜索，保留运动折返前的首次阈值交点。
- `prediction.metrics()` 记录候选索引、版本、`generated_action_tokens=N*H`、
  候选方差均值／最大值以及包含采样与选择的 `predict_action_elapsed_sec`。
  提供 `to_entropy` 时方差在该变换后的空间计算，否则在 AAC 动作空间计算。
- 非法候选索引在采样前报错；核心函数另外验证实际候选数量。

本次同步的是 ALOHA 最新变更；保留本项目 Cartesian 增量语义及论文夹爪权重 1。
ALOHA 的绝对关节目标、`low_var_eval` 开关和 policy queue 属于其模型接入层，
这里通过 sampler 回调接入模型，并直接返回可执行前缀。

## Threading 动作适配

本项目 Cartesian 部署使用 `[dxyz,drotvec,dwidth]`，不是论文的二值夹爪格式：

```python
from AAC import AACInference, threading_delta_to_aac

prediction = AACInference().predict(
    sample_chunks,  # 闭包：同一观测批量产生 N 条完整物理动作序列
    to_aac=lambda actions: threading_delta_to_aac(
        actions,
        current_width=current_width_m,
        closed_width_threshold=closed_width_threshold_m,
    ),
)
actions_to_execute = prediction.actions  # 保留原始 dwidth，不执行二值标签
```

宽度阈值须按实际夹爪配置；先累计 `dwidth` 得到预测宽度再二值化。
如部署会覆盖夹爪动作（当前 pi0.5 路径将 `dwidth` 置零），sampler 必须先应用同样的覆盖。
`alpha=3` 是论文默认值，不是跨单位通用的物理阈值：本项目平移使用米、旋转使用弧度，
可能长期无法达到阈值而返回完整 horizon；应结合 `magnitude_threshold_reached` 和 `action_magnitude`
在离线数据上明确校准。实现不会暗中缩放位移或改阈值。

实机 Cartesian CLI 已提供 pi0.5 和 MVT/PlanARP 接入：`--aac --aac-alpha 3 --aac-num-samples 20`，
使用 `--prediction-mode full_then_truncate`，不传 `--chunk-selector`。
`--aac-execution-candidate-index` 默认 0。每次同一观测批量采样 N 个完整 horizon，
连续熵使用归一化输出，运动量使用反归一化动作，保持原部署的夹爪增量置零行为。
执行长度由 AAC 决定，不受固定 `--execute-steps` 截断。
支持任意正 horizon。pi0.5 采样对观测扩展批量，未实现视觉 KV cache 跨候选共享。
需要已有可用的 checkpoint、兼容 LeRobot 的 Python 环境及机器人/相机依赖。
裁剪 checkpoint 通过 `pi05/deployment/cropped.py` 入口加载保存的 ROI 元数据，
未裁剪 checkpoint 使用 `scripts/deployment/cartesian.py`。
关节绝对目标不能直接传入该 Cartesian 核心；需要先用运动学转换为 TCP 增量。

Maze/Threading PlanARP 通过 `AAC.arp.predict_arp_aac` 接入；先进行一次视觉编码，
再将特征扩展到 N 条候选，启用 ARP 空间 token 随机采样及非低方差 GMM 采样。
非 AAC 模式仍保持默认确定性预测。Maze XY 补零为 XYZ，旋转和夹爪固定为零；
这些恒定熵项不改变前缀差分。PlanARP 连续熵使用物理动作增量，
不把空间像素 token 当作归一化 Cartesian 动作。
完整评估命令和结果登记见仓库根目录 `doc/deploy_selector.md`。

Threading Cartesian CLI 的 ARP AAC 默认使用 `--aac-sample-batch-size 1`，
复用一次视觉编码，分批生成 `--aac-num-samples` 条候选后统一计算熵。
这降低采样峰值显存，但可能增加推理耗时；有显存余量时可增大批大小。
trace 中保存 `sample_batch_size`。不同批大小会改变随机数消耗顺序，
因此即使随机种子相同，也不保证得到相同的候选轨迹。

## 验证

```bash
python -m pytest tests/test_aac.py -q
```

覆盖独立 NumPy 协方差计算、SciPy 非交换旋转复合、下标、严格阈值、单位夹爪权重、
宽度增量适配、无交点、单步 horizon、归一化隔离、采样次数及原动作前缀保持。

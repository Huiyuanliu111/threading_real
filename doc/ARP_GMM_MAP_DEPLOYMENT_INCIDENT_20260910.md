# ARP 部署中 MAP 模式未生效的问题

日期：2026-09-10

## 结论

> 2026-09-10 后续复核：`threading_new_1` checkpoint 从训练开始就是
> `sideview + frontview` 两路固定相机输入，不包含 wrist。此前把旧的三相机 checkpoint
> 运行记录用于解释 `threading_new_1` 的成功/失败变化，是错误归因，现已撤回。

Threading ARP 的两个实机部署入口都存在同一个配置错误：命令行参数
`--gmm-eval-mode map` 虽然默认选择 MAP 模式，但部署代码只给 GMM predictor
安装了 MAP 实现，没有把策略实际传给 ARP generator 的 `sample` 参数改成
`"map"`。

当前 ARP checkpoint 的配置为 `arp_cfg.sample: true`。因此修复前即使部署命令使用
默认的 `--gmm-eval-mode map`，推理仍然执行随机 GMM sampling。同一相机图像和机器人
状态可能产生不同动作；少量概率选中另一个 mixture component 时，会表现为偶发的
方向或幅度突变。

MAP 错误是真实存在的部署语义错误，但 Git 和后续实机结果不支持把它作为“同一
`threading_new_1` checkpoint 以前成功、现在失败”的根因。以前的成功运行也经过同一段
未生效的 MAP 代码，修复后确定性 MAP 仍然输出了错误方向。因此它只能解释修复前的随机
波动，不能解释当前可重复的失败。

这不是模型训练时使用 GMM 的问题，也不是 MAP 算法本身的问题。问题是部署命令声明
使用 MAP，但代码没有把该选择传到生成过程。

## 受影响的代码

受影响的两个入口是：

- `scripts/deploy_threading_real_cartesian.py`：当前 7D Cartesian delta ARP 实机部署入口；
- `scripts/deploy_threading_real.py`：较早的 8D joint target/delta 部署入口。

当前 Cartesian ARP checkpoint 为：

```text
outputs/threading_new_1_arp/20260907_134130/checkpoints/latest.ckpt
```

该 checkpoint 的实际输入为：

```yaml
camera_output_keys:
  - sideview
  - frontview
shape_meta:
  obs:
    sideview: {type: rgb}
    frontview: {type: rgb}
    agent_pos: {type: low_dim, shape: [8]}
```

其中 `agent_pos` 是 7 维关节角加 gripper width。没有 wrist 图像输入。

其关键配置为：

```yaml
policy:
  action_mode: cartesian_delta
  horizon: 20
  n_action_steps: 20
  n_obs_steps: 2
  arp_cfg:
    action_predictor: gmm
    sample: true
    low_var_eval: true
```

`ThreadingARPolicy` 初始化时执行：

```python
self.use_sample = arp_cfg.get("sample", True)
```

所以该 checkpoint 加载后，`policy.use_sample` 的值是布尔值 `True`。

## 预期调用链

命令行默认值为：

```python
parser.add_argument(
    "--gmm-eval-mode",
    choices=("map", "mean", "sample"),
    default="map",
)
```

三种模式应当对应以下传值：

| 命令行模式 | `policy.use_sample` | 预期行为 |
|---|---:|---|
| `map` | `"map"` | 选择概率最大的 mixture component，并输出其均值 |
| `mean` | `False` | 使用 predictor 的非采样输出 |
| `sample` | `True` | 从 GMM 分布随机采样 |

策略最终在 `ThreadingARPolicy.predict_action()` 中调用：

```python
generated = self.policy.generate(
    sequence,
    future_spec,
    contexts={"visual-token": visual_tokens},
    sample=self.use_sample,
)
```

因此决定实际推理模式的是 `policy.use_sample`，而不是部署参数本身。

## 修复前的错误

修复前的部署逻辑是：

```python
if args.gmm_eval_mode == "map":
    enable_map_gmm_inference(policy)
else:
    policy.use_sample = args.gmm_eval_mode == "sample"
```

`enable_map_gmm_inference()` 会替换每个 `GMMPredictor.sample()` 方法，但替换后的方法
只有收到字符串 `do_sample == "map"` 时才走 MAP 分支：

```python
if do_sample != "map":
    return original_sample(distributions, do_sample, **extra_contexts)
```

由于部署代码没有设置 `policy.use_sample = "map"`，generator 实际传入的是 checkpoint
中的布尔值 `True`。`True != "map"`，所以调用被转发给原来的随机采样实现。

换句话说，安装 MAP 方法只提供了处理 `"map"` 的能力；它不会主动把策略切换到 MAP
模式。

## 为什么会表现为“偶发的奇怪动作”

ARP 的 action predictor 是 GMM。随机模式会先按照 mixture probabilities 选择一个
component，再从该 component 的分布采样。多数推理可能落在主 component 附近，因此
动作看起来稳定；当某次选择不同 component 或采到尾部值时，动作可能突然改变。

这类现象具有以下特征：

- 相机、机器人位置和 gripper 状态没有明显变化；
- 大部分动作相似，少数动作突然偏离；
- 重新运行后异常不一定能复现；
- 同一输入连续推理也不能得到相同输出。

这些特征与本次观察到的情况一致，但不能据此断言此前每一个失败动作都只由随机采样
造成。视觉分布偏移、机器人状态、训练覆盖范围和 checkpoint 本身仍可能产生错误的
确定性动作。

## Git 复核：为什么 MAP 修复不能解释成功变失败

`threading_new_1` checkpoint 在 2026-09-07 13:56 写完，SHA-256 为：

```text
3d8f0d653e07f27b70ba746a9435484d73bacde68e7c321ed19ebcf1fdce98e6
```

文件修改时间没有变化，当前 `latest.ckpt` 与 `epoch=0055-val_loss=-6.067.ckpt` 完全相同，
没有证据表明模型权重后来被覆盖。

双相机 checkpoint 部署支持进入 Git 的提交是 `89e52e2`（2026-09-09 18:59）。该提交前
的已提交部署脚本强制要求 `sideview + wrist + frontview`，会直接拒绝
`threading_new_1`，所以同一 checkpoint 的成功实机运行只能发生在 `89e52e2` 对应代码或
其提交前的等价未提交工作区版本上。

从 `89e52e2` 到当前 `cd5676f`，以下 ARP 相关文件的 Git 差异只有 MAP 分支增加的一行：

```python
policy.use_sample = "map"
```

`threading_task/policy.py`、数据处理、图像 encoder、checkpoint 配置以及双相机映射在这段
时间内均没有改变。因此 Git 中不存在另一个能把此前成功行为变成当前失败行为的 ARP
代码改动。若此前与现在确实使用同一个 checkpoint，差异来自运行命令、机器人/夹爪状态、
相机与场景，或修复前 GMM sampling 的具体随机结果，而不是 wrist 输入被增删。

## 当前更直接的异常：关节状态越过训练范围

恢复位置后的只读诊断记录到首次推理关节状态：

```text
q = [0.21093, 0.72673, 0.10387, -2.22385, 0.16512, 2.92331, 2.52262]
```

checkpoint 内保存的训练范围显示：

| 状态维度 | 当前值 | 训练最小值 | 训练最大值 | 结论 |
|---|---:|---:|---:|---|
| q1 | 0.21093 | 0.31615 | 0.81443 | 低于训练范围 |
| q3 | 0.10387 | -0.93229 | -0.02833 | 高于训练范围 |

按 checkpoint 的 min-max normalizer，q1 和 q3 分别被映射到 `-1.422` 和 `1.293`，超过
训练归一化区间 `[-1, 1]`。关节状态在训练中不是只用于执行，它直接进入策略的
`agent_pos` 条件。因此恢复相似的 TCP/相机构图不等于恢复相同的模型输入；Franka 的冗余
关节构型可以在末端位姿接近时仍让 q1、q3 明显不同。

一次实机测试后的闭合夹爪状态仍有同样问题：q1 为 `0.26466`、q3 为 `0.06511`，两者继续
越过训练范围。这比亮度差异更直接地说明当前 observation 落在训练分布之外。

当前证据支持的解释是：模型在视觉特征和关节状态都偏离训练分布时，确定性 MAP 输出了
一个接近数据集总体动作偏置的计划。MAP 修复消除了随机性，但不会把分布外输入变回训练
条件，也不会保证主 mixture component 的动作方向正确。

## 动作级诊断结果

诊断使用当前恢复后的机器人位置、同一组实时相机图像和同一组机器人状态。测试只读取
相机与状态，没有向机器人发送 arm 或 gripper 命令。

使用 checkpoint 原始的随机 GMM 模式，对同一 observation 重复推理 32 次：

| 指标 | 结果 |
|---|---:|
| MAP 首步平移幅度 | 4.405 mm |
| 随机首步相对 MAP 的平均平移偏差 | 0.138 mm |
| 随机首步相对 MAP 的最大平移偏差 | 4.411 mm |
| 随机首步相对 MAP 的最大旋转偏差 | 0.777° |
| 随机首步平移幅度范围 | 4.405–5.084 mm |
| 20 步内任一动作分量的最大绝对值 | 0.0502 |

最大首步偏差和正常首步平移幅度相当，说明一次低概率采样足以明显改变动作方向。平均值
较小并不能排除偶发异常，因为实机风险由单次尾部样本决定。

修复后使用 MAP 模式，在完全相同的 observation 上重复推理，最大动作绝对差为：

```text
0.0
```

这证明修复后的 ARP 推理在该测试条件下是确定性的。

完整动作诊断保存在：

```text
artifacts/live_arp_action_shift_20260910_restored_pose/report.json
```

复现命令：

```bash
cd /home/huiyuan/teleoperation/threading_real
conda run --no-capture-output -n pushbox \
  env LD_LIBRARY_PATH=/home/huiyuan/miniconda3/envs/pushbox/lib:/home/huiyuan/miniconda3/envs/pushbox/lib/python3.10/site-packages/torch/lib \
  python scripts/diagnose_live_arp_action_shift.py \
  --checkpoint outputs/threading_new_1_arp/20260907_134130/checkpoints/latest.ckpt \
  --dataset /home/huiyuan/teleoperation/data/threading_combined_pi05_15hz_sg5_nozero \
  --output artifacts/live_arp_action_shift_20260910_restored_pose/report.json
```

这里的 `threading_combined_pi05_15hz_sg5_nozero` 是本次诊断使用的参考数据集，并非
checkpoint 配置中记录的 `threading_new_1_smolvla_6hz_sg5_nozero` 训练数据集。因此
下面的 held-out 数值只表示诊断参考集内部的基线，不能解释为 checkpoint 验证集指标。

该脚本会连接 RealSense 和只读机器人状态 UDP/RPC。它不会创建 TrackC streamer，也不会
调用 `movej`、`grasp`、`gripper_release` 或其他执行接口。

## 与相机视觉差异的关系

发现 GMM 模式问题之前，诊断目标是判断实时相机的亮度和色偏是否导致了奇怪动作。恢复
相机位置后，ARP encoder 仍检测到一定的视觉分布差异：

| 视角 | projected feature 最近邻距离 / 诊断参考集 held-out 基线 |
|---|---:|
| sideview | 4.53× |
| frontview | 11.31× |

frontview 的差异较大，而且不仅包含亮度和绿色偏色，也包含物体、针具、末端执行器和背景
构图差异。这个结果说明当前视觉条件仍低于理想的训练分布匹配程度。

为了把颜色影响与 GMM 随机性分开，动作反事实测试先启用确定性 MAP，再保持机器人状态和
图像内容不变，只把实时图像的每通道均值与标准差匹配到诊断参考数据：

| 修改条件 | 首步平移变化 | 首步旋转变化 | 平移方向余弦 |
|---|---:|---:|---:|
| 只校正 sideview | 0.145 mm | 0.0109° | 0.99957 |
| 只校正 frontview | 0.0758 mm | 0.0059° | 0.99989 |
| 两路同时校正 | 0.221 mm | 0.0168° | 0.99903 |

两路颜色同时校正造成的首步平移变化约为当前 4.405 mm 首步动作的 5%，动作方向基本不变。
因此在当前恢复位置，亮度和色偏会减少视觉裕量，但它们不像是本次偶发动作突变的主要
原因。该结论只适用于当前 observation；在遮挡、接近针具或训练覆盖较弱的阶段，视觉
分布偏移仍可能造成更大的动作变化。

视觉特征报告保存在：

```text
artifacts/live_arp_visual_comparison_20260910_restored_pose/report.json
```

## 修复

两个部署入口的 MAP 分支都增加了同一行：

```python
enable_map_gmm_inference(policy)
policy.use_sample = "map"
```

修改位置：

- `scripts/deploy_threading_real_cartesian.py` 的 GMM mode 配置分支；
- `scripts/deploy_threading_real.py` 的 GMM mode 配置分支。

修复保持另外两个显式模式不变：

```text
--gmm-eval-mode mean    -> policy.use_sample = False
--gmm-eval-mode sample  -> policy.use_sample = True
```

因此需要随机 rollout 时仍可以显式选择 `sample`，但实机默认的 `map` 现在与命令行语义
一致。

## 验证

完成了以下验证：

```bash
conda run --no-capture-output -n pushbox \
  python -m py_compile \
  scripts/deploy_threading_real.py \
  scripts/deploy_threading_real_cartesian.py \
  scripts/diagnose_live_arp_action_shift.py

conda run --no-capture-output -n pushbox \
  pytest -q \
  tests/test_deploy_threading_real.py \
  tests/test_deploy_threading_real_cartesian.py
```

结果：

```text
13 passed
```

动作级重复推理验证结果为 `determinism_max_abs_action_difference = 0.0`。

## 实机运行要求

已经启动的 Python 部署进程不会自动加载代码修改，必须结束并重新启动。ARP 实机初测建议
明确保留以下参数：

```text
--gmm-eval-mode map
--execute-steps 1
```

`--execute-steps 1` 使策略在每次执行后重新观察和规划，便于在确认修复和视觉对齐期间限制
单次错误计划的影响。确认稳定后，再根据闭环测试结果逐步增加执行长度。

修复只保证同一输入不会因为 GMM sampling 而随机改变，不保证模型输出一定正确。如果修复
后仍出现异常动作，应依次检查：

1. 部署进程是否在修复后重新启动，并确认没有显式传入 `--gmm-eval-mode sample`；
2. sideview/frontview 序列号、映射、相机位姿、曝光和白平衡是否与训练一致；
3. 当前 TCP、关节状态和场景阶段是否落在训练数据范围内；
4. 异常动作在固定 observation 的 MAP 重复推理中是否可复现；
5. 原始预测是否在 Cartesian clipping 和 workspace clipping 前已经异常。

## 与 pi0.5 的区别

本问题和 pi0.5 每次从新高斯噪声开始进行 flow-matching 推理不同。pi0.5 的随机初始噪声
是其标准生成过程；是否固定 noise seed 是部署策略选择。ARP 本次问题则是用户选择了 MAP，
但部署代码仍执行随机 GMM sampling，属于命令行配置没有生效。

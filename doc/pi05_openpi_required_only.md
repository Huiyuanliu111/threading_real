# OpenPI π0.5 Required-only：原理与实现设想

日期：2026-09-24。状态：设计提案，尚未实现或验证性能收益。

本文针对本仓库 `pi05_openpi/` 的 JAX/OpenPI 路径。核心设想是：selector 在动作生成前选择长度 `k`，base policy 从一开始就只生成 `k` 步动作，再执行这 `k` 步。Flow matching 在原理上允许这样做；工程实现的可行性较高，但现有 checkpoint 在短 horizon 下的动作质量和端到端加速幅度需要实验确认。

## 1. 问题与术语

| 符号 | 含义 |
|---|---|
| `H` | checkpoint 训练时的动作预测长度，由 `action_horizon` 描述 |
| `k` | 本次选择的执行长度；初期限定为 `1 <= k <= H` |
| `N` | flow matching 采样的数值积分次数，对应 `num_steps` |
| `D` | 模型内部动作维度；当前配置为 32，输出转换后保留 6 维物理动作 |

两种预测模式使用相同的执行长度 `k`：

| 模式 | 实际生成 | 实际执行 |
|---|---|---|
| `full_then_truncate` | `H` 步 | 前 `k` 步 |
| `required_only` | `k` 步 | 全部 `k` 步 |

这里的 action token 指一个动作时间位置在网络中的表示。π0.5 的这条推理路径通过 flow matching 同时更新整段连续动作；`k` 控制序列长度，`N` 控制更新次数。初期比较应固定 `N`，单独研究长度变化。

## 2. 为什么原理上可行

可以从形状为 `[B, k, D]` 的高斯噪声开始，令条件观测为 `o`，通过速度场逐步生成动作：

```text
x_1 ~ Normal(0, I), shape = [B, k, D]
dt = -1 / N
x_(t+dt) = x_t + dt * v_theta(x_t, t, o)
最终输出 x_0，shape = [B, k, D]
```

Flow matching 本身没有要求动作序列必须是固定的 `H` 步。实现上需要网络能接收长度为 `k` 的序列，并使噪声、attention mask、位置索引和输出长度相互一致。

但“能够计算”与“学到了正确的短序列分布”是两个问题。对于只按 `H` 步训练的 checkpoint，缩短序列会改变动作之间的注意力关系；直接生成 `k` 步，不保证等于生成 `H` 步后取前 `k` 步，也不保证具有相同的边缘分布或任务成功率。这是本方案主要的模型效果风险。

OpenPI 的动作投影按动作维度映射，序列长度参与 mask 和采样形状的构造。这提供了复用现有权重实现短序列采样的结构基础。参见固定版本的 [Pi0 / π0.5 实现](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0.py)。


## 4. 推荐实现

### 4.1 显式传入本次预测长度

新增接口的设计示意如下，当前代码尚不提供这些参数：

```python
policy.infer(observation, prediction_horizon=k)
model.sample_actions(rng, observation, prediction_horizon=k, num_steps=N)
```

未传长度时使用 `H`。在进入编译函数前，校验 `k` 是允许集合中的整数且不超过 `H`。初版按单条观测、每次调用一个 `k` 实现；同一个 batch 内混合不同长度不在初版范围内。

`action_horizon` 继续描述 checkpoint 的训练长度。每次调用把 `k` 作为自己的参数，避免在推理期间修改共享模型属性。

### 4.2 让计算形状真正缩短

需要统一处理以下位置：

1. 初始噪声创建为 `[B, k, D]`。如果调用者提供噪声，显式检查其形状。
2. `embed_suffix()` 根据 `noisy_actions.shape[1]` 构造动作 mask。当前继承实现仍使用固定 horizon，因此只提供短噪声不足以完成改造。
3. 去噪循环始终处理这 `k` 个动作位置，输出投影取对应的实际动作长度。
4. 位置索引和 prefix/suffix attention 按实际形状构造，维持原有注意力语义。
5. 后处理返回 `[k, 6]`，执行侧以本次返回的长度安排执行和下一次重规划。

验收要确认每轮动作后缀的实际长度为 `k`。构造完整 `H` 步后再切片，或者仅将后面的动作遮住但仍保持完整计算形状，均不能作为“计算量随 `k` 缩短”的证明。

### 4.3 JAX：静态长度、提前编译、运行时选择

JAX 编译需要可确定的数组形状。建议将 `prediction_horizon` 声明为 JIT 静态参数，为允许的长度缓存编译结果，启动时逐一预热并等待计算完成。

例如，H50 的首轮实验可选择 `{10, 25, 50}`。这只是实验候选，不是已验证的最优长度。不同长度共用同一套模型参数，但会产生各自的编译结果和运行内存需求；不应通过反复加载 checkpoint 来实现长度选择。

如果 selector 输出任意整数，就需要覆盖所有允许长度的编译与预热，或明确将输出量化到有限候选集合。量化会改变 selector 的实际决策，必须记录原始选择和最终使用值。

尤其应避免直接沿用“临时修改模型配置”的方式。OpenPI 的 `Policy` 使用 `module_jit` 包装采样；该工具冻结包装时的模型状态。包装后再修改原对象属性不会直接更新包装函数捕获的状态。参见 [Policy](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/policies/policy.py) 与 [module_jit](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/shared/nnx_utils.py)。

### 4.4 Selector 与视觉特征复用

第一阶段先使用固定 `k` 验证生成机制，再接入 selector。最终建议的流程是：

```text
观测 → 图像编码 → 视觉特征 → selector → k
                      ↓                 ↓
             构建 prefix / KV cache → 选择 k 对应的采样函数
                                        ↓
                              k 个动作位置，N 次积分
                                        ↓
                                  后处理并执行 k 步
```

应以显式参数传递本次视觉特征或 prefix cache，避免临时替换图像编码函数。缓存只在同一次观测的推理内复用；初版不假定不同观测之间可以复用。

selector 必须能在完整动作生成之前给出 `k`。如果选择依赖先生成完整 `H` 步才能取得的信息，就不能直接获得这里设想的生成节省。

### 4.5 部署协议与并发

远程部署应由服务端在输入变换前读取请求长度，或在服务端运行 selector；不能假定往现有观测字典加入 `horizon` 就会自动传入模型。建议响应记录：

```text
prediction_mode, training_horizon, requested_horizon,
predicted_horizon, execution_horizon, actions
```

`required_only` 下验证 `predicted_horizon == execution_horizon == k`；`full_then_truncate` 下生成长度为 `H`，执行长度为 `k`。本地适配器、远程客户端和控制循环都应区分训练长度与本次长度。

显式传参可以消除临时修改 horizon 带来的竞争，但不自动保证整个服务可并发。请求 RNG、观测缓存、通信连接和执行队列仍需独立管理。初版可以继续串行处理推理请求。

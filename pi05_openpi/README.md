# OpenPI π0.5 Threading：6 维动作，无夹爪

本机实机部署见 [部署命令](../../doc/deploy_pi05_openpi.md)：2000 步权重已复制到本机，4060 Ti 直接推理，本机连接相机。使用 `bash threading_real/pi05_openpi/run_deploy.sh`；默认完整预测 H50、固定执行 1 步，需显式添加 `--execute --confirm-real-robot` 才执行模型动作。

当前默认：LoRA、30 Hz、chunk=50、10,000 steps；每 2,000 steps 完整验证，仅保留 val loss 最低的 checkpoint。
新启动训练默认启用 W&B **online**（项目 `threading_pi05_openpi`），可显式用 `--no-wandb` 关闭。
启用时强制 online，不受遗留 WANDB_MODE=offline 影响；服务器已存在登录凭据。
当前已经运行的 v4 进程是在 W&B 关闭时启动，此默认值修改不会热更新该进程，也不会为切换日志而重启训练。

基础权重为 `gs://openpi-assets/checkpoints/pi05_base/params`，没有使用 LIBERO 数据或权重。
OpenPI 源码基于 revision `215abfb217dbac7d5f1273282331b9b1866c0479`。

**当前 6D v4 训练已于 2026-09-20 按用户要求停止，GPU 已释放。** 服务器旧实验 v3 的第 2,000 步 checkpoint 已按用户要求删除；日志和旧配置保留。
旧实验夹爪归一化尺度过小，首次 val loss=3104.78，不能视为有效收敛。
新实验名 `threading_lora_tcp6_30hz_h50_best_v4`；不要用旧 checkpoint 的 `--resume` 恢复本实验。

## 数据和物理语义

原始记录来自 `/home/huiyuan/threading_new/threading_new_1` 和 `threading_new_2`。
当前数据集为 `data/threading_tcp6_nosmooth_30hz`，80 episodes、45,415 帧。

- **动作 6 维**：`[dx, dy, dz, drotvec_x, drotvec_y, drotvec_z]`，相邻实测 TCP 位姿增量，基座坐标系，单位米/弧度。
- **状态 9 维**：TCP 位置 3 维 + 旋转矩阵前两列 6 维；没有夹爪宽度。
- 彻底移除 `action[6]` 和 `observation.state[9]`，包括数据 schema、逐 episode 统计、模型输入/输出和训练归一化统计。不是保留第七维后清零。
- 为匹配 π0.5 基础权重，模型内部 state/action 仍补零至 32 维；推理输出为 `50×6`，不产生夹爪指令。
- 夹爪保持闭合由外部控制器负责；本目录的离线推理不连接机器人，也不会主动控制夹爪。

## 数据处理：完整披露

| 项目 | 行为 |
|---|---|
| 图像 | 原始 cam1、cam3 完整 RGB，不使用深度或 ROI |
| 尺寸 | 全图等比例 INTER_AREA 缩放，居中补黑边至 224×224 |
| 时间对齐 | cam1 为参考，最近邻匹配 cam3 和机器人主机时间戳；相机 ≤25ms、机器人 ≤5ms，不插值 |
| 频率 | 保留全部有效对齐帧，30Hz，不隔帧、不降采样；真实间隔要求 20–50ms |
| 平滑 | 无滤波、平滑或轨迹重建 |
| 筛选 | 不删静止/微小动作、不按 teleop-active 截取、不按成功筛选 |
| 运动学 | 原始实测关节角做 Panda FK：panda_link0 → panda_hand_tcp |
| 动作 | 实测相邻位姿差和 `log(R_next R_current^T)`，不是下发目标或速度 |
| 夹爪 | 状态和动作中均删除；保留原始记录供追溯 |
| 尾帧 | 最后实测位姿仅作为前一步的动作终点 |
| 存储 | 无损 PNG 内嵌 parquet，保留全图像素；索引时间戳为 frame_index/30 |
| 追溯 | meta/alignment 保存原始帧号、机器人行号、真实时间戳和对齐误差 |

已有全图 30Hz 数据逐帧核对过原始记录。本次通过 `remove_gripper_dataset.py` 仅删除两列，
没有重复解码视频；所有保留数值、图像 bytes、时间戳、索引逐文件精确比较。
证明记录在 `meta/gripper_removal.json`，再由 `verify_raw_dataset.py` 验证剩余状态/动作与原始 FK 一致。
直接从原始数据重新构建时，`build_raw_dataset.py` 也默认生成同样的 9D state / 6D action。

```bash
# 从已核对的原始数据派生产物中精确删除夹爪（输出拒绝覆盖）
python threading_real/pi05_openpi/remove_gripper_dataset.py
# 或从原始视频/机器人记录重新构建，需要现有 LeRobot v3 数据构建环境
threading_real/pi05/.venv-deploy/bin/python threading_real/pi05_openpi/build_raw_dataset.py
threading_real/pi05/.venv-deploy/bin/python threading_real/pi05_openpi/verify_raw_dataset.py
```

## 训练时处理与技巧

- cam3/front → `base_0_rgb`；cam1/side → `left_wrist_0_rgb`；第三视角补零并屏蔽。第二槽位不代表物理腕部相机。
- 官方训练图像增强：front 随机约 95% 裁剪并缩放、旋转 ±5°；两路均有颜色扰动（brightness=.3、contrast=.4、saturation=.5）。验证/推理关闭这些增强。
- 归一化使用训练集 1%/99% 分位数，不裁剪异常值；50 步重叠 chunk 包括 episode 尾部重复最后动作的填充，不跨 episode。
- state 离散化进文本 token，state/action 补零至 32 维；没有额外 state dropout。
- 官方 flow matching，加高斯噪声，时间参数为 `Beta(1.5,1)*0.999+0.001`；loss 沿用官方补齐维度平均。
- LoRA 使用 `gemma_2b_lora` 和 `gemma_300m_lora`，关闭 EMA。官方冻结规则仅冻结 llm 内非 LoRA 参数；视觉编码器、动作投影等 llm 外参数仍可训练，不是仅训练 adapter。
- AdamW、梯度范数裁剪 1.0、warmup=250、peak LR=2.5e-5、cosine decay；不额外修改学习率分组。
- seed=42，固定 72 条训练 / 8 条验证，40,722 / 4,693 帧。验证 episode `[0,25,28,33,40,51,60,61]`，不参与归一化统计。
- 每 2,000 steps 验证全部样本；固定随机噪声/时间种子，尾 batch 填充不计入 loss。LoRA 评估实际保存的当前参数。
- 只有有限且严格更低的 val loss 才保存，新 checkpoint 成功后删除旧最优；不另存 latest/final。val loss 不是实机成功率。

## 准备与训练命令

```bash
bash threading_real/pi05_openpi/bootstrap.sh
OPENPI_PY="$PWD/threading_real/pi05_openpi/vendor/openpi/.venv/bin/python"
COMMON=(--mode lora --exp-name threading_lora_tcp6_30hz_h50_best_v4 --batch-size 12 --fsdp-devices 2)
JAX_PLATFORMS=cpu "$OPENPI_PY" threading_real/pi05_openpi/run.py check "${COMMON[@]}"
JAX_PLATFORMS=cpu "$OPENPI_PY" threading_real/pi05_openpi/run.py norm "${COMMON[@]}"
# 启动新实验：
CUDA_VISIBLE_DEVICES=0,1,3,4,5,6 "$OPENPI_PY" threading_real/pi05_openpi/run.py train "${COMMON[@]}"
```

原始数据构建使用 LeRobot v3 环境，训练读取器使用固定版本的 LeRobot v2.1 API；数据格式为 v2.1。
统计使用官方 RunningStats，全部训练起始帧参与。训练循环复用官方 init_train_state/train_step，增加完整验证与 best-only 保存。
资产、配置、checkpoint 按实验隔离。训练开始后禁止重算统计或改配置；恢复同一实验使用 `--resume`。
恢复从最优步继续，未保存的后续更新会丢失；数据加载器会重新初始化，不保证精确复现中断时 batch 顺序。

```text
assets/<exp-name>/pi05_threading_lora/<repo-id>/norm_stats.json
checkpoints/pi05_threading_lora/<exp-name>.json
checkpoints/pi05_threading_lora/<exp-name>/<best_step>/{params,train_state,assets,metrics}
checkpoints/pi05_threading_lora/<exp-name>/validation.jsonl
```

## 离线推理

`infer_one.py --base` 使用基础权重及新数据集统计做接口检查，输出 `50×6`；不是已微调的任务策略。
训练后的模型使用 `--run-config <新实验.json> --checkpoint <新实验/best_step>`，读取 checkpoint 自带统计。
`--num-steps` 默认 10，是 flow-matching 去噪步数，与 chunk=50 不同；`--warmup 1` 可预热后计时。
旧 v3 的 7D 动作 / 10D 状态 checkpoint 已删除；原始代码仍在服务器 `archive/gripper_v3_code`，与新 v4 不兼容。

## 服务器 10.157.174.249

目录 `/home/huiyuan/threading_real/pi05_openpi`。旧 v3 已停止，其 checkpoint 2000 已删除。
6D v4 已停止（原 PID=4109880），最后记录第 8940 步；日志 `logs/train_tcp6_v4_20260919.log`。
保留最优第 2000 步 checkpoint，val loss=0.02955778；没有另存停止时的非最优权重，不自动重启。
`logs/train_20260919.log` 是旧 v3 日志，不要用于判断新训练进度；旧暂停记录已归档。

现有 `.venv` 只读复用 `/home/huiyuan/pi05/.venv-pi05` 的共享依赖，并在本环境覆盖 datasets=3.6.0、huggingface-hub=0.32.0。
JAX=0.5.3、Flax=0.10.2、Orbax=0.11.13、Torch=2.7.1、cuDNN=9.5.1.17。共享依赖目录必须保留。
固定版读取器在 `vendor/lerobot`；上游源码在 `vendor/openpi`；缓存隔离到本实验 `.cache`。
`start_remote.sh` 默认六张卡 0,1,3,4,5,6、global batch=12、FSDP=2，检查 GPU 空闲及磁盘剩余至少 55GiB，flock 防重复启动。
首次 JAX 编译和初始化需要数分钟。显存预分配比例 0.85；这不是实时张量显存用量。

用户授权的旧 threading 实验清理清单在 `outputs/threading_cleanup_20260919.json`；其他实验、环境和原始数据未删除。

## 第 2000 步模型的视觉敏感度诊断

2026-09-20 已完成 8 条验证轨迹 × 3 个阶段 × 2 个噪声种子的配对检查，共 384 次推理。
固定 state、prompt、初始 flow noise 和 JAX RNG，仅替换/遮黑 front 或 side 图像，另有重复输入及噪声对照。
没有训练或机器人控制。front=cam3，side=cam1。结果保存在 `outputs/visual_sensitivity_2000_tcp6/`。

替换 front / side / 两路的平均 chunk 平移响应分别为 0.03580 / 0.29022 / 0.29031 mm/步；
相同输入噪声重复差异为零，仅改变噪声为 0.53456 mm/步。说明模型有视觉响应，且本次更敏感于 side。
该诊断不能证明正确视觉定位或实机任务成功；详见 report_zh.md 的限制说明。

服务器复现命令（输出需使用新的目录名）：

```bash
cd /home/huiyuan/threading_real/pi05_openpi
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_MEM_FRACTION=0.75 \
  PYTHONPATH="$PWD/vendor/lerobot" HF_HOME="$PWD/.cache/huggingface" \
  OPENPI_DATA_HOME="$PWD/.cache/openpi" HF_LEROBOT_HOME="$PWD/.cache/lerobot" \
  .venv/bin/python ../scripts/diagnostics/openpi_visual_sensitivity.py \
  checkpoints/pi05_threading_lora/threading_lora_tcp6_30hz_h50_best_v4/2000 \
  --run-config checkpoints/pi05_threading_lora/threading_lora_tcp6_30hz_h50_best_v4.json \
  --output outputs/visual_sensitivity_2000_tcp6_repeat
```

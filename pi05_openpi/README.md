# OpenPI π0.5 Threading：6 维动作，无夹爪

> 2026-09-21：v4 已按用户要求删除，本机和服务器的2000步权重及对应配置已移除。下文引用v4的部署命令和验证结果仅为历史记录，不能直接启动；需先配置新的checkpoint。

历史本机部署见 [部署命令](../../doc/deploy_pi05_openpi.md)：2000 步权重当时复制到本机，4060 Ti 直接推理，本机连接相机。使用 `bash threading_real/pi05_openpi/run_deploy.sh`；默认完整预测 H50、固定执行 50 步，需显式添加 `--execute --confirm-real-robot` 才执行模型动作。

训练参数默认沿用 v6 配方（不表示该实验仍在运行）：action expert 全量微调（427,932,672 参数），视觉编码器注意力 Q/K/V/out 与 MLP 使用 LoRA（rank=16、alpha=16，8,695,296 参数），动作投影及时间 MLP 全量训练（2,165,792 参数）。总可训练参数 438,793,760；语言模型与视觉非 LoRA 权重冻结。30 Hz、chunk=50、10,000 steps；每 1,000 steps 验证，每 2,000 steps 保存并保留全部定期 checkpoint。
新启动训练默认启用 W&B **online**（项目 `threading_pi05_openpi`），可显式用 `--no-wandb` 关闭。
启用时强制 online，不受遗留 WANDB_MODE=offline 影响；服务器已存在登录凭据。
v4/v5/v6 均已按用户要求停止。v5 于 2026-09-21 按用户要求停止，最后记录 step=700，未到第一个 checkpoint 保存步。v6 使用独立实验，从 pi05_base 初始化。

基础权重为 `gs://openpi-assets/checkpoints/pi05_base/params`，没有使用 LIBERO 数据或权重。
OpenPI 源码基于 revision `215abfb217dbac7d5f1273282331b9b1866c0479`。

**当前 6D v4 训练已于 2026-09-20 按用户要求停止，GPU 已释放。** 服务器旧实验 v3 的第 2,000 步 checkpoint 已按用户要求删除；日志和旧配置保留。
旧实验夹爪归一化尺度过小，首次 val loss=3104.78，不能视为有效收敛。
新实验名 `threading_tcp6_30hz_h50_vision_lora_action_full_v6`，从 pi05_base 初始化；不要使用 v4 checkpoint 的 `--resume`。历史配置缺少冻结开关时仍按旧规则加载，兼容 v4 推理。

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
- 默认 `--mode vision_lora_action_full`：语言模型使用冻结的 `gemma_2b`，action expert 使用全量可训练的 `gemma_300m`（含自适应归一化参数）；视觉 LoRA 由本目录 `siglip_lora.py` 和 `vision_lora_config.py` 提供。保留原始视觉参数路径，A 随机初始化、B 零初始化；patch/position embedding、LayerNorm 和视觉输出 head 冻结。`--vision-lora-rank 16`，alpha 等于 rank。关闭 EMA。历史 `--mode lora` 配置仍兼容。
- AdamW、梯度范数裁剪 1.0、warmup=250、peak LR=2.5e-5、cosine decay；不额外修改学习率分组。
- seed=42，固定 72 条训练 / 8 条验证，40,722 / 4,693 帧。验证 episode `[0,25,28,33,40,51,60,61]`，不参与归一化统计。
- `--eval-interval 1000`：每 1,000 steps 验证全部样本；固定随机噪声/时间种子，尾 batch 填充不计入 loss。LoRA 评估当前参数。
- `--save-interval 2000`：每 2,000 steps 保存并保留全部 checkpoint，无论 val loss 是否改善。保存周期必须是验证周期的整数倍。验证记录中的 improved 表示评估最优，不保证该步有 checkpoint（如第 1,000 步）；从已保存步中按 val loss 选择部署权重。val loss 不是实机成功率。

## 准备与训练命令

```bash
bash threading_real/pi05_openpi/bootstrap.sh
OPENPI_PY="$PWD/threading_real/pi05_openpi/vendor/openpi/.venv/bin/python"
COMMON=(--mode vision_lora_action_full --exp-name threading_tcp6_30hz_h50_vision_lora_action_full_v6 --vision-lora-rank 16 --eval-interval 1000 --save-interval 2000 --wandb --batch-size 12 --fsdp-devices 2)
JAX_PLATFORMS=cpu "$OPENPI_PY" threading_real/pi05_openpi/run.py check "${COMMON[@]}"
JAX_PLATFORMS=cpu "$OPENPI_PY" threading_real/pi05_openpi/run.py norm "${COMMON[@]}"
# 启动新实验：
CUDA_VISIBLE_DEVICES=0,1,3,4,5,6 "$OPENPI_PY" threading_real/pi05_openpi/run.py train "${COMMON[@]}"
```

原始数据构建使用 LeRobot v3 环境，训练读取器使用固定版本的 LeRobot v2.1 API；数据格式为 v2.1。
统计使用官方 RunningStats，全部训练起始帧参与。训练循环复用官方 init_train_state/train_step，增加完整验证与定期 checkpoint 保存。
资产、配置、checkpoint 按实验隔离。训练开始后禁止重算统计或改配置；恢复同一实验使用 `--resume`。
恢复从最新已保存步继续，未保存的后续更新会丢失；数据加载器会重新初始化，不保证精确复现中断时 batch 顺序。

```text
assets/<exp-name>/pi05_threading_<mode>/<repo-id>/norm_stats.json
checkpoints/pi05_threading_<mode>/<exp-name>.json
checkpoints/pi05_threading_<mode>/<exp-name>/<step>/{params,train_state,assets,metrics}
checkpoints/pi05_threading_<mode>/<exp-name>/validation.jsonl
```

## 离线推理

`infer_one.py --base` 使用基础权重及新数据集统计做接口检查，输出 `50×6`；不是已微调的任务策略。
训练后的模型使用 `--run-config <新实验.json> --checkpoint <新实验/best_step>`，读取 checkpoint 自带统计。
`--num-steps` 默认 10，是 flow-matching 去噪步数，与 chunk=50 不同；`--warmup 1` 可预热后计时。
旧 v3 的 7D 动作 / 10D 状态 checkpoint 已删除；原始代码仍在服务器 `archive/gripper_v3_code`，与新 v4 不兼容。

## 服务器 10.157.174.249

历史记录：v6 曾于 2026-09-21 启动（原 PID 4144880，现已停止），首步 loss=0.08424887；日志 `logs/train_tcp6_vision_lora_action_full_v6_20260921.log`；[W&B v6](https://wandb.ai/huiyuan_tac/threading_pi05_openpi/runs/jzrhwvvb)。

历史清理记录（早于 v4 删除）：已删除旧 v3 的 `data/threading_full_nosmooth_30hz/data`（7D 动作 Parquet）及来源对应的 Hugging Face 缓存，释放约 6.31 GiB。旧元数据、配置、日志保留；当前 tcp6 数据、v4 部署 checkpoint、基础权重及环境保留。清单见 `outputs/threading_cleanup_20260921.json`。

目录 `/home/huiyuan/threading_real/pi05_openpi`。旧 v3 已停止，其 checkpoint 2000 已删除。
6D v4 已停止（原 PID=4109880），最后记录第 8940 步；日志 `logs/train_tcp6_v4_20260919.log`。
历史最优第 2000 步 checkpoint 的 val loss=0.02955778，该权重现已删除；没有另存停止时的非最优权重，不自动重启。
`logs/train_20260919.log` 是旧 v3 日志，不要用于判断新训练进度；旧暂停记录已归档。

现有 `.venv` 只读复用 `/home/huiyuan/pi05/.venv-pi05` 的共享依赖，并在本环境覆盖 datasets=3.6.0、huggingface-hub=0.32.0。
JAX=0.5.3、Flax=0.10.2、Orbax=0.11.13、Torch=2.7.1、cuDNN=9.5.1.17。共享依赖目录必须保留。
固定版读取器在 `vendor/lerobot`；上游源码在 `vendor/openpi`；缓存隔离到本实验 `.cache`。
`start_remote.sh` 默认六张卡 0,1,3,4,5,6、global batch=12、FSDP=2，检查 GPU 空闲及磁盘剩余至少 55GiB，flock 防重复启动。
首次 JAX 编译和初始化需要数分钟。显存预分配比例 0.85；这不是实时张量显存用量。

用户授权的旧 threading 实验清理清单在 `outputs/threading_cleanup_20260919.json`；其他实验、环境和原始数据未删除。

## chunk=10 对照实验（v11）

启动脚本 `start_v11_remote.sh`，参考v6归档的 `archive/pre_native640_20260921/run.py` 默认值及本README中的v6启动参数，唯一训练配方变化为horizon=50→10（30Hz下约0.333秒）。v6原始run配置和W&B记录目前不可用，不能声称已对其原始manifest逐项比对。

恢复原80条224数据，seed42按72条训练/8条验证；cam1+cam3、本体状态、默认图像增强、vision LoRA rank16 + action expert全量、batch12/FSDP2、lr2.5e-5、warmup250、10000步均沿用v6。每1000步验证、2000步保存，保留全部5份；从pi05_base初始化，单独重算chunk10归一化。新实验名 `threading_tcp6_30hz_h10_vision_lora_action_full_v11`，日志 `logs/train_tcp6_h10_v11_20260921.log`。首次启动设置 `V10_HANDOFF_PID=4181918`，等待v10的500步checkpoint提交后停止v10再接管六卡；独立启动时不设置该变量。

## 单轨迹过拟合实验（v10）

2026-09-21 按用户要求停止 v9，改用 `start_overfit1_remote.sh` 从 pi05_base 重新训练。`--overfit-episodes 1 --seed 42` 选择原10条中的 episode 33，共655帧；训练、归一化及同集评估仅使用该轨迹。其余配置沿用v9：单cam1 native640、vision LoRA + action expert全量、batch=6、10000步、每500步评估保存、保留最近5份。独立实验名为 `threading_tcp6_cam1_native640_overfit1_v10`，日志 `logs/train_cam1_native640_overfit1_v10_20260921.log`。

v10首次六卡运行55分钟未完成首步，已停止并归档到服务器 `archive/overfit1_v10_stalled_20260921`。2026-09-21改用GPU 0、1，保持global batch=6、FSDP=2，确认step20参数范数发生变化，约1.8秒/步；新W&B为 https://wandb.ai/huiyuan_tac/threading_pi05_openpi/runs/hvepooxy 。具体卡住原因尚未定位。训练日志现在记录前5步，非评估步即时提交W&B，评估步与评估指标合并提交。

## 10 episodes 过拟合实验（v9，已停止）

2026-09-21 按用户要求停止 v8（最后记录 step 780，未产生 checkpoint），启动入口为 `start_overfit10_remote.sh`。沿用单 cam1 原始像素、vision LoRA + action expert 全量微调，重新从 pi05_base 初始化。

`--overfit-episodes 10 --seed 42` 固定选择 episode `[0, 21, 25, 28, 33, 37, 40, 51, 60, 61]`，共 5932 帧。训练、归一化和评估仅使用这 10 条完整轨迹；复用原始数据集，通过全局帧索引筛选，不修改标签或复制图像。batch=6，10000 steps，每500 steps评估、500 steps保存，保留最近5份checkpoint以控制磁盘占用。归一化和checkpoint均使用独立实验名 `threading_tcp6_cam1_native640_overfit10_v9`。

W&B评估指标为 `train_eval_loss`，scope为 `train_reconstruction`：它是同一训练集上的固定噪声flow loss，不是独立验证误差，也不等价于真实动作误差或实机成功率。日志为 `logs/train_cam1_native640_overfit10_v9_20260921.log`。

清理审计见 `logs/cleanup_before_overfit10_v9.json`：仅移除服务器旧224数据和对应可重建HF缓存，释放约6.3GiB；删除前167个数据文件SHA256全部与本机持久副本一致。保留基础权重、原始数据和新实验复用的native数据缓存。

## cam1 原始 640×480 输入（v8，已停止）

`run_native640.sh` 默认选择独立 v8 实验 `threading_tcp6_cam1_native640_vision_lora_action_full_v8`，通过 `--camera-views cam1` 只使用 cam1；保留 v6 的 action expert 全量 + vision LoRA 训练范围，总可训练参数仍为 438,793,760。
图像来自原始 640×480 视频帧，数据集保存未缩放 RGB；输入模型前左右各补 2、上下各补 5 像素黑边，得到 644×490。没有裁剪、缩放或图像增强。
视觉位置编码保留预训练 16×16 参数，运行时双三次插值到 35×46；cam1 为唯一视觉输入，共 1,610 tokens，保留 `left_wrist_0_rgb` 槽位名；cam3 和空白槽位不进入视觉编码器或语言模型。
高分辨率数据在 `data/threading_tcp6_native640_30hz`；数值标签和对齐不变。预览：`outputs/input_native640_examples/`。
输入尺寸和相机选择由 run-config 的 `image_profile`、`camera_views` 和服务端 metadata 决定；单路部署仅采集/发送 cam1，旧 checkpoint 缺省仍走 224×224。不要把 224 数据放大冒充原图。

```bash
# 先完成数据重建；原始视频在本机 /home/huiyuan/threading_new。
OPENPI_PY=<OpenPI环境的python>
"$OPENPI_PY" threading_real/pi05_openpi/build_native_dataset.py
JAX_PLATFORMS=cpu OPENPI_PY="$OPENPI_PY" bash threading_real/pi05_openpi/run_native640.sh check
JAX_PLATFORMS=cpu OPENPI_PY="$OPENPI_PY" bash threading_real/pi05_openpi/run_native640.sh norm
```

已通过服务器单卡完整基础模型推理测试，输出 50×6 有限值，见 `outputs/native640_setup/model_smoke.json`。训练使用 batch=6/FSDP=2，启动结果见 `outputs/cam1_native640_setup/launch.json`（生成后）。旧权重不代表已经学会高分辨率单相机输入。

服务器启动脚本为 `start_native640_remote.sh`，先计算训练集归一化，再启动训练；日志为 `logs/train_tcp6_cam1_native640_v8_20260921.log`。由于持久磁盘空间有限，服务器数据副本位于 `/dev/shm/huiyuan_pi05_v8/data/threading_tcp6_native640_30hz`，HF datasets 缓存也放在同一内存文件系统；checkpoint、归一化统计和日志仍保存到项目的持久磁盘。服务器重启后须从本机重新同步数据，再使用相同配置恢复训练；本机完整数据不受影响。该脚本用于首次启动，不用于已有 checkpoint 的恢复。

单路模型通过 loss/sample 形状、冻结参数量和输入像素测试，完整 GPU 推理输出 50×6；固定噪声时新增/替换 cam3 输出完全不变。见 `outputs/cam1_native640_setup/model_smoke.json`。原双路数据可复用，仅训练输入选择 cam1，不删除原始 cam3 数据。缺少 `camera_views` 的历史配置仍使用双路。

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

远程推理可使用独立入口 `run_deploy_remote.sh`，相机与follower控制仍在本机，见 [服务器推理部署](../../doc/deploy_pi05_openpi_remote.md)。

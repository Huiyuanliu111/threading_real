# ARP 单帧输入、连续数据与六维训练方案

日期：2026-09-10。落实重新审计后的调整；本次没有连接机器人或发送运动指令。

## 已完成的改动

输入改为同一观测时刻的两路固定相机各一帧，以及一组 TCP rotation-6D+夹爪宽度状态。`policy.n_obs_steps=1`，预测 20 步，dataset horizon 同步改为 20，首个动作目标索引为 0。没有 wrist，也没有语言输入。

单帧解决的是历史两帧时间间隔不一致；它不消除采集/推理延迟，也不保证一次执行长 chunk 时闭环足够快。现有 Cartesian 部署脚本会从 checkpoint 读取帧数，无需强行覆盖旧模型参数。旧两帧、七维预测头 checkpoint 不能作为本方案的直接恢复训练入口。

模型内部 fine-action token 从 7D 改为 6D，只学习平移和旋转。对外动作接口仍返回 7D，最后一维强制为零，与现有锁夹爪部署兼容。新参数 `arm_only_action=true` 只在新配置中启用；旧配置默认行为保留。

修复验证时的状态 dropout：现在同时检查 teacher-forcing 标志和 `self.training`。新配置训练时也设状态 dropout=0，保留亮度、对比度、饱和度、色相和噪声增强，取消未经几何校准的随机图像平移。

## 新数据

路径：`/home/huiyuan/teleoperation/data/threading_combined_80_arp_continuous_15hz`

- 80 条轨迹，22659 帧，15 Hz；64 train / 16 validation。
- 每个 episode 建立固定时间网格，按机器人原始时间戳插值 joint+width，再做 FK。
- cam1 和 cam3 独立按时间戳匹配到该网格，保存 224×224 视频，默认训练缩小到 96×96。
- action 直接由相邻网格上的 TCP 位姿计算，采用基座坐标系平移增量、左乘旋转增量。
- 不做 SG 平滑，不按动作大小删除内部帧，保留微调和静止段。
- 检测夹爪宽度相对初始抓取增加超过 2 mm 且连续 3 个网格点保持的释放事件，在事件开始处裁掉尾段。共有 4 条记录触发：episode 22、27、30、31。其他记录在相机有效时间范围内没有触发该条件。
- 最后一个真实状态作为最后 action 的端点，不制造末尾零动作标签。保留源时间戳与两路相机原始帧号，便于追溯。

全量校验结果：160 个视频的帧数均与对应 parquet 一致；全部相邻源时间间隔为 1/15 秒（纳秒取整误差以内）；全部相邻 action 与 FK 重算逐值一致；夹爪 action 全零。整条轨迹平移累加的最大闭合误差约 1.91e-6 mm。最大相机时间匹配误差 17.295 ms，最大机器人最近时间点误差 2.659 ms。

这些检查证明时间/标签一致性，不证明每条示范插孔成功。释放检测也不等价于成功标注。本次查看了被裁剪的 4 条记录尾段图示；如需按接触或成功阶段进一步裁剪，builder 提供 `--crop-manifest` 显式时间范围，不能用“最后百分之几”替代阶段标注。

新目录采用现有 `ThreadingRealLeRobotDataset` 支持的本地 LeRobot v2 文件布局，未生成 Hub 发布所需的全部元数据。它与现有 pi0.5 数据目录分开。

## 验证和 checkpoint 选择

每个 epoch（从 epoch 1 开始）完整验证全部 16 条验证轨迹，共 4388 个窗口；`max_val_steps=null`、`max_validation_sequences=null`。单帧窗口允许在尾部 padding，但无效目标不参与物理误差，累计 10/20 步指标仅使用对应长度完整的窗口。

新日志包含：

- `val_xyz1_mm`、`val_rotvec1_deg`：首步平移、旋转向量误差。
- `val_xyz_mm`、`val_rotvec_deg`：全部有效预测动作误差。
- `val_xyz_sum10_mm`、`val_xyz_sum20_mm`：累计平移误差。
- 对应的 `baseline_mean_*` 和 `baseline_zero_*` 指标。
- `val_observations`、`val_valid_actions`：实际覆盖数。

GMM 使用最高混合权重分量均值做确定性物理指标评估。常量均值基线只由训练 episode 的动作计算。top-k checkpoint 按 `val_xyz1_mm` 最小值选择，同时应检查旋转和累计误差；六维 NLL 仍记录，但不是唯一选模标准。

完整训练器 smoke 检查已实际跑过一轮全量验证：4388 个 observation、84720 个有效目标，保存了 latest 和按毫米误差命名的 top-k checkpoint。该 smoke 仅做两次优化器更新，权重不能用于部署。保存后的 EMA 经部署 loader 严格加载 220/220 参数，单帧输入输出 `[1,20,7]`，夹爪增量全零。

## 小样本拟合结果

训练 episode 0 均匀抽取 32 个完整 20-step 窗口，固定随机种子 42，训练和评估使用相同窗口。关闭所有随机增强和 dropout；每步 batch=16。优化器主干与投影 lr=2e-4，视觉 encoder lr=1e-5，不使用 EMA。其目的仅为检查拟合能力。

| 方法 | 更新步数 | 首步平移误差 | 首步旋转向量误差 | 10 步累计平移误差 |
|---|---:|---:|---:|---:|
| 样本平均动作 | — | 1.082 mm | 0.1241° | 9.525 mm |
| GMM6 | 400 | 0.802 mm | 0.1045° | 4.877 mm |
| Regression6 / Smooth-L1 | 400 | 0.827 mm | 0.1100° | 5.710 mm |
| GMM6 | 2000 | 0.105 mm | 0.0158° | 0.832 mm |
| Regression6 / Smooth-L1 | 2000 | 0.032 mm | 0.0061° | 0.288 mm |

两种预测头都能拟合这些精细动作，回归头在该受控实验中更好。不能把这张表解释为留出验证集性能或实机成功率；也不能将它直接与旧数据 80 个验证窗口的误差比较。正式训练保留原先的较低学习率（主网络和投影 5e-5，encoder 1e-6），并启用图像增强，训练设置与容量测试不同。

## 正式训练命令

数据已生成，无需再次运行 builder。建议先训练六维回归版本，并以同数据、同训练参数的 GMM 版本做对照。两者仍是 ARP，区别是连续动作预测头与损失。

```bash
cd /home/huiyuan/teleoperation/threading_real
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/home/huiyuan/miniconda3/envs/pushbox/bin/python pushbox/train.py \
  --config-name=threading_combined_80_arp_single_frame_regression
```

GMM 对照只替换配置名：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/home/huiyuan/miniconda3/envs/pushbox/bin/python pushbox/train.py \
  --config-name=threading_combined_80_arp_single_frame
```

均为从头训练 80 epochs、batch=64、无旧 checkpoint 恢复、默认 W&B offline。Hydra 自动创建新的输出目录。本次尚未启动上述正式 80-epoch 训练。

另有 `threading_combined_80_arp_single_frame_224` 配置供分辨率对照：两路全图 224×224、batch=16、梯度累积 4 次。它尚未训练，不能声称高分辨率已改善性能。它也不是经过标定的孔区域 ROI。

## 文件与测试

- 构建脚本：`scripts/build_arp_continuous_dataset.py`。
- 容量测试：`scripts/check_arp_single_frame_fit.py`，可用 `--config`、`--steps`、`--episode` 调整。
- 数据报告：新数据的 `meta/continuous_build_report.json`；独立全量校验为 `artifacts/arp_single_frame_data_check.json`。
- 拟合报告：`artifacts/arp_single_frame_fit_{gmm6,regression6}{,_2000}/report.json`。这些目录的 `diagnostic_only.ckpt` 只适合离线诊断。
- 训练器检查：`artifacts/arp_single_frame_workspace_smoke/`。
- 裁剪图示：`artifacts/arp_single_frame_release_crops.png`。

新增 5 项测试全部通过，覆盖单帧六维 GMM/回归反向传播、验证无状态 dropout、夹爪标签不影响损失、七维部署接口、padding 指标、释放裁剪，以及视频缓存到网络输入的数值范围。视频缓存改为 uint8，避免完整轨迹缓存被 float32 放大四倍。

已有集成/Cartesian 部署/FK 测试共 34 项通过，3 项因仓库缺少旧仿真配置 `threading_arp.yaml`、`threading_arp_v2.yaml`、`threading_arp_v3.yaml` 而失败，未修改这些无关配置。

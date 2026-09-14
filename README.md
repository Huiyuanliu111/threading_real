# Threading Real

这个仓库只保留真实 Threading 的两条策略管道：ARP 与 π0.5。两者都经过相同阶段：
训练数据与模型、chunk selector、推理与真机部署。

| 阶段 | ARP | π0.5 |
| --- | --- | --- |
| 训练 | `scripts/arp/train.py`、`scripts/arp/validate_dataset.py` | `pi05/training/` |
| Chunk selector | `scripts/chunk_selector/`（训练工具） | `pi05/chunk_selector/` 与 `scripts/chunk_selector/` |
| 推理与真机 | `scripts/deployment/cartesian.py` | `pi05/deployment/` 调用同一 Cartesian runner |
| 离线诊断 | `scripts/diagnostics/` | `pi05/diagnostics/` |

`scripts/deployment/cartesian.py` 是唯一的实机部署入口：它会按 checkpoint 自动加载 ARP 或
π0.5。π0.5 的 `run_required_only.sh` 和 `run_full_then_truncate.sh` 记录每周期时延及人工
标注的 episode 成功结果，`summarize_trials.py` 汇总两条路径的成功率与推理时间。

当前 adaptive selector 已接入 π0.5 和 MVT ARP 部署。MVT ARP 使用冻结的点云视觉
特征训练 Transformer selector，支持基于训练轨迹的 spatial rule soft label；部署复用
一次 MVT 编码并按概率期望选择整数执行步数，目前支持 full-then-truncate。
标注、节点视频帧可视化、训练与推理命令见 [Chunk selector](../doc/Chunk_selector.md)。

空间 ARP 的相机标定工具位于 `scripts/calibration/`。历史 PushBox 与 MimicGen Threading
仿真脚本已移除。

点云 ARP 的 PushT 式稀疏计划监督使用 `threading_new_1_mvt_planarp` 配置，训练和推理
均按“稀疏计划 → 密集动作”顺序运行，详见 [PlanARP](doc/planarp.md)。

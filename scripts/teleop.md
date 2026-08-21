## 遥操作录制训练数据

目标：采集 **100 条** 成功演示，用于 ARP 模仿学习。

```bash
python scripts/teleop.py --out data/demos
python scripts/teleop.py --out data/demos --horizon 3000   # 自定义超时步数（默认 2000）
python scripts/teleop.py --debug --out data/demos          # 排查渲染问题
```

数据默认保存到 `data/demos/ep_NNNN.npz`。

### 操作说明

**机械臂（OSC 末端位姿）**


| 按键     | 作用                       |
| ---------- | ---------------------------- |
| W / S    | 沿桌面 X 轴后退 / 前进     |
| A / D    | 沿桌面 Y 轴左 / 右         |
| Q / E    | 末端上 / 下（Z）           |
| Z / C    | 夹爪绕垂直轴旋转（yaw ±） |
| ↑↓←→ | 同 W/S、A/D                |
| Space    | 夹爪开 / 关（关闭后末端自动上移 3 cm） |
| F11      | 全屏 / 窗口切换            |

**视角（左侧画面区）**


| 操作            | 作用                                                |
| ----------------- | ----------------------------------------------------- |
| 鼠标左键拖拽    | 旋转视角（orbit）                                   |
| 鼠标右键拖拽    | 平移视角                                            |
| 滚轮 /`+` `-`   | 拉近 / 拉远                                         |
| Tab / Shift+Tab | 下一个 / 上一个视角                                 |
| `1`–`4`        | `birdview` / `agentview` / `frontview` / `sideview` |
| `5`             | 回到 orbit 自由视角                                 |

**录制**


| 按键  | 作用                              |
| ------- | ----------------------------------- |
| R     | 开始 / 停止录制当前 episode       |
| Enter | 保存当前 episode 到磁盘并重置环境 |
| N     | 丢弃当前 episode 并重置           |
| Esc   | 退出（未保存的缓冲数据丢弃）      |

每条 episode 保存为 `<out>/ep_NNNN.npz`，主要字段：


| 键                | 形状           | 说明                                                   |
| ------------------- | ---------------- | -------------------------------------------------------- |
| `actions`         | (T, 7)         | OSC_POSE：`[dx, dy, dz, droll, dpitch, dyaw, gripper]` |
| `rewards`         | (T,)           | 逐步奖励                                               |
| `dones`           | (T,)           | 是否终止                                               |
| `obs/birdview`    | (T, 96, 96, 3) | birdview RGB，供 ARP 训练                              |
| `obs/agent_state` | (T, 9)         | 机器人关节状态：7 关节位置 + 2 夹爪 qpos               |
| `obs/box_to_goal`  | (T, 2)         | 积木到目标的相对位移                                   |
| `obs/*`           | (T, …)        | robosuite 本体观测（如`robot0_eef_pos` 等）            |

单局默认最多 **2000** 步（约 100 s @20 Hz），超时记为失败。
# 机器人轨迹生成的 Catmull-Rom 样条曲线技术规范

本文档提供了用于生成平滑机器人运动轨迹的向心 Catmull-Rom 样条（Centripetal Catmull-Rom Spline）的数学规范与实现逻辑。

---

## 1. 核心数学公式

计算单段 Catmull-Rom 样条曲线需要 **4 个连续的控制点**：$P_{i-1}$, $P_i$, $P_{i+1}$, $P_{i+2}$。算法在归一化参数 $t \in [0, 1]$ 的控制下，插值生成 $P_i$ 到 $P_{i+1}$ 之间的轨迹曲线。

### 向心参数化 ($\alpha = 0.5$)
为防止机器人在运行轨迹中出现尖角（Cusps）或自交（Self-intersections），必须使用 $\alpha = 0.5$ 的向心参数化来计算控制点之间的时间间隔 $\Delta t$：

$$\Delta t_k = ||P_{k+1} - P_k||^\alpha \quad \text{其中 } k \in \{i-1, i, i+1\}$$

节点时间序列（Knot Sequences）定义如下：
* $t_0 = 0$
* $t_1 = t_0 + \Delta t_{i-1}$
* $t_2 = t_1 + \Delta t_{i}$
* $t_3 = t_2 + \Delta t_{i+1}$

将全局插值参数 $t$ 映射到当前特定曲线片段的参数 $u$ 为：
$$t = t_1 + u \cdot (t_2 - t_1), \quad u \in [0, 1]$$

### 标准矩阵方程（均匀近似）
对于控制点间距均匀的片段，点 $P(u)$ 的标准矩阵计算公式为：

$$P(u) = \frac{1}{2} \begin{bmatrix} 1 & u & u^2 & u^3 \end{bmatrix} \begin{bmatrix} 0 & 2 & 0 & 0 \\ -1 & 0 & 1 & 0 \\ 2 & -5 & 4 & -1 \\ -1 & 3 & -3 & 1 \end{bmatrix} \begin{bmatrix} P_{i-1} \\ P_i \\ P_{i+1} \\ P_{i+2} \end{bmatrix}$$

---

## 2. 多点/无限点轨迹的滑动窗口逻辑

要处理包含 $N$ 个控制点的任意长序列 $[P_1, P_2, \dots, P_N]$，应使用窗口大小为 4、步长（Stride）为 1 的滑动窗口机制：

* **第 1 段**（$P_1$ 到 $P_2$ 之间）：使用 $[P_{\text{ext\_start}}, P_1, P_2, P_3]$ 进行计算
* **第 $i$ 段**（$P_i$ 到 $P_{i+1}$ 之间）：使用 $[P_{i-1}, P_i, P_{i+1}, P_{i+2}]$ 进行计算
* **第 $N-1$ 段**（$P_{N-1}$ 到 $P_N$ 之间）：使用 $[P_{N-2}, P_{N-1}, P_N, P_{\text{ext\_end}}]$ 进行计算

### 边界条件处理（外推法）
由于首尾两段曲线缺乏足够的邻近控制点，必须人工外推合成虚拟控制点：
* **起点外推**：$P_{\text{ext\_start}} = P_1 + (P_1 - P_2)$
* **终点外推**：$P_{\text{ext\_end}} = P_N + (P_N - P_{N-1})$

---

## 3. 数据增强（多度化生成）实现要求

1. **输入数据**：形状为 `(N, D)` 的张量（Tensor），表示 $D$ 维空间中的 $N$ 个控制点（如 $X,Y,Z$ 笛卡尔坐标或机器人关节角度）。
2. **扰动逻辑（多样性）**：仅对中间控制点 $P_2 \dots P_{N-1}$ 施加高斯噪声 $\epsilon \sim \mathcal{N}(0, \sigma^2)$。保持起点 $P_1$ 和终点 $P_N$ 固定，以确保任务的初始状态和目标状态约束不变。
3. **输出数据**：形状为 `(M, D)` 的稠密轨迹张量，其中 $M$ 为根据分辨率参数插值后的总步数。

---

## 4. 数据生成 UI 操作指南（`scripts/datagen.py`）

本脚本提供一个 pygame 图形界面：用鼠标点选关键路径点 → 实时拟合 Catmull-Rom 样条 → 碰撞检测 → 在模拟器中执行并录制为 **LeRobot v3** 数据集。完全独立于 `teleop.py`。

### 4.1 启动

```bash
# 启动 UI（需要显示器 + GPU EGL）
python scripts/datagen.py --out data/datagen --repo-id pushbox/datagen

# 仅做无界面的数学校验（样条 / 碰撞 / schema，不需要显示器和模拟器）
python scripts/datagen.py --validate
```

常用参数：

| 参数                  | 默认值            | 说明                          |
| --------------------- | ----------------- | ----------------------------- |
| `--out`               | `data/datagen`    | LeRobot 数据集输出目录        |
| `--repo-id`           | `pushbox/datagen` | LeRobot 数据集 repo_id        |
| `--samples-per-seg`   | `24`              | 每段控制点之间的样条插值点数  |
| `--horizon`           | `2000`            | 单条 episode 的最大仿真步数   |

### 4.2 界面布局

```
┌───────────────────────────┬──────────────┐
│  左侧：俯视示意画布 600×600 │ top45 视图    │
│   - 橙/黄填充：障碍物       ├──────────────┤
│   - 蓝色框：积木初始区域    │ sideview 视图 │
│   - 绿色圆：目标区域        │（实时参考）   │
│   - 白色圆点：关键点（≤5）  │              │
│   - 样条曲线：绿=可行/红=碰撞│              │
├───────────────────────────┴──────────────┤
│  状态栏：当前提示 + waypoints: N/5  saved: K │
└──────────────────────────────────────────┘
```

左侧画布是世界坐标的**俯视精确映射**（world ↔ 像素一一对应），鼠标点击位置即为桌面上的 `(x, y)` 目标点。右侧两个面板是模拟器 `top45` / `sideview` 相机的实时画面，仅作视觉参考，**不可点击**。

### 4.3 操作按键

| 操作              | 功能                                                   |
| ----------------- | ------------------------------------------------------ |
| 鼠标左键（画布内）| 在空白处添加关键点（**最多 5 个**）；点在已有点上则拖拽 |
| 鼠标右键（画布内）| 删除最近的关键点                                       |
| `N`               | 清空所有关键点                                          |
| `Enter`           | 若路径无碰撞，则在模拟器中执行并保存为一条 episode      |
| `Esc`             | 退出程序（**会 finalize 数据集**，写入 meta/统计信息）  |

### 4.4 标准工作流程

1. **画路径**：在左侧画布依次左键点击，添加 2~5 个关键点。脚本会实时拟合样条曲线。
   - 曲线为**绿色** = 路径未碰撞，可录制；
   - 曲线为**红色** = 路径与障碍物相交或越界，**无法录制**（按 Enter 会被拒绝）。
   - 需要微调时，直接拖拽某个点；右键删除单个点；`N` 清空重画。
   - 提示：障碍物之间的通道很窄（约 8 cm），叠加 3 cm 安全半径后自由空间更小，中段关键点需贴近通道中心（x ≈ 0 附近）才能保持绿色。

2. **保存当前 episode（如何保存）**：确认曲线为绿色后按 **`Enter`**。脚本会：
   - 重置环境并自动抓取积木（`init_with_grip`，抓取步数不计入 horizon）；
   - 用比例控制器驱动末端执行器沿稠密样条点运动，z 高度与夹爪状态保持不变；
   - 逐帧录制 `top45`/`sideview` 视频帧 + `state`/`box_pos`/`action`；
   - 调用 `writer.save_episode()` 把这条轨迹追加进 LeRobot 数据集。
   - 状态栏显示 `saved ep <idx> (<steps> steps) success=<bool>`，`saved:` 计数 +1。

3. **进入下一个 episode（如何进入下一条）**：每次按 `Enter` 录制时都会自动 `init_with_grip` 重置环境，**积木初始位置在 `BOX_INIT` 区域内重新随机**。因此录制完一条后：
   - 按 **`N`** 清空当前关键点（或拖动已有点改成新路径）；
   - 重新点选关键点画出新路径；
   - 再次按 **`Enter`** 即录制下一条 episode。
   - 如此循环采集多条，`saved:` 计数持续累加。

4. **结束采集**：按 **`Esc`** 退出。此时才会执行 `writer.finalize()`，写入 `meta/episodes`、`meta/stats.json` 并更新 `meta/info.json` 的总数。**务必通过 `Esc` 正常退出**，否则数据集元信息可能不完整。

### 4.5 输出格式（LeRobot v3）

与 `teleop.py` 输出完全一致，可直接进入训练管线：

```
data/datagen/
├── meta/info.json, tasks.parquet, stats.json, episodes/chunk-000/*.parquet
├── data/chunk-000/file-000.parquet                       # state / action / box_pos
└── videos/observation.images.{top45,sideview}/chunk-000/*.mp4
```

每帧字段：`observation.images.top45`、`observation.images.sideview`（uint8 96×96×3 视频）、`observation.state`（9D）、`observation.box_pos`（2D）、`action`（7D OSC_POSE）。

---

## 5. 数据集校验（`scripts/validate_demos.py`）

`validate_demos.py` 用于校验 LeRobot v3 数据集是否符合 ARP 训练管线的要求，检查数据集结构、特征完整性和逐 episode 数据合理性。

### 5.1 启动

```bash
# 校验默认目录 data/demos
python scripts/validate_demos.py

# 校验指定目录
python scripts/validate_demos.py --dir data/datagen

# 安静模式，只打印失败项
python scripts/validate_demos.py --dir data/demos --quiet
#生成视频
python scripts/replay_demo.py data/datagen --episode 0 --out replay.mp4
```

### 5.2 参数

| 参数      | 默认值       | 说明                        |
| --------- | ------------ | --------------------------- |
| `--dir`   | `data/demos` | LeRobot 数据集目录          |
| `--quiet` | `false`      | 只输出 FAIL 项，不打印 PASS |

### 5.3 校验项目

| 阶段        | 检查项                                                        |
| ----------- | ------------------------------------------------------------- |
| 结构        | `meta/info.json` 是否存在，`fps` / `features` 是否可读        |
| 特征        | 必须包含 `observation.images.top45`、`observation.images.sideview`、`observation.state`、`action`；各特征 shape / dtype 是否正确 |
| 逐 episode  | state/action 维度（`state` 9D，`action` 7D）                 |
|             | 关节角度是否在 Panda 限位内（含 0.05 rad 容差）              |
|             | 夹爪 qpos 是否在 [-0.05, 0.05] 内                            |
|             | action 前 6 维绝对值是否 ≤ 1.05                               |
|             | 图像数值是否在 [0, 1] 归一化范围内                           |
|             | state / action 帧数是否一致                                   |

### 5.4 输出样例

```
Validating LeRobot dataset at data/datagen...

  OK LeRobot dataset found at: data/datagen
  OK info.json: fps=20, features=['action', 'observation.images.sideview', 'observation.images.top45', 'observation.state']
  OK All expected features present
  OK observation.state: shape=[9], dtype=float32
  OK action: shape=[7], dtype=float32
  OK observation.images.top45: video feature OK
  OK observation.images.sideview: video feature OK
  OK Total episodes: 12
  Episode 0: PASS  frames=187
  Episode 1: PASS  frames=203
  ...

--- Summary ---
  PASS: 12  WARN: 0  FAIL: 0  TOTAL: 12
  Status: OK
```

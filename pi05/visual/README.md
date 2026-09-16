# π0.5 固定任务区域裁剪实验

第一版保持两路相机和原 SigLIP 编码器，用 cam1/cam3 各自的固定 ROI 替代整幅图。
输入处理为 **原始分辨率 RGB → 固定裁剪 → 等比例缩放 → 黑边补齐至 224×224**。
不新增视觉分支，也不拼接全局与局部图。全图基线与裁剪实验应使用相同 episodes、
split、训练预算、动作频率与执行步数。由于原转换器直接拉伸整图，新流程采用
letterbox，若需严格分离“裁剪”和“保持比例”的收益，可另做全幅 ROI 的 letterbox 基线。

所有新增实现位于 `pi05/`；原始转换器和机器人 runner 通过专用入口复用，
只在入口进程内适配视频读取和观测处理函数，不修改共享文件。

## 1. 选择区域并查看输入

命令在 teleoperation 仓库根目录执行，使用安装了 numpy、OpenCV 的 Python。
数据构建、训练与实机还需要现有 LeRobot / Panda 运行环境。`--select` 需要有桌面
的 `opencv-python`；无桌面机器可用 `opencv-python-headless` 配合显式坐标。

下面的 `RAW_EPISODE` 是实际原始 episode 目录，必须包含 `cam1.mp4` 和 `cam3.mp4`，
不要指向已缩到 224 的 LeRobot 视频。frame 是各视频的原始帧号；预览不承担跨相机
时间对齐，正式构建沿用原转换器的时间戳对齐。

```bash
RAW_EPISODE=/absolute/path/to/raw/episode_001
python threading_real/pi05/visual/preview.py \
  --episode "$RAW_EPISODE" --frames 0 100 200 \
  --select \
  --save-config threading_real/pi05/visual/threading_roi.json \
  --output threading_real/pi05/outputs/crop_preview/episode_001
```

在第一帧上依次为两相机拖框，Enter 确认。取消或空框报错，不会保存空配置。
选择能够看清目标的首帧，例如将 `--frames` 改为 `100 200 300`。
覆盖针、方块及其接近/穿入范围；检查起点、中段、终点以及不同 episode 的极端位置。
输出包括原图 `raw`、绿色框 `roi`、全幅 letterbox `full224`、实际输入 `input224`。

无 GUI 时可传 `--cam1-roi X0 Y0 X1 Y1 --cam3-roi X0 Y0 X1 Y1` 代替 `--select`。
坐标为原图像素，右下边界不包含在 ROI 内。工具自动记录实际原图宽高。
不提供未经检查的默认框。配置和预览目录均拒绝覆盖。

用同一个框检查其他 episode：

```bash
python threading_real/pi05/visual/preview.py \
  --episode /absolute/path/to/raw/episode_020 --frames 0 100 200 \
  --config threading_real/pi05/visual/threading_roi.json \
  --output threading_real/pi05/outputs/crop_preview/episode_020
```

如需全图 letterbox 对照，使用 `[0, 0, width, height]` 作为每个相机的 ROI。
改变相机位姿、镜头或采集视野后，应重新检查配置；只保持像素宽高相同并不代表几何相同。

## 2. 从原始数据重建

```bash
python threading_real/pi05/training/build_dataset.py \
  --raw-root /absolute/path/to/threading_new_1 \
  --raw-root /absolute/path/to/threading_new_2 \
  --output threading_real/pi05/data/threading_crop_15hz \
  --repo-id threading_real/threading_crop_15hz \
  --visual-config threading_real/pi05/visual/threading_roi.json \
  --expected-episodes 80 --image-size 224 \
  --state-representation tcp_pose_6d
```

保留连续 15 Hz 时间线；不传 `--drop-zero-actions`。裁剪在原始视频解码后、第一次
缩放前发生，不改变机器人状态、动作或时间戳匹配。cam1 对应
`observation.images.exterior_image_2_right`，cam3 对应
`observation.images.exterior_image_1_left`。

最终数据的 `meta/visual_preprocessing.json` 保存完整 ROI 和输入尺寸。
训练直接读取已裁剪视频，不重复裁剪。原始分辨率不符会报错。
构建成功后清理中间数据；失败时保留中间数据供检查，输出路径仍禁止覆盖。

## 3. 训练并保存配置

```bash
DATASET_ROOT="$PWD/threading_real/pi05/data/threading_crop_15hz" \
REPO_ID=threading_real/threading_crop_15hz \
OUTPUT_DIR="$PWD/threading_real/pi05/outputs/threading_crop_v1" \
JOB_NAME=threading_crop_v1 \
  bash threading_real/pi05/training/train.sh
```

GPU、batch、steps 等沿用现有脚本，按训练机设置环境变量。加 `PRINT_CONFIG_ONLY=true`
可以检查启动参数。训练 wrapper 将数据中的 `visual_preprocessing.json` 复制到每个
`pretrained_model` 目录，与权重绑定。直接调用 wrapper 时也可通过 `--dataset.root`
识别元数据；请使用该 wrapper 保存权重。离线推理仍使用已裁剪数据和原诊断入口。

## 4. 实机使用专用入口

```bash
python threading_real/pi05/deployment/cropped.py \
  threading_real/pi05/outputs/threading_crop_v1/checkpoints/last/pretrained_model \
  --policy-kind pi05 \
  --task "insert the grasped block through the needle" \
  --server-url http://10.157.175.22:8008/RPC2 \
  --server-ip 10.157.175.22 --udp-ip 10.157.175.211 \
  --policy-hz 15 --execute-steps 1 --max-cycles 20
```

以上为 dry-run，沿用 runner 的实际执行开关。专用入口读取 checkpoint 的 ROI，
按配置请求原始相机分辨率，再调用与训练构建相同的裁剪函数；相机必须支持该分辨率
下的 30 FPS。现有 rig 要求两路相机宽高一致。缺少 ROI 元数据、错误输入大小或
`--pre-resize-image-size` 都会拒绝运行。

**裁剪权重必须使用 `deployment/cropped.py`。** 共享的普通 runner 不读取这些元数据，
直接调用它会输入全图，导致训练与部署不一致。旧的 selector 使用全图特征训练，
本轮裁剪入口暂不接受 `--chunk-selector`，先用固定执行长度完成对照。

## 验证

```bash
python -m pytest threading_real/pi05/visual/test_crop.py -q
```

测试覆盖像素裁剪、letterbox、非法 ROI/分辨率拒绝、重复解码不重复裁剪、BGR 数据
构建与 RGB 实机路径一致、合成视频预览与元数据、流水线配置传递及失败保留中间数据。
不需要 GPU、LeRobot 或机器人。像素一致性测试发生在视频编码之前；训练视频再次编码
会引入压缩差异，因此不宣称解码后的训练图与在线图逐像素相同。

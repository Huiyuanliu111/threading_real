# Cartesian 部署视频

`scripts/deployment/cartesian.py` 默认自动录制 cam1（sideview）和 cam3（frontview），
包括 dry-run。即使策略只使用 cam1，也会开启 cam3 供录制使用。

每次运行创建独立时间戳目录，每个 episode 从初始抓取完成后的相机预热开始录制，
到 episode 停止执行后结束；场景重置和结果填写期间不录制。
正常结束、Ctrl+C、SIGTERM 或 Python 异常退出时会关闭并保存视频。

默认路径：

```text
threading_real/outputs/deployment_videos/<时间戳>/episode_0001/cam1.mp4
threading_real/outputs/deployment_videos/<时间戳>/episode_0001/cam3.mp4
```

视频使用相机原始 640×480 RGB 画面，MP4/mp4v 编码，目标帧率 30 FPS。
后台线程持续采集并编码，模型从同一采集线程获取最新画面，因此推理等待期间仍会录制。
实际采集帧率受相机、USB 和编码性能影响。

现有部署命令无需修改。用 `--video-output-dir /path/to/videos` 更改保存目录，
用 `--no-record-video` 关闭录制并恢复只开启策略所需相机的行为。

## Maze 部署

`maze_real/scripts/deployment/cartesian.py` 同样默认录制 cam1 和 cam3，使用相同的
`--video-output-dir`、`--no-record-video` 参数。默认保存到
`maze_real/outputs/deployment_videos/<时间戳>/episode_0001/`，每个 episode 生成
`cam1.mp4` 和 `cam3.mp4`。初始抓取完成后、TrackC 启动前开始录制，停止运动后结束，
不录制结果填写和 guide 手动复位阶段。cam1 序列号可通过 `--sideview-serial` 指定。
启动时打印保存目录；修改脚本后需重新启动部署进程才能生效。

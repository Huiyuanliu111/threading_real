"""LeRobot v3 dataset writer / reader (byte-compatible with official lerobot package).

Produces datasets readable by LeRobotDataset and the dataset visualizer:
- Tabular data: Apache Parquet (fixed_size_list schema, int64 indices)
- Video data: MP4 via PyAV (h264)
- Metadata: meta/info.json, meta/episodes/, meta/tasks.parquet, meta/stats.json

Usage (recording):
    writer = LeRobotWriter.create("pushbox/demos", fps=20, features=..., root="data/demos")
    for step in range(n_steps):
        writer.add_frame(frame_dict)  # must include 'task' key
    writer.save_episode()
    writer.finalize()

Usage (loading):
    episodes = load_lerobot_episodes("data/demos")
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_BOX_POS_ZEROS = np.zeros(2, dtype=np.float32)


class LeRobotWriter:
    """Writes a LeRobot v3 dataset incrementally, byte-compatible with lerobot."""

    def __init__(
        self,
        repo_id: str,
        fps: int,
        features: dict,
        root: str,
        data_chunk: int = 0,
        _use_videos: bool = True,  # noqa: FBT001, FBT002
    ):
        self.repo_id = repo_id
        self.fps = fps
        self.features = features
        self.root = Path(root)
        self.data_chunk = data_chunk
        self._use_videos = _use_videos
        self._episode_buffer: list[dict] = []
        self._episodes_meta: list[dict] = []
        self._global_frame_offset = 0
        self._ep_counter = 0
        self._data_writer = None
        self._video_muxers: dict[str, object] = {}  # pyav muxers
        self._video_streams: dict[str, object] = {}
        self._parquet_schema = None
        self._closed = False

    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        features: dict,
        root: str,
        use_videos: bool = True,  # noqa: FBT001, FBT002
        data_chunk: int = 0,
    ) -> LeRobotWriter:
        root_p = Path(root)
        _mkdir(root_p / "meta" / "episodes" / f"chunk-{data_chunk:03d}")
        _mkdir(root_p / "data" / f"chunk-{data_chunk:03d}")

        # Build info.json matching official lerobot format
        info_features = {}
        for feat_name, feat_info in features.items():
            entry = {"dtype": feat_info["dtype"], "shape": list(feat_info["shape"])}
            if "names" in feat_info:
                entry["names"] = feat_info["names"]
            info_features[feat_name] = entry

        info = {
            "codebase_version": "v3.0",
            "fps": fps,
            "robot_type": "panda",
            "total_episodes": 0,
            "total_frames": 0,
            "total_tasks": 1,
            "features": info_features,
        }
        (root_p / "meta" / "info.json").write_text(json.dumps(info, indent=2))

        # Create tasks.parquet
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq  # noqa: F811
            task_table = pa.table({
                "task_index": pa.array([0], type=pa.int64()),
                "task": pa.array(["push_box"], type=pa.string()),
            })
            pq.write_table(task_table, str(root_p / "meta" / "tasks.parquet"))
        except ImportError:
            pass

        if use_videos:
            for feat_name, feat_info in features.items():
                if feat_info.get("dtype") == "video":
                    _mkdir(root_p / "videos" / feat_name / f"chunk-{data_chunk:03d}")

        return cls(repo_id=repo_id, fps=fps, features=features, root=root,
                   data_chunk=data_chunk, _use_videos=use_videos)

    @classmethod
    def resume_or_create(
        cls,
        repo_id: str,
        fps: int,
        features: dict,
        root: str,
        use_videos: bool = True,  # noqa: FBT001, FBT002
    ) -> LeRobotWriter:
        """Create a new dataset or resume appending to an existing one.

        If meta/info.json exists and has total_episodes > 0, a new data chunk
        is created with episode / frame indices continuing from where the
        previous session left off.  Existing data is never overwritten.
        """
        root_p = Path(root)
        info_path = root_p / "meta" / "info.json"

        if info_path.exists():
            try:
                info = json.loads(info_path.read_text())
                existing_eps = int(info.get("total_episodes", 0))
                existing_frames = int(info.get("total_frames", 0))
            except (json.JSONDecodeError, KeyError, TypeError):
                existing_eps = 0
                existing_frames = 0

            if existing_eps > 0:
                # Find next available chunk number
                data_dir = root_p / "data"
                next_chunk = 0
                if data_dir.exists():
                    chunks = sorted(data_dir.glob("chunk-*"))
                    if chunks:
                        next_chunk = max(int(c.name.split("-")[-1]) for c in chunks) + 1

                # Create dirs for the new chunk only
                _mkdir(root_p / "meta" / "episodes" / f"chunk-{next_chunk:03d}")
                _mkdir(root_p / "data" / f"chunk-{next_chunk:03d}")

                if use_videos:
                    for feat_name, feat_info in features.items():
                        if feat_info.get("dtype") == "video":
                            _mkdir(root_p / "videos" / feat_name / f"chunk-{next_chunk:03d}")

                # Ensure tasks.parquet exists (might have been deleted)
                tasks_path = root_p / "meta" / "tasks.parquet"
                if not tasks_path.exists():
                    try:
                        import pyarrow as pa
                        task_table = pa.table({
                            "task_index": pa.array([0], type=pa.int64()),
                            "task": pa.array(["push_box"], type=pa.string()),
                        })
                        task_table.write_table(task_table, str(tasks_path))
                    except ImportError:
                        pass

                writer = cls(repo_id=repo_id, fps=fps, features=features, root=root,
                             data_chunk=next_chunk, _use_videos=use_videos)
                writer._global_frame_offset = existing_frames
                writer._ep_counter = existing_eps
                print(f"[lerobot_io] resuming dataset at chunk {next_chunk:03d}, "
                      f"ep_offset={existing_eps}, frame_offset={existing_frames}")
                return writer

        # Fall through to fresh create
        return cls.create(repo_id, fps, features, root, use_videos)

    def add_frame(self, frame: dict) -> None:
        if self._closed:
            raise RuntimeError("Writer is already finalized")
        self._episode_buffer.append(frame)

    def save_episode(self) -> int:
        if not self._episode_buffer:
            return self._ep_counter
        n_frames = len(self._episode_buffer)
        ep_idx = self._ep_counter
        self._ep_counter += 1
        self._write_data_chunk(ep_idx, self._episode_buffer)
        self._write_video_chunk(self._episode_buffer)
        self._episodes_meta.append({
            "episode_index": ep_idx,
            "length": n_frames,
            "dataset_from_index": self._global_frame_offset - n_frames,
            "dataset_to_index": self._global_frame_offset,
        })
        self._episode_buffer.clear()
        return ep_idx

    def finalize(self) -> None:
        if self._episode_buffer:
            self.save_episode()
        self._close_writers()
        self._flush_meta()
        self._closed = True

    def _close_writers(self) -> None:
        if self._data_writer is not None:
            self._data_writer.close()
            self._data_writer = None
        # Close video writers (flush + finalize)
        import av
        for feat_name, muxer in self._video_muxers.items():
            stream = self._video_streams[feat_name]
            for packet in stream.encode():
                muxer.mux(packet)
            muxer.close()
        self._video_muxers.clear()
        self._video_streams.clear()

    def _flush_meta(self) -> None:
        self._write_episodes_meta()
        self._update_info()

    # ── Parquet ──────────────────────────────────────────────────────────

    def _get_data_path(self) -> Path:
        return self.root / "data" / f"chunk-{self.data_chunk:03d}" / "file-000.parquet"

    def _get_video_path(self, feature_name: str) -> Path:
        return self.root / "videos" / feature_name / f"chunk-{self.data_chunk:03d}" / "file-000.mp4"

    def _ensure_data_writer(self):
        if self._data_writer is not None:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq  # noqa: F811

        # Match official lerobot schema: index, int64, fixed_size_list<float32>
        fields = [
            pa.field("episode_index", pa.int64()),
            pa.field("frame_index", pa.int64()),
            pa.field("index", pa.int64()),
            pa.field("timestamp", pa.float32()),
            pa.field("task_index", pa.int64()),
        ]
        for feat_name, feat_info in self.features.items():
            dtype = feat_info.get("dtype", "float32")
            if dtype == "video":
                continue
            shape = feat_info.get("shape", (1,))
            n = shape[-1] if isinstance(shape, (list, tuple)) else shape
            fields.append(pa.field(
                feat_name,
                pa.list_(pa.field("element", pa.float32()), n),
            ))

        # Add rich metadata matching lerobot
        meta = {
            "info": {
                "features": {
                    k: {"dtype": v.get("dtype"), "shape": v.get("shape")}
                    for k, v in self.features.items()
                }
            }
        }
        self._parquet_schema = pa.schema(fields).with_metadata(
            {"huggingface": json.dumps(meta)}
        )
        self._data_writer = pq.ParquetWriter(
            str(self._get_data_path()),
            self._parquet_schema,
            compression="snappy",
        )

    def _write_data_chunk(self, ep_idx: int, frames: list[dict]) -> None:
        self._ensure_data_writer()
        import pyarrow as pa

        n = len(frames)
        start = self._global_frame_offset

        columns: dict[str, pa.Array] = {
            "episode_index": pa.array([ep_idx] * n, type=pa.int64()),
            "frame_index": pa.array(list(range(n)), type=pa.int64()),
            "index": pa.array([start + i for i in range(n)], type=pa.int64()),
            "timestamp": pa.array(
                [float(start + i) / self.fps for i in range(n)], type=pa.float32()
            ),
            "task_index": pa.array([0] * n, type=pa.int64()),
        }
        for feat_name, feat_info in self.features.items():
            if feat_info.get("dtype") == "video":
                continue
            values = np.stack([f[feat_name] for f in frames]).astype(np.float32)
            columns[feat_name] = pa.array(values.tolist(), type=pa.list_(pa.float32()))

        batch = pa.RecordBatch.from_pydict(columns, schema=self._parquet_schema)
        self._data_writer.write_batch(batch)
        self._global_frame_offset += n

    # ── Video (PyAV MP4) ─────────────────────────────────────────────────

    def _ensure_video_muxer(self, feat_name: str, shape: tuple):
        if feat_name in self._video_muxers:
            return
        import av
        h, w = shape[0], shape[1]
        container = av.open(str(self._get_video_path(feat_name)), "w")
        stream = container.add_stream("h264", rate=self.fps)
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuv420p"
        stream.options["preset"] = "ultrafast"
        stream.options["crf"] = "30"
        self._video_muxers[feat_name] = container
        self._video_streams[feat_name] = stream

    def _write_video_chunk(self, frames: list[dict]) -> None:
        if not self._use_videos:
            return
        import av
        for feat_name, feat_info in self.features.items():
            if feat_info.get("dtype") != "video":
                continue
            shape = feat_info.get("shape", (96, 96, 3))
            self._ensure_video_muxer(feat_name, shape)
            container = self._video_muxers[feat_name]
            stream = self._video_streams[feat_name]
            for frame_dict in frames:
                img = frame_dict[feat_name]  # (H, W, 3) uint8
                video_frame = av.VideoFrame.from_ndarray(img, format="rgb24")
                for packet in stream.encode(video_frame):
                    container.mux(packet)

    # ── Metadata ─────────────────────────────────────────────────────────

    def _write_episodes_meta(self) -> None:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq  # noqa: F811
        except ImportError:
            return
        if not self._episodes_meta:
            return
        table = pa.table({
            "episode_index": pa.array([m["episode_index"] for m in self._episodes_meta], type=pa.int64()),
            "length": pa.array([m["length"] for m in self._episodes_meta], type=pa.int64()),
            "dataset_from_index": pa.array([m["dataset_from_index"] for m in self._episodes_meta], type=pa.int64()),
            "dataset_to_index": pa.array([m["dataset_to_index"] for m in self._episodes_meta], type=pa.int64()),
        })
        ep_path = self.root / "meta" / "episodes" / f"chunk-{self.data_chunk:03d}" / "file-000.parquet"
        pq.write_table(table, str(ep_path))

    def _write_stats(self) -> None:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq  # noqa: F811
        except ImportError:
            return

        data_dir = self.root / "data"
        if not data_dir.exists():
            return

        all_tables = []
        for chunk_dir in sorted(data_dir.glob("chunk-*")):
            for pf in sorted(chunk_dir.glob("*.parquet")):
                if pf.exists() and pf.stat().st_size > 0:
                    all_tables.append(pq.read_table(str(pf)))

        if not all_tables:
            return
        table = pa.concat_tables(all_tables) if len(all_tables) > 1 else all_tables[0]
        if len(table) == 0:
            return

        stats = {}
        for feat_name, feat_info in self.features.items():
            if feat_info.get("dtype") == "video":
                continue
            col = table.column(feat_name).to_pylist()
            arr = np.array(col, dtype=np.float32)
            stats[feat_name] = {
                "mean": arr.mean(axis=0).tolist(),
                "std": arr.std(axis=0).tolist(),
                "min": arr.min(axis=0).tolist(),
                "max": arr.max(axis=0).tolist(),
            }
        (self.root / "meta" / "stats.json").write_text(json.dumps(stats, indent=2))

    def _update_info(self) -> None:
        info_path = self.root / "meta" / "info.json"
        if not info_path.exists():
            return
        info = json.loads(info_path.read_text())

        # Compute true totals from all episode-metadata chunks
        ep_meta_dir = self.root / "meta" / "episodes"
        total_eps = 0
        total_frames = 0
        if ep_meta_dir.exists():
            try:
                import pyarrow.parquet as pq  # noqa: F811
            except ImportError:
                pass
            else:
                for chunk_dir in sorted(ep_meta_dir.glob("chunk-*")):
                    for pf in sorted(chunk_dir.glob("*.parquet")):
                        if pf.exists() and pf.stat().st_size > 0:
                            ep_table = pq.read_table(str(pf))
                            total_eps += len(ep_table)
                            if "length" in ep_table.column_names:
                                lengths = ep_table.column("length").to_pylist()
                                total_frames += int(sum(lengths))

        info["total_episodes"] = total_eps if total_eps > 0 else self._ep_counter
        info["total_frames"] = total_frames if total_frames > 0 else self._global_frame_offset
        info_path.write_text(json.dumps(info, indent=2))


# ── Loading ───────────────────────────────────────────────────────────────────


def load_lerobot_episodes(root: str | Path, max_frames_per_ep: int = 0) -> list[dict]:
    """Load all episodes from a LeRobot v3 dataset directory.

    If max_frames_per_ep > 0, each episode is evenly subsampled to at most that many frames
    to reduce memory usage.  Parquet data is read first (small), then video frames are
    decoded only at the selected indices.
    """
    import pyarrow.parquet as pq  # noqa: F811

    root_p = Path(root).expanduser()
    if not (root_p / "meta" / "info.json").exists():
        root_p = _find_dataset_root(root_p)
        if root_p is None:
            raise FileNotFoundError(f"LeRobot dataset not found at {root}")

    info = json.loads((root_p / "meta" / "info.json").read_text())
    features = info.get("features", {})

    # Read all data parquet files (multiple chunks) and concatenate
    import pyarrow as pa
    data_files = sorted((root_p / "data").rglob("*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No parquet in {root_p / 'data'}")
    tables = [pq.read_table(str(f)) for f in data_files]
    table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]

    # Read episode metadata (all chunks)
    ep_meta_dir = root_p / "meta" / "episodes"
    ep_files = sorted(ep_meta_dir.rglob("*.parquet"))
    if ep_files:
        ep_tables = [pq.read_table(str(f)) for f in ep_files]
        ep_table = pa.concat_tables(ep_tables) if len(ep_tables) > 1 else ep_tables[0]
        ep_indices = ep_table.column("episode_index").to_pylist()
        ep_lengths = ep_table.column("length").to_pylist()
        ep_from = ep_table.column("dataset_from_index").to_pylist()
        ep_to = ep_table.column("dataset_to_index").to_pylist()
    else:
        total_frames = len(table)
        ep_indices, ep_lengths = [0], [total_frames]
        ep_from, ep_to = [0], [total_frames]

    state_col = _col_to_numpy(table.column("observation.state"))
    action_col = _col_to_numpy(table.column("action"))
    box_col = None
    if "observation.box_pos" in table.column_names:
        box_col = _col_to_numpy(table.column("observation.box_pos"))

    # Build per-episode frame index maps BEFORE loading video (saves memory)
    ep_frame_indices: list[np.ndarray] = []
    for ep_i in range(len(ep_indices)):
        s, e = int(ep_from[ep_i]), int(ep_to[ep_i])
        all_idx = np.arange(s, e)
        if max_frames_per_ep and max_frames_per_ep > 0 and len(all_idx) > max_frames_per_ep:
            all_idx = np.linspace(s, e - 1, max_frames_per_ep, dtype=int)
        ep_frame_indices.append(all_idx)

    # Load video frames only at the needed indices (per-episode, skipping extras)
    global_idx_map: dict[int, int] = {}
    for ep_i, indices in enumerate(ep_frame_indices):
        for local_i, gidx in enumerate(indices):
            global_idx_map[int(gidx)] = (ep_i, local_i)

    video_readers: dict[str, dict] = {}
    videos_dir = root_p / "videos"
    if videos_dir.exists():
        import av
        for feat_name, feat_info in features.items():
            if feat_info.get("dtype") != "video":
                continue
            video_files = sorted((videos_dir / feat_name).rglob("*.mp4"))
            video_readers[feat_name] = {}  # ep_i -> np.ndarray (local_T, H, W, 3) uint8
            gframe = 0
            for vf in video_files:
                container = av.open(str(vf))
                for frame in container.decode(video=0):
                    if gframe in global_idx_map:
                        ep_i, local_i = global_idx_map[gframe]
                        img = frame.to_ndarray(format="rgb24")
                        if feat_name not in video_readers:
                            video_readers[feat_name] = {}
                        if ep_i not in video_readers[feat_name]:
                            n_needed = len(ep_frame_indices[ep_i])
                            video_readers[feat_name][ep_i] = np.zeros(
                                (n_needed, *img.shape), dtype=np.uint8
                            )
                        video_readers[feat_name][ep_i][local_i] = img
                    gframe += 1
                container.close()

    episodes = []
    for ep_i in range(len(ep_indices)):
        indices = ep_frame_indices[ep_i]
        n = len(indices)
        # slice parquet data at these indices
        state = state_col[indices]
        action = action_col[indices]
        box_pos = box_col[indices] if box_col is not None else np.tile(_BOX_POS_ZEROS, (n, 1))
        top45 = video_readers.get("observation.images.top45", {}).get(ep_i)
        sideview = video_readers.get("observation.images.sideview", {}).get(ep_i)
        if top45 is None:
            top45 = np.zeros((n, 3, 96, 96), dtype=np.float32)
        else:
            top45 = top45.astype(np.float32) / 255.0
            top45 = np.moveaxis(top45, -1, 1)  # (T,H,W,C) → (T,C,H,W)
        if sideview is None:
            sideview = np.zeros((n, 3, 96, 96), dtype=np.float32)
        else:
            sideview = sideview.astype(np.float32) / 255.0
            sideview = np.moveaxis(sideview, -1, 1)
        episodes.append({
            "top45": top45, "sideview": sideview,
            "state": state, "action": action, "box_pos": box_pos,
        })
    return episodes


# ── helpers ───────────────────────────────────────────────────────────────────


def _mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _col_to_numpy(col) -> np.ndarray:
    vals = col.to_pylist()
    return np.array(vals, dtype=np.float32)


def _read_video_frames_av(path: str) -> np.ndarray:
    """Read all frames from MP4 via PyAV. Returns (N, H, W, 3) uint8."""
    try:
        import av
        container = av.open(path)
        frames = []
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
        container.close()
        if not frames:
            return np.zeros((0, 96, 96, 3), dtype=np.uint8)
        return np.stack(frames)
    except Exception:
        return np.zeros((0, 96, 96, 3), dtype=np.uint8)


def _get_video_slice(video_readers: dict, feat_name: str, start: int, end: int,
                     img_size: int = 96) -> np.ndarray:
    n = end - start
    if n <= 0:
        return np.zeros((0, 3, img_size, img_size), dtype=np.float32)
    if feat_name not in video_readers:
        return np.zeros((n, 3, img_size, img_size), dtype=np.float32)
    frames = video_readers[feat_name][start:end].astype(np.float32) / 255.0
    if len(frames) == 0:
        return np.zeros((n, 3, img_size, img_size), dtype=np.float32)
    if len(frames) < n:
        # pad with zeros if video is shorter than expected
        padded = np.zeros((n, frames.shape[1], frames.shape[2], frames.shape[3]), dtype=np.float32)
        padded[:len(frames)] = frames
        return padded
    return np.moveaxis(frames, -1, 1)


def _find_dataset_root(start: Path) -> Path | None:
    for p in [start] + list(start.parents):
        if (p / "meta" / "info.json").exists():
            return p
    return None

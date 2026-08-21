"""Replay an episode from a LeRobot v3 dataset via the environment.

Usage:
    python scripts/replay_demo.py data/old_data/datagen --episode 57 --no-render
    python scripts/replay_demo.py data/old_data/datagen --episode 57 --size 128
    python scripts/replay_demo.py data/old_data/datagen --episode 57 --out replay.npz
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.pop("DISPLAY", None)
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/10_nvidia.json")

DEFAULT_SIZE = 256


def _load_single_episode_actions(dataset_dir: Path, ep_idx: int) -> np.ndarray:
    """Load only actions for a single episode (no video decoding)."""
    import pyarrow.parquet as pq
    import pyarrow as pa

    ep_meta_dir = dataset_dir / "meta" / "episodes"
    ep_files = sorted(ep_meta_dir.rglob("*.parquet"))
    if not ep_files:
        raise FileNotFoundError(f"No episode metadata in {ep_meta_dir}")
    ep_tables = [pq.read_table(str(f)) for f in ep_files]
    ep_table = pa.concat_tables(ep_tables) if len(ep_tables) > 1 else ep_tables[0]
    ep_indices = ep_table.column("episode_index").to_pylist()
    ep_from = ep_table.column("dataset_from_index").to_pylist()
    ep_to = ep_table.column("dataset_to_index").to_pylist()

    if ep_idx >= len(ep_indices):
        raise IndexError(f"Episode {ep_idx} out of range (total: {len(ep_indices)})")

    s, e = int(ep_from[ep_idx]), int(ep_to[ep_idx])
    n_frames = e - s
    print(f"Episode {ep_idx}: frames [{s}, {e}), {n_frames} actions")

    data_files = sorted((dataset_dir / "data").rglob("*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No data parquet in {dataset_dir / 'data'}")
    tables = [pq.read_table(str(f)) for f in data_files]
    table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
    actions = table.column("action").to_pylist()[s:e]
    return np.array(actions, dtype=np.float32)


def _bind_egl(env):
    try:
        rc = getattr(env.sim, "_render_context_offscreen", None)
        if rc is not None and hasattr(rc, "gl_ctx") and rc.gl_ctx is not None:
            rc.gl_ctx.make_current()
    except Exception:
        pass


def _render_safe(env, camera_name: str, size: int):
    _bind_egl(env)
    try:
        return env.sim.render(width=size, height=size, camera_name=camera_name)
    except Exception:
        return None


def capture_dual(env, size: int = DEFAULT_SIZE):
    bird = _render_safe(env, "birdview", size)
    if bird is None:
        return None
    front = _render_safe(env, "frontview", size)
    if front is None:
        return None
    bird = np.flipud(bird).astype(np.uint8)
    front = np.flipud(front).astype(np.uint8)
    return np.concatenate([bird, front], axis=1)


def main():
    parser = argparse.ArgumentParser(description="Replay an episode from LeRobot v3 dataset")
    parser.add_argument("dataset_dir", type=str, help="path to LeRobot dataset dir")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--out", type=str, default=None, help="output .npz path")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE, help="render resolution per camera")
    parser.add_argument("--no-render", action="store_true", help="physics-only replay")
    parser.add_argument("--skip", type=int, default=0, help="skip first N frames when rendering")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    actions = _load_single_episode_actions(dataset_dir, args.episode)
    n_frames = len(actions)
    print(f"Replaying episode {args.episode}: {n_frames} actions")

    from envs import PushBoxEnv

    env = PushBoxEnv(
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=not args.no_render,
        use_camera_obs=False,
        use_object_obs=True,
        reward_shaping=True,
        horizon=n_frames + 50,
        hard_reset=False,
        camera_names=["birdview", "frontview"] if not args.no_render else [],
        camera_heights=args.size,
        camera_widths=args.size,
    )

    obs, grip_steps = env.init_with_grip()
    print(f"  gripped box in {grip_steps} steps")

    out = args.out or f"replay_ep{args.episode:04d}.npz"
    do_render = not args.no_render
    frames = []  # collect in memory (no disk I/O, no ffmpeg during loop)

    for i in range(n_frames):
        if i == 0 or i % max(1, n_frames // 20) == 0:
            print(f"  frame {i}/{n_frames}")

        if do_render:
            if i >= args.skip:
                frame = capture_dual(env, args.size)
                if frame is not None:
                    frames.append(frame)
                else:
                    print(f"  render failed at frame {i}, stopping render")
                    do_render = False
            if i % 100 == 0:
                gc.collect()

        obs, _, done, info = env.step(actions[i])
        if done:
            outcome = "SUCCESS" if info.get("success") else (
                "COLLISION" if info.get("arm_collision") else
                ("OUT-OF-BOUNDS" if info.get("out_of_bounds") else
                 ("BOX_FELL" if info.get("box_fell") else "DONE")))
            print(f"  episode ended at step {i}: {outcome}")
            break

    # Report final box position
    box = obs.get("box_pos_xy", np.zeros(2))
    print(f"  final box_pos: ({box[0]:.4f}, {box[1]:.4f})  dist_to_goal={np.linalg.norm(box - np.array([0.0, 0.25])):.3f}m")

    env.close()

    if frames:
        frames_arr = np.stack(frames, axis=0)
        np.savez_compressed(out, frames=frames_arr)
        print(f"Saved {len(frames)} frames to {out}")
    elif do_render:
        print("No frames rendered")
    else:
        print(f"Physics-only replay done")

    # Optional: try ffmpeg conversion (subprocess, isolated from EGL)
    if frames and out.endswith(".npz"):
        mp4_out = out.replace(".npz", ".mp4")
        import shutil
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            import subprocess
            import tempfile
            tmp_raw = tempfile.NamedTemporaryFile(suffix=".raw", delete=False)
            tmp_raw.close()
            tmp_path = tmp_raw.name
            with open(tmp_path, "wb") as f:
                f.write(frames_arr.tobytes())
            h, w = frames_arr.shape[1], frames_arr.shape[2]
            cmd = [
                ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{w}x{h}", "-r", "20", "-i", tmp_path,
                "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p", mp4_out,
            ]
            try:
                subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60)
                print(f"  also saved {mp4_out}")
            except Exception as e:
                print(f"  ffmpeg skipped ({e})")
            os.unlink(tmp_path)


if __name__ == "__main__":
    main()

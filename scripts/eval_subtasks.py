"""Evaluate ARP policy on decoupled PushBox subtasks.

Each subtask (approach / cross / reach) is evaluated independently with its
own box initial position, so later subtasks are *not* gated by earlier ones.

Usage:
    # Evaluate all three subtasks, 20 episodes each
    python scripts/eval_subtasks.py <checkpoint_dir>

    # Single subtask only
    python scripts/eval_subtasks.py <checkpoint_dir> --tasks cross

    # Custom episode count, save all videos
    python scripts/eval_subtasks.py <checkpoint_dir> --episodes 50 --all-videos
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pushbox import arp
from pushbox.diffusion_policy.model.common.normalizer import LinearNormalizer
from chunk_selector.chunk_dataset import CoarseChunkFeatureCollector, parse_subtask_chunks
from chunk_selector.chunk_selector import ChunkSelector
from pushbox.policy import ACTION_DIM, AGENT_STATE_DIM, PushBoxARPolicy
from scripts.eval_policy import load_policy as load_shared_policy

# ── Constants ────────────────────────────────────────────────────────────────
POLICY_IMG_SIZE = 96
RECORD_IMG_SIZE = 256

# Phase y-thresholds from PushBoxEnv
PHASE_Y_APPROACH = -0.10
PHASE_Y_CROSS = 0.10

# Subtask definitions: each subtask maps to a phase and a box-init y range.
# The y-range places the box in the *starting* region for that phase so
# the episode begins right at the subtask boundary.
# ── Subtask 1 ──
# Box starts before the corridor, succeeds when pushed close to it.
# ── Subtask 2 ──
# Box starts inside the obstacle corridor, succeeds when pushed out the
# far side (past y=0.10).
# ── Subtask 3 ──
# Box starts after the corridor, succeeds when pushed to the goal zone.
SUBTASK_DEFS: dict[str, dict] = {
    "1": {
        "phase_idx": 0,
        "y_range": (-0.25, -0.12),
        "x_range": (0.01, 0.05),
        "success_y": -0.15,
        "description": "push box from start area past y=-0.15",
    },
    "2": {
        "phase_idx": 1,
        "y_range": (-0.20, -0.15),
        # Bias resets toward the upper edge: x=0.06 has the highest
        # probability and density decreases linearly toward x=-0.01.
        "x_range": (-0.01, 0.06),
        "success_y": PHASE_Y_CROSS,
        "description": "push box through obstacle corridor past y=0.10",
    },
    "3": {
        "phase_idx": 2,
        "y_range": (0.11, 0.13),
        "x_range": (0.01, 0.05),
        "description": "push box from corridor exit to goal (0.0, 0.25)",
    },
}

# ── Low-level helpers  (keep in sync with eval_policy.py) ────────────────────


def capture_view(env, camera_name: str, size: int = RECORD_IMG_SIZE) -> np.ndarray:
    if (
        hasattr(env.sim, "_render_context_offscreen")
        and env.sim._render_context_offscreen is not None
    ):
        env.sim._render_context_offscreen.gl_ctx.make_current()
    frame = env.sim.render(width=size, height=size, camera_name=camera_name)
    return np.flipud(frame).astype(np.uint8)


def get_agent_state(obs: dict) -> np.ndarray:
    joints = obs.get("robot0_joint_pos", np.zeros(7, dtype=np.float32))
    gripper = obs.get("robot0_gripper_qpos", np.zeros(2, dtype=np.float32))
    return np.concatenate([joints, gripper]).astype(np.float32)


def get_box_pos(obs: dict) -> np.ndarray:
    return obs.get("box_pos_xy", np.zeros(2, dtype=np.float32)).astype(np.float32)


def save_video(frames: list, out_path: str, fps: int = 20) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        print(f"    video SKIPPED (0 frames): {out_path}")
        return

    # Ensure all frames are valid uint8 RGB
    clean_frames = []
    for f in frames:
        f = np.asarray(f)
        if f.dtype != np.uint8:
            f = np.clip(f, 0, 255).astype(np.uint8)
        if len(f.shape) == 3 and f.shape[-1] == 4:
            f = f[..., :3]
        clean_frames.append(f)
    frames = clean_frames

    # Save .npz backup for safety
    npz_path = out_path.replace(".mp4", ".npz")
    try:
        np.savez_compressed(npz_path, frames=np.stack(frames))
    except Exception:
        pass

    # Write mp4 via imageio (more portable than cv2)
    try:
        import imageio

        writer = imageio.get_writer(out_path, fps=fps, codec="libx264")
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        print(f"    video saved ({len(frames)} frames): {out_path}")
    except Exception as e:
        print(f"    mp4 failed ({e}), npz saved: {npz_path}")


# ── Checkpoint loading (shared with eval_policy.py) ──────────────────────────


def _detect_checkpoint_config(ckpt_sd: dict) -> dict:
    """Inspect checkpoint state_dict and return pretrained + arp_cfg + tokens."""
    tk_embed_key = "policy.chunk_embedder.token_type_embed.weight"
    if tk_embed_key in ckpt_sd:
        num_tokens = ckpt_sd[tk_embed_key].shape[0]
    else:
        num_tokens = 4

    token_dims = {}
    for i in range(num_tokens):
        embed_key = f"policy.token_embedders.{i}.embed.weight"
        if embed_key in ckpt_sd:
            token_dims[i] = ckpt_sd[embed_key].shape[1]
        else:
            token_dims[i] = 9

    chunk_embed_w = ckpt_sd.get("policy.chunk_embedder.chunk_embed.weight")
    action_chunk_size = chunk_embed_w.shape[0] if chunk_embed_w is not None else 50
    n_embd = chunk_embed_w.shape[1] if chunk_embed_w is not None else 64

    num_latents = 1
    for i in range(num_tokens):
        pred_key = f"policy.token_predictors.{i}.mlp.fc2.weight"
        if pred_key in ckpt_sd:
            out_dim = ckpt_sd[pred_key].shape[0]
            dim = token_dims.get(i, 9)
            num_latents = out_dim // (2 * dim + 1)
            break

    pos_emb = ckpt_sd.get("policy.pos_emb.weight")
    n_obs_steps = 2
    if pos_emb is not None:
        max_seq_len = pos_emb.shape[0]
        horizon = action_chunk_size
        plan_steps = max_seq_len // 2 - n_obs_steps - horizon
    else:
        plan_steps = 5
        horizon = action_chunk_size

    token_names = {0: "pos", 1: "coarse-plan", 2: "fine-action", 3: "box_pos"}

    tokens = []
    for i in range(num_tokens):
        name = token_names.get(i, f"token-{i}")
        dim = token_dims[i]
        kwargs = {
            "name": name,
            "dim": dim,
            "is_continuous": True,
            "embedding": "linear",
            "is_control": (i == 0),
        }
        if not kwargs["is_control"]:
            kwargs["predictor"] = "gmm"
            kwargs["predictor_kwargs"] = {"num_latents": num_latents, "low_var_eval": True}
        tokens.append(arp.TokenType.make(**kwargs))

    config = {
        "pretrained": True,
        "arp_cfg": {
            "n_embd": n_embd,
            "embd_pdrop": 0.1,
            "layer_norm_every_block": False,
            "num_layers": 6,
            "layer_cfg": {
                "n_head": 8,
                "mlp_ratio": 4.0,
                "AdaLN": True,
                "mlp_dropout": 0.1,
                "attn_kwargs": {"attn_pdrop": 0.1, "resid_pdrop": 0.1},
                "cond_attn_kwargs": {"attn_pdrop": 0.1, "resid_pdrop": 0.1},
            },
            "plan_steps": plan_steps,
            "plan_chunk_size": 1,
            "action_chunk_size": action_chunk_size,
            "num_latents": num_latents,
            "low_var_eval": True,
            "sample": False,
        },
        "horizon": horizon,
        "n_action_steps": action_chunk_size,
        "n_obs_steps": n_obs_steps,
    }

    print(
        f"[load] detected: num_tokens={num_tokens}, n_embd={n_embd}, "
        f"plan_steps={plan_steps}, action_chunk_size={action_chunk_size}, "
        f"num_latents={num_latents}, horizon={horizon}"
    )
    print(
        f"[load] tokens: {['{}:{}D {}'.format(tk['name'], tk['dim'], 'ctrl' if tk.get('is_control') else 'pred') for tk in tokens]}"
    )
    config["tokens"] = tokens
    return config


def load_policy(checkpoint_dir: str, device: str = "cuda:0"):
    ckpt_dir = Path(checkpoint_dir)
    if ckpt_dir.is_file():
        ckpt_path = ckpt_dir
    else:
        ckpts = sorted(ckpt_dir.glob("*.ckpt"))
        if not ckpts:
            # Try checkpoints/ subdirectory
            ckpt_sub = ckpt_dir / "checkpoints"
            ckpts = sorted(ckpt_sub.glob("*.ckpt")) if ckpt_sub.is_dir() else []
        if not ckpts:
            raise FileNotFoundError(f"No .ckpt files found in {ckpt_dir}")
        ckpt_path = ckpts[0]

    print(f"\n[load] checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dicts" in ckpt:
        sd = ckpt["state_dicts"]["model"]
    elif "model" in ckpt:
        sd = ckpt["model"]
    else:
        sd = ckpt

    ckpt_config = _detect_checkpoint_config(sd)

    shape_meta = {
        "action": {"shape": [ACTION_DIM]},
        "obs": {
            "agent_pos": {"shape": [AGENT_STATE_DIM], "type": "low_dim"},
            "box_pos": {"shape": [2], "type": "low_dim"},
            "top45": {"shape": [3, 96, 96], "type": "rgb"},
            "sideview": {"shape": [3, 96, 96], "type": "rgb"},
        },
    }
    policy = PushBoxARPolicy(
        shape_meta=shape_meta,
        horizon=ckpt_config.get("horizon", 20),
        n_action_steps=ckpt_config.get("n_action_steps", 20),
        n_obs_steps=ckpt_config.get("n_obs_steps", 2),
        crop_shape=(84, 84),
        obs_encoder_group_norm=True,
        eval_fixed_crop=False,
        pretrained=ckpt_config["pretrained"],
        arp_cfg=ckpt_config["arp_cfg"],
        tokens=ckpt_config.get("tokens") or [],
    )

    cleaned = {}
    for k, v in sd.items():
        if k.startswith("ema_model."):
            k = k[len("ema_model.") :]
        elif k.startswith("model."):
            k = k[len("model.") :]
        cleaned[k] = v

    norm_sd = {}
    for k, v in cleaned.items():
        if k.startswith("normalizer."):
            norm_sd[k[len("normalizer.") :]] = v
    normalizer = LinearNormalizer()
    normalizer.load_state_dict(norm_sd)
    policy.set_normalizer(normalizer)

    policy_sd = {k: v for k, v in cleaned.items() if not k.startswith("normalizer.")}
    own = policy.state_dict()
    matched = {k: v for k, v in policy_sd.items() if k in own}
    print(f"[load] matched {len(matched)}/{len(own)} param keys")
    own.update(matched)
    policy.load_state_dict(own, strict=False)
    policy.to(device)
    policy.eval()
    return policy


# ── Episode runner ───────────────────────────────────────────────────────────


def run_episode(
    env,
    policy,
    device,
    *,
    max_steps: int,
    target_phase: int,
    record_video: bool,
    success_y: float | None = None,
    debug_init: bool = False,
    rnd_obs: bool = False,
    feature_collector: CoarseChunkFeatureCollector | None = None,
    episode_id: str = "",
    subtask: str = "",
):
    """Run one episode, return (success, steps, fail_reason, frames, init_frames, wall_time, inf_passes).

    *success* is scoped to *target_phase* (only that phase must succeed).
    If *success_y* is given for target_phase=0, approach succeeds when
    box_y > success_y (overriding the env's PHASE_Y_APPROACH).

    When *debug_init* is True, a separate video of the grip initialization
    is recorded and returned as the fourth element.

    *wall_time* is the real elapsed wall-clock time in seconds (excluding grip init).
    *inf_passes* is the number of policy.predict_action() calls in this episode.
    """
    # ── Grip initialization (with optional diagnostics) ──
    init_frames: list | None = None

    if debug_init:

        def _grip_capture(env):
            f = capture_view(env, "top45", size=RECORD_IMG_SIZE)
            s = capture_view(env, "sideview", size=RECORD_IMG_SIZE)
            return np.concatenate([f, s], axis=1)

        obs, grip_steps, init_frames = env.init_with_grip(verbose=True, record_fn=_grip_capture)
    else:
        obs, grip_steps, _ = env.init_with_grip()

    # Apply light obstacle randomisation after grip init (overrides fixed config)
    if rnd_obs:
        env._place_obstacles(_light_obstacle_configs(np.random))

    t_start = time.time()
    inf_passes = 0

    top45_buf: list = []
    side_buf: list = []
    state_buf: list = []
    box_buf: list = []
    step = 0
    success = False
    done = False
    fail_reason = ""
    frames: list = []

    while step < max_steps:
        top45 = capture_view(env, "top45", size=POLICY_IMG_SIZE).astype(np.float32) / 255.0
        sideview = capture_view(env, "sideview", size=POLICY_IMG_SIZE).astype(np.float32) / 255.0

        state = get_agent_state(obs)
        box_pos = get_box_pos(obs)

        if len(top45_buf) < 2:
            top45_buf = [top45, top45]
            side_buf = [sideview, sideview]
            state_buf = [state, state]
            box_buf = [box_pos, box_pos]
        else:
            top45_buf.append(top45)
            side_buf.append(sideview)
            state_buf.append(state)
            box_buf.append(box_pos)
            top45_buf = top45_buf[-2:]
            side_buf = side_buf[-2:]
            state_buf = state_buf[-2:]
            box_buf = box_buf[-2:]

        top45_np = np.stack(top45_buf, axis=0).transpose(0, 3, 1, 2)
        side_np = np.stack(side_buf, axis=0).transpose(0, 3, 1, 2)
        top45_t = torch.from_numpy(top45_np).unsqueeze(0).to(device)
        side_t = torch.from_numpy(side_np).unsqueeze(0).to(device)
        state_t = torch.from_numpy(np.stack(state_buf, axis=0)).unsqueeze(0).to(device)
        box_t = torch.from_numpy(np.stack(box_buf, axis=0)).unsqueeze(0).to(device)

        with torch.no_grad():
            result = policy.predict_action(
                {"top45": top45_t, "sideview": side_t, "agent_pos": state_t, "box_pos": box_t}
            )
        inf_passes += 1
        if feature_collector is not None:
            features = getattr(policy, "last_chunk_features", None)
            if features is None:
                raise RuntimeError("PushBox policy did not expose shared visual features")
            feature_collector.record(
                features,
                episode_id=episode_id,
                decision_step=step,
                subtask=subtask,
            )
        actions = result["action"][0].cpu().numpy()

        for i in range(len(actions)):
            obs, reward, done, info = env.step(actions[i])
            step += 1

            # Record video *after* each step so we see the effect of the action
            if record_video:
                top_hd = capture_view(env, "top45", size=RECORD_IMG_SIZE)
                side_hd = capture_view(env, "sideview", size=RECORD_IMG_SIZE)
                frames.append(np.concatenate([top_hd, side_hd], axis=1))

            # Check if target phase already succeeded (may have happened earlier
            # in this action chunk or even in a previous chunk).
            # For approach subtask with success_y, also check box_y directly.
            phase_ok = info.get("phase_success", [False] * 3)
            target_ok = bool(phase_ok[target_phase]) if target_phase < len(phase_ok) else False
            if not target_ok and success_y is not None:
                box_y = float(get_box_pos(obs)[1])
                target_ok = box_y > success_y

            # Early termination when sub-goal (success_y) is reached.
            # Only for subtasks with an explicit success_y; subtask 3
            # relies on the env's _check_success (goal zone).
            if target_ok and success_y is not None:
                done = True
                success = True

            if info.get("arm_collision", False):
                done = True
                success = target_ok
                if not success:
                    box_y = env._box_y() if hasattr(env, "_box_y") else float(get_box_pos(obs)[1])
                    fail_reason = f"arm_collision (step={step}, box_y={box_y:.3f}, phase={info.get('current_phase', '?')})"
            elif info.get("out_of_bounds", False):
                done = True
                success = target_ok
                if not success:
                    fail_reason = "out_of_bounds"
            elif info.get("box_fell", False):
                done = True
                success = target_ok
                if not success:
                    fail_reason = "box_fell"
            elif info.get("phase_timeout", False):
                done = True
                success = target_ok
                if not success:
                    fail_reason = f"phase_timeout (phase={info.get('current_phase', '?')})"
            elif done:
                success = target_ok
                if not success:
                    fail_reason = f"done_without_success (phase_ok={phase_ok})"
            if done:
                break

        if done:
            break

    # Sanity-check frame consistency before returning
    if frames:
        shapes = {f.shape for f in frames}
        if len(shapes) > 1:
            print(f"    [WARN] inconsistent frame shapes: {shapes}")

    wall_time = time.time() - t_start
    return success, step, fail_reason, frames, init_frames, wall_time, inf_passes


def _make_env(
    max_steps: int,
    y_range: tuple[float, float],
    x_range: tuple[float, float] | None = None,
    seed: int | None = None,
    rnd_obs: bool = False,
):
    """Create a PushBoxEnv with a custom box-init y range."""
    from envs import PushBoxEnv
    from envs.pushbox_env import V2_FIXED_OBSTACLES

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault(
        "__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
    )
    os.environ.pop("DISPLAY", None)

    return PushBoxEnv(
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        use_object_obs=True,
        reward_shaping=True,
        horizon=max_steps,
        hard_reset=False,
        camera_names="top45,sideview",
        camera_heights=256,
        camera_widths=256,
        box_init_y_range=y_range,
        box_init_x_range=x_range,
        box_x_sampler=_edge_bias_sampler if x_range is not None else None,
        obstacle_configs=V2_FIXED_OBSTACLES,
        seed=seed,
    )


# ── Light obstacle randomisation ───────────────────────────────────────────


def _edge_bias_sampler(rng, lo, hi):
    """Triangular distribution: lo min probability, hi max probability."""
    return float(rng.triangular(lo, hi, hi))


def _light_obstacle_configs(rng) -> list[dict]:
    """Generate light-random obstacle configs: y-offset +-0.5cm only."""
    y_offset = float(rng.uniform(-0.005, 0.005))
    return [
        {"pos": [0.215, y_offset], "half_size": [0.18, 0.04, 0.05], "euler": [0.0, 0.0, 0.0]},
        {"pos": [-0.225, y_offset], "half_size": [0.18, 0.04, 0.05], "euler": [0.0, 0.0, 0.0]},
    ]


# ── Main ─────────────────────────────────────────────────────────────────────


def _eval_subtasks(
    policy,
    device: str,
    tasks: list[str],
    episodes: int,
    max_steps: int,
    no_video: bool,
    all_videos: bool,
    debug_init: bool,
    video_out_dir: Path,
    seed: int | None = None,
    rnd_obs: bool = False,
    feature_collector: CoarseChunkFeatureCollector | None = None,
) -> dict:
    """Run subtask evaluation for current policy settings.

    Returns a dict with keys: "subtasks", "overall_successes", "overall_total".
    """
    all_subtask_results: dict[str, dict] = {}
    overall_successes = 0
    overall_total = 0

    for st_name in tasks:
        st_def = SUBTASK_DEFS[st_name]
        target_phase = st_def["phase_idx"]
        y_range = st_def["y_range"]

        print(f"\n{'─' * 48}")
        print(f"  [Subtask {st_name}] box y ∈ {y_range}  →  {st_def['description']}")
        print(f"{'─' * 48}")

        env = _make_env(
            max_steps,
            y_range,
            x_range=st_def.get("x_range"),
            seed=(seed + target_phase) if seed is not None else None,
            rnd_obs=rnd_obs,
        )

        successes = 0
        steps_all: list[int] = []
        wall_times_all: list[float] = []
        inf_passes_all: list[int] = []
        steps_success: list[int] = []
        wall_times_success: list[float] = []
        inf_passes_success: list[int] = []

        for ep in range(1, episodes + 1):
            # Pair stochastic policy samples across chunk candidates. Each
            # candidate recreates the env with the same seed, while this reset
            # makes the policy RNG identical for the corresponding episode.
            if seed is not None:
                episode_seed = int(seed + target_phase * 100_000 + ep - 1)
                random.seed(episode_seed)
                np.random.seed(episode_seed)
                torch.manual_seed(episode_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(episode_seed)
            record = not no_video
            success, steps, fail_reason, frames, init_frames, wall_time, inf_passes = run_episode(
                env,
                policy,
                device,
                max_steps=max_steps,
                target_phase=target_phase,
                record_video=record,
                success_y=st_def.get("success_y"),
                debug_init=debug_init,
                rnd_obs=rnd_obs,
                feature_collector=feature_collector,
                episode_id=f"subtask{st_name}:episode{ep:04d}",
                subtask=st_name,
            )
            successes += int(success)
            steps_all.append(steps)
            wall_times_all.append(wall_time)
            inf_passes_all.append(inf_passes)
            if success:
                steps_success.append(steps)
                wall_times_success.append(wall_time)
                inf_passes_success.append(inf_passes)

            status = "OK" if success else "FAIL"
            detail = f"  ({fail_reason})" if fail_reason else ""
            print(
                f"    ep {ep:3d}/{episodes}: {status:4s}  steps={steps:5d}  wall={wall_time:.1f}s  inf_passes={inf_passes}{detail}"
            )

            # Save video if requested
            if record and frames and (all_videos or not success):
                sub_dir = video_out_dir / st_name
                sub_dir.mkdir(parents=True, exist_ok=True)
                out_path = str(sub_dir / f"ep{ep:03d}_{status}_{steps}steps.mp4")
                save_video(frames, out_path)

            # Save grip-init debug video
            if debug_init and init_frames:
                grip_dir = video_out_dir / st_name
                grip_dir.mkdir(parents=True, exist_ok=True)
                grip_path = str(grip_dir / f"ep{ep:03d}_grip_init_{status}.mp4")
                save_video(init_frames, grip_path, fps=10)

        env.close()

        # ── Per-subtask stats ──────────────────────────────────────────
        rate = 100.0 * successes / episodes
        avg_steps = float(np.mean(steps_all)) if steps_all else 0.0
        min_steps = int(np.min(steps_all)) if steps_all else 0
        max_steps_seen = int(np.max(steps_all)) if steps_all else 0

        print(
            f"\n    [Subtask {st_name}] {successes}/{episodes} ({rate:.1f}%)  "
            f"avg(all)={avg_steps:.0f}  min={min_steps}  max={max_steps_seen}"
        )

        all_subtask_results[st_name] = {
            "successes": successes,
            "failures": episodes - successes,
            "success_rate": round(float(successes) / episodes, 4),
            "steps": {
                "mean": round(avg_steps, 1),
                "min": min_steps,
                "max": max_steps_seen,
                "mean_success": round(float(np.mean(steps_success)), 1)
                if steps_success
                else 0,
            },
            "wall_time": {
                "mean": round(float(np.mean(wall_times_all)), 2) if wall_times_all else 0,
                "min": round(float(np.min(wall_times_all)), 2) if wall_times_all else 0,
                "max": round(float(np.max(wall_times_all)), 2) if wall_times_all else 0,
                "mean_success": round(float(np.mean(wall_times_success)), 2)
                if wall_times_success
                else 0,
            },
            "inference_passes": {
                "mean": round(float(np.mean(inf_passes_all)), 1) if inf_passes_all else 0,
                "min": int(np.min(inf_passes_all)) if inf_passes_all else 0,
                "max": int(np.max(inf_passes_all)) if inf_passes_all else 0,
                "mean_success": round(float(np.mean(inf_passes_success)), 1)
                if inf_passes_success
                else 0,
            },
        }

        overall_successes += successes
        overall_total += episodes

    return {
        "subtasks": all_subtask_results,
        "overall_successes": overall_successes,
        "overall_total": overall_total,
    }


def _plot_chunk_comparison(
    all_chunk_results: dict,
    tasks: list[str],
    output_dir: Path,
):  # pyright: ignore[reportUnknownMemberType]
    """Generate grouped bar chart comparing success rate, wall time, and inference passes across chunk sizes."""
    chunk_sizes = sorted(all_chunk_results.keys())
    subtasks = sorted(tasks, key=lambda t: int(t))

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 11), sharex=True)

    x = np.arange(len(subtasks))
    width = 0.2
    colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99", "#c2c2f0"]

    # --- Top: Success Rate ---
    for i, cs in enumerate(chunk_sizes):
        rates = []
        for st in subtasks:
            r = all_chunk_results[cs]["subtasks"].get(st, {})
            rates.append(r.get("success_rate", 0) * 100)
        offset = (i - len(chunk_sizes) / 2 + 0.5) * width
        bars = ax1.bar(
            x + offset,
            rates,
            width,
            label=f"chunk={cs}",
            color=colors[i % len(colors)],
        )
        for bar, rate in zip(bars, rates):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{rate:.0f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax1.set_ylabel("Success Rate (%)")
    ax1.set_title("Chunk Size Comparison: Success Rate")
    ax1.set_ylim(0, 105)
    ax1.legend(loc="lower right")
    ax1.grid(axis="y", alpha=0.3)

    # --- Bottom: Avg Wall Time ---
    for i, cs in enumerate(chunk_sizes):
        times = []
        for st in subtasks:
            r = all_chunk_results[cs]["subtasks"].get(st, {})
            t = r.get("wall_time", {}).get("mean", 0)
            times.append(t)
        offset = (i - len(chunk_sizes) / 2 + 0.5) * width
        bars = ax2.bar(
            x + offset,
            times,
            width,
            label=f"chunk={cs}",
            color=colors[i % len(colors)],
        )
        for bar, t in zip(bars, times):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{t:.1f}s",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax2.set_ylabel("Avg Wall Time (s)")
    ax2.set_title("Chunk Size Comparison: Wall Time")
    ax2.legend(loc="upper right")
    ax2.grid(axis="y", alpha=0.3)

    # --- Bottom: Avg Inference Passes ---
    for i, cs in enumerate(chunk_sizes):
        passes = []
        for st in subtasks:
            r = all_chunk_results[cs]["subtasks"].get(st, {})
            p = r.get("inference_passes", {}).get("mean", 0)
            passes.append(p)
        offset = (i - len(chunk_sizes) / 2 + 0.5) * width
        bars = ax3.bar(
            x + offset,
            passes,
            width,
            label=f"chunk={cs}",
            color=colors[i % len(colors)],
        )
        for bar, p in zip(bars, passes):
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(passes) * 0.02,
                f"{p:.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax3.set_ylabel("Avg Inference Passes")
    ax3.set_xlabel("Subtask")
    ax3.set_title("Chunk Size Comparison: Inference Passes")
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"Subtask {st}" for st in subtasks])
    ax3.legend(loc="upper right")
    ax3.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / "chunk_comparison.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")  # type: ignore[attr-defined]
    plt.close(fig)
    print(f"\n  Comparison chart saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ARP policy on decoupled PushBox subtasks"
    )
    parser.add_argument("checkpoint", type=str, help="checkpoint dir or .ckpt file")
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["1", "2", "3"],
        default=None,
        help="which subtasks to evaluate (default: all three)",
    )
    parser.add_argument(
        "--episodes", type=int, default=20, help="episodes per subtask (default: 20)"
    )
    parser.add_argument(
        "--max-steps", type=int, default=8000, help="max steps per episode (default: 8000)"
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--weights",
        choices=("ema", "model"),
        default="ema",
        help="checkpoint weights to evaluate (default: ema)",
    )
    parser.add_argument(
        "--failures-only",
        dest="all_videos",
        action="store_false",
        help="only save videos for failed episodes (default: save all)",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="disable video saving entirely",
    )
    parser.add_argument(
        "--debug-init",
        action="store_true",
        help="record video of the grip initialization phase + verbose diagnostics",
    )
    parser.add_argument(
        "--video-dir", type=str, default=None, help="custom output directory for videos"
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed for reproducibility")
    parser.add_argument("--no-summary", action="store_true", help="suppress final summary table")
    parser.add_argument(
        "--chunk-selector",
        type=Path,
        default=None,
        help="optional adaptive_chunk directory containing a trained selector sidecar",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="limit actions per inference chunk (default: full horizon from ckpt). "
        "Smaller values mean more frequent re-inference.",
    )
    parser.add_argument(
        "--all-chunks",
        action="store_true",
        help="evaluate candidate chunks and generate comparison chart. "
        "Mutually exclusive with --chunk-size.",
    )
    parser.add_argument(
        "--candidate-chunks",
        type=int,
        nargs="+",
        default=None,
        help="fixed chunks compared by --all-chunks; default: 1 2 5 10 and maximum",
    )
    obstacle_randomization = parser.add_mutually_exclusive_group()
    obstacle_randomization.add_argument(
        "--rnd-obs",
        action="store_true",
        dest="rnd_obs",
        help="enable light obstacle randomisation (+-0.5cm; disabled by default)",
    )
    obstacle_randomization.add_argument(
        "--no-rnd-obs",
        action="store_false",
        dest="rnd_obs",
        help="explicitly keep obstacles fixed (the default)",
    )
    parser.set_defaults(rnd_obs=False)
    parser.add_argument(
        "--feature-dataset",
        type=Path,
        default=None,
        help="write coarse-labeled shared visual features to this HDF5 file",
    )
    parser.add_argument(
        "--coarse-labels",
        nargs="+",
        default=None,
        metavar="SUBTASK=CHUNK",
        help="required for collection, for example 1=5 2=15 3=5",
    )
    parser.add_argument(
        "--selector-candidates",
        type=int,
        nargs="+",
        default=None,
        help="selector classes; coarse collection defaults to the coarse-label values",
    )
    parser.add_argument(
        "--collection-chunk",
        type=int,
        default=None,
        help="fixed behavior chunk used for collection; default: maximum executable chunk",
    )
    parser.add_argument("--overwrite-feature-dataset", action="store_true")
    args = parser.parse_args()

    # Validate mutual exclusivity
    if args.all_chunks and args.chunk_size is not None:
        parser.error("--all-chunks and --chunk-size are mutually exclusive")
    if args.chunk_selector is not None and (args.all_chunks or args.chunk_size is not None):
        parser.error("--chunk-selector cannot be combined with fixed chunk evaluation")
    if args.feature_dataset is not None and (
        args.all_chunks or args.chunk_size is not None or args.chunk_selector is not None
    ):
        parser.error(
            "--feature-dataset cannot be combined with --all-chunks, --chunk-size, "
            "or --chunk-selector"
        )

    tasks = args.tasks if args.tasks else list(SUBTASK_DEFS.keys())

    # ── Load policy ────────────────────────────────────────────────────
    policy = load_shared_policy(
        args.checkpoint,
        device=args.device,
        weights=args.weights,
        use_checkpoint_config=True,
    )
    maximum_chunk = int(
        getattr(
            policy,
            "max_selector_chunk_label",
            getattr(policy, "max_selector_chunk", policy.horizon),
        )
    )
    if args.chunk_size is not None and not 1 <= args.chunk_size <= maximum_chunk:
        parser.error(f"--chunk-size must lie in [1, {maximum_chunk}]")
    comparison_chunks = sorted(
        set(
            args.candidate_chunks
            or [
                value
                for value in (1, 2, 5, 10, maximum_chunk)
                if value <= maximum_chunk
            ]
        )
    )
    invalid_comparison_chunks = [
        value for value in comparison_chunks if value <= 0 or value > maximum_chunk
    ]
    if invalid_comparison_chunks:
        parser.error(
            f"--candidate-chunks must lie in [1, {maximum_chunk}]: "
            f"{invalid_comparison_chunks}"
        )
    feature_collector = None
    if args.feature_dataset is not None:
        if not args.coarse_labels:
            parser.error("--coarse-labels is required with --feature-dataset")
        default_candidates = [
            value
            for value in (1, 2, 5, 10, 15, maximum_chunk)
            if value <= maximum_chunk
        ]
        selector_candidates = sorted(
            set(args.selector_candidates or default_candidates)
        )
        invalid_candidates = [
            value
            for value in selector_candidates
            if value <= 0 or value > maximum_chunk
        ]
        if invalid_candidates:
            parser.error(
                f"selector candidates must lie in [1, {maximum_chunk}]: "
                f"{invalid_candidates}"
            )
        try:
            coarse_labels = parse_subtask_chunks(args.coarse_labels)
        except ValueError as error:
            parser.error(str(error))
        if args.selector_candidates is None:
            selector_candidates = sorted(set(coarse_labels.values()))
        out_of_range_labels = sorted(
            value
            for value in set(coarse_labels.values())
            if value > maximum_chunk
        )
        if out_of_range_labels:
            parser.error(
                f"coarse labels exceed maximum executable chunk {maximum_chunk}: "
                f"{out_of_range_labels}"
            )
        missing_labels = [task for task in tasks if task not in coarse_labels]
        if missing_labels:
            parser.error(f"missing coarse labels for tasks: {missing_labels}")
        invalid_labels = sorted(set(coarse_labels.values()) - set(selector_candidates))
        if invalid_labels:
            parser.error(
                f"coarse labels {invalid_labels} are absent from selector candidates "
                f"{selector_candidates}"
            )
        unused_candidates = sorted(
            set(selector_candidates) - set(coarse_labels.values())
        )
        if unused_candidates:
            parser.error(
                "coarse collection would create classes without positive samples: "
                f"{unused_candidates}"
            )
        behavior_chunk = args.collection_chunk or maximum_chunk
        if behavior_chunk <= 0 or behavior_chunk > maximum_chunk:
            parser.error(f"--collection-chunk must lie in [1, {maximum_chunk}]")
        if args.feature_dataset.exists() and not args.overwrite_feature_dataset:
            parser.error(
                f"{args.feature_dataset} already exists; pass --overwrite-feature-dataset"
            )
        policy.inference_chunk_size = behavior_chunk
        feature_collector = CoarseChunkFeatureCollector(
            args.feature_dataset,
            candidate_chunks=selector_candidates,
            subtask_chunks=coarse_labels,
            overwrite=args.overwrite_feature_dataset,
            metadata={
                "policy_type": type(policy).__name__,
                "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
                "behavior_chunk": behavior_chunk,
            },
        )
    if args.chunk_selector is not None:
        if not hasattr(policy, "set_chunk_selector"):
            parser.error("checkpoint policy does not support adaptive chunk selection")
        policy.set_chunk_selector(
            ChunkSelector.from_pretrained(args.chunk_selector, device=args.device)
        )
        print(f"[eval] adaptive chunk selector={args.chunk_selector.expanduser().resolve()}")

    # ── Apply chunk size limit ─────────────────────────────────────────
    chunk_size = args.chunk_size
    if chunk_size is not None:
        policy.inference_chunk_size = chunk_size  # type: ignore[assignment]

    # ── Set random seeds ──────────────────────────────────────────────
    if args.seed is not None:
        import random

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # ── Video output directory (compute before cue card) ──────────────
    if args.video_dir:
        video_dir = Path(args.video_dir)
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        video_dir = Path("/home/huiyuan/pushbox/eval_output") / timestamp
    epoch_video_dir = video_dir

    # Cue card
    print("\n" + "=" * 64)
    print("  Subtask Evaluation")
    print(f"  Tasks:      {', '.join(tasks)}")
    print(
        f"  Episodes:   {args.episodes} per task ({len(tasks)} × {args.episodes} = {len(tasks) * args.episodes} total)"
    )
    print(f"  Max steps:  {args.max_steps}")
    print(f"  Device:     {args.device}")
    print(f"  Weights:    {args.weights}")
    seed_display = str(args.seed) if args.seed is not None else "random"
    print(f"  Seed:       {seed_display}")
    chunk_display = f"{chunk_size}" if chunk_size is not None else "full (ckpt default)"
    print(f"  Chunk size: {chunk_display}")
    print(
        f"  Video:      {'all' if args.all_videos else 'failures only' if not args.no_video else 'none'}"
    )
    print(f"  Obstacles:  {'randomized (+-0.5cm)' if args.rnd_obs else 'fixed'}")
    print(f"  Output:     {epoch_video_dir}")
    print("=" * 64)

    save_videos = not args.no_video
    if save_videos:
        epoch_video_dir.mkdir(parents=True, exist_ok=True)

    # ── Run evaluation ──────────────────────────────────────────────────
    if args.all_chunks:
        all_chunk_data: dict[int, dict] = {}

        for cs in comparison_chunks:
            chunk_label = str(cs)
            policy.inference_chunk_size = cs  # type: ignore[assignment]

            print(f"\n{'#' * 64}")
            print(
                f"  Chunk size block: {chunk_label}  "
                f"(inference_chunk_size={cs})"
            )
            print(f"{'#' * 64}")

            chunk_video_dir = epoch_video_dir / f"chunk_{chunk_label}"
            save_videos_chunk = save_videos

            result = _eval_subtasks(
                policy,
                args.device,
                tasks,
                args.episodes,
                args.max_steps,
                no_video=not save_videos_chunk,
                all_videos=args.all_videos,
                debug_init=args.debug_init,
                video_out_dir=chunk_video_dir,
                seed=args.seed,
                rnd_obs=args.rnd_obs,
            )
            all_chunk_data[cs] = result

            # Per-chunk quick summary
            if not args.no_summary:
                r = result
                sorted_tasks = sorted(tasks, key=lambda t: int(t))
                print(f"\n  --- Chunk {chunk_label} Summary ---")
                print(
                    f"  {'Subtask':<10s} {'Success':>10s} {'Rate':>8s} {'Wall(s)':>8s} {'Inf(#)':>8s}"
                )
                for st_name in sorted_tasks:
                    sr = r["subtasks"][st_name]
                    wall_mean = sr.get("wall_time", {}).get("mean", 0)
                    inf_mean = sr.get("inference_passes", {}).get("mean", 0)
                    print(
                        f"  Subtask {st_name:<3s} {sr['successes']:>4d}/{args.episodes:<4d} "
                        f"{sr['success_rate'] * 100:>6.1f}%  {wall_mean:>6.1f}s  {inf_mean:>6.0f}"
                    )

        # ── Generate comparison chart ────────────────────────────────
        _plot_chunk_comparison(all_chunk_data, tasks, epoch_video_dir)

        # ── Write comparison JSON ─────────────────────────────────────
        comparison_stats = {
            "checkpoint": str(args.checkpoint),
            "weights": args.weights,
            "episodes_per_subtask": args.episodes,
            "seed": args.seed,
            "chunk_sizes": {str(cs): val for cs, val in all_chunk_data.items()},
            "best_chunk_by_subtask": {
                task: min(
                    comparison_chunks,
                    key=lambda candidate: (
                        -all_chunk_data[candidate]["subtasks"][task]["success_rate"],
                        all_chunk_data[candidate]["subtasks"][task]["wall_time"]["mean"],
                        candidate,
                    ),
                )
                for task in tasks
            },
            "best_overall_chunk": min(
                comparison_chunks,
                key=lambda candidate: (
                    -all_chunk_data[candidate]["overall_successes"],
                    candidate,
                ),
            ),
        }
        comp_path = epoch_video_dir / "chunk_comparison.json"
        comp_path.parent.mkdir(parents=True, exist_ok=True)
        with open(comp_path, "w") as f:
            json.dump(comparison_stats, f, indent=2)
        print(f"  Comparison stats saved → {comp_path}")
        print("  Coarse labels:")
        for task, best_chunk in comparison_stats["best_chunk_by_subtask"].items():
            print(f"    {task}={best_chunk}")

    else:
        # ── Single chunk evaluation ──────────────────────────────────
        result = _eval_subtasks(
            policy,
            args.device,
            tasks,
            args.episodes,
            args.max_steps,
            no_video=args.no_video,
            all_videos=args.all_videos,
            debug_init=args.debug_init,
            video_out_dir=epoch_video_dir,
            seed=args.seed,
            rnd_obs=args.rnd_obs,
            feature_collector=feature_collector,
        )
        all_subtask_results = result["subtasks"]
        overall_successes = result["overall_successes"]
        overall_total = result["overall_total"]

        # ── Final summary ────────────────────────────────────────────
        if not args.no_summary:
            sorted_tasks = sorted(tasks, key=lambda t: int(t))
            print("\n" + "=" * 64)
            print("  Summary")
            print("=" * 64)
            print(
                f"  {'Subtask':<10s} {'Success':>10s} {'Rate':>8s} {'Avg steps':>10s} {'Min':>6s} {'Max':>6s} {'Inf(#)':>8s}"
            )
            print(f"  {'─' * 10} {'─' * 10} {'─' * 8} {'─' * 10} {'─' * 6} {'─' * 6} {'─' * 8}")
            for st_name in sorted_tasks:
                r = all_subtask_results[st_name]
                inf_mean = r.get("inference_passes", {}).get("mean", 0)
                print(
                    f"  Subtask {st_name:<3s} {r['successes']:>4d}/{args.episodes:<4d} "
                    f"{r['success_rate'] * 100:>6.1f}%  {r['steps']['mean']:>8.0f}  "
                    f"{r['steps']['min']:>4d}  {r['steps']['max']:>4d}  {inf_mean:>6.0f}"
                )
            print(f"  {'─' * 10} {'─' * 10} {'─' * 8} {'─' * 10} {'─' * 6} {'─' * 6} {'─' * 8}")
            overall_rate = 100.0 * overall_successes / overall_total if overall_total else 0
            print(
                f"  {'OVERALL':<14s} {overall_successes:>4d}/{overall_total:<4d} {overall_rate:>6.1f}%"
            )
            print()

        # ── Write stats JSON ─────────────────────────────────────────
        stats = {
            "checkpoint": str(args.checkpoint),
            "weights": args.weights,
            "episodes_per_subtask": args.episodes,
            "chunk_size": chunk_size,
            "seed": args.seed,
            "subtasks": all_subtask_results,
            "overall": {
                "successes": overall_successes,
                "total": overall_total,
                "rate": round(float(overall_successes) / overall_total, 4) if overall_total else 0,
            },
        }
        stats_path = epoch_video_dir / "subtask_stats.json"
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  Stats saved → {stats_path}")

    if feature_collector is not None:
        feature_collector.close()
        print(f"  Feature dataset saved → {args.feature_dataset.resolve()}")


if __name__ == "__main__":
    main()

"""Evaluate a trained ARP policy in the PushBox environment.

Usage:
    python scripts/eval_policy.py <checkpoint_dir> [--episodes 20] [--save-video] [--video-dir videos]
    python scripts/eval_policy.py <checkpoint_dir> --subtasks approach  # single subtask
    python scripts/eval_policy.py <checkpoint_dir> --subtasks all      # all subtasks (default)

    The checkpoint dir should contain files like:
      epoch=0100-val_loss=0.123.ckpt  (policy weights)
      latest.ckpt                     (or latest checkpoint)

    Each subtask runs --episodes independent evaluations with its own
    box initialization region for decoupled assessment.

    Also loads the normalizer state from the checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import hydra

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pushbox import arp
from pushbox.diffusion_policy.model.common.normalizer import LinearNormalizer
from pushbox.policy import ACTION_DIM, AGENT_STATE_DIM, PushBoxARPolicy
from chunk_selector.chunk_selector import ChunkSelector

# ── Env helpers ────────────────────────────────────────────────────────────
POLICY_IMG_SIZE = 96  # must match training image_shape (96x96)
RECORD_IMG_SIZE = 256  # higher res for video output
PHASE_NAMES = ("approach", "cross", "reach")


def capture_view(env, camera_name: str, size: int = RECORD_IMG_SIZE) -> np.ndarray:
    if (
        hasattr(env.sim, "_render_context_offscreen")
        and env.sim._render_context_offscreen is not None
    ):
        env.sim._render_context_offscreen.gl_ctx.make_current()
    frame = env.sim.render(width=size, height=size, camera_name=camera_name)
    frame = np.flipud(frame).astype(np.uint8)
    return frame


def get_agent_state(obs: dict) -> np.ndarray:
    joints = obs.get("robot0_joint_pos", np.zeros(7, dtype=np.float32))
    gripper = obs.get("robot0_gripper_qpos", np.zeros(2, dtype=np.float32))
    return np.concatenate([joints, gripper]).astype(np.float32)


def get_box_pos(obs: dict) -> np.ndarray:
    return obs.get("box_pos_xy", np.zeros(2, dtype=np.float32)).astype(np.float32)


# ── Checkpoint loading ─────────────────────────────────────────────────────


def _detect_checkpoint_config(ckpt_sd: dict) -> dict:
    """Inspect checkpoint state_dict and return pretrained + arp_cfg + tokens."""
    tk_embed_key = "policy.chunk_embedder.token_type_embed.weight"
    if tk_embed_key in ckpt_sd:
        num_tokens = ckpt_sd[tk_embed_key].shape[0]
    else:
        num_tokens = 4

    # Detect token dimensions
    token_dims = {}
    for i in range(num_tokens):
        embed_key = f"policy.token_embedders.{i}.embed.weight"
        if embed_key in ckpt_sd:
            token_dims[i] = ckpt_sd[embed_key].shape[1]
        else:
            token_dims[i] = 9

    # Infer architecture params from checkpoint shapes
    chunk_embed_w = ckpt_sd.get("policy.chunk_embedder.chunk_embed.weight")
    action_chunk_size = chunk_embed_w.shape[0] if chunk_embed_w is not None else 50
    n_embd = chunk_embed_w.shape[1] if chunk_embed_w is not None else 64

    # Infer num_latents from token predictor output size
    num_latents = 1
    for i in range(num_tokens):
        pred_key = f"policy.token_predictors.{i}.mlp.fc2.weight"
        if pred_key in ckpt_sd:
            out_dim = ckpt_sd[pred_key].shape[0]
            dim = token_dims.get(i, 9)
            # For GMM predictor: output = 2*L*dim + L (means + scales + logits)
            num_latents = out_dim // (2 * dim + 1)
            break

    # Infer plan_steps from pos_emb shape:
    # max_seq_len = (n_obs_steps + plan_steps + horizon) * 2
    # Assuming horizon = action_chunk_size (typical for pushbox)
    pos_emb = ckpt_sd.get("policy.pos_emb.weight")
    n_obs_steps = 2  # standard for pushbox
    if pos_emb is not None:
        max_seq_len = pos_emb.shape[0]
        horizon = action_chunk_size
        plan_steps = max_seq_len // 2 - n_obs_steps - horizon
    else:
        plan_steps = 5
        horizon = action_chunk_size

    # Token name mapping: 0=pos, 1=coarse-plan, 2=fine-action, 3=box_pos
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
            "is_control": (i == 0),  # only pos token is control
        }
        # non-control tokens need a predictor for generate() to work,
        # even if ckpt doesn't have matching weights (strict=False)
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
        f"[eval] detected: num_tokens={num_tokens}, n_embd={n_embd}, "
        f"num_layers={config['arp_cfg']['num_layers']}, plan_steps={plan_steps}, "
        f"action_chunk_size={action_chunk_size}, num_latents={num_latents}, "
        f"horizon={horizon}"
    )
    print(
        f"[eval] tokens: {['{}:{}D {}'.format(tk['name'], tk['dim'], 'ctrl' if tk.get('is_control') else 'pred') for tk in tokens]}"
    )
    config["tokens"] = tokens
    return config


def load_policy(
    checkpoint_dir: str,
    device: str = "cuda:0",
    weights: str = "model",
    use_checkpoint_config: bool = False,
):
    ckpt_dir = Path(checkpoint_dir)
    if ckpt_dir.is_file():
        ckpt_path = ckpt_dir
    else:
        ckpts = sorted(ckpt_dir.glob("*.ckpt"))
        if not ckpts:
            raise FileNotFoundError(f"No .ckpt files found in {ckpt_dir}")
        ckpt_path = ckpts[0]

    print(f"[eval] loading checkpoint: {ckpt_path}")

    # Load checkpoint first to detect config
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dicts" in ckpt:
        state_dicts = ckpt["state_dicts"]
        if weights == "ema" and "ema_model" in state_dicts:
            state_key = "ema_model"
        elif "model" in state_dicts:
            state_key = "model"
        else:
            raise KeyError(f"Checkpoint has no model weights: {sorted(state_dicts)}")
        sd = state_dicts[state_key]
        print(f"[eval] weights: {state_key}")
    elif "model" in ckpt:
        sd = ckpt["model"]
    else:
        sd = ckpt

    if use_checkpoint_config and ckpt.get("cfg") is not None:
        cfg = ckpt["cfg"]
        if "policy" not in cfg:
            raise KeyError("Checkpoint cfg has no policy section")
        policy = hydra.utils.instantiate(cfg.policy)
        print("[eval] policy config: checkpoint cfg.policy")
    else:
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

    # Extract and load normalizer separately
    norm_sd = {}
    for k, v in cleaned.items():
        if k.startswith("normalizer."):
            norm_sd[k[len("normalizer.") :]] = v
    normalizer = LinearNormalizer()
    normalizer.load_state_dict(norm_sd)
    policy.set_normalizer(normalizer)

    if use_checkpoint_config:
        # Exact checkpoint config should reproduce every parameter. Fail loudly
        # instead of silently evaluating a partially initialized policy.
        policy.load_state_dict(cleaned, strict=True)
        print(f"[eval] matched {len(cleaned)}/{len(policy.state_dict())} param keys (strict)")
    else:
        # Backward-compatible fallback for old PushBox checkpoints without a
        # usable Hydra policy config.
        policy_sd = {k: v for k, v in cleaned.items() if not k.startswith("normalizer.")}
        own = policy.state_dict()
        matched = {k: v for k, v in policy_sd.items() if k in own}
        print(f"[eval] matched {len(matched)}/{len(own)} param keys")
        own.update(matched)
        policy.load_state_dict(own, strict=False)
    policy.to(device)
    policy.eval()
    return policy


# ── Video saving ───────────────────────────────────────────────────────────


def save_video(frames: list, out_path: str, fps: int = 10):
    """Save a list of uint8 numpy frames (H, W, 3) as an mp4 video."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio

        writer = imageio.get_writer(out_path, fps=fps, format="FFMPEG", codec="libx264")
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        print(f"  video saved: {out_path}")
    except Exception as e:
        # Fallback: save frames as npz
        print(f"  video save failed ({e}), saving frames as npz instead")
        np.savez_compressed(out_path.replace(".mp4", ".npz"), frames=np.stack(frames))


# ── Evaluation loop ────────────────────────────────────────────────────────


def run_episode(
    env,
    policy,
    device,
    n_action_steps: int = 8,
    max_steps: int = 2000,
    record_video: bool = False,
    target_phase: int | None = None,
):
    """Run one episode, return (success, steps, phase_info, [video_frames]).

    If target_phase is given (0/1/2), success is scoped to that phase only
    rather than requiring all three phases.
    """
    import os as _os

    _os.environ.setdefault("MUJOCO_GL", "egl")
    _os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    _os.environ.setdefault(
        "__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
    )

    obs, grip_steps = env.init_with_grip()
    print(f"[init] auto-grip completed in {grip_steps} steps")

    top45_buf = []
    side_buf = []
    state_buf = []
    box_buf = []
    step = 0
    success = False
    done = False
    frames = []
    last_info = {}

    while step < max_steps:
        # Policy input: render at 96x96 dual cameras (must match training resolution)
        top45 = capture_view(env, "top45", size=POLICY_IMG_SIZE).astype(np.float32) / 255.0
        sideview = capture_view(env, "sideview", size=POLICY_IMG_SIZE).astype(np.float32) / 255.0

        if record_video:
            bird = capture_view(env, "birdview", size=RECORD_IMG_SIZE)
            front = capture_view(env, "frontview", size=RECORD_IMG_SIZE)
            frames.append(np.concatenate([bird, front], axis=1))

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
        actions = result["action"][0].cpu().numpy()

        for i in range(len(actions)):
            obs, reward, done, info = env.step(actions[i])
            last_info = info
            step += 1
            if info.get("arm_collision", False):
                done = True
                success = False
            elif done:
                if target_phase is not None:
                    phase_ok = info.get("phase_success", [False] * 3)
                    success = (
                        bool(phase_ok[target_phase]) if target_phase < len(phase_ok) else False
                    )
                else:
                    success = bool(info.get("success", False))
            if done:
                break

        if done:
            break

    phase_info = {
        "success": bool(last_info.get("success", success)),
        "all_phases_success": bool(last_info.get("all_phases_success", False)),
        "phase_success": [bool(x) for x in last_info.get("phase_success", [False] * 3)],
        "phase_steps": [int(x) for x in last_info.get("phase_steps", [0, 0, 0])],
        "current_phase": int(last_info.get("current_phase", 0)),
        "total_steps": step,
    }
    return success, step, phase_info, frames


def main():
    parser = argparse.ArgumentParser(description="Evaluate PushBox ARP policy")
    parser.add_argument("checkpoint", type=str, help="checkpoint dir or .ckpt file")
    parser.add_argument(
        "--episodes", type=int, default=20, help="number of eval episodes per subtask"
    )
    parser.add_argument("--max-steps", type=int, default=8000, help="max steps per episode")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--subtasks",
        choices=["1", "2", "3", "all"],
        default="all",
        help="which subtask(s) to evaluate (default: all)",
    )
    parser.add_argument("--save-video", action="store_true", help="save video for failed episodes")
    parser.add_argument(
        "--save-all-videos", action="store_true", help="save video for every episode"
    )
    parser.add_argument("--video-dir", type=str, default=None, help="output directory for videos")
    parser.add_argument(
        "--chunk-selector",
        type=Path,
        default=None,
        help="optional adaptive_chunk directory containing a trained selector sidecar",
    )
    args = parser.parse_args()

    policy = load_policy(args.checkpoint, device=args.device)
    if args.chunk_selector is not None:
        if not hasattr(policy, "set_chunk_selector"):
            parser.error("checkpoint policy does not support adaptive chunk selection")
        policy.set_chunk_selector(
            ChunkSelector.from_pretrained(args.chunk_selector, device=args.device)
        )
        print(f"[eval] adaptive chunk selector={args.chunk_selector.expanduser().resolve()}")

    # Must set EGL before env creation (robosuite init calls _reset_internal)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault(
        "__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
    )
    os.environ.pop("DISPLAY", None)
    from envs import PushBoxEnv

    # Resolve checkpoint path first
    ckpt_path = Path(args.checkpoint)
    if ckpt_path.is_file():
        run_dir = ckpt_path.parent.parent  # .../outputs/<date>/<time>/
    else:
        run_dir = ckpt_path.parent  # .../outputs/<date>/<time>/checkpoints/ → parent

    video_dir = Path(args.video_dir) if args.video_dir else run_dir / "eval_videos"
    if args.save_video or args.save_all_videos:
        video_dir.mkdir(parents=True, exist_ok=True)
        print(f"[eval] videos will be saved to: {video_dir}")

    ckpt_path_obj = Path(args.checkpoint)
    ckpt_file = (
        ckpt_path_obj
        if ckpt_path_obj.is_file()
        else ckpt_path_obj / sorted(ckpt_path_obj.glob("*.ckpt"))[0].name
    )
    ckpt_name = ckpt_file.name
    m = re.search(r"epoch=(\d+)", ckpt_name)
    epoch_num = int(m.group(1)) if m else 0

    epoch_video_dir = video_dir / f"epoch={epoch_num:04d}"
    if args.save_video or args.save_all_videos:
        epoch_video_dir.mkdir(parents=True, exist_ok=True)

    # ── Subtask definitions ──────────────────────────────────────────────
    # Each subtask has its own box-init-y range so the episode starts in the
    # correct phase region and the success condition is scoped to that phase.
    #
    # Phase thresholds (from env):
    #   PHASE_Y_APPROACH = -0.10   PHASE_Y_CROSS = 0.10
    #
    # Subtask 1 (phase 0):  box_y in [-0.25, -0.12]  →  succeed when y > -0.15
    # Subtask 2 (phase 1):  box_y in [-0.08, -0.02]  →  succeed when y >  0.10
    # Subtask 3 (phase 2):  box_y in [ 0.12,  0.18]  →  succeed when box at goal (0.0, 0.25)

    SUBTASK_DEFS = {
        "1": {"phase_idx": 0, "y_range": (-0.25, -0.12)},
        "2": {"phase_idx": 1, "y_range": (-0.08, -0.02)},
        "3": {"phase_idx": 2, "y_range": (0.12, 0.18)},
    }

    subtask_names = ["1", "2", "3"] if args.subtasks == "all" else [args.subtasks]

    all_subtask_results = {}

    for st_name in subtask_names:
        st_def = SUBTASK_DEFS[st_name]
        print(f"\n{'=' * 60}")
        print(f"  Subtask: {st_name}  (box init y ∈ {st_def['y_range']})")
        print(f"{'=' * 60}")

        env = PushBoxEnv(
            robots="Panda",
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=False,
            use_object_obs=True,
            reward_shaping=True,
            horizon=args.max_steps,
            hard_reset=False,
            camera_names="top45,sideview",
            camera_heights=256,
            camera_widths=256,
            box_init_y_range=st_def["y_range"],
        )

        successes = 0
        steps_list = []
        episode_phase_infos = []

        for i in range(args.episodes):
            record = args.save_all_videos or args.save_video
            success, steps, phase_info, frames = run_episode(
                env,
                policy,
                args.device,
                max_steps=args.max_steps,
                record_video=record,
                target_phase=st_def["phase_idx"],
            )
            successes += int(success)
            steps_list.append(steps)
            episode_phase_infos.append(phase_info)

            status = "SUCCESS" if success else "FAIL"
            print(f"  Ep {i + 1:3d}/{args.episodes}: {status:7s}  steps={steps:4d}")

            save_this = args.save_all_videos or (args.save_video and not success)
            if save_this and frames:
                sub_dir = epoch_video_dir / st_name
                sub_dir.mkdir(parents=True, exist_ok=True)
                out_path = str(sub_dir / f"ep{i + 1:03d}_{status}_{steps}steps.mp4")
                save_video(frames, out_path)

        env.close()

        rate = 100.0 * successes / args.episodes
        avg_steps = float(np.mean(steps_list)) if steps_list else 0.0
        print(f"\n  Subtask {st_name} results:")
        print(f"    Success rate: {successes}/{args.episodes} ({rate:.1f}%)")
        print(f"    Avg steps:    {avg_steps:.1f}")
        print(f"    Min steps:    {np.min(steps_list) if steps_list else 0}")
        print(f"    Max steps:    {np.max(steps_list) if steps_list else 0}")

        all_subtask_results[st_name] = {
            "successes": int(successes),
            "failures": args.episodes - int(successes),
            "success_rate": float(successes) / args.episodes,
            "steps": {
                "mean": float(avg_steps),
                "min": int(np.min(steps_list)) if steps_list else 0,
                "max": int(np.max(steps_list)) if steps_list else 0,
            },
            "episodes": episode_phase_infos,
        }

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Overall Summary")
    print("=" * 60)
    for st_name in subtask_names:
        r = all_subtask_results[st_name]
        print(
            f"  {st_name:10s}: {r['successes']}/{args.episodes} ({r['success_rate'] * 100:.1f}%)  avg_steps={r['steps']['mean']:.1f}"
        )

    stats = {
        "checkpoint": str(args.checkpoint),
        "num_episodes_per_subtask": args.episodes,
        "subtasks": all_subtask_results,
    }
    stats_path = epoch_video_dir / "eval_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  Stats saved to: {stats_path}")


if __name__ == "__main__":
    main()

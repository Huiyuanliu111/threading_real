"""Online eval runner for PushBox ARP training — runs policy in simulation and reports success rate."""
from __future__ import annotations

import numpy as np
import os as _os
import torch
from typing import Dict

from pushbox.diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from pushbox.diffusion_policy.policy.base_image_policy import BaseImagePolicy

RECORD_IMG_SIZE = 96


def _capture_cam(env, camera_name: str, size: int = RECORD_IMG_SIZE) -> np.ndarray:
    if hasattr(env.sim, "_render_context_offscreen") and env.sim._render_context_offscreen is not None:
        env.sim._render_context_offscreen.gl_ctx.make_current()
    frame = env.sim.render(width=size, height=size, camera_name=camera_name)
    return np.flipud(frame).astype(np.float32) / 255.0


def _get_agent_state(obs: dict) -> np.ndarray:
    joints = obs.get("robot0_joint_pos", np.zeros(7, dtype=np.float32))
    gripper = obs.get("robot0_gripper_qpos", np.zeros(2, dtype=np.float32))
    return np.concatenate([joints, gripper]).astype(np.float32)


def _get_box_pos(obs: dict) -> np.ndarray:
    return obs.get("box_pos_xy", np.zeros(2, dtype=np.float32)).astype(np.float32)


class PushBoxImageRunner(BaseImageRunner):
    def __init__(self, output_dir, n_eval_episodes: int = 10, max_steps: int = 500, **kwargs):
        super().__init__(output_dir)
        self.n_eval_episodes = n_eval_episodes
        self.max_steps = max_steps
        self._env = None

    def _get_env(self):
        if self._env is None:
            import sys as _sys
            from pathlib import Path as _Path
            _pushbox_root = str(_Path(__file__).resolve().parent.parent)
            if _pushbox_root not in _sys.path:
                _sys.path.insert(0, _pushbox_root)
            # Pop DISPLAY so mujoco uses pure EGL; DON'T restore (no pygame window)
            _os.environ.pop("DISPLAY", None)
            from envs import PushBoxEnv
            self._env = PushBoxEnv(
                robots="Panda",
                has_renderer=False,
                has_offscreen_renderer=True,  # required for sim.render()
                use_camera_obs=False,
                use_object_obs=True,
                reward_shaping=True,
                horizon=self.max_steps,
                hard_reset=False,
                camera_names="top45,sideview",
                camera_heights=256,
                camera_widths=256,
            )
        return self._env

    def run(self, policy: BaseImagePolicy) -> Dict:
        env = self._get_env()
        device = next(policy.parameters()).device
        n_action_steps = policy.n_action_steps

        successes = 0
        steps_list = []

        for _ in range(self.n_eval_episodes):
            obs, grip_steps = env.init_with_grip()

            top45_buf, side_buf, state_buf, box_buf = [], [], [], []
            step, done = 0, False

            while not done and step < self.max_steps:
                top45 = _capture_cam(env, "top45")
                sideview = _capture_cam(env, "sideview")
                state = _get_agent_state(obs)
                box_pos = _get_box_pos(obs)

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
                    result = policy.predict_action({
                        "top45": top45_t, "sideview": side_t,
                        "agent_pos": state_t, "box_pos": box_t
                    })
                actions = result["action"][0].cpu().numpy()

                for i in range(len(actions)):
                    obs, _, done, info = env.step(actions[i])
                    step += 1
                    if done:
                        if info.get("success", False):
                            successes += 1
                        break

            steps_list.append(step)

        return {
            "test_mean_score": successes / max(self.n_eval_episodes, 1),
            "test_success_rate": successes / max(self.n_eval_episodes, 1),
            "test_avg_steps": float(np.mean(steps_list)),
        }

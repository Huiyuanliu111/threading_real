"""Quick smoke-test: step the PushBoxEnv with random actions and print diagnostics."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from envs import PushBoxEnv


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="open MuJoCo viewer window")
    args = parser.parse_args()

    env = PushBoxEnv(
        robots="Panda",
        has_renderer=args.render,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        reward_shaping=True,
        horizon=200,
        hard_reset=True,
        render_camera="frontview",
    )

    print("Action space dim:", env.action_spec[0].shape)

    for ep in range(3):
        obs = env.reset()
        total_reward = 0.0
        done = False
        step = 0

        while not done:
            action = np.random.uniform(*env.action_spec, size=env.action_spec[0].shape)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            step += 1

        box_xy = obs["box_pos_xy"]
        print(
            f"Episode {ep+1:2d} | steps={step:4d} | return={total_reward:7.3f} | "
            f"box_xy=[{box_xy[0]:.3f}, {box_xy[1]:.3f}] | "
            f"success={info['success']} arm_col={info['arm_collision']} oob={info['out_of_bounds']}"
        )

    env.close()


if __name__ == "__main__":
    main()

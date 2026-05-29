import argparse
import ale_py
import cv2
import gymnasium as gym
import numpy as np

import stable_worldmodel as swm
from stable_worldmodel.data.formats.lance import LanceWriter

gym.register_envs(ale_py)

parser = argparse.ArgumentParser(description="Collect Pong offline dataset.")
parser.add_argument("--frames", type=int, default=50000)
args = parser.parse_args()

dataset_path = (
    swm.data.utils.get_cache_dir(sub_folder='datasets') / 'pong_offline.lance'
)
print(f"Saving to {dataset_path}")

env = gym.make("ALE/Pong-v5", render_mode="rgb_array")

current_frames = 0
ep = 0

print(f"Collecting {args.frames} frames...")

with LanceWriter(str(dataset_path), mode="overwrite") as writer:
    while current_frames < args.frames:
        obs, _ = env.reset()
        ep_pixels, ep_actions = [], []

        while current_frames < args.frames:
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)

            frame = cv2.resize(env.render(), (224, 224), interpolation=cv2.INTER_NEAREST)
            ep_pixels.append(frame)
            ep_actions.append(np.array([action], dtype=np.float32))
            current_frames += 1

            if terminated or truncated:
                break

        writer.write_episode({'pixels': ep_pixels, 'action': ep_actions})
        ep += 1

        if ep % 10 == 0 or current_frames >= args.frames:
            print(f"  {current_frames}/{args.frames} frames, {ep} episodes")

env.close()
print(f"Done! Dataset saved to {dataset_path}")

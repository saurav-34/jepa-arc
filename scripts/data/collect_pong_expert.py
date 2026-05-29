import argparse
import os

import ale_py
import cv2
import gymnasium as gym
import lance
import numpy as np
import pyarrow as pa
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

gym.register_envs(ale_py)

parser = argparse.ArgumentParser()
parser.add_argument("--frames", type=int, default=50000)
parser.add_argument("--train-steps", type=int, default=1_000_000,
                    help="Timesteps to train a local agent when HF download fails")
args = parser.parse_args()

LOCAL_CHECKPOINT = os.path.expanduser("~/.stable_worldmodel/ppo-ALE-Pong-v5.zip")


def load_model():
    if os.path.exists(LOCAL_CHECKPOINT):
        print(f"Loading local PPO agent from {LOCAL_CHECKPOINT}")
        return PPO.load(LOCAL_CHECKPOINT)

    try:
        from huggingface_sb3 import load_from_hub
        print("Downloading pre-trained PPO agent for Pong from HuggingFace...")
        checkpoint = load_from_hub(
            repo_id="sb3/ppo-ALE-Pong-v5",
            filename="ppo-ALE-Pong-v5.zip",
        )
        model = PPO.load(checkpoint)
        os.makedirs(os.path.dirname(LOCAL_CHECKPOINT), exist_ok=True)
        model.save(LOCAL_CHECKPOINT)
        return model
    except Exception as e:
        print(f"HuggingFace download failed ({e.__class__.__name__}). Training local PPO agent...")

    # Train with a properly wrapped env so the policy obs space is correct
    train_env = make_atari_env("ALE/Pong-v5", n_envs=4)
    train_env = VecFrameStack(train_env, n_stack=4)
    model = PPO("CnnPolicy", train_env, verbose=1)
    model.learn(total_timesteps=args.train_steps)
    train_env.close()
    os.makedirs(os.path.dirname(LOCAL_CHECKPOINT), exist_ok=True)
    model.save(LOCAL_CHECKPOINT)
    print(f"Saved locally trained agent to {LOCAL_CHECKPOINT}")
    return model


model = load_model()

# The SB3 PPO Atari CnnPolicy expects obs from a frame-stacked Atari env
# (grayscale 84x84, 4-frame stack). We use make_atari_env + VecFrameStack
# to produce the right observations, and pull the RGB render from the
# underlying base env for saving.
vec_env = make_atari_env("ALE/Pong-v5", n_envs=1, env_kwargs={"render_mode": "rgb_array"})
vec_env = VecFrameStack(vec_env, n_stack=4)

dataset_path = "datasets/pong_expert.lance"
os.makedirs("datasets", exist_ok=True)

image_size = 224 * 224 * 3
schema = pa.schema([
    pa.field("episode", pa.int32()),
    pa.field("action", pa.int32()),
    pa.field("pixels", pa.list_(pa.uint8(), image_size)),
])

batches = []
current_frames = 0
ep = 0

print(f"Collecting {args.frames} expert frames...")

obs = vec_env.reset()

ep_actions, ep_pixels = [], []

while current_frames < args.frames:
    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, _ = vec_env.step(action)

    # Render from the base env (before Atari wrappers strip color/size)
    raw_frame = vec_env.envs[0].render()
    frame = cv2.resize(raw_frame, (224, 224), interpolation=cv2.INTER_NEAREST)

    ep_actions.append(int(action[0]))
    ep_pixels.append(frame.flatten())
    current_frames += 1

    if done[0]:
        batch = pa.RecordBatch.from_arrays([
            pa.array([ep] * len(ep_actions), type=pa.int32()),
            pa.array(ep_actions, type=pa.int32()),
            pa.array(ep_pixels, type=pa.list_(pa.uint8(), image_size)),
        ], schema=schema)
        batches.append(batch)
        ep += 1

        if ep % 5 == 0 or current_frames >= args.frames:
            print(f"  {current_frames}/{args.frames} frames, {ep} episodes")

        ep_actions, ep_pixels = [], []
        # VecEnv auto-resets; obs is already the new episode obs

# Flush any partial episode at the end
if ep_actions:
    batch = pa.RecordBatch.from_arrays([
        pa.array([ep] * len(ep_actions), type=pa.int32()),
        pa.array(ep_actions, type=pa.int32()),
        pa.array(ep_pixels, type=pa.list_(pa.uint8(), image_size)),
    ], schema=schema)
    batches.append(batch)

lance.write_dataset(batches, dataset_path, schema=schema, mode="overwrite")
print(f"Done! {lance.dataset(dataset_path).count_rows()} frames saved to {dataset_path}")

"""Collect Pong expert frames with game state labels from ALE RAM.

RAM addresses:
  49 = ball_x, 54 = ball_y, 51 = player_y (right), 50 = opp_y (left)
"""

import argparse
import io
import os
import random

import ale_py
import cv2
import gymnasium as gym
import lance
import numpy as np
import pyarrow as pa
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack


def encode_frame(frame: np.ndarray, jpeg_quality: int = 95) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(frame.astype(np.uint8)).save(buf, format='JPEG', quality=jpeg_quality)
    return buf.getvalue()

gym.register_envs(ale_py)

parser = argparse.ArgumentParser()
parser.add_argument("--frames", type=int, default=50000)
parser.add_argument("--out", type=str, default="datasets/pong_with_state.lance")
parser.add_argument("--train-steps", type=int, default=1_000_000)
parser.add_argument("--epsilon", type=float, default=0.30, help="30% chance to take a random action for better WM exploration")
args = parser.parse_args()

LOCAL_CHECKPOINT = os.path.expanduser("~/.stable_worldmodel/ppo-ALE-Pong-v5.zip")

IMG_SIZE = 224

def load_model():
    if os.path.exists(LOCAL_CHECKPOINT):
        print(f"Loading local PPO agent from {LOCAL_CHECKPOINT}")
        return PPO.load(LOCAL_CHECKPOINT)
    try:
        from huggingface_sb3 import load_from_hub
        print("Downloading pre-trained PPO agent from HuggingFace...")
        checkpoint = load_from_hub("sb3/ppo-ALE-Pong-v5", "ppo-ALE-Pong-v5.zip")
        model = PPO.load(checkpoint)
        os.makedirs(os.path.dirname(LOCAL_CHECKPOINT), exist_ok=True)
        model.save(LOCAL_CHECKPOINT)
        return model
    except Exception as e:
        print(f"HuggingFace download failed ({e}). Training local PPO agent...")

    train_env = make_atari_env("ALE/Pong-v5", n_envs=4)
    train_env = VecFrameStack(train_env, n_stack=4)
    model = PPO("CnnPolicy", train_env, verbose=1)
    model.learn(total_timesteps=args.train_steps)
    train_env.close()
    os.makedirs(os.path.dirname(LOCAL_CHECKPOINT), exist_ok=True)
    model.save(LOCAL_CHECKPOINT)
    return model


def get_state(ale):
    ram = ale.getRAM()
    return {
        'ball_x': int(ram[49]),
        'ball_y': int(ram[54]),
        'player_y': int(ram[51]),
        'opp_y': int(ram[50]),
    }

model = load_model()

vec_env = make_atari_env("ALE/Pong-v5", n_envs=1, env_kwargs={"render_mode": "rgb_array"})
vec_env = VecFrameStack(vec_env, n_stack=4)
ale = vec_env.envs[0].unwrapped.ale

schema = pa.schema([
    pa.field("episode_idx", pa.int32()),
    pa.field("step_idx", pa.int32()),
    pa.field("action", pa.int32()),
    pa.field("pixels", pa.binary()),
    pa.field("ball_x", pa.float32()),
    pa.field("ball_y", pa.float32()),
    pa.field("ball_vx", pa.float32()),
    pa.field("ball_vy", pa.float32()),
    pa.field("player_y", pa.float32()),
    pa.field("opp_y", pa.float32()),
])

os.makedirs(os.path.dirname(args.out), exist_ok=True)
batches = []
current_frames = 0
ep = 0

obs = vec_env.reset()
ep_actions, ep_pixels = [], []
ep_ball_x, ep_ball_y = [], []
ep_player_y, ep_opp_y = [], []

print(f"Collecting {args.frames} frames with {int((1-args.epsilon)*100)}/{int(args.epsilon*100)} expert/random split...")

while current_frames < args.frames:
    # 1. Capture the State BEFORE taking the action (Fixes the alignment bug)
    raw_frame = vec_env.envs[0].render()
    frame = cv2.resize(raw_frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    state = get_state(ale)

    # 2. Decide the Action (Epsilon-Greedy for better WM data)
    if random.random() < args.epsilon:
        action = vec_env.action_space.sample() # Take a random exploration step
    else:
        action_array, _ = model.predict(obs, deterministic=True)
        action = int(action_array[0])

    # 3. Save the aligned step data
    ep_actions.append(action)
    ep_pixels.append(encode_frame(frame))
    ep_ball_x.append(float(state['ball_x']))
    ep_ball_y.append(float(state['ball_y']))
    ep_player_y.append(float(state['player_y']))
    ep_opp_y.append(float(state['opp_y']))
    
    # 4. Advance the environment
    obs, _, done, _ = vec_env.step([action])
    current_frames += 1

    # 5. Process the episode when finished
    if done[0] or current_frames >= args.frames:
        # Calculate velocities ONLY at the end of the episode (Fixes the CPU bottleneck)
        MAX_BALL_SPEED = 50.0  # pixels/step; larger deltas = ball reset after a point scored (artifacts reach ~76–205)
        ep_vx = [0.0] + [ep_ball_x[i] - ep_ball_x[i-1] for i in range(1, len(ep_ball_x))]
        ep_vy = [0.0] + [ep_ball_y[i] - ep_ball_y[i-1] for i in range(1, len(ep_ball_y))]
        ep_vx = [v if abs(v) <= MAX_BALL_SPEED else 0.0 for v in ep_vx]
        ep_vy = [v if abs(v) <= MAX_BALL_SPEED else 0.0 for v in ep_vy]

        ep_len = len(ep_actions)
        batch = pa.RecordBatch.from_arrays([
            pa.array([ep] * ep_len, type=pa.int32()),
            pa.array(list(range(ep_len)), type=pa.int32()),
            pa.array(ep_actions, type=pa.int32()),
            pa.array(ep_pixels, type=pa.binary()),
            pa.array(ep_ball_x, type=pa.float32()),
            pa.array(ep_ball_y, type=pa.float32()),
            pa.array(ep_vx, type=pa.float32()),
            pa.array(ep_vy, type=pa.float32()),
            pa.array(ep_player_y, type=pa.float32()),
            pa.array(ep_opp_y, type=pa.float32()),
        ], schema=schema)
        
        batches.append(batch)
        ep += 1

        if ep % 5 == 0 or current_frames >= args.frames:
            print(f"  {current_frames}/{args.frames} frames, {ep} episodes")

        # Reset buffers
        ep_actions, ep_pixels = [], []
        ep_ball_x, ep_ball_y = [], []
        ep_player_y, ep_opp_y = [], []
        if not done[0]:  # VecEnv auto-resets on done; only reset manually on early exit
            obs = vec_env.reset()

lance.write_dataset(batches, args.out, schema=schema, mode="overwrite")
print(f"Done! {lance.dataset(args.out).count_rows()} frames saved to {args.out}")
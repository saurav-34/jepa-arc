import argparse
import ale_py
import gymnasium as gym
import pyarrow as pa
gym.register_envs(ale_py)
import lance
import cv2
import os

# 1. Set up the argument parser for easy testing
parser = argparse.ArgumentParser(description="Collect Pong offline dataset into Lance format.")
parser.add_argument("--frames", type=int, default=50000, help="Total number of frames to collect")
args = parser.parse_args()

TARGET_FRAMES = args.frames
current_frames = 0
ep = 0

env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
dataset_path = "datasets/pong_offline.lance"
os.makedirs("datasets", exist_ok=True)

# 2. FIX: Ensure image_size perfectly matches the 224x224 resize!
image_size = 224 * 224 * 3 
schema = pa.schema([
    pa.field("episode", pa.int32()),
    pa.field("action", pa.int32()),
    pa.field("pixels", pa.list_(pa.uint8(), image_size)), 
])

batches = []

print(f"Collecting exactly {TARGET_FRAMES} frames directly to {dataset_path}...")

while current_frames < TARGET_FRAMES:
    obs, _ = env.reset()
    ep_actions, ep_pixels = [], []
    
    # Run the episode until it terminates OR we hit our target frame count
    while current_frames < TARGET_FRAMES:
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        
        # Resize to exactly 224x224 with Nearest Neighbor to keep sharp edges
        frame = cv2.resize(env.render(), (224, 224), interpolation=cv2.INTER_NEAREST)
        
        ep_actions.append(action)
        ep_pixels.append(frame.flatten())
        current_frames += 1
        
        if terminated or truncated:
            break
            
    # Package the episode into PyArrow
    batch = pa.RecordBatch.from_arrays([
        pa.array([ep] * len(ep_actions), type=pa.int32()),
        pa.array(ep_actions, type=pa.int32()),
        pa.array(ep_pixels, type=pa.list_(pa.uint8(), image_size))
    ], schema=schema)
    
    batches.append(batch)
    ep += 1
    
    # Print progress every 10 episodes
    if ep % 10 == 0 or current_frames == TARGET_FRAMES:
        print(f"Collected {current_frames}/{TARGET_FRAMES} frames...")

# 3. Write everything to disk securely
lance.write_dataset(batches, dataset_path, schema=schema, mode="overwrite")

dataset = lance.dataset(dataset_path)
print(f"Collection complete! Successfully wrote {dataset.count_rows()} rows to Lance.")
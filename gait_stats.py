import argparse
from email import policy

import gymnasium as gym
import numpy as np
from common import load_policy

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--episodes", type=int, default=10)
args = parser.parse_args()

env = gym.make("Walker2d-v5")
policy = load_policy(args.model, "Walker2d-v5")

speeds, distances, lengths, efforts, falls = [], [], [], [], 0

for episode in range(args.episodes):
    obs, info = env.reset(seed=1000 + episode)
    steps, done = 0, False
    while not done:
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        speeds.append(info["x_velocity"])          # pillanatnyi sebesség előre (m/s)
        efforts.append(np.abs(action).mean())      # mennyire hajtja a motorokat (0..1)
        steps += 1
        done = terminated or truncated
    distances.append(info["x_position"])           # megtett távolság az epizód végén (m)
    lengths.append(steps)
    falls += int(terminated)

env.close()

print(f"Modell: {args.model}")
print(f"Átlagos sebesség:       {np.mean(speeds):.2f} m/s")
print(f"Átlagos megtett út:     {np.mean(distances):.1f} m")
print(f"Átlagos epizódhossz:    {np.mean(lengths):.0f} lépés ({np.mean(lengths) * 0.008:.1f} mp)")
print(f"Elesett:                {falls} / {args.episodes} epizódban")
print(f"Átlagos motorhasználat: {np.mean(efforts):.2f} (0 = pihen, 1 = teljes erő)")
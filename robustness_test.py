import argparse

import gymnasium as gym
import numpy as np

from common import load_policy
from wrappers import PushWrapper

parser = argparse.ArgumentParser()
parser.add_argument("--env", default="Walker2d-v5")
parser.add_argument("--model", required=True)
parser.add_argument("--forces", type=float, nargs="+", default=[0, 50, 100, 150, 200, 300, 400])
parser.add_argument("--episodes", type=int, default=30)
parser.add_argument("--interval", type=int, default=200)
parser.add_argument("--duration", type=int, default=5)
parser.add_argument("--side", type=int, default=0, choices=[-1, 0, 1], help="0 = véletlen, 1 = hátulról, -1 = elölről")
args = parser.parse_args()

policy = load_policy(args.model, args.env)
print(f"Modell: {args.model} | {args.episodes} epizód erőnként | lökés átlagosan {args.interval} lépésenként, {args.duration} lépésig")
print(f"{'Erő (N)':>8} {'Elesett':>10} {'Átl. jutalom':>13} {'Átl. hossz':>11} {'Lökés/epizód':>13}")

for force in args.forces:
    env = PushWrapper(gym.make(args.env), force=force, interval=args.interval, duration=args.duration, side=args.side)
    returns, lengths, falls, pushes = [], [], 0, []
    for episode in range(args.episodes):
        obs, info = env.reset(seed=episode)      # minden erőnél ugyanazok a kezdőállapotok és lökésidőpontok
        total, steps, done = 0.0, 0, False
        while not done:
            obs, reward, terminated, truncated, info = env.step(policy(obs))
            total += reward
            steps += 1
            done = terminated or truncated
        returns.append(total)
        lengths.append(steps)
        falls += int(terminated)
        pushes.append(info["push_count"])
    env.close()
    print(f"{force:8.0f} {falls:>6}/{args.episodes:<3} {np.mean(returns):13.0f} {np.mean(lengths):11.0f} {np.mean(pushes):13.1f}")
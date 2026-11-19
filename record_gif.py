import argparse
from email import policy
import os

import gymnasium as gym
import imageio
from common import load_policy

parser = argparse.ArgumentParser()
parser.add_argument("--env", default="Walker2d-v5")
parser.add_argument("--model", default=None, help="mentett modell; ha nincs, véletlen stratégia")
parser.add_argument("--out", default="videos/out.gif")
parser.add_argument("--steps", type=int, default=300, help="legfeljebb ennyi lépést vesz fel")
parser.add_argument("--camera", default=None, help="kamera neve; a Walker2d-nél 'track' követi a robotot")
args = parser.parse_args()


camera = {"camera_name": args.camera} if args.camera else {}
env = gym.make(args.env, render_mode="rgb_array", **camera)
policy = load_policy(args.model, args.env) if args.model else None

frames = []
obs, info = env.reset(seed=0)
for step in range(args.steps):
    if policy is None:
        action = env.action_space.sample()
    else:
        action = policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    frames.append(env.render())
    if terminated or truncated:
        obs, info = env.reset()

env.close()

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
# A GIF kb. 30 képkocka/mp legyen, valós sebességgel: ehhez ritkítjuk a képkockákat
render_fps = env.metadata["render_fps"]
skip = max(1, round(render_fps / 30))
frames = frames[::skip]
imageio.mimsave(args.out, frames, fps=render_fps / skip, loop=0)
print(f"Kész: {len(frames)} képkocka -> {args.out}")
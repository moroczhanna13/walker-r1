import argparse
from email import policy
import os

import gymnasium as gym
import imageio
from common import load_policy
from wrappers import PushWrapper

parser = argparse.ArgumentParser()
parser.add_argument("--env", default="Walker2d-v5")
parser.add_argument("--model", default=None, help="mentett modell; ha nincs, véletlen stratégia")
parser.add_argument("--out", default="videos/out.gif")
parser.add_argument("--steps", type=int, default=300, help="legfeljebb ennyi lépést vesz fel")
parser.add_argument("--camera", default=None, help="kamera neve; a Walker2d-nél 'track' követi a robotot")
parser.add_argument("--push", type=float, default=0, help="lökések ereje newtonban (0 = nincs)")
parser.add_argument("--interval", type=int, default=100, help="átlagosan hány lépésenként jöjjön lökés")
args = parser.parse_args()


camera = {"camera_name": args.camera} if args.camera else {}
env = gym.make(args.env, render_mode="rgb_array", **camera)
if args.push > 0:
    env = PushWrapper(env, force=args.push, interval=args.interval)
policy = load_policy(args.model, args.env) if args.model else None

frames = []
obs, info = env.reset(seed=0)
for step in range(args.steps):
    if policy is None:
        action = env.action_space.sample()
    else:
        action = policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    frame = env.render().copy()
    if info.get("pushing"):
        frame[:12, :] = [255, 0, 0]          # piros csík a kép tetején, amíg a lökés tart
    frames.append(frame)
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
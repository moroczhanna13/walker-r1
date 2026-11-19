import argparse
import time

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

parser = argparse.ArgumentParser()
parser.add_argument("--env", default="Walker2d-v5")
parser.add_argument("--timesteps", type=int, default=1_000_000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--name", default="walker_ppo", help="a futtatás neve: ez lesz a modell és a napló neve")
args = parser.parse_args()

env = gym.make(args.env)

model = PPO("MlpPolicy", env, verbose=1, seed=args.seed, tensorboard_log="logs")

# 100 ezer lépésenként elmenti az aktuális modellt a models/checkpoints mappába
checkpoints = CheckpointCallback(
    save_freq=100_000,
    save_path="models/checkpoints",
    name_prefix=args.name,
)

start = time.time()
model.learn(
    total_timesteps=args.timesteps,
    tb_log_name=args.name,
    callback=checkpoints,
    progress_bar=True,
)
minutes = (time.time() - start) / 60
print(f"Tanítás kész: {minutes:.1f} perc")

model.save(f"models/{args.name}")
print(f"Modell elmentve: models/{args.name}.zip")

env.close()
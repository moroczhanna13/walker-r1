import os

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Betölt egy mentett modellt, és visszaad egy függvényt: megfigyelés -> akció
# Ha a modell mellett van <modell>_vecnormalize.pkl fájl, a megfigyeléseket ugyanúgy normalizálja, ahogy a tanítás közben történt
def load_policy(model_path, env_id):
    if model_path.endswith(".zip"):
        model_path = model_path[:-4]
    model = PPO.load(model_path)

    stats_path = model_path + "_vecnormalize.pkl"
    if not os.path.exists(stats_path):
        return lambda obs: model.predict(obs, deterministic=True)[0]

    dummy_env = DummyVecEnv([lambda: gym.make(env_id)])
    normalizer = VecNormalize.load(stats_path, dummy_env)
    normalizer.training = False        # kiértékeléskor a statisztikák ne változzanak
    print(f"Normalizáló statisztikák betöltve: {stats_path}")
    return lambda obs: model.predict(normalizer.normalize_obs(obs), deterministic=True)[0]
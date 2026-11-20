import json
import os
import zipfile

import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Kiolvassa a mentett fájlból, hogy PPO vagy SAC modell-e, és azzal tölti be
def load_model(model_path):
    with zipfile.ZipFile(model_path + ".zip") as archive:
        policy_class = json.loads(archive.read("data"))["policy_class"]["__module__"]
    algo = SAC if ".sac." in policy_class else PPO
    return algo.load(model_path)


def load_policy(model_path, env_id):
    """Betölt egy mentett modellt, és visszaad egy függvényt: megfigyelés -> akció.

    Ha a modell mellett van <modell>_vecnormalize.pkl fájl, a megfigyeléseket
    ugyanúgy normalizálja, ahogy a tanítás közben történt.
    """
    if model_path.endswith(".zip"):
        model_path = model_path[:-4]
    model = load_model(model_path)

    stats_path = model_path + "_vecnormalize.pkl"
    if not os.path.exists(stats_path):
        return lambda obs: model.predict(obs, deterministic=True)[0]

    dummy_env = DummyVecEnv([lambda: gym.make(env_id)])
    normalizer = VecNormalize.load(stats_path, dummy_env)
    normalizer.training = False        # kiértékeléskor a statisztikák ne változzanak
    print(f"Normalizáló statisztikák betöltve: {stats_path}")
    return lambda obs: model.predict(normalizer.normalize_obs(obs), deterministic=True)[0]
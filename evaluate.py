import argparse
from email import policy

import gymnasium as gym
import numpy as np
from common import load_policy

#Lefuttat episodes db epizódot, és visszaadja az összjutalmakat és a hosszakat
#Ha model=None, véletlen akciókat használ (kiinduló érték)
def evaluate(env_id, model=None, episodes=10):
    env = gym.make(env_id)
    returns, lengths = [], []

    for episode in range(episodes):
        obs, info = env.reset(seed=1000 + episode)
        total_reward, steps, done = 0.0, 0, False

        while not done:
            if model is None:
                action = env.action_space.sample()
            else:
                action = policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        returns.append(total_reward)
        lengths.append(steps)

    env.close()
    return np.array(returns), np.array(lengths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Walker2d-v5")
    parser.add_argument("--model", default=None, help="mentett modell útvonala; ha nincs, véletlen stratégia")
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    policy = load_policy(args.model, args.env) if args.model else None
    returns, lengths = evaluate(args.env, policy, args.episodes)

    name = args.model if args.model else "véletlen stratégia"
    print(f"Környezet: {args.env} | Stratégia: {name} | Epizódok: {args.episodes}")
    print(f"Átlagos összjutalom: {returns.mean():.1f} +/- {returns.std():.1f}")
    print(f"Legrosszabb / legjobb epizód: {returns.min():.1f} / {returns.max():.1f}")
    print(f"Átlagos epizódhossz: {lengths.mean():.0f} lépés")
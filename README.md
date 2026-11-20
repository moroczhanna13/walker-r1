# Walker RL

Teaching a 2D biped (MuJoCo Walker2d) to walk with reinforcement learning.

**Status:** work in progress (step 5 of 8)

## Before and after

| Random policy | PPO, 1M steps | PPO, normalization |
| --- | --- | --- |
| ![Before](videos/before.gif) | ![After](videos/after.gif) | ![After normalization](videos/after_improved.gif) |

First: an untrained robot falls in under a second.
Second: after 1 million steps of PPO it moves forward at about 2.4 m/s, but still falls after a few seconds.
Third: using PPO and observation/reward normalization, it moves at 4.1 m/s, and rarely falls.

![SAC](videos/sac.gif)

Training with SAC (seed 0), it moves at 2,7 m/s.

## Setup

    python -m venv .venv
    pip install -r requirements.txt

## How to run

Train with PPO + normalization, with seed 0:

    python train.py --normalize --seed 0 --name ppo_s0

Train with SAC, 300k steps, with seed 0:

    python train.py --algo sac --timesteps 300000 --eval-freq 20000 --seed 0 --name sac_s0

Get a graph with the results:

    python plot_results.py

## Roadmap

- [x] Environment setup, random-policy baseline
- [x] First PPO training
- [x] Tuned PPO
- [x] PPO vs SAC
- [ ] Own experiment

## Results

Mean return over 10 evaluation episodes (deterministic policy).

| Policy | Return | Episode length | Speed | Falls |
| --- | --- | --- | --- | --- |
| Random policy | 0.1 +/- 5.2 | 19 steps | - | 10/10 |
| PPO, default settings, 1M steps | 1465.6 +/- 200.4 | 446 steps | 2.29 m/s | 10/10 |
| PPO + observation/reward normalization | 4950.0 +/- 991.0 | 960 steps |  4.12 m/s | 1/10 |
| PPO + normalization + 4 parallel envs | 5566.5 +/- 13.3 | 1000 steps | 4.57 m/s | 0/10 |
| PPO + normalization + RL Zoo hyperparameters | 4277.8 +/- 196.8 | 991 steps | 3.30 m/s | 0/10 |

![Learning curves](plots/week4_comparison.png)

Normalizing observations and rewards was the single most important change:
it tripled the return and eliminated falls. Tuned hyperparameters from RL Zoo
did not beat the defaults in this single-seed run.

### PPO vs SAC

![PPO vs SAC learning curves](plots/ppo_vs_sac.png)

Best-model return over 20 evaluation episodes, mean ± std across 3 seeds.

| Algorithm | Training steps | Wall-clock per run | Return (3 seeds) | Per-seed returns |
| --- | --- | --- | --- | --- |
| PPO + normalization | 1M | ~23 min | 5164 +/- 423 | 4950 / 4787 / 5754 |
| SAC | 300k | ~79 min | 2571 +/- 932 | 3305 / 3151 / 1256 |

- **Sample efficiency:** at 300k steps SAC reached more return than PPO.
- **Wall-clock:** PPO finished 1M steps in a quarter of the time SAC needed for 300k.
- **Variance:** seed-to-seed spread is large for both algorithms.

Limitations: only 3 seeds per algorithm, and SAC was trained for fewer steps,
so this is not a claim about which algorithm is better in general.
# Walker RL

Teaching a 2D biped (MuJoCo Walker2d) to walk with reinforcement learning.

**Status:** work in progress (week 1 of 8)

## Before training

![Random policy](videos/before.gif)

An untrained robot taking random actions falls over in under a second.

## Setup

    python -m venv .venv
    pip install -r requirements.txt

## Roadmap

- [x] Environment setup, random-policy baseline
- [ ] First PPO training
- [ ] Tuned PPO, PPO vs SAC
- [ ] Own experiment
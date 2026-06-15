#!/usr/bin/env python3
"""Quick eval of the latest vision checkpoint - 5 deterministic episodes."""
import sys, numpy as np
sys.path.insert(0, 'src')

from stable_baselines3 import PPO
from simulation.drone_gym_env import RacingEnv

print('Loading checkpoint...')
model = PPO.load('trained_models/rl_vision_policy_latest.zip')
print('Loaded OK.')

env = RacingEnv(substeps=4, use_camera_obs=True)
totals = []

for run in range(5):
    obs, _ = env.reset()
    gates = 0
    for _ in range(1500):
        action, _ = model.predict(obs, deterministic=True)
        obs, _rew, term, trunc, info = env.step(action)
        gates = info.get('gates_passed', gates)
        if term or trunc:
            break
    reason = info.get('reason', '?')
    totals.append(gates)
    print(f'  Run {run+1}: {gates} gates  ({reason})')

env.close()
print(f'\nAvg: {np.mean(totals):.2f}  Max: {max(totals)}  Min: {min(totals)}')

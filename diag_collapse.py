import sys; sys.path.insert(0,'src')
from simulation.drone_gym_env import RacingEnv
from stable_baselines3 import PPO
import numpy as np

model = PPO.load('trained_models/rl_vision_policy_latest.zip', device='cpu')
env = RacingEnv(use_camera_obs=True)
obs, _ = env.reset()

actions, gate_counts = [], []
for step in range(600):
    action, _ = model.predict(obs, deterministic=True)
    obs, rew, term, trunc, info = env.step(action)
    actions.append(action.copy())
    if term or trunc:
        g = info.get('gates_passed', 0)
        gate_counts.append(g)
        reason = info.get('reason', '?')
        print(f"ep{len(gate_counts):02d} step={step+1:4d}  gates={g}  reason={reason}")
        obs, _ = env.reset()
        if len(gate_counts) >= 5:
            break

arr = np.array(actions)
print()
print(f"Action means : vx={arr[:,0].mean():+.3f}  vy={arr[:,1].mean():+.3f}  vz={arr[:,2].mean():+.3f}  yaw={arr[:,3].mean():+.3f}")
print(f"Action stds  : vx={arr[:,0].std():.3f}  vy={arr[:,1].std():.3f}  vz={arr[:,2].std():.3f}  yaw={arr[:,3].std():.3f}")
print(f"vx range     : [{arr[:,0].min():+.3f}, {arr[:,0].max():+.3f}]")
print(f"Avg gates    : {np.mean(gate_counts):.2f}")
env.close()

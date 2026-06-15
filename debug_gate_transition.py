"""Debug the BC policy's behaviour around the gate-0 → gate-1 transition."""
import sys, os
sys.path.insert(0, 'src')
from stable_baselines3 import PPO
from simulation.drone_gym_env import RacingEnv
import numpy as np

m = PPO.load('trained_models/rl_vision_policy_latest', device='cpu')
env = RacingEnv(substeps=4, use_camera_obs=True)
obs, _ = env.reset()

last_gates = 0
gate0_step = None

print("step | gates | obs[0:5] (cam: bx,by,dist,vis,appr)           | act [vx,vy,vz,yaw] norm")
print("-" * 100)

for step in range(400):
    act, _ = m.predict(obs, deterministic=True)
    obs, _, term, trunc, info = env.step(act)
    new_gates = info['gates_passed']

    if new_gates != last_gates:
        gate0_step = step
        print(f"[step {step:4d}] >>> GATE {new_gates} PASSED <<<  act={[f'{a:+.4f}' for a in act]}")
        last_gates = new_gates

    # Print every step from 85 to 145 (gate-0 crossing + first 50 post-gate steps)
    if 85 <= step <= 145:
        obs_str = " ".join(f"{x:+.3f}" for x in obs[0:5])
        act_str = " ".join(f"{a:+.4f}" for a in act)
        phys = env._sim.physics.get_state()
        pos = phys['position']
        print(f"[step {step:4d}] gates={new_gates}  cam=[{obs_str}]  act=[{act_str}]  pos=[{pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f}]")

    if term or trunc:
        print(f"\n[step {step:4d}] TERMINATED  final_gates={new_gates}")
        break

env.close()

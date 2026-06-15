import sys; sys.path.insert(0,'src')
from simulation.racing_simulator import RacingSimulator
import numpy as np
sim = RacingSimulator(substeps=4)
sim.reset(seed=0)
for i, g in enumerate(sim.gates):
    print(f'Gate {i}: pos={[round(x,2) for x in g.position.tolist()]}  normal={[round(x,2) for x in g.normal.tolist()]}')
print()
# Also check what happens with camera obs at post-gate-0 positions
from simulation.drone_gym_env import RacingEnv
env = RacingEnv(substeps=4, use_camera_obs=True)
obs, _ = env.reset()
# Manually check gate positions from env
for i, g in enumerate(env._sim.gates):
    print(f'EnvGate {i}: pos={[round(x,2) for x in g.position.tolist()]}')
env.close()

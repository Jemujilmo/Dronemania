import sys; sys.path.insert(0, 'src')
from simulation.drone_gym_env import RacingEnv
import numpy as np

e = RacingEnv(substeps=4, use_camera_obs=True)
obs, _ = e.reset()

g0    = e._sim.gates[0]
state = e._sim.physics.get_state()
pos   = state['position']
print(f"Gate 0 pos : {g0.position}")
print(f"Spawn pos  : {pos}")
print(f"y_diff={abs(pos[1]-g0.position[1]):.4f}  z_diff={abs(pos[2]-g0.position[2]):.4f}  (both should be ~0)")

# Always-forward policy
step, gates = 0, 0
total_rew = 0.0
while True:
    act = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    obs, rew, term, trunc, info = e.step(act)
    step += 1
    total_rew += rew
    gates = info.get('gates_passed', gates)
    if gates >= 1 or term or trunc or step > 300:
        break
print(f"Always-forward: {gates} gate(s) in {step} steps  total_rew={total_rew:.2f}  (term={term} trunc={trunc})")
e.close()

import sys, numpy as np
sys.path.insert(0, 'src')
from simulation.drone_gym_env import RacingEnv

e = RacingEnv(substeps=4, use_camera_obs=True)
obs, _ = e.reset()

gates = e._sim.gates
spawn = e._sim.physics.get_state()['position']

print(f"Track: {len(gates)} gates  |  Spawn: [{spawn[0]:.1f}, {spawn[1]:.1f}, {spawn[2]:.1f}]")
print()
prev = spawn
for i, g in enumerate(gates):
    gap = np.linalg.norm(g.position - prev)
    print(f"  Gate {i}: [{g.position[0]:5.1f}, {g.position[1]:4.1f}, {g.position[2]:4.1f}]"
          f"  {g.width:.1f}x{g.height:.1f}m  gap={gap:.1f}m  {g.shape}")
    prev = g.position

# Test: naive forward-only pass
step, gates_passed = 0, 0
while True:
    act = np.array([1.0, 0.0, 0.0, 0.0])
    obs, rew, term, trunc, info = e.step(act)
    step += 1
    gates_passed = info.get('gates_passed', gates_passed)
    if term or trunc or step > 500:
        break

print(f"\nNaive forward: {gates_passed}/{len(e._sim.gates)} gates in {step} steps (term={term} trunc={trunc})")
e.close()

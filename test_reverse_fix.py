#!/usr/bin/env python3
"""Smoke test: backward-flight penalty + game-day min_forward_vx clamp."""
import sys, numpy as np
sys.path.insert(0, 'src')
from simulation.drone_gym_env import RacingEnv

# Training mode (no clamp)
e = RacingEnv(substeps=4, use_camera_obs=True, min_forward_vx=0.0)
o, _ = e.reset()
print(f'Training env OK  obs={o.shape}  spawn_pos={e._spawn_pos.round(2)}  min_fwd={e._min_forward_vx}')

# Step with backward action — should get NEGATIVE reward contribution from penalty
a_backward = np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32)
a_forward  = np.array([ 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
o, r_back, *_ = e.step(a_backward)
e.reset()
o, r_fwd,  *_ = e.step(a_forward)
print(f'Reward  forward={r_fwd:.4f}  backward={r_back:.4f}  gap={r_fwd - r_back:.4f}  (gap should be positive)')
assert r_fwd > r_back, 'FAIL: backward action should get lower reward than forward'
e.close()

# Game-day mode (min vx = 1.5)
e2 = RacingEnv(substeps=4, use_camera_obs=True, min_forward_vx=1.5)
o2, _ = e2.reset()
print(f'Game-day env OK  min_fwd={e2._min_forward_vx}')
o3, r, t, tr, info = e2.step(a_backward)
print(f'Game-day step with action[0]=-1 -> rew={r:.3f}  gates={info["gates_passed"]}')
e2.close()

# Anti-stall: verify 150-step cutoff fires when drone goes backward
e3 = RacingEnv(substeps=4, use_camera_obs=True)
e3.reset()
spawn_x = e3._spawn_pos[0]
# Force drone 4m behind spawn
e3._sim.physics.set_state(
    position=np.array([spawn_x - 4.0, 0.0, 2.0]),
    velocity=np.zeros(3), orientation=np.zeros(3), angular_velocity=np.zeros(3))
e3._ep_steps = 150
o, r, t, tr, info = e3.step(np.zeros(4, dtype=np.float32))
print(f'Anti-stall tier-1 truncated={tr}  (should be True)')
assert tr, 'FAIL: tier-1 anti-stall should fire at step 150 when 4m behind spawn'
e3.close()

print('\nAll checks PASSED')

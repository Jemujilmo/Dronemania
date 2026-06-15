"""Deep diagnostic: run 100 random-policy episodes, trace why gate 0 is never reached."""
import sys, numpy as np
sys.path.insert(0, 'src')
from simulation.drone_gym_env import RacingEnv

e = RacingEnv(substeps=4, use_camera_obs=True)
gate_passes = 0
end_reasons = {}
x_max_seen = []
x_at_end = []

for ep in range(100):
    obs, _ = e.reset()
    gate_x = e._sim.gates[0].position[0]
    spawn_x = e._spawn_pos[0]
    max_x = spawn_x
    for step in range(300):
        action = np.random.uniform(-1, 1, 4).astype(np.float32)
        obs, r, term, trunc, info = e.step(action)
        pos = e._sim.physics.get_state()['position']
        max_x = max(max_x, pos[0])
        if info.get('gates_passed', 0) > 0:
            gate_passes += 1
            break
        if term or trunc:
            reason = info.get('reason', 'antistall')
            end_reasons[reason] = end_reasons.get(reason, 0) + 1
            x_at_end.append(pos[0])
            break
    x_max_seen.append(max_x)

e.close()

print(f"Gate 0 passes in 100 random episodes: {gate_passes}")
print(f"Episode end reasons: {end_reasons}")
print(f"Max x ever reached (avg): {np.mean(x_max_seen):.2f}m  (gate is at {gate_x:.2f}m)")
print(f"Max x ever reached (max): {np.max(x_max_seen):.2f}m  spawn={spawn_x:.2f}m")
print(f"x at episode end (avg):   {np.mean(x_at_end):.2f}m")
print(f"Farthest reach: {max(x_max_seen):.2f}m  (needed {gate_x:.2f}m)")

# Quick check: with pure forward action, does gate register?
print("\n--- Pure forward test ---")
e2 = RacingEnv(substeps=4, use_camera_obs=True)
obs, _ = e2.reset()
print(f"Gate 0 pos: {e2._sim.gates[0].position.round(2)}")
print(f"Spawn pos:  {e2._spawn_pos.round(2)}")
print(f"Gate normal (yaw): {np.degrees(e2._sim.gates[0].orientation[2]):.1f} deg")
for step in range(100):
    obs, r, term, trunc, info = e2.step(np.array([1.,0.,0.,0.], dtype=np.float32))
    pos = e2._sim.physics.get_state()['position']
    if info.get('gates_passed', 0) > 0:
        print(f"  Gate 0 PASSED at step {step}!  pos={pos.round(2)}")
        break
    if step % 15 == 0:
        print(f"  step {step:3d}  x={pos[0]:.2f}  (gate_x={e2._sim.gates[0].position[0]:.2f})")
    if term or trunc:
        print(f"  Episode ended at {step}: {info}")
        break
e2.close()

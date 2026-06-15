import sys, numpy as np
sys.path.insert(0, 'src')
from simulation.drone_gym_env import RacingEnv

e = RacingEnv(substeps=4, use_camera_obs=True)

hover_rewards, fwd_rewards = [], []
for _ in range(10):
    e.reset()
    _, r, *_ = e.step(np.zeros(4, dtype=np.float32))
    hover_rewards.append(r)

for _ in range(10):
    e.reset()
    _, r, *_ = e.step(np.array([1., 0., 0., 0.], dtype=np.float32))
    fwd_rewards.append(r)

print(f"Hover reward/step:   {np.mean(hover_rewards):.4f}")
print(f"Forward reward/step: {np.mean(fwd_rewards):.4f}")
winner = "HOVER" if np.mean(hover_rewards) > np.mean(fwd_rewards) else "FORWARD"
print(f"PPO prefers: {winner}  (gap={abs(np.mean(hover_rewards)-np.mean(fwd_rewards)):.4f}/step)")

# Check what the camera obs looks like at spawn
e.reset()
obs = e._get_obs()
print(f"\nAt spawn: bearing_x={obs[0]:.3f}  bearing_y={obs[1]:.3f}  dist={obs[2]:.3f}  visibility={obs[3]:.3f}")
e.close()

# Simulate an episode to see reward-over-time for forward policy
e2 = RacingEnv(substeps=4, use_camera_obs=True)
obs, _ = e2.reset()
cum, steps_to_gate = 0.0, None
for step in range(300):
    action = np.array([1., 0., 0., 0.], dtype=np.float32)
    obs, r, term, trunc, info = e2.step(action)
    cum += r
    if info.get('gates_passed', 0) > 0 and steps_to_gate is None:
        steps_to_gate = step
        print(f"GATE 0 PASSED at step {step}!  cumulative_reward={cum:.2f}")
    if term or trunc:
        print(f"Episode ended at step {step}: reason={info.get('reason','?')}  gates={info['gates_passed']}  cum_rew={cum:.2f}")
        break
e2.close()

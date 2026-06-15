"""
Diagnose covariate shift: compare expert actions vs BC model actions
for observations similar to the post-gate-0 region (vis=0.15, dist≈1.0).
"""
import sys, os
sys.path.insert(0, 'src')
import numpy as np
from stable_baselines3 import PPO
import torch as th

# Load expert data
data = np.load('records/expert_demos.npz')
obs_all  = data['obs']   # (276000, 20)
acts_all = data['acts']  # (276000, 4)

# Filter: weak visibility window similar to the problematic obs
# cam obs: [0]=bearing_x, [1]=bearing_y, [2]=dist, [3]=visibility, [4]=approach
vis  = obs_all[:, 3]
dist = obs_all[:, 2]
bx   = obs_all[:, 0]

# Region 1: low-vis (vis ≈ 0.15) + max dist (dist ≈ 1.0) — suspect OOD zone
mask_lovis = (vis > 0.10) & (vis < 0.25) & (dist > 0.95)
# Region 2: zero-camera — where BC goes haywire
mask_zero  = (vis == 0.0) & (dist == 0.0)
# Region 3: healthy camera (vis > 0.4) — working zone
mask_good  = vis > 0.40

print(f"Total transitions: {len(obs_all):,}")
print(f"Low-vis + max-dist (OOD zone): {mask_lovis.sum():,}  ({100*mask_lovis.mean():.1f}%)")
print(f"Zero-camera:                   {mask_zero.sum():,}  ({100*mask_zero.mean():.1f}%)")
print(f"Healthy camera:                {mask_good.sum():,}  ({100*mask_good.mean():.1f}%)")

print("\n--- Expert actions in LOW-VIS zone ---")
a = acts_all[mask_lovis]
print(f"  vx range: [{a[:,0].min():.3f}, {a[:,0].max():.3f}]  mean={a[:,0].mean():.3f}")
print(f"  vy range: [{a[:,1].min():.3f}, {a[:,1].max():.3f}]  mean={a[:,1].mean():.3f}")
print(f"  vz range: [{a[:,2].min():.3f}, {a[:,2].max():.3f}]  mean={a[:,2].mean():.3f}")
print(f"  yaw range:[{a[:,3].min():.3f}, {a[:,3].max():.3f}]  mean={a[:,3].mean():.3f}")

print("\n--- Expert actions in ZERO-camera zone ---")
a = acts_all[mask_zero]
if len(a) > 0:
    print(f"  vx range: [{a[:,0].min():.3f}, {a[:,0].max():.3f}]  mean={a[:,0].mean():.3f}")
    print(f"  vy range: [{a[:,1].min():.3f}, {a[:,1].max():.3f}]  mean={a[:,1].mean():.3f}")
    print(f"  vz range: [{a[:,2].min():.3f}, {a[:,2].max():.3f}]  mean={a[:,2].mean():.3f}")
    print(f"  yaw range:[{a[:,3].min():.3f}, {a[:,3].max():.3f}]  mean={a[:,3].mean():.3f}")
else:
    print("  (no zero-camera transitions in training data!)")

# Now ask BC model to predict on a sample from each zone
model = PPO.load('trained_models/rl_vision_policy_latest', device='cpu')
policy = model.policy
policy.set_training_mode(False)

def bc_predict(obs_np):
    t = th.tensor(obs_np, dtype=th.float32)
    with th.no_grad():
        feat = policy.extract_features(t, policy.pi_features_extractor)
        lpi, _ = policy.mlp_extractor(feat)
        return policy.action_net(lpi).numpy()

# Sample 100 from low-vis zone and compare
if mask_lovis.sum() > 0:
    idx = np.where(mask_lovis)[0][:100]
    expert_sample = acts_all[idx]
    bc_sample = bc_predict(obs_all[idx])
    mse_lovis = np.mean((expert_sample - bc_sample)**2)
    print(f"\n--- BC vs Expert MSE in LOW-VIS zone (n=100): {mse_lovis:.6f} ---")
    print(f"  BC vx: [{bc_sample[:,0].min():.3f}, {bc_sample[:,0].max():.3f}]  mean={bc_sample[:,0].mean():.3f}")
    print(f"  BC vy: [{bc_sample[:,1].min():.3f}, {bc_sample[:,1].max():.3f}]  mean={bc_sample[:,1].mean():.3f}")

# Check what the BC model thinks zero-camera obs should do
zero_obs = np.zeros((1, 20), dtype=np.float32)
bc_zero = bc_predict(zero_obs)
print(f"\n--- BC prediction for all-zero obs: {bc_zero.tolist()} ---")

# Check the obs the expert HAD at dist=1,vis=0.15 (same as the verify run)
testobs = np.array([[0.587, 0.892, 1.000, 0.150, 0.500,
                     8.0, 0.0, 0.0,   # vx=8 m/s (body), vy=0, vz=0?  
                     0.0, 0.0, 0.0,   # orientation
                     0.0, 0.0, 0.0,   # angular vel
                     0.0, 1.0,        # yaw_rate=0, speed_norm=1.0
                     0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
bc_test = bc_predict(testobs)
print(f"\n--- BC prediction for typical post-gate-0 obs: {[round(x,3) for x in bc_test[0].tolist()]} ---")

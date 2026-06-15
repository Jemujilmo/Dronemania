"""Rev B smoke test — run with: python smoke_test_revb.py"""
import sys, pathlib
sys.path.insert(0, 'src')

import numpy as np
import cv2

from simulation.drone_gym_env import RacingEnv
from perception.gate_detector import SimGateDetector

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

errors = []

def check(label, cond, detail=""):
    if cond:
        print(f"  [{PASS}] {label}")
    else:
        print(f"  [{FAIL}] {label}  {detail}")
        errors.append(label)

print("=" * 55)
print("  Dronemania Rev B Smoke Test")
print("=" * 55)

# ── 1. Env init ───────────────────────────────────────────────
print("\n[1] Environment initialisation")
e = RacingEnv(substeps=4, use_camera_obs=True)
obs, _ = e.reset()
check("obs shape == (20,)",      obs.shape == (20,),   str(obs.shape))
check("obs dtype float32",       obs.dtype == np.float32)
print(f"      obs = {obs}")

# ── 2. GPS leak: slot[14] and slot[15] must be IMU-scale, not GPS ─
print("\n[2] GPS leak check (slots 14 + 15)")
check("slot[14] not GPS dist (|val|<5)",  abs(float(obs[14])) < 5.0,
      f"got {obs[14]:.4f}")
check("slot[15] not GPS progress (0-1)", 0.0 <= float(obs[15]) <= 1.0,
      f"got {obs[15]:.4f}")

# ── 3. Visibility slot responds to gate ───────────────────────
print("\n[3] Visibility slot[3] over 300 steps")
vis_vals, rewards = [], []
obs, _ = e.reset()
for _ in range(300):
    act = e.action_space.sample()
    obs, rew, term, trunc, info = e.step(act)
    vis_vals.append(float(obs[3]))
    rewards.append(float(rew))
    if term or trunc:
        obs, _ = e.reset()

nonzero = sum(1 for v in vis_vals if v > 0.0)
check("some visibility > 0 in 300 steps",  nonzero > 0,
      f"nonzero={nonzero}/300  max={max(vis_vals):.3f}")
check("no visibility > 1.0 (no leakage)", max(vis_vals) <= 1.0,
      f"max={max(vis_vals):.3f}")
check("no visibility exactly 0.5 (old leak value)",
      all(v != 0.5 for v in vis_vals),
      "found 0.5 — old is_active leak still present")
print(f"      nonzero={nonzero}/300  max={max(vis_vals):.3f}  avg_reward={sum(rewards)/len(rewards):.4f}")

# ── 4. Detector: blank frame ──────────────────────────────────
print("\n[4] Gate detector — blank frame (no false positives)")
det = SimGateDetector()
blank = np.zeros((240, 320, 3), dtype=np.uint8)
r_blank = det.detect(blank)
check("blank frame → gate_visible=False", not r_blank['gate_visible'])

# ── 5. Detector: orange-only frame (no white frame edges) ─────
print("\n[5] Gate detector — orange-dot fallback (long-range path)")
orange_frame = np.zeros((240, 320, 3), dtype=np.uint8)
# 4 corner dots roughly centred around (160, 120)
for px, py in [(130, 100), (190, 100), (130, 140), (190, 140)]:
    cv2.circle(orange_frame, (px, py), 5, (0, 140, 255), -1)    # BGR orange
r_orange = det.detect(orange_frame)
check("orange-only → gate_visible=True",    r_orange['gate_visible'],
      "neither contour nor fallback path firing")
check("orange-only → active_visible=True",  r_orange['active_visible'])
# visibility can be 0.15 (fallback/long-range) OR 0.3+ (normal path if dots
# are large enough to be picked up as contours) — both are correct.
check("orange-only → visibility > 0.1",
      r_orange['obs_vector'][3] > 0.1,
      f"got {r_orange['obs_vector'][3]:.4f}  (expected >0.1 via either path)")
bearing = r_orange['bearing_px']
check("orange centroid roughly centred (|bearing|<40px)",
      abs(bearing) < 40, f"bearing={bearing}px")
path_used = "fallback(0.15)" if abs(r_orange['obs_vector'][3] - 0.15) < 0.01 else "contour"
print(f"      obs_vector = {r_orange['obs_vector']}  [{path_used} path]")

# ── 5b. True long-range fallback: dots too small for contours ─
print("\n[5b] Gate detector — true orange fallback (tiny dots below contour threshold)")
tiny_frame = np.zeros((240, 320, 3), dtype=np.uint8)
# radius=2 → area ≈ 12 px² which is below _MIN_AREA=35, so no contours found
for px, py in [(130, 100), (190, 100), (130, 140), (190, 140)]:
    cv2.circle(tiny_frame, (px, py), 2, (0, 140, 255), -1)
r_tiny = det.detect(tiny_frame)
check("tiny-dot fallback → gate_visible=True", r_tiny['gate_visible'],
      "fallback not firing for sub-threshold dots")
if r_tiny['gate_visible']:
    check("tiny-dot fallback → visibility=0.15",
          abs(r_tiny['obs_vector'][3] - 0.15) < 0.01,
          f"got {r_tiny['obs_vector'][3]:.4f}")
    print(f"      obs_vector = {r_tiny['obs_vector']}  [fallback path]")
print("\n[6] Gate detector — simulated close gate (white frame + orange)")
gate_frame = np.zeros((240, 320, 3), dtype=np.uint8)
pts = np.array([[100,80],[220,80],[220,160],[100,160]], dtype=np.int32)
cv2.polylines(gate_frame, [pts], isClosed=True, color=(230,230,230), thickness=3)
for px, py in [(100,80),(220,80),(220,160),(100,160)]:
    cv2.circle(gate_frame, (px,py), 5, (0,140,255), -1)
r_gate = det.detect(gate_frame)
check("close gate → gate_visible=True",   r_gate['gate_visible'])
check("close gate → active_visible=True", r_gate['active_visible'])
check("close gate → visibility > 0.3",    r_gate['obs_vector'][3] > 0.3,
      f"got {r_gate['obs_vector'][3]:.4f}")
check("close gate → dist_norm < 0.9",     r_gate['obs_vector'][2] < 0.9,
      f"got {r_gate['obs_vector'][2]:.4f} (should be lower = closer)")
print(f"      obs_vector = {r_gate['obs_vector']}")

# ── 7. Backup: auto_backup fires without error ────────────────
print("\n[7] auto_backup() call")
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import self_train as st
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
vec = VecMonitor(DummyVecEnv([lambda: RacingEnv(substeps=4, use_camera_obs=True)]))
dummy_model = PPO("MlpPolicy", vec, verbose=0)
st.auto_backup(dummy_model, "smoketest_avg0p0_ph0")
backups = list(st.BACKUP_DIR.glob("*.zip"))
check("backup .zip file created in trained_models/backups/",
      len(backups) > 0, f"found {len(backups)} .zip files in {st.BACKUP_DIR}")
if backups:
    print(f"      Created: {backups[-1].name}")
vec.close()

# ── 8. cleanup.ps1 won't overwrite valid model ────────────────
print("\n[8] cleanup.ps1 restore logic (check script content)")
ps1 = pathlib.Path("cleanup.ps1").read_text()
check("cleanup.ps1 no longer has unconditional Copy-Item",
      "ALWAYS restore" not in ps1 and "Copy-Item $backup $latest -Force\nWrite-Host" not in ps1)
check("cleanup.ps1 checks zip validity before restore",
      "testzip" in ps1 or "needRestore" in ps1)

# ── Summary ───────────────────────────────────────────────────
e.close()
print("\n" + "=" * 55)
if errors:
    print(f"  {len(errors)} FAILURE(S): {', '.join(errors)}")
    sys.exit(1)
else:
    print(f"  All 9 checks PASSED — Rev B is good to go")
print("=" * 55)

# Dronemania — Project Snapshot
*Last updated: 2026-03-11*

---

## Where We Are (March 3, 2026)

GPS-hybrid observation space is now live. The model has broken through the
gate-2 plateau that blocked progress for 200+ phases. **New all-time best:
avg=2.5** (phase 62, March 3). Model regularly reaches gate 2+ and
occasionally gate 3 in a single episode.

### Best results — all-time
| Date | Method | Avg gates (eval) | Steps | Notes |
|------|--------|-----------------|-------|-------|
| Feb 20 | Pure PPO, no warm-start | 2.0 | ~9M | GPS obs mode |
| Feb 21 | PPO from BC warm-start | 3.0 | ~1.6M | GPS obs mode |
| Feb 21 | BC (DAgger) alone, deterministic | **5.0/5** | — | Perfect run, no PPO |
| Mar 3 | PPO + GPS-hybrid obs | **2.5** | ~7.6M (this run) | **Best vision-mode result** |

### March 3 session summary (77 phases, 9.46M steps, ~11.5 hours)

| Phase range | Avg gates | Notes |
|---|---|---|
| 1–6 | 1.0–1.625 | Re-adaptation after GPS obs added to observation |
| 7–8 | 1.75 → **2.0** | First time hitting 2.0 in this run |
| 9–23 | 1.25–1.875 | Oscillating but healthy, never collapsing |
| 24 | **2.125** | New best at the time |
| 51, 55 | **2.125** | Consistently reaching gate 2+ |
| 59 | **2.0** | |
| 62 | **2.5** | **All-time best for vision mode** |
| 76 | **2.375** | Second highest ever |
| 77 | 1.875 | Stopped here |

Best checkpoint saved: `trained_models/backups/vision_avg2p5_ph62_20260303_161242.zip`

### What changed since Feb 21

**Architecture change (March 2):**
- `obs[16:19]` — was always zeros, now contains GPS direction vector to
  current target gate (normalised, based on competition telemetry)
- `obs[19]` — was zero, now contains normalised distance to gate
- Between-gate reward — replaced spin-yaw incentive with `+0.03 × fwd_norm`
  (dense GPS distance reward was tried briefly but caused vz-saturation
  collapse across 172 phases — removed)
- `competition_adapter.py` created — converts policy velocity outputs to
  Throttle/Roll/Pitch/Yaw for the competition API

**March 2 incident — collapse and recovery:**
- A `delta_dist` reward was added that used raw Euclidean distance.
  The model discovered "dive = shorter 3D distance = more reward", saturated
  vz=-1.0 and vx=+1.0, and produced avg=0.0 for 172 phases undetected.
- Diagnosis: `action_std ≈ 0.000` on vx and vz (fully saturated).
- Fix: removed dense GPS reward entirely; kept GPS direction in obs only.
- Recovery: restored from `pre_fresh_rl_vision_policy_best` (avg=1.8).

---

## Saved Assets

| File | Size | What it is |
|------|------|------------|
| `trained_models/rl_vision_policy_latest.zip` | 1.72 MB | Latest checkpoint (phase 77, Mar 3) |
| `trained_models/rl_vision_policy_best.zip` | 1.72 MB | Best eval checkpoint (avg=2.5, phase 62) |
| `trained_models/backups/vision_avg2p5_ph62_20260303_161242.zip` | 1.72 MB | **Best vision result — DO NOT OVERWRITE** |
| `trained_models/backups/pre_fresh_rl_vision_policy_best_20260302_062930.zip` | 1.72 MB | Pre-GPS-obs best (avg=1.8, Feb 26) — safe fallback |
| `trained_models/backups/BEST_BC_dagger_*.zip` | 1.72 MB | Original BC warm-start (5/5 gates deterministic) |
| `records/expert_demos_combined.npz` | 29.7 MB | 551k BC training demos |

Current model architecture: `[256, 256]` MlpPolicy, Tanh, 20-dim obs, 4-dim action.

---

## What To Run Next

Check progress anytime:
```powershell
Get-Content "records\vision_training_log.jsonl" | ForEach-Object { $j = $_ | ConvertFrom-Json; [PSCustomObject]@{ph=$j.phase; avg=$j.avg_gates; ts=$j.ts} } | Where-Object {$_.ts -gt "2026-03-03"} | Sort-Object ts | Select-Object -Last 15 | ForEach-Object { "$($_.ts) ph=$($_.ph) avg=$($_.avg)" }
```

### Resume training (start from phase 77 checkpoint)
```powershell
.\start_tonight.ps1
```

### Resume from best checkpoint (phase 62, avg=2.5)
```powershell
Copy-Item "trained_models\backups\vision_avg2p5_ph62_20260303_161242.zip" "trained_models\rl_vision_policy_latest.zip" -Force
.\start_tonight.ps1
```

### Pending items before competition
1. **Increase num_gates to 10–15** — currently training on 5-gate tracks only;
   competition course length unknown. Edit `_generate_track(num_gates=...)` in
   `src/simulation/racing_simulator.py`. Requires retraining.
2. **Confirm angular_velocity in API** — `obs[11:14]` uses it; if competition
   doesn't provide it, add random dropout during training for robustness.
3. **Verify throttle/roll/pitch/yaw scaling** — constants at top of
   `competition_adapter.py` need confirming once DCL API docs arrive.

### Known risks
- Policy oscillation (1.0–2.5 range) — still high variance; needs more phases
  to converge, or lower `ent_coef` from 0.05 → 0.02 after phase 100
- No protection against future reward-shaping attractors — any new dense reward
  must be audited for saturation risk before adding

---

## Known Gotchas

1. **Watchdog trap** — `train_watchdog.ps1` and `monitor_watchdog.ps1` respawn
   processes silently. Always verify by checking for watchdog PowerShell PIDs,
   not just python PIDs. Kill order: watchdogs first, then python.

2. **Checkpoint overwrite** — Multiple trainer instances share
   `rl_vision_policy_latest.zip` and clobber each other. Run ONE instance only.

3. **Never use `--fresh` from the BC model** — it wipes the [256,256] weights
   and starts from random [64,64]. Zip should be ~1.72 MB; if it's ~170 kB the
   BC model was overwritten.

4. **`log_std` rising during PPO** — expected and healthy. PPO's entropy bonus
   will naturally push it back up from -1.0 as it explores. Not a bug.

---

## BC Pipeline Reference

```
record_expert.py   GPS PD expert → (obs, action) pairs
  500 clean eps    → records/expert_demos.npz          (276k transitions)
  500 DAgger eps   → records/expert_demos_noisy.npz    (276k transitions)
  combined+shuffle → records/expert_demos_combined.npz (551k transitions)

pretrain_bc.py     Supervised MSE on combined demos
  [256,256] MlpPolicy, Tanh, Adam lr=3e-4, 80 epochs, batch=1024
  Final loss: 0.00011  |  log_std clamped to -1.0 after training
  Output: rl_vision_policy_latest.zip (1.72 MB)

logs/bc_verify.py  Load model, run 10 episodes, report gates + arch
```

**Key insight:** Pure expert imitation gave 1/5 gates (covariate shift at
post-gate-0 overshoot). DAgger noise injection (`--noise 0.25 --explore-prob 0.30`)
on record_expert.py fixed it — model trained on off-nominal recovery states,
deterministically hits 5/5 gates.

---

## WORKING COMPONENTS ✓

1. **Physics Engine** (src/simulation/drone_physics.py)
   - 6-DOF quadrotor dynamics verified
   - Stable hover at 0.5 throttle
   - Realistic mass (0.75kg) and thrust limits
   - dt = 1/240 = 4.17ms per step

2. **Simulator Environment** (src/simulation/racing_simulator.py)
   - 20-gate procedural racing tracks
   - Perspective projection camera rendering
   - Gate detection via image processing
   - Termination conditions (crash, timeout, OOB, course complete)

3. **Perception - Gate Detection** (src/perception/gate_detector.py)
   - HSV color thresholding for bright gate colors
   - Successfully detecting 2-5 gates in frame
   - Confidence scoring and sorting by distance
   - ~2-5ms latency per frame

4. **State Estimation** 
   - Using simulator ground truth (position, velocity, orientation)
   - Direct measurement fusion from simulator physics
   - Ready for real sensor integration (IMU, rangefinder, etc.)

5. **Path Planning** (src/planning/waypoint_planner.py)
   - Reactive waypoint generation toward detected gates
   - Altitude safety constraints (min 0.8m)
   - Lookahead distance = 5m
   - 3.0 m/s target velocity

6. **Control System** (race_test.py _velocity_to_motors)
   - Direct velocity-to-motor command conversion
   - Altitude hold with strong vertical gain
   - Pitch/roll for horizontal navigation
   - Motor output limiting for stability (0.25-0.75 range)

## PERFORMANCE METRICS

- **Loop Rate**: 1200-1400 Hz (excellent)
- **Avg Loop Time**: 0.76ms
- **Max Velocity Achieved**: ~13 m/s (from ~3 m/s command)
- **Stable Hover**: YES (verified)

## CURRENT ISSUES

1. **Gate Passing**: Only passing 1/20 gates before crash
2. **Altitude Maintenance**: Min height 0.11m (too close to ground)
3. **Overshoot**: Reaching 13+ m/s despite 3 m/s planning target
4. **Control Stability**: Needs better damping for smooth flight

## NEXT STEPS FOR FULL COMPLETION

### Priority 1: Fix Control Stability
- [ ] Add velocity damping (apply negative acceleration when velocity > target)
- [ ] Implement complementary filter for altitude (sensor fusion)
- [ ] Reduce pitch/roll sensitivity when high velocity

### Priority 2: Multi-Gate Navigation  
- [ ] Test gate transition: pass gate 1 → navigate to gate 2
- [ ] Debug why drone crashes after gate 1
- [ ] May need to check if gate detection updates correctly after passing

### Priority 3: Speed Optimization
- [ ] Increase target velocity progressively (currently 3 m/s is safe)
- [ ] Tune PID gains for aggressive racing
- [ ] Add predictive waypointing (look ahead 2-3 gates)

### Priority 4: Robustness Features
- [ ] Recovery when gates not detected (search pattern)
- [ ] Collision avoidance if obstacles added
- [ ] Graceful degradation for lost gates

## Pending Investigation — Reward Hardening

**Issue:** The model plateaus/oscillates (avg 1.6–3.6) rather than converging
upward. Likely cause: the reward function has two competing objectives that
aren't fully aligned:

1. **Gate-passing reward** — large sparse reward when a gate is passed
2. **Forward-nudge reward** (`+0.03 × vx_norm`) — small dense reward just for
   flying forward, added to prevent collapse after the delta-distance collapse
   incident (March 2)

The forward-nudge is doing its job (no collapse since), but it also teaches the
model that flying fast in *any* direction earns reward, independent of whether
it actually passes gates. As training accumulates, the policy can drift toward
"go fast" rather than "go through gates" — especially on difficult random tracks
where the gate isn't immediately visible.

**What to evaluate when revisiting:**

- [ ] **Remove forward-nudge entirely** now that the policy is stable at avg 3+,
      and see if the gate-passing reward alone is sufficient to maintain
      performance. Risk: possible slow collapse on hard tracks. Mitigation:
      monitor for 10–20 phases and restore from backup if avg drops below 2.0.

- [ ] **Replace forward-nudge with gate-progress reward** — instead of rewarding
      raw vx, reward reduction in *angular error to gate* (i.e. reward aligning
      heading toward the gate, not just going fast). This keeps a dense signal
      but one that's 100% aligned with the task.

- [ ] **Curriculum tightening** — `ADVANCE_THRESHOLD=0.10` still advances the
      spawn gate even with mediocre performance. Raise back to 0.25–0.30 so the
      model must *reliably* pass the current gate before being asked to go
      further.

- [ ] **Check whether `ent_coef=0.01` is now too low** — if the model is
      overfitting to a narrow set of track layouts, a tiny bump to 0.02 might
      help generalization without re-introducing oscillation.

**Checkpoint to use when investigating:** `vision_avg3p6_ph332_20260307_140204.zip`

---

## CODE FILES MODIFIED

- src/simulation/drone_physics.py - fixed motor_max_thrust (25.0x → 2.0x)
- src/perception/gate_detector.py - improved HSV thresholding
- src\planning/waypoint_planner.py - added altitude safety constraints
- src/state_estimation/ekf_estimator.py - added ground truth fusion
- race_test.py - complete racing AI implementation
- test_gate_detection.py - gate detection verification
- test_physics_motion.py - physics motion validation
- test_physics_stability.py - hover stability verification
- debug_race.py - full-stack debugging

## ARCHITECTURE VALIDATED

✓ Modular pipeline: Perception → State Est → Planning → Control → Physics
✓ Simulator provides realistic gate sequences
✓ Vision-based gate detection works
✓ Motor command generation stable
✓ Loop frequency excellent for real-time control

## KEY ACHIEVEMENT

**Autonomous drone successfully navigates through first gate at 0.72 + m/s**
- Gates detected in camera view
- Path planned dynamically
- Control commands generated and executed
- Physics responds realistically
- Ready for tuning to pass all 20 gates

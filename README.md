# Dronemania — AI-GP Autonomous Drone Racing Stack

Autonomous drone racing system built for the **AI Grand Prix** competition. The stack navigates a drone through a sequence of 3D gates as fast as possible using a combination of a scripted PD baseline and a vision-based reinforcement learning policy.

---

## Competition Timeline

| Event | Opens |
|---|---|
| 1st Virtual Qualifier | May 2026 |
| 2nd Virtual Qualifier | June 2026 |
| Physical Qualifier | September 2026 |
| AI Grand Prix Final | November 2026 |

**Objective:** Complete all gates in the shortest time possible. Fully autonomous — no manual control, no hardware advantages. Python-based autonomy stack only.

---

## Current State (February 20, 2026)

### Models

| Model | File | Gates Cleared | Steps Trained | Notes |
|---|---|---|---|---|
| Scripted PD baseline | *(in-memory)* | 20/20 | — | ~214s full course, deterministic |
| RL GPS (state obs) | `rl_policy_latest.zip` | ~9/20 | ~1.28M | GPS + state vector obs |
| RL Vision — milestone | `rl_vision_policy_2026-02-20_3gates_1.97Msteps.zip` | **3/20** | ~1.97M | **Best saved weights** |
| RL Vision — resume from | `rl_vision_policy_2gates_checkpoint.zip` | 3/20 | ~1.97M | Alias of milestone above |

> **Resume training** from `rl_vision_policy_2gates_checkpoint.zip` — do **not** use `--fresh`.

### Training Progress (Vision Mode — session log)

| Phase | Total Steps | Avg Gates | Notes |
|---|---|---|---|
| 1–3 | 0–369k | 0 | Learning to stay airborne |
| 4 | 491k | 1 | First gate cleared |
| 6 | 638k | 1 | Holding 1 gate |
| 8–9 | 852k–958k | 0–1 | Variance — still finding gate 1 |
| 12 | 1.47M | 0 | Regression after timeout issues |
| **15** | **1.84M** | **3** | **Peak — 3/20 gates, all 5 eval eps** |
| 16 | 1.97M | 2 | Minor drop — checkpoint saved here |
| 1+ | 2.0M+ | ongoing | Overnight run in progress |

- Training speed: **~470 fps** with `fast_render=True` across 12 parallel envs

### Changes Made Feb 20

- **Gate timeout raised 20s → 40s** — model was being cut off mid-approach between gates; scripted PD averages ~10s/gate so RL needs 3–4x margin while learning
- **Episode cap raised 3000 → 6000 steps** — comment claimed "200s at 15Hz" but env actually runs at 60Hz; was silently capping episodes at 50s total
- **Grid monitor** (`monitor.py`) — default view is now 2×2 with 4 simultaneous instances; `--single` flag restores original cockpit view
- **Watchdog scripts** — `train_watchdog.ps1` + `monitor_watchdog.ps1` auto-restart on crash; `start_tonight.ps1` launches both

---

## Architecture

```
Camera (320x240, 110 deg FOV)
        |
        v
SimGateDetector          <- Canny edges + approxPolyDP shape matching
  Pentagon (even gates)    99.8% detection rate on 3D projected shapes
  Square   (odd gates)
        |
        v
  cam_vec [dx, dy, dist_n, visibility]
  (GPS compass fallback when gate not visible)
        |
        v
PPO Policy (MlpPolicy)   <- Stable Baselines3, 4x8192-step rollouts
  Obs: 4-dim camera vec
  Act: [vx, vy, vz, yaw_rate] (continuous)
        |
        v
_velocity_to_motors()    <- PD inner loop, 240Hz physics substeps (x4/step)
        |
        v
DronePhysics             <- Custom 240Hz rigid-body sim, full 6DOF
```

### Reward Shaping (Vision Mode)

| Signal | Value | Purpose |
|---|---|---|
| Gate pass | +80 x quality | Dominant learning signal |
| Crash | -25 | Low — don't over-penalise exploration |
| Survival | +0.03/step | Stay airborne |
| Time | -0.06/step | Net -0.03 when idle, incentivises speed |
| Bearing | +0.08 x (1-err) x visibility | Keep gate centred in FOV |
| Forward progress | +0.12 x vel·gate_dir | GPS-based always-on shaping |
| Course complete | +200 + max(0, 300-time) | Endgame bonus |
| Smoothness | -0.002 x angular_vel^2 | Penalise erratic rotation |

Anti-stall: episode truncated at step 500 if still at gate 0 and behind starting line.

---

## Project Structure

```
Dronemania/
 src/
    simulation/
       racing_simulator.py   # 20-gate track, camera render, gate logic
       drone_physics.py      # 240Hz 6DOF rigid-body physics
       drone_gym_env.py      # Gymnasium env, reward, obs, RL interface
       racing_env.py
    perception/
       gate_detector.py      # Shape-based detector (pentagon/square)
    control/
       pid_controller.py
       simple_pid_controller.py
    planning/
       waypoint_planner.py
    state_estimation/
        ekf_estimator.py
 trained_models/
    rl_policy_latest.zip                              # GPS RL model
    rl_vision_policy_2gates_checkpoint.zip            # Resume point (= milestone below)
    rl_vision_policy_2026-02-20_3gates_1.97Msteps.zip # <- dated milestone, best weights
 records/
    race_records.json        # Best times + history
    training_log.jsonl       # Per-phase eval results (GPS mode)
    vision_training_log.jsonl# Per-phase eval results (vision mode)
 self_train.py        # Infinite PPO training loop with curriculum + eval
 monitor.py           # Live monitor: 2x2 grid of 4 instances (--single for cockpit view)
 visualize_race.py    # Race replay, cockpit HUD, gate detection overlay
 train_watchdog.ps1   # Auto-restarts trainer on crash; logs to logs/watchdog_train.log
 monitor_watchdog.ps1 # Auto-restarts monitor on crash
 start_tonight.ps1    # One-click launcher: kills stale Python, opens both watchdog windows
 cleanup.ps1          # Nuclear option: kills ALL watchdog windows + Python, then relaunches clean
 config/
    autonomy.yaml
    training_config.yaml
 docker/
     Dockerfile    # (containerisation pending)
```

---

## Training

### Overnight / unattended training

Use the watchdog launcher — auto-restarts on any crash, survives VS Code being closed:
```powershell
.\start_tonight.ps1
```
This opens two detached terminal windows, plus the monitor spawns its render:
- **"Dronemania Trainer"** — text only, PPO rollout tables + progress bar (PowerShell window)
- **"Dronemania Monitor"** — text only, watchdog restart log (PowerShell window)
- **2× "Dronemania Monitor" OpenCV windows** — the 3D simulation renders opened by the monitor process
- **Windows PowerShell** — your VS Code / existing terminal stays open separately

You should see exactly **5 windows** total when running correctly: 1 trainer terminal + 1 monitor terminal + 2 simulation render windows + your shell.
Expected **Python process count: 2** (1 `self_train.py --vision` + 1 `monitor.py`) — verify with:
```powershell
Get-Process python | Measure-Object | Select-Object Count
```

Check progress without opening VS Code:
```powershell
Get-Content "records\vision_training_log.jsonl" | Select-Object -Last 5 | ForEach-Object { $j = $_ | ConvertFrom-Json; "$($j.ts) ph=$($j.phase) steps=$($j.total_steps) avg=$($j.avg_gates) $($j.trend)" }
```

### Start / resume vision training (manual)
```bash
python self_train.py --vision
```

### Fresh restart (deletes existing vision checkpoint)
```bash
python self_train.py --vision --fresh
```

### GPS state-obs training
```bash
python self_train.py
```

### Live monitor (separate terminal — auto-reloads on new checkpoints)
```bash
python monitor.py           # 2x2 grid: 4 instances running simultaneously
python monitor.py --single  # Original single cockpit view
python monitor.py --gps     # Monitor GPS model instead
```
Keys: `R` = force reload, `Space` = pause, `N` = reset all, `Q`/`Esc` = quit

### Visualise a saved race
```bash
python visualize_race.py
```

---

## Simulation Details

| Parameter | Value |
|---|---|
| Physics rate | 240 Hz |
| RL decision rate | 60 Hz (4 substeps/action) |
| Camera resolution | 320 x 240 |
| FOV | 110 deg diagonal |
| Gate size | 1.8 m x 1.8 m |
| Gate shapes | Pentagon (even), Square (odd) |
| Course | 20 gates, 3D Grand Prix layout, ~345 m |
| Gate spacing | ~15-20 m with lateral/vertical variation |
| Training speed | ~470 fps (12 envs, fast_render mode) |
| Gate timeout | 40s per gate (RL training) |
| Episode cap | 6000 steps (~100s) |

---

## Dependencies

```
Python             3.13.7
stable-baselines3  2.7.1
gymnasium          1.2.3
torch              2.10.0+cpu
opencv-python      4.x
numpy
```

Install:
```bash
pip install -r requirements.txt
```

---

## Roadmap

- [x] Custom 240Hz physics simulation
- [x] 20-gate 3D Grand Prix track generation
- [x] Scripted PD baseline  20/20 gates (~214s)
- [x] Shape-based gate detector  99.8% detection rate
- [x] PPO reinforcement learning  GPS observation mode (~9/20 gates)
- [x] PPO reinforcement learning  Vision (camera) observation mode
- [x] GPS compass fallback when gate not in FOV
- [x] fast_render training mode (~470 fps)
- [x] Live training monitor with cockpit view
- [x] Watchdog scripts for unattended overnight training
- [x] Vision RL clears **3/20 gates** consistently (Feb 20 milestone)
## This has been a major hangup point
- [ ] Vision RL clears full 20-gate course
- [ ] Beat scripted PD lap time
- [ ] Docker containerisation
- [ ] Qualifier submission package

---

## Architecture Philosophy

**NOT:** Image -> Black Box Neural Network -> Commands

**YES:** Modular perception -> state estimation -> planning -> control pipeline; RL as the high-level policy

Priorities:
- **Generalization** over memorization - randomized track layout per episode
- **Sim-to-real transfer** - standardized interfaces, no simulator-specific tricks
- **Fault tolerance** - GPS compass fallback when vision fails
- **Reproducibility** - deterministic startup, pinned dependencies



## Competition Notes (Round 1 - AI Grand Prix)

- Round 1 runs inside a **Windows downloadable app** built on DCL's platform
- Submit Python-based autonomy code only - no manual control, no hardware advantages
- Standardized 3D racecourse, defined gate sequence, **time-based scoring** (fastest valid run wins)
- Gates are standard DCL squares/pentagons - same geometry as the current sim
- Mid-tier PC with dedicated GPU recommended for the competition client
- Full code IP stays with the team; DCL gets a read-only license for competition duration only

---

## Troubleshooting (Windows)

### Multiple render windows / duplicate process pairs
**Cause:** `start_tonight.ps1` was run more than once, opening multiple watchdog window pairs that each independently restart Python, causing 2+ trainers and 2+ monitors competing over the same checkpoint file.

**Fix:**
1. **Manually close** all "Dronemania Trainer" and "Dronemania Monitor" PowerShell windows from the taskbar
2. Then run `cleanup.ps1` which kills remaining Python, validates checkpoint, and relaunches:
```powershell
powershell -ExecutionPolicy Bypass -File cleanup.ps1
```

Expected clean state: **2 Python processes** (1 trainer + 1 monitor). Verify:
```powershell
Get-Process python | Measure-Object | Select-Object Count
```

---

### Trainer exits with code 1 every phase (PermissionError on checkpoint save)
**Cause:** Windows holds file locks on `.zip` files. The monitor (or SB3 internals) keeps `rl_vision_policy_latest.zip` open while the trainer's `safe_save()` tries to atomically rename the temp file over it - `[WinError 32] file in use by another process`. This was the root cause of all "exit code 1" failures throughout the project.

**Fix (already applied in `self_train.py`):** `safe_save()` retries the rename up to 10x with 0.5s sleep. The message `[safe_save] destination locked, retrying...` in trainer output is normal. If all 10 retries fail the trainer crashes; close the monitor window and restart via `start_tonight.ps1`.

---

### Watchdog window closes immediately with no output

**Cause A - Non-ASCII characters in .ps1 file:** Em-dashes, curly quotes, etc. cause PowerShell 5 to throw a parse error on startup and exit silently.

**Diagnosis:**
```powershell
powershell.exe -ExecutionPolicy Bypass -File train_watchdog.ps1 2>&1 | Select-Object -First 5
```
If you see `Unexpected token`, open the file and replace any curly/special punctuation with plain ASCII equivalents (hyphens, straight quotes).

**Cause B - $Args is a reserved variable:** PowerShell's `$Args` is read-only. Scripts that set `$Args = @("self_train.py")` silently ignore the assignment; Python launches with no arguments, exits immediately (clean code 0), and the watchdog tight-loops without actually training.

**Fix (already applied):** Renamed to `$TrainArgs` / `$MonitorArgs` in both watchdog files.

---

### Training log not updating / model stuck at 0 gates for many phases
**Cause:** Gate timeout (previously 20s) cut episodes short before the model could reach gate 1. Separately, `MAX_EP_STEPS = 3000` was commented "200s @ 15Hz" but the env runs at 60Hz - actual cap was 50s total.

**Fix (already applied):**
- Gate timeout: `> 40.0` in `src/simulation/racing_simulator.py`
- Episode cap: `MAX_EP_STEPS = 6000` (~100s at 60Hz) in `src/simulation/drone_gym_env.py`

---

### Monitor shows blank/black cells at startup
**Expected behaviour.** The monitor sleeps 10s (configurable at top of `monitor_watchdog.ps1`) to let the trainer write its first checkpoint. Cells populate once `rl_vision_policy_latest.zip` exists and loads successfully.
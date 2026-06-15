# PyBullet Simulator Setup Guide

## Installation

### 1. Install Dependencies

```powershell
# Use the updated requirements file
pip install -r requirements-new.txt
```

**Key packages:**
- `pybullet` - Physics simulation engine
- `gymnasium` - Reinforcement learning environment standard
- `opencv-python` - Image processing (camera rendering)
- `simple-pid` - PID controller
- `filterpy` - Kalman filtering

### 2. Verify Installation

```powershell
python -c "import pybullet; print('PyBullet OK')"
python -c "import gymnasium; print('Gym OK')"
```

## Running the Simulator

### Quick Test (With Visualization)

```powershell
python test_runner.py --mode gui --episodes 1
```

This will:
1. Launch PyBullet with GUI window
2. Create a procedural racing track with 20 gates
3. Test the drone (currently moves forward with random actions)
4. Show FPS and episode completion

### Headless Mode (For Training/Benchmarking)

```powershell
python test_runner.py --mode headless --episodes 5
```

This runs without GUI visualization, faster for multiple runs.

## Project Structure

```
src/
├── simulation/
│   ├── sim_interface.py      ← PyBulletDroneEnv (core simulator)
│   ├── racing_env.py          ← Gymnasium wrapper
│   └── __init__.py
├── autonomy/
│   └── main.py                ← Integration point
├── perception/
├── control/
├── planning/
└── state_estimation/
```

## What's Implemented

### ✅ PyBullet Simulator (`sim_interface.py`)

- **Physics engine:** Realistic quadrotor dynamics
- **Procedural tracks:** Auto-generates race courses with gates
- **Camera simulation:** Forward-looking camera with realistic rendering
- **IMU simulation:** Accelerometer and gyroscope data
- **Gate detection:** Can detect when drone passes through gates
- **Collision detection:** Detects crashes

### ✅ Gymnasium Integration (`racing_env.py`)

- Standard RL environment interface
- Reward function optimized for racing (speed + gate completion)
- Observation: Camera image (240×320 RGB)
- Action: [vx, vy, vz, yaw_rate]

### ✅ Configuration (`config/autonomy.yaml`)

- Racing-optimized PID gains
- Aggressive control parameters
- Color-based gate detection settings

## What's Next

### Phase 1: Get Drone Following Gates Autonomously

Right now, the test_runner uses **random actions**. Next steps:

1. **Connect autonomy stack** - Replace random action with full pipeline:
   ```
   Camera → Gate Detection → State Estimation → Planning → Control
   ```

2. **Test gate detection** - Make sure perception finds gates in simulation

3. **Tune PID controller** - Get aggressive but stable response

### Phase 2: Measure Performance

Track these metrics as you improve:

```python
# Metrics to optimize
- Course completion time (seconds)
- Gates per run (should be 100%)
- Average velocity through gates
- Smoothness (low acceleration variance)
- Latency (perception→control time)
```

### Phase 3: Create Challenging Tracks

```python
# Already supported in PyBullet:
- Variable gate spacing
- Gate orientation variations
- Different track layouts
- Lighting changes (simulator can add shadows)
```

## Troubleshooting

### PyBullet window opens but is blank

**Issue:** Graphics mode problem
**Solution:** 
```powershell
# Use headless mode
python test_runner.py --mode headless
```

### "ImportError: No module named pybullet"

**Issue:** Package not installed
**Solution:**
```powershell
pip install pybullet numpy opencv-python --upgrade
```

### Simulation runs slowly

**Issue:** GUI rendering overhead
**Solution:** Use headless mode or lower simulation rate:
```yaml
# In config/autonomy.yaml
sim_rate: 120  # Reduce from 240
```

### Drone doesn't move

**Issue:** Test runner using dummy actions
**Solution:** Integrate full autonomy stack (next phase)

## Performance Baseline

Current test_runner baseline (random action):
- FPS: 30-60 (depends on hardware)
- Course complete time: ~30-60 seconds
- Gates passed: 0-5 (random)

**Goal:** After integration:
- Course complete time: < 10 seconds
- Gates passed: 20/20 (100%)
- Consistent runs

## Reference

- [PyBullet Docs](https://docs.google.com/document/d/e/2PACX-1vQktvLQiL_TM1-PU-LNlMFNg8-4BFz2e_7MHtxFLnP8hDDPOuYs7hUXW6f0EvFxQX8TmWGZNp6VFGQG/pub)
- [Gymnasium Docs](https://gymnasium.farama.org/)
- [PID Controller Tuning](https://en.wikipedia.org/wiki/PID_controller#Tuning)

#!/usr/bin/env python3
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from simulation.drone_physics import DronePhysics
from simulation.racing_simulator import RacingSimulator

print("Testing physics engine...")
physics = DronePhysics()
print(f"Position: {physics.get_state()['position']}")

for i in range(50):
    physics.step(np.array([0.5, 0.5, 0.5, 0.5]))

print(f"After 50 steps: {physics.get_state()['position']}")

print("\nTesting racing simulator...")
sim = RacingSimulator({'sim_rate': 240})
sim.reset()
print(f"Gates: {len(sim.gates)}")
print(f"First gate: {sim.gates[0].position}")

print("\nRunning 100 simulation steps...")
start = time.time()
for i in range(100):
    action = np.array([2.0, 0.0, 0.0, 0.0])
    obs, info = sim.step(action)

elapsed = time.time() - start
hz = 100 / elapsed
print(f"Speed: {hz:.1f} Hz")
print(f"Gate idx: {sim.current_gate_idx}")
print(f"Drone pos: {sim.physics.get_state()['position']}")

print("\nSUCCESS: Physics engine working!")

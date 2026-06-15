#!/usr/bin/env python3
"""Test if physics engine actually moves the drone."""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from simulation.racing_simulator import RacingSimulator

config = {'sim_rate': 240}
sim = RacingSimulator(config)
sim.reset()

print("Testing physics engine motion...")
print(f"Initial position: {sim.physics.get_state()['position']}")

# Apply forward + upward thrust
for i in range(240):  # 1 second of simulation
    # Forward pitch: m1,m2 lower, m3,m4 higher
    motors = np.array([0.4, 0.4, 0.6, 0.6])  # Pitch forward
    sim.set_motor_commands(motors)
    
    if i % 48 == 0:  # Every 200ms
        pos = sim.physics.get_state()['position']
        vel = sim.physics.get_state()['velocity']
        print(f"  t={i/240:.2f}s: pos={pos}, vel={vel}")

final_pos = sim.physics.get_state()['position']
print(f"\nFinal position: {final_pos}")
print(f"Expected: should have moved forward (X > 0) and possibly up (Z > 1)")

#!/usr/bin/env python3
"""Test physics stability with 50% hover throttle."""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from simulation.racing_simulator import RacingSimulator

config = {'sim_rate': 240}
sim = RacingSimulator(config)
sim.reset()

print("Testing physics with hover throttle only...")
print(f"Initial: pos={sim.physics.get_state()['position']}, vel={sim.physics.get_state()['velocity']}")

# Hold at hover for 240 steps (1 second)
for i in range(480):  # 2 seconds
    # Exactly hover
    motors = np.array([0.5, 0.5, 0.5, 0.5])
    sim.set_motor_commands(motors)
    
    if i % 120 == 0:  # Every 500ms
        state = sim.physics.get_state()
        pos = state['position']
        vel = state['velocity']
        print(f"  t={i/240:.2f}s: pos={pos}, vel_mag={np.linalg.norm(vel):.3f}")

final_state = sim.physics.get_state()
print(f"\nFinal: pos={final_state['position']}, vel={final_state['velocity']}")
print(f"Expected: position should stay near [0, 0, 1], velocity near zero")

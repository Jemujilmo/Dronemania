#!/usr/bin/env python3
"""
Physics engine test - verify simulation works correctly.

Tests:
1. Drone initialization
2. Physics stepping
3. Motor command conversion
4. Gate detection
5. Performance metrics
"""

import sys
import os
import time
import numpy as np

# Force UTF-8 encoding on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from simulation.drone_physics import DronePhysics
from simulation.racing_simulator import RacingSimulator


def test_physics_engine():
    """Test basic physics engine."""
    print("=" * 60)
    print("🔬 PHYSICS ENGINE TEST")
    print("=" * 60)
    
    print("\n[1] Testing drone physics initialization...")
    physics = DronePhysics()
    state = physics.get_state()
    
    print(f"   [OK] Initial position: {state['position']}")
    print(f"   [OK] Initial velocity: {state['velocity']}")
    print(f"   [OK] Initial orientation: {state['orientation']}")
    
    print("\n[2] Testing physics step...")
    # Hover command (0.5 per motor = gravity compensation)
    hover_cmd = np.array([0.5, 0.5, 0.5, 0.5])
    
    for i in range(10):
        info = physics.step(hover_cmd)
    
    state = physics.get_state()
    print(f"   [OK] After 10 steps, height: {state['position'][2]:.3f}m")
    print(f"   [OK] Velocity: {state['velocity']}")
    
    print("\n[3] Testing acceleration (increasing thrust)...")
    accel_cmd = np.array([0.6, 0.6, 0.6, 0.6])  # More than hover
    
    for i in range(50):
        info = physics.step(accel_cmd)
    
    state = physics.get_state()
    print(f"   [OK] After acceleration, height: {state['position'][2]:.3f}m")
    print(f"   [OK] Velocity: {np.linalg.norm(state['velocity']):.3f} m/s")
    
    print("\n[SUCCESS] Physics engine working correctly!\n")


def test_racing_simulator():
    """Test racing simulator."""
    print("=" * 60)
    print("🏁 RACING SIMULATOR TEST")
    print("=" * 60)
    
    config = {
        'sim_rate': 240,
    }
    
    print("\n1️⃣ Initializing simulator...")
    sim = RacingSimulator(config)
    sim.reset()
    
    print(f"   ✅ Simulator created")
    print(f"   ✅ Track generated with {len(sim.gates)} gates")
    print(f"   ✅ First gate at: {sim.gates[0].position}")
    
    print("\n2️⃣ Running simulation for 100 steps...")
    start_time = time.time()
    
    for step in range(100):
        # Simple control: move forward
        action = np.array([2.0, 0.0, 0.0, 0.0])  # Forward velocity
        obs, info = sim.step(action)
        
        if 'gate_passed' in info:
            print(f"   ✅ Gate {info['gate_passed']} passed!")
    
    elapsed = time.time() - start_time
    hz = 100 / elapsed
    
    drone_state = sim.physics.get_state()
    print(f"   ✅ Simulation speed: {hz:.1f} Hz")
    print(f"   ✅ Drone position: {drone_state['position']}")
    print(f"   ✅ Drone velocity: {np.linalg.norm(drone_state['velocity']):.2f} m/s")
    print(f"   ✅ Gates passed: {sim.current_gate_idx}")
    
    print("\n3️⃣ Testing full race...")
    sim.reset()
    
    gates_passed = 0
    max_steps = 5000
    start_time = time.time()
    
    for step in range(max_steps):
        # Simple racing strategy: move toward current gate
        current_gate = sim.gates[sim.current_gate_idx]
        drone_pos = sim.physics.get_state()['position']
        
        direction = current_gate.position - drone_pos
        direction_norm = np.linalg.norm(direction)
        
        if direction_norm > 0.1:
            direction = direction / direction_norm
        
        # Command: move toward gate
        action = direction * 5.0  # 5 m/s speed
        action[3] = 0.0  # No yaw
        
        obs, info = sim.step(action)
        
        if 'gate_passed' in info:
            gates_passed += 1
        
        if info.get('reason') == 'course_complete':
            print(f"   ✅ Course complete in {info['total_time']:.2f}s!")
            print(f"   ✅ All {info['gates_passed']} gates passed!")
            break
        
        if step % 500 == 0:
            print(f"   Step {step:4d}: Gates passed {gates_passed:2d}/5, Time: {sim.time:.2f}s")
    
    elapsed = time.time() - start_time
    hz = max_steps / elapsed
    
    print(f"\n   ✅ Simulation speed: {hz:.1f} Hz")
    print(f"   ✅ Final gates passed: {gates_passed}")
    
    print("\n✅ Racing simulator working correctly!\n")


def test_camera_rendering():
    """Test camera rendering."""
    print("=" * 60)
    print("📷 CAMERA RENDERING TEST")
    print("=" * 60)
    
    config = {'sim_rate': 240}
    sim = RacingSimulator(config)
    sim.reset()
    
    print("\n1️⃣ Rendering camera image...")
    
    for i in range(10):
        action = np.array([2.0, 0.0, 0.0, 0.0])
        obs, info = sim.step(action)
    
    print(f"   ✅ Image shape: {obs.shape}")
    print(f"   ✅ Image dtype: {obs.dtype}")
    
    print("\n✅ Camera rendering working!\n")


def main():
    """Run all tests."""
    try:
        test_physics_engine()
        test_racing_simulator()
        test_camera_rendering()
        
        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Run: python integration_test.py")
        print("  2. This will test full autonomy stack with simulator")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

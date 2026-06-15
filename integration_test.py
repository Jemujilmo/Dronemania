#!/usr/bin/env python3
"""
Full integration test: Autonomy stack + PyBullet simulator

This script runs the complete pipeline:
    Camera → Gate Detection → State Estimation → Planning → Control → Simulation
"""

import sys
import os
import yaml
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from simulation.racing_env import DroneRacingEnv
from perception.gate_detector import GateDetector
from state_estimation.ekf_estimator import EKFEstimator
from planning.waypoint_planner import WaypointPlanner
from control.pid_controller import PIDController


def load_config(config_path: str) -> dict:
    """Load configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def visualize_perception(image: np.ndarray, detection: dict, 
                         step: int, save_path: str = None) -> np.ndarray:
    """Draw detection results on image for debugging."""
    import cv2
    
    vis = image.copy()
    
    # Draw detected gates
    if detection.get('next_gate'):
        gate = detection['next_gate']
        x, y, w, h = gate['bbox']
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
        con_text = f"Conf: {gate['confidence']:.2f}"
        cv2.putText(vis, con_text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (0, 255, 0), 1)
    
    # Draw info
    info_text = f"Step: {step} | Gates: {detection.get('num_detected', 0)}"
    cv2.putText(vis, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
               0.7, (255, 255, 255), 2)
    
    return vis


def run_integration_test(config_path: str = 'config/autonomy.yaml'):
    """Run full autonomy stack on simulator."""
    
    print("=" * 60)
    print("🚁 FULL INTEGRATION TEST")
    print("=" * 60)
    
    # Load config
    config = load_config(config_path)
    print(f"\n📋 Config loaded: {config_path}")
    
    # Create environment
    env = DroneRacingEnv(config, render_mode='human')
    print("✅ Environment created")
    
    # Initialize modules
    perception = GateDetector(config['perception'])
    state_estimator = EKFEstimator(config['state_estimation'])
    planner = WaypointPlanner(config['planning'])
    controller = PIDController(config['control'])
    print("✅ All modules initialized")
    
    # Reset and start
    print("\n🚀 Starting autonomy loop...")
    obs, info = env.reset()
    print(f"   Initial observation: {obs.shape}")
    
    episode_reward = 0
    step = 0
    max_steps = 5000
    gates_passed = 0
    start_time = time.time()
    step_times = []
    
    try:
        while step < max_steps:
            loop_start = time.time()
            
            # 1. Perception: Detect gates
            perception_out = perception.process(obs)
            
            # 2. State Estimation: Get state estimate
            # (In full version, would have real IMU data)
            state_estimate = state_estimator.get_state()
            
            # 3. Planning: Get target
            target = planner.plan(state_estimate, perception_out)
            
            # 4. Control: Compute command
            command = controller.compute(state_estimate, target, dt=0.01)
            
            # 5. Step simulator
            obs, reward, terminated, truncated, sim_info = env.step(command['velocity'])
            
            episode_reward += reward
            step += 1
            
            # Track performance
            step_time = time.time() - loop_start
            step_times.append(step_time)
            
            # Update gates passed
            if 'gate_passed' in sim_info:
                gates_passed += 1
                print(f"   ✅ Gate {gates_passed} passed!")
            
            # Log periodically
            if step % 500 == 0:
                avg_hz = 500 / sum(step_times[-500:]) if len(step_times) >= 500 else 0
                print(f"   Step {step:4d} | Gates: {gates_passed:2d} | " + 
                      f"Reward: {episode_reward:7.1f} | Hz: {avg_hz:.1f}")
            
            # Check termination
            if terminated or truncated:
                print(f"\n   Episode ended: {sim_info}")
                break
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    finally:
        elapsed = time.time() - start_time
        avg_hz = step / elapsed if elapsed > 0 else 0
        
        print("\n" + "=" * 60)
        print("📊 RESULTS")
        print("=" * 60)
        print(f"Total steps: {step}")
        print(f"Gates passed: {gates_passed}")
        print(f"Total reward: {episode_reward:.2f}")
        print(f"Elapsed time: {elapsed:.2f}s")
        print(f"Average Hz: {avg_hz:.1f}")
        print(f"Avg latency/step: {np.mean(step_times)*1000:.2f}ms")
        
        env.close()
        print("\n✅ Integration test complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Full Integration Test')
    parser.add_argument('--config', type=str, default='config/autonomy.yaml',
                       help='Config path')
    args = parser.parse_args()
    
    run_integration_test(args.config)

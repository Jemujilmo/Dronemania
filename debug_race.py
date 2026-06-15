#!/usr/bin/env python3
"""
Debug version of racing AI - add detailed logging
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from simulation.racing_simulator import RacingSimulator
from perception.gate_detector import GateDetector
from state_estimation.ekf_estimator import EKFEstimator
from planning.waypoint_planner import WaypointPlanner
from control.simple_pid_controller import SimplePIDController


def main():
    """Run debug test."""
    
    config = {
        'sim_rate': 240,
        'perception': {
            'enabled': True,
            'camera': {
                'input_resolution': [640, 480],
                'processing_resolution': [320, 240],
            },
            'gate_detection': {
                'method': 'color_threshold',
                'min_confidence': 0.3,
                'color_ranges': {
                    'lower': [0, 50, 50],
                    'upper': [180, 255, 255],
                },
            },
        },
        'state_estimation': {
            'enabled': True,
            'method': 'ekf',
            'imu_weight': 0.5,
            'vision_weight': 0.5,
            'initial_state': {
                'position': [0.0, 0.0, 1.0],
                'velocity': [0.0, 0.0, 0.0],
            },
        },
        'planning': {
            'enabled': True,
            'method': 'reactive',
            'lookahead_distance': 5.0,
            'target_velocity': 8.0,
            'collision_check': True,
            'replan_rate_hz': 20,
        },
        'control': {
            'type': 'pid',
            'pid_gains': {
                'position': {'kp': 1.2, 'ki': 0.02, 'kd': 0.3},
                'velocity': {'kp': 0.5, 'ki': 0.01, 'kd': 0.1},
                'yaw': {'kp': 1.5, 'ki': 0.05, 'kd': 0.3},
            },
            'max_velocity': 15.0,
            'max_angular_velocity': 3.0,
        },
    }
    
    # Initialize
    sim = RacingSimulator(config)
    gate_detector = GateDetector(config['perception'])
    state_estimator = EKFEstimator(config['state_estimation'])
    planner = WaypointPlanner(config['planning'])
    controller = SimplePIDController(config['control'])
    
    sim.reset()
    state_estimator.reset()
    controller.reset()
    
    print("Starting debug race...")
    initial_state = sim.physics.get_state()
    print(f"Initial drone position: {initial_state['position']}")
    print(f"First few gates: {sim.gates[:3]}\n")
    
    for step in range(500):  # 500 steps = 2+ seconds
        # Get observation
        obs = sim.get_observation()
        camera_image = obs['camera']
        imu_data = obs['imu']
        
        # Gate detection
        perception_output = gate_detector.process(camera_image)
        
        # State estimation
        state_estimate = state_estimator.update(
            imu_data=imu_data,
            vision_data=perception_output,
            ground_truth_state=obs.get('drone_state'),
            dt=1.0/240.0
        )
        
        # Planning
        target = planner.plan(
            current_state=state_estimate,
            perception=perception_output
        )
        
        # Control
        control_cmd = controller.compute(
            current_state=state_estimate,
            target=target,
            dt=1.0/240.0
        )
        
        # Motor commands
        vx, vy, vz = control_cmd['velocity']
        hover = 0.5
        z_adj = np.clip(vz / 20.0, -0.3, 0.3)
        pitch = np.clip(vx / 20.0, -0.2, 0.2)
        roll = np.clip(vy / 20.0, -0.2, 0.2)
        
        m1 = hover + z_adj + pitch - roll
        m2 = hover + z_adj + pitch + roll
        m3 = hover + z_adj - pitch - roll
        m4 = hover + z_adj - pitch + roll
        
        motors = np.array([m1, m2, m3, m4])
        motors = np.clip(motors, 0.0, 1.0)
        
        # Debug output frequently to diagnose crash
        if step % 50 == 0:
            drone_pos = state_estimate['position']
            drone_vel = state_estimate['velocity']
            target_pos = target['position']
            target_vel = target.get('velocity', [0,0,0])
            
            print(f"\nStep {step}:")
            print(f"  Drone pos: {drone_pos}, vel: {drone_vel}")
            print(f"  Target pos: {target_pos}, vel: {target_vel}")
            print(f"  Motors: [{motors[0]:.2f}, {motors[1]:.2f}, {motors[2]:.2f}, {motors[3]:.2f}]")
            
            if 'detected_gates' in perception_output and perception_output['detected_gates']:
                print(f"  Gates: {len(perception_output['detected_gates'])} detected")
            else:
                print(f"  Gates: None")
        
        sim.set_motor_commands(motors)
        
        # Check termination
        terminated, truncated, info = sim.check_termination()
        
        if terminated:
            print(f"\nTerminated at step {step}: {info.get('reason')}")
            final_state = sim.physics.get_state()
            print(f"Final position: {final_state['position']}")
            break


if __name__ == "__main__":
    main()

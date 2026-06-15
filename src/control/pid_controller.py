"""
PID controller for drone control.

Translates high-level targets (position, velocity) into low-level commands.
"""

import numpy as np
from typing import Dict, Any
from simple_pid import PID


class PIDController:
    """PID controller for 3D position and yaw control."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize PID controllers for each axis."""
        self.config = config
        
        # Get PID gains from config
        pos_gains = config['pid_gains']['position']
        vel_gains = config['pid_gains']['velocity']
        yaw_gains = config['pid_gains']['yaw']
        
        # Position controllers (output is velocity command)
        self.pid_x = PID(pos_gains['kp'], pos_gains['ki'], pos_gains['kd'])
        self.pid_y = PID(pos_gains['kp'], pos_gains['ki'], pos_gains['kd'])
        self.pid_z = PID(pos_gains['kp'], pos_gains['ki'], pos_gains['kd'])
        
        # Velocity controllers (output is acceleration/thrust)
        self.pid_vx = PID(vel_gains['kp'], vel_gains['ki'], vel_gains['kd'])
        self.pid_vy = PID(vel_gains['kp'], vel_gains['ki'], vel_gains['kd'])
        self.pid_vz = PID(vel_gains['kp'], vel_gains['ki'], vel_gains['kd'])
        
        # Yaw controller
        self.pid_yaw = PID(yaw_gains['kp'], yaw_gains['ki'], yaw_gains['kd'])
        self.pid_yaw.output_limits = (-2.0, 2.0)  # Limit yaw rate
        
        # Set output limits
        max_vel = config.get('max_velocity', 10.0)
        self.pid_x.output_limits = (-max_vel, max_vel)
        self.pid_y.output_limits = (-max_vel, max_vel)
        self.pid_z.output_limits = (-max_vel, max_vel)
        
    def compute(self, current_state: Dict[str, Any], 
                target: Dict[str, Any], 
                dt: float) -> Dict[str, Any]:
        """
        Compute control commands.
        
        Args:
            current_state: Current state estimate
            target: Target from planner
            dt: Time step
            
        Returns:
            Control command dictionary
        """
        # Ensure dt is positive (simple_pid requirement)
        if dt <= 0:
            dt = 1.0 / 240.0  # Default 240Hz
        
        # Set sample time for PIDs
        self.pid_x.sample_time = dt
        self.pid_y.sample_time = dt
        self.pid_z.sample_time = dt
        self.pid_vx.sample_time = dt
        self.pid_vy.sample_time = dt
        self.pid_vz.sample_time = dt
        self.pid_yaw.sample_time = dt
        
        # Current state
        current_pos = current_state['position']
        current_vel = current_state['velocity']
        current_yaw = current_state['orientation'][2]
        
        # Target state
        target_pos = target['position']
        target_vel = target.get('velocity', np.zeros(3))
        target_yaw = target.get('yaw', current_yaw)
        
        # Position control (outputs desired velocity)
        vel_cmd_x = self.pid_x(current_pos[0], target_pos[0])
        vel_cmd_y = self.pid_y(current_pos[1], target_pos[1])
        vel_cmd_z = self.pid_z(current_pos[2], target_pos[2])
        
        # Blend with target velocity
        desired_vel = np.array([vel_cmd_x, vel_cmd_y, vel_cmd_z])
        desired_vel = 0.7 * desired_vel + 0.3 * target_vel
        
        # Velocity control (outputs acceleration)
        acc_x = self.pid_vx(current_vel[0], desired_vel[0])
        acc_y = self.pid_vy(current_vel[1], desired_vel[1])
        acc_z = self.pid_vz(current_vel[2], desired_vel[2])
        
        # Yaw control
        yaw_rate = self.pid_yaw(current_yaw, target_yaw)
        
        # Convert to standard control format
        # Depending on simulator interface, this might be:
        # - Body-frame velocities
        # - Acceleration commands
        # - Motor thrust values
        
        return {
            'velocity': desired_vel,
            'acceleration': np.array([acc_x, acc_y, acc_z]),
            'yaw_rate': yaw_rate,
            'timestamp': current_state.get('timestamp', 0.0)
        }
    
    def reset(self):
        """Reset all PID controllers."""
        self.pid_x.reset()
        self.pid_y.reset()
        self.pid_z.reset()
        self.pid_vx.reset()
        self.pid_vy.reset()
        self.pid_vz.reset()
        self.pid_yaw.reset()

"""
Simple inline PID controller without external dependencies.
"""

import numpy as np
from typing import Dict, Any


class SimplePID:
    """Minimal PID controller."""
    
    def __init__(self, kp: float, ki: float, kd: float, 
                 output_limits: tuple = None):
        """Initialize PID gains."""
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        
        self.last_error = 0.0
        self.integral = 0.0
    
    def update(self, error: float, dt: float) -> float:
        """
        Compute PID output.
        
        Args:
            error: Setpoint - measured
            dt: Time step
            
        Returns:
            Control output
        """
        if dt <= 0:
            return 0.0
        
        # Proportional
        p_term = self.kp * error
        
        # Integral
        self.integral += error * dt
        i_term = self.ki * self.integral
        
        # Derivative
        d_term = self.kd * (error - self.last_error) / dt if dt > 0 else 0
        self.last_error = error
        
        # Sum
        output = p_term + i_term + d_term
        
        # Saturate
        if self.output_limits:
            output = np.clip(output, self.output_limits[0], self.output_limits[1])
        
        return output
    
    def reset(self):
        """Reset integrator and derivative."""
        self.last_error = 0.0
        self.integral = 0.0


class SimplePIDController:
    """Racing-optimized PID controller without external PID library."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize PID controllers for each axis."""
        self.config = config
        
        # Get PID gains from config
        pos_gains = config['pid_gains']['position']
        vel_gains = config['pid_gains']['velocity']
        yaw_gains = config['pid_gains']['yaw']
        
        max_vel = config.get('max_velocity', 10.0)
        
        # Position controllers (output is velocity command)
        self.pid_x = SimplePID(
            pos_gains['kp'], pos_gains['ki'], pos_gains['kd'],
            output_limits=(-max_vel, max_vel)
        )
        self.pid_y = SimplePID(
            pos_gains['kp'], pos_gains['ki'], pos_gains['kd'],
            output_limits=(-max_vel, max_vel)
        )
        self.pid_z = SimplePID(
            pos_gains['kp'], pos_gains['ki'], pos_gains['kd'],
            output_limits=(-max_vel, max_vel)
        )
        
        # Velocity controllers (output is acceleration/thrust)
        self.pid_vx = SimplePID(
            vel_gains['kp'], vel_gains['ki'], vel_gains['kd']
        )
        self.pid_vy = SimplePID(
            vel_gains['kp'], vel_gains['ki'], vel_gains['kd']
        )
        self.pid_vz = SimplePID(
            vel_gains['kp'], vel_gains['ki'], vel_gains['kd']
        )
        
        # Yaw controller
        self.pid_yaw = SimplePID(
            yaw_gains['kp'], yaw_gains['ki'], yaw_gains['kd'],
            output_limits=(-2.0, 2.0)
        )
        
        self.last_dt = 1.0 / 240.0  # Cache for safety
    
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
        # Ensure dt is positive
        if dt <= 0:
            dt = self.last_dt
        self.last_dt = dt
        
        # Current state
        current_pos = current_state['position']
        current_vel = current_state['velocity']
        current_yaw = current_state['orientation'][2]
        
        # Target state
        target_pos = target['position']
        target_vel = target.get('velocity', np.zeros(3))
        target_yaw = target.get('yaw', current_yaw)
        
        # Position control (outputs desired velocity)
        err_x = target_pos[0] - current_pos[0]
        err_y = target_pos[1] - current_pos[1]
        err_z = target_pos[2] - current_pos[2]
        
        vel_cmd_x = self.pid_x.update(err_x, dt)
        vel_cmd_y = self.pid_y.update(err_y, dt)
        vel_cmd_z = self.pid_z.update(err_z, dt)
        
        # Blend with target velocity
        desired_vel = np.array([vel_cmd_x, vel_cmd_y, vel_cmd_z])
        desired_vel = 0.7 * desired_vel + 0.3 * target_vel
        
        # Velocity control (outputs acceleration)
        err_vx = desired_vel[0] - current_vel[0]
        err_vy = desired_vel[1] - current_vel[1]
        err_vz = desired_vel[2] - current_vel[2]
        
        acc_x = self.pid_vx.update(err_vx, dt)
        acc_y = self.pid_vy.update(err_vy, dt)
        acc_z = self.pid_vz.update(err_vz, dt)
        
        # Yaw control
        err_yaw = target_yaw - current_yaw
        yaw_rate = self.pid_yaw.update(err_yaw, dt)
        
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

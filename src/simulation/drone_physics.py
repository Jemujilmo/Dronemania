"""
Custom drone physics simulator - optimized for racing.

Implements 6-DOF rigid body dynamics for quadrotor:
- Position (x, y, z)
- Orientation (roll, pitch, yaw)
- Linear velocity
- Angular velocity

No external dependencies beyond numpy - pure Python physics.
"""

import numpy as np
from typing import Dict, Tuple, Any
import math


class DronePhysics:
    """6-DOF quadrotor physics engine."""
    
    def __init__(self):
        """Initialize drone physics parameters."""
        # Mass and inertia
        self.mass = 0.75  # kg
        self.gravity = 9.81  # m/s^2
        
        # Moment of inertia (assume rectangular body)
        # Ixx, Iyy, Izz
        self.inertia = np.array([0.008, 0.008, 0.013])  # kg*m^2
        self.inertia_inv = 1.0 / self.inertia
        
        # State vector: [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
        self.state = np.zeros(12)
        self.state[2] = 1.0  # Start at 1m height
        
        # Motor parameters
        self.motor_thrust_coefficient = 0.1  # Thrust per motor
        # Each motor should produce about 1/4 of weight at full throttle for reasonable dynamics
        self.motor_max_thrust = 2.0 * self.mass * self.gravity / 4  # Per motor, allows 2x gravity at full
        # This means: 0.5 throttle = hover (0.5 * TMax * 4 = weight)
        
        # Drag coefficients
        self.drag_linear = 0.05  # Linear drag
        self.drag_angular = 0.01  # Angular drag
        
        # Simulation parameters
        self.dt = 1.0 / 240.0  # 240Hz simulation
        
    def set_state(self, position: np.ndarray, velocity: np.ndarray,
                  orientation: np.ndarray, angular_velocity: np.ndarray):
        """Set drone state directly."""
        self.state[0:3] = position
        self.state[3:6] = velocity
        self.state[6:9] = orientation  # [roll, pitch, yaw]
        self.state[9:12] = angular_velocity
    
    def get_state(self) -> Dict[str, np.ndarray]:
        """Get current drone state."""
        return {
            'position': self.state[0:3].copy(),
            'velocity': self.state[3:6].copy(),
            'orientation': self.state[6:9].copy(),  # [roll, pitch, yaw]
            'angular_velocity': self.state[9:12].copy()
        }
    
    def motor_command_to_forces(self, motor_commands: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert motor commands [0-1] to forces and torques.
        
        Motor layout (quadcopter X configuration):
            1   2
             \ /
              X
             / \
            3   4
        
        Args:
            motor_commands: [m1, m2, m3, m4] in range [0, 1]
            
        Returns:
            (force_vector, torque_vector)
        """
        # Clamp commands
        commands = np.clip(motor_commands, 0.0, 1.0)
        
        # Calculate thrust from each motor
        thrusts = commands * self.motor_max_thrust
        
        # Total vertical thrust
        total_thrust = np.sum(thrusts)
        
        # Torques from differential thrusts
        # Roll (pitch motors): 3, 4 vs 1, 2
        torque_roll = (thrusts[2] + thrusts[3] - thrusts[0] - thrusts[1]) * 0.1
        
        # Pitch (front/back motors): 1, 3 vs 2, 4
        torque_pitch = (thrusts[0] + thrusts[2] - thrusts[1] - thrusts[3]) * 0.1
        
        # Yaw (counter-rotation)
        # Diagonal pairs spin same direction: m1,m4 CW  |  m2,m3 CCW
        # Net yaw = (m2+m3) - (m1+m4)  (CCW pair minus CW pair)
        torque_yaw = (thrusts[1] + thrusts[2] - thrusts[0] - thrusts[3]) * 0.02
        
        force = np.array([0.0, 0.0, total_thrust])
        torque = np.array([torque_roll, torque_pitch, torque_yaw])
        
        return force, torque
    
    def _euler_to_rotation_matrix(self, euler: np.ndarray) -> np.ndarray:
        """Convert Euler angles (roll, pitch, yaw) to rotation matrix."""
        roll, pitch, yaw = euler
        
        # Rotation matrices for each axis
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        return Rz @ Ry @ Rx  # ZYX convention
    
    def _rotation_matrix_to_euler(self, R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to Euler angles."""
        # Extract Euler angles from rotation matrix
        sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
        
        singular = sy < 1e-6
        
        if not singular:
            roll = math.atan2(R[2, 1], R[2, 2])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = math.atan2(R[1, 0], R[0, 0])
        else:
            roll = math.atan2(-R[1, 2], R[1, 1])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = 0
        
        return np.array([roll, pitch, yaw])
    
    def step(self, motor_commands: np.ndarray) -> Dict[str, Any]:
        """
        Step physics simulation by one time step.
        
        Args:
            motor_commands: [0-1] thrust values for 4 motors
            
        Returns:
            Dictionary with updated state and info
        """
        pos, vel, euler, ang_vel = (
            self.state[0:3], self.state[3:6], self.state[6:9], self.state[9:12]
        )
        
        # Get forces and torques from motor commands
        force_body, torque_body = self.motor_command_to_forces(motor_commands)
        
        # Get rotation matrix
        R = self._euler_to_rotation_matrix(euler)
        
        # Transform body-frame force to world frame
        force_world = R @ force_body
        force_world[2] -= self.mass * self.gravity  # Add gravity
        
        # Linear acceleration (Newton's second law)
        accel = force_world / self.mass
        
        # Apply linear drag
        accel -= self.drag_linear * vel
        
        # Update linear velocity and position
        new_vel = vel + accel * self.dt
        new_pos = pos + vel * self.dt  # Use previous velocity (symplectic)
        
        # Angular acceleration (from torques and inertia)
        # τ = I * α
        ang_accel = self.inertia_inv * torque_body
        
        # Gyroscopic effects (simplified)
        # This is more complex, but we'll use a simple approximation
        
        # Apply angular drag
        ang_accel -= self.drag_angular * ang_vel
        
        # Update angular velocity
        new_ang_vel = ang_vel + ang_accel * self.dt
        
        # Update orientation using angle velocity
        # Euler angle derivatives: deuler/dt = some function of angular velocity
        # Simplified: assume small angles
        deuler = self._angular_velocity_to_euler_derivative(euler, new_ang_vel)
        new_euler = euler + deuler * self.dt
        
        # Wrap yaw to [-pi, pi]
        new_euler[2] = self._wrap_angle(new_euler[2])
        
        # Update state
        self.state[0:3] = new_pos
        self.state[3:6] = new_vel
        self.state[6:9] = new_euler
        self.state[9:12] = new_ang_vel
        
        info = {
            'position': new_pos.copy(),
            'velocity': new_vel.copy(),
            'accel': accel.copy(),
            'thrust': force_world[2]
        }
        
        return info
    
    def _angular_velocity_to_euler_derivative(self, euler: np.ndarray, 
                                              ang_vel: np.ndarray) -> np.ndarray:
        """Convert angular velocity to Euler angle rates."""
        roll, pitch, yaw = euler
        
        # Relationship between Euler angle rates and angular velocity
        # d(roll)/dt = wx + wy*sin(roll)*tan(pitch) + wz*cos(roll)*tan(pitch)
        # d(pitch)/dt = wy*cos(roll) - wz*sin(roll)
        # d(yaw)/dt = wy*sin(roll)/cos(pitch) + wz*cos(roll)/cos(pitch)
        
        cos_roll = np.cos(roll)
        sin_roll = np.sin(roll)
        cos_pitch = np.cos(pitch)
        sin_pitch = np.sin(pitch)
        tan_pitch = sin_pitch / cos_pitch if abs(cos_pitch) > 1e-6 else 0
        
        wx, wy, wz = ang_vel
        
        euler_rates = np.array([
            wx + wy * sin_roll * tan_pitch + wz * cos_roll * tan_pitch,
            wy * cos_roll - wz * sin_roll,
            (wy * sin_roll + wz * cos_roll) / cos_pitch if abs(cos_pitch) > 1e-6 else 0
        ])
        
        return euler_rates
    
    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Wrap angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def reset(self):
        """Reset drone to initial state."""
        self.state = np.zeros(12)
        self.state[2] = 1.0  # Start at 1m height

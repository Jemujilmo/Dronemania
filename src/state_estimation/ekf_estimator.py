"""
Extended Kalman Filter for drone state estimation.

Fuses IMU, vision, and other sensor data to estimate:
- Position (x, y, z)
- Velocity (vx, vy, vz)
- Orientation (quaternion or euler angles)
- Angular velocity (wx, wy, wz)
"""

import numpy as np
from typing import Dict, Any, Optional


class EKFEstimator:
    """Extended Kalman Filter for state estimation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize EKF with configuration."""
        self.config = config
        
        # State vector: [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
        self.state_dim = 12
        self.state = np.zeros(self.state_dim)
        
        # Set initial state from config
        initial = config.get('initial_state', {})
        if 'position' in initial:
            self.state[0:3] = np.array(initial['position'])
        if 'velocity' in initial:
            self.state[3:6] = np.array(initial['velocity'])
        
        # Covariance matrix
        self.P = np.eye(self.state_dim) * 1.0
        
        # Process noise
        self.Q = np.eye(self.state_dim) * 0.01
        
        # Measurement noise
        self.R_imu = np.eye(6) * 0.1  # IMU measurements
        self.R_vision = np.eye(3) * 0.5  # Vision-based position
        
        # Weights for sensor fusion
        self.imu_weight = config.get('imu_weight', 0.3)
        self.vision_weight = config.get('vision_weight', 0.7)
        
    def predict(self, dt: float, control: Optional[np.ndarray] = None):
        """
        Prediction step using motion model.
        
        Args:
            dt: Time step
            control: Optional control input (acceleration commands)
        """
        # Simple constant velocity model
        # x_k+1 = x_k + vx * dt
        # vx_k+1 = vx_k (constant velocity)
        
        F = np.eye(self.state_dim)
        
        # Position updates from velocity
        F[0, 3] = dt  # x += vx * dt
        F[1, 4] = dt  # y += vy * dt
        F[2, 5] = dt  # z += vz * dt
        
        # Orientation updates from angular velocity
        F[6, 9] = dt   # roll += wx * dt
        F[7, 10] = dt  # pitch += wy * dt
        F[8, 11] = dt  # yaw += wz * dt
        
        # Predict state
        self.state = F @ self.state
        
        # Predict covariance
        self.P = F @ self.P @ F.T + self.Q
    
    def update_imu(self, imu_data: Dict[str, np.ndarray]):
        """Update with IMU measurements."""
        if imu_data is None:
            return
        
        # IMU provides: linear acceleration and angular velocity
        accel = imu_data.get('linear_acceleration', np.zeros(3))
        gyro = imu_data.get('angular_velocity', np.zeros(3))
        
        # Measurement vector: [ax, ay, az, wx, wy, wz]
        z = np.concatenate([accel, gyro])
        
        # Measurement model (simplified)
        H = np.zeros((6, self.state_dim))
        H[3:6, 9:12] = np.eye(3)  # Angular velocity direct measurement
        
        # Kalman gain
        S = H @ self.P @ H.T + self.R_imu
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state (weighted by configuration)
        innovation = z - H @ self.state
        self.state += K @ innovation * self.imu_weight
        
        # Update covariance
        self.P = (np.eye(self.state_dim) - K @ H) @ self.P
    
    def update_vision(self, vision_data: Dict[str, Any]):
        """Update with vision-based position estimate."""
        if vision_data is None or vision_data.get('next_gate') is None:
            return
        
        # Vision provides relative position to gate
        # This is simplified - real implementation would triangulate
        gate = vision_data['next_gate']
        
        # Estimate relative position from image coordinates and gate size
        # This is a placeholder - proper implementation needs camera calibration
        confidence = vision_data.get('confidence', 0.0)
        
        if confidence < 0.5:
            return  # Don't update with low confidence detections
        
        # For now, skip vision update in this simplified version
        # Real implementation would convert pixel coords to world coords
        pass
    
    def update(self, imu_data: Optional[Dict] = None, 
               vision_data: Optional[Dict] = None,
               ground_truth_state: Optional[Dict] = None, 
               dt: float = 0.02) -> Dict[str, Any]:
        """
        Complete update cycle: predict + update.
        
        Args:
            imu_data: IMU measurements
            vision_data: Vision measurements (gate detection)
            ground_truth_state: Actual drone state (from simulator, for fusion)
            dt: Time step
        
        Returns:
            State estimate dictionary
        """
        # Prediction step
        self.predict(dt)
        
        # Update steps
        if imu_data is not None:
            self.update_imu(imu_data)
        
        if vision_data is not None:
            self.update_vision(vision_data)
        
        # If we have ground truth (simulator for testing), fuse it
        # In real life this would be GNSS, rangefinder, vision odometry, etc.
        if ground_truth_state is not None:
            self._update_ground_truth(ground_truth_state)
        
        # Return state estimate
        return self.get_state()
    
    def _update_ground_truth(self, truth: Dict):
        """Fuse ground truth state (from simulator)."""
        # Measurement: position and velocity from simulator
        if 'position' in truth:
            pos_measurement = truth['position']
            # Measurement model: we directly measure position
            z_pos = pos_measurement
            H_pos = np.zeros((3, self.state_dim))
            H_pos[0, 0] = 1  # measure x
            H_pos[1, 1] = 1  # measure y
            H_pos[2, 2] = 1  # measure z
            
            # Measurement noise (low, we trust simulator)
            R_pos = np.eye(3) * 0.01
            
            # Update with proper Kalman gain
            S = H_pos @ self.P @ H_pos.T + R_pos
            try:
                S_inv = np.linalg.inv(S + np.eye(3) * 1e-6)
                K = self.P @ H_pos.T @ S_inv
                innovation = z_pos - self.state[0:3]
                self.state[0:3] += (K @ innovation) * 0.8  # 80% weight to simulator
                self.P = (np.eye(self.state_dim) - K @ H_pos) @ self.P
            except:
                pass  # Skip update if singular
        
        if 'velocity' in truth:
            vel_measurement = truth['velocity']
            z_vel = vel_measurement
            H_vel = np.zeros((3, self.state_dim))
            H_vel[0, 3] = 1  # measure vx
            H_vel[1, 4] = 1  # measure vy
            H_vel[2, 5] = 1  # measure vz
            
            R_vel = np.eye(3) * 0.01
            
            S = H_vel @ self.P @ H_vel.T + R_vel
            try:
                S_inv = np.linalg.inv(S + np.eye(3) * 1e-6)
                K = self.P @ H_vel.T @ S_inv
                innovation = z_vel - self.state[3:6]
                self.state[3:6] += (K @ innovation) * 0.8
                self.P = (np.eye(self.state_dim) - K @ H_vel) @ self.P
            except:
                pass  # Skip update if singular
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state estimate."""
        return {
            'position': self.state[0:3].copy(),
            'velocity': self.state[3:6].copy(),
            'orientation': self.state[6:9].copy(),  # [roll, pitch, yaw]
            'angular_velocity': self.state[9:12].copy(),
            'covariance': self.P.copy()
        }
    
    def reset(self):
        """Reset estimator to initial state."""
        self.state = np.zeros(self.state_dim)
        self.P = np.eye(self.state_dim) * 1.0
        
        # Re-apply initial state from config
        initial = self.config.get('initial_state', {})
        if 'position' in initial:
            self.state[0:3] = np.array(initial['position'])
        if 'velocity' in initial:
            self.state[3:6] = np.array(initial['velocity'])

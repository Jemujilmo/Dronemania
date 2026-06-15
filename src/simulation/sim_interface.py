"""
PyBullet simulator implementation for drone racing.

Provides physics-based quadrotor simulation with:
- Realistic drone dynamics
- Procedurally generated gate sequences
- Camera sensor simulation for vision processing
- IMU sensor simulation
"""

import numpy as np
import pybullet as p
import pybullet_data
import cv2
from typing import Dict, Any, Tuple, List, Optional
import math


class PyBulletDroneEnv:
    """PyBullet-based drone racing environment."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize PyBullet environment."""
        self.config = config
        
        # Connect to PyBullet
        self.client = p.connect(p.GUI)  # Use GUI=True for visualization, GUI=False for headless
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Load plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Drone parameters
        self.drone_mass = 0.75  # kg
        self.arm_length = 0.1575  # meters
        self.max_rpm = 10000
        
        # Load drone URDF (or create simple box drone)
        self.drone_id = None
        self._load_drone()
        
        # Camera parameters
        self.camera_height = 480
        self.camera_width = 640
        self.fov = 60  # degrees
        
        # Gates in environment
        self.gates: List[Dict[str, Any]] = []
        self.current_gate_idx = 0
        
        # Simulation state
        self.time_step = 1.0 / self.config.get('sim_rate', 240)
        p.setTimeStep(self.time_step)
        self.step_count = 0
        
        # Motor control state
        self.motor_thrusts = np.zeros(4)
        
    def _load_drone(self):
        """Load drone model into PyBullet."""
        # Create a simple drone model from primitives
        # In production, load URDF file
        
        base_mass = self.drone_mass
        base_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.02])
        
        self.drone_id = p.createMultiBody(
            baseMass=base_mass,
            baseCollisionShapeIndex=base_shape,
            basePosition=[0, 0, 1.0],
            baseOrientation=[0, 0, 0, 1]
        )
        
    def add_gate(self, position: np.ndarray, orientation: np.ndarray, 
                 width: float = 1.0, height: float = 1.0, color: Tuple = (1, 0, 0)):
        """
        Add a gate to the environment.
        
        Args:
            position: [x, y, z] position
            orientation: [roll, pitch, yaw] in radians
            width: Gate width
            height: Gate height
            color: RGB color
        """
        # Create gate frame from cylinders
        # For now, simple visual representation
        
        # Create collision shape for gate
        gate_shape = p.createCollisionShape(p.GEOM_BOX, 
                                           halfExtents=[width/2, height/2, 0.01])
        
        # Convert euler to quaternion
        quat = p.getQuaternionFromEuler(orientation)
        
        # Create gate body
        gate_id = p.createMultiBody(
            baseMass=0,  # Static
            baseCollisionShapeIndex=gate_shape,
            basePosition=position,
            baseOrientation=quat,
            baseVisualShapeIndex=self._create_gate_visual(color)
        )
        
        self.gates.append({
            'id': gate_id,
            'position': np.array(position),
            'orientation': np.array(orientation),
            'width': width,
            'height': height
        })
    
    def _create_gate_visual(self, color: Tuple) -> int:
        """Create visual shape for gate."""
        shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.5, 0.5, 0.01],
            rgbaColor=color + (0.8,)  # Add alpha
        )
        return shape
    
    def procedural_track(self, num_gates: int = 5, spacing: float = 5.0):
        """
        Generate procedural race track with gates.
        
        Args:
            num_gates: Number of gates in course
            spacing: Spacing between gates
        """
        self.gates = []
        
        for i in range(num_gates):
            # Simple linear track with slight variations
            x = i * spacing
            y = np.sin(i * 0.3) * 2  # Slight curve
            z = 1.5
            
            # Randomize gate orientation slightly
            roll = np.random.uniform(-0.1, 0.1)
            pitch = np.random.uniform(-0.1, 0.1)
            yaw = np.arctan2(1, spacing)  # Face down track
            
            self.add_gate(
                position=np.array([x, y, z]),
                orientation=np.array([roll, pitch, yaw]),
                color=self._get_gate_color(i)
            )
    
    def _get_gate_color(self, gate_idx: int) -> Tuple:
        """Get color for gate based on index."""
        # Cycle through colors
        colors = [
            (1, 0, 0),  # Red
            (0, 1, 0),  # Green
            (0, 0, 1),  # Blue
            (1, 1, 0),  # Yellow
            (1, 0, 1),  # Magenta
        ]
        return colors[gate_idx % len(colors)]
    
    def reset(self) -> Dict[str, Any]:
        """Reset environment and return initial observation."""
        # Reset drone position and velocity
        p.resetBasePositionAndOrientation(
            self.drone_id,
            [0, 0, 1.0],
            [0, 0, 0, 1]
        )
        p.resetBaseVelocity(self.drone_id, [0, 0, 0], [0, 0, 0])
        
        # Reset simulation
        self.step_count = 0
        self.current_gate_idx = 0
        self.motor_thrusts = np.zeros(4)
        
        # Generate procedural track
        self.procedural_track(num_gates=5, spacing=8.0)
        
        # Return initial observation
        return self.get_observation()
    
    def get_observation(self) -> Dict[str, Any]:
        """Get current sensor observations."""
        # Get drone state
        pos, orn = p.getBasePositionAndOrientation(self.drone_id)
        lin_vel, ang_vel = p.getBaseVelocity(self.drone_id)
        
        # Get drone orientation as euler angles
        roll, pitch, yaw = p.getEulerFromQuaternion(orn)
        
        # Simulate camera
        camera_image = self._simulate_camera()
        
        # Simulate IMU
        imu_data = {
            'linear_acceleration': np.array(lin_vel) / self.time_step,  # Simplified
            'angular_velocity': np.array(ang_vel)
        }
        
        return {
            'camera': camera_image,
            'imu': imu_data,
            'drone_state': {
                'position': np.array(pos),
                'orientation': np.array([roll, pitch, yaw]),
                'velocity': np.array(lin_vel),
                'angular_velocity': np.array(ang_vel)
            },
            'timestamp': self.step_count * self.time_step
        }
    
    def _simulate_camera(self) -> np.ndarray:
        """Simulate drone forward-looking camera."""
        # Get drone position and orientation
        pos, orn = p.getBasePositionAndOrientation(self.drone_id)
        roll, pitch, yaw = p.getEulerFromQuaternion(orn)
        
        # Look-ahead direction (where drone is facing)
        # This is simplified - real implementation would use proper camera matrix
        forward = np.array([
            np.cos(yaw) * np.cos(pitch),
            np.sin(yaw) * np.cos(pitch),
            -np.sin(pitch)
        ])
        
        # Camera target is 10m ahead
        target = np.array(pos) + forward * 10
        
        # Camera up vector
        up = np.array([0, 0, 1])
        
        # Get view and projection matrices
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=pos,
            cameraTargetPosition=target,
            cameraUpVector=up
        )
        
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=self.camera_width / self.camera_height,
            nearVal=0.1,
            farVal=100
        )
        
        # Render image
        width, height, rgb, depth, mask = p.getCameraImage(
            width=self.camera_width,
            height=self.camera_height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix
        )
        
        # Convert to numpy array and resize
        image = np.array(rgb, dtype=np.uint8)[:, :, :3]  # Drop alpha
        image = cv2.resize(image, (320, 240))  # Resize to config size
        
        return image
    
    def send_command(self, command: Dict[str, Any]) -> Tuple[bool, bool, Dict]:
        """
        Send control command to drone.
        
        Args:
            command: Control command with velocity/acceleration
            
        Returns:
            (terminated, truncated, info) tuple
        """
        # Extract command
        target_velocity = command.get('velocity', np.zeros(3))
        yaw_rate = command.get('yaw_rate', 0.0)
        
        # Simple velocity tracking using forces
        # In reality, this would be motor RPM control
        current_vel, _ = p.getBaseVelocity(self.drone_id)
        current_vel = np.array(current_vel)
        
        # PID-like force application
        force_scale = 10.0
        force = (target_velocity - current_vel) * force_scale
        
        # Add gravity compensation
        force[2] += self.drone_mass * 9.81
        
        # Apply force to drone
        p.applyExternalForce(
            self.drone_id,
            -1,  # Link index (-1 = base link)
            force,
            [0, 0, 0],  # Force position (center of mass)
            p.WORLD_FRAME
        )
        
        # Apply torque for yaw
        _, ang_vel = p.getBaseVelocity(self.drone_id)
        torque = np.array([0, 0, (yaw_rate - ang_vel[2]) * 2.0])
        p.applyExternalTorque(self.drone_id, -1, torque, p.WORLD_FRAME)
        
        # Step simulation
        p.stepSimulation()
        self.step_count += 1
        
        # Check termination conditions
        pos, _ = p.getBasePositionAndOrientation(self.drone_id)
        
        terminated = False
        truncated = False
        info = {'step': self.step_count}
        
        # Crash detection (below ground)
        if pos[2] < 0:
            terminated = True
            info['reason'] = 'crashed'
        
        # Gate passed detection
        for i, gate in enumerate(self.gates):
            if i == self.current_gate_idx:
                dist_to_gate = np.linalg.norm(pos - gate['position'])
                if dist_to_gate < 2.0:  # Passed through gate
                    self.current_gate_idx += 1
                    info['gate_passed'] = i
        
        # Course complete
        if self.current_gate_idx >= len(self.gates):
            terminated = True
            info['reason'] = 'course_complete'
            info['total_time'] = self.step_count * self.time_step
        
        # Max time limit
        if self.step_count > 10000:  # ~40 seconds at 240Hz
            truncated = True
            info['reason'] = 'timeout'
        
        return terminated, truncated, info
    
    def close(self):
        """Disconnect from PyBullet."""
        p.disconnect(self.client)


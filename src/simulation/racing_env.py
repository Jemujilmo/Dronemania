"""
Drone racing environment wrapper.

Integrates PyBullet simulator with autonomy stack.
Implements Gymnasium interface for compatibility.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.sim_interface import PyBulletDroneEnv


class DroneRacingEnv(gym.Env):
    """Drone racing environment for gate navigation challenge."""
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    
    def __init__(self, config: Dict[str, Any], render_mode: Optional[str] = None):
        """Initialize racing environment."""
        super().__init__()
        
        self.config = config
        self.render_mode = render_mode
        
        # Initialize PyBullet simulator
        self.simulator = PyBulletDroneEnv(config)
        
        # Action space: [vx, vy, vz, yaw_rate]
        # Velocities in m/s, yaw_rate in rad/s
        self.action_space = spaces.Box(
            low=np.array([-10, -10, -5, -2]),
            high=np.array([10, 10, 10, 2]),
            dtype=np.float32
        )
        
        # Observation space: camera image
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(240, 320, 3),
            dtype=np.uint8
        )
        
        self.episode_steps = 0
        self.max_steps = config.get('max_episode_steps', 10000)
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset environment."""
        super().reset(seed=seed)
        
        self.episode_steps = 0
        observation_dict = self.simulator.reset()
        
        observation = observation_dict['camera']
        info = {}
        
        return observation, info
    
    def step(self, action: np.ndarray):
        """Execute one step of environment."""
        self.episode_steps += 1
        
        # Convert action to control command
        command = {
            'velocity': action[:3],
            'yaw_rate': action[3],
            'timestamp': self.episode_steps * 0.00417  # ~240Hz
        }
        
        # Send to simulator
        terminated, truncated, info = self.simulator.send_command(command)
        
        # Get new observation
        obs_dict = self.simulator.get_observation()
        observation = obs_dict['camera']
        
        # Calculate reward
        reward = self._calculate_reward(info)
        
        # Check max steps
        if self.episode_steps >= self.max_steps:
            truncated = True
        
        return observation, reward, terminated, truncated, info
    
    def _calculate_reward(self, info: Dict) -> float:
        """Calculate reward for this step."""
        reward = 0.0
        
        # Per-step penalty (encourages speed)
        reward -= 0.01
        
        # Gate passed bonus
        if 'gate_passed' in info:
            reward += 50.0
        
        # Crash penalty
        if info.get('reason') == 'crashed':
            reward -= 100.0
        
        # Course complete bonus
        if info.get('reason') == 'course_complete':
            reward += 500.0
        
        return reward
    
    def render(self):
        """Render environment (handled by PyBullet GUI)."""
        if self.render_mode == "rgb_array":
            return self.simulator._simulate_camera()
    
    def close(self):
        """Close environment."""
        self.simulator.close()

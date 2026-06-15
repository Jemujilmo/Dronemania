"""
Base drone interface - abstraction layer for both simulation and real drones.
This allows the same control code to work in sim and on real hardware.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Dict, Any


class BaseDrone(ABC):
    """Abstract base class for drone interface."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_simulation = True
        
    @abstractmethod
    def reset(self) -> np.ndarray:
        """
        Reset the drone to initial state.
        
        Returns:
            observation: Initial observation (camera image, sensors, etc.)
        """
        pass
    
    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one timestep of drone control.
        
        Args:
            action: Control action (velocities, thrust, etc.)
            
        Returns:
            observation: Current observation
            reward: Reward signal
            terminated: Whether episode ended (crash, goal)
            truncated: Whether episode was truncated (timeout)
            info: Additional information
        """
        pass
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """
        Get current drone state.
        
        Returns:
            Dictionary containing position, velocity, orientation, etc.
        """
        pass
    
    @abstractmethod
    def get_camera_image(self) -> np.ndarray:
        """
        Get current camera image.
        
        Returns:
            RGB image as numpy array
        """
        pass
    
    @abstractmethod
    def close(self):
        """Clean up resources."""
        pass

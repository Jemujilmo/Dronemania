"""
Waypoint-based path planner.

Generates target waypoints based on gate detections and current state.
Handles collision avoidance and replanning.
"""

import numpy as np
from typing import Dict, Any, List, Optional


class WaypointPlanner:
    """Waypoint-based path planner for gate navigation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize planner with configuration."""
        self.config = config
        self.lookahead_distance = config.get('lookahead_distance', 2.5)  # Reduced for tighter tracking
        self.target_velocity = config.get('target_velocity', 4.0)  # High-speed racing
        self.camera_center_bias_px = config.get('camera_center_bias_px', 0.0)  # No bias by default
        
        # Current plan
        self.waypoints: List[np.ndarray] = []
        self.current_waypoint_idx = 0
        
    def plan(self, current_state: Dict[str, Any], 
             perception: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate target waypoint based on current state and perception.
        
        Args:
            current_state: From state estimator
            perception: From gate detector
            
        Returns:
            Target dictionary with position and velocity
        """
        current_pos = current_state['position']
        current_vel = current_state['velocity']
        
        # Get gate tracking info
        next_gate = perception.get('next_gate')
        detected_gates = perception.get('gates', [])
        current_gate_idx = perception.get('current_gate_idx', 0)
        total_gates = perception.get('total_gates', 20)
        
        # CRITICAL: Reject gates that are significantly behind drone's current X position
        # This prevents targeting missed gates in rear view
        drone_x = current_pos[0]
        expected_gate_pos = perception.get('expected_gate_pos')
        
        if next_gate and detected_gates:
            gate_center_x = next_gate['center'][0]
            gate_area = next_gate.get('area', 0)
            
            # POSITION-BASED FILTER: If we know where the next gate SHOULD be:
            if expected_gate_pos is not None:
                expected_x = expected_gate_pos[0]
                
                # If the gate is ahead of us (where it should be), reject off-center detections
                if expected_x > drone_x - 10.0:  # Gate is ahead (within 10m tolerance)
                    # STRICT centering: accept gates within ±30px of center (130-190)
                    # This still prevents left-bias while allowing some tracking tolerance
                    if gate_center_x < 130 or gate_center_x > 190:
                        next_gate = None
                        # print(f"[PLANNER] Rejected off-center gate: pixel_x={gate_center_x} (need 130-190)")
            else:
                # Fallback to old filter if no position info
                # SIMPLE RULE: Always accept large gates (they're close/next)
                # Only reject very small gates (area < 200) in left-center of image
                if gate_area < 200 and gate_center_x < 140:
                    # Very small gate in left-center = likely distant gate behind us
                    next_gate = None
        
        # Priority order for gate position:
        # 1. Known world position (always accurate) — use regardless of camera quality.
        # 2. Camera-based projection — only when world pos is unavailable.
        # 3. Dead-reckoning fallback — last resort.
        if expected_gate_pos is not None:
            gate_position = np.array(expected_gate_pos, dtype=float)
        elif next_gate is not None:
            gate_position = self._gate_to_world_position(next_gate, current_state)
        else:
            # No gate info at all - keep moving forward
            return self._fallback_target(current_state)
        
        # Generate target waypoint
        # AIM DIRECTLY AT GATE CENTER - don't overshoot!
        direction = gate_position - current_pos
        distance = np.linalg.norm(direction)
        
        if distance > 0.1:
            direction = direction / distance
        else:
            direction = np.array([1.0, 0.0, 0.0])  # Default forward
        
        # Target position: aim AT the gate, not beyond it
        # Use small lookahead only when gate is far (>10m), but always use gate Z
        # to avoid the slanted-path Z interpolation causing altitude bumps.
        if distance > 10.0:
            target_pos = current_pos + direction * min(self.lookahead_distance, distance)
        else:
            # Close to gate - aim directly at it for precision
            target_pos = gate_position.copy()

        # Always target the gate's actual Z — never the interpolated mid-path Z.
        # This lets the altitude controller smoothly fly to the correct level
        # rather than chasing a moving intermediate height.
        target_pos[2] = max(gate_position[2], 0.5)  # absolute minimum safety floor
        
        # Target velocity
        target_vel = direction * self.target_velocity
        
        return {
            'position': target_pos,
            'velocity': target_vel,
            'yaw': np.arctan2(direction[1], direction[0]),  # Face towards gate
            'gate_distance': distance,
            'confidence': perception.get('confidence', 0.0)
        }
    
    def _gate_to_world_position(self, gate_info: Dict, 
                                current_state: Dict) -> np.ndarray:
        """
        Convert gate pixel coordinates to world position.
        
        This is simplified - real implementation needs:
        - Camera intrinsic calibration
        - Depth estimation from gate size
        - Proper coordinate transformation
        """
        # Get gate center in image coordinates
        center_x, center_y = gate_info['center']
        area = gate_info['area']
        
        # Estimate distance from gate area (inverse square law)
        # Larger area = closer gate
        # This is a rough heuristic
        estimated_distance = np.sqrt(10000.0 / max(area, 1.0))
        estimated_distance = np.clip(estimated_distance, 1.0, 20.0)
        
        # Assume camera is centered on drone
        # Image coordinates to normalized device coordinates
        # (this assumes 320x240 image from config)
        image_width = 320
        image_height = 240
        
        # Normalized coordinates (-1 to 1)
        # Right-bias correction: shift center_x RIGHT to increase rightward correction
        # This counteracts left-of-gate drift by exaggerating gate right offset.
        center_x_corrected = center_x + self.camera_center_bias_px
        norm_x = (center_x_corrected - image_width / 2) / (image_width / 2)
        norm_y = (center_y - image_height / 2) / (image_height / 2)
        
        # Assume 90-degree FOV for simplicity
        # Real implementation needs camera calibration matrix
        angle_x = norm_x * (np.pi / 4)  # 45 degrees half FOV
        angle_y = -norm_y * (np.pi / 4)  # Negative because image y is down
        
        # Convert to world coordinates (relative to drone)
        # Assuming drone is facing +X direction
        current_pos = current_state['position']
        current_yaw = current_state['orientation'][2]  # Yaw from state
        
        # Relative position from drone perspective
        rel_x = estimated_distance * np.cos(angle_y)
        rel_y = estimated_distance * np.sin(angle_x) * np.cos(angle_y)
        rel_z = estimated_distance * np.sin(angle_y)
        
        # Rotate by drone yaw to get world coordinates
        world_x = current_pos[0] + rel_x * np.cos(current_yaw) - rel_y * np.sin(current_yaw)
        world_y = current_pos[1] + rel_x * np.sin(current_yaw) + rel_y * np.cos(current_yaw)
        world_z = current_pos[2] + rel_z
        
        return np.array([world_x, world_y, world_z])
    
    def _fallback_target(self, current_state: Dict) -> Dict[str, Any]:
        """Generate fallback target when no gates detected."""
        current_pos = current_state['position']
        current_vel = current_state['velocity']
        current_yaw = current_state['orientation'][2]
        
        # CRITICAL: Don't try to stop - that causes aggressive pitch-back and climbing!
        # Instead, continue moving forward at racing speed
        forward_dir = np.array([np.cos(current_yaw), np.sin(current_yaw), 0.0])
        
        # Target is lookahead distance forward along current heading
        target_pos = current_pos + forward_dir * self.lookahead_distance
        target_pos[2] = max(current_pos[2], 0.8)  # Maintain current altitude in fallback
        
        return {
            'position': target_pos,
            'velocity': forward_dir * self.target_velocity,  # Keep racing forward
            'yaw': current_yaw,
            'gate_distance': float('inf'),
            'confidence': 0.0
        }
    
    def reset(self):
        """Reset planner state."""
        self.waypoints = []
        self.current_waypoint_idx = 0

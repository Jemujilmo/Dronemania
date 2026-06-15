"""
Main autonomy stack orchestrator.

This is the entry point that:
1. Loads configuration
2. Initializes all modules (perception, state estimation, planning, control)
3. Manages the main control loop
4. Ensures deterministic behavior
5. Handles graceful shutdown

Designed for AI-GP compliance - no human intervention required.
"""

import os
import sys
import time
import yaml
import logging
import numpy as np
from typing import Dict, Any, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perception.gate_detector import GateDetector
from state_estimation.ekf_estimator import EKFEstimator
from planning.waypoint_planner import WaypointPlanner
from control.pid_controller import PIDController
from simulation.sim_interface import PyBulletDroneEnv


class AutonomyStack:
    """Main autonomy stack orchestrator."""
    
    def __init__(self, config_path: str):
        """Initialize autonomy stack with configuration."""
        self.config = self._load_config(config_path)
        self._setup_logging()
        self._set_deterministic()
        
        self.logger.info("🚁 Initializing Dronemania Autonomy Stack")
        
        # Initialize modules
        self.simulator = PyBulletDroneEnv(self.config['simulator'])
        self.perception = GateDetector(self.config['perception'])
        self.state_estimator = EKFEstimator(self.config['state_estimation'])
        self.planner = WaypointPlanner(self.config['planning'])
        self.controller = PIDController(self.config['control'])
        
        # Control loop parameters
        self.control_rate = self.config['autonomy']['control_rate_hz']
        self.dt = 1.0 / self.control_rate
        
        # State
        self.running = False
        self.emergency_stop = False
        
        self.logger.info("✅ Autonomy stack initialized successfully")
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = getattr(logging, self.config['logging']['level'])
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('AutonomyStack')
        
    def _set_deterministic(self):
        """Set deterministic behavior for reproducibility."""
        if self.config['autonomy'].get('deterministic', False):
            np.random.seed(42)
            # Add torch seed if using neural networks
            # torch.manual_seed(42)
            os.environ['PYTHONHASHSEED'] = '0'
    
    def run(self):
        """Main control loop - runs until completion or failure."""
        self.running = True
        self.logger.info("🚀 Starting main control loop")
        
        # Reset simulator and get initial state
        initial_observation = self.simulator.reset()
        
        loop_count = 0
        start_time = time.time()
        
        try:
            while self.running and not self.emergency_stop:
                loop_start = time.time()
                
                # 1. Get sensor data
                observation = self.simulator.get_observation()
                
                # 2. Perception: Detect gates and obstacles
                perception_output = self.perception.process(observation['camera'])
                
                # 3. State Estimation: Fuse sensor data
                state_estimate = self.state_estimator.update(
                    imu_data=observation.get('imu'),
                    vision_data=perception_output,
                    dt=self.dt
                )
                
                # 4. Planning: Generate target waypoint
                target = self.planner.plan(
                    current_state=state_estimate,
                    perception=perception_output
                )
                
                # 5. Control: Compute control commands
                control_command = self.controller.compute(
                    current_state=state_estimate,
                    target=target,
                    dt=self.dt
                )
                
                # 6. Send command to simulator
                terminated, truncated, info = self.simulator.send_command(control_command)
                
                # 7. Check termination conditions
                if terminated or truncated:
                    self.logger.info(f"Episode ended: {info}")
                    self.running = False
                
                # 8. Rate limiting
                loop_count += 1
                elapsed = time.time() - loop_start
                sleep_time = max(0, self.dt - elapsed)
                time.sleep(sleep_time)
                
                # Log stats periodically
                if loop_count % 100 == 0:
                    hz = loop_count / (time.time() - start_time)
                    self.logger.debug(f"Loop {loop_count}, Rate: {hz:.1f} Hz")
                
        except KeyboardInterrupt:
            self.logger.info("⚠️  Interrupted by user")
        except Exception as e:
            self.logger.error(f"❌ Error in main loop: {e}", exc_info=True)
            self.emergency_stop = True
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Graceful shutdown of all modules."""
        self.logger.info("🛑 Shutting down autonomy stack")
        self.running = False
        
        # Close all modules in reverse order
        if hasattr(self, 'simulator'):
            self.simulator.close()
        
        self.logger.info("✅ Shutdown complete")


def main():
    """Entry point for autonomy stack."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Dronemania Autonomy Stack')
    parser.add_argument(
        '--config',
        type=str,
        default='config/autonomy.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # Create and run autonomy stack
    stack = AutonomyStack(args.config)
    stack.run()


if __name__ == "__main__":
    main()

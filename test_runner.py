#!/usr/bin/env python3
"""
Test runner for drone racing simulator.

Launches PyBullet simulation with autonomy stack.
Usage:
    python test_runner.py --mode headless  # No GUI
    python test_runner.py --mode gui       # With visual feedback
"""

import argparse
import sys
import os
import yaml

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from simulation.racing_env import DroneRacingEnv


def run_simulation(config_path: str, gui: bool = True, num_episodes: int = 1):
    """Run simulation with racing environment."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create environment
    env = DroneRacingEnv(config, render_mode='human' if gui else None)
    
    print(f"🚁 Starting drone racing simulation ({'GUI' if gui else 'HEADLESS'})")
    print(f"   Episodes: {num_episodes}")
    print(f"   Action space: {env.action_space}")
    print(f"   Observation space: {env.observation_space}")
    
    try:
        for episode in range(num_episodes):
            print(f"\n📍 Episode {episode + 1}/{num_episodes}")
            
            obs, info = env.reset()
            print(f"   Reset complete. Obs shape: {obs.shape}")
            
            episode_reward = 0
            step = 0
            max_steps = 5000
            
            while step < max_steps:
                # Random action for testing (replace with autonomy stack)
                # For now, move forward and try to follow gates
                action = [2.0, 0.0, 0.0, 0.0]  # Move forward
                
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                step += 1
                
                if step % 100 == 0:
                    print(f"   Step {step}: Reward={episode_reward:.2f} | Info={info}")
                
                if terminated or truncated:
                    print(f"   Episode ended: {info}")
                    break
            
            print(f"   Episode complete - Total reward: {episode_reward:.2f}")
            
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    finally:
        env.close()
        print("\n✅ Simulation closed")


def main():
    parser = argparse.ArgumentParser(description='Drone Racing Simulator Test')
    parser.add_argument('--config', type=str, default='config/autonomy.yaml',
                       help='Path to configuration file')
    parser.add_argument('--mode', type=str, default='gui', choices=['gui', 'headless'],
                       help='Rendering mode')
    parser.add_argument('--episodes', type=int, default=1,
                       help='Number of episodes to run')
    
    args = parser.parse_args()
    
    run_simulation(
        config_path=args.config,
        gui=(args.mode == 'gui'),
        num_episodes=args.episodes
    )


if __name__ == "__main__":
    main()

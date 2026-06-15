"""
Main training script for drone racing AI.

For the automated self-improving loop use self_train.py in the project root.
This script is a single-run convenience wrapper.
"""

import os
import sys
import yaml
import pathlib
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from simulation.drone_gym_env import RacingEnv  # noqa


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def make_env(config: dict):
    """Create environment factory."""
    def _init():
        return RacingEnv(substeps=config.get('substeps', 4))
    return _init


def main():
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'training_config.yaml')
    config = load_config(config_path)
    
    print("Dronemania Training")
    print(f"Algorithm: {config['training']['algorithm']}")
    print(f"Total timesteps: {config['training']['total_timesteps']}")
    
    # Create environment
    env = DummyVecEnv([make_env(config)])
    
    # Create model (MlpPolicy — obs is 20-float state vector, not images)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config['training']['learning_rate'],
        n_steps=config['training']['n_steps'],
        batch_size=config['training']['batch_size'],
        gamma=config['training']['gamma'],
        verbose=1,
        tensorboard_log=config['logging']['tensorboard_log']
    )
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=config['logging']['save_freq'],
        save_path=config['logging']['checkpoint_dir'],
        name_prefix='drone_model'
    )
    
    # Train the model
    print("\n📚 Training started...")
    model.learn(
        total_timesteps=config['training']['total_timesteps'],
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    # Save final model
    final_model_path = os.path.join(config['logging']['checkpoint_dir'], 'drone_model_final')
    model.save(final_model_path)
    print(f"\n✅ Training complete! Model saved to {final_model_path}")
    
    env.close()


if __name__ == "__main__":
    main()

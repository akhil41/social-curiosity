"""
Default configuration for Social Curiosity tabular experiments.
Contains hyperparameters for Q-learning training and environment settings.
"""
from typing import Dict, Any

# Q-Learning Hyperparameters
QLEARNING_CONFIG: Dict[str, Any] = {
    # Learning parameters
    'learning_rate': 0.1,          # Learning rate (α) for Q-learning updates
    'discount_factor': 0.95,       # Discount factor (γ) for future rewards
    'initial_epsilon': 1.0,        # Initial exploration rate for ε-greedy policy
    'epsilon_decay': 0.999,        # Decay rate for epsilon after each episode
    'min_epsilon': 0.01,           # Minimum exploration rate
}

# Environment Configuration
ENV_CONFIG: Dict[str, Any] = {
    'grid_size': 5,                # Size of the GridWorld (5x5)
    'max_episode_steps': 100,      # Maximum steps per episode
}

# Training Configuration
TRAINING_CONFIG: Dict[str, Any] = {
    'episodes': 50000,             # Number of episodes to train for (extended duration)
    'log_interval': 100,           # Log progress every N episodes
    'save_interval': 1000,         # Save model every N episodes
    'eval_interval': 500,          # Evaluate model every N episodes
    'n_eval_episodes': 10,         # Number of episodes for evaluation
}

# Social Intrinsic Motivation Configuration
SIM_CONFIG: Dict[str, Any] = {
    'intrinsic_coef': 0.2,         # Bonus coefficient for social intrinsic motivation
}

# Path Configuration
PATH_CONFIG: Dict[str, str] = {
    'models_dir': './results/tabular/models/',  # Directory to save trained models
    'logs_dir': './results/tabular/logs/',      # Directory for training logs
    'plots_dir': './plots/tabular/',            # Directory for generated plots
    'checkpoints_dir': './results/tabular/checkpoints/',  # Directory for training checkpoints
}

# Default configuration for experiments
DEFAULT_CONFIG: Dict[str, Any] = {
    'qlearning': QLEARNING_CONFIG,
    'env': ENV_CONFIG,
    'training': TRAINING_CONFIG,
    'sim': SIM_CONFIG,
    'paths': PATH_CONFIG,
}

def get_config() -> Dict[str, Any]:
    """
    Get the default configuration for tabular experiments.
    
    Returns:
        Dictionary containing all configuration parameters
    """
    return DEFAULT_CONFIG.copy()

def update_config(base_config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update a configuration dictionary with new values.
    
    Args:
        base_config: Base configuration to update
        updates: Dictionary of updates to apply
        
    Returns:
        Updated configuration
    """
    config = base_config.copy()
    for key, value in updates.items():
        if key in config and isinstance(config[key], dict) and isinstance(value, dict):
            config[key] = update_config(config[key], value)
        else:
            config[key] = value
    return config
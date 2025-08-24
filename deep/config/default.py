"""
Default configuration for Social Curiosity deep learning experiments.
Contains hyperparameters for PPO training and environment settings.
"""
from typing import Dict, Any

# PPO Hyperparameters
PPO_CONFIG: Dict[str, Any] = {
    # Learning parameters
    'learning_rate': 3e-4,
    'n_steps': 2048,           # Number of steps to run for each environment per update
    'batch_size': 64,          # Minibatch size
    'n_epochs': 10,            # Number of epoch when optimizing the surrogate loss
    
    # Advantage estimation
    'gamma': 0.99,             # Discount factor
    'gae_lambda': 0.95,        # Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    
    # Clipping parameters
    'clip_range': 0.2,         # Clipping parameter for the policy
    'clip_range_vf': None,     # Clipping parameter for the value function
    
    # Optimization parameters
    'normalize_advantage': True,
    'ent_coef': 0.0,           # Entropy coefficient for exploration
    'vf_coef': 0.5,            # Value function coefficient for loss calculation
    'max_grad_norm': 0.5,      # Maximum norm for gradient clipping
    
    # SDE (State-Dependent Exploration)
    'use_sde': False,
    'sde_sample_freq': -1,
    
    # Early stopping
    'target_kl': None,         # Target KL divergence threshold
    
    # Miscellaneous
    'verbose': 1,
    'seed': None,
    'device': 'cuda',          # Device to use for training ('cpu', 'cuda', 'auto')
}

# Environment Configuration
ENV_CONFIG: Dict[str, Any] = {
    'max_episode_steps': 100,  # Maximum steps per episode
}

# Training Configuration
TRAINING_CONFIG: Dict[str, Any] = {
    'total_timesteps': 1_000_000,  # Total number of timesteps to train for (extended duration)
    'log_interval': 10,            # Log progress every N episodes
    'save_interval': 100,          # Save model every N episodes
    'eval_interval': 500,          # Evaluate model every N episodes
    'n_eval_episodes': 10,         # Number of episodes for evaluation
}

# Social Intrinsic Motivation Configuration
SIM_CONFIG: Dict[str, Any] = {
    'intrinsic_coef': 0.2,         # Bonus coefficient for social intrinsic motivation
}

# Logging Configuration
LOGGING_CONFIG: Dict[str, Any] = {
    'tensorboard_log': './logs/tensorboard/',  # TensorBoard log directory
    'wandb_project': 'social-curiosity-deep',  # WandB project name
    'wandb_entity': None,                      # WandB entity/username
    'wandb_tags': ['ppo', 'multi-agent', 'social-curiosity'],
}

# Model Architecture Configuration
MODEL_CONFIG: Dict[str, Any] = {
    'features_dim': 64,        # Dimension of extracted features
    'policy_network': [64, 64], # Architecture of policy network
    'value_network': [64, 64],  # Architecture of value network
}

# Path Configuration
PATH_CONFIG: Dict[str, str] = {
    'models_dir': './results/deep/models/',    # Directory to save trained models
    'logs_dir': './results/deep/logs/',        # Directory for training logs
    'plots_dir': './plots/deep/',              # Directory for generated plots
    'checkpoints_dir': './results/deep/checkpoints/',  # Directory for training checkpoints
}

# Default configuration for experiments
DEFAULT_CONFIG: Dict[str, Any] = {
    'ppo': PPO_CONFIG,
    'env': ENV_CONFIG,
    'training': TRAINING_CONFIG,
    'logging': LOGGING_CONFIG,
    'model': MODEL_CONFIG,
    'paths': PATH_CONFIG,
    'sim': SIM_CONFIG,
}

def get_config() -> Dict[str, Any]:
    """
    Get the default configuration for deep learning experiments.
    
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
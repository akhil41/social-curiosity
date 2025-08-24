"""
Configuration for deep learning implementation of Social Curiosity project.
Contains hyperparameters, model architectures, and training configurations.
"""

from .default import get_config, update_config, PPO_CONFIG, ENV_CONFIG, TRAINING_CONFIG, LOGGING_CONFIG, MODEL_CONFIG, PATH_CONFIG, DEFAULT_CONFIG

__all__ = [
    'get_config',
    'update_config',
    'PPO_CONFIG',
    'ENV_CONFIG',
    'TRAINING_CONFIG',
    'LOGGING_CONFIG',
    'MODEL_CONFIG',
    'PATH_CONFIG',
    'DEFAULT_CONFIG'
]
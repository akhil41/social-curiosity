"""
Deep learning-specific implementation code for Social Curiosity project.
Contains training scripts, PettingZoo environment, and stable-baselines3 agents.
"""

from .environment import SocialCuriosityEnv, make_env
from .agents import create_ppo_agent, load_agent, save_agent, CustomActorCriticPolicy
from .utils import SocialCuriosityCallback, setup_wandb, save_config, load_config

__all__ = [
    'SocialCuriosityEnv',
    'make_env',
    'create_ppo_agent',
    'load_agent',
    'save_agent',
    'CustomActorCriticPolicy',
    'SocialCuriosityCallback',
    'setup_wandb',
    'save_config',
    'load_config'
]
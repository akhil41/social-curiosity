"""
Agent definitions for Social Curiosity project using stable-baselines3.
Contains PPO agent configurations and policy networks for multi-agent environments.
"""
from typing import Dict, Any, Optional, List
import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for the Social Curiosity environment.
    This extracts meaningful features from the raw observation space.
    """
    
    def __init__(self, observation_space, features_dim: int = 64):
        """
        Initialize the custom feature extractor.
        
        Args:
            observation_space: The observation space of the environment
            features_dim: Dimension of the extracted features (default: 64)
        """
        super().__init__(observation_space, features_dim)
        
        # Calculate input dimension from observation space
        n_input_features = observation_space.shape[0]
        
        # Define the feature extraction network
        self.network = nn.Sequential(
            nn.Linear(n_input_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        Forward pass through the feature extractor.
        
        Args:
            observations: Input observations
            
        Returns:
            Extracted features
        """
        return self.network(observations)


class CustomActorCriticPolicy(ActorCriticPolicy):
    """
    Custom Actor-Critic policy for Social Curiosity environment.
    Uses the custom feature extractor and defines appropriate network architecture.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize the custom policy.
        """
        super().__init__(
            *args,
            **kwargs,
            features_extractor_class=CustomFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=64),
            net_arch=dict(pi=[64, 64], vf=[64, 64])
        )


def create_ppo_agent(env, config: Dict[str, Any]) -> PPO:
    """
    Create a PPO agent with the specified configuration.
    
    Args:
        env: The environment to train on
        config: Configuration dictionary containing hyperparameters
        
    Returns:
        PPO agent instance
    """
    return PPO(
        policy=CustomActorCriticPolicy,
        env=env,
        learning_rate=config.get('learning_rate', 3e-4),
        n_steps=config.get('n_steps', 2048),
        batch_size=config.get('batch_size', 64),
        n_epochs=config.get('n_epochs', 10),
        gamma=config.get('gamma', 0.99),
        gae_lambda=config.get('gae_lambda', 0.95),
        clip_range=config.get('clip_range', 0.2),
        clip_range_vf=config.get('clip_range_vf', None),
        normalize_advantage=config.get('normalize_advantage', True),
        ent_coef=config.get('ent_coef', 0.0),
        vf_coef=config.get('vf_coef', 0.5),
        max_grad_norm=config.get('max_grad_norm', 0.5),
        use_sde=config.get('use_sde', False),
        sde_sample_freq=config.get('sde_sample_freq', -1),
        target_kl=config.get('target_kl', None),
        tensorboard_log=config.get('tensorboard_log', None),
        policy_kwargs=config.get('policy_kwargs', dict()),
        verbose=config.get('verbose', 1),
        seed=config.get('seed', None),
        device=config.get('device', 'auto'),
        _init_setup_model=config.get('_init_setup_model', True),
    )


def create_multi_ppo_agents(env, config: Dict[str, Any], num_agents: int = 2) -> List[PPO]:
    """
    Create multiple PPO agents for multi-agent environment.
    
    Args:
        env: The environment to train on
        config: Configuration dictionary containing hyperparameters
        num_agents: Number of agents to create (default: 2)
        
    Returns:
        List of PPO agent instances
    """
    agents = []
    for i in range(num_agents):
        agent = create_ppo_agent(env, config)
        agents.append(agent)
    return agents


def load_agent(model_path: str, env=None) -> PPO:
    """
    Load a trained PPO agent from disk.
    
    Args:
        model_path: Path to the saved model
        env: Optional environment (if None, will need to set later)
        
    Returns:
        Loaded PPO agent
    """
    return PPO.load(model_path, env=env)


def save_agent(agent: PPO, model_path: str):
    """
    Save a trained PPO agent to disk.
    
    Args:
        agent: The PPO agent to save
        model_path: Path where to save the model
    """
    agent.save(model_path)


def load_multi_agents(model_paths: List[str], envs: Optional[List] = None) -> List[PPO]:
    """
    Load multiple trained PPO agents from disk.
    
    Args:
        model_paths: List of paths to the saved models
        envs: Optional list of environments (if None, will need to set later)
        
    Returns:
        List of loaded PPO agents
    """
    agents = []
    for i, model_path in enumerate(model_paths):
        env = envs[i] if envs and i < len(envs) else None
        agent = load_agent(model_path, env)
        agents.append(agent)
    return agents


def save_multi_agents(agents: List[PPO], model_paths: List[str]):
    """
    Save multiple trained PPO agents to disk.
    
    Args:
        agents: List of PPO agents to save
        model_paths: List of paths where to save the models
    """
    for agent, model_path in zip(agents, model_paths):
        save_agent(agent, model_path)
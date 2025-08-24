"""
PettingZoo environment wrapper for Social Curiosity project.
Uses simple_spread environment from PettingZoo's multi-agent particle environments.
"""
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pettingzoo.mpe import simple_spread_v3


class SocialCuriosityEnv(gym.Env):
    """
    Wrapper for PettingZoo's simple_spread environment with proper observation
    and action space definitions for stable-baselines3 integration.
    
    This environment wraps the multi-agent PettingZoo environment into a format
    suitable for single-agent RL algorithms from stable-baselines3.
    """
    
    def __init__(self, num_agents: int = 2, local_ratio: float = 0.5):
        """
        Initialize the Social Curiosity environment.
        
        Args:
            num_agents: Number of agents in the environment (default: 2)
            local_ratio: Ratio of local reward to global reward (default: 0.5)
        """
        self.num_agents = num_agents
        self.local_ratio = local_ratio
        
        # Create the underlying PettingZoo environment
        self.env = simple_spread_v3.parallel_env(
            N=num_agents,
            local_ratio=local_ratio,
            max_cycles=25,
            continuous_actions=False
        )
        
        # Get observation and action space from the first agent
        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()
        
        # Initialize state
        self.reset()
    
    def _get_observation_space(self) -> spaces.Box:
        """Get the observation space for a single agent."""
        # Get observation space from the first agent
        agent_id = list(self.env.observation_spaces.keys())[0]
        obs_space = self.env.observation_spaces[agent_id]
        return obs_space
    
    def _get_action_space(self) -> spaces.Discrete:
        """Get the action space for a single agent."""
        # Get action space from the first agent
        agent_id = list(self.env.action_spaces.keys())[0]
        action_space = self.env.action_spaces[agent_id]
        return action_space
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Optional random seed
            
        Returns:
            Tuple containing:
            - observation: Initial observation for the first agent
            - info: Additional information
        """
        observations, infos = self.env.reset(seed=seed)
        self.current_observations = observations
        self.current_agent_idx = 0
        return self._get_current_observation(), {}
    
    def _get_current_observation(self) -> np.ndarray:
        """Get observation for the current agent."""
        agent_id = list(self.current_observations.keys())[self.current_agent_idx]
        return self.current_observations[agent_id]
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one time step in the environment.
        
        Args:
            action: Action for the current agent
            
        Returns:
            Tuple containing:
            - observation: Next observation
            - reward: Reward for the current agent
            - terminated: Whether episode is terminated
            - truncated: Whether episode is truncated
            - info: Additional information
        """
        # Get current agent ID
        agent_ids = list(self.current_observations.keys())
        current_agent_id = agent_ids[self.current_agent_idx]
        
        # Store action for current agent
        actions = {current_agent_id: action}
        
        # If this is the last agent, execute the step
        if self.current_agent_idx == self.num_agents - 1:
            # Get actions for all agents (others take no-op action)
            for agent_id in agent_ids:
                if agent_id != current_agent_id:
                    actions[agent_id] = 0  # No-op action
            
            # Execute step in the underlying environment
            observations, rewards, terminations, truncations, infos = self.env.step(actions)
            
            # Store new observations
            self.current_observations = observations
            
            # Check if episode is done
            terminated = any(terminations.values())
            truncated = any(truncations.values())
            
            # Get reward for the current agent
            reward = rewards[current_agent_id]
            
            # Reset agent index for next step
            self.current_agent_idx = 0
            
        else:
            # Move to next agent, no environment step yet
            self.current_agent_idx += 1
            reward = 0.0
            terminated = False
            truncated = False
            infos = {}
        
        # Get observation for next agent
        next_observation = self._get_current_observation()
        
        return next_observation, reward, terminated, truncated, infos
    
    def render(self):
        """Render the environment."""
        self.env.render()
    
    def close(self):
        """Close the environment."""
        self.env.close()


# Factory function for creating the environment
def make_env(num_agents: int = 2, local_ratio: float = 0.5) -> SocialCuriosityEnv:
    """
    Create a Social Curiosity environment instance.
    
    Args:
        num_agents: Number of agents
        local_ratio: Ratio of local to global reward
        
    Returns:
        SocialCuriosityEnv instance
    """
    return SocialCuriosityEnv(num_agents=num_agents, local_ratio=local_ratio)
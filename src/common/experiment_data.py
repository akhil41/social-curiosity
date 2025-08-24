"""
Common data structures and classes for Social Curiosity experiments.
Contains shared data models for results, configurations, and experiment tracking.
"""
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
import numpy as np


@dataclass
class EpisodeResult:
    """Data class for storing episode-level results."""
    episode: int
    rewards: Tuple[float, float]  # (agent_a_reward, agent_b_reward)
    length: int
    success: bool
    sim_bonuses: Tuple[float, float] = (0.0, 0.0)  # (agent_a_bonus, agent_b_bonus)
    timestamp: Optional[float] = None


@dataclass
class ExperimentConfig:
    """Data class for storing experiment configuration."""
    run_name: str
    intrinsic_coef: float
    episodes: int
    seed: int
    algorithm: str  # 'tabular' or 'deep'
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    start_time: Optional[str] = None


@dataclass
class ExperimentResults:
    """Data class for storing complete experiment results."""
    config: ExperimentConfig
    episode_results: List[EpisodeResult] = field(default_factory=list)
    agent_stats: Dict[str, Any] = field(default_factory=dict)
    
    def add_episode_result(self, result: EpisodeResult) -> None:
        """Add an episode result to the experiment."""
        self.episode_results.append(result)
    
    def get_rewards_array(self) -> np.ndarray:
        """Get array of episode rewards for both agents."""
        return np.array([r.rewards for r in self.episode_results])
    
    def get_average_rewards(self) -> Tuple[float, float]:
        """Get average rewards for both agents across all episodes."""
        if not self.episode_results:
            return (0.0, 0.0)
        rewards = self.get_rewards_array()
        return (float(np.mean(rewards[:, 0])), float(np.mean(rewards[:, 1])))
    
    def get_success_rate(self) -> float:
        """Get overall success rate."""
        if not self.episode_results:
            return 0.0
        successes = [1 if r.success else 0 for r in self.episode_results]
        return float(np.mean(successes))
    
    def get_episode_lengths(self) -> List[int]:
        """Get list of episode lengths."""
        return [r.length for r in self.episode_results]
    
    def get_sim_bonuses_array(self) -> np.ndarray:
        """Get array of SIM bonuses for both agents."""
        return np.array([r.sim_bonuses for r in self.episode_results])


def calculate_moving_average(data: List[float], window_size: int = 10) -> List[float]:
    """
    Calculate moving average of a data series.
    
    Args:
        data: Input data series
        window_size: Size of the moving window
        
    Returns:
        Smoothed data series
    """
    if len(data) < window_size:
        return data
    
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid').tolist()


def calculate_success_rate(rewards: List[float], threshold: float = 8.0) -> float:
    """
    Calculate success rate from episode rewards.
    
    Args:
        rewards: List of episode rewards
        threshold: Reward threshold for success
        
    Returns:
        Success rate (0.0 to 1.0)
    """
    if not rewards:
        return 0.0
    successful_episodes = sum(1 for r in rewards if r >= threshold)
    return successful_episodes / len(rewards)
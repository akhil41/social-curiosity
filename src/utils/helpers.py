"""
Utility functions for Social Curiosity project.
Contains helper functions that can be used by both tabular and deep implementations.
"""
import os
import json
from typing import Dict, Any, List, Optional
from pathlib import Path


def setup_experiment_directory(run_name: str, base_dir: str = "results") -> Path:
    """
    Create and return experiment directory path.
    
    Args:
        run_name: Name of the experiment run
        base_dir: Base directory for results (default: "results")
        
    Returns:
        Path to experiment directory
    """
    exp_dir = Path(base_dir) / run_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def save_config(config: Dict[str, Any], path: str) -> None:
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        path: Path to save the configuration
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)


def load_config(path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        path: Path to the configuration file
        
    Returns:
        Loaded configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is not valid JSON
    """
    with open(path, 'r') as f:
        return json.load(f)


def save_results(results: Dict[str, Any], path: str) -> None:
    """
    Save experiment results to JSON file.
    
    Args:
        results: Results dictionary
        path: Path to save the results
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)


def moving_average(data: List[float], window_size: int = 10) -> List[float]:
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
    
    import numpy as np
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


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def ensure_directory_exists(path: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        path: Path to directory
    """
    os.makedirs(path, exist_ok=True)


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to project root directory
    """
    return Path(__file__).parent.parent.parent
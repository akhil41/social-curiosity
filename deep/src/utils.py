"""
Utility functions for Social Curiosity deep learning implementation.
Contains helper functions for logging, evaluation, and data processing.
"""
import os
import json
import numpy as np
from typing import Dict, Any, List, Optional
from tqdm import tqdm


from stable_baselines3.common.callbacks import BaseCallback

class SocialCuriosityCallback(BaseCallback):
    """
    Custom callback for Social Curiosity training.
    Handles logging, model saving, and evaluation.
    """
    
    def __init__(
        self,
        save_path: str,
        save_interval: int = 100,
        eval_env: Optional[Any] = None,
        eval_interval: int = 50,
        n_eval_episodes: int = 10,
        verbose: int = 1,
        model: Optional[Any] = None,
        wandb_run: Optional[Any] = None
    ):
        super().__init__(verbose)
        """
        Initialize the callback.
        
        Args:
            save_path: Path to save models
            save_interval: Save model every N episodes
            eval_env: Environment for evaluation
            eval_interval: Evaluate model every N episodes
            n_eval_episodes: Number of episodes for evaluation
            verbose: Verbosity level
            model: The RL model to save and evaluate
            wandb_run: WandB run object for logging
        """
        self.save_path = save_path
        self.save_interval = save_interval
        self.eval_env = eval_env
        self.eval_interval = eval_interval
        self.n_eval_episodes = n_eval_episodes  # Use the parameter passed in
        self.episode_count = 0
        self.verbose = verbose
        self.model = model
        self.wandb_run = wandb_run
        
        # Create save directory
        os.makedirs(save_path, exist_ok=True)
    
    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        
        Returns:
            bool: If the callback returns False, training is aborted early.
        """
        # Check if episode ended
        if self.locals.get('done', False):
            self.episode_count += 1
            
            # Save model at specified intervals
            if self.episode_count % self.save_interval == 0:
                model_path = os.path.join(self.save_path, f"model_{self.episode_count}")
                if hasattr(self, 'model') and self.model is not None:
                    self.model.save(model_path)
                    if self.verbose >= 1:
                        print(f"Model saved to {model_path}")
            
            # Evaluate model at specified intervals
            if (self.eval_env is not None and
                self.episode_count % self.eval_interval == 0):
                self._evaluate_model()
        
        return True
    
    def _evaluate_model(self):
        """Evaluate the current model."""
        if not hasattr(self, 'model'):
            return
            
        if self.verbose >= 1:
            print(f"Evaluating model after {self.episode_count} episodes...")
        
        total_rewards = []
        for episode in tqdm(range(self.n_eval_episodes), desc="Evaluating", leave=False):
            obs, _ = self.eval_env.reset()
            episode_reward = 0.0
            terminated = False
            truncated = False
            
            while not (terminated or truncated):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                episode_reward += reward
            
            total_rewards.append(episode_reward)
        
        avg_reward = sum(total_rewards) / len(total_rewards)
        success_rate = calculate_success_rate(total_rewards)
        
        if self.verbose >= 1:
            print(f"Evaluation results - Avg reward: {avg_reward:.2f}, Success rate: {success_rate:.2f}")
        
        # Log to WandB if available
        if hasattr(self, 'wandb_run') and self.wandb_run:
            self.wandb_run.log({
                "eval/avg_reward": avg_reward,
                "eval/success_rate": success_rate,
                "eval/episode": self.episode_count
            })


def setup_wandb(config: Dict[str, Any], run_name: str) -> Optional[Any]:
    """
    Setup Weights & Biases logging.
    
    Args:
        config: Configuration dictionary
        run_name: Name of the training run
        
    Returns:
        WandB run object or None if not configured
    """
    wandb_config = config.get('logging', {})
    
    if wandb_config.get('wandb_project'):
        try:
            import wandb
            wandb.init(
                project=wandb_config['wandb_project'],
                entity=wandb_config.get('wandb_entity'),
                name=run_name,
                tags=wandb_config.get('wandb_tags', []),
                config=config
            )
            return wandb
        except ImportError:
            print("WandB not available, skipping logging")
    
    return None


def save_config(config: Dict[str, Any], path: str):
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
    """
    with open(path, 'r') as f:
        return json.load(f)


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
    
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid').tolist()
#!/usr/bin/env python3
"""
Training script for Social Curiosity in Multi-Agent GridWorld (Tabular version).

Supports both baseline (intrinsic_coef=0.0) and SIM (intrinsic_coef=0.2) experiments.
Trains for 10k episodes with proper logging, checkpointing, and result tracking.
"""

import argparse
import numpy as np
import time
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tabular.src.environment import GridWorldEnv
from tabular.src.agents import TabularQLearningAgent
from tabular.src.social_motivation import SocialIntrinsicMotivation


class TrainingLogger:
    """Logger for training progress and results."""
    
    def __init__(self, run_name: str, results_dir: str = "results/tabular"):
        """
        Initialize the training logger.
        
        Args:
            run_name: Name of the training run
            results_dir: Base directory for saving results
        """
        self.run_name = run_name
        self.results_dir = Path(results_dir) / run_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tracking dictionaries
        self.episode_rewards: List[Tuple[float, float]] = []
        self.episode_lengths: List[int] = []
        self.success_rates: List[bool] = []
        self.exploration_stats: List[Dict[str, Any]] = []
        self.sim_bonuses: List[Tuple[float, float]] = []
        
        # Create log file
        self.log_file = self.results_dir / "training_log.txt"
        self.config_file = self.results_dir / "config.json"
        self.results_file = self.results_dir / "results.json"
        
    def log_config(self, config: Dict[str, Any]) -> None:
        """Log training configuration."""
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def log_episode(self, 
                   episode: int, 
                   rewards: Tuple[float, float], 
                   length: int, 
                   success: bool,
                   agent_stats: Dict[str, Any],
                   sim_bonus: Tuple[float, float] = (0.0, 0.0)) -> None:
        """
        Log episode results.
        
        Args:
            episode: Episode number
            rewards: Tuple of (reward_a, reward_b)
            length: Episode length in steps
            success: Whether episode was successful
            agent_stats: Agent statistics
            sim_bonus: SIM bonuses (bonus_a, bonus_b)
        """
        self.episode_rewards.append(rewards)
        self.episode_lengths.append(length)
        self.success_rates.append(success)
        self.sim_bonuses.append(sim_bonus)
        
        # Add exploration stats
        stats = {
            'episode': episode,
            'agent_a_epsilon': agent_stats.get('agent_a_epsilon', 0.0),
            'agent_b_epsilon': agent_stats.get('agent_b_epsilon', 0.0),
            'agent_a_states': agent_stats.get('agent_a_states', 0),
            'agent_b_states': agent_stats.get('agent_b_states', 0),
            'agent_a_visits': agent_stats.get('agent_a_visits', 0),
            'agent_b_visits': agent_stats.get('agent_b_visits', 0),
        }
        self.exploration_stats.append(stats)
        
        # Log to file every 100 episodes
        if episode % 100 == 0:
            with open(self.log_file, 'a') as f:
                f.write(f"Episode {episode}: "
                       f"Rewards A={rewards[0]:.2f}, B={rewards[1]:.2f}, "
                       f"Length={length}, Success={success}, "
                       f"SIM Bonus A={sim_bonus[0]:.2f}, B={sim_bonus[1]:.2f}\n")
    
    def save_results(self) -> None:
        """Save all results to JSON file."""
        results = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'success_rates': self.success_rates,
            'sim_bonuses': self.sim_bonuses,
            'exploration_stats': self.exploration_stats,
            'total_episodes': len(self.episode_rewards)
        }
        
        with open(self.results_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    def print_progress(self, episode: int, total_episodes: int, 
                      rewards: Tuple[float, float], length: int) -> None:
        """Print training progress."""
        if episode % 100 == 0:
            avg_reward_a = np.mean([r[0] for r in self.episode_rewards[-100:]])
            avg_reward_b = np.mean([r[1] for r in self.episode_rewards[-100:]])
            success_rate = np.mean(self.success_rates[-100:]) * 100
            
            print(f"Episode {episode}/{total_episodes} | "
                  f"Avg Rewards: A={avg_reward_a:.2f}, B={avg_reward_b:.2f} | "
                  f"Length: {length} | Success: {success_rate:.1f}%")


def train_agents(args: argparse.Namespace) -> None:
    """
    Main training function for both baseline and SIM experiments.
    
    Args:
        args: Command-line arguments
    """
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    
    # Initialize environment, agents, and SIM module
    env = GridWorldEnv()
    agent_a = TabularQLearningAgent(
        learning_rate=args.learning_rate,
        initial_epsilon=args.epsilon_start,
        epsilon_decay=args.epsilon_decay,
        min_epsilon=args.epsilon_min
    )
    agent_b = TabularQLearningAgent(
        learning_rate=args.learning_rate,
        initial_epsilon=args.epsilon_start,
        epsilon_decay=args.epsilon_decay,
        min_epsilon=args.epsilon_min
    )
    
    sim = SocialIntrinsicMotivation(intrinsic_coef=args.intrinsic_coef)
    
    # Initialize logger
    logger = TrainingLogger(args.run_name)
    
    # Log configuration
    config = {
        'run_name': args.run_name,
        'intrinsic_coef': args.intrinsic_coef,
        'episodes': args.episodes,
        'learning_rate': args.learning_rate,
        'epsilon_start': args.epsilon_start,
        'epsilon_decay': args.epsilon_decay,
        'epsilon_min': args.epsilon_min,
        'seed': args.seed,
        'start_time': datetime.now().isoformat()
    }
    logger.log_config(config)
    
    print(f"Starting training: {args.run_name}")
    print(f"Intrinsic coefficient: {args.intrinsic_coef}")
    print(f"Episodes: {args.episodes}")
    print(f"Seed: {args.seed}")
    print("-" * 50)
    
    start_time = time.time()
    
    # Training loop
    for episode in range(1, args.episodes + 1):
        # Reset environment and SIM
        state = env.reset()
        sim.reset()
        done = False
        episode_reward_a = 0.0
        episode_reward_b = 0.0
        episode_length = 0
        total_sim_bonus_a = 0.0
        total_sim_bonus_b = 0.0
        
        while not done:
            # Store previous state for SIM calculation
            prev_state = state.copy()
            
            # Agents choose actions
            action_a = agent_a.choose_action(state)
            action_b = agent_b.choose_action(state)
            
            # Environment step
            next_state, (reward_a, reward_b), done, info = env.step(action_a, action_b)
            
            # Update SIM visited cells
            sim.update(prev_state, next_state, 'A')
            sim.update(prev_state, next_state, 'B')
            
            # Calculate SIM bonuses if using SIM
            sim_bonus_a, sim_bonus_b = (0.0, 0.0)
            if args.intrinsic_coef > 0:
                sim_bonus_a, sim_bonus_b = sim.calculate_bonus(prev_state, next_state, 'A', 'B')
                total_sim_bonus_a += sim_bonus_a
                total_sim_bonus_b += sim_bonus_b
            
            # Combine extrinsic and intrinsic rewards
            total_reward_a = reward_a + sim_bonus_a
            total_reward_b = reward_b + sim_bonus_b
            
            # Agent learning updates
            agent_a.update(state, action_a, total_reward_a, next_state, done)
            agent_b.update(state, action_b, total_reward_b, next_state, done)
            
            # Update state and track rewards
            state = next_state
            episode_reward_a += total_reward_a
            episode_reward_b += total_reward_b
            episode_length += 1
        
        # Decay exploration rates
        agent_a.decay_epsilon()
        agent_b.decay_epsilon()
        
        # Check if episode was successful (both coins collected)
        success = all(info['coins_taken'])
        
        # Get agent statistics
        agent_stats = {
            'agent_a_epsilon': agent_a.epsilon,
            'agent_b_epsilon': agent_b.epsilon,
            'agent_a_states': agent_a.get_stats()['total_states'],
            'agent_b_states': agent_b.get_stats()['total_states'],
            'agent_a_visits': agent_a.get_stats()['total_visits'],
            'agent_b_visits': agent_b.get_stats()['total_visits'],
        }
        
        # Log episode results
        logger.log_episode(
            episode=episode,
            rewards=(episode_reward_a, episode_reward_b),
            length=episode_length,
            success=success,
            agent_stats=agent_stats,
            sim_bonus=(total_sim_bonus_a, total_sim_bonus_b)
        )
        
        # Print progress
        logger.print_progress(episode, args.episodes, 
                             (episode_reward_a, episode_reward_b), episode_length)
        
        # Save checkpoint every 1000 episodes
        if episode % 1000 == 0:
            checkpoint_dir = logger.results_dir / f"checkpoint_ep{episode}"
            checkpoint_dir.mkdir(exist_ok=True)
            
            agent_a.save(str(checkpoint_dir / "agent_a.pkl"))
            agent_b.save(str(checkpoint_dir / "agent_b.pkl"))
            
            # Save SIM state if using SIM
            if args.intrinsic_coef > 0:
                sim_data = {
                    'visited_cells': sim.visited_cells,
                    'intrinsic_coef': sim.intrinsic_coef
                }
                with open(checkpoint_dir / "sim.pkl", 'wb') as f:
                    import pickle
                    pickle.dump(sim_data, f)
    
    # Training completed
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    # Save final agents and results
    final_dir = logger.results_dir / "final"
    final_dir.mkdir(exist_ok=True)
    
    agent_a.save(str(final_dir / "agent_a.pkl"))
    agent_b.save(str(final_dir / "agent_b.pkl"))
    
    if args.intrinsic_coef > 0:
        sim_data = {
            'visited_cells': sim.visited_cells,
            'intrinsic_coef': sim.intrinsic_coef
        }
        with open(final_dir / "sim.pkl", 'wb') as f:
            import pickle
            pickle.dump(sim_data, f)
    
    logger.save_results()
    
    # Print final statistics
    avg_reward_a = np.mean([r[0] for r in logger.episode_rewards])
    avg_reward_b = np.mean([r[1] for r in logger.episode_rewards])
    avg_length = np.mean(logger.episode_lengths)
    success_rate = np.mean(logger.success_rates) * 100
    
    print(f"\nFinal Statistics:")
    print(f"Average Reward - Agent A: {avg_reward_a:.2f}")
    print(f"Average Reward - Agent B: {avg_reward_b:.2f}")
    print(f"Average Episode Length: {avg_length:.1f}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Agent A States: {agent_a.get_stats()['total_states']}")
    print(f"Agent B States: {agent_b.get_stats()['total_states']}")
    print(f"Results saved to: {logger.results_dir}")


def main():
    """Main function to parse arguments and start training."""
    parser = argparse.ArgumentParser(description="Train agents in Social Curiosity GridWorld")
    
    # Experiment configuration
    parser.add_argument("--run_name", type=str, required=True,
                       help="Name for this run (e.g., 'baseline', 'sim')")
    parser.add_argument("--intrinsic_coef", type=float, default=0.0,
                       help="SIM bonus coefficient (0.0 for baseline, 0.2 for SIM)")
    parser.add_argument("--episodes", type=int, default=10000,
                       help="Number of episodes to train")
    
    # Learning parameters
    parser.add_argument("--learning_rate", type=float, default=0.1,
                       help="Learning rate for Q-learning")
    parser.add_argument("--epsilon_start", type=float, default=1.0,
                       help="Initial exploration rate")
    parser.add_argument("--epsilon_decay", type=float, default=0.999,
                       help="Exploration rate decay")
    parser.add_argument("--epsilon_min", type=float, default=0.01,
                       help="Minimum exploration rate")
    
    # Reproducibility
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.intrinsic_coef < 0:
        raise ValueError("intrinsic_coef must be non-negative")
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    if args.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    
    # Create results directory
    results_dir = Path("results/tabular")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Start training
    train_agents(args)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Create sample experimental data for Social Curiosity project plotting demonstration.

This script generates realistic sample data for both tabular and deep learning experiments
to demonstrate the plotting capabilities.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import os


def create_tabular_sample_data():
    """Create sample data for tabular experiments."""
    print("Creating sample tabular experimental data...")

    # Create results directories
    results_dir = Path("results/tabular")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Configuration for baseline experiment
    baseline_config = {
        'run_name': 'baseline',
        'intrinsic_coef': 0.0,
        'learning_rate': 0.1,
        'discount_factor': 0.9,
        'exploration_rate': 0.1,
        'total_episodes': 1000,
        'grid_size': 5,
        'max_steps': 100
    }

    # Configuration for SIM experiment
    sim_config = {
        'run_name': 'sim',
        'intrinsic_coef': 0.2,
        'learning_rate': 0.1,
        'discount_factor': 0.9,
        'exploration_rate': 0.1,
        'total_episodes': 1000,
        'grid_size': 5,
        'max_steps': 100
    }

    # Generate baseline results
    baseline_results = generate_tabular_experiment_data(baseline_config, sim_bonus=False)
    baseline_dir = results_dir / "baseline"
    baseline_dir.mkdir(exist_ok=True)

    with open(baseline_dir / "config.json", 'w') as f:
        json.dump(baseline_config, f, indent=2)

    with open(baseline_dir / "results.json", 'w') as f:
        json.dump(baseline_results, f, indent=2)

    # Generate SIM results
    sim_results = generate_tabular_experiment_data(sim_config, sim_bonus=True)
    sim_dir = results_dir / "sim"
    sim_dir.mkdir(exist_ok=True)

    with open(sim_dir / "config.json", 'w') as f:
        json.dump(sim_config, f, indent=2)

    with open(sim_dir / "results.json", 'w') as f:
        json.dump(sim_results, f, indent=2)

    print(f"Created tabular sample data in {results_dir}")


def create_deep_sample_data():
    """Create sample data for deep learning experiments."""
    print("Creating sample deep learning experimental data...")

    # Create results directories
    results_dir = Path("results/deep")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Generate baseline results
    baseline_dir = results_dir / "baseline"
    baseline_dir.mkdir(exist_ok=True)

    baseline_data = generate_deep_experiment_data("baseline", intrinsic_coef=0.0)
    with open(baseline_dir / "results.json", 'w') as f:
        json.dump(baseline_data, f, indent=2)

    # Generate SIM results
    sim_dir = results_dir / "sim"
    sim_dir.mkdir(exist_ok=True)

    sim_data = generate_deep_experiment_data("sim", intrinsic_coef=0.2)
    with open(sim_dir / "results.json", 'w') as f:
        json.dump(sim_data, f, indent=2)

    print(f"Created deep learning sample data in {results_dir}")


def generate_tabular_experiment_data(config: Dict[str, Any], sim_bonus: bool = False) -> Dict[str, Any]:
    """Generate realistic tabular experiment data."""
    total_episodes = config['total_episodes']
    intrinsic_coef = config['intrinsic_coef']

    # Generate episode rewards with realistic learning curves
    episode_rewards = []
    episode_lengths = []
    success_rates = []
    sim_bonuses = []
    exploration_stats = []

    # Learning curve parameters
    baseline_reward = -10  # Starting reward (mostly failures)
    max_reward = 15  # Peak reward when agents cooperate well

    for episode in range(total_episodes):
        # Learning progress (sigmoid-like curve)
        progress = episode / total_episodes
        learning_factor = 1 / (1 + np.exp(-6 * (progress - 0.3)))  # Sigmoid

        # Add some noise and variability
        noise = np.random.normal(0, 2)

        # Calculate rewards for both agents
        base_reward = baseline_reward + (max_reward - baseline_reward) * learning_factor + noise

        # Agent A reward
        reward_a = base_reward + np.random.normal(0, 1)

        # Agent B reward (slightly correlated with Agent A)
        reward_b = base_reward * 0.8 + reward_a * 0.2 + np.random.normal(0, 1)

        episode_rewards.append([float(reward_a), float(reward_b)])

        # Episode length (decreases as agents learn)
        max_length = config['max_steps']
        length = int(max_length * (1 - learning_factor * 0.7) + np.random.randint(5, 15))
        episode_lengths.append(length)

        # Success rate (increases with learning)
        success_prob = learning_factor * 0.9 + 0.05  # Max 95% success
        success = np.random.random() < success_prob
        success_rates.append(int(success))

        # SIM bonuses (only if sim_bonus is True)
        if sim_bonus and intrinsic_coef > 0:
            # Social curiosity bonus when agents help each other explore
            bonus_a = np.random.exponential(intrinsic_coef * 2) if np.random.random() < 0.3 else 0
            bonus_b = np.random.exponential(intrinsic_coef * 2) if np.random.random() < 0.3 else 0
        else:
            bonus_a = 0.0
            bonus_b = 0.0
        sim_bonuses.append([float(bonus_a), float(bonus_b)])

        # Exploration statistics
        exploration_stats.append({
            'episode': episode,
            'agent_a_states': int(50 + episode * 0.5 + np.random.randint(0, 20)),
            'agent_b_states': int(45 + episode * 0.5 + np.random.randint(0, 20)),
            'unique_joint_states': int(100 + episode * 0.8 + np.random.randint(0, 30))
        })

    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'success_rates': success_rates,
        'sim_bonuses': sim_bonuses,
        'exploration_stats': exploration_stats,
        'total_episodes': total_episodes
    }


def generate_deep_experiment_data(run_name: str, intrinsic_coef: float) -> Dict[str, Any]:
    """Generate realistic deep learning experiment data."""
    total_episodes = 1000

    # Generate episode rewards with deep learning characteristics
    episode_rewards = []
    episode_lengths = []
    success_rates = []

    # Deep learning typically has more stable but slower learning
    baseline_reward = -15
    max_reward = 18

    for episode in range(total_episodes):
        # Slower learning curve for deep RL
        progress = episode / total_episodes
        learning_factor = 1 / (1 + np.exp(-4 * (progress - 0.5)))  # Slower sigmoid

        # More stable learning with less noise
        noise = np.random.normal(0, 1.5)

        # Calculate average reward
        avg_reward = baseline_reward + (max_reward - baseline_reward) * learning_factor + noise
        episode_rewards.append(float(avg_reward))

        # Episode length
        max_length = 100
        length = int(max_length * (1 - learning_factor * 0.8) + np.random.randint(10, 20))
        episode_lengths.append(length)

        # Success rate
        success_prob = learning_factor * 0.95 + 0.02
        success = np.random.random() < success_prob
        success_rates.append(int(success))

    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'success_rates': success_rates,
        'total_episodes': total_episodes,
        'intrinsic_coef': intrinsic_coef,
        'run_name': run_name
    }


def main():
    """Main function to create all sample data."""
    print("Creating sample experimental data for Social Curiosity project...")

    # Create sample data for both implementations
    create_tabular_sample_data()
    create_deep_sample_data()

    print("\nSample data creation completed!")
    print("\nGenerated directories:")
    print("- results/tabular/baseline/")
    print("- results/tabular/sim/")
    print("- results/deep/baseline/")
    print("- results/deep/sim/")
    print("\nYou can now run the plotting scripts to generate visualizations.")


if __name__ == "__main__":
    main()
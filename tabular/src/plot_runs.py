#!/usr/bin/env python3
"""
Plotting functionality for comparing results between baseline and SIM experiments.

This script generates comparison plots showing performance differences between
baseline (intrinsic_coef=0.0) and SIM (intrinsic_coef=0.2) experiments.

Usage:
    python plot_runs.py --baseline results/tabular/baseline --sim results/tabular/sim --output plots/compare.png
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import os
import sys
from dataclasses import dataclass
import seaborn as sns

# Set matplotlib style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")


@dataclass
class ExperimentData:
    """Container for experiment data and metadata."""
    run_name: str
    intrinsic_coef: float
    episode_rewards: List[Tuple[float, float]]
    episode_lengths: List[int]
    success_rates: List[bool]
    sim_bonuses: List[Tuple[float, float]]
    exploration_stats: List[Dict[str, Any]]
    total_episodes: int
    config: Dict[str, Any]


def load_experiment_data(results_dir: Path) -> ExperimentData:
    """
    Load experiment data from results directory.
    
    Args:
        results_dir: Path to experiment results directory
        
    Returns:
        ExperimentData object containing all experiment data
    """
    results_dir = Path(results_dir)
    
    # Load results
    with open(results_dir / "results.json", 'r') as f:
        results = json.load(f)
    
    # Load config
    with open(results_dir / "config.json", 'r') as f:
        config = json.load(f)
    
    return ExperimentData(
        run_name=config['run_name'],
        intrinsic_coef=config['intrinsic_coef'],
        episode_rewards=results['episode_rewards'],
        episode_lengths=results['episode_lengths'],
        success_rates=results['success_rates'],
        sim_bonuses=results['sim_bonuses'],
        exploration_stats=results['exploration_stats'],
        total_episodes=results['total_episodes'],
        config=config
    )


def load_multiple_experiments(directories: List[Path]) -> List[ExperimentData]:
    """
    Load multiple experiment directories.
    
    Args:
        directories: List of paths to experiment directories
        
    Returns:
        List of ExperimentData objects
    """
    experiments = []
    for dir_path in directories:
        try:
            exp_data = load_experiment_data(dir_path)
            experiments.append(exp_data)
            print(f"Loaded experiment: {exp_data.run_name} (intrinsic_coef={exp_data.intrinsic_coef})")
        except Exception as e:
            print(f"Error loading {dir_path}: {e}")
    
    return experiments


def plot_learning_curves(experiments: List[ExperimentData], ax: plt.Axes) -> None:
    """
    Plot learning curves (episode rewards over time).
    
    Args:
        experiments: List of ExperimentData objects
        ax: Matplotlib axes to plot on
    """
    ax.set_title('Learning Curves: Episode Rewards')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    
    for exp in experiments:
        episodes = range(1, exp.total_episodes + 1)
        rewards_a = [r[0] for r in exp.episode_rewards]
        rewards_b = [r[1] for r in exp.episode_rewards]
        
        # Plot moving average for smoother curves
        window_size = max(1, exp.total_episodes // 100)
        rewards_a_smooth = np.convolve(rewards_a, np.ones(window_size)/window_size, mode='valid')
        rewards_b_smooth = np.convolve(rewards_b, np.ones(window_size)/window_size, mode='valid')
        
        label = f"{exp.run_name} (coef={exp.intrinsic_coef})"
        ax.plot(episodes[window_size-1:], rewards_a_smooth, label=f"{label} - Agent A", alpha=0.8)
        ax.plot(episodes[window_size-1:], rewards_b_smooth, label=f"{label} - Agent B", alpha=0.8, linestyle='--')
    
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_success_rates(experiments: List[ExperimentData], ax: plt.Axes) -> None:
    """
    Plot success rates over episodes.
    
    Args:
        experiments: List of ExperimentData objects
        ax: Matplotlib axes to plot on
    """
    ax.set_title('Success Rates Over Episodes')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate')
    ax.set_ylim(0, 1.1)
    
    for exp in experiments:
        episodes = range(1, exp.total_episodes + 1)
        
        # Calculate cumulative success rate
        cumulative_success = np.cumsum(exp.success_rates) / np.arange(1, exp.total_episodes + 1)
        
        label = f"{exp.run_name} (coef={exp.intrinsic_coef})"
        ax.plot(episodes, cumulative_success, label=label, linewidth=2)
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Perfect Success')


def plot_exploration_comparison(experiments: List[ExperimentData], ax: plt.Axes) -> None:
    """
    Plot exploration statistics comparison.
    
    Args:
        experiments: List of ExperimentData objects
        ax: Matplotlib axes to plot on
    """
    ax.set_title('Exploration Comparison: Unique States Visited')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Unique States')
    
    for exp in experiments:
        episodes = [stats['episode'] for stats in exp.exploration_stats]
        states_a = [stats['agent_a_states'] for stats in exp.exploration_stats]
        states_b = [stats['agent_b_states'] for stats in exp.exploration_stats]
        
        label = f"{exp.run_name} (coef={exp.intrinsic_coef})"
        ax.plot(episodes, states_a, label=f"{label} - Agent A", alpha=0.8)
        ax.plot(episodes, states_b, label=f"{label} - Agent B", alpha=0.8, linestyle='--')
    
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_performance_metrics(experiments: List[ExperimentData], ax: plt.Axes) -> None:
    """
    Plot performance metrics comparison.
    
    Args:
        experiments: List of ExperimentData objects
        ax: Matplotlib axes to plot on
    """
    metrics_data = []
    labels = []
    
    for exp in experiments:
        # Calculate performance metrics
        final_100_rewards_a = [r[0] for r in exp.episode_rewards[-100:]]
        final_100_rewards_b = [r[1] for r in exp.episode_rewards[-100:]]
        final_success_rate = np.mean(exp.success_rates[-100:]) * 100
        
        metrics_data.append({
            'Avg Reward A': np.mean(final_100_rewards_a),
            'Avg Reward B': np.mean(final_100_rewards_b),
            'Success Rate': final_success_rate,
            'Total States': exp.exploration_stats[-1]['agent_a_states'] + exp.exploration_stats[-1]['agent_b_states']
        })
        labels.append(f"{exp.run_name}\n(coef={exp.intrinsic_coef})")
    
    # Create bar plot
    x = np.arange(len(metrics_data))
    width = 0.2
    
    metrics = ['Avg Reward A', 'Avg Reward B', 'Success Rate', 'Total States']
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
    
    for i, metric in enumerate(metrics):
        values = [data[metric] for data in metrics_data]
        ax.bar(x + i * width, values, width, label=metric, color=colors[i], alpha=0.8)
    
    ax.set_title('Performance Metrics Comparison (Final 100 Episodes)')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')


def calculate_statistics(experiments: List[ExperimentData]) -> Dict[str, Any]:
    """
    Calculate statistical comparisons between experiments.
    
    Args:
        experiments: List of ExperimentData objects
        
    Returns:
        Dictionary with statistical comparisons
    """
    stats = {}
    
    for exp in experiments:
        # Final performance metrics
        final_100_rewards = [r[0] + r[1] for r in exp.episode_rewards[-100:]]  # Total reward per episode
        final_success_rate = np.mean(exp.success_rates[-100:]) * 100
        
        stats[exp.run_name] = {
            'final_avg_reward': np.mean(final_100_rewards),
            'final_std_reward': np.std(final_100_rewards),
            'final_success_rate': final_success_rate,
            'total_unique_states': exp.exploration_stats[-1]['agent_a_states'] + exp.exploration_stats[-1]['agent_b_states'],
            'total_episodes': exp.total_episodes
        }
    
    return stats


def create_comparison_plots(experiments: List[ExperimentData], output_path: str) -> None:
    """
    Create comprehensive comparison plots.
    
    Args:
        experiments: List of ExperimentData objects
        output_path: Path to save the output plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Social Curiosity: Baseline vs SIM Experiment Comparison', fontsize=16, fontweight='bold')
    
    # Plot individual components
    plot_learning_curves(experiments, axes[0, 0])
    plot_success_rates(experiments, axes[0, 1])
    plot_exploration_comparison(experiments, axes[1, 0])
    plot_performance_metrics(experiments, axes[1, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plot saved to: {output_path}")


def main():
    """Main function to parse arguments and generate plots."""
    parser = argparse.ArgumentParser(description="Compare results between baseline and SIM experiments")
    
    # Input directories
    parser.add_argument("--baseline", type=str, required=True,
                       help="Path to baseline experiment results directory")
    parser.add_argument("--sim", type=str, required=True,
                       help="Path to SIM experiment results directory")
    parser.add_argument("--output", type=str, default="plots/compare.png",
                       help="Output path for comparison plot")
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"],
                       help="Output format for plots")
    
    args = parser.parse_args()
    
    # Load experiments
    experiments = load_multiple_experiments([args.baseline, args.sim])
    
    if len(experiments) < 2:
        print("Error: Need at least 2 experiments for comparison")
        return
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate comparison plots
    create_comparison_plots(experiments, args.output)
    
    # Print statistical comparison
    stats = calculate_statistics(experiments)
    print("\nStatistical Comparison:")
    print("-" * 50)
    for exp_name, exp_stats in stats.items():
        print(f"{exp_name}:")
        print(f"  Final Avg Reward: {exp_stats['final_avg_reward']:.2f} ± {exp_stats['final_std_reward']:.2f}")
        print(f"  Final Success Rate: {exp_stats['final_success_rate']:.1f}%")
        print(f"  Total Unique States: {exp_stats['total_unique_states']}")
        print()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Comprehensive plotting system for Social Curiosity project.

This script generates comprehensive visualizations for both tabular and deep learning experiments,
including learning curves, performance comparisons, exploration patterns, and state coverage analysis.

Usage:
    python generate_comprehensive_plots.py
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from dataclasses import dataclass
import os

# Set matplotlib style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")


@dataclass
class ExperimentData:
    """Container for experiment data and metadata."""
    run_name: str
    implementation: str  # 'tabular' or 'deep'
    intrinsic_coef: float
    episode_rewards: List[float]
    episode_lengths: List[int]
    success_rates: List[int]
    sim_bonuses: Optional[List[Tuple[float, float]]] = None
    exploration_stats: Optional[List[Dict[str, Any]]] = None
    total_episodes: int = 0
    config: Dict[str, Any] = None


def load_tabular_experiment_data(results_dir: Path) -> ExperimentData:
    """Load tabular experiment data from results directory."""
    results_dir = Path(results_dir)

    # Load results
    with open(results_dir / "results.json", 'r') as f:
        results = json.load(f)

    # Load config
    with open(results_dir / "config.json", 'r') as f:
        config = json.load(f)

    return ExperimentData(
        run_name=config['run_name'],
        implementation='tabular',
        intrinsic_coef=config['intrinsic_coef'],
        episode_rewards=results['episode_rewards'],
        episode_lengths=results['episode_lengths'],
        success_rates=results['success_rates'],
        sim_bonuses=results.get('sim_bonuses', []),
        exploration_stats=results.get('exploration_stats', []),
        total_episodes=results['total_episodes'],
        config=config
    )


def load_deep_experiment_data(results_dir: Path) -> ExperimentData:
    """Load deep learning experiment data from results directory."""
    results_dir = Path(results_dir)

    # Load results
    with open(results_dir / "results.json", 'r') as f:
        results = json.load(f)

    return ExperimentData(
        run_name=results.get('run_name', 'deep_experiment'),
        implementation='deep',
        intrinsic_coef=results.get('intrinsic_coef', 0.0),
        episode_rewards=results['episode_rewards'],
        episode_lengths=results['episode_lengths'],
        success_rates=results['success_rates'],
        total_episodes=results['total_episodes']
    )


def load_all_experiments() -> List[ExperimentData]:
    """Load all available experimental data."""
    experiments = []

    # Load tabular experiments
    tabular_dir = Path("results/tabular")
    if tabular_dir.exists():
        for exp_dir in tabular_dir.iterdir():
            if exp_dir.is_dir():
                try:
                    exp_data = load_tabular_experiment_data(exp_dir)
                    experiments.append(exp_data)
                    print(f"Loaded tabular experiment: {exp_data.run_name} (coef={exp_data.intrinsic_coef})")
                except Exception as e:
                    print(f"Error loading {exp_dir}: {e}")

    # Load deep learning experiments
    deep_dir = Path("results/deep")
    if deep_dir.exists():
        for exp_dir in deep_dir.iterdir():
            if exp_dir.is_dir():
                try:
                    exp_data = load_deep_experiment_data(exp_dir)
                    experiments.append(exp_data)
                    print(f"Loaded deep learning experiment: {exp_data.run_name} (coef={exp_data.intrinsic_coef})")
                except Exception as e:
                    print(f"Error loading {exp_dir}: {e}")

    return experiments


def create_learning_curves_plot(experiments: List[ExperimentData], output_dir: Path):
    """Create comprehensive learning curves plot."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Social Curiosity: Learning Curves Comparison', fontsize=16, fontweight='bold')

    # Separate experiments by implementation and intrinsic coefficient
    tabular_baseline = [e for e in experiments if e.implementation == 'tabular' and e.intrinsic_coef == 0.0]
    tabular_sim = [e for e in experiments if e.implementation == 'tabular' and e.intrinsic_coef > 0.0]
    deep_baseline = [e for e in experiments if e.implementation == 'deep' and e.intrinsic_coef == 0.0]
    deep_sim = [e for e in experiments if e.implementation == 'deep' and e.intrinsic_coef > 0.0]

    # Plot 1: Tabular Learning Curves
    ax = axes[0, 0]
    ax.set_title('Tabular Q-Learning: Episode Rewards')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')

    for exp in tabular_baseline + tabular_sim:
        if exp.episode_rewards:
            episodes = range(1, len(exp.episode_rewards) + 1)
            if exp.implementation == 'tabular':
                # For tabular, episode_rewards is list of [reward_a, reward_b] tuples
                rewards = [np.mean(r) if isinstance(r, list) else r for r in exp.episode_rewards]
            else:
                rewards = exp.episode_rewards

            # Calculate moving average
            window_size = max(1, len(rewards) // 100)
            rewards_smooth = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')

            label = f"{exp.run_name} (coef={exp.intrinsic_coef})"
            ax.plot(episodes[window_size-1:], rewards_smooth, label=label, linewidth=2)

    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Deep Learning Curves
    ax = axes[0, 1]
    ax.set_title('Deep RL (PPO): Episode Rewards')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')

    for exp in deep_baseline + deep_sim:
        if exp.episode_rewards:
            episodes = range(1, len(exp.episode_rewards) + 1)
            rewards = exp.episode_rewards

            # Calculate moving average
            window_size = max(1, len(rewards) // 100)
            rewards_smooth = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')

            label = f"{exp.run_name} (coef={exp.intrinsic_coef})"
            ax.plot(episodes[window_size-1:], rewards_smooth, label=label, linewidth=2)

    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Success Rates Comparison
    ax = axes[1, 0]
    ax.set_title('Success Rates Over Time')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate')
    ax.set_ylim(0, 1.1)

    for exp in experiments:
        if exp.success_rates:
            episodes = range(1, len(exp.success_rates) + 1)
            success_rates = np.array(exp.success_rates, dtype=float)

            # Calculate cumulative success rate
            cumulative_success = np.cumsum(success_rates) / np.arange(1, len(success_rates) + 1)

            label = f"{exp.implementation.upper()}: {exp.run_name} (coef={exp.intrinsic_coef})"
            ax.plot(episodes, cumulative_success, label=label, linewidth=2)

    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Perfect Success')

    # Plot 4: Performance Comparison
    ax = axes[1, 1]
    ax.set_title('Final Performance Comparison (Last 100 Episodes)')
    ax.set_ylabel('Performance Metric')

    exp_names = []
    final_rewards = []
    final_success_rates = []

    for exp in experiments:
        if len(exp.episode_rewards) >= 100:
            # Calculate final metrics
            if exp.implementation == 'tabular':
                final_rewards_data = [np.mean(r) if isinstance(r, list) else r for r in exp.episode_rewards[-100:]]
            else:
                final_rewards_data = exp.episode_rewards[-100:]

            final_reward = np.mean(final_rewards_data)
            final_success = np.mean(exp.success_rates[-100:]) * 100

            exp_names.append(f"{exp.implementation.upper()}\n{exp.run_name}\n(coef={exp.intrinsic_coef})")
            final_rewards.append(final_reward)
            final_success_rates.append(final_success)

    x = np.arange(len(exp_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, final_rewards, width, label='Avg Reward', alpha=0.8)
    bars2 = ax.bar(x + width/2, final_success_rates, width, label='Success Rate (%)', alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(exp_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'learning_curves_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved learning curves plot: {output_dir / 'learning_curves_comparison.png'}")


def create_exploration_analysis_plot(experiments: List[ExperimentData], output_dir: Path):
    """Create exploration patterns and state coverage analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Social Curiosity: Exploration Analysis', fontsize=16, fontweight='bold')

    # Filter tabular experiments (only they have exploration stats)
    tabular_experiments = [e for e in experiments if e.implementation == 'tabular' and e.exploration_stats]

    if tabular_experiments:
        # Plot 1: State Exploration Over Time
        ax = axes[0, 0]
        ax.set_title('State Exploration: Unique States Visited')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Unique States')

        for exp in tabular_experiments:
            if exp.exploration_stats:
                episodes = [stats['episode'] for stats in exp.exploration_stats]
                states_a = [stats['agent_a_states'] for stats in exp.exploration_stats]
                states_b = [stats['agent_b_states'] for stats in exp.exploration_stats]

                label = f"{exp.run_name} (coef={exp.intrinsic_coef})"
                ax.plot(episodes, states_a, label=f"{label} - Agent A", alpha=0.8)
                ax.plot(episodes, states_b, label=f"{label} - Agent B", alpha=0.8, linestyle='--')

        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Joint State Coverage
        ax = axes[0, 1]
        ax.set_title('Joint State Coverage')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Joint Unique States')

        for exp in tabular_experiments:
            if exp.exploration_stats:
                episodes = [stats['episode'] for stats in exp.exploration_stats]
                joint_states = [stats.get('unique_joint_states', stats['agent_a_states'] + stats['agent_b_states'])
                              for stats in exp.exploration_stats]

                label = f"{exp.run_name} (coef={exp.intrinsic_coef})"
                ax.plot(episodes, joint_states, label=label, linewidth=2)

        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Exploration Efficiency
        ax = axes[1, 0]
        ax.set_title('Exploration Efficiency')
        ax.set_xlabel('Episode')
        ax.set_ylabel('New States per Episode')

        for exp in tabular_experiments:
            if len(exp.exploration_stats) > 1:
                episodes = [stats['episode'] for stats in exp.exploration_stats[1:]]
                states_a = [stats['agent_a_states'] for stats in exp.exploration_stats[1:]]
                prev_states_a = [stats['agent_a_states'] for stats in exp.exploration_stats[:-1]]
                states_b = [stats['agent_b_states'] for stats in exp.exploration_stats[1:]]
                prev_states_b = [stats['agent_b_states'] for stats in exp.exploration_stats[:-1]]

                new_states_a = [max(0, curr - prev) for curr, prev in zip(states_a, prev_states_a)]
                new_states_b = [max(0, curr - prev) for curr, prev in zip(states_b, prev_states_b)]
                total_new_states = [a + b for a, b in zip(new_states_a, new_states_b)]

                label = f"{exp.run_name} (coef={exp.intrinsic_coef})"
                ax.plot(episodes, total_new_states, label=label, linewidth=2)

        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Social Curiosity Impact
        ax = axes[1, 1]
        ax.set_title('Social Curiosity Reward Distribution')
        ax.set_xlabel('Episode')
        ax.set_ylabel('SIM Bonus Magnitude')

        for exp in tabular_experiments:
            if exp.sim_bonuses:
                episodes = range(1, len(exp.sim_bonuses) + 1)
                bonus_magnitudes = [abs(a) + abs(b) for a, b in exp.sim_bonuses]

                # Calculate moving average
                window_size = max(1, len(bonus_magnitudes) // 50)
                bonuses_smooth = np.convolve(bonus_magnitudes, np.ones(window_size)/window_size, mode='valid')

                label = f"{exp.run_name} (coef={exp.intrinsic_coef})"
                ax.plot(episodes[window_size-1:], bonuses_smooth, label=label, linewidth=2)

        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'exploration_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved exploration analysis plot: {output_dir / 'exploration_analysis.png'}")


def create_performance_comparison_plot(experiments: List[ExperimentData], output_dir: Path):
    """Create performance comparison between implementations."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Implementation Performance Comparison', fontsize=16, fontweight='bold')

    # Separate by implementation
    tabular_experiments = [e for e in experiments if e.implementation == 'tabular']
    deep_experiments = [e for e in experiments if e.implementation == 'deep']

    # Plot 1: Learning Speed Comparison
    ax = axes[0, 0]
    ax.set_title('Learning Speed: Episodes to Reach Threshold')
    ax.set_ylabel('Episodes to 80% Success Rate')

    threshold = 0.8
    implementations = []
    episodes_to_threshold = []

    for exp in experiments:
        if exp.success_rates:
            success_rates = np.array(exp.success_rates, dtype=float)
            cumulative_success = np.cumsum(success_rates) / np.arange(1, len(success_rates) + 1)

            # Find first episode where cumulative success >= threshold
            threshold_episode = None
            for i, cum_success in enumerate(cumulative_success):
                if cum_success >= threshold:
                    threshold_episode = i + 1
                    break

            if threshold_episode:
                implementations.append(f"{exp.implementation.upper()}\n{exp.run_name}\n(coef={exp.intrinsic_coef})")
                episodes_to_threshold.append(threshold_episode)

    if episodes_to_threshold:
        x = np.arange(len(implementations))
        bars = ax.bar(x, episodes_to_threshold, alpha=0.8, color=['skyblue' if 'TABULAR' in impl else 'lightcoral' for impl in implementations])
        ax.set_xticks(x)
        ax.set_xticklabels(implementations, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')

    # Plot 2: Final Performance by Implementation
    ax = axes[0, 1]
    ax.set_title('Final Performance by Implementation')
    ax.set_ylabel('Performance Metric')

    if tabular_experiments and deep_experiments:
        # Calculate final metrics for each implementation
        tabular_final_reward = np.mean([np.mean([np.mean(r) if isinstance(r, list) else r for r in e.episode_rewards[-100:]])
                                      for e in tabular_experiments if len(e.episode_rewards) >= 100])
        deep_final_reward = np.mean([np.mean(e.episode_rewards[-100:]) for e in deep_experiments if len(e.episode_rewards) >= 100])

        tabular_final_success = np.mean([np.mean(e.success_rates[-100:]) for e in tabular_experiments if len(e.success_rates) >= 100])
        deep_final_success = np.mean([np.mean(e.success_rates[-100:]) for e in deep_experiments if len(e.success_rates) >= 100])

        metrics = ['Average Reward', 'Success Rate']
        tabular_values = [tabular_final_reward, tabular_final_success]
        deep_values = [deep_final_reward, deep_final_success]

        x = np.arange(len(metrics))
        width = 0.35

        ax.bar(x - width/2, tabular_values, width, label='Tabular Q-Learning', alpha=0.8)
        ax.bar(x + width/2, deep_values, width, label='Deep RL (PPO)', alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Stability Analysis
    ax = axes[1, 0]
    ax.set_title('Learning Stability: Reward Variance')
    ax.set_ylabel('Reward Standard Deviation (Last 100 Episodes)')

    exp_names = []
    reward_stds = []

    for exp in experiments:
        if len(exp.episode_rewards) >= 100:
            if exp.implementation == 'tabular':
                final_rewards = [np.mean(r) if isinstance(r, list) else r for r in exp.episode_rewards[-100:]]
            else:
                final_rewards = exp.episode_rewards[-100:]

            reward_std = np.std(final_rewards)
            exp_names.append(f"{exp.implementation.upper()}\n{exp.run_name}\n(coef={exp.intrinsic_coef})")
            reward_stds.append(reward_std)

    if reward_stds:
        x = np.arange(len(exp_names))
        bars = ax.bar(x, reward_stds, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(exp_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Social Curiosity Effectiveness
    ax = axes[1, 1]
    ax.set_title('Social Curiosity Effectiveness')
    ax.set_ylabel('Performance Improvement (%)')

    if len(tabular_experiments) >= 2:
        # Compare baseline vs SIM for tabular
        baseline_exp = next((e for e in tabular_experiments if e.intrinsic_coef == 0.0), None)
        sim_exp = next((e for e in tabular_experiments if e.intrinsic_coef > 0.0), None)

        if baseline_exp and sim_exp and len(baseline_exp.episode_rewards) >= 100 and len(sim_exp.episode_rewards) >= 100:
            baseline_reward = np.mean([np.mean(r) if isinstance(r, list) else r for r in baseline_exp.episode_rewards[-100:]])
            sim_reward = np.mean([np.mean(r) if isinstance(r, list) else r for r in sim_exp.episode_rewards[-100:]])

            baseline_success = np.mean(baseline_exp.success_rates[-100:])
            sim_success = np.mean(sim_exp.success_rates[-100:])

            reward_improvement = ((sim_reward - baseline_reward) / abs(baseline_reward)) * 100
            success_improvement = ((sim_success - baseline_success) / baseline_success) * 100

            improvements = [reward_improvement, success_improvement]
            labels = ['Reward Improvement', 'Success Rate Improvement']

            x = np.arange(len(labels))
            bars = ax.bar(x, improvements, alpha=0.8, color=['green' if imp > 0 else 'red' for imp in improvements])

            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved performance comparison plot: {output_dir / 'performance_comparison.png'}")


def create_summary_report(experiments: List[ExperimentData], output_dir: Path):
    """Create a comprehensive summary report."""
    report = []
    report.append("# Social Curiosity Project: Comprehensive Analysis Report\n")
    report.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Summary statistics
    report.append("## Experiment Summary\n")
    report.append(f"- Total experiments analyzed: {len(experiments)}")
    report.append(f"- Tabular experiments: {len([e for e in experiments if e.implementation == 'tabular'])}")
    report.append(f"- Deep learning experiments: {len([e for e in experiments if e.implementation == 'deep'])}")
    report.append(f"- Baseline experiments (coef=0.0): {len([e for e in experiments if e.intrinsic_coef == 0.0])}")
    report.append(f"- SIM experiments (coef>0.0): {len([e for e in experiments if e.intrinsic_coef > 0.0])}")
    report.append("")

    # Performance comparison
    report.append("## Performance Comparison\n")
    report.append("| Experiment | Implementation | Intrinsic Coef | Final Avg Reward | Final Success Rate | Total Episodes |")
    report.append("|------------|----------------|----------------|------------------|-------------------|----------------|")

    for exp in experiments:
        if len(exp.episode_rewards) >= 100:
            if exp.implementation == 'tabular':
                final_reward = np.mean([np.mean(r) if isinstance(r, list) else r for r in exp.episode_rewards[-100:]])
            else:
                final_reward = np.mean(exp.episode_rewards[-100:])

            final_success = np.mean(exp.success_rates[-100:]) * 100

            report.append(f"| {exp.run_name} | {exp.implementation.upper()} | {exp.intrinsic_coef} | {final_reward:.2f} | {final_success:.1f}% | {exp.total_episodes} |")

    report.append("")

    # Key findings
    report.append("## Key Findings\n")

    # Compare baseline vs SIM
    baseline_experiments = [e for e in experiments if e.intrinsic_coef == 0.0]
    sim_experiments = [e for e in experiments if e.intrinsic_coef > 0.0]

    if baseline_experiments and sim_experiments:
        report.append("### Social Curiosity Impact\n")

        for baseline, sim in zip(baseline_experiments, sim_experiments):
            if len(baseline.episode_rewards) >= 100 and len(sim.episode_rewards) >= 100:
                if baseline.implementation == 'tabular':
                    baseline_reward = np.mean([np.mean(r) if isinstance(r, list) else r for r in baseline.episode_rewards[-100:]])
                    sim_reward = np.mean([np.mean(r) if isinstance(r, list) else r for r in sim.episode_rewards[-100:]])
                else:
                    baseline_reward = np.mean(baseline.episode_rewards[-100:])
                    sim_reward = np.mean(sim.episode_rewards[-100:])

                reward_improvement = ((sim_reward - baseline_reward) / abs(baseline_reward)) * 100
                report.append(f"- **{baseline.implementation.upper()}**: Social curiosity improved final reward by {reward_improvement:+.1f}%")

    # Compare implementations
    tabular_experiments = [e for e in experiments if e.implementation == 'tabular']
    deep_experiments = [e for e in experiments if e.implementation == 'deep']

    if tabular_experiments and deep_experiments:
        report.append("\n### Implementation Comparison\n")

        tabular_final_reward = np.mean([np.mean([np.mean(r) if isinstance(r, list) else r for r in e.episode_rewards[-100:]])
                                      for e in tabular_experiments if len(e.episode_rewards) >= 100])
        deep_final_reward = np.mean([np.mean(e.episode_rewards[-100:]) for e in deep_experiments if len(e.episode_rewards) >= 100])

        if tabular_final_reward > deep_final_reward:
            report.append(f"- Tabular Q-Learning achieved higher final reward ({tabular_final_reward:.2f} vs {deep_final_reward:.2f})")
        else:
            report.append(f"- Deep RL (PPO) achieved higher final reward ({deep_final_reward:.2f} vs {tabular_final_reward:.2f})")

    # Save report
    report_path = output_dir / 'analysis_report.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"Saved analysis report: {report_path}")


def main():
    """Main function to generate all plots and analysis."""
    print("Generating comprehensive plots for Social Curiosity project...")

    # Create output directory
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)

    # Load all experimental data
    experiments = load_all_experiments()

    if not experiments:
        print("No experimental data found. Please run experiments first or create sample data.")
        return

    print(f"Loaded {len(experiments)} experiments")

    # Generate all plots
    create_learning_curves_plot(experiments, output_dir)
    create_exploration_analysis_plot(experiments, output_dir)
    create_performance_comparison_plot(experiments, output_dir)
    create_summary_report(experiments, output_dir)

    print("\nPlot generation completed!")
    print(f"All plots saved in: {output_dir}")
    print("\nGenerated files:")
    print("- learning_curves_comparison.png")
    print("- exploration_analysis.png")
    print("- performance_comparison.png")
    print("- analysis_report.md")


if __name__ == "__main__":
    main()
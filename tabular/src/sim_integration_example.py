#!/usr/bin/env python3
"""
Example integration of SocialIntrinsicMotivation with the GridWorld environment.
This demonstrates how the SIM signal would be used in a training loop.
"""

import numpy as np
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from tabular.src.environment import GridWorldEnv
from tabular.src.social_motivation import SocialIntrinsicMotivation


def sim_integration_example():
    """Example of how to integrate SIM with the GridWorld environment."""
    print("Social Intrinsic Motivation Integration Example")
    print("=" * 50)
    
    # Initialize environment and SIM module
    env = GridWorldEnv()
    sim = SocialIntrinsicMotivation(intrinsic_coef=0.2)
    
    # Reset environment and SIM
    state = env.reset()
    sim.reset()
    
    print("Initial state:", state)
    print("Initial visited cells:", sim.get_visited_stats())
    
    # Run a few steps to demonstrate SIM integration
    for step in range(10):
        print(f"\n--- Step {step + 1} ---")
        
        # Store previous state for SIM calculation
        prev_state = state.copy()
        
        # Choose random actions (in a real training loop, agents would choose actions)
        action_a = np.random.randint(0, 5)  # 5 actions: stay, up, down, left, right
        action_b = np.random.randint(0, 5)
        
        print(f"Actions: A={env.ACTION_NAMES[action_a]}, B={env.ACTION_NAMES[action_b]}")
        
        # Take step in environment
        next_state, (reward_a, reward_b), done, info = env.step(action_a, action_b)
        
        # Update SIM visited cells
        sim.update(prev_state, next_state, 'A')
        sim.update(prev_state, next_state, 'B')
        
        # Calculate SIM bonuses
        sim_bonus_a, sim_bonus_b = sim.calculate_bonus(prev_state, next_state, 'A', 'B')
        
        # Combine extrinsic and intrinsic rewards
        total_reward_a = reward_a + sim_bonus_a
        total_reward_b = reward_b + sim_bonus_b
        
        print(f"Extrinsic rewards: A={reward_a:.2f}, B={reward_b:.2f}")
        print(f"SIM bonuses: A={sim_bonus_a:.2f}, B={sim_bonus_b:.2f}")
        print(f"Total rewards: A={total_reward_a:.2f}, B={total_reward_b:.2f}")
        print(f"Agent positions: A={info['agent_a_pos']}, B={info['agent_b_pos']}")
        print(f"Door open: {info['door_open']}, Coins taken: {info['coins_taken']}")
        print(f"Visited cells: {sim.get_visited_stats()}")
        
        # Update state for next iteration
        state = next_state
        
        if done:
            print("\nEpisode completed!")
            break
    
    print(f"\nFinal visited cells:")
    print(f"Agent A: {sorted(sim.visited_cells['A'])}")
    print(f"Agent B: {sorted(sim.visited_cells['B'])}")


def training_loop_structure():
    """
    Example structure of how SIM would be integrated into a training loop.
    This is a template that would be used in the actual train.py script.
    """
    print("\n" + "="*60)
    print("TRAINING LOOP INTEGRATION STRUCTURE")
    print("="*60)
    
    # Pseudocode for training loop integration:
    print("""
# In training loop initialization:
env = GridWorldEnv()
sim = SocialIntrinsicMotivation(intrinsic_coef=0.2)  # Configurable coefficient
agent_a = TabularQLearningAgent()
agent_b = TabularQLearningAgent()

for episode in range(num_episodes):
    state = env.reset()
    sim.reset()  # Reset visited cells for new episode
    episode_rewards = [0, 0]
    
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
        
        # Calculate SIM bonuses
        sim_bonus_a, sim_bonus_b = sim.calculate_bonus(prev_state, next_state, 'A', 'B')
        
        # Combine rewards
        total_reward_a = reward_a + sim_bonus_a
        total_reward_b = reward_b + sim_bonus_b
        
        # Agent learning updates
        agent_a.update(state, action_a, total_reward_a, next_state, done)
        agent_b.update(state, action_b, total_reward_b, next_state, done)
        
        # Update state and track rewards
        state = next_state
        episode_rewards[0] += total_reward_a
        episode_rewards[1] += total_reward_b
    
    # Episode statistics and logging...
    agent_a.decay_epsilon()
    agent_b.decay_epsilon()
    """)


if __name__ == "__main__":
    sim_integration_example()
    training_loop_structure()
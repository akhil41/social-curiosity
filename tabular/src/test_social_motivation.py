#!/usr/bin/env python3
"""
Test script for SocialIntrinsicMotivation class.
"""

import numpy as np
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from tabular.src.social_motivation import SocialIntrinsicMotivation


def test_social_motivation():
    """Test the SocialIntrinsicMotivation implementation."""
    print("Testing SocialIntrinsicMotivation...")
    
    # Initialize the SIM module
    sim = SocialIntrinsicMotivation(intrinsic_coef=0.2)
    
    # Test state representation: (ax, ay, bx, by, door_open, c1_taken, c2_taken)
    # Initial state: A at (0,0), B at (4,0), door closed, coins not taken
    initial_state = np.array([0, 0, 4, 0, 0, 0, 0], dtype=np.float32)
    
    # Reset to initialize visited cells with starting positions
    sim.reset()
    
    print("Initial visited cells:")
    print(f"Agent A: {sim.visited_cells['A']}")
    print(f"Agent B: {sim.visited_cells['B']}")
    
    # Test 1: No movement - should give no bonus
    next_state1 = np.array([0, 0, 4, 0, 0, 0, 0], dtype=np.float32)  # No movement
    bonus_a, bonus_b = sim.calculate_bonus(initial_state, next_state1, 'A', 'B')
    print(f"\nTest 1 - No movement:")
    print(f"Bonus A: {bonus_a}, Bonus B: {bonus_b}")
    
    # Update visited cells
    sim.update(initial_state, next_state1, 'A')
    sim.update(initial_state, next_state1, 'B')
    
    # Test 2: Agent B moves to new cell, but Agent A is too far
    next_state2 = np.array([0, 0, 4, 1, 0, 0, 0], dtype=np.float32)  # B moves down to (4,1)
    bonus_a, bonus_b = sim.calculate_bonus(next_state1, next_state2, 'A', 'B')
    print(f"\nTest 2 - B moves to (4,1), A at (0,0) (distance 5):")
    print(f"Bonus A: {bonus_a}, Bonus B: {bonus_b}")
    
    # Update visited cells
    sim.update(next_state1, next_state2, 'A')
    sim.update(next_state1, next_state2, 'B')
    
    # Test 3: Agent A moves adjacent to B, then B moves to new cell
    # First, move A closer to B
    state_a_near = np.array([3, 0, 4, 1, 0, 0, 0], dtype=np.float32)  # A at (3,0), B at (4,1)
    sim.update(next_state2, state_a_near, 'A')
    sim.update(next_state2, state_a_near, 'B')
    
    # Now B moves to new cell while A is adjacent
    state_b_new = np.array([3, 0, 4, 2, 0, 0, 0], dtype=np.float32)  # A at (3,0), B at (4,2)
    bonus_a, bonus_b = sim.calculate_bonus(state_a_near, state_b_new, 'A', 'B')
    print(f"\nTest 3 - A at (3,0), B moves from (4,1) to (4,2) (distance 1):")
    print(f"Bonus A: {bonus_a}, Bonus B: {bonus_b}")
    
    # Update visited cells
    sim.update(state_a_near, state_b_new, 'A')
    sim.update(state_a_near, state_b_new, 'B')
    
    # Test 4: B moves to already visited cell
    state_b_visited = np.array([3, 0, 4, 1, 0, 0, 0], dtype=np.float32)  # B moves back to (4,1)
    bonus_a, bonus_b = sim.calculate_bonus(state_b_new, state_b_visited, 'A', 'B')
    print(f"\nTest 4 - B moves back to visited cell (4,1):")
    print(f"Bonus A: {bonus_a}, Bonus B: {bonus_b}")
    
    print(f"\nFinal visited cells:")
    print(f"Agent A: {sorted(sim.visited_cells['A'])}")
    print(f"Agent B: {sorted(sim.visited_cells['B'])}")
    print(f"Visited stats: {sim.get_visited_stats()}")


if __name__ == "__main__":
    test_social_motivation()
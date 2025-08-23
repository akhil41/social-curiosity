#!/usr/bin/env python3
"""
Simple test script for the GridWorld environment.
"""

import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from environment import GridWorldEnv

def test_basic_functionality():
    """Test basic environment functionality."""
    print("Testing GridWorld Environment Basic Functionality")
    print("=" * 50)
    
    env = GridWorldEnv()
    
    # Test reset
    state = env.reset()
    print(f"Initial state: {state}")
    env.render()
    
    # Test some basic movements
    print("Testing basic movements...")
    
    # Move agent A right
    print("\nMoving Agent A right...")
    state, rewards, done, info = env.step(env.ACTION_RIGHT, env.ACTION_STAY)
    print(f"State: {state}")
    print(f"Rewards: {rewards}")
    print(f"Done: {done}")
    env.render()
    
    # Move agent B right
    print("\nMoving Agent B right...")
    state, rewards, done, info = env.step(env.ACTION_STAY, env.ACTION_RIGHT)
    print(f"State: {state}")
    print(f"Rewards: {rewards}")
    print(f"Done: {done}")
    env.render()
    
    # Test boundary conditions
    print("\nTesting boundary conditions...")
    env.reset()
    # Try to move agent A left (should stay at boundary)
    state, rewards, done, info = env.step(env.ACTION_LEFT, env.ACTION_STAY)
    print(f"State after trying to move left from boundary: {state}")
    env.render()

def test_pressure_plates_and_door():
    """Test pressure plate and door functionality."""
    print("\n" + "=" * 50)
    print("Testing Pressure Plates and Door Functionality")
    print("=" * 50)
    
    env = GridWorldEnv()
    env.reset()
    
    # Move agents to pressure plates
    print("Moving agents to pressure plates...")
    
    # Agent A to pressure plate A (1,1)
    # From (0,0): right, down
    env.step(env.ACTION_RIGHT, env.ACTION_STAY)  # A: (1,0), B: (4,0)
    env.step(env.ACTION_DOWN, env.ACTION_STAY)   # A: (1,1), B: (4,0)
    
    # Agent B to pressure plate B (3,1)  
    # From (4,0): left, down
    env.step(env.ACTION_STAY, env.ACTION_LEFT)   # A: (1,1), B: (3,0)
    env.step(env.ACTION_STAY, env.ACTION_DOWN)   # A: (1,1), B: (3,1)
    
    # Both should now be on pressure plates - door should open
    state, rewards, done, info = env.step(env.ACTION_STAY, env.ACTION_STAY)
    print(f"State with both on pressure plates: {state}")
    print(f"Door open: {info['door_open']}")
    print(f"Rewards: {rewards} (should include +3 for door opening)")
    env.render()

def test_coin_collection():
    """Test coin collection functionality."""
    print("\n" + "=" * 50)
    print("Testing Coin Collection Functionality")
    print("=" * 50)
    
    env = GridWorldEnv()
    env.reset()
    
    # First open the door
    # Move agents to pressure plates
    env.step(env.ACTION_RIGHT, env.ACTION_STAY)  # A: (1,0), B: (4,0)
    env.step(env.ACTION_DOWN, env.ACTION_STAY)   # A: (1,1), B: (4,0)
    env.step(env.ACTION_STAY, env.ACTION_LEFT)   # A: (1,1), B: (3,0)
    env.step(env.ACTION_STAY, env.ACTION_DOWN)   # A: (1,1), B: (3,1)
    state, rewards, done, info = env.step(env.ACTION_STAY, env.ACTION_STAY)  # Door opens
    
    print("Door opened, now testing coin collection...")
    
    # Move agent A to collect coin 1 at (2,3)
    # From (1,1): right, right, down, down
    env.step(env.ACTION_RIGHT, env.ACTION_STAY)  # A: (2,1), B: (3,1)
    env.step(env.ACTION_RIGHT, env.ACTION_STAY)  # A: (3,1), B: (3,1) - collision!
    env.reset()  # Reset due to collision
    
    # Try again with better path
    env.step(env.ACTION_RIGHT, env.ACTION_STAY)  # A: (1,0), B: (4,0)
    env.step(env.ACTION_DOWN, env.ACTION_STAY)   # A: (1,1), B: (4,0)
    env.step(env.ACTION_STAY, env.ACTION_LEFT)   # A: (1,1), B: (3,0)
    env.step(env.ACTION_STAY, env.ACTION_DOWN)   # A: (1,1), B: (3,1)
    state, rewards, done, info = env.step(env.ACTION_STAY, env.ACTION_STAY)  # Door opens
    
    # Now move agent A to coin
    env.step(env.ACTION_RIGHT, env.ACTION_STAY)  # A: (2,1), B: (3,1)
    env.step(env.ACTION_DOWN, env.ACTION_STAY)   # A: (2,2), B: (3,1)
    env.step(env.ACTION_DOWN, env.ACTION_STAY)   # A: (2,3), B: (3,1) - collect coin 1!
    
    state, rewards, done, info = env.step(env.ACTION_STAY, env.ACTION_STAY)
    print(f"State after coin collection: {state}")
    print(f"Coins taken: {info['coins_taken']}")
    print(f"Rewards: {rewards} (should include +1 for coin collection)")
    env.render()

if __name__ == "__main__":
    test_basic_functionality()
    test_pressure_plates_and_door()
    test_coin_collection()
    
    print("\n" + "=" * 50)
    print("All tests completed!")
    print("=" * 50)
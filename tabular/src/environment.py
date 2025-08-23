import numpy as np
from typing import Tuple, Dict, Any, List, Optional

class GridWorldEnv:
    """
    5x5 GridWorld environment for two agents with pressure plates, door, and coins.
    
    State representation: (ax, ay, bx, by, door_open, c1_taken, c2_taken)
    - ax, ay: Agent A's x,y coordinates (0-4)
    - bx, by: Agent B's x,y coordinates (0-4)
    - door_open: Boolean (0 or 1) indicating if door is open
    - c1_taken, c2_taken: Boolean (0 or 1) indicating if coins are collected
    
    Actions per agent: stay, up, down, left, right (5 actions each)
    """
    
    # Action mapping
    ACTION_STAY = 0
    ACTION_UP = 1
    ACTION_DOWN = 2
    ACTION_LEFT = 3
    ACTION_RIGHT = 4
    
    ACTION_NAMES = {
        ACTION_STAY: "stay",
        ACTION_UP: "up",
        ACTION_DOWN: "down",
        ACTION_LEFT: "left",
        ACTION_RIGHT: "right"
    }
    
    def __init__(self):
        """Initialize the 5x5 GridWorld environment."""
        # Grid dimensions
        self.grid_size = 5
        
        # Fixed positions
        self.pressure_plate_a = (1, 1)  # Pressure plate A position
        self.pressure_plate_b = (3, 1)  # Pressure plate B position
        self.door_position = (2, 2)     # Door position
        self.coin_positions = [(2, 3), (2, 4)]  # Coin positions
        
        # Initial positions
        self.initial_a_pos = (0, 0)  # Agent A starting position
        self.initial_b_pos = (4, 0)  # Agent B starting position
        
        # Action space
        self.action_space = 5  # 5 actions per agent
        
        # State space dimensions
        self.state_dim = 7  # (ax, ay, bx, by, door_open, c1_taken, c2_taken)
        
        # Step penalty
        self.step_penalty = -0.01
        
        # Initialize state variables
        self.agent_a_pos: List[int] = [0, 0]
        self.agent_b_pos: List[int] = [0, 0]
        self.door_open: bool = False
        self.coins_taken: List[bool] = [False, False]
        
        # Reset the environment to initial state
        self.reset()
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment to initial state.
        
        Returns:
            np.ndarray: Initial state vector
        """
        self.agent_a_pos = list(self.initial_a_pos)
        self.agent_b_pos = list(self.initial_b_pos)
        self.door_open = False
        self.coins_taken = [False, False]  # [coin1_taken, coin2_taken]
        
        return self._get_state()
    
    def step(self, action_a: int, action_b: int) -> Tuple[np.ndarray, Tuple[float, float], bool, Dict[str, Any]]:
        """
        Execute one time step in the environment.
        
        Args:
            action_a: Action for agent A (0-4)
            action_b: Action for agent B (0-4)
            
        Returns:
            Tuple containing:
            - next_state: New state after actions
            - rewards: Tuple of (reward_a, reward_b)
            - done: Whether episode is done
            - info: Additional information
        """
        # Store previous state for reward calculation
        prev_door_open = self.door_open
        prev_coins_taken = self.coins_taken.copy()
        
        # Move agents
        self._move_agent('A', action_a)
        self._move_agent('B', action_b)
        
        # Check for collisions (agents can't occupy same cell)
        self._handle_collisions()
        
        # Check pressure plates and update door status
        self._update_door_status()
        
        # Check coin collection
        self._check_coin_collection()
        
        # Calculate rewards
        rewards = self._calculate_rewards(prev_door_open, prev_coins_taken)
        
        # Check termination
        done = self._check_termination()
        
        # Get next state
        next_state = self._get_state()
        
        info = {
            'door_open': self.door_open,
            'coins_taken': self.coins_taken.copy(),
            'agent_a_pos': tuple(self.agent_a_pos),
            'agent_b_pos': tuple(self.agent_b_pos)
        }
        
        return next_state, rewards, done, info
    
    def _move_agent(self, agent: str, action: int):
        """Move an agent based on the given action."""
        if agent == 'A':
            pos = self.agent_a_pos
        else:
            pos = self.agent_b_pos
        
        # Create new position by copying the current position
        new_x, new_y = pos[0], pos[1]
        
        if action == self.ACTION_UP:
            new_y = max(0, pos[1] - 1)
        elif action == self.ACTION_DOWN:
            new_y = min(self.grid_size - 1, pos[1] + 1)
        elif action == self.ACTION_LEFT:
            new_x = max(0, pos[0] - 1)
        elif action == self.ACTION_RIGHT:
            new_x = min(self.grid_size - 1, pos[0] + 1)
        # ACTION_STAY: no movement
        
        # Update position
        if agent == 'A':
            self.agent_a_pos = [new_x, new_y]
        else:
            self.agent_b_pos = [new_x, new_y]
    
    def _handle_collisions(self):
        """Handle collisions between agents (agents can't occupy same cell)."""
        if (self.agent_a_pos[0] == self.agent_b_pos[0] and 
            self.agent_a_pos[1] == self.agent_b_pos[1]):
            # Agents collided, revert to previous positions (simplified handling)
            # In a more complex implementation, you might want different collision logic
            self.agent_a_pos = list(self.initial_a_pos)
            self.agent_b_pos = list(self.initial_b_pos)
    
    def _update_door_status(self):
        """Update door status based on pressure plate activation."""
        # Check if both agents are on their respective pressure plates
        a_on_plate = (self.agent_a_pos[0] == self.pressure_plate_a[0] and 
                      self.agent_a_pos[1] == self.pressure_plate_a[1])
        b_on_plate = (self.agent_b_pos[0] == self.pressure_plate_b[0] and 
                      self.agent_b_pos[1] == self.pressure_plate_b[1])
        
        # Door opens only if both pressure plates are activated simultaneously
        if a_on_plate and b_on_plate and not self.door_open:
            self.door_open = True
    
    def _check_coin_collection(self):
        """Check if agents collect coins (only when door is open)."""
        if not self.door_open:
            return
        
        # Check coin 1
        coin1_pos = self.coin_positions[0]
        if (not self.coins_taken[0] and 
            (self.agent_a_pos[0] == coin1_pos[0] and self.agent_a_pos[1] == coin1_pos[1] or
             self.agent_b_pos[0] == coin1_pos[0] and self.agent_b_pos[1] == coin1_pos[1])):
            self.coins_taken[0] = True
        
        # Check coin 2
        coin2_pos = self.coin_positions[1]
        if (not self.coins_taken[1] and 
            (self.agent_a_pos[0] == coin2_pos[0] and self.agent_a_pos[1] == coin2_pos[1] or
             self.agent_b_pos[0] == coin2_pos[0] and self.agent_b_pos[1] == coin2_pos[1])):
            self.coins_taken[1] = True
    
    def _calculate_rewards(self, prev_door_open: bool, prev_coins_taken: List[bool]) -> Tuple[float, float]:
        """Calculate rewards for both agents."""
        reward_a = self.step_penalty
        reward_b = self.step_penalty
        
        # Door opening reward (both agents get +3 when door opens)
        if self.door_open and not prev_door_open:
            reward_a += 3.0
            reward_b += 3.0
        
        # Coin collection rewards (individual rewards)
        for i in range(2):
            if self.coins_taken[i] and not prev_coins_taken[i]:
                # Check which agent collected the coin
                coin_pos = self.coin_positions[i]
                if (self.agent_a_pos[0] == coin_pos[0] and self.agent_a_pos[1] == coin_pos[1]):
                    reward_a += 1.0
                elif (self.agent_b_pos[0] == coin_pos[0] and self.agent_b_pos[1] == coin_pos[1]):
                    reward_b += 1.0
        
        return (reward_a, reward_b)
    
    def _check_termination(self) -> bool:
        """Check if the episode should terminate."""
        # Episode ends when both coins are collected
        return all(self.coins_taken)
    
    def _get_state(self) -> np.ndarray:
        """Get the current state as a numpy array."""
        return np.array([
            self.agent_a_pos[0],  # ax
            self.agent_a_pos[1],  # ay
            self.agent_b_pos[0],  # bx
            self.agent_b_pos[1],  # by
            1.0 if self.door_open else 0.0,  # door_open
            1.0 if self.coins_taken[0] else 0.0,  # c1_taken
            1.0 if self.coins_taken[1] else 0.0   # c2_taken
        ], dtype=np.float32)
    
    def render(self):
        """Render the current state of the environment (text-based)."""
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        # Mark special positions
        grid[self.pressure_plate_a[1]][self.pressure_plate_a[0]] = 'P'
        grid[self.pressure_plate_b[1]][self.pressure_plate_b[0]] = 'P'
        grid[self.door_position[1]][self.door_position[0]] = 'D' if not self.door_open else ' '
        
        # Mark coins
        for i, (cx, cy) in enumerate(self.coin_positions):
            if not self.coins_taken[i]:
                grid[cy][cx] = 'C'
        
        # Mark agents
        grid[self.agent_a_pos[1]][self.agent_a_pos[0]] = 'A'
        grid[self.agent_b_pos[1]][self.agent_b_pos[0]] = 'B'
        
        # Print the grid
        print("GridWorld State:")
        for row in grid:
            print(' '.join(row))
        print(f"Door open: {self.door_open}")
        print(f"Coins taken: {self.coins_taken}")
        print()
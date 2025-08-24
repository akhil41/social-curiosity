"""
Gymnasium environment for Social Curiosity project.
Implements the same 5x5 GridWorld as the tabular version with pressure plates, door, and coins.
"""
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SocialCuriosityGridWorld(gym.Env):
    """
    5x5 GridWorld environment for two agents with pressure plates, door, and coins.
    
    State representation: (ax, ay, bx, by, door_open, c1_taken, c2_taken)
    - ax, ay: Agent A's x,y coordinates (0-4)
    - bx, by: Agent B's x,y coordinates (0-4)
    - door_open: Boolean (0 or 1) indicating if door is open
    - c1_taken, c2_taken: Boolean (0 or 1) indicating if coins are collected
    
    Actions: Combined actions for both agents (25 total actions)
    - Action encoding: agent_a_action * 5 + agent_b_action
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
    
    # Combined action space for two agents (5x5 = 25 actions)
    # Action encoding: agent_a_action * 5 + agent_b_action
    
    def __init__(self, intrinsic_coef: float = 0.0):
        """
        Initialize the 5x5 GridWorld environment.
        
        Args:
            intrinsic_coef: Coefficient for intrinsic motivation reward (default: 0.0)
        """
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
        self.action_space = spaces.Discrete(25)  # Combined actions for both agents (5x5)
        
        # State space dimensions
        self.state_dim = 7  # (ax, ay, bx, by, door_open, c1_taken, c2_taken)
        
        # Step penalty
        self.step_penalty = -0.01
        
        # Intrinsic motivation coefficient
        self.intrinsic_coef = intrinsic_coef
        
        # Initialize state variables
        self.agent_a_pos: List[int] = [0, 0]
        self.agent_b_pos: List[int] = [0, 0]
        self.door_open: bool = False
        self.coins_taken: List[bool] = [False, False]
        
        # Track visited cells for social curiosity reward
        self.visited_cells: Dict[str, set] = {
            'agent_0': set(),
            'agent_1': set()
        }
        
        # Track previous positions for social curiosity calculation
        self.previous_positions: Dict[str, tuple] = {
            'agent_0': (0, 0),
            'agent_1': (4, 0)
        }
        
        # Agent names
        self.agents = ["agent_0", "agent_1"]
        self.possible_agents = ["agent_0", "agent_1"]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        
        # Observation and action spaces
        # Observation space (single-agent interface)
        self.observation_space = spaces.Box(low=0, high=4, shape=(self.state_dim,), dtype=np.float32)
        
        # Reset the environment to initial state
        self.reset()
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset
            
        Returns:
            Tuple containing:
            - observation: Current state observation
            - info: Additional information dictionary
        """
        if seed is not None:
            np.random.seed(seed)
            
        self.agent_a_pos = list(self.initial_a_pos)
        self.agent_b_pos = list(self.initial_b_pos)
        self.door_open = False
        self.coins_taken = [False, False]  # [coin1_taken, coin2_taken]
        
        # Reset visited cells
        self.visited_cells = {
            'agent_0': {(0, 0)},
            'agent_1': {(4, 0)}
        }
        
        # Reset previous positions
        self.previous_positions = {
            'agent_0': (0, 0),
            'agent_1': (4, 0)
        }
        
        # Get observation
        observation = self._get_state()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one time step in the environment.
        
        Args:
            action: Combined action for both agents (0-24)
            
        Returns:
            Tuple containing:
            - observation: Current state observation
            - reward: Combined reward for both agents
            - terminated: Episode termination flag
            - truncated: Episode truncation flag
            - info: Additional information dictionary
        """
        # Store previous state for reward calculation
        prev_door_open = self.door_open
        prev_coins_taken = self.coins_taken.copy()
        prev_agent_a_pos = tuple(self.agent_a_pos)
        prev_agent_b_pos = tuple(self.agent_b_pos)
        
        # Decode combined action into individual agent actions
        action_a = action // 5  # Integer division to get agent A's action
        action_b = action % 5   # Modulo to get agent B's action
        
        # Move agents
        self._move_agent('A', action_a)
        self._move_agent('B', action_b)
        
        # Check for collisions (agents can't occupy same cell)
        self._handle_collisions()
        
        # Check pressure plates and update door status
        self._update_door_status()
        
        # Check coin collection
        self._check_coin_collection()
        
        # Update visited cells
        # Store current positions as previous positions for next step
        self.previous_positions['agent_0'] = (self.agent_a_pos[0], self.agent_a_pos[1])
        self.previous_positions['agent_1'] = (self.agent_b_pos[0], self.agent_b_pos[1])
        
        # Add current positions to visited cells
        self.visited_cells['agent_0'].add(tuple(self.agent_a_pos))
        self.visited_cells['agent_1'].add(tuple(self.agent_b_pos))
        
        # Calculate rewards
        rewards = self._calculate_rewards(prev_door_open, prev_coins_taken,
                                        prev_agent_a_pos, prev_agent_b_pos)
        
        # Combine rewards (average of both agents' rewards)
        combined_reward = (rewards["agent_0"] + rewards["agent_1"]) / 2.0
        
        # Check termination
        done = self._check_termination()
        
        # Get observation
        observation = self._get_state()
        
        # Info
        info = self._get_info()
        
        return observation, combined_reward, done, False, info
    
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
    
    def _calculate_rewards(self, prev_door_open: bool, prev_coins_taken: List[bool],
                          prev_agent_a_pos: tuple, prev_agent_b_pos: tuple) -> Dict[str, float]:
        """Calculate rewards for both agents including social curiosity bonus."""
        # Base rewards
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
        
        # Add social curiosity bonus if enabled
        if self.intrinsic_coef > 0:
            bonus_a, bonus_b = self._calculate_social_curiosity_bonus(
                prev_agent_a_pos, prev_agent_b_pos)
            reward_a += bonus_a
            reward_b += bonus_b
        
        return {"agent_0": reward_a, "agent_1": reward_b}
    
    def _calculate_social_curiosity_bonus(self, prev_agent_a_pos: tuple,
                                         prev_agent_b_pos: tuple) -> Tuple[float, float]:
        """
        Calculate social curiosity bonus for both agents.
        
        Args:
            prev_agent_a_pos: Previous position of agent A
            prev_agent_b_pos: Previous position of agent B
            
        Returns:
            Tuple of (bonus_a, bonus_b) for the two agents
        """
        bonus_a = 0.0
        bonus_b = 0.0
        
        # Check for agent A -> agent B social bonus
        # Agent A gets bonus if agent B explores a new cell and A was nearby
        if self._should_give_bonus(prev_agent_a_pos, self.previous_positions['agent_1'], 'agent_1'):
            bonus_a += self.intrinsic_coef
        
        # Check for agent B -> agent A social bonus
        # Agent B gets bonus if agent A explores a new cell and B was nearby
        if self._should_give_bonus(prev_agent_b_pos, self.previous_positions['agent_0'], 'agent_0'):
            bonus_b += self.intrinsic_coef
        
        return bonus_a, bonus_b
    
    def _should_give_bonus(self, observer_prev_pos: tuple,
                          explorer_prev_pos: tuple, explorer_id: str) -> bool:
        """
        Check if social bonus should be given to observer based on explorer's action.
        
        Conditions:
        1. Explorer enters a previously unseen cell at t+1
        2. Observer was within Manhattan distance 1 of explorer at time t
        
        Args:
            observer_prev_pos: Previous position of observer
            explorer_prev_pos: Previous position of explorer
            explorer_id: ID of explorer agent
            
        Returns:
            True if bonus should be given, False otherwise
        """
        # Get explorer's current position
        if explorer_id == 'agent_0':
            explorer_curr_pos = (self.agent_a_pos[0], self.agent_a_pos[1])
        else:
            explorer_curr_pos = (self.agent_b_pos[0], self.agent_b_pos[1])
        
        # If explorer didn't move, no bonus
        if explorer_prev_pos == explorer_curr_pos:
            return False
        
        # Check if explorer's new cell was previously unseen
        # We check if the current position is already in visited cells before this step
        # If it's not, then this is a new exploration
        if explorer_curr_pos in self.visited_cells[explorer_id]:
            # If it was already visited, this is not a new exploration
            return False
        
        # Check if observer was within Manhattan distance 1 of explorer at time t
        distance = self._manhattan_distance(observer_prev_pos, explorer_prev_pos)
        
        return distance <= 1
    
    def _manhattan_distance(self, pos1: tuple, pos2: tuple) -> int:
        """
        Calculate Manhattan distance between two positions.
        
        Args:
            pos1: First position (x1, y1)
            pos2: Second position (x2, y2)
            
        Returns:
            Manhattan distance: |x1 - x2| + |y1 - y2|
        """
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
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
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about the current state."""
        return {
            'door_open': self.door_open,
            'coins_taken': self.coins_taken.copy(),
            'agent_a_pos': tuple(self.agent_a_pos),
            'agent_b_pos': tuple(self.agent_b_pos)
        }
    
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
    
# Factory function for creating the environment
def make_env(intrinsic_coef: float = 0.0) -> SocialCuriosityGridWorld:
    """
    Create a Social Curiosity GridWorld environment instance.
    
    Args:
        intrinsic_coef: Coefficient for intrinsic motivation reward (default: 0.0)
        
    Returns:
        SocialCuriosityGridWorld instance
    """
    env = SocialCuriosityGridWorld(intrinsic_coef=intrinsic_coef)
    return env
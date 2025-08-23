import numpy as np
from typing import Dict, Tuple, Optional
import pickle
import os


class TabularQLearningAgent:
    """
    Independent Tabular Q-learning Agent with ε-greedy exploration.
    
    This agent implements the standard Q-learning algorithm with a tabular
    representation of the state-action value function.
    
    State representation: (ax, ay, bx, by, door_open, c1_taken, c2_taken)
    - ax, ay: Agent A's x,y coordinates (0-4)
    - bx, by: Agent B's x,y coordinates (0-4)
    - door_open: Boolean (0 or 1) indicating if door is open
    - c1_taken, c2_taken: Boolean (0 or 1) indicating if coins are collected
    
    Actions: stay, up, down, left, right (5 actions)
    """
    
    def __init__(self, 
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 initial_epsilon: float = 1.0,
                 epsilon_decay: float = 0.999,
                 min_epsilon: float = 0.01):
        """
        Initialize the Q-learning agent.
        
        Args:
            learning_rate (float): Learning rate (α) for Q-learning updates
            discount_factor (float): Discount factor (γ) for future rewards
            initial_epsilon (float): Initial exploration rate for ε-greedy policy
            epsilon_decay (float): Decay rate for epsilon after each episode
            min_epsilon (float): Minimum exploration rate
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        # Initialize Q-table as a dictionary for sparse state representation
        # Key: state tuple (ax, ay, bx, by, door_open, c1_taken, c2_taken)
        # Value: numpy array of Q-values for each action (5 actions)
        self.q_table: Dict[Tuple[int, int, int, int, int, int, int], np.ndarray] = {}
        
        # Track number of updates for each state-action pair
        self.visit_counts: Dict[Tuple[Tuple[int, int, int, int, int, int, int], int], int] = {}
    
    def _get_state_key(self, state: np.ndarray) -> Tuple[int, int, int, int, int, int, int]:
        """
        Convert state array to hashable tuple key for Q-table.
        
        Args:
            state: State array (ax, ay, bx, by, door_open, c1_taken, c2_taken)
            
        Returns:
            Hashable tuple representation of the state
        """
        return (
            int(state[0]),  # ax
            int(state[1]),  # ay
            int(state[2]),  # bx
            int(state[3]),  # by
            int(state[4]),  # door_open
            int(state[5]),  # c1_taken
            int(state[6])   # c2_taken
        )
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """
        Get Q-values for the current state.
        
        Args:
            state: Current state array
            
        Returns:
            numpy array of Q-values for each action (5 actions)
        """
        state_key = self._get_state_key(state)
        
        # Initialize Q-values to zeros if state not seen before
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(5, dtype=np.float32)
            
        return self.q_table[state_key]
    
    def choose_action(self, state: np.ndarray, epsilon: Optional[float] = None) -> int:
        """
        Choose an action using ε-greedy policy.
        
        Args:
            state: Current state array
            epsilon: Optional exploration rate (uses self.epsilon if None)
            
        Returns:
            Selected action (0-4)
        """
        if epsilon is None:
            epsilon = self.epsilon
            
        q_values = self.get_q_values(state)
        state_key = self._get_state_key(state)
        
        # Exploration: random action
        if np.random.random() < epsilon:
            action = np.random.randint(0, 5)
        # Exploitation: best action
        else:
            action = np.argmax(q_values)
            
        # Track visit count
        visit_key = (state_key, action)
        self.visit_counts[visit_key] = self.visit_counts.get(visit_key, 0) + 1
            
        return action
    
    def update(self, 
               state: np.ndarray, 
               action: int, 
               reward: float, 
               next_state: np.ndarray, 
               done: bool) -> None:
        """
        Update Q-values using the Q-learning algorithm.
        
        Q(s, a) ← Q(s, a) + α * [r + γ * max_a' Q(s', a') - Q(s, a)]
        
        Args:
            state: Current state array
            action: Action taken
            reward: Reward received
            next_state: Next state array
            done: Whether episode terminated
        """
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)
        
        # Get current Q-value
        current_q = self.get_q_values(state)[action]
        
        # Calculate target Q-value
        if done:
            target = reward
        else:
            next_q_values = self.get_q_values(next_state)
            max_next_q = np.max(next_q_values)
            target = reward + self.discount_factor * max_next_q
        
        # Update Q-value using learning rate
        self.q_table[state_key][action] += self.learning_rate * (target - current_q)
    
    def decay_epsilon(self) -> None:
        """Decay exploration rate after each episode."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath: str) -> None:
        """
        Save Q-table and agent parameters to file.
        
        Args:
            filepath: Path to save the agent data
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        data = {
            'q_table': self.q_table,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'min_epsilon': self.min_epsilon,
            'visit_counts': self.visit_counts
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: str) -> None:
        """
        Load Q-table and agent parameters from file.
        
        Args:
            filepath: Path to load the agent data from
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Agent file not found: {filepath}")
            
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        self.q_table = data['q_table']
        self.learning_rate = data['learning_rate']
        self.discount_factor = data['discount_factor']
        self.epsilon = data['epsilon']
        self.epsilon_decay = data['epsilon_decay']
        self.min_epsilon = data['min_epsilon']
        self.visit_counts = data.get('visit_counts', {})
    
    def get_stats(self) -> Dict[str, float]:
        """
        Get agent statistics.
        
        Returns:
            Dictionary containing agent statistics
        """
        total_states = len(self.q_table)
        total_visits = sum(self.visit_counts.values())
        
        return {
            'total_states': total_states,
            'total_visits': total_visits,
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate
        }
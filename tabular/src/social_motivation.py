import numpy as np
from typing import Dict, Set, Tuple, Optional


class SocialIntrinsicMotivation:
    """
    Social Intrinsic Motivation (SIM) signal for multi-agent GridWorld.
    
    Provides a bonus when a teammate enters a previously unseen cell at t+1
    and the agent was within Manhattan distance 1 of the teammate at time t.
    This approximates social influence via proximity-enabled exploration.
    
    State representation: (ax, ay, bx, by, door_open, c1_taken, c2_taken)
    - ax, ay: Agent A's x,y coordinates (0-4)
    - bx, by: Agent B's x,y coordinates (0-4)
    - door_open: Boolean (0 or 1) indicating if door is open
    - c1_taken, c2_taken: Boolean (0 or 1) indicating if coins are collected
    """
    
    def __init__(self, intrinsic_coef: float = 0.2):
        """
        Initialize the Social Intrinsic Motivation module.
        
        Args:
            intrinsic_coef: Bonus coefficient for social intrinsic motivation (default: 0.2)
        """
        self.intrinsic_coef = intrinsic_coef
        
        # Track visited cells per agent
        self.visited_cells: Dict[str, Set[Tuple[int, int]]] = {
            'A': set(),
            'B': set()
        }
    
    def _extract_agent_positions(self, state: np.ndarray) -> Dict[str, Tuple[int, int]]:
        """
        Extract agent positions from state array.
        
        Args:
            state: State array (ax, ay, bx, by, door_open, c1_taken, c2_taken)
            
        Returns:
            Dictionary mapping agent IDs to their (x, y) positions
        """
        return {
            'A': (int(state[0]), int(state[1])),  # (ax, ay)
            'B': (int(state[2]), int(state[3]))   # (bx, by)
        }
    
    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """
        Calculate Manhattan distance between two positions.
        
        Args:
            pos1: First position (x1, y1)
            pos2: Second position (x2, y2)
            
        Returns:
            Manhattan distance: |x1 - x2| + |y1 - y2|
        """
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def update(self, prev_state: np.ndarray, next_state: np.ndarray, agent_id: str) -> None:
        """
        Update visited cells for an agent based on state transition.
        
        Args:
            prev_state: Previous state array
            next_state: Next state array
            agent_id: Agent ID ('A' or 'B')
        """
        # Extract agent's position from next state
        agent_positions = self._extract_agent_positions(next_state)
        current_pos = agent_positions[agent_id]
        
        # Add current position to visited cells
        self.visited_cells[agent_id].add(current_pos)
    
    def calculate_bonus(self, prev_state: np.ndarray, next_state: np.ndarray, 
                       agent_a_id: str, agent_b_id: str) -> Tuple[float, float]:
        """
        Calculate social intrinsic motivation bonuses for both agents.
        
        Args:
            prev_state: Previous state array at time t
            next_state: Next state array at time t+1
            agent_a_id: ID of first agent ('A' or 'B')
            agent_b_id: ID of second agent ('A' or 'B')
            
        Returns:
            Tuple of (bonus_a, bonus_b) for the two agents
        """
        # Extract positions from previous and next states
        prev_positions = self._extract_agent_positions(prev_state)
        next_positions = self._extract_agent_positions(next_state)
        
        bonus_a = 0.0
        bonus_b = 0.0
        
        # Check for agent A -> agent B social bonus
        if self._should_give_bonus(prev_positions, next_positions, agent_a_id, agent_b_id):
            bonus_a += self.intrinsic_coef
        
        # Check for agent B -> agent A social bonus
        if self._should_give_bonus(prev_positions, next_positions, agent_b_id, agent_a_id):
            bonus_b += self.intrinsic_coef
        
        return bonus_a, bonus_b
    
    def _should_give_bonus(self, prev_positions: Dict[str, Tuple[int, int]], 
                          next_positions: Dict[str, Tuple[int, int]], 
                          observer_id: str, explorer_id: str) -> bool:
        """
        Check if social bonus should be given from observer to explorer.
        
        Conditions:
        1. Explorer enters a previously unseen cell at t+1
        2. Observer was within Manhattan distance 1 of explorer at time t
        
        Args:
            prev_positions: Agent positions at time t
            next_positions: Agent positions at time t+1
            observer_id: ID of agent that might receive bonus
            explorer_id: ID of agent that explored new cell
            
        Returns:
            True if bonus should be given, False otherwise
        """
        # Check if explorer moved to a new cell
        explorer_prev_pos = prev_positions[explorer_id]
        explorer_next_pos = next_positions[explorer_id]
        
        # If explorer didn't move, no bonus
        if explorer_prev_pos == explorer_next_pos:
            return False
        
        # Check if explorer's new cell was previously unseen
        if explorer_next_pos in self.visited_cells[explorer_id]:
            return False
        
        # Check if observer was within Manhattan distance 1 of explorer at time t
        observer_prev_pos = prev_positions[observer_id]
        distance = self._manhattan_distance(observer_prev_pos, explorer_prev_pos)
        
        return distance <= 1
    
    def reset(self) -> None:
        """
        Reset visited cells for both agents.
        """
        self.visited_cells['A'] = set()
        self.visited_cells['B'] = set()
        
        # Add initial positions to visited cells (agents start at (0,0) and (4,0))
        self.visited_cells['A'].add((0, 0))
        self.visited_cells['B'].add((4, 0))
    
    def get_visited_stats(self) -> Dict[str, int]:
        """
        Get statistics about visited cells.
        
        Returns:
            Dictionary with number of visited cells per agent
        """
        return {
            'A': len(self.visited_cells['A']),
            'B': len(self.visited_cells['B'])
        }
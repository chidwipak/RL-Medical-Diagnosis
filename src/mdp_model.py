"""
MDP Model for Medical Diagnosis Environment

This module provides an explicit model of the MDP including:
- Transition probabilities P(s'|s,a)
- Reward function R(s,a,s')

Used by Policy Iteration and Value Iteration algorithms.
"""

import numpy as np
from typing import Tuple, Dict, List


class MDPModel:
    """
    Explicit MDP Model for the Medical Diagnosis Environment.
    
    This class provides the transition and reward functions needed for
    dynamic programming algorithms (Policy Iteration, Value Iteration).
    
    Attributes:
        n_states (int): Number of states (32)
        n_actions (int): Number of actions (12)
        gamma (float): Discount factor
        P (np.ndarray): Transition probabilities P[s,a,s']
        R (np.ndarray): Expected rewards R[s,a]
    """
    
    # Disease patterns (same as environment)
    DISEASE_PATTERNS = {
        0: np.array([1, 0, 1, 0, 1]),  # Flu
        1: np.array([0, 1, 0, 1, 1])   # Pneumonia
    }
    
    # Reward constants
    REWARD_ASK_SYMPTOM = 0.2
    REWARD_ORDER_TEST = -0.3
    REWARD_CORRECT_DIAGNOSIS = 10.0
    REWARD_WRONG_DIAGNOSIS = -5.0
    REWARD_STEP_COST = -0.01
    
    ACTION_NAMES = [
        "Ask Fever", "Ask Cough", "Ask Fatigue", 
        "Ask Breath", "Ask Chest Pain",
        "Test Fever", "Test Cough", "Test Fatigue",
        "Test Breath", "Test Chest Pain",
        "Diagnose Flu", "Diagnose Pneumonia"
    ]
    
    SYMPTOM_NAMES = ["Fever", "Cough", "Fatigue", "Breath", "Chest Pain"]
    
    def __init__(self, gamma: float = 0.9, disease: int = None):
        """
        Initialize the MDP model.
        
        Args:
            gamma: Discount factor (default: 0.9)
            disease: If specified (0 or 1), model for that disease only.
                    If None, average over both diseases (50% each).
        """
        self.n_states = 32  # 2^5 states
        self.n_actions = 12  # 5 asks + 5 tests + 2 diagnoses
        self.n_symptoms = 5
        self.gamma = gamma
        self.disease = disease
        
        # Terminal state indicator
        self.terminal_state = 32  # Use state 32 as absorbing terminal state
        self.n_states_with_terminal = 33
        
        # Build transition and reward matrices
        self._build_model()
    
    def _build_model(self):
        """Build transition probability and reward matrices."""
        n_s = self.n_states_with_terminal
        n_a = self.n_actions
        
        # P[s, a, s'] = probability of transitioning to s' from s via a
        self.P = np.zeros((n_s, n_a, n_s))
        
        # R[s, a] = expected reward for taking action a in state s
        self.R = np.zeros((n_s, n_a))
        
        # Terminal state is absorbing
        for a in range(n_a):
            self.P[self.terminal_state, a, self.terminal_state] = 1.0
            self.R[self.terminal_state, a] = 0.0
        
        # Build transitions for non-terminal states
        for s in range(self.n_states):
            for a in range(n_a):
                self._set_transition(s, a)
    
    def _set_transition(self, s: int, a: int):
        """Set transition probability and reward for state-action pair."""
        if a < 5:
            # Ask symptom
            symptom_idx = a
            reward = self.REWARD_ASK_SYMPTOM + self.REWARD_STEP_COST
            
            # New state with symptom revealed
            s_new = s | (1 << symptom_idx)
            
            self.P[s, a, s_new] = 1.0
            self.R[s, a] = reward
            
        elif a < 10:
            # Order test
            symptom_idx = a - 5
            reward = self.REWARD_ORDER_TEST + self.REWARD_STEP_COST
            
            # New state with symptom revealed
            s_new = s | (1 << symptom_idx)
            
            self.P[s, a, s_new] = 1.0
            self.R[s, a] = reward
            
        else:
            # Diagnose (terminal action)
            diagnosed_disease = a - 10  # 0=Flu, 1=Pneumonia
            
            if self.disease is not None:
                # Single disease model
                if diagnosed_disease == self.disease:
                    reward = self.REWARD_CORRECT_DIAGNOSIS + self.REWARD_STEP_COST
                else:
                    reward = self.REWARD_WRONG_DIAGNOSIS + self.REWARD_STEP_COST
            else:
                # Average over both diseases (50% each)
                # P(correct) = 0.5 if diagnosed matches true
                # Expected reward = 0.5 * correct + 0.5 * wrong
                reward = 0.5 * (self.REWARD_CORRECT_DIAGNOSIS + self.REWARD_STEP_COST) + \
                         0.5 * (self.REWARD_WRONG_DIAGNOSIS + self.REWARD_STEP_COST)
            
            # Transition to terminal state
            self.P[s, a, self.terminal_state] = 1.0
            self.R[s, a] = reward
    
    def get_transition_prob(self, s: int, a: int, s_next: int) -> float:
        """
        Get transition probability P(s'|s,a).
        
        Args:
            s: Current state
            a: Action taken
            s_next: Next state
        
        Returns:
            Probability of transitioning to s_next
        """
        return self.P[s, a, s_next]
    
    def get_reward(self, s: int, a: int) -> float:
        """
        Get expected reward R(s,a).
        
        Args:
            s: Current state
            a: Action taken
        
        Returns:
            Expected reward for taking action a in state s
        """
        return self.R[s, a]
    
    def get_next_states(self, s: int, a: int) -> List[Tuple[int, float]]:
        """
        Get all possible next states and their probabilities.
        
        Args:
            s: Current state
            a: Action taken
        
        Returns:
            List of (next_state, probability) tuples
        """
        next_states = []
        for s_next in range(self.n_states_with_terminal):
            prob = self.P[s, a, s_next]
            if prob > 0:
                next_states.append((s_next, prob))
        return next_states
    
    def is_terminal(self, s: int) -> bool:
        """Check if state is terminal."""
        return s == self.terminal_state
    
    def get_state_description(self, s: int) -> str:
        """Get human-readable state description."""
        if s == self.terminal_state:
            return "Terminal"
        
        binary = format(s, '05b')
        known = []
        for i in range(5):
            if s & (1 << i):
                known.append(self.SYMPTOM_NAMES[i])
        
        if not known:
            return f"State {s} ({binary}): No symptoms known"
        return f"State {s} ({binary}): {', '.join(known)} known"
    
    def print_model_summary(self):
        """Print a summary of the MDP model."""
        print("=" * 60)
        print("MDP Model Summary")
        print("=" * 60)
        print(f"States: {self.n_states} (+ 1 terminal = {self.n_states_with_terminal})")
        print(f"Actions: {self.n_actions}")
        print(f"Discount Factor (γ): {self.gamma}")
        print(f"Disease: {'Both (averaged)' if self.disease is None else ['Flu', 'Pneumonia'][self.disease]}")
        print()
        
        print("Reward Structure:")
        print(f"  Ask Symptom: {self.REWARD_ASK_SYMPTOM:+.2f}")
        print(f"  Order Test:  {self.REWARD_ORDER_TEST:+.2f}")
        print(f"  Correct Dx:  {self.REWARD_CORRECT_DIAGNOSIS:+.2f}")
        print(f"  Wrong Dx:    {self.REWARD_WRONG_DIAGNOSIS:+.2f}")
        print(f"  Step Cost:   {self.REWARD_STEP_COST:+.2f}")
        print("=" * 60)


class DiseaseSpecificMDP:
    """
    MDP model that handles both diseases properly.
    
    For Policy Iteration and Value Iteration, we need to consider
    that the true disease is hidden. We create separate models for
    each disease and can compute values for both.
    """
    
    def __init__(self, gamma: float = 0.9):
        """
        Initialize disease-specific MDP models.
        
        Args:
            gamma: Discount factor
        """
        self.gamma = gamma
        self.flu_model = MDPModel(gamma=gamma, disease=0)
        self.pneumonia_model = MDPModel(gamma=gamma, disease=1)
        
        self.n_states = 32
        self.n_states_with_terminal = 33
        self.n_actions = 12
        self.terminal_state = 32
    
    def get_value_for_disease(self, V: np.ndarray, s: int, a: int, disease: int) -> float:
        """
        Get expected value for a state-action pair given a specific disease.
        
        Args:
            V: Value function
            s: Current state
            a: Action
            disease: Disease (0=Flu, 1=Pneumonia)
        
        Returns:
            Expected value
        """
        model = self.flu_model if disease == 0 else self.pneumonia_model
        
        value = 0.0
        for s_next, prob in model.get_next_states(s, a):
            r = model.get_reward(s, a)
            value += prob * (r + self.gamma * V[s_next])
        
        return value
    
    def get_expected_value(self, V: np.ndarray, s: int, a: int) -> float:
        """
        Get expected value averaged over both diseases.
        
        Args:
            V: Value function
            s: Current state
            a: Action
        
        Returns:
            Expected value (averaged over diseases)
        """
        # Equal probability for each disease
        v_flu = self.get_value_for_disease(V, s, a, 0)
        v_pneumonia = self.get_value_for_disease(V, s, a, 1)
        return 0.5 * v_flu + 0.5 * v_pneumonia


if __name__ == "__main__":
    # Test the MDP model
    print("\n" + "=" * 60)
    print("Testing MDP Model")
    print("=" * 60)
    
    # Create model for Flu
    model_flu = MDPModel(disease=0)
    model_flu.print_model_summary()
    
    # Test transitions
    print("\nTransition examples from State 0:")
    for a in range(12):
        next_states = model_flu.get_next_states(0, a)
        reward = model_flu.get_reward(0, a)
        print(f"  Action {a} ({model_flu.ACTION_NAMES[a]}): "
              f"→ {next_states}, R = {reward:.2f}")
    
    # Test state descriptions
    print("\nState descriptions:")
    for s in [0, 1, 3, 7, 31]:
        print(f"  {model_flu.get_state_description(s)}")

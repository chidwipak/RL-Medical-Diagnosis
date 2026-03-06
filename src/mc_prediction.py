"""
Monte Carlo Prediction for Medical Diagnosis

Evaluates a given policy (e.g., random policy) to learn the State-Value function V(s)
without needing transition probabilities.

Algorithm (First-Visit MC):
1. Initialize V(s) = 0, N(s) = 0 for all states
2. Generate episode using policy pi
3. For each state s in episode (first occurrence):
   - Calculate return G from s onwards
   - N(s) += 1
   - V(s) += (1/N(s)) * (G - V(s))
"""

import numpy as np
import time
from typing import List, Tuple, Dict

class MCPrediction:
    DISEASE_PATTERNS = {
        0: np.array([1, 1, 1, 0, 0]), 1: np.array([1, 1, 0, 1, 0]),
        2: np.array([1, 0, 1, 0, 1]), 3: np.array([1, 0, 0, 1, 1]),
        4: np.array([0, 1, 1, 0, 1]), 5: np.array([0, 1, 0, 1, 0]),
        6: np.array([0, 0, 1, 1, 0]), 7: np.array([0, 0, 0, 0, 1]),
    }

    def __init__(self, gamma: float = 0.9):
        self.n_states = 244
        self.gamma = gamma
        self.terminal_state = 243
        self.R_ASK = -0.1
        self.R_CORRECT = 10.0
        self.R_WRONG = -5.0
        
        self.V = np.zeros(self.n_states)
        self.N = np.zeros(self.n_states)

    def state_to_symptom_status(self, state: int) -> List[int]:
        statuses = []
        for i in range(5):
            statuses.append(state % 3)
            state //= 3
        return statuses

    def symptom_status_to_state(self, statuses: List[int]) -> int:
        state = 0
        for i in range(4, -1, -1):
            state = state * 3 + statuses[i]
        return state

    def get_valid_actions(self, state: int) -> List[int]:
        if state == self.terminal_state:
            return []
        statuses = self.state_to_symptom_status(state)
        valid = [i for i in range(5) if statuses[i] == 0]
        valid.extend(range(5, 13))
        return valid

    def random_policy(self, state: int) -> int:
        """Evaluate a uniform random policy."""
        valid = self.get_valid_actions(state)
        return np.random.choice(valid) if valid else 0

    def step(self, state: int, action: int, disease: int) -> Tuple[int, float, bool]:
        """Simulate one step."""
        if action >= 5:
            diag = action - 5
            return self.terminal_state, self.R_CORRECT if diag == disease else self.R_WRONG, True

        symptom_idx = action
        statuses = self.state_to_symptom_status(state)
        
        if statuses[symptom_idx] != 0:
            return state, self.R_ASK, False
            
        statuses[symptom_idx] = self.DISEASE_PATTERNS[disease][symptom_idx] + 1
        return self.symptom_status_to_state(statuses), self.R_ASK, False

    def generate_episode(self, disease: int) -> List[Tuple[int, float]]:
        """Generate episode using random policy."""
        episode = []
        state = 0
        for _ in range(15):
            action = self.random_policy(state)
            next_state, reward, done = self.step(state, action, disease)
            episode.append((state, reward))
            if done:
                break
            state = next_state
        return episode

    def run(self, n_episodes: int = 10000):
        print("Starting Monte Carlo Prediction (Random Policy)...")
        self.V = np.zeros(self.n_states)
        self.N = np.zeros(self.n_states)
        
        for ep in range(n_episodes):
            disease = np.random.randint(0, 8)
            episode = self.generate_episode(disease)
            
            G = 0.0
            visited = set()
            for t in range(len(episode) - 1, -1, -1):
                s, r = episode[t]
                G = r + self.gamma * G
                if s not in visited:
                    visited.add(s)
                    self.N[s] += 1
                    self.V[s] += (G - self.V[s]) / self.N[s]
                    
            if (ep + 1) % 2000 == 0:
                print(f"Episode {ep+1:5d} | V(s=0) = {self.V[0]:.4f}")
                
        print(f"Final V(s=0): {self.V[0]:.4f}")
        return self.V

if __name__ == "__main__":
    mc = MCPrediction()
    mc.run(10000)

"""
TD(0) Prediction (One-Step TD) for Medical Diagnosis

Evaluates a given policy (e.g., random policy) to learn the State-Value function V(s)
by bootstrapping after every step.

Algorithm:
1. Initialize V(s) = 0
2. For each episode step (s, a, r, s'):
   - V(s) += alpha * [r + gamma * V(s') - V(s)]
"""

import numpy as np
from typing import List, Tuple

class TDPrediction:
    DISEASE_PATTERNS = {
        0: np.array([1, 1, 1, 0, 0]), 1: np.array([1, 1, 0, 1, 0]),
        2: np.array([1, 0, 1, 0, 1]), 3: np.array([1, 0, 0, 1, 1]),
        4: np.array([0, 1, 1, 0, 1]), 5: np.array([0, 1, 0, 1, 0]),
        6: np.array([0, 0, 1, 1, 0]), 7: np.array([0, 0, 0, 0, 1]),
    }

    def __init__(self, gamma: float = 0.9, alpha: float = 0.1):
        self.n_states = 244
        self.gamma = gamma
        self.alpha = alpha
        self.terminal_state = 243
        self.R_ASK = -0.1
        self.R_CORRECT = 10.0
        self.R_WRONG = -5.0
        self.V = np.zeros(self.n_states)

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
        valid = self.get_valid_actions(state)
        return np.random.choice(valid) if valid else 0

    def step(self, state: int, action: int, disease: int) -> Tuple[int, float, bool]:
        if action >= 5:
            diag = action - 5
            return self.terminal_state, self.R_CORRECT if diag == disease else self.R_WRONG, True
        symptom_idx = action
        statuses = self.state_to_symptom_status(state)
        if statuses[symptom_idx] != 0:
            return state, self.R_ASK, False
        statuses[symptom_idx] = self.DISEASE_PATTERNS[disease][symptom_idx] + 1
        return self.symptom_status_to_state(statuses), self.R_ASK, False

    def run(self, n_episodes: int = 10000):
        print("Starting TD(0) Prediction (Random Policy)...")
        self.V = np.zeros(self.n_states)
        
        for ep in range(n_episodes):
            state = 0
            disease = np.random.randint(0, 8)
            
            for _ in range(15):
                action = self.random_policy(state)
                next_state, reward, done = self.step(state, action, disease)
                
                # TD(0) Update
                td_target = reward + self.gamma * self.V[next_state]
                td_error = td_target - self.V[state]
                self.V[state] += self.alpha * td_error
                
                if done:
                    break
                state = next_state
                
            if (ep + 1) % 2000 == 0:
                print(f"Episode {ep+1:5d} | V(s=0) = {self.V[state]:.4f}")
                
        print(f"Final V(s=0): {self.V[0]:.4f}")
        return self.V

if __name__ == "__main__":
    td = TDPrediction(alpha=0.05)
    td.run(10000)

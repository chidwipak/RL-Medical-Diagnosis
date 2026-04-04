"""
Value Iteration for Medical Diagnosis

I implemented Value Iteration as an alternative to Policy Iteration.
This algorithm directly computes optimal values using Bellman optimality equation.

My Algorithm Steps:
1. I initialize V(s) = 0 for all states
2. I update each V(s) = max_a [Q(s,a)] where Q(s,a) = R + γ * E[V(s')]
3. I repeat until the maximum change (delta) is below threshold
4. I extract the final policy by picking argmax_a Q(s,a) for each state

Author: K. Chidwipak
Roll No: S20230010131
"""

import numpy as np
from typing import Dict, List
import time


class ValueIteration:
    """
    My implementation of Value Iteration for medical diagnosis.
    Same disease patterns as Policy Iteration for consistency.
    """
    
    # Same 8 diseases as Policy Iteration
    DISEASE_PATTERNS = {
        0: np.array([1, 1, 1, 0, 0]),  # Flu
        1: np.array([1, 1, 0, 1, 0]),  # Strep
        2: np.array([1, 0, 1, 0, 1]),  # Pneumonia
        3: np.array([1, 0, 0, 1, 1]),  # Bronchitis
        4: np.array([0, 1, 1, 0, 1]),  # Cold
        5: np.array([0, 1, 0, 1, 0]),  # Allergy
        6: np.array([0, 0, 1, 1, 0]),  # Asthma
        7: np.array([0, 0, 0, 0, 1]),  # Migraine
    }
    
    DISEASE_NAMES = ["Flu", "Strep", "Pneumonia", "Bronchitis",
                     "Cold", "Allergy", "Asthma", "Migraine"]
    SYMPTOM_NAMES = ["Fever", "Cough", "Fatigue", "Breath", "Headache"]
    
    def __init__(self, gamma: float = 0.9, theta: float = 1e-6):
        """Initialize my Value Iteration solver."""
        self.n_states = 244
        self.n_actions = 13
        self.gamma = gamma
        self.theta = theta
        self.terminal_state = 243
        
        self.R_ASK = -0.1
        self.R_CORRECT = 10.0
        self.R_WRONG = -5.0
        
        self.V = None
        self.policy = None
        self.history = []
    
    def state_to_symptom_status(self, state: int) -> List[int]:
        """I convert state to symptom status list."""
        statuses = []
        for i in range(5):
            statuses.append(state % 3)
            state //= 3
        return statuses
    
    def symptom_status_to_state(self, statuses: List[int]) -> int:
        """I convert symptom statuses back to state number."""
        state = 0
        for i in range(4, -1, -1):
            state = state * 3 + statuses[i]
        return state
    
    def get_compatible_diseases(self, state: int) -> List[int]:
        """I find which diseases match the known symptoms."""
        statuses = self.state_to_symptom_status(state)
        compatible = []
        
        for disease in range(8):
            is_compatible = True
            for i in range(5):
                if statuses[i] == 0:
                    continue
                expected = self.DISEASE_PATTERNS[disease][i] + 1
                if statuses[i] != expected:
                    is_compatible = False
                    break
            if is_compatible:
                compatible.append(disease)
        
        return compatible if compatible else list(range(8))
    
    def get_next_state(self, state: int, action: int, disease: int) -> int:
        """I compute the next state after an action."""
        if action >= 5:
            return self.terminal_state
        
        statuses = self.state_to_symptom_status(state)
        symptom_idx = action
        
        if statuses[symptom_idx] != 0:
            return state
        
        symptom_value = self.DISEASE_PATTERNS[disease][symptom_idx]
        statuses[symptom_idx] = symptom_value + 1
        
        return self.symptom_status_to_state(statuses)
    
    def get_reward(self, action: int, disease: int) -> float:
        """I return reward for action given true disease."""
        if action < 5:
            return self.R_ASK
        else:
            diagnosed = action - 5
            return self.R_CORRECT if diagnosed == disease else self.R_WRONG
    
    def _get_expected_value(self, s: int, a: int, V: np.ndarray) -> float:
        """I calculate Q(s,a) by averaging over compatible diseases."""
        if s == self.terminal_state:
            return 0.0
        
        compatible = self.get_compatible_diseases(s)
        if not compatible:
            return -100.0
        
        total = 0.0
        p_each = 1.0 / len(compatible)
        
        for disease in compatible:
            r = self.get_reward(a, disease)
            s_next = self.get_next_state(s, a, disease)
            total += p_each * (r + self.gamma * V[s_next])
        
        return total
    
    def run(self, verbose: bool = True) -> Dict:
        """
        I run Value Iteration to find optimal values and policy.
        Unlike Policy Iteration, I directly maximize Q-values each step.
        """
        start = time.time()
        
        self.V = np.zeros(self.n_states)
        self.history = []
        
        if verbose:
            print("=" * 60)
            print("VALUE ITERATION - Medical Diagnosis")
            print("=" * 60)
        
        for iteration in range(1000):
            delta = 0.0
            
            # Update V(s) for each state
            for s in range(self.n_states - 1):
                v_old = self.V[s]
                Q = np.array([self._get_expected_value(s, a, self.V) 
                             for a in range(self.n_actions)])
                self.V[s] = np.max(Q)
                delta = max(delta, abs(v_old - self.V[s]))
            
            self.history.append({'delta': delta, 'V0': self.V[0]})
            
            if verbose and iteration < 10:
                print(f"Iter {iteration+1}: Δ = {delta:.6f}, V(s=0) = {self.V[0]:.4f}")
            
            if delta < self.theta:
                if verbose:
                    print(f"Converged at iteration {iteration+1}!")
                break
        
        # Extract optimal policy
        self.policy = np.zeros(self.n_states, dtype=np.int32)
        for s in range(self.n_states - 1):
            Q = np.array([self._get_expected_value(s, a, self.V) 
                         for a in range(self.n_actions)])
            self.policy[s] = np.argmax(Q)
        
        return {
            'V': self.V,
            'policy': self.policy,
            'iterations': len(self.history),
            'elapsed_time': time.time() - start,
            'converged': True
        }
    
    def get_action_name(self, action: int) -> str:
        """I return human-readable action name."""
        if action < 5:
            return f"Ask {self.SYMPTOM_NAMES[action]}"
        return f"Diagnose {self.DISEASE_NAMES[action - 5]}"

    def simulate_episode(self, patient_symptoms, verbose=False):
        """Simulate a diagnosis episode for given patient symptoms."""
        best_disease = 0
        best_match = -1
        for d in range(8):
            match = sum(patient_symptoms[i] == self.DISEASE_PATTERNS[d][i] for i in range(5))
            if match > best_match:
                best_match, best_disease = match, d

        state = 0
        path_bits = [0]
        actions_taken = []

        for step in range(10):
            action = self.policy[state]
            actions_taken.append(action)

            if action >= 5:
                diagnosed = action - 5
                return {
                    'true_disease': best_disease,
                    'diagnosed': diagnosed,
                    'success': diagnosed == best_disease,
                    'path_bits': path_bits,
                    'actions': actions_taken,
                    'steps': step + 1
                }

            symptom_idx = action
            symptom_val = patient_symptoms[symptom_idx]
            statuses = self.state_to_symptom_status(state)
            statuses[symptom_idx] = symptom_val + 1
            state = self.symptom_status_to_state(statuses)

            new_bits = sum((1 << i) if statuses[i] != 0 else 0 for i in range(5))
            path_bits.append(new_bits)

        return {'diagnosed': None, 'success': False, 'path_bits': path_bits, 'steps': 10}


if __name__ == "__main__":
    vi = ValueIteration()
    results = vi.run()
    print(f"\nFinal V(s=0) = {results['V'][0]:.4f}")

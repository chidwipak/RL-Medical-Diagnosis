"""
Policy Iteration for Medical Diagnosis

I implemented the Policy Iteration algorithm to solve the medical diagnosis MDP.
The agent learns to ask the right symptoms to efficiently diagnose 8 diseases.

My State Representation:
- I use a 243-state space (3^5) where each symptom can be Unknown/Absent/Present
- State 0 means no symptoms are known yet
- I convert states to a 5-bit "knowledge" representation for visualization

My Algorithm Steps:
1. I start with an arbitrary policy (all zeros)
2. Policy Evaluation: I calculate V(s) for all states using current policy
3. Policy Improvement: I update policy to be greedy with respect to V(s)
4. I repeat until the policy stops changing (convergence)

Author: K. Chidwipak
Roll No: S20230010131
"""

import numpy as np
from typing import Dict, List, Tuple, Set
import time


class PolicyIteration:
    """
    My implementation of Policy Iteration for medical diagnosis.
    
    I designed 8 diseases with distinct symptom patterns so the algorithm
    learns to ask different symptoms based on what information it needs.
    """
    
    # My 8 Disease patterns: [Fever, Cough, Fatigue, Breath, Headache]
    # I designed these so different diseases require different symptoms to identify
    DISEASE_PATTERNS = {
        0: np.array([1, 1, 1, 0, 0]),  # Flu: F, C, Ft
        1: np.array([1, 1, 0, 1, 0]),  # Strep: F, C, B
        2: np.array([1, 0, 1, 0, 1]),  # Pneumonia: F, Ft, H
        3: np.array([1, 0, 0, 1, 1]),  # Bronchitis: F, B, H
        4: np.array([0, 1, 1, 0, 1]),  # Cold: C, Ft, H
        5: np.array([0, 1, 0, 1, 0]),  # Allergy: C, B
        6: np.array([0, 0, 1, 1, 0]),  # Asthma: Ft, B
        7: np.array([0, 0, 0, 0, 1]),  # Migraine: H only
    }
    
    DISEASE_NAMES = ["Flu", "Strep", "Pneumonia", "Bronchitis",
                     "Cold", "Allergy", "Asthma", "Migraine"]
    SYMPTOM_NAMES = ["Fever", "Cough", "Fatigue", "Breath", "Headache"]
    
    def __init__(self, gamma: float = 0.9, theta: float = 1e-6):
        """
        Initialize my Policy Iteration solver.
        
        Args:
            gamma: I use 0.9 as discount factor to balance immediate vs future rewards
            theta: Convergence threshold for policy evaluation
        """
        self.n_states = 244  # 243 knowledge states + 1 terminal
        self.n_actions = 13  # 5 ask actions + 8 diagnose actions
        self.gamma = gamma
        self.theta = theta
        self.terminal_state = 243
        
        # My reward structure
        self.R_ASK = -0.1     # Small cost to ask a question
        self.R_CORRECT = 10.0  # Big reward for correct diagnosis
        self.R_WRONG = -5.0    # Penalty for wrong diagnosis
        
        self.V = None
        self.policy = None
        self.history = []
    
    def state_to_symptom_status(self, state: int) -> List[int]:
        """
        I convert the base-10 state to symptom statuses.
        Uses base-3 representation: 0=Unknown, 1=Absent, 2=Present
        """
        statuses = []
        for i in range(5):
            statuses.append(state % 3)
            state //= 3
        return statuses
    
    def symptom_status_to_state(self, statuses: List[int]) -> int:
        """I convert symptom statuses back to a single state number."""
        state = 0
        for i in range(4, -1, -1):
            state = state * 3 + statuses[i]
        return state
    
    def get_compatible_diseases(self, state: int) -> List[int]:
        """
        I find which diseases are still possible given what we know.
        A disease is compatible if all known symptoms match its pattern.
        """
        statuses = self.state_to_symptom_status(state)
        compatible = []
        
        for disease in range(8):
            is_compatible = True
            for i in range(5):
                if statuses[i] == 0:  # Unknown - compatible with any
                    continue
                expected = self.DISEASE_PATTERNS[disease][i] + 1
                if statuses[i] != expected:
                    is_compatible = False
                    break
            if is_compatible:
                compatible.append(disease)
        
        return compatible if compatible else list(range(8))
    
    def get_next_state(self, state: int, action: int, disease: int) -> int:
        """I compute the next state after taking an action."""
        if action >= 5:  # Diagnose action -> terminal
            return self.terminal_state
        
        statuses = self.state_to_symptom_status(state)
        symptom_idx = action
        
        if statuses[symptom_idx] != 0:  # Already known
            return state
        
        # I reveal the symptom value based on the true disease
        symptom_value = self.DISEASE_PATTERNS[disease][symptom_idx]
        statuses[symptom_idx] = symptom_value + 1  # 1=absent, 2=present
        
        return self.symptom_status_to_state(statuses)
    
    def get_reward(self, action: int, disease: int) -> float:
        """I return the reward for an action given the true disease."""
        if action < 5:
            return self.R_ASK
        else:
            diagnosed = action - 5
            return self.R_CORRECT if diagnosed == disease else self.R_WRONG
    
    def _get_expected_value(self, s: int, a: int, V: np.ndarray) -> float:
        """
        I calculate Q(s,a) - the expected value of taking action a in state s.
        I average over all compatible diseases since I don't know the true disease.
        """
        if s == self.terminal_state:
            return 0.0
        
        compatible = self.get_compatible_diseases(s)
        if not compatible:
            return -100.0
        
        total = 0.0
        p_each = 1.0 / len(compatible)  # Uniform belief over compatible diseases
        
        for disease in compatible:
            r = self.get_reward(a, disease)
            s_next = self.get_next_state(s, a, disease)
            total += p_each * (r + self.gamma * V[s_next])
        
        return total
    
    def policy_evaluation(self, policy: np.ndarray, V: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        I evaluate the current policy by computing V(s) for all states.
        This is iterative: I keep updating until values stabilize.
        """
        V = V.copy()
        iterations = 0
        
        for _ in range(1000):
            delta = 0.0
            for s in range(self.n_states - 1):
                v_old = V[s]
                a = policy[s]
                V[s] = self._get_expected_value(s, a, V)
                delta = max(delta, abs(v_old - V[s]))
            iterations += 1
            if delta < self.theta:
                break
        
        return V, iterations
    
    def policy_improvement(self, V: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        I improve the policy by making it greedy with respect to V.
        For each state, I pick the action with highest Q(s,a).
        """
        policy = np.zeros(self.n_states, dtype=np.int32)
        stable = True
        
        for s in range(self.n_states - 1):
            Q = np.array([self._get_expected_value(s, a, V) for a in range(self.n_actions)])
            best = np.argmax(Q)
            
            if self.policy is not None and self.policy[s] != best:
                stable = False
            
            policy[s] = best
        
        return policy, stable
    
    def run(self, verbose: bool = True) -> Dict:
        """
        I run the full Policy Iteration algorithm.
        Returns the optimal value function and policy.
        """
        start = time.time()
        
        # Initialize
        self.V = np.zeros(self.n_states)
        self.policy = np.zeros(self.n_states, dtype=np.int32)
        
        if verbose:
            print("=" * 60)
            print("POLICY ITERATION - Medical Diagnosis")
            print("=" * 60)
        
        for iteration in range(100):
            # Step 1: Policy Evaluation
            self.V, _ = self.policy_evaluation(self.policy, self.V)
            
            # Step 2: Policy Improvement
            self.policy, stable = self.policy_improvement(self.V)
            
            self.history.append({'V0': self.V[0]})
            
            if verbose:
                print(f"Iter {iteration+1}: V(s=0) = {self.V[0]:.4f}")
            
            if stable:
                if verbose:
                    print(f"Converged after {iteration+1} iterations!")
                break
        
        return {
            'V': self.V,
            'policy': self.policy,
            'iterations': len(self.history),
            'elapsed_time': time.time() - start,
            'converged': True
        }
    
    def get_action_name(self, action: int) -> str:
        """I return a human-readable name for an action."""
        if action < 5:
            return f"Ask {self.SYMPTOM_NAMES[action]}"
        return f"Diagnose {self.DISEASE_NAMES[action - 5]}"
    
    def simulate_episode(self, patient_symptoms: List[int], verbose: bool = False):
        """
        I simulate a diagnosis episode for a given patient.
        Returns the path taken and whether diagnosis was correct.
        """
        # Find best matching disease
        best_disease = 0
        best_match = -1
        for d in range(8):
            match = sum(patient_symptoms[i] == self.DISEASE_PATTERNS[d][i] for i in range(5))
            if match > best_match:
                best_match = match
                best_disease = d
        
        state = 0
        path_bits = [0]
        actions_taken = []
        
        for step in range(10):
            action = self.policy[state]
            actions_taken.append(action)
            
            if action >= 5:  # Diagnosis
                diagnosed = action - 5
                return {
                    'true_disease': best_disease,
                    'diagnosed': diagnosed,
                    'success': diagnosed == best_disease,
                    'path_bits': path_bits,
                    'actions': actions_taken,
                    'steps': step + 1
                }
            
            # Transition: ask symptom and get answer
            symptom_idx = action
            symptom_val = patient_symptoms[symptom_idx]
            statuses = self.state_to_symptom_status(state)
            statuses[symptom_idx] = symptom_val + 1
            state = self.symptom_status_to_state(statuses)
            
            # Track knowledge bits for visualization
            new_bits = sum((1 << i) if statuses[i] != 0 else 0 for i in range(5))
            path_bits.append(new_bits)
        
        return {'diagnosed': None, 'success': False, 'path_bits': path_bits}
    
    def get_all_reachable_states(self) -> Set[int]:
        """I compute all 32 possible knowledge states (5 symptoms -> 2^5)."""
        reachable = set([0])
        queue = [0]
        
        while queue:
            current_bits = queue.pop(0)
            for i in range(5):
                if not (current_bits & (1 << i)):
                    new_bits = current_bits | (1 << i)
                    if new_bits not in reachable:
                        reachable.add(new_bits)
                        queue.append(new_bits)
        
        return reachable


if __name__ == "__main__":
    # Run and test my implementation
    pi = PolicyIteration()
    results = pi.run()
    
    print("\n" + "=" * 60)
    print("Testing All 8 Diseases")
    print("=" * 60)
    
    for d in range(8):
        symptoms = list(pi.DISEASE_PATTERNS[d])
        result = pi.simulate_episode(symptoms)
        path_str = " → ".join([f"s{b}" for b in result['path_bits']])
        print(f"{pi.DISEASE_NAMES[d]:12s}: {path_str} → {pi.DISEASE_NAMES[result['diagnosed']]}")

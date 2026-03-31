"""
Semi-Gradient SARSA with Function Approximation for Medical Diagnosis

TD-based on-policy control with linear function approximation.
Updates weights at every step using the semi-gradient TD(0) rule.

Update Rule (from FA slides):
    Δw = α (r + γ q̂(s',a',w) - q̂(s,a,w)) ∇_w q̂(s,a,w)
    
For linear FA: ∇_w q̂ = φ(s,a), so:
    Δw = α (r + γ φ(s',a')ᵀw - φ(s,a)ᵀw) φ(s,a)

Called "semi-gradient" because we only differentiate the prediction,
not the target (which also depends on w).

Author: K. Chidwipak
Roll No: S20230010131
"""

import numpy as np
from typing import Dict, List, Tuple
import time


class SARSAFunctionApprox:
    """
    Semi-Gradient SARSA with Linear Function Approximation.
    One-step TD control that updates weights every step.
    """

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

    def __init__(self, gamma: float = 0.9, alpha: float = 0.01,
                 epsilon_decay: float = 500.0):
        self.n_states = 244
        self.n_actions = 13
        self.gamma = gamma
        self.alpha_init = alpha
        self.epsilon_decay = epsilon_decay
        self.terminal_state = 243

        self.R_ASK = -0.1
        self.R_CORRECT = 10.0
        self.R_WRONG = -5.0

        self.n_features = 215
        self.w = None
        self.policy = None
        self.history = []

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

    def get_features(self, state: int, action: int) -> np.ndarray:
        """Build tile-coded feature vector φ(s,a) — same design as MC FA."""
        phi = np.zeros(self.n_features)

        if state == self.terminal_state:
            return phi

        statuses = self.state_to_symptom_status(state)

        # Tile-coded interaction: symptom × status × action
        for i in range(5):
            idx = i * (3 * 13) + statuses[i] * 13 + action
            phi[idx] = 1.0

        phi[195 + action] = 1.0

        n_known = sum(1 for s in statuses if s != 0)
        phi[208 + n_known] = 1.0

        phi[214] = 1.0

        return phi

    def q_hat(self, state: int, action: int) -> float:
        return np.dot(self.get_features(state, action), self.w)

    def get_valid_actions(self, state: int) -> List[int]:
        if state == self.terminal_state:
            return []
        statuses = self.state_to_symptom_status(state)
        valid = []
        for i in range(5):
            if statuses[i] == 0:
                valid.append(i)
        valid.extend(range(5, 13))
        return valid

    def epsilon_greedy_action(self, state: int, epsilon: float) -> int:
        valid = self.get_valid_actions(state)
        if not valid:
            return 0

        if np.random.random() < epsilon:
            return np.random.choice(valid)
        else:
            q_vals = [self.q_hat(state, a) for a in valid]
            return valid[np.argmax(q_vals)]

    def step(self, state: int, action: int, disease: int) -> Tuple[int, float, bool]:
        if action >= 5:
            diagnosed = action - 5
            reward = self.R_CORRECT if diagnosed == disease else self.R_WRONG
            return self.terminal_state, reward, True

        statuses = self.state_to_symptom_status(state)
        symptom_idx = action

        if statuses[symptom_idx] != 0:
            return state, self.R_ASK, False

        symptom_value = self.DISEASE_PATTERNS[disease][symptom_idx]
        statuses[symptom_idx] = symptom_value + 1
        next_state = self.symptom_status_to_state(statuses)

        return next_state, self.R_ASK, False

    def run(self, n_episodes: int = 50000, verbose: bool = True) -> Dict:
        """
        Run Semi-Gradient SARSA with Function Approximation.
        Updates weights at every step using TD(0) semi-gradient rule.
        """
        start = time.time()

        self.w = np.zeros(self.n_features)
        self.history = []

        if verbose:
            print("=" * 60)
            print("SEMI-GRADIENT SARSA WITH FUNCTION APPROXIMATION")
            print("=" * 60)

        total_rewards = []

        for ep in range(n_episodes):
            epsilon = 1.0 / (1.0 + ep / self.epsilon_decay)
            alpha = self.alpha_init / (1.0 + ep / (self.epsilon_decay * 20))

            disease = np.random.randint(0, 8)
            state = 0
            action = self.epsilon_greedy_action(state, epsilon)

            total_reward = 0.0

            for _ in range(15):
                next_state, reward, done = self.step(state, action, disease)
                total_reward += reward

                phi = self.get_features(state, action)

                if done:
                    # Terminal: TD target is just reward
                    td_error = reward - np.dot(phi, self.w)
                    self.w += alpha * td_error * phi
                    break

                next_action = self.epsilon_greedy_action(next_state, epsilon)

                # Semi-gradient SARSA update
                phi_next = self.get_features(next_state, next_action)
                td_error = reward + self.gamma * np.dot(phi_next, self.w) - np.dot(phi, self.w)
                self.w += alpha * td_error * phi

                state = next_state
                action = next_action

            total_rewards.append(total_reward)

            if (ep + 1) % 5000 == 0:
                avg_reward = np.mean(total_rewards[-5000:])
                q_s0_best = max(self.q_hat(0, a) for a in self.get_valid_actions(0))
                self.history.append({
                    'episode': ep + 1,
                    'epsilon': epsilon,
                    'avg_reward': avg_reward,
                    'Q_s0_max': q_s0_best,
                })
                if verbose:
                    print(f"Episode {ep+1:6d}: ε={epsilon:.4f}, "
                          f"Avg Reward={avg_reward:.3f}, "
                          f"Q(s=0,best)={q_s0_best:.4f}")

        self._extract_policy()

        elapsed = time.time() - start

        if verbose:
            print(f"\nCompleted in {elapsed:.2f}s")

        return {
            'w': self.w.copy(),
            'policy': self.policy,
            'history': self.history,
            'total_rewards': total_rewards,
            'elapsed_time': elapsed,
            'n_episodes': n_episodes,
        }

    def _extract_policy(self):
        self.policy = np.zeros(self.n_states, dtype=np.int32)
        for s in range(self.n_states - 1):
            valid = self.get_valid_actions(s)
            if valid:
                q_vals = [self.q_hat(s, a) for a in valid]
                self.policy[s] = valid[np.argmax(q_vals)]

    def get_action_name(self, action: int) -> str:
        if action < 5:
            return f"Ask {self.SYMPTOM_NAMES[action]}"
        return f"Diagnose {self.DISEASE_NAMES[action - 5]}"

    def simulate_episode(self, patient_symptoms: List[int], verbose: bool = False):
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

        return {'diagnosed': None, 'success': False, 'path_bits': path_bits}


if __name__ == "__main__":
    sarsa_fa = SARSAFunctionApprox()
    results = sarsa_fa.run(n_episodes=50000)

    print("\nTesting all 8 diseases:")
    correct = 0
    for d in range(8):
        symptoms = list(sarsa_fa.DISEASE_PATTERNS[d])
        result = sarsa_fa.simulate_episode(symptoms)
        status = "✓" if result['success'] else "✗"
        if result['success']:
            correct += 1
        print(f"  {sarsa_fa.DISEASE_NAMES[d]:12s}: {status} "
              f"→ {sarsa_fa.DISEASE_NAMES[result['diagnosed']]}, "
              f"steps={result['steps']}")
    print(f"\nAccuracy: {correct}/8 ({100*correct/8:.0f}%)")

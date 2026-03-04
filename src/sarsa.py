"""
SARSA (TD(0) On-Policy Control) for Medical Diagnosis

Model-free on-policy control that learns Q(s,a) using one-step TD updates.
Updates happen after every step (unlike Monte Carlo which waits for episode end).

Update Rule:
    Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]

Where (s,a,r,s',a') comes from following ε-greedy policy.

Author: K. Chidwipak
Roll No: S20230010131
"""

import numpy as np
from typing import Dict, List, Tuple
import time


class SARSA:
    """
    SARSA (TD(0)) On-Policy Control for medical diagnosis.
    Learns Q-values using one-step temporal difference updates.
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

    def __init__(self, gamma: float = 0.9, alpha: float = 0.1, epsilon_decay: float = 500.0):
        """
        Initialize SARSA.

        Args:
            gamma: Discount factor
            alpha: Initial learning rate
            epsilon_decay: Controls epsilon decay rate
        """
        self.n_states = 244
        self.n_actions = 13
        self.gamma = gamma
        self.alpha_init = alpha
        self.epsilon_decay = epsilon_decay
        self.terminal_state = 243

        self.R_ASK = -0.1
        self.R_CORRECT = 10.0
        self.R_WRONG = -5.0

        self.Q = None
        self.policy = None
        self.history = []

    def state_to_symptom_status(self, state: int) -> List[int]:
        """Convert state to symptom status list (base-3)."""
        statuses = []
        for i in range(5):
            statuses.append(state % 3)
            state //= 3
        return statuses

    def symptom_status_to_state(self, statuses: List[int]) -> int:
        """Convert symptom statuses back to state number."""
        state = 0
        for i in range(4, -1, -1):
            state = state * 3 + statuses[i]
        return state

    def get_valid_actions(self, state: int) -> List[int]:
        """Get valid actions (don't re-ask known symptoms)."""
        if state == self.terminal_state:
            return []
        statuses = self.state_to_symptom_status(state)
        valid = []
        for i in range(5):
            if statuses[i] == 0:
                valid.append(i)
        valid.extend(range(5, 13))
        return valid

    def epsilon_greedy_action(self, state: int, Q: np.ndarray, epsilon: float) -> int:
        """Select action using epsilon-greedy policy."""
        valid = self.get_valid_actions(state)
        if not valid:
            return 0

        if np.random.random() < epsilon:
            return np.random.choice(valid)
        else:
            q_vals = [Q[state, a] for a in valid]
            best_idx = np.argmax(q_vals)
            return valid[best_idx]

    def step(self, state: int, action: int, disease: int) -> Tuple[int, float, bool]:
        """Simulate one environment step."""
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
        Run SARSA algorithm.

        SARSA update: Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
        """
        start = time.time()

        self.Q = np.zeros((self.n_states, self.n_actions))
        self.history = []

        if verbose:
            print("=" * 60)
            print("SARSA (TD(0)) CONTROL - Medical Diagnosis")
            print("=" * 60)

        total_rewards = []

        for ep in range(n_episodes):
            epsilon = 1.0 / (1.0 + ep / self.epsilon_decay)
            alpha = self.alpha_init / (1.0 + ep / (self.epsilon_decay * 10))

            disease = np.random.randint(0, 8)
            state = 0
            action = self.epsilon_greedy_action(state, self.Q, epsilon)

            total_reward = 0.0

            for _ in range(15):  # Max steps
                next_state, reward, done = self.step(state, action, disease)
                total_reward += reward

                if done:
                    # Terminal: Q(s',a') = 0
                    td_error = reward - self.Q[state, action]
                    self.Q[state, action] += alpha * td_error
                    break

                # Choose next action (this is the second A in S-A-R-S-A)
                next_action = self.epsilon_greedy_action(next_state, self.Q, epsilon)

                # SARSA update
                td_error = reward + self.gamma * self.Q[next_state, next_action] - self.Q[state, action]
                self.Q[state, action] += alpha * td_error

                state = next_state
                action = next_action

            total_rewards.append(total_reward)

            if (ep + 1) % 5000 == 0:
                avg_reward = np.mean(total_rewards[-5000:])
                self.history.append({
                    'episode': ep + 1,
                    'epsilon': epsilon,
                    'alpha': alpha,
                    'avg_reward': avg_reward,
                    'Q_s0_max': np.max(self.Q[0]),
                })
                if verbose:
                    print(f"Episode {ep+1:6d}: ε={epsilon:.4f}, α={alpha:.4f}, "
                          f"Avg Reward={avg_reward:.3f}, "
                          f"Q(s=0,best)={np.max(self.Q[0]):.4f}")

        # Extract greedy policy
        self.policy = np.zeros(self.n_states, dtype=np.int32)
        for s in range(self.n_states - 1):
            valid = self.get_valid_actions(s)
            if valid:
                q_vals = [self.Q[s, a] for a in valid]
                self.policy[s] = valid[np.argmax(q_vals)]

        elapsed = time.time() - start

        if verbose:
            print(f"\nCompleted in {elapsed:.2f}s")
            print(f"Final Q(s=0, best) = {np.max(self.Q[0]):.4f}")

        return {
            'Q': self.Q,
            'policy': self.policy,
            'history': self.history,
            'total_rewards': total_rewards,
            'elapsed_time': elapsed,
            'n_episodes': n_episodes,
        }

    def get_action_name(self, action: int) -> str:
        """Return human-readable action name."""
        if action < 5:
            return f"Ask {self.SYMPTOM_NAMES[action]}"
        return f"Diagnose {self.DISEASE_NAMES[action - 5]}"

    def simulate_episode(self, patient_symptoms: List[int], verbose: bool = False):
        """Simulate diagnosis using learned policy."""
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
    sarsa = SARSA()
    results = sarsa.run(n_episodes=50000)
    print(f"\nFinal Q(s=0, best) = {np.max(results['Q'][0]):.4f}")

    print("\nTesting all 8 diseases:")
    for d in range(8):
        symptoms = list(sarsa.DISEASE_PATTERNS[d])
        result = sarsa.simulate_episode(symptoms)
        status = "✓" if result['success'] else "✗"
        print(f"  {sarsa.DISEASE_NAMES[d]:12s}: {status} "
              f"→ {sarsa.DISEASE_NAMES[result['diagnosed']]}, "
              f"steps={result['steps']}")

"""
Monte Carlo Control with Function Approximation for Medical Diagnosis

Instead of storing Q-values in a table (like Assignment 2), I use a
linear function approximator: q̂(s,a,w) = φ(s,a)ᵀ w

The feature vector φ(s,a) encodes state-action information and the
weight vector w is learned from episode returns.

Update Rule (from FA slides):
    Δw = α (Gₜ - q̂(sₜ,aₜ,w)) ∇_w q̂(sₜ,aₜ,w)
    
For linear FA: ∇_w q̂ = φ(s,a), so:
    Δw = α (Gₜ - φ(sₜ,aₜ)ᵀw) φ(sₜ,aₜ)

Author: K. Chidwipak
Roll No: S20230010131
"""

import numpy as np
from typing import Dict, List, Tuple
import time


class MCFunctionApprox:
    """
    Monte Carlo Control with Linear Function Approximation.
    Uses feature vectors instead of Q-table for generalization.
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

        # Feature dimension: tile-coded interaction features
        # 195 (5 symptoms × 3 statuses × 13 actions) + 13 (action) + 6 (n_known bins) + 1 (bias) = 215
        self.n_features = 215
        self.w = None  # Weight vector
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

    def get_features(self, state: int, action: int) -> np.ndarray:
        """
        Build tile-coded feature vector φ(s,a) for linear function approximation.

        Features (tile coding for state-action interactions):
        - 195 tiles: for each symptom i, status j, action a → one feature
          Index = i * (3 * 13) + status[i] * 13 + action
          This captures "what is the value of taking action a when symptom i has status j"
        - 13 action one-hot features
        - 6 features: n_known (0..5) one-hot (how many symptoms known)
        - 1 bias feature
        Total: 195 + 13 + 6 + 1 = 215
        """
        phi = np.zeros(self.n_features)

        if state == self.terminal_state:
            return phi

        statuses = self.state_to_symptom_status(state)

        # Tile-coded interaction features: symptom × status × action
        for i in range(5):
            idx = i * (3 * 13) + statuses[i] * 13 + action
            phi[idx] = 1.0

        # Action one-hot (indices 195..207)
        phi[195 + action] = 1.0

        # Number of known symptoms one-hot (indices 208..213)
        n_known = sum(1 for s in statuses if s != 0)
        phi[208 + n_known] = 1.0

        # Bias
        phi[214] = 1.0

        return phi

    def q_hat(self, state: int, action: int) -> float:
        """Compute q̂(s,a,w) = φ(s,a)ᵀ w"""
        return np.dot(self.get_features(state, action), self.w)

    def get_valid_actions(self, state: int) -> List[int]:
        """Get valid actions for a state."""
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
        """Select action using epsilon-greedy with FA."""
        valid = self.get_valid_actions(state)
        if not valid:
            return 0

        if np.random.random() < epsilon:
            return np.random.choice(valid)
        else:
            q_vals = [self.q_hat(state, a) for a in valid]
            return valid[np.argmax(q_vals)]

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
        Run MC Control with Function Approximation.
        Uses complete episode returns to update weight vector.
        """
        start = time.time()

        self.w = np.zeros(self.n_features)
        self.history = []

        if verbose:
            print("=" * 60)
            print("MC CONTROL WITH FUNCTION APPROXIMATION")
            print("=" * 60)

        total_rewards = []

        for ep in range(n_episodes):
            epsilon = 1.0 / (1.0 + ep / self.epsilon_decay)
            alpha = self.alpha_init / (1.0 + ep / (self.epsilon_decay * 20))

            disease = np.random.randint(0, 8)

            # Generate complete episode
            episode = []
            state = 0
            for _ in range(15):
                action = self.epsilon_greedy_action(state, epsilon)
                next_state, reward, done = self.step(state, action, disease)
                episode.append((state, action, reward))
                if done:
                    break
                state = next_state

            total_reward = sum(r for _, _, r in episode)
            total_rewards.append(total_reward)

            # MC update: process episode backwards to compute returns
            G = 0.0
            for t in range(len(episode) - 1, -1, -1):
                s, a, r = episode[t]
                G = r + self.gamma * G

                # Gradient update: w += α(Gₜ - q̂(s,a,w)) φ(s,a)
                phi = self.get_features(s, a)
                q_val = np.dot(phi, self.w)
                self.w += alpha * (G - q_val) * phi

            # Log progress
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

        # Extract greedy policy
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
        """Extract greedy policy from learned weights."""
        self.policy = np.zeros(self.n_states, dtype=np.int32)
        for s in range(self.n_states - 1):
            valid = self.get_valid_actions(s)
            if valid:
                q_vals = [self.q_hat(s, a) for a in valid]
                self.policy[s] = valid[np.argmax(q_vals)]

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
    mc_fa = MCFunctionApprox()
    results = mc_fa.run(n_episodes=50000)

    print("\nTesting all 8 diseases:")
    correct = 0
    for d in range(8):
        symptoms = list(mc_fa.DISEASE_PATTERNS[d])
        result = mc_fa.simulate_episode(symptoms)
        status = "✓" if result['success'] else "✗"
        if result['success']:
            correct += 1
        print(f"  {mc_fa.DISEASE_NAMES[d]:12s}: {status} "
              f"→ {mc_fa.DISEASE_NAMES[result['diagnosed']]}, "
              f"steps={result['steps']}")
    print(f"\nAccuracy: {correct}/8 ({100*correct/8:.0f}%)")

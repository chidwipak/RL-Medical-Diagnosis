"""
Least Squares Policy Iteration (LSPI) for Medical Diagnosis

A batch method that solves for optimal weights directly using
least-squares instead of gradient descent. Uses LSTDQ for
policy evaluation and greedy improvement.

Algorithm (from FA slides):
1. Collect samples (s,a,r,s') using ε-greedy exploration
2. Policy Evaluation (LSTDQ):
   - Build A = Σ φ(sₜ,aₜ)(φ(sₜ,aₜ) - γ φ(s_{t+1}, π(s_{t+1})))ᵀ
   - Build b = Σ φ(sₜ,aₜ) r_{t+1}
   - Solve w = A⁻¹ b
3. Policy Improvement: π(s) = argmax_a q̂(s,a,w)
4. Repeat until policy converges

LSPI is sample-efficient: it reuses ALL collected data in every iteration.

Author: K. Chidwipak
Roll No: S20230010131
"""

import numpy as np
from typing import Dict, List, Tuple
import time


class LSPI:
    """
    Least Squares Policy Iteration for medical diagnosis.
    Uses LSTDQ for efficient batch policy evaluation.
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

    def __init__(self, gamma: float = 0.9, epsilon: float = 0.1):
        self.n_states = 244
        self.n_actions = 13
        self.gamma = gamma
        self.epsilon = epsilon
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
        """Build tile-coded feature vector φ(s,a) — same design as other FA algorithms."""
        phi = np.zeros(self.n_features)

        if state == self.terminal_state:
            return phi

        statuses = self.state_to_symptom_status(state)

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

    def greedy_action(self, state: int) -> int:
        """Select best action according to current weights."""
        valid = self.get_valid_actions(state)
        if not valid:
            return 0
        q_vals = [self.q_hat(state, a) for a in valid]
        return valid[np.argmax(q_vals)]

    def epsilon_greedy_action(self, state: int) -> int:
        valid = self.get_valid_actions(state)
        if not valid:
            return 0
        if np.random.random() < self.epsilon:
            return np.random.choice(valid)
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

    def collect_samples(self, n_episodes: int = 10000) -> List[Tuple]:
        """Collect experience samples using ε-greedy exploration."""
        samples = []

        for _ in range(n_episodes):
            disease = np.random.randint(0, 8)
            state = 0

            for _ in range(15):
                action = self.epsilon_greedy_action(state)
                next_state, reward, done = self.step(state, action, disease)

                samples.append((state, action, reward, next_state, done))

                if done:
                    break
                state = next_state

        return samples

    def lstdq(self, samples: List[Tuple]) -> np.ndarray:
        """
        LSTDQ: Least Squares Temporal Difference for Q-values.
        Solves for w = A⁻¹ b directly.
        """
        k = self.n_features
        A = np.eye(k) * 0.01  # Small regularization for stability
        b = np.zeros(k)

        for s, a, r, s_next, done in samples:
            phi = self.get_features(s, a)

            if done:
                phi_next = np.zeros(k)
            else:
                # Use greedy action under current policy for s'
                a_next = self.greedy_action(s_next)
                phi_next = self.get_features(s_next, a_next)

            A += np.outer(phi, phi - self.gamma * phi_next)
            b += phi * r

        # Solve w = A⁻¹ b
        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(A, b, rcond=None)[0]

        return w

    def run(self, n_sample_episodes: int = 10000, max_iterations: int = 20,
            verbose: bool = True) -> Dict:
        """
        Run LSPI: iterate between LSTDQ evaluation and greedy improvement.
        """
        start = time.time()

        self.w = np.zeros(self.n_features)
        self.history = []

        if verbose:
            print("=" * 60)
            print("LEAST SQUARES POLICY ITERATION (LSPI)")
            print("=" * 60)
            print(f"Collecting {n_sample_episodes} sample episodes...")

        # Collect samples once (LSPI reuses them)
        samples = self.collect_samples(n_sample_episodes)

        if verbose:
            print(f"Collected {len(samples)} transitions")

        total_rewards = []

        for iteration in range(max_iterations):
            w_old = self.w.copy()

            # LSTDQ: solve for best weights given current policy
            self.w = self.lstdq(samples)

            # Extract policy
            self._extract_policy()

            # Check convergence
            w_change = np.linalg.norm(self.w - w_old)

            # Evaluate current policy
            correct = 0
            for d in range(8):
                symptoms = list(self.DISEASE_PATTERNS[d])
                result = self.simulate_episode(symptoms)
                if result['success']:
                    correct += 1

            q_s0_best = max(self.q_hat(0, a) for a in self.get_valid_actions(0))

            self.history.append({
                'episode': (iteration + 1) * n_sample_episodes,
                'iteration': iteration + 1,
                'w_change': w_change,
                'accuracy': correct / 8,
                'Q_s0_max': q_s0_best,
                'avg_reward': q_s0_best,  # Approximation for compatibility
            })

            if verbose:
                print(f"Iter {iteration+1:3d}: Δw={w_change:.6f}, "
                      f"Accuracy={correct}/8, "
                      f"Q(s=0,best)={q_s0_best:.4f}")

            if w_change < 1e-6:
                if verbose:
                    print("Converged!")
                break

            # Collect new samples with updated policy every few iterations
            if (iteration + 1) % 5 == 0:
                new_samples = self.collect_samples(n_sample_episodes // 2)
                samples.extend(new_samples)

        elapsed = time.time() - start

        if verbose:
            print(f"\nCompleted in {elapsed:.2f}s")

        return {
            'w': self.w.copy(),
            'policy': self.policy,
            'history': self.history,
            'total_rewards': total_rewards,
            'elapsed_time': elapsed,
            'n_episodes': n_sample_episodes,
            'iterations': len(self.history),
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
    lspi = LSPI()
    results = lspi.run(n_sample_episodes=10000)

    print("\nTesting all 8 diseases:")
    correct = 0
    for d in range(8):
        symptoms = list(lspi.DISEASE_PATTERNS[d])
        result = lspi.simulate_episode(symptoms)
        status = "✓" if result['success'] else "✗"
        if result['success']:
            correct += 1
        print(f"  {lspi.DISEASE_NAMES[d]:12s}: {status} "
              f"→ {lspi.DISEASE_NAMES[result['diagnosed']]}, "
              f"steps={result['steps']}")
    print(f"\nAccuracy: {correct}/8 ({100*correct/8:.0f}%)")

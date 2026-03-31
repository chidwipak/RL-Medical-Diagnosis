"""
REINFORCE (Monte Carlo Policy Gradient) for Medical Diagnosis

Instead of learning Q-values and deriving a policy, REINFORCE directly
parameterizes and learns the policy using gradient ascent.

Policy: Softmax with linear features
    π_θ(a|s) = exp(φ(s,a)ᵀθ) / Σ_a' exp(φ(s,a')ᵀθ)

Score Function:
    ∇_θ log π_θ(s,a) = φ(s,a) - E_π[φ(s,·)]

Update Rule (from PG slides):
    Δθ = α ∇_θ log π_θ(sₜ,aₜ) Gₜ

Where Gₜ is the Monte Carlo return from timestep t.

Author: K. Chidwipak
Roll No: S20230010131
"""

import numpy as np
from typing import Dict, List, Tuple
import time


class REINFORCE:
    """
    REINFORCE (Monte Carlo Policy Gradient) for medical diagnosis.
    Directly learns a softmax policy parameterized by θ.
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

    def __init__(self, gamma: float = 0.9, alpha: float = 0.005):
        self.n_states = 244
        self.n_actions = 13
        self.gamma = gamma
        self.alpha_init = alpha
        self.terminal_state = 243

        self.R_ASK = -0.1
        self.R_CORRECT = 10.0
        self.R_WRONG = -5.0

        self.n_features = 215
        self.theta = None  # Policy parameters
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
        """Build tile-coded feature vector φ(s,a) — same design."""
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

    def softmax_probs(self, state: int) -> Tuple[List[int], np.ndarray]:
        """
        Compute softmax probabilities for valid actions.
        π_θ(a|s) = exp(φ(s,a)ᵀθ) / Σ exp(φ(s,a')ᵀθ)
        """
        valid = self.get_valid_actions(state)
        if not valid:
            return [], np.array([])

        logits = np.array([np.dot(self.get_features(state, a), self.theta) for a in valid])
        # Numerically stable softmax
        logits -= np.max(logits)
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits)

        return valid, probs

    def sample_action(self, state: int) -> int:
        """Sample action from softmax policy."""
        valid, probs = self.softmax_probs(state)
        if not valid:
            return 0
        return np.random.choice(valid, p=probs)

    def score_function(self, state: int, action: int) -> np.ndarray:
        """
        Compute score function: ∇_θ log π_θ(s,a)
        For softmax: φ(s,a) - E_π[φ(s,·)] = φ(s,a) - Σ_a' π(a'|s) φ(s,a')
        """
        valid, probs = self.softmax_probs(state)
        phi_sa = self.get_features(state, action)

        # Expected features under current policy
        expected_phi = np.zeros(self.n_features)
        for a, p in zip(valid, probs):
            expected_phi += p * self.get_features(state, a)

        return phi_sa - expected_phi

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
        Run REINFORCE algorithm.
        Generates episodes, computes returns, updates policy parameters.
        """
        start = time.time()

        self.theta = np.zeros(self.n_features)
        self.history = []

        if verbose:
            print("=" * 60)
            print("REINFORCE (MONTE CARLO POLICY GRADIENT)")
            print("=" * 60)

        total_rewards = []

        for ep in range(n_episodes):
            alpha = self.alpha_init / (1.0 + ep / 50000.0)

            disease = np.random.randint(0, 8)

            # Generate episode
            episode = []
            state = 0
            for _ in range(15):
                action = self.sample_action(state)
                next_state, reward, done = self.step(state, action, disease)
                episode.append((state, action, reward))
                if done:
                    break
                state = next_state

            total_reward = sum(r for _, _, r in episode)
            total_rewards.append(total_reward)

            # Compute returns and update θ
            G = 0.0
            for t in range(len(episode) - 1, -1, -1):
                s, a, r = episode[t]
                G = r + self.gamma * G

                # REINFORCE update: θ += α ∇_θ log π_θ(s,a) (Gₜ - baseline)
                score = self.score_function(s, a)
                # Clip gradient for stability
                score_norm = np.linalg.norm(score)
                if score_norm > 10.0:
                    score = score * 10.0 / score_norm
                self.theta += alpha * score * G

            # Log progress
            if (ep + 1) % 5000 == 0:
                avg_reward = np.mean(total_rewards[-5000:])
                # Estimate Q(s=0, best) from policy
                valid = self.get_valid_actions(0)
                q_est = max(np.dot(self.get_features(0, a), self.theta) for a in valid)
                self.history.append({
                    'episode': ep + 1,
                    'avg_reward': avg_reward,
                    'Q_s0_max': q_est,
                })
                if verbose:
                    print(f"Episode {ep+1:6d}: "
                          f"Avg Reward={avg_reward:.3f}, "
                          f"Score(s=0,best)={q_est:.4f}")

        self._extract_policy()

        elapsed = time.time() - start

        if verbose:
            print(f"\nCompleted in {elapsed:.2f}s")

        return {
            'theta': self.theta.copy(),
            'policy': self.policy,
            'history': self.history,
            'total_rewards': total_rewards,
            'elapsed_time': elapsed,
            'n_episodes': n_episodes,
        }

    def _extract_policy(self):
        """Extract greedy policy from softmax parameters."""
        self.policy = np.zeros(self.n_states, dtype=np.int32)
        for s in range(self.n_states - 1):
            valid = self.get_valid_actions(s)
            if valid:
                scores = [np.dot(self.get_features(s, a), self.theta) for a in valid]
                self.policy[s] = valid[np.argmax(scores)]

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
    reinforce = REINFORCE()
    results = reinforce.run(n_episodes=50000)

    print("\nTesting all 8 diseases:")
    correct = 0
    for d in range(8):
        symptoms = list(reinforce.DISEASE_PATTERNS[d])
        result = reinforce.simulate_episode(symptoms)
        status = "✓" if result['success'] else "✗"
        if result['success']:
            correct += 1
        print(f"  {reinforce.DISEASE_NAMES[d]:12s}: {status} "
              f"→ {reinforce.DISEASE_NAMES[result['diagnosed']]}, "
              f"steps={result['steps']}")
    print(f"\nAccuracy: {correct}/8 ({100*correct/8:.0f}%)")

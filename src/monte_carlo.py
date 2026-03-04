"""
GLIE Monte Carlo Control for Medical Diagnosis

Model-free control algorithm that learns Q(s,a) from complete episodes.
Unlike Policy/Value Iteration, this does NOT use transition probabilities.
Instead, it learns directly from experience using epsilon-greedy exploration.

GLIE = Greedy in the Limit with Infinite Exploration
- All state-action pairs are visited infinitely often
- Policy converges to greedy as epsilon → 0

Algorithm:
1. Generate episode using epsilon-greedy policy
2. For each first-visit (s,a) in episode, compute return G
3. Update Q(s,a) incrementally: Q(s,a) += (1/N(s,a)) * (G - Q(s,a))
4. Decay epsilon: epsilon = 1 / (1 + episode / decay_factor)

Author: K. Chidwipak
Roll No: S20230010131
"""

import numpy as np
from typing import Dict, List, Tuple
import time


class GLIEMonteCarlo:
    """
    GLIE Monte Carlo Control for medical diagnosis.
    Learns optimal Q-values and policy from complete episodes
    without needing transition probabilities.
    """

    # Same 8 diseases as Assignment 1
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

    def __init__(self, gamma: float = 0.9, epsilon_decay: float = 500.0):
        """
        Initialize GLIE Monte Carlo Control.

        Args:
            gamma: Discount factor
            epsilon_decay: Controls how fast epsilon decays. Higher = slower decay.
        """
        self.n_states = 244  # 243 knowledge states + 1 terminal
        self.n_actions = 13  # 5 ask + 8 diagnose
        self.gamma = gamma
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
        """Get valid actions for a state (don't re-ask known symptoms)."""
        if state == self.terminal_state:
            return []
        statuses = self.state_to_symptom_status(state)
        valid = []
        for i in range(5):
            if statuses[i] == 0:  # Unknown symptom
                valid.append(i)
        valid.extend(range(5, 13))  # All diagnose actions always valid
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
        """
        Simulate one environment step (no transition probabilities needed).

        Args:
            state: Current state
            action: Action taken
            disease: True disease of the patient

        Returns:
            (next_state, reward, done)
        """
        if action >= 5:
            # Diagnose action
            diagnosed = action - 5
            reward = self.R_CORRECT if diagnosed == disease else self.R_WRONG
            return self.terminal_state, reward, True

        # Ask symptom action
        statuses = self.state_to_symptom_status(state)
        symptom_idx = action

        if statuses[symptom_idx] != 0:
            # Already known - no change, small cost
            return state, self.R_ASK, False

        # Reveal symptom based on true disease
        symptom_value = self.DISEASE_PATTERNS[disease][symptom_idx]
        statuses[symptom_idx] = symptom_value + 1  # 1=absent, 2=present
        next_state = self.symptom_status_to_state(statuses)

        return next_state, self.R_ASK, False

    def generate_episode(self, Q: np.ndarray, epsilon: float, disease: int,
                         max_steps: int = 15) -> List[Tuple[int, int, float]]:
        """
        Generate one complete episode using epsilon-greedy policy.

        Returns:
            List of (state, action, reward) tuples
        """
        episode = []
        state = 0  # Start with no symptoms known

        for _ in range(max_steps):
            action = self.epsilon_greedy_action(state, Q, epsilon)
            next_state, reward, done = self.step(state, action, disease)

            episode.append((state, action, reward))

            if done:
                break
            state = next_state

        return episode

    def run(self, n_episodes: int = 50000, verbose: bool = True) -> Dict:
        """
        Run GLIE Monte Carlo Control.

        Args:
            n_episodes: Number of training episodes
            verbose: Print progress

        Returns:
            Dictionary with Q, policy, history, etc.
        """
        start = time.time()

        self.Q = np.zeros((self.n_states, self.n_actions))
        N = np.zeros((self.n_states, self.n_actions))  # Visit counts
        self.history = []

        if verbose:
            print("=" * 60)
            print("GLIE MONTE CARLO CONTROL - Medical Diagnosis")
            print("=" * 60)

        total_rewards = []

        for ep in range(n_episodes):
            # GLIE epsilon schedule
            epsilon = 1.0 / (1.0 + ep / self.epsilon_decay)

            # Random disease for this episode
            disease = np.random.randint(0, 8)

            # Generate episode
            episode = self.generate_episode(self.Q, epsilon, disease)

            # Calculate returns and update Q (first-visit MC)
            G = 0.0
            visited = set()
            total_ep_reward = sum(r for _, _, r in episode)
            total_rewards.append(total_ep_reward)

            # Process episode backwards
            for t in range(len(episode) - 1, -1, -1):
                s, a, r = episode[t]
                G = r + self.gamma * G

                if (s, a) not in visited:
                    visited.add((s, a))
                    N[s, a] += 1
                    # Incremental mean update
                    self.Q[s, a] += (G - self.Q[s, a]) / N[s, a]

            # Log progress
            if (ep + 1) % 5000 == 0:
                avg_reward = np.mean(total_rewards[-5000:])
                self.history.append({
                    'episode': ep + 1,
                    'epsilon': epsilon,
                    'avg_reward': avg_reward,
                    'Q_s0_max': np.max(self.Q[0]),
                })
                if verbose:
                    print(f"Episode {ep+1:6d}: ε={epsilon:.4f}, "
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
        """Simulate a diagnosis episode using learned policy."""
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
    mc = GLIEMonteCarlo()
    results = mc.run(n_episodes=50000)
    print(f"\nFinal Q(s=0, best) = {np.max(results['Q'][0]):.4f}")

    print("\nTesting all 8 diseases:")
    for d in range(8):
        symptoms = list(mc.DISEASE_PATTERNS[d])
        result = mc.simulate_episode(symptoms)
        status = "✓" if result['success'] else "✗"
        print(f"  {mc.DISEASE_NAMES[d]:12s}: {status} "
              f"→ {mc.DISEASE_NAMES[result['diagnosed']]}, "
              f"steps={result['steps']}")

"""
Medical Diagnosis Environment for Reinforcement Learning

A custom Gymnasium environment implementing a cost-aware medical diagnosis MDP.
The agent must diagnose between Flu and Pneumonia by asking about symptoms
and ordering tests, balancing information gathering with costs.

MDP Specification:
- States: 32 (5-bit knowledge vector, each bit = symptom known/unknown)
- Actions: 12 (5 symptom asks, 5 tests, 2 diagnoses)
- Transitions: Deterministic
- Discount Factor: γ = 0.9
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any, Optional


class DiagnosisEnv(gym.Env):
    """
    Medical Diagnosis Environment
    
    The agent acts as a doctor diagnosing patients with either Flu or Pneumonia.
    Starting from zero knowledge, the agent can:
    - Ask about symptoms (cheap: +0.2 reward)
    - Order tests (expensive: -0.3 reward)
    - Make a diagnosis (terminal: +10 correct, -5 wrong)
    
    Attributes:
        n_symptoms (int): Number of symptoms/tests (5)
        n_states (int): Total number of states (32 = 2^5)
        n_actions (int): Total number of actions (12)
        gamma (float): Discount factor (0.9)
    """
    
    metadata = {"render_modes": ["human", "ansi"]}
    
    # Disease symptom patterns: 1 = present, 0 = absent
    DISEASE_PATTERNS = {
        0: np.array([1, 0, 1, 0, 1]),  # Flu: Fever, Fatigue, Chest Pain
        1: np.array([0, 1, 0, 1, 1])   # Pneumonia: Cough, Shortness of Breath, Chest Pain
    }
    
    DISEASE_NAMES = {0: "Flu", 1: "Pneumonia"}
    
    SYMPTOM_NAMES = [
        "Fever",           # Bit 0
        "Cough",           # Bit 1
        "Fatigue",         # Bit 2
        "Shortness of Breath",  # Bit 3
        "Chest Pain"       # Bit 4
    ]
    
    ACTION_NAMES = [
        "Ask Fever", "Ask Cough", "Ask Fatigue", 
        "Ask Breath", "Ask Chest Pain",
        "Test Fever", "Test Cough", "Test Fatigue",
        "Test Breath", "Test Chest Pain",
        "Diagnose Flu", "Diagnose Pneumonia"
    ]
    
    # Reward structure
    REWARD_ASK_SYMPTOM = 0.2
    REWARD_ORDER_TEST = -0.3
    REWARD_CORRECT_DIAGNOSIS = 10.0
    REWARD_WRONG_DIAGNOSIS = -5.0
    REWARD_STEP_COST = -0.01
    REWARD_TIMEOUT = -2.0
    
    def __init__(self, render_mode: Optional[str] = None, max_steps: int = 20, gamma: float = 0.9):
        """
        Initialize the Diagnosis Environment.
        
        Args:
            render_mode: Rendering mode ('human' or 'ansi')
            max_steps: Maximum steps before timeout (default: 20)
            gamma: Discount factor (default: 0.9)
        """
        super().__init__()
        
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.gamma = gamma
        
        # Environment dimensions
        self.n_symptoms = 5
        self.n_states = 2 ** self.n_symptoms  # 32 states
        self.n_actions = 12  # 5 asks + 5 tests + 2 diagnoses
        
        # Gymnasium spaces
        self.observation_space = spaces.Discrete(self.n_states)
        self.action_space = spaces.Discrete(self.n_actions)
        
        # Episode state
        self.state = 0  # Current knowledge state (5-bit integer)
        self.true_disease = 0  # Hidden true disease (0=Flu, 1=Pneumonia)
        self.true_symptoms = None  # True symptom values
        self.revealed_symptoms = None  # What symptoms are revealed
        self.step_count = 0
        self.done = False
        self.episode_reward = 0.0
        
        # History for visualization
        self.history = []
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[int, Dict]:
        """
        Reset the environment for a new episode.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (can specify 'disease' to fix the disease)
        
        Returns:
            observation: Initial state (0 = no knowledge)
            info: Additional information about the episode
        """
        super().reset(seed=seed)
        
        # Reset episode state
        self.state = 0  # State 0: no symptoms known
        self.step_count = 0
        self.done = False
        self.episode_reward = 0.0
        self.history = []
        
        # Randomly select true disease (or use specified)
        if options and 'disease' in options:
            self.true_disease = options['disease']
        else:
            self.true_disease = self.np_random.choice([0, 1])
        
        # Set true symptom pattern based on disease
        self.true_symptoms = self.DISEASE_PATTERNS[self.true_disease].copy()
        self.revealed_symptoms = np.zeros(self.n_symptoms, dtype=np.int32)
        
        info = {
            "true_disease": self.DISEASE_NAMES[self.true_disease],
            "true_disease_id": self.true_disease,
            "step": 0
        }
        
        return self.state, info
    
    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict]:
        """
        Execute an action in the environment.
        
        Args:
            action: Action to take (0-11)
        
        Returns:
            observation: New state
            reward: Reward received
            terminated: Whether episode ended (diagnosis made)
            truncated: Whether episode was cut short (timeout)
            info: Additional information
        """
        if self.done:
            raise RuntimeError("Episode already done. Call reset() first.")
        
        self.step_count += 1
        reward = self.REWARD_STEP_COST  # Base step cost
        terminated = False
        truncated = False
        info = {"action": self.ACTION_NAMES[action], "action_id": action}
        
        if action < 5:
            # Ask symptom (actions 0-4)
            symptom_idx = action
            reward += self.REWARD_ASK_SYMPTOM
            
            # Reveal symptom (set bit in state)
            if not (self.state & (1 << symptom_idx)):
                self.revealed_symptoms[symptom_idx] = self.true_symptoms[symptom_idx]
                self.state = self.state | (1 << symptom_idx)
            
            info["symptom"] = self.SYMPTOM_NAMES[symptom_idx]
            info["value"] = int(self.true_symptoms[symptom_idx])
            
        elif action < 10:
            # Order test (actions 5-9)
            symptom_idx = action - 5
            reward += self.REWARD_ORDER_TEST
            
            # Reveal symptom (same as ask, but more expensive)
            if not (self.state & (1 << symptom_idx)):
                self.revealed_symptoms[symptom_idx] = self.true_symptoms[symptom_idx]
                self.state = self.state | (1 << symptom_idx)
            
            info["test"] = self.SYMPTOM_NAMES[symptom_idx]
            info["value"] = int(self.true_symptoms[symptom_idx])
            
        else:
            # Diagnose (actions 10-11)
            diagnosed_disease = action - 10  # 0=Flu, 1=Pneumonia
            
            if diagnosed_disease == self.true_disease:
                reward += self.REWARD_CORRECT_DIAGNOSIS
                info["correct"] = True
            else:
                reward += self.REWARD_WRONG_DIAGNOSIS
                info["correct"] = False
            
            info["diagnosed"] = self.DISEASE_NAMES[diagnosed_disease]
            info["actual"] = self.DISEASE_NAMES[self.true_disease]
            terminated = True
        
        # Check for timeout
        if not terminated and self.step_count >= self.max_steps:
            reward += self.REWARD_TIMEOUT
            truncated = True
        
        self.done = terminated or truncated
        self.episode_reward += reward
        
        # Record history
        self.history.append({
            "step": self.step_count,
            "action": action,
            "action_name": self.ACTION_NAMES[action],
            "state": self.state,
            "reward": reward,
            "cumulative_reward": self.episode_reward,
            "terminated": terminated,
            "truncated": truncated
        })
        
        info["step"] = self.step_count
        info["state"] = self.state
        info["knowledge"] = self._state_to_binary_str(self.state)
        
        return self.state, reward, terminated, truncated, info
    
    def _state_to_binary_str(self, state: int) -> str:
        """Convert state integer to binary string representation."""
        return format(state, f"0{self.n_symptoms}b")
    
    def get_state_description(self, state: int) -> Dict:
        """
        Get detailed description of a state.
        
        Args:
            state: State integer (0-31)
        
        Returns:
            Dictionary with state details
        """
        binary = self._state_to_binary_str(state)
        known_symptoms = []
        unknown_symptoms = []
        
        for i in range(self.n_symptoms):
            if state & (1 << i):
                known_symptoms.append(self.SYMPTOM_NAMES[i])
            else:
                unknown_symptoms.append(self.SYMPTOM_NAMES[i])
        
        return {
            "state": state,
            "binary": binary,
            "known_symptoms": known_symptoms,
            "unknown_symptoms": unknown_symptoms,
            "n_known": len(known_symptoms)
        }
    
    def render(self) -> Optional[str]:
        """Render the current state of the environment."""
        if self.render_mode == "ansi":
            output = []
            output.append(f"\n{'='*50}")
            output.append(f"Step: {self.step_count} | State: {self.state} ({self._state_to_binary_str(self.state)})")
            output.append(f"True Disease: {self.DISEASE_NAMES[self.true_disease]}")
            output.append("-" * 50)
            
            for i, name in enumerate(self.SYMPTOM_NAMES):
                known = "✓" if (self.state & (1 << i)) else "?"
                value = self.revealed_symptoms[i] if (self.state & (1 << i)) else "?"
                present = "Present" if value == 1 else ("Absent" if value == 0 else "Unknown")
                output.append(f"  [{known}] {name}: {present}")
            
            output.append(f"\nEpisode Reward: {self.episode_reward:.2f}")
            output.append(f"{'='*50}\n")
            
            return "\n".join(output)
        
        return None
    
    def get_valid_actions(self, state: int) -> list:
        """
        Get list of valid/useful actions for a state.
        
        Args:
            state: Current state
        
        Returns:
            List of action indices that are useful (not already revealed symptoms)
        """
        valid = []
        
        for i in range(5):  # Symptom asks
            if not (state & (1 << i)):
                valid.append(i)
        
        for i in range(5, 10):  # Tests
            symptom_idx = i - 5
            if not (state & (1 << symptom_idx)):
                valid.append(i)
        
        # Diagnose actions are always valid
        valid.extend([10, 11])
        
        return valid


# Register environment with Gymnasium
gym.register(
    id="DiagnosisEnv-v1",
    entry_point="src.diagnosis_env:DiagnosisEnv"
)


if __name__ == "__main__":
    # Quick test of the environment
    env = DiagnosisEnv(render_mode="ansi")
    
    print("Testing DiagnosisEnv...")
    print(f"State space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Number of states: {env.n_states}")
    print(f"Number of actions: {env.n_actions}")
    
    # Run a random episode
    state, info = env.reset(seed=42)
    print(f"\nStarting episode with {info['true_disease']}")
    print(env.render())
    
    done = False
    while not done:
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {info['action']}, Reward: {reward:.2f}")
        print(env.render())
        done = terminated or truncated
    
    print(f"Episode finished! Total reward: {env.episode_reward:.2f}")

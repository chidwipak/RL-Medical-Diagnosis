# RL Medical Diagnosis System

**Live Demo:** [ai-rl-doctor.streamlit.app](https://ai-rl-doctor.streamlit.app)

## About This Project

I built this Reinforcement Learning project to demonstrate how an AI agent can learn to diagnose diseases by intelligently asking about symptoms. The project is a comprehensive analysis of **10 different Reinforcement Learning algorithms** across 4 major categories.

The final project concludes that **Policy Iteration** is the optimal algorithm for this specific clinical decision-making MDP.

### The Algorithm Progression
1. **Assignment 1:** Exact DP methods (Policy Iteration, Value Iteration) leveraging the known transition model.
2. **Assignment 2:** Model-Free methods (GLIE Monte Carlo, SARSA, SARSA(λ)) learning purely from interaction experience using tabular representations.
3. **Assignment 3:** Scalable methods including Function Approximation (MC-FA, SARSA-FA, LSPI) and Policy Gradients (REINFORCE, Actor-Critic).
4. **Final Project:** A comparative analysis of all 10 algorithms, justifying the best clinical solver.

## The Environment (MDP)

I created a custom medical diagnosis Markov Decision Process (MDP):
1. **State Space (244 states)**: A base-3 knowledge representation tracking 5 symptoms (Fever, Cough, Fatigue, Breath Difficulty, Headache). Each state represents whether a symptom is *Unknown*, *Absent*, or *Present*.
2. **Action Space (13 actions)**: 5 'Ask' actions (gathering information) and 8 'Diagnose' actions (making a final prediction).
3. **Reward Structure**:
   - Ask symptom: -0.1 (encourages efficiency / fewer questions)
   - Correct diagnosis: +10.0
   - Wrong diagnosis: -5.0

## The 10 Algorithms Investigated

All 10 algorithms successfully converged and achieved **100% accuracy**. 

| Category | Algorithms Implemented |
|---|---|
| **Dynamic Programming** | Policy Iteration, Value Iteration |
| **Model-Free Control** | GLIE Monte Carlo, SARSA, SARSA(λ) |
| **Function Approximation** | MC with Linear FA, SARSA with Linear FA, LSPI |
| **Policy Gradients** | REINFORCE, Actor-Critic |

## Why Policy Iteration Wins

After implementing all 10 algorithms, I selected **Policy Iteration** as the best algorithm for this problem because:
- **Exact Optimality**: It mathematically guarantees finding the true optimal policy using Bellman equations.
- **Extreme Efficiency**: It achieves the theoretical minimum average diagnosis steps (4.0 steps).
- **Incredible Speed**: Training converges in `< 0.5s`, compared to minutes for FA/PG methods.
- **Zero Hyperparameters**: Unlike model-free methods, there is no $\epsilon$, $\lambda$, or $\alpha$ to tune.
- **Deterministic**: In medical environments, deterministic and repeatable behavior is mandatory.

*Note: While DP works perfectly here because our state space is small (244 states) and we know the exact transition model, the Function Approximation and Policy Gradient algorithms remain crucial for scaling this approach to larger real-world medical datasets.*

## Interactive Dashboard

I built an interactive Streamlit dashboard featuring:
- **Live Diagnosis Mode**: Select symptoms and watch the AI (Policy Iteration) dynamically diagnose the correct disease step-by-step.
- **Comprehensive Comparison**: Side-by-side performance metrics across all 10 algorithms.
- **Algorithm Deep Dives**: Specific tabs exploring the training and mechanics of DP, Model-Free, FA, and PG methods.

## How to Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the interactive dashboard
streamlit run app.py
```

## Author

**K. Chidwipak**  
Roll No: S20230010131  
Course: Reinforcement Learning

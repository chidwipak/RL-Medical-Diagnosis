# RL Medical Diagnosis System

**Live Demo:** [ai-rl-doctor.streamlit.app](https://ai-rl-doctor.streamlit.app)

## About This Project

I built this Reinforcement Learning project to demonstrate how an AI agent can learn to diagnose diseases by intelligently asking about symptoms. The project is divided into two parts:
1. **Assignment 1:** Dynamic Programming (Policy Iteration & Value Iteration) using known transition probabilities.
2. **Assignment 2:** Model-Free Reinforcement Learning (GLIE Monte Carlo, SARSA, SARSA(λ)) learning directly from experience without transition probabilities.

## What I Implemented

I created a medical diagnosis system where an AI doctor:
1. Starts with no knowledge about the patient
2. Asks about 5 symptoms: Fever, Cough, Fatigue, Breath Difficulty, Headache
3. Learns which questions to ask to efficiently identify 8 different diseases
4. Makes a diagnosis when confident

### The 8 Diseases Modeled

| Disease | Symptoms Present |
|---------|-----------------|
| Flu | Fever, Cough, Fatigue |
| Strep | Fever, Cough, Breath |
| Pneumonia | Fever, Fatigue, Headache |
| Bronchitis | Fever, Breath, Headache |
| Cold | Cough, Fatigue, Headache |
| Allergy | Cough, Breath |
| Asthma | Fatigue, Breath |
| Migraine | Headache |

## MDP Design

### State Space (243 states)
I used a base-3 representation where each of the 5 symptoms can be:
- **Unknown (0)**: Not yet asked
- **Absent (1)**: Patient doesn't have it
- **Present (2)**: Patient has it

Total states = 3^5 = 243 knowledge states + 1 terminal = 244 states.

### Action Space (13 actions)
- **5 Ask actions**: Ask about each symptom
- **8 Diagnose actions**: Diagnose each disease

### Reward Structure
- **Ask symptom**: -0.1 (small cost for asking)
- **Correct diagnosis**: +10.0
- **Wrong diagnosis**: -5.0

## The Algorithms Investigated

### Part 1: Dynamic Programming (Model-Based)
Requires full knowledge of transition probabilities $P(s'|s,a)$.
- **Policy Iteration**: Converges in 4-5 iterations
- **Value Iteration**: Converges in 15-20 iterations

### Part 2: Model-Free RL (Learning from Experience)
Learns without transition probabilities using $\epsilon$-greedy exploration.
- **GLIE Monte Carlo Control**: Learns from complete episodes. Unbiased but high variance.
- **SARSA (One-Step TD)**: Learns step-by-step. Bootstraps value estimates.
- **SARSA(λ) with Eligibility Traces**: Bridges MC and One-Step TD using eligibility traces to propagate rewards backward.

*All 3 model-free algorithms successfully converged to policies achieving 100% accuracy across all 8 diseases after 50,000+ episodes of training.*

## Visualization Dashboard

I built an interactive Streamlit dashboard with two tabs:
- **Assignment 1:** Visualize the optimal policy via DP.
- **Assignment 2:** Compare Q-value convergence, accuracy, and average reward for the 3 Model-Free algorithms.

Features include:
- **32-State Grid**: Shows all logically possible knowledge paths
- **Animated Paths**: Watch the AI traverse through states to reach a diagnosis
- **Training Analytics**: Real-time convergence plot visualizations

## How to Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run Assignment 1 training (optional)
python train.py

# Run Assignment 2 model-free training and generating plots (optional)
python train_model_free.py

# Launch the dashboard locally
streamlit run app.py
```

## Project Structure

```
RL_Project/
├── app.py                    # Streamlit dashboard
├── train.py                  # DP Training script
├── train_model_free.py       # Model-Free Training script
├── presentation_assignment2.tex # LaTeX presentation for Assigment 2
├── requirements.txt          # Dependencies
├── src/
│   ├── policy_iteration.py   
│   ├── value_iteration.py    
│   ├── monte_carlo.py        # GLIE MC implementation
│   ├── sarsa.py              # SARSA implementation
│   ├── sarsa_lambda.py       # SARSA(λ) implementation
│   └── __init__.py
├── results/                  # Saved plots, Q-tables, and policies
└── README.md
```

## Author

**K. Chidwipak**  
Roll No: S20230010131  
Course: Reinforcement Learning

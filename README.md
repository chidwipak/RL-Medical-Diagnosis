# RL Medical Diagnosis System

## About This Project

I built this Reinforcement Learning project to demonstrate how an AI agent can learn to diagnose diseases by intelligently asking about symptoms. The agent uses **Policy Iteration** and **Value Iteration** algorithms to find the optimal strategy for diagnosis.

## What I Implemented

I created a medical diagnosis system where an AI doctor:
1. Starts with no knowledge about the patient
2. Asks about 5 symptoms: Fever, Cough, Fatigue, Breath Difficulty, Headache
3. Learns which questions to ask to efficiently identify 8 different diseases
4. Makes a diagnosis when confident

### The 8 Diseases I Modeled

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

## How I Designed the MDP

### State Space (243 states)
I used a base-3 representation where each of the 5 symptoms can be:
- **Unknown (0)**: Not yet asked
- **Absent (1)**: Patient doesn't have it
- **Present (2)**: Patient has it

So total states = 3^5 = 243 knowledge states + 1 terminal = 244 states.

### Action Space (13 actions)
- **5 Ask actions**: Ask about each symptom
- **8 Diagnose actions**: Diagnose each disease

### Reward Structure
| Action | Reward |
|--------|--------|
| Ask symptom | -0.1 (small cost for asking) |
| Correct diagnosis | +10.0 |
| Wrong diagnosis | -5.0 |

### Discount Factor
I used γ = 0.9 to balance immediate vs future rewards.

## The Algorithms I Used

### Policy Iteration
1. Start with a random policy
2. **Policy Evaluation**: Calculate V(s) for all states under current policy
3. **Policy Improvement**: Update policy to be greedy with respect to V
4. Repeat until policy converges

### Value Iteration
1. Initialize V(s) = 0 for all states
2. Update V(s) = max_a [R(s,a) + γ * Σ P(s'|s,a) * V(s')]
3. Repeat until values converge
4. Extract optimal policy from final values

## Key Results

| Algorithm | Iterations to Converge | V(initial state) |
|-----------|----------------------|------------------|
| Policy Iteration | 4-5 | ~7.0 |
| Value Iteration | 15-20 | ~7.0 |

## Visualization Dashboard

I built an interactive Streamlit dashboard with:
- **32-State Grid**: Shows all possible knowledge states
- **8 Disease Endpoints**: Visual representation of diagnosis targets
- **Animated Paths**: Watch the AI traverse through states
- **Two Modes**:
  - Exploration: Random symptom order to cover all 32 states
  - Optimal: Shows the efficient learned policy

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run training (optional)
python train.py

# Launch the dashboard
streamlit run app.py
```

## Project Structure

```
RL_Project/
├── src/
│   ├── policy_iteration.py   # Policy Iteration implementation
│   ├── value_iteration.py    # Value Iteration implementation
│   └── __init__.py
├── app.py                    # Streamlit dashboard
├── train.py                  # Training script
├── requirements.txt          # Dependencies
└── README.md
```

## What I Learned

Through this project, I understood:
- How to model real-world problems as MDPs
- The importance of state representation (243-state vs 32-state)
- How information gathering has value in decision making
- The trade-off between asking more questions vs immediate diagnosis
- Practical implementation of dynamic programming algorithms

## Author

**K. Chidwipak**  
Roll No: S20230010131  
Course: Reinforcement Learning

# Detailed Project Explanation

## The Big Picture

This project is about teaching an AI to be a doctor. Imagine you go to a clinic and the doctor asks you questions like "Do you have fever?", "Do you have cough?" etc. After asking some questions, the doctor tells you what disease you might have.

I built an AI that learns to do exactly this - but smarter. Instead of asking all possible questions (which wastes time), the AI learns which questions are most useful to ask first.

---

## Understanding the Problem as an MDP

### What is an MDP?

MDP = Markov Decision Process. It has 4 parts:

1. **States (S)**: Where the agent currently is
2. **Actions (A)**: What the agent can do
3. **Transitions (P)**: What happens after an action
4. **Rewards (R)**: How good or bad an action was

### My State Design

**Question**: What should a "state" represent?

**Answer**: It should represent what the AI currently knows about the patient.

I have 5 symptoms: Fever, Cough, Fatigue, Breath Difficulty, Headache

Each symptom can be in one of 3 conditions:
- **Unknown (0)**: AI hasn't asked about this yet
- **Absent (1)**: AI asked, patient said "No"
- **Present (2)**: AI asked, patient said "Yes"

So total states = 3 × 3 × 3 × 3 × 3 = 3^5 = 243 states

Plus 1 terminal state (after diagnosis) = **244 states total**

**State Encoding**: I use base-3 numbers.
- State 0 = [0,0,0,0,0] = nothing known
- State 1 = [1,0,0,0,0] = only Fever is absent
- State 2 = [2,0,0,0,0] = only Fever is present
- etc.

### My Action Design

The AI can do 13 things:

**Ask Actions (0-4):**
- Action 0: Ask about Fever
- Action 1: Ask about Cough
- Action 2: Ask about Fatigue
- Action 3: Ask about Breath
- Action 4: Ask about Headache

**Diagnose Actions (5-12):**
- Action 5: Diagnose Flu
- Action 6: Diagnose Strep
- Action 7: Diagnose Pneumonia
- Action 8: Diagnose Bronchitis
- Action 9: Diagnose Cold
- Action 10: Diagnose Allergy
- Action 11: Diagnose Asthma
- Action 12: Diagnose Migraine

### My Transition Design

When AI asks a symptom:
1. Get the patient's true answer (from their symptom pattern)
2. Update the state to mark that symptom as known

Example:
- Current state: [0,0,0,0,0] (nothing known)
- AI asks about Fever (action 0)
- Patient has Flu (pattern: [1,1,1,0,0]), so Fever = 1 (present)
- New state: [2,0,0,0,0] (Fever is present)

When AI diagnoses:
- Go to terminal state (state 243)
- Get reward based on correctness

### My Reward Design

| Action Type | Reward |
|------------|--------|
| Ask any symptom | -0.1 |
| Diagnose correctly | +10.0 |
| Diagnose incorrectly | -5.0 |

**Why these values?**

- **Ask = -0.1**: Small negative so AI doesn't ask forever, but small enough that asking is usually worth it
- **Correct = +10**: Big positive to encourage correct diagnosis
- **Wrong = -5**: Medium negative to discourage guessing

The math: If AI guesses randomly (1/8 chance of correct):
- Expected reward = (1/8 × 10) + (7/8 × -5) = 1.25 - 4.375 = -3.125

If AI asks 3 questions then diagnoses correctly:
- Expected reward = (-0.1 × 3) + 10 = -0.3 + 10 = 9.7

So asking before diagnosing is much better!

---

## Understanding the Algorithms

### Policy Iteration

**Core Idea**: Keep improving the policy until it's optimal.

**Step 1 - Policy Evaluation**:
Given a policy π, calculate V^π(s) for all states.

V^π(s) = Expected total reward starting from s, following policy π

Formula (Bellman equation for policy π):
```
V^π(s) = R(s, π(s)) + γ × Σ P(s'|s, π(s)) × V^π(s')
```

This is solved iteratively until values stabilize.

**Step 2 - Policy Improvement**:
Create a new policy that's greedy with respect to V^π.

For each state s:
```
π_new(s) = argmax_a [ R(s,a) + γ × Σ P(s'|s,a) × V^π(s') ]
```

This means: pick the action with highest expected value.

**Repeat** until policy doesn't change.

**Why it converges?**
Each improvement step makes the policy at least as good as before (monotonic improvement). Since there are finite policies, it must converge.

### Value Iteration

**Core Idea**: Find optimal V* directly, then extract policy.

**Update Rule** (Bellman optimality equation):
```
V(s) = max_a [ R(s,a) + γ × Σ P(s'|s,a) × V(s') ]
```

This is basically Policy Improvement + Evaluation combined into one step.

**Repeat** until max change (delta) < threshold.

**Extract Policy**:
After V* is found:
```
π*(s) = argmax_a [ R(s,a) + γ × Σ P(s'|s,a) × V*(s') ]
```

### Key Difference

| Policy Iteration | Value Iteration |
|-----------------|----------------|
| Full policy evaluation each iteration | One Bellman update each iteration |
| Fewer outer iterations | More outer iterations |
| Each iteration is expensive | Each iteration is cheap |
| Guarantees optimal policy | Guarantees optimal values |

---

## Understanding Compatible Diseases

This is a crucial concept in my project.

**Situation**: AI doesn't know the true disease. It only knows the symptoms it has revealed.

**Question**: Given current knowledge, which diseases are still possible?

**Example**:
- AI knows: Fever = Present, Cough = Unknown, ...
- Compatible diseases: Any disease that has Fever = 1

**Why this matters**:
When calculating Q(s,a), we average over all compatible diseases:
```
Q(s,a) = Σ P(disease | knowledge) × [ R(s,a,disease) + γ × V(s') ]
```

I assume uniform probability over compatible diseases.

---

## Understanding the 32-State Visualization

The 243-state space is too complex to visualize. So I use a simpler "knowledge" representation.

**Knowledge State**: Which symptoms are known (regardless of their value)

This is a 5-bit number:
- Bit 0: Fever known? (1 or 0)
- Bit 1: Cough known? (1 or 0)
- Bit 2: Fatigue known? (1 or 0)
- Bit 3: Breath known? (1 or 0)
- Bit 4: Headache known? (1 or 0)

Total: 2^5 = 32 knowledge states

**Examples**:
- State 0 (00000): Nothing known
- State 1 (00001): Only Fever known
- State 3 (00011): Fever and Cough known
- State 31 (11111): All symptoms known

**In the Grid**:
- 8 rows × 4 columns = 32 cells
- Each cell shows: state number, known symptoms
- Path shows how AI moves through states

---

## How the Dashboard Works

1. **User selects symptoms** (checkboxes)
2. **System finds matching disease** (best match)
3. **AI runs diagnosis**:
   - Starts at state 0
   - Follows policy to pick action
   - If Ask: reveal symptom, move to new state
   - If Diagnose: check if correct
4. **Visualization shows path** through 32-state grid

**Two Modes**:
- **Exploration**: Random symptom order (visits many states)
- **Optimal**: Follows learned policy (efficient path)

---

## Key Insights

1. **Value of Information**: Asking questions has value because it leads to correct diagnosis. The -0.1 cost is much less than the difference between correct (+10) and wrong (-5) diagnosis.

2. **Optimal Strategy**: The optimal policy asks symptoms that best split the remaining possible diseases. Usually fever is best first (splits 4+4).

3. **Greedy Works**: In this problem, greedy policy improvement works because we have a proper MDP structure with discounting.

4. **Both algorithms find same answer**: Policy Iteration and Value Iteration converge to the same optimal policy, just via different paths.

---

## Summary

- **State**: What AI knows (3^5 = 243 states)
- **Action**: Ask or Diagnose (13 actions)
- **Reward**: -0.1 ask, +10 correct, -5 wrong
- **Gamma**: 0.9 (care about future)
- **Policy Iteration**: Evaluate → Improve → Repeat
- **Value Iteration**: Bellman update → Repeat → Extract policy
- **Result**: AI learns efficient diagnostic strategy

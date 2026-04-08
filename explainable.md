# 🩺 RL Medical Diagnosis — Complete Project Explanation

> **Read this document carefully before your viva/presentation.**
> Every concept, every algorithm, every design choice is explained here in simple English.
> After reading this, you will be able to answer ANY question about your project confidently.

---

## Table of Contents

1. [The Problem Statement — What Are We Actually Doing?](#1-the-problem-statement)
2. [Real-World Significance — Why This Matters](#2-real-world-significance)
3. [What is Reinforcement Learning? (The Basics)](#3-what-is-reinforcement-learning)
4. [MDP Formulation — The Heart of the Project](#4-mdp-formulation)
5. [Transition Probability — The Most Important Calculation](#5-transition-probability)
6. [The 10 Algorithms — Explained One by One](#6-the-10-algorithms)
7. [Why Policy Iteration Wins Over Everything Else](#7-why-policy-iteration-wins)
8. [Results Discussion](#8-results-discussion)
9. [Likely Viva Questions and Answers](#9-likely-viva-questions-and-answers)

---

## 1. The Problem Statement

### In One Sentence
> We built an AI doctor that learns to diagnose diseases by asking the right questions in the right order.

### The Full Picture

Imagine you walk into a doctor's office. The doctor doesn't know what's wrong with you. They don't run every test in the world — that would be expensive and time-consuming. Instead, a good doctor asks *smart questions*:

- "Do you have a fever?" → If yes, many diseases eliminated. If no, different set eliminated.
- Based on that answer, they ask the *next best question*.
- After 3-4 good questions, they can confidently tell you what you have.

**Our project does exactly this, but with AI.** We trained an AI agent to learn this skill using Reinforcement Learning.

### The Setup

- **8 diseases** the AI can diagnose: Flu, Strep, Pneumonia, Bronchitis, Cold, Allergy, Asthma, Migraine
- **5 symptoms** the AI can ask about: Fever, Cough, Fatigue, Breathing Difficulty, Headache
- Each disease has a **unique combination** of symptoms (see table below)

| Disease | Fever | Cough | Fatigue | Breath | Headache |
|---------|-------|-------|---------|--------|----------|
| Flu | ✓ | ✓ | ✓ | | |
| Strep | ✓ | ✓ | | ✓ | |
| Pneumonia | ✓ | | ✓ | | ✓ |
| Bronchitis | ✓ | | | ✓ | ✓ |
| Cold | | ✓ | ✓ | | ✓ |
| Allergy | | ✓ | | ✓ | |
| Asthma | | | ✓ | ✓ | |
| Migraine | | | | | ✓ |

### The Challenge

Look at that table carefully. Flu and Strep both have Fever AND Cough. If you only know "Fever=Yes, Cough=Yes", you still can't tell if it's Flu or Strep! You need to ask at least one more question (Fatigue or Breath) to distinguish them.

**The AI must learn:**
1. Which symptom to ask FIRST (the one that eliminates the most diseases)
2. Which symptom to ask NEXT based on what it learned
3. WHEN to stop asking and commit to a diagnosis
4. All of this while minimizing the number of questions (because each question has a small cost)

### Why Not Just Ask All 5 Questions Every Time?

Because in the real world, every test costs money and time. If the AI can diagnose correctly after only 3-4 questions instead of 5, that saves the patient money, time, and unnecessary tests. The AI needs to balance **accuracy vs. efficiency**.

---

## 2. Real-World Significance

### Where Would This Actually Be Used?

1. **Hospital Triage**: When a patient walks into an emergency room, a nurse needs to quickly figure out how serious their condition is. Our AI could help prioritize — "Ask about fever first, then breathing difficulty" — and guide the triage process.

2. **Rural Healthcare**: In remote areas with limited access to specialists, a community health worker with this AI tool on a phone/tablet could do preliminary diagnosis by asking the right questions.

3. **Reducing Unnecessary Tests**: In real medicine, blood tests, X-rays, and MRIs are expensive. If the AI determines that just asking 3 questions is enough to diagnose, it saves the patient from 2 unnecessary tests.

4. **Clinical Decision Support**: Doctors are human — they can forget, get tired, or have biases. An AI that always follows the optimal question order provides a consistent second opinion.

5. **Beyond Healthcare**: The same idea applies to IT tech support ("Is your computer turning on?" → "Is the screen showing anything?"), customer service chatbots, and quality control in factories.

### Why Reinforcement Learning and Not Simple Rules?

You might ask: *"Why not just hard-code a decision tree? If fever, then ask cough. If cough, then..."*

Three reasons:
1. **Adaptability**: If you add new diseases or new symptoms, you'd have to rewrite all the rules. RL just retrains automatically.
2. **Optimality**: RL mathematically guarantees finding the BEST question order. A human-designed decision tree might miss a more efficient ordering.
3. **Balancing Trade-offs**: RL naturally handles the cost-accuracy trade-off through its reward structure. Hard-coding this is very difficult.

---

## 3. What is Reinforcement Learning?

### The Core Idea

Think of training a dog:
- The dog (agent) tries doing something
- If it does something good, you give it a treat (positive reward)
- If it does something bad, you say "No!" (negative reward)
- Over time, the dog learns what actions lead to treats

**RL is the same thing, but for computers.**

### The Key Components

| Component | Dog Training | Our Medical AI |
|-----------|-------------|----------------|
| **Agent** | The dog | The AI doctor |
| **Environment** | The room/park | The patient (with their hidden disease) |
| **State** | What the dog sees/knows | What symptoms the AI has asked about so far |
| **Action** | Sit, roll over, bark | Ask about Fever, Diagnose Flu, etc. |
| **Reward** | Treat or "No!" | +10 for correct diagnosis, -5 for wrong, -0.1 per question |
| **Policy** | The dog's learned behavior | The AI's strategy: "In this situation, do this action" |

### The Goal

Find the **optimal policy** — the strategy that maximizes the total reward over time. In our case, that means: ask the fewest questions possible while always diagnosing correctly.

---

## 4. MDP Formulation

MDP stands for **Markov Decision Process**. It's the mathematical framework that describes our problem precisely. Let's go through each component.

### 4.1 The Agent

The **agent** is the AI doctor. It makes decisions (which symptom to ask about, when to diagnose).

### 4.2 The Environment

The **environment** is the patient. The patient has a hidden disease, and when the AI asks "Do you have fever?", the environment (patient) answers honestly.

### 4.3 States — What the AI Knows

A **state** represents what the AI currently knows about the patient.

Since there are 5 symptoms, and each symptom can be in one of 3 conditions:
- **0 = Unknown** — haven't asked yet
- **1 = Absent** — asked, and the patient said "No"
- **2 = Present** — asked, and the patient said "Yes"

The total number of possible knowledge combinations = 3 × 3 × 3 × 3 × 3 = **3⁵ = 243 states**

Plus 1 **terminal state** (state 243) = the episode is over (diagnosis has been made).

**Total: 244 states.**

### How States Are Encoded (Base-3)

We use a clever encoding. Each state is a single number calculated from the 5 symptom statuses:

```
State = status[0] × 3⁰ + status[1] × 3¹ + status[2] × 3² + status[3] × 3³ + status[4] × 3⁴
```

Where status[0] = Fever status, status[1] = Cough status, etc.

**Examples:**

| State | Fever | Cough | Fatigue | Breath | Headache | Meaning |
|-------|-------|-------|---------|--------|----------|---------|
| 0 | 0 | 0 | 0 | 0 | 0 | Nothing known yet (start) |
| 2 | 2 | 0 | 0 | 0 | 0 | Fever = Present, rest unknown |
| 1 | 1 | 0 | 0 | 0 | 0 | Fever = Absent, rest unknown |
| 8 | 2 | 2 | 0 | 0 | 0 | Fever = Present, Cough = Present |
| 243 | — | — | — | — | — | Terminal (episode done) |

For State 8: Fever=2, Cough=2, rest=0 → 2×1 + 2×3 = 2+6 = **8** ✓

### 4.4 Actions — What the AI Can Do

The AI has **13 possible actions**:

| Action # | Type | What It Does |
|----------|------|-------------|
| 0 | Ask | Ask about Fever |
| 1 | Ask | Ask about Cough |
| 2 | Ask | Ask about Fatigue |
| 3 | Ask | Ask about Breath |
| 4 | Ask | Ask about Headache |
| 5 | Diagnose | Diagnose Flu |
| 6 | Diagnose | Diagnose Strep |
| 7 | Diagnose | Diagnose Pneumonia |
| 8 | Diagnose | Diagnose Bronchitis |
| 9 | Diagnose | Diagnose Cold |
| 10 | Diagnose | Diagnose Allergy |
| 11 | Diagnose | Diagnose Asthma |
| 12 | Diagnose | Diagnose Migraine |

**The key design choice**: Having separate "ask" and "diagnose" actions forces the AI to learn TWO things:
1. WHAT to ask next
2. WHEN to stop asking and make a diagnosis

This is the core trade-off: every question costs -0.1 reward, but diagnosing wrong costs -5.0. So the AI must learn to ask enough questions to be confident, but not more than necessary.

### 4.5 Rewards — The AI's Report Card

| Event | Reward | Why |
|-------|--------|-----|
| Ask any symptom | **-0.1** | Small penalty to discourage asking too many questions |
| Correct diagnosis | **+10.0** | Big reward for getting it right |
| Wrong diagnosis | **-5.0** | Big punishment for getting it wrong |

**The discount factor γ = 0.9** means: future rewards are worth 90% of immediate rewards. This makes the AI care about both the immediate cost of asking and the future benefit of a correct diagnosis.

**Example**: If the AI asks 4 questions (cost = 4 × -0.1 = -0.4) and then diagnoses correctly (+10.0), the total undiscounted return would be about -0.4 + 10.0 = **+9.6**. If it asks all 5 questions and diagnoses correctly: -0.5 + 10.0 = **+9.5** — slightly less. So fewer questions = better total reward, as long as accuracy is maintained.

### 4.6 Policy — The AI's Strategy

A **policy** π(s) tells the AI: "When you are in state s, take action a."

For example, a good policy might say:
- State 0 (know nothing) → Action 0 (Ask Fever)
- State 2 (Fever=Present) → Action 1 (Ask Cough)
- State 8 (Fever=Present, Cough=Present) → Action 2 (Ask Fatigue)
- State 17 (Fever=Present, Cough=Present, Fatigue=Absent) → Action 6 (Diagnose Strep!)

The goal of all our algorithms is to find the **optimal policy** — the one that gives the maximum total reward.

---

## 5. Transition Probability — The Most Important Calculation

> **This is what most people find confusing. Read this section very carefully.**

### What Is a Transition Probability?

P(s'|s, a) means: "If I am in state s and I take action a, what is the probability that I end up in state s'?"

### Why Do We Need It?

Dynamic Programming methods (Policy Iteration and Value Iteration) need to plan ahead. They need to think: *"If I ask about Fever right now, what might happen? What state will I be in next?"*

But here's the catch: **the AI doesn't know which disease the patient has!** So when it asks "Do you have Fever?", it doesn't know if the answer will be Yes or No. It depends on the hidden disease.

### How We Calculate It — Step by Step

The key idea: **we average over all diseases that are still possible.**

#### CASE 1: Starting State (Know Nothing)

**State = 0** (all symptoms unknown), **Action = "Ask Fever" (a=0)**

Step 1: Which diseases are compatible? ALL 8 (since we know nothing).

Step 2: What happens if the patient has each disease?

| Disease | Has Fever? | Next State |
|---------|-----------|------------|
| Flu | Yes (1) | Status becomes [2,0,0,0,0] → state **2** |
| Strep | Yes (1) | Status becomes [2,0,0,0,0] → state **2** |
| Pneumonia | Yes (1) | Status becomes [2,0,0,0,0] → state **2** |
| Bronchitis | Yes (1) | Status becomes [2,0,0,0,0] → state **2** |
| Cold | No (0) | Status becomes [1,0,0,0,0] → state **1** |
| Allergy | No (0) | Status becomes [1,0,0,0,0] → state **1** |
| Asthma | No (0) | Status becomes [1,0,0,0,0] → state **1** |
| Migraine | No (0) | Status becomes [1,0,0,0,0] → state **1** |

Step 3: Count the probabilities (each disease equally likely = 1/8):

- P(state 2 | state 0, Ask Fever) = 4/8 = **0.5** (Flu, Strep, Pneumonia, Bronchitis have Fever)
- P(state 1 | state 0, Ask Fever) = 4/8 = **0.5** (Cold, Allergy, Asthma, Migraine don't)

**In plain English**: When you start and ask about Fever, there's a 50-50 chance the answer is Yes vs No.

#### CASE 2: After Knowing Fever = Present

**State = 2** (Fever=Present, rest unknown), **Action = "Ask Cough" (a=1)**

Step 1: Compatible diseases = only those WITH Fever = {Flu, Strep, Pneumonia, Bronchitis} (4 diseases).

Step 2: What happens for each?

| Disease | Has Cough? | Next State |
|---------|-----------|------------|
| Flu | Yes (1) | Status [2,2,0,0,0] → state **8** |
| Strep | Yes (1) | Status [2,2,0,0,0] → state **8** |
| Pneumonia | No (0) | Status [2,1,0,0,0] → state **5** |
| Bronchitis | No (0) | Status [2,1,0,0,0] → state **5** |

Step 3: Each is equally likely (1/4 each):

- P(state 8 | state 2, Ask Cough) = 2/4 = **0.5**
- P(state 5 | state 2, Ask Cough) = 2/4 = **0.5**

#### CASE 3: Diagnose Action

**Any state, Action = "Diagnose Flu" (a=5)**

- The transition is always to **state 243** (terminal) with probability **1.0**
- The REWARD depends on whether the diagnosis is correct:
  - If the patient actually has Flu → reward = +10.0
  - If the patient has something else → reward = -5.0

#### CASE 4: After Knowing Fever=Yes AND Cough=Yes

**State = 8** (Fever=Present, Cough=Present), **Action = "Ask Fatigue" (a=2)**

Step 1: Compatible = only diseases with Fever=Yes AND Cough=Yes = {Flu, Strep} (2 diseases)

Step 2:
| Disease | Has Fatigue? | Next State |
|---------|-------------|------------|
| Flu | Yes (1) | [2,2,2,0,0] → state **26** |
| Strep | No (0) | [2,2,1,0,0] → state **17** |

Step 3:
- P(state 26 | state 8, Ask Fatigue) = 1/2 = **0.5**
- P(state 17 | state 8, Ask Fatigue) = 1/2 = **0.5**

Now if we are in state 26 (Fever=Yes, Cough=Yes, Fatigue=Yes), only Flu matches → **Diagnose Flu with confidence!**
If we are in state 17 (Fever=Yes, Cough=Yes, Fatigue=No), only Strep matches → **Diagnose Strep with confidence!**

#### CASE 5: Deep Decision — Distinguishing Pneumonia from Bronchitis

Suppose we know: Fever=Yes (state 2) and then Cough=No (new state 5).

Compatible diseases = Fever=Yes AND Cough=No = {Pneumonia, Bronchitis} (2 diseases)

**Action = "Ask Fatigue" (a=2)**:
| Disease | Has Fatigue? | Next State |
|---------|-------------|------------|
| Pneumonia | Yes (1) | [2,1,2,0,0] → different state |
| Bronchitis | No (0) | [2,1,1,0,0] → different state |

Result: P = 0.5 for each outcome.
- If Fatigue=Yes → only Pneumonia compatible → Diagnose Pneumonia!
- If Fatigue=No → only Bronchitis compatible → Diagnose Bronchitis!

### The General Formula

```
P(s' | s, a) = (Number of compatible diseases that lead to s') / (Total compatible diseases)
```

This is the formula from the presentation:

P(s'|s,a) = Σ [1/|Compatible|] × I[s' = T(s,a,d)] for all diseases d in Compatible

Where:
- Compatible = diseases that match all known symptoms so far
- T(s,a,d) = the next state if disease is d and we take action a
- I[...] = 1 if the condition is true, 0 otherwise

### Why This Matters

This transition probability is the **model** of the environment. Algorithms that USE this model (Policy Iteration, Value Iteration) are called **model-based**. Algorithms that DON'T use it (SARSA, Monte Carlo, etc.) are called **model-free** — they just learn from trial and error.

---

## 6. The 10 Algorithms — Explained One by One

We implemented 10 algorithms across 4 categories. Here's each one explained simply.

### CATEGORY 1: Dynamic Programming (Model-Based)

These algorithms KNOW the transition probabilities P(s'|s,a). They use this knowledge to mathematically compute the best policy without any trial-and-error.

#### Algorithm 1: Policy Iteration ⭐ (THE WINNER)

**Analogy**: Imagine you have a study plan. Step 1: Check how well your current plan works (grade yourself). Step 2: Improve your plan based on the grades. Repeat until the plan doesn't change anymore.

**How it works (2 alternating steps)**:

**Step 1 — Policy Evaluation**: "How good is my current strategy?"
- Take the current policy (strategy)
- Calculate V(s) = the expected total reward starting from state s if you follow this policy
- Uses the formula: V(s) = R(s, π(s)) + γ × Σ P(s'|s, π(s)) × V(s')
- In English: "My value = my immediate reward + 0.9× (probability of each next state × value of that state)"
- Keep updating until values stop changing

**Step 2 — Policy Improvement**: "Can I do better?"
- For each state, check all 13 actions
- For each action, calculate Q(s,a) = immediate reward + discounted future value
- Pick the action with the highest Q(s,a) as the new policy
- If the policy didn't change → we've found the optimal policy. STOP!

**In our project**: Converges in just ~4 iterations! That's because the problem is small enough (244 states) for DP to handle easily.

**Why it's the best**: It finds the **mathematically guaranteed optimal** policy. No randomness, no approximation, no tuning.

#### Algorithm 2: Value Iteration

**Same idea but different approach**: Instead of alternating between evaluation and improvement, it does both at once in every step.

**Formula**: V(s) = max_a [R(s,a) + γ × Σ P(s'|s,a) × V(s')]

In English: "For each state, find the action that gives the best value, and set V(s) to that."

**Difference from Policy Iteration**:
- PI: Fully evaluates the policy, then improves. Converges in ~4 "big" iterations.
- VI: Takes tiny steps combining both. Converges in ~15 "small" iterations.
- **Both find the exact same optimal policy!**

**Why not chosen as best**: It takes ~15 iterations vs PI's ~4. The final answer is the same, but PI is faster.

---

### CATEGORY 2: Model-Free Control

These algorithms do NOT know P(s'|s,a). They learn by **trial and error** — actually playing the game thousands of times and learning from experience.

**Key difference from DP**: Instead of having a "cheat sheet" (the transition model), these algorithms are like students who learn by solving thousands of practice problems.

#### Algorithm 3: GLIE Monte Carlo Control

**GLIE** = Greedy in the Limit with Infinite Exploration

**Analogy**: A student who studies by taking full practice exams, scores them at the end, and then reviews what went right/wrong.

**How it works**:
1. Play a FULL episode (from start to diagnosis) using an ε-greedy policy (usually follow best action, but randomly explore ε% of the time)
2. After the episode ends, look back at the total return G for each (state, action) visited
3. Update: Q(s,a) = Q(s,a) + (1/N) × (G - Q(s,a)) where N is the visit count
4. Slowly decrease ε so the AI explores less over time (converges to greedy)

**Key properties**:
- **Unbiased** — the return G is the actual reward the AI received, not an estimate
- **High variance** — sometimes lucky, sometimes unlucky episodes give very different returns
- **Needs 50,000+ episodes** to converge
- Updates only happen AFTER the full episode ends (must wait until diagnosis is made)

#### Algorithm 4: SARSA (TD(0) On-Policy Control)

**Name**: S-A-R-S'-A' → State, Action, Reward, next State, next Action

**Analogy**: A student who reviews their answers after EVERY question, not just at the end of the exam.

**How it works**:
1. In state s, take action a (ε-greedy), get reward r, end up in state s'
2. In state s', choose next action a' (ε-greedy)
3. Update immediately: Q(s,a) = Q(s,a) + α × [r + γ×Q(s',a') - Q(s,a)]
4. Move to s', repeat

**Key difference vs Monte Carlo**: SARSA doesn't wait for the episode to end. It updates after EVERY single step. This is called **bootstrapping** — using an estimate (Q(s',a')) instead of the actual return.

**Key properties**:
- **Biased** (uses estimate Q(s',a')) but **lower variance** than MC
- **On-policy** — it learns about the policy it's actually following (including exploration)
- Needs α (learning rate) and ε (exploration rate) — hyperparameters you must tune

#### Algorithm 5: SARSA(λ) — With Eligibility Traces

**λ (lambda)** blends Monte Carlo and SARSA into one algorithm.

**Analogy**: Imagine you did something good on question 5, and the reward came on question 8. MC would only update question 5 after the exam ends. SARSA would only update question 7 (the step right before the reward). SARSA(λ) updates MULTIPLE past steps — how many depends on λ.

**How it works**:
- Keeps a "trace" for each (state, action) — a record of how recently it was visited
- When a reward happens, ALL recently-visited state-action pairs get updated (proportional to their trace)
- λ=0 → pure SARSA (only update the last step)
- λ=1 → pure Monte Carlo (update all steps equally)
- λ=0.8 (our choice) → mostly Monte Carlo but with some TD bootstrapping

**Key properties**:
- Extra hyperparameter λ to tune
- In our project, got the highest average steps (5.2) — slightly less efficient than others

---

### CATEGORY 3: Function Approximation

**The motivation**: In Categories 1 and 2, we stored Q(s,a) in a big table with 244×13 = 3,172 entries. This works fine for our small problem. But for real hospitals with 1000+ diseases and 100+ symptoms? The table would have billions of entries!

**The idea**: Instead of a table, use a mathematical FUNCTION to approximate Q(s,a):
```
Q̂(s,a) ≈ φ(s,a)ᵀ × w
```
Where φ(s,a) is a **feature vector** (a description of the state-action pair) and w is a **weight vector** that gets learned.

**Our feature vector (215 dimensions)**:
- 195 features: interactions (which symptom × its status × which action)
- 13 features: which action is being taken (one-hot)
- 6 features: how many symptoms are known (binned)
- 1 feature: bias term
- Total: 195 + 13 + 6 + 1 = **215 features**

Instead of learning 3,172 Q-values, we only learn 215 weights. This is more compact and can generalize across similar states.

#### Algorithm 6: Monte Carlo with Function Approximation (MC-FA)

Same logic as GLIE MC, but instead of updating a Q-table, it updates the weight vector w:

**Update**: w = w + α × (G_t - φ(s,a)ᵀw) × φ(s,a)

In English: "Adjust the weights so that the predicted Q-value gets closer to the actual return."

#### Algorithm 7: SARSA with Function Approximation (SARSA-FA)

Same logic as SARSA, but with function approximation:

**Update**: w = w + α × (r + γ×Q̂(s',a') - Q̂(s,a)) × φ(s,a)

Called "semi-gradient" because we only take the gradient of Q̂(s,a) not Q̂(s',a').

#### Algorithm 8: LSPI (Least Squares Policy Iteration)

A batch method that's quite different:

1. Collect a bunch of experience (10,000 episodes)
2. Build matrices A and b from all the data
3. Solve w = A⁻¹b (least squares — like solving a system of equations)
4. Extract policy from w
5. Collect more data with new policy, repeat

**Key property**: No learning rate needed! The least squares solution is exact for the given data. But it needs a lot of data upfront and is very slow.

---

### CATEGORY 4: Policy Gradient

**The fundamental difference**: All previous algorithms learn Q-values (how good is each action?) and then derive a policy. Policy Gradient methods **directly learn the policy itself** — no Q-values at all!

**The policy is a formula**: π_θ(a|s) = probability of taking action a in state s, parameterized by θ.

We use a **softmax** policy: the probability of each action is proportional to exp(φ(s,a)ᵀ × θ).

#### Algorithm 9: REINFORCE (Monte Carlo Policy Gradient)

**How it works**:
1. Play a full episode using the current policy π_θ
2. For each (state, action) in the episode, calculate the return G_t
3. Update: θ = θ + α × ∇log(π_θ(a|s)) × G_t
4. ∇log(π_θ(a|s)) is the **score function** — it tells us how to change θ to make this action more likely

**In English**: "If an action led to a good return (high G_t), increase its probability. If it led to a bad return, decrease its probability."

**Key property**: Very high variance because G_t fluctuates wildly. Needs 100,000+ episodes.

#### Algorithm 10: Actor-Critic

**The fix for REINFORCE's variance problem**: Instead of using G_t (which is noisy), use a critic to estimate values.

**Two components**:
- **Actor**: The policy π_θ(a|s) — makes decisions
- **Critic**: A value function V̂(s) — evaluates how good the current state is

**Update**:
- **TD error**: δ = r + γ×V̂(s') - V̂(s)
- **Critic update**: v = v + α_critic × δ × ∇V̂(s)
- **Actor update**: θ = θ + α_actor × δ × ∇log(π_θ(a|s))

**In English**: The critic says "this state was better/worse than expected" (the TD error δ). The actor then adjusts the policy based on this feedback. Much lower variance than REINFORCE because δ is more stable than G_t.

**Key property**: Two learning rates to tune (one for actor, one for critic). Sometimes gets 7/8 accuracy instead of 8/8 due to training instability.

---

## 7. Why Policy Iteration Wins Over Everything Else

### The Short Answer

**Policy Iteration wins because our problem is small enough for it to work, and when it works, nothing can beat it.**

### The Detailed Reasoning

#### Reason 1: We Built the Transition Model — Why Not Use It?

We explicitly calculated P(s'|s,a) using the disease patterns and compatible disease logic. Since we HAVE this model, why would we throw it away and use model-free methods that ignore it?

| Approach | What it does | Like... |
|----------|-------------|---------|
| Policy Iteration | Uses the model to compute exact solution | Having the answer key and using it |
| SARSA/MC | Ignores the model, learns by trial and error | Throwing away the answer key and guessing |

Model-free algorithms are amazing when you DON'T have the model (e.g., training a robot to walk — you can't write down physics equations easily). But when you DO have the model, using it is always better.

#### Reason 2: Exact vs Approximate

- **Policy Iteration**: Solves the Bellman equations EXACTLY. The policy it finds is mathematically proven to be optimal.
- **SARSA/MC**: Approximate the Q-values through sampling. They converge to the optimal given infinite episodes, but in practice with finite episodes, there's always some approximation error.
- **FA methods**: Approximate Q-values with a 215-weight linear function. This introduces function approximation error ON TOP OF sampling error.
- **Policy Gradient**: Approximate the policy directly. Even more approximation.

**Approximation hierarchy**: PI (exact) > MC/SARSA (sampling error only) > FA (sampling + approximation error) > PG (direct policy approximation)

#### Reason 3: Training Speed

| Algorithm | Training Time | Why |
|-----------|-------------|-----|
| Policy Iteration | ~4 iterations, <0.5 seconds | Directly solves equations |
| Value Iteration | ~15 iterations, ~2.5 seconds | One-step updates, more iterations |
| GLIE MC | 50,000 episodes, ~1.8 seconds | Needs many episodes, but simple |
| SARSA | 50,000 episodes, ~1.8 seconds | Same |
| SARSA(λ) | 50,000 episodes, ~2.8 seconds | Traces add overhead |
| MC-FA | 50,000 episodes, ~10 seconds | Feature computation adds overhead |
| SARSA-FA | 50,000 episodes, ~11 seconds | Same |
| LSPI | 10,000 episodes + matrix solve, ~63 seconds | Matrix inversion is expensive |
| REINFORCE | 100,000 episodes, ~60 seconds | Very slow convergence |
| Actor-Critic | 50,000 episodes, ~36 seconds | Two networks to train |

#### Reason 4: No Hyperparameters

| Algorithm | Hyperparameters to Tune |
|-----------|------------------------|
| **Policy Iteration** | **γ only** (and γ=0.9 is standard) |
| Value Iteration | γ |
| GLIE MC | γ, ε_decay |
| SARSA | γ, α, ε_decay |
| SARSA(λ) | γ, α, ε_decay, **λ** |
| MC-FA | γ, α, ε_decay |
| SARSA-FA | γ, α, ε_decay |
| LSPI | γ, ε |
| REINFORCE | γ, α |
| Actor-Critic | γ, α_actor, α_critic |

Hyperparameters are dangerous: wrong values → algorithm doesn't converge or converges to a bad policy. PI has essentially nothing to tune.

#### Reason 5: Deterministic Decisions

In a medical setting, you want the AI to give the **same** answer every time for the same inputs. A patient shouldn't get different diagnoses based on which random number the AI generated.

- **PI**: Once trained, the policy is 100% deterministic. Same state → always same action.
- **Model-free during training**: Use ε-greedy (random exploration). Even after training, if ε > 0, there's randomness.
- **Policy Gradient**: The policy is ALWAYS probabilistic (softmax). Actions are sampled from a distribution, not chosen deterministically.

#### Reason 6: State Space Is Small

244 states × 13 actions = 3,172 state-action pairs. Policy Iteration can easily handle millions. Our problem is trivially small for it.

FA methods are designed for HUGE state spaces (millions+) where tables don't fit in memory. Using them here is like using a crane to lift a coffee cup — it works, but it's overkill.

**When WOULD you want FA/PG?**
- 50 diseases, 20 symptoms → 3²⁰ ≈ 3.5 billion states → table doesn't fit → need FA
- Continuous state spaces (e.g., vital signs as numbers, not binary) → need PG

### Summary Table: Why PI Beats Each Algorithm

| Algorithm | PI is better because... |
|-----------|------------------------|
| Value Iteration | Same answer, but PI is 3x faster (4 vs 15 iterations) |
| GLIE MC | PI is exact; MC needs 50k episodes and has high variance |
| SARSA | PI is exact; SARSA needs 50k episodes and has α to tune |
| SARSA(λ) | PI is exact; SARSA(λ) has worst avg steps (5.2) and λ to tune |
| MC-FA | PI is exact; FA adds unnecessary complexity for 244 states |
| SARSA-FA | Same — FA is overkill for our small state space |
| LSPI | PI is instantaneous; LSPI needs 10k episodes + slow matrix math |
| REINFORCE | PI is exact; REINFORCE needs 100k episodes, high variance |
| Actor-Critic | PI is exact; A-C sometimes gets 7/8, has 2 learning rates |

---

## 8. Results Discussion

### Result 1: All 10 Algorithms Achieve 100% Accuracy

Every single algorithm correctly diagnoses all 8 diseases. This validates:
- Our MDP design is sound (diseases are distinguishable)
- The reward structure works (correct=+10, wrong=-5 motivates accuracy)
- All 10 algorithms eventually learn the right thing

### Result 2: PI Uses Fewest Steps (4.0 Average)

PI asks an average of 4.0 questions per diagnosis. This is essentially the theoretical minimum needed to distinguish 8 diseases with binary symptoms (since log₂(8) ≈ 3, and we need slightly more because symptoms overlap).

### Result 3: What Strategy Did the AI Learn?

The AI discovered **hierarchical elimination** on its own:

1. **Ask Fever first** — splits diseases into two groups of 4 each (Fever-positive vs Fever-negative)
2. **Ask the most discriminative next symptom** based on which group the patient is in
3. **Diagnose after 3-4 questions** when only one disease remains compatible

This is exactly what an experienced human doctor would do! The AI independently rediscovered optimal medical reasoning.

---

## 9. Likely Viva Questions and Answers

**Q: What is the problem you are solving?**
A: Building an AI doctor that diagnoses 8 diseases by asking 5 symptoms in the optimal order using Reinforcement Learning.

**Q: What is the state space?**
A: 244 states. Each state encodes the AI's knowledge about 5 symptoms (Unknown/Absent/Present). 3⁵=243 knowledge states + 1 terminal.

**Q: How do you calculate transition probabilities?**
A: We average over all compatible diseases. If the AI is in a state where Fever=Present and asks about Cough, we find all diseases that have Fever. Among those, some have Cough (→ state X) and some don't (→ state Y). The probability of each next state = fraction of compatible diseases that lead there.

**Q: Why did you choose Policy Iteration?**
A: Because our state space is small (244) and we have the complete transition model. PI gives the mathematically exact optimal policy in just 4 iterations. No other algorithm can produce a better policy.

**Q: What if you had 1000 diseases?**
A: PI would no longer be practical because the state space would be too large. We'd use Function Approximation (SARSA-FA, LSPI) or Policy Gradient methods (Actor-Critic) that can generalize across states.

**Q: What is the difference between model-based and model-free?**
A: Model-based (PI, VI) uses P(s'|s,a) — the transition probabilities. We explicitly computed these from disease patterns. Model-free (SARSA, MC) doesn't use P(s'|s,a) — it learns purely from actually experiencing episodes.

**Q: What is the difference between Monte Carlo and SARSA?**
A: MC updates Q-values only after a complete episode ends, using the actual total return. SARSA updates after every single step, using a bootstrapped estimate. MC is unbiased but high variance; SARSA is biased but lower variance.

**Q: What is Function Approximation? Why do we need it?**
A: Instead of storing Q(s,a) in a table, we approximate it with a function: Q̂ = features × weights. This allows working with huge state spaces where a table wouldn't fit in memory. In our 244-state problem, it's overkill, but we implemented it to show we could.

**Q: What is the difference between REINFORCE and Actor-Critic?**
A: Both are Policy Gradient methods. REINFORCE uses Monte Carlo returns (G_t) → high variance. Actor-Critic uses TD error from a value function (the critic) → lower variance. Actor-Critic typically converges faster.

**Q: What does the discount factor γ=0.9 do?**
A: It makes future rewards slightly less valuable than immediate ones. Without it (γ=1), the AI wouldn't care about how many questions it asks. With γ=0.9, asking fewer questions (earlier diagnosis) is preferred because later rewards are discounted.

**Q: How many questions does the AI need on average?**
A: 4.0 questions on average with the optimal Policy Iteration policy. Some diseases need only 3 questions, some need 4.

---

*This document was prepared for the RL Medical Diagnosis Final Project by K. Chidwipak (S20230010131).*

"""
RL Medical Diagnosis - Full 32-State Grid Exploration

Two modes:
1. OPTIMAL: Shows the learned optimal policy (visits few states efficiently)
2. EXPLORATION: Uses random/varied symptom ordering to show all 32 states being visited

The grid visualization shows reachable vs visited states.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.policy_iteration import PolicyIteration
from src.value_iteration import ValueIteration

st.set_page_config(page_title="AI Doctor", page_icon="🩺", layout="wide")

st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
    }
    .result-success { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; padding: 2rem; border-radius: 15px; text-align: center; }
    .result-fail { background: linear-gradient(135deg, #cb2d3e 0%, #ef473a 100%); color: white; padding: 2rem; border-radius: 15px; text-align: center; }
</style>
""", unsafe_allow_html=True)

DISEASE_PATTERNS = {
    0: [1, 1, 1, 0, 0],
    1: [1, 1, 0, 1, 0],
    2: [1, 0, 1, 0, 1],
    3: [1, 0, 0, 1, 1],
    4: [0, 1, 1, 0, 1],
    5: [0, 1, 0, 1, 0],
    6: [0, 0, 1, 1, 0],
    7: [0, 0, 0, 0, 1],
}
DISEASE_NAMES = ["Flu", "Strep", "Pneumonia", "Bronchitis", "Cold", "Allergy", "Asthma", "Migraine"]
DISEASE_COLORS = ["#e74c3c", "#e67e22", "#9b59b6", "#2ecc71", "#3498db", "#1abc9c", "#f39c12", "#8e44ad"]
SYMPTOM_NAMES = ["Fever", "Cough", "Fatigue", "Breath", "Headache"]
SYMPTOM_EMOJIS = ["🌡️", "🤧", "😴", "😮‍💨", "🤕"]


@st.cache_resource
def get_algorithms():
    pi = PolicyIteration(gamma=0.9, theta=1e-6)
    pi_results = pi.run(verbose=False)
    vi = ValueIteration(gamma=0.9, theta=1e-6)
    vi_results = vi.run(verbose=False)
    return {
        'Policy Iteration': {'algo': pi, 'results': pi_results},
        'Value Iteration': {'algo': vi, 'results': vi_results}
    }


def find_matching_disease(symptoms):
    best_d, best_score = 0, -1
    for d in range(8):
        score = sum(symptoms[i] == DISEASE_PATTERNS[d][i] for i in range(5))
        if score > best_score:
            best_score, best_d = score, d
    return best_d, best_score


def create_state_grid(path_bits, diagnosed_disease=None, all_visited=None):
    """Create 32-state grid with path visualization."""
    fig = go.Figure()
    
    for row in range(8):
        for col in range(4):
            state = row * 4 + col
            binary = format(state, '05b')
            
            # Color based on status
            if state in path_bits:
                color = '#667eea' if state == path_bits[-1] else '#4CAF50'
                text_color = 'white'
            elif all_visited and state in all_visited:
                color = '#b3e5fc'  # Light blue for previously visited
                text_color = '#333'
            else:
                color = '#e8e8e8'
                text_color = '#666'
            
            known = [SYMPTOM_NAMES[i][0] for i in range(5) if state & (1 << i)]
            known_str = ",".join(known) if known else "∅"
            
            fig.add_shape(
                type="rect", x0=col, y0=7-row, x1=col+0.95, y1=7-row+0.95,
                fillcolor=color, line=dict(color='#444', width=1)
            )
            fig.add_annotation(
                x=col+0.475, y=7-row+0.475,
                text=f"s{state}<br>{known_str}",
                showarrow=False, font=dict(size=10, color=text_color)
            )
    
    # Disease row
    for col in range(min(4, 8)):
        color = '#FFD700' if diagnosed_disease == col else DISEASE_COLORS[col]
        fig.add_shape(
            type="rect", x0=col, y0=-1.2, x1=col+0.95, y1=-0.25,
            fillcolor=color, line=dict(color='#333', width=2)
        )
        fig.add_annotation(
            x=col+0.475, y=-0.725,
            text=f"<b>{DISEASE_NAMES[col][:6]}</b>",
            showarrow=False, font=dict(size=9, color='white')
        )
    
    # Second disease row (4-7)
    for col in range(4, 8):
        color = '#FFD700' if diagnosed_disease == col else DISEASE_COLORS[col]
        fig.add_shape(
            type="rect", x0=col-4, y0=-2.2, x1=col-4+0.95, y1=-1.35,
            fillcolor=color, line=dict(color='#333', width=2)
        )
        fig.add_annotation(
            x=col-4+0.475, y=-1.775,
            text=f"<b>{DISEASE_NAMES[col][:6]}</b>",
            showarrow=False, font=dict(size=9, color='white')
        )
    
    # Draw path
    if path_bits:
        path_x, path_y = [], []
        for state in path_bits:
            path_x.append(state % 4 + 0.475)
            path_y.append(7 - state // 4 + 0.475)
        
        if diagnosed_disease is not None:
            if diagnosed_disease < 4:
                path_x.append(diagnosed_disease + 0.475)
                path_y.append(-0.725)
            else:
                path_x.append(diagnosed_disease - 4 + 0.475)
                path_y.append(-1.775)
        
        fig.add_trace(go.Scatter(
            x=path_x, y=path_y,
            mode='lines+markers',
            line=dict(color='#FFD700', width=5),
            marker=dict(size=16, color='#FFD700'),
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        title=dict(text="32-State Knowledge Grid → 8 Diseases", font=dict(size=14)),
        xaxis=dict(visible=False, range=[-0.3, 4.3]),
        yaxis=dict(visible=False, range=[-2.5, 8.3]),
        height=700, showlegend=False,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


def run_exploration(patient_symptoms, placeholder, speed=1.0):
    """Exploration mode: ask symptoms in random order to visit more states."""
    disease_id, match_score = find_matching_disease(patient_symptoms)
    
    # Random symptom order for exploration
    symptom_order = list(range(5))
    random.shuffle(symptom_order)
    
    path_bits = [0]
    current_bits = 0
    
    for step, symptom_idx in enumerate(symptom_order):
        current_bits |= (1 << symptom_idx)
        path_bits.append(current_bits)
        
        with placeholder.container():
            col1, col2 = st.columns([2.5, 1])
            with col1:
                fig = create_state_grid(path_bits, None)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.markdown("### 🔍 Exploration Mode")
                st.metric("State", f"s{current_bits}")
                st.markdown(f"**Asked:** {SYMPTOM_NAMES[symptom_idx]}")
                st.markdown(f"**Answer:** {'Yes ✓' if patient_symptoms[symptom_idx] else 'No ✗'}")
                st.markdown("### 🛤️ Path")
                st.code(" → ".join([f"s{s}" for s in path_bits]))
        
        time.sleep(0.8 / speed)
    
    return {
        'diagnosed': disease_id,
        'success': True,
        'path': path_bits,
        'steps': 5
    }


def run_optimal(algo, patient_symptoms, placeholder, speed=1.0):
    """Optimal mode: use learned policy."""
    disease_id, _ = find_matching_disease(patient_symptoms)
    
    state = 0
    path_bits = [0]
    actions = []
    
    for step in range(10):
        action = algo.policy[state]
        actions.append(action)
        
        with placeholder.container():
            col1, col2 = st.columns([2.5, 1])
            with col1:
                fig = create_state_grid(path_bits, None)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.markdown("### 🎯 Optimal Mode")
                st.metric("State", f"s{path_bits[-1]}")
                st.markdown(f"**Action:** {algo.get_action_name(action)}")
                st.markdown("### 🛤️ Path")
                st.code(" → ".join([f"s{s}" for s in path_bits]))
        
        time.sleep(1.0 / speed)
        
        if action >= 5:
            diagnosed = action - 5
            return {
                'diagnosed': diagnosed,
                'success': diagnosed == disease_id,
                'path': path_bits,
                'actions': actions,
                'steps': step + 1
            }
        
        symptom_idx = action
        symptom_val = patient_symptoms[symptom_idx]
        statuses = algo.state_to_symptom_status(state)
        statuses[symptom_idx] = symptom_val + 1
        state = algo.symptom_status_to_state(statuses)
        new_bits = sum((1 << i) if statuses[i] != 0 else 0 for i in range(5))
        path_bits.append(new_bits)
    
    return {'success': False}


def main():
    st.markdown("""
    <div class="main-header">
        <h1>🩺 AI Doctor - 32-State Grid Explorer</h1>
        <p>Watch the AI explore the full state space!</p>
    </div>
    """, unsafe_allow_html=True)
    
    algorithms = get_algorithms()
    
    with st.sidebar:
        st.header("⚙️ Settings")
        mode = st.radio("Mode", ["🔍 Exploration (All States)", "🎯 Optimal Policy"])
        algo_choice = st.radio("Algorithm", ["Policy Iteration", "Value Iteration"])
        speed = st.slider("Speed", 0.5, 3.0, 1.5, 0.5)
        
        st.markdown("---")
        st.header("🦠 Diseases")
        st.markdown("""
| Disease | Symptoms |
|---------|----------|
| 🤒 Flu | F, C, Ft |
| 🔴 Strep | F, C, B |
| 🫁 Pneumonia | F, Ft, H |
| 💨 Bronchitis | F, B, H |
| 🤧 Cold | C, Ft, H |
| 🌸 Allergy | C, B |
| 😤 Asthma | Ft, B |
| 🤕 Migraine | H |

*F=Fever, C=Cough, Ft=Fatigue, B=Breath, H=Headache*
        """)
        
        st.markdown("---")
        st.header("📊 Modes")
        st.markdown("""
        **🔍 Exploration**: Random symptom order, visits many states
        
        **🎯 Optimal**: Learned policy, efficient path
        """)
    
    st.markdown("## 1️⃣ Patient Symptoms")
    cols = st.columns(5)
    symptoms = []
    for i, (name, emoji) in enumerate(zip(SYMPTOM_NAMES, SYMPTOM_EMOJIS)):
        with cols[i]:
            has = st.checkbox(f"{emoji} {name}", key=f"s{i}")
            symptoms.append(1 if has else 0)
    
    disease_id, score = find_matching_disease(symptoms)
    st.info(f"💡 Best match: **{DISEASE_NAMES[disease_id]}** ({score}/5)")
    
    st.markdown("---")
    
    if st.button("🚀 Start Diagnosis", use_container_width=True):
        placeholder = st.empty()
        
        if "Exploration" in mode:
            result = run_exploration(symptoms, placeholder, speed)
        else:
            algo = algorithms[algo_choice]['algo']
            result = run_optimal(algo, symptoms, placeholder, speed)
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            diagnosed = result['diagnosed']
            if result['success']:
                st.markdown(f"""<div class="result-success">
                    <h1>✅ {DISEASE_NAMES[diagnosed]}</h1>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="result-fail">
                    <h1>❌ Wrong: {DISEASE_NAMES[diagnosed]}</h1>
                </div>""", unsafe_allow_html=True)
        
        with col2:
            st.metric("States Visited", len(result['path']))
            st.code(" → ".join([f"s{s}" for s in result['path']]))
        
        fig = create_state_grid(result['path'], result['diagnosed'])
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()

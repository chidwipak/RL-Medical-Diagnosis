"""
RL Medical Diagnosis - Complete Dashboard
Assignment 1 (DP): Policy Iteration & Value Iteration
Assignment 2 (Model-Free): GLIE Monte Carlo, SARSA, SARSA(λ)
Assignment 3 (FA & PG): MC-FA, SARSA-FA, LSPI, REINFORCE, Actor-Critic

Tab-based layout with 32-state grid visualization for all algorithms.
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
from src.monte_carlo import GLIEMonteCarlo
from src.sarsa import SARSA
from src.sarsa_lambda import SARSALambda
from src.mc_fa import MCFunctionApprox
from src.sarsa_fa import SARSAFunctionApprox
from src.lspi import LSPI
from src.reinforce import REINFORCE
from src.actor_critic import ActorCritic

st.set_page_config(page_title="AI Doctor - RL Medical Diagnosis", page_icon="🩺", layout="wide")

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
    .algo-card { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 1.5rem; border-radius: 15px; margin-bottom: 1rem; }
    .metric-card { background: white; padding: 1rem; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); text-align: center; }
</style>
""", unsafe_allow_html=True)

DISEASE_PATTERNS = {
    0: [1, 1, 1, 0, 0], 1: [1, 1, 0, 1, 0], 2: [1, 0, 1, 0, 1], 3: [1, 0, 0, 1, 1],
    4: [0, 1, 1, 0, 1], 5: [0, 1, 0, 1, 0], 6: [0, 0, 1, 1, 0], 7: [0, 0, 0, 0, 1],
}
DISEASE_NAMES = ["Flu", "Strep", "Pneumonia", "Bronchitis", "Cold", "Allergy", "Asthma", "Migraine"]
DISEASE_COLORS = ["#e74c3c", "#e67e22", "#9b59b6", "#2ecc71", "#3498db", "#1abc9c", "#f39c12", "#8e44ad"]
SYMPTOM_NAMES = ["Fever", "Cough", "Fatigue", "Breath", "Headache"]
SYMPTOM_EMOJIS = ["🌡️", "🤧", "😴", "😮‍💨", "🤕"]


# ===== CACHED ALGORITHM LOADING =====

@st.cache_resource
def get_dp_algorithms():
    pi = PolicyIteration(gamma=0.9, theta=1e-6)
    pi_results = pi.run(verbose=False)
    vi = ValueIteration(gamma=0.9, theta=1e-6)
    vi_results = vi.run(verbose=False)
    return {
        'Policy Iteration': {'algo': pi, 'results': pi_results},
        'Value Iteration': {'algo': vi, 'results': vi_results}
    }

@st.cache_resource
def get_model_free_algorithms():
    n_episodes = 50000
    mc = GLIEMonteCarlo(gamma=0.9, epsilon_decay=500.0)
    mc_results = mc.run(n_episodes=n_episodes, verbose=False)
    sarsa = SARSA(gamma=0.9, alpha=0.1, epsilon_decay=500.0)
    sarsa_results = sarsa.run(n_episodes=n_episodes, verbose=False)
    sarsa_l = SARSALambda(gamma=0.9, lambda_=0.8, alpha=0.1, epsilon_decay=500.0)
    sarsa_l_results = sarsa_l.run(n_episodes=n_episodes, verbose=False)
    return {
        'GLIE Monte Carlo': {'algo': mc, 'results': mc_results},
        'SARSA': {'algo': sarsa, 'results': sarsa_results},
        'SARSA(λ)': {'algo': sarsa_l, 'results': sarsa_l_results},
    }

@st.cache_resource
def get_fa_pg_algorithms():
    """Train Assignment 3 FA & Policy Gradient algorithms."""
    n_episodes = 50000

    mc_fa = MCFunctionApprox(gamma=0.9, alpha=0.01, epsilon_decay=500.0)
    mc_fa_results = mc_fa.run(n_episodes=n_episodes, verbose=False)

    sarsa_fa = SARSAFunctionApprox(gamma=0.9, alpha=0.01, epsilon_decay=500.0)
    sarsa_fa_results = sarsa_fa.run(n_episodes=n_episodes, verbose=False)

    lspi = LSPI(gamma=0.9, epsilon=0.1)
    lspi_results = lspi.run(n_sample_episodes=10000, max_iterations=20, verbose=False)

    reinforce = REINFORCE(gamma=0.9, alpha=0.005)
    reinforce_results = reinforce.run(n_episodes=n_episodes, verbose=False)

    ac = ActorCritic(gamma=0.9, alpha_actor=0.005, alpha_critic=0.01)
    ac_results = ac.run(n_episodes=n_episodes, verbose=False)

    return {
        'MC with FA': {'algo': mc_fa, 'results': mc_fa_results},
        'SARSA with FA': {'algo': sarsa_fa, 'results': sarsa_fa_results},
        'LSPI': {'algo': lspi, 'results': lspi_results},
        'REINFORCE': {'algo': reinforce, 'results': reinforce_results},
        'Actor-Critic': {'algo': ac, 'results': ac_results},
    }


# ===== SHARED HELPER FUNCTIONS =====

def find_matching_disease(symptoms):
    best_d, best_score = 0, -1
    for d in range(8):
        score = sum(symptoms[i] == DISEASE_PATTERNS[d][i] for i in range(5))
        if score > best_score:
            best_score, best_d = score, d
    return best_d, best_score


def create_state_grid(path_bits, diagnosed_disease=None, all_visited=None):
    fig = go.Figure()

    for row in range(8):
        for col in range(4):
            state = row * 4 + col
            binary = format(state, '05b')

            if state in path_bits:
                color = '#667eea' if state == path_bits[-1] else '#4CAF50'
                text_color = 'white'
            elif all_visited and state in all_visited:
                color = '#b3e5fc'
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


def get_symptom_inputs(key_prefix=""):
    cols = st.columns(5)
    symptoms = []
    for i, (name, emoji) in enumerate(zip(SYMPTOM_NAMES, SYMPTOM_EMOJIS)):
        with cols[i]:
            has = st.checkbox(f"{emoji} {name}", key=f"{key_prefix}s{i}")
            symptoms.append(1 if has else 0)
    return symptoms


def run_exploration(patient_symptoms, placeholder, speed=1.0):
    disease_id, _ = find_matching_disease(patient_symptoms)
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


def run_optimal(algo, patient_symptoms, placeholder, speed=1.0, algo_name="Algorithm"):
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
                st.markdown(f"### 🎯 {algo_name} - Optimal")
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


def show_result(result):
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


def disease_table_sidebar():
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


# ===== CONVERGENCE & Q-VALUE PLOTS =====

def plot_convergence_plotly(results_dict, colors_map=None):
    fig = go.Figure()
    if colors_map is None:
        colors_map = {}

    for name, res in results_dict.items():
        history = res['results']['history']
        if history:
            episodes = [h['episode'] for h in history]
            q_vals = [h['Q_s0_max'] for h in history]
            fig.add_trace(go.Scatter(
                x=episodes, y=q_vals,
                mode='lines+markers',
                name=name,
                line=dict(color=colors_map.get(name, '#666'), width=3),
                marker=dict(size=8)
            ))

    fig.update_layout(
        title="Q-Value / Score Convergence",
        xaxis_title="Episode",
        yaxis_title="Q(s=0, best)",
        height=400,
        template="plotly_white"
    )
    return fig


def plot_reward_convergence_plotly(results_dict, colors_map=None):
    fig = go.Figure()
    if colors_map is None:
        colors_map = {}

    window = 1000
    for name, res in results_dict.items():
        rewards = res['results'].get('total_rewards', [])
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            x_vals = list(range(window-1, len(rewards)))
            step = max(1, len(smoothed) // 500)
            fig.add_trace(go.Scatter(
                x=x_vals[::step], y=smoothed[::step],
                mode='lines',
                name=name,
                line=dict(color=colors_map.get(name, '#666'), width=2)
            ))

    fig.update_layout(
        title="Average Episode Reward (Smoothed)",
        xaxis_title="Episode",
        yaxis_title="Reward",
        height=400,
        template="plotly_white"
    )
    return fig


def _simulate_dp_episode(algo, patient_symptoms):
    best_disease = 0
    best_match = -1
    for d in range(8):
        match = sum(patient_symptoms[i] == algo.DISEASE_PATTERNS[d][i] for i in range(5))
        if match > best_match:
            best_match, best_disease = match, d

    state = 0
    for step in range(10):
        action = algo.policy[state]
        if action >= 5:
            diagnosed = action - 5
            return {'success': diagnosed == best_disease}
        symptom_idx = action
        symptom_val = patient_symptoms[symptom_idx]
        statuses = algo.state_to_symptom_status(state)
        statuses[symptom_idx] = symptom_val + 1
        state = algo.symptom_status_to_state(statuses)
    return {'success': False}


def plot_accuracy_comparison(algos_dict):
    names = []
    accuracies = []

    for name, data in algos_dict.items():
        algo = data['algo']
        correct = 0
        for d in range(8):
            symptoms = list(algo.DISEASE_PATTERNS[d])
            if hasattr(algo, 'simulate_episode'):
                result = algo.simulate_episode(symptoms)
            else:
                result = _simulate_dp_episode(algo, symptoms)
            if result['success']:
                correct += 1
        names.append(name)
        accuracies.append(100 * correct / 8)

    colors_map = {
        'GLIE Monte Carlo': '#3498db', 'SARSA': '#e74c3c', 'SARSA(λ)': '#2ecc71',
        'Policy Iteration': '#9b59b6', 'Value Iteration': '#f39c12',
        'MC with FA': '#1a5276', 'SARSA with FA': '#c0392b', 'LSPI': '#27ae60',
        'REINFORCE': '#8e44ad', 'Actor-Critic': '#d35400'
    }
    bar_colors = [colors_map.get(n, '#666') for n in names]

    fig = go.Figure(data=[go.Bar(
        x=names, y=accuracies,
        marker_color=bar_colors,
        text=[f"{a:.0f}%" for a in accuracies],
        textposition='outside'
    )])
    fig.update_layout(
        title="Diagnosis Accuracy (8 Diseases)",
        yaxis_title="Accuracy (%)",
        yaxis=dict(range=[0, 110]),
        height=400,
        template="plotly_white",
        xaxis_tickangle=-30
    )
    return fig


def plot_q_heatmap(algo, algo_name):
    key_states = [0, 1, 2, 3, 4, 8, 16]
    action_labels = [algo.get_action_name(a) for a in range(13)]

    z_data = []
    y_labels = []
    for s in key_states:
        if hasattr(algo, 'Q'):
            z_data.append([algo.Q[s, a] for a in range(13)])
        elif hasattr(algo, 'q_hat'):
            z_data.append([algo.q_hat(s, a) for a in range(13)])
        else:
            z_data.append([0.0 for a in range(13)])
        statuses = algo.state_to_symptom_status(s)
        known = [SYMPTOM_NAMES[i][0] for i in range(5) if statuses[i] != 0]
        y_labels.append(f"s{s} ({','.join(known) if known else '∅'})")

    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=action_labels,
        y=y_labels,
        colorscale='RdYlGn',
        text=[[f"{v:.2f}" for v in row] for row in z_data],
        texttemplate="%{text}",
        textfont={"size": 9},
    ))
    fig.update_layout(
        title=f"{algo_name}: Q-values for Key States",
        xaxis_title="Action",
        yaxis_title="State",
        height=350,
        template="plotly_white"
    )
    return fig


# ===== MAIN APP =====

def main():
    st.markdown("""
    <div class="main-header">
        <h1>🩺 AI Doctor - RL Medical Diagnosis</h1>
        <p>Dynamic Programming · Model-Free RL · Function Approximation · Policy Gradient</p>
    </div>
    """, unsafe_allow_html=True)

    # ===== SIDEBAR =====
    with st.sidebar:
        st.header("⚙️ Assignment 1 Settings")
        dp_mode = st.radio("Mode", ["🔍 Exploration (All States)", "🎯 Optimal Policy"], key="dp_mode")
        dp_algo_choice = st.radio("Algorithm", ["Policy Iteration", "Value Iteration"], key="dp_algo")
        dp_speed = st.slider("Speed", 0.5, 3.0, 1.5, 0.5, key="dp_speed")

        st.markdown("---")
        st.header("⚙️ Assignment 2 Settings")
        mf_mode = st.radio("Mode", ["🔍 Exploration (ε-Greedy)", "🎯 Learned Policy"], key="mf_mode")
        mf_algo_choice = st.radio("Algorithm",
                                   ["GLIE Monte Carlo", "SARSA", "SARSA(λ)"],
                                   key="mf_algo")
        mf_speed = st.slider("Speed", 0.5, 3.0, 1.5, 0.5, key="mf_speed")

        st.markdown("---")
        st.header("⚙️ Assignment 3 Settings")
        fa_mode = st.radio("Mode", ["🔍 Exploration", "🎯 Learned Policy"], key="fa_mode")
        fa_algo_choice = st.radio("Algorithm",
                                   ["MC with FA", "SARSA with FA", "LSPI", "REINFORCE", "Actor-Critic"],
                                   key="fa_algo")
        fa_speed = st.slider("Speed", 0.5, 3.0, 1.5, 0.5, key="fa_speed")

        st.markdown("---")
        st.header("🦠 Diseases")
        disease_table_sidebar()

    tab1, tab2, tab3 = st.tabs([
        "📊 Assignment 1: Dynamic Programming",
        "🧠 Assignment 2: Model-Free",
        "📐 Assignment 3: FA & Policy Gradient"
    ])

    # ===== ASSIGNMENT 1 TAB =====
    with tab1:
        dp_algorithms = get_dp_algorithms()

        st.markdown("## 1️⃣ Patient Symptoms")
        symptoms = get_symptom_inputs("dp_")

        disease_id, score = find_matching_disease(symptoms)
        st.info(f"💡 Best match: **{DISEASE_NAMES[disease_id]}** ({score}/5)")
        st.markdown("---")

        if st.button("🚀 Start Diagnosis", use_container_width=True, key="dp_start"):
            placeholder = st.empty()
            if "Exploration" in dp_mode:
                result = run_exploration(symptoms, placeholder, dp_speed)
            else:
                algo = dp_algorithms[dp_algo_choice]['algo']
                result = run_optimal(algo, symptoms, placeholder, dp_speed, dp_algo_choice)
            show_result(result)

    # ===== ASSIGNMENT 2 TAB =====
    with tab2:
        st.markdown("""
        <div class="algo-card">
            <h3>🧠 Model-Free Control Algorithms</h3>
            <p>These algorithms learn <b>without transition probabilities</b> — directly from episodes of experience using ε-greedy exploration.</p>
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("Training model-free algorithms (first load only)..."):
            mf_algorithms = get_model_free_algorithms()

        st.markdown("## 1️⃣ Patient Symptoms")
        symptoms = get_symptom_inputs("mf_")

        disease_id, score = find_matching_disease(symptoms)
        st.info(f"💡 Best match: **{DISEASE_NAMES[disease_id]}** ({score}/5)")
        st.markdown("---")

        if st.button("🚀 Start Diagnosis", use_container_width=True, key="mf_start"):
            placeholder = st.empty()
            if "Exploration" in mf_mode:
                result = run_exploration(symptoms, placeholder, mf_speed)
            else:
                algo = mf_algorithms[mf_algo_choice]['algo']
                result = run_optimal(algo, symptoms, placeholder, mf_speed, mf_algo_choice)
            show_result(result)

        st.markdown("---")
        st.markdown("## 📈 Training Analytics")

        mf_colors = {'GLIE Monte Carlo': '#3498db', 'SARSA': '#e74c3c', 'SARSA(λ)': '#2ecc71'}

        anal_tab1, anal_tab2, anal_tab3, anal_tab4 = st.tabs([
            "📉 Convergence", "🏆 Accuracy", "🗺️ Q-Values", "📊 Summary"
        ])

        with anal_tab1:
            col1, col2 = st.columns(2)
            with col1:
                fig = plot_convergence_plotly(mf_algorithms, mf_colors)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = plot_reward_convergence_plotly(mf_algorithms, mf_colors)
                st.plotly_chart(fig, use_container_width=True)

        with anal_tab2:
            all_algos = {}
            dp_algos = get_dp_algorithms()
            for name, data in dp_algos.items():
                all_algos[name] = data
            for name, data in mf_algorithms.items():
                all_algos[name] = data
            fig = plot_accuracy_comparison(all_algos)
            st.plotly_chart(fig, use_container_width=True)

        with anal_tab3:
            selected_algo = st.selectbox("Select Algorithm", list(mf_algorithms.keys()), key="q_select")
            algo = mf_algorithms[selected_algo]['algo']
            fig = plot_q_heatmap(algo, selected_algo)
            st.plotly_chart(fig, use_container_width=True)

        with anal_tab4:
            st.markdown("### 📊 Algorithm Comparison")
            summary_data = []
            for name, data in mf_algorithms.items():
                algo = data['algo']
                results = data['results']
                correct = sum(1 for d in range(8)
                              if algo.simulate_episode(list(algo.DISEASE_PATTERNS[d]))['success'])
                summary_data.append({
                    'Algorithm': name,
                    'Episodes': results['n_episodes'],
                    'Time (s)': f"{results['elapsed_time']:.2f}",
                    'Q(s=0, best)': f"{np.max(results['Q'][0]):.4f}",
                    'Accuracy': f"{correct}/8 ({100*correct/8:.0f}%)",
                })
            st.table(summary_data)

    # ===== ASSIGNMENT 3 TAB =====
    with tab3:
        st.markdown("""
        <div class="algo-card">
            <h3>📐 Function Approximation & Policy Gradient</h3>
            <p><b>FA algorithms</b> use feature vectors φ(s,a) instead of Q-tables — enabling generalization across states.<br>
            <b>Policy Gradient</b> algorithms directly learn the policy π_θ(a|s) instead of Q-values.</p>
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("Training FA & Policy Gradient algorithms (first load only)..."):
            fa_algorithms = get_fa_pg_algorithms()

        st.markdown("## 1️⃣ Patient Symptoms")
        symptoms = get_symptom_inputs("fa_")

        disease_id, score = find_matching_disease(symptoms)
        st.info(f"💡 Best match: **{DISEASE_NAMES[disease_id]}** ({score}/5)")
        st.markdown("---")

        if st.button("🚀 Start Diagnosis", use_container_width=True, key="fa_start"):
            placeholder = st.empty()
            if "Exploration" in fa_mode:
                result = run_exploration(symptoms, placeholder, fa_speed)
            else:
                algo = fa_algorithms[fa_algo_choice]['algo']
                result = run_optimal(algo, symptoms, placeholder, fa_speed, fa_algo_choice)
            show_result(result)

        # ===== TRAINING ANALYTICS =====
        st.markdown("---")
        st.markdown("## 📈 Training Analytics")

        fa_colors = {
            'MC with FA': '#1a5276', 'SARSA with FA': '#c0392b', 'LSPI': '#27ae60',
            'REINFORCE': '#8e44ad', 'Actor-Critic': '#d35400'
        }

        a3_tab1, a3_tab2, a3_tab3, a3_tab4 = st.tabs([
            "📉 Convergence", "🏆 Accuracy", "🗺️ Q-Values", "📊 Summary"
        ])

        with a3_tab1:
            col1, col2 = st.columns(2)
            with col1:
                fig = plot_convergence_plotly(fa_algorithms, fa_colors)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = plot_reward_convergence_plotly(fa_algorithms, fa_colors)
                st.plotly_chart(fig, use_container_width=True)

        with a3_tab2:
            # Compare ALL algorithms across all assignments
            all_algos = {}
            for name, data in get_dp_algorithms().items():
                all_algos[name] = data
            for name, data in get_model_free_algorithms().items():
                all_algos[name] = data
            for name, data in fa_algorithms.items():
                all_algos[name] = data
            fig = plot_accuracy_comparison(all_algos)
            st.plotly_chart(fig, use_container_width=True)

        with a3_tab3:
            selected_algo = st.selectbox("Select Algorithm", list(fa_algorithms.keys()), key="fa_q_select")
            algo = fa_algorithms[selected_algo]['algo']
            fig = plot_q_heatmap(algo, selected_algo)
            st.plotly_chart(fig, use_container_width=True)

        with a3_tab4:
            st.markdown("### 📊 Algorithm Comparison")

            summary_data = []
            for name, data in fa_algorithms.items():
                algo = data['algo']
                results = data['results']
                correct = sum(1 for d in range(8)
                              if algo.simulate_episode(list(algo.DISEASE_PATTERNS[d]))['success'])

                algo_type = "Policy Gradient" if name in ["REINFORCE", "Actor-Critic"] else "Function Approx"
                summary_data.append({
                    'Algorithm': name,
                    'Type': algo_type,
                    'Time (s)': f"{results['elapsed_time']:.2f}",
                    'Accuracy': f"{correct}/8 ({100*correct/8:.0f}%)",
                })
            st.table(summary_data)

            st.markdown("""
            ### 🔑 Key Differences

            | Feature | MC-FA | SARSA-FA | LSPI | REINFORCE | Actor-Critic |
            |---------|-------|----------|------|-----------|--------------|
            | Approach | Value-based | Value-based | Value-based | Policy-based | Both |
            | FA Type | Linear q̂ | Linear q̂ | Linear q̂ | Softmax π_θ | Softmax + Linear V̂ |
            | Update | End of episode | Every step | Batch | End of episode | Every step |
            | Method | MC + SGD | TD + semi-gradient | Least squares | Policy gradient | Actor + Critic |
            """)

            st.markdown("""
            ### 💡 Why Function Approximation?

            In Assignment 2, we stored Q-values in a **table** (244 × 13 = 3,172 entries).
            With FA, we use a **weight vector** of only **31 parameters** — a 100× reduction!

            This matters because:
            1. **Generalization**: Similar states share information through features
            2. **Scalability**: Works for much larger state spaces
            3. **Memory efficient**: Store weights, not entire Q-table
            """)

            st.markdown("""
            ### 💡 Why Policy Gradient?

            Value-based methods learn Q(s,a) and derive policy. Policy gradient methods
            **directly** optimize the policy — no need for Q-values!

            Advantages:
            1. **Stochastic policies**: Can learn mixed strategies
            2. **Continuous actions**: Naturally handles continuous action spaces
            3. **Convergence**: Guaranteed convergence to at least a local optimum
            """)


if __name__ == "__main__":
    main()

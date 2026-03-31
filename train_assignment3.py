"""
Training Script for Assignment 3: Function Approximation & Policy Gradient

Runs all 5 algorithms:
  FA: MC with FA, Semi-Gradient SARSA with FA, LSPI
  PG: REINFORCE, Actor-Critic

Compares results and generates visualization plots.

Author: K. Chidwipak
Roll No: S20230010131
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.mc_fa import MCFunctionApprox
from src.sarsa_fa import SARSAFunctionApprox
from src.lspi import LSPI
from src.reinforce import REINFORCE
from src.actor_critic import ActorCritic


def create_results_dir():
    dirs = [
        'results/assignment3',
        'results/assignment3/mc_fa',
        'results/assignment3/sarsa_fa',
        'results/assignment3/lspi',
        'results/assignment3/reinforce',
        'results/assignment3/actor_critic',
        'results/assignment3/comparison'
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def plot_convergence(history, total_rewards, algo_name, save_path):
    """Plot convergence curves for an algorithm."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Average reward over training
    window = 1000
    if len(total_rewards) >= window:
        smoothed = np.convolve(total_rewards, np.ones(window)/window, mode='valid')
        axes[0].plot(smoothed, 'b-', linewidth=1, alpha=0.8)
    elif total_rewards:
        axes[0].plot(total_rewards, 'b-', linewidth=1, alpha=0.8)
    axes[0].set_xlabel('Episode', fontsize=12)
    axes[0].set_ylabel('Average Reward (smoothed)', fontsize=12)
    axes[0].set_title(f'{algo_name}: Episode Rewards', fontsize=14)
    axes[0].grid(True, alpha=0.3)

    # Q(s=0, best) over time
    if history:
        episodes = [h['episode'] for h in history]
        q_vals = [h['Q_s0_max'] for h in history]
        axes[1].plot(episodes, q_vals, 'g-o', linewidth=2, markersize=5)
        axes[1].set_xlabel('Episode', fontsize=12)
        axes[1].set_ylabel('Q(s=0, best action)', fontsize=12)
        axes[1].set_title(f'{algo_name}: Q-value Convergence', fontsize=14)
        axes[1].grid(True, alpha=0.3)
        if q_vals:
            axes[1].axhline(y=q_vals[-1], color='r', linestyle='--',
                            label=f'Final: {q_vals[-1]:.4f}')
            axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def test_all_diseases(algo, algo_name):
    """Test algorithm on all 8 diseases."""
    print(f"\n  Testing {algo_name} on all 8 diseases:")
    correct = 0
    for d in range(8):
        symptoms = list(algo.DISEASE_PATTERNS[d])
        result = algo.simulate_episode(symptoms)
        status = "✓" if result['success'] else "✗"
        if result['success']:
            correct += 1
        diag_name = algo.DISEASE_NAMES[result['diagnosed']] if result.get('diagnosed') is not None else "None"
        path_str = " → ".join([f"s{b}" for b in result['path_bits']])
        print(f"    {algo.DISEASE_NAMES[d]:12s}: {status} → {diag_name:12s} | {path_str}")
    print(f"  Accuracy: {correct}/8 ({100*correct/8:.0f}%)")
    return correct


def compare_all(results_dict, save_path):
    """Compare all algorithms."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {
        'MC-FA': '#3498db', 'SARSA-FA': '#e74c3c', 'LSPI': '#2ecc71',
        'REINFORCE': '#9b59b6', 'Actor-Critic': '#f39c12'
    }

    # 1. Reward convergence
    ax = axes[0, 0]
    window = 1000
    for name, res in results_dict.items():
        rewards = res.get('total_rewards', [])
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(smoothed, color=colors.get(name, '#666'), linewidth=1, alpha=0.8, label=name)
        elif rewards:
            ax.plot(rewards, color=colors.get(name, '#666'), linewidth=1, alpha=0.8, label=name)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Reward')
    ax.set_title('Episode Reward Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Q(s=0) convergence
    ax = axes[0, 1]
    for name, res in results_dict.items():
        history = res.get('history', [])
        if history:
            episodes = [h['episode'] for h in history]
            q_vals = [h['Q_s0_max'] for h in history]
            ax.plot(episodes, q_vals, '-o', color=colors.get(name, '#666'), linewidth=2,
                    markersize=5, label=name)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Q(s=0, best)')
    ax.set_title('Q-Value Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Diagnosis accuracy
    ax = axes[1, 0]
    algo_names = list(results_dict.keys())
    accuracies = []
    for name, res in results_dict.items():
        algo = res['algo']
        c = sum(1 for d in range(8)
                if algo.simulate_episode(list(algo.DISEASE_PATTERNS[d]))['success'])
        accuracies.append(100 * c / 8)

    bars = ax.bar(algo_names, accuracies, color=[colors.get(n, '#666') for n in algo_names], alpha=0.8)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Diagnosis Accuracy (8 Diseases)')
    ax.set_ylim(0, 110)
    plt.setp(ax.get_xticklabels(), rotation=15, ha='right')
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{acc:.0f}%', ha='center', fontsize=11, fontweight='bold')

    # 4. Summary table
    ax = axes[1, 1]
    ax.axis('off')
    summary = "ASSIGNMENT 3 COMPARISON\n" + "=" * 40 + "\n\n"
    for name, res in results_dict.items():
        algo = res['algo']
        summary += f"{name}:\n"
        summary += f"  Time: {res.get('elapsed_time', 0):.2f}s\n"
        c = sum(1 for d in range(8)
                if algo.simulate_episode(list(algo.DISEASE_PATTERNS[d]))['success'])
        summary += f"  Accuracy: {c}/8 diseases\n\n"

    ax.text(0.1, 0.95, summary, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    print("\n" + "=" * 70)
    print("  RL MEDICAL DIAGNOSIS - ASSIGNMENT 3 TRAINING")
    print("  Function Approximation & Policy Gradient")
    print("=" * 70 + "\n")

    create_results_dir()

    results_dict = {}

    # ===== 1. MC Control with FA =====
    print("\n" + "=" * 70)
    print("  1. MC CONTROL WITH FUNCTION APPROXIMATION")
    print("=" * 70)
    mc_fa = MCFunctionApprox(gamma=0.9, alpha=0.01, epsilon_decay=500.0)
    mc_fa_results = mc_fa.run(n_episodes=50000, verbose=True)
    mc_fa_results['algo'] = mc_fa
    test_all_diseases(mc_fa, "MC-FA")
    plot_convergence(mc_fa_results['history'], mc_fa_results['total_rewards'],
                     'MC with Function Approximation', 'results/assignment3/mc_fa/convergence.png')
    results_dict['MC-FA'] = mc_fa_results

    # ===== 2. Semi-Gradient SARSA with FA =====
    print("\n" + "=" * 70)
    print("  2. SEMI-GRADIENT SARSA WITH FUNCTION APPROXIMATION")
    print("=" * 70)
    sarsa_fa = SARSAFunctionApprox(gamma=0.9, alpha=0.01, epsilon_decay=500.0)
    sarsa_fa_results = sarsa_fa.run(n_episodes=50000, verbose=True)
    sarsa_fa_results['algo'] = sarsa_fa
    test_all_diseases(sarsa_fa, "SARSA-FA")
    plot_convergence(sarsa_fa_results['history'], sarsa_fa_results['total_rewards'],
                     'Semi-Gradient SARSA with FA', 'results/assignment3/sarsa_fa/convergence.png')
    results_dict['SARSA-FA'] = sarsa_fa_results

    # ===== 3. LSPI =====
    print("\n" + "=" * 70)
    print("  3. LEAST SQUARES POLICY ITERATION (LSPI)")
    print("=" * 70)
    lspi = LSPI(gamma=0.9, epsilon=0.1)
    lspi_results = lspi.run(n_sample_episodes=10000, max_iterations=20, verbose=True)
    lspi_results['algo'] = lspi
    test_all_diseases(lspi, "LSPI")
    plot_convergence(lspi_results['history'], lspi_results.get('total_rewards', []),
                     'LSPI', 'results/assignment3/lspi/convergence.png')
    results_dict['LSPI'] = lspi_results

    # ===== 4. REINFORCE =====
    print("\n" + "=" * 70)
    print("  4. REINFORCE (MONTE CARLO POLICY GRADIENT)")
    print("=" * 70)
    reinforce = REINFORCE(gamma=0.9, alpha=0.005)
    reinforce_results = reinforce.run(n_episodes=50000, verbose=True)
    reinforce_results['algo'] = reinforce
    test_all_diseases(reinforce, "REINFORCE")
    plot_convergence(reinforce_results['history'], reinforce_results['total_rewards'],
                     'REINFORCE', 'results/assignment3/reinforce/convergence.png')
    results_dict['REINFORCE'] = reinforce_results

    # ===== 5. Actor-Critic =====
    print("\n" + "=" * 70)
    print("  5. ADVANTAGE ACTOR-CRITIC (A2C)")
    print("=" * 70)
    ac = ActorCritic(gamma=0.9, alpha_actor=0.005, alpha_critic=0.01)
    ac_results = ac.run(n_episodes=50000, verbose=True)
    ac_results['algo'] = ac
    test_all_diseases(ac, "Actor-Critic")
    plot_convergence(ac_results['history'], ac_results['total_rewards'],
                     'Actor-Critic', 'results/assignment3/actor_critic/convergence.png')
    results_dict['Actor-Critic'] = ac_results

    # ===== Comparison =====
    print("\n" + "=" * 70)
    print("  COMPARISON")
    print("=" * 70)
    compare_all(results_dict, 'results/assignment3/comparison/comparison.png')

    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE!")
    print("  Results saved to: results/assignment3/")
    print("  Run 'streamlit run app.py' for interactive visualization")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

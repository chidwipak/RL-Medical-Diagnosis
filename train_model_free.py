"""
Training Script for Model-Free RL Medical Diagnosis (Assignment 2)

Runs GLIE Monte Carlo, SARSA, and SARSA(λ) algorithms,
compares results, and generates visualization plots.

These algorithms learn WITHOUT transition probabilities.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.monte_carlo import GLIEMonteCarlo
from src.sarsa import SARSA
from src.sarsa_lambda import SARSALambda


def create_results_dir():
    dirs = [
        'results/model_free',
        'results/model_free/glie_mc',
        'results/model_free/sarsa',
        'results/model_free/sarsa_lambda',
        'results/model_free/comparison'
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def plot_convergence(history, total_rewards, algo_name, save_path):
    """Plot convergence curves for a model-free algorithm."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Average reward over training
    window = 1000
    if len(total_rewards) >= window:
        smoothed = np.convolve(total_rewards, np.ones(window)/window, mode='valid')
        axes[0].plot(smoothed, 'b-', linewidth=1, alpha=0.8)
    axes[0].set_xlabel('Episode', fontsize=12)
    axes[0].set_ylabel('Average Reward (smoothed)', fontsize=12)
    axes[0].set_title(f'{algo_name}: Episode Rewards', fontsize=14)
    axes[0].grid(True, alpha=0.3)

    # Q(s=0, best) over time
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


def compare_all(results_dict, save_path):
    """Compare all 3 model-free algorithms."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {'GLIE MC': 'blue', 'SARSA': 'red', 'SARSA(λ)': 'green'}

    # 1. Reward convergence
    ax = axes[0, 0]
    window = 1000
    for name, res in results_dict.items():
        rewards = res['total_rewards']
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(smoothed, color=colors[name], linewidth=1, alpha=0.8, label=name)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Reward')
    ax.set_title('Episode Reward Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Q(s=0) convergence
    ax = axes[0, 1]
    for name, res in results_dict.items():
        episodes = [h['episode'] for h in res['history']]
        q_vals = [h['Q_s0_max'] for h in res['history']]
        ax.plot(episodes, q_vals, '-o', color=colors[name], linewidth=2,
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
        correct = 0
        for d in range(8):
            symptoms = list(algo.DISEASE_PATTERNS[d])
            result = algo.simulate_episode(symptoms)
            if result['success']:
                correct += 1
        accuracies.append(100 * correct / 8)

    bars = ax.bar(algo_names, accuracies, color=[colors[n] for n in algo_names], alpha=0.8)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Diagnosis Accuracy (8 Diseases)')
    ax.set_ylim(0, 110)
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{acc:.0f}%', ha='center', fontsize=12, fontweight='bold')

    # 4. Summary table
    ax = axes[1, 1]
    ax.axis('off')
    summary = "MODEL-FREE COMPARISON\n" + "=" * 40 + "\n\n"
    for name, res in results_dict.items():
        summary += f"{name}:\n"
        summary += f"  Episodes: {res['n_episodes']}\n"
        summary += f"  Time: {res['elapsed_time']:.2f}s\n"
        summary += f"  Q(s=0,best): {np.max(res['Q'][0]):.4f}\n"
        algo = res['algo']
        correct = sum(1 for d in range(8)
                      if algo.simulate_episode(list(algo.DISEASE_PATTERNS[d]))['success'])
        summary += f"  Accuracy: {correct}/8 diseases\n\n"

    ax.text(0.1, 0.95, summary, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

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
        diag_name = algo.DISEASE_NAMES[result['diagnosed']] if result['diagnosed'] is not None else "None"
        path_str = " → ".join([f"s{b}" for b in result['path_bits']])
        print(f"    {algo.DISEASE_NAMES[d]:12s}: {status} → {diag_name:12s} | {path_str}")
    print(f"  Accuracy: {correct}/8 ({100*correct/8:.0f}%)")
    return correct


def main():
    print("\n" + "=" * 70)
    print("  RL MEDICAL DIAGNOSIS - MODEL-FREE TRAINING (Assignment 2)")
    print("  GLIE Monte Carlo | SARSA | SARSA(λ)")
    print("=" * 70 + "\n")

    create_results_dir()
    n_episodes = 50000

    # ===== GLIE Monte Carlo =====
    print("\n" + "=" * 70)
    print("  1. GLIE MONTE CARLO CONTROL")
    print("=" * 70)
    mc = GLIEMonteCarlo(gamma=0.9, epsilon_decay=500.0)
    mc_results = mc.run(n_episodes=n_episodes, verbose=True)
    mc_results['algo'] = mc
    test_all_diseases(mc, "GLIE MC")
    plot_convergence(mc_results['history'], mc_results['total_rewards'],
                     'GLIE Monte Carlo', 'results/model_free/glie_mc/convergence.png')

    # ===== SARSA =====
    print("\n" + "=" * 70)
    print("  2. SARSA (TD(0))")
    print("=" * 70)
    sarsa = SARSA(gamma=0.9, alpha=0.1, epsilon_decay=500.0)
    sarsa_results = sarsa.run(n_episodes=n_episodes, verbose=True)
    sarsa_results['algo'] = sarsa
    test_all_diseases(sarsa, "SARSA")
    plot_convergence(sarsa_results['history'], sarsa_results['total_rewards'],
                     'SARSA', 'results/model_free/sarsa/convergence.png')

    # ===== SARSA(λ) =====
    print("\n" + "=" * 70)
    print("  3. SARSA(λ=0.8)")
    print("=" * 70)
    sarsa_l = SARSALambda(gamma=0.9, lambda_=0.8, alpha=0.1, epsilon_decay=500.0)
    sarsa_l_results = sarsa_l.run(n_episodes=n_episodes, verbose=True)
    sarsa_l_results['algo'] = sarsa_l
    test_all_diseases(sarsa_l, "SARSA(λ)")
    plot_convergence(sarsa_l_results['history'], sarsa_l_results['total_rewards'],
                     'SARSA(λ)', 'results/model_free/sarsa_lambda/convergence.png')

    # ===== Comparison =====
    print("\n" + "=" * 70)
    print("  COMPARISON")
    print("=" * 70)

    results_dict = {
        'GLIE MC': mc_results,
        'SARSA': sarsa_results,
        'SARSA(λ)': sarsa_l_results,
    }
    compare_all(results_dict, 'results/model_free/comparison/comparison.png')

    # Save Q-values and policies
    for name, prefix in [('glie_mc', mc), ('sarsa', sarsa), ('sarsa_lambda', sarsa_l)]:
        np.save(f'results/model_free/{name}/Q.npy', prefix.Q)
        np.save(f'results/model_free/{name}/policy.npy', prefix.policy)

    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE!")
    print("  Results saved to: results/model_free/")
    print("  Run 'streamlit run app.py' for interactive visualization")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

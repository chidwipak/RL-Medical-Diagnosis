"""
Training Script for Medical Diagnosis RL

Runs both Policy Iteration and Value Iteration algorithms,
compares results, and generates visualizations.

Uses 243-state MDP where each symptom is Unknown/Absent/Present.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.policy_iteration import PolicyIteration
from src.value_iteration import ValueIteration


def create_results_dir():
    dirs = ['results', 'results/policy_iteration', 'results/value_iteration', 'results/comparison']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return dirs


def plot_convergence(history, algorithm_name, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    iterations = [h['iteration'] + 1 for h in history]
    v_initial = [h['V_initial_state'] for h in history]
    
    axes[0].plot(iterations, v_initial, 'b-', linewidth=2, marker='o', markersize=4)
    axes[0].set_xlabel('Iteration', fontsize=12)
    axes[0].set_ylabel('V(s=0)', fontsize=12)
    axes[0].set_title(f'{algorithm_name}: Value of Initial State', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=v_initial[-1], color='r', linestyle='--', label=f'Final: {v_initial[-1]:.4f}')
    axes[0].legend()
    
    if 'delta' in history[0]:
        deltas = [h['delta'] for h in history]
    else:
        deltas = [h['final_delta'] for h in history]
    
    axes[1].semilogy(iterations, deltas, 'g-', linewidth=2, marker='o', markersize=4)
    axes[1].set_xlabel('Iteration', fontsize=12)
    axes[1].set_ylabel('Delta (log scale)', fontsize=12)
    axes[1].set_title(f'{algorithm_name}: Convergence', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def compare_algorithms(pi_results, vi_results, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Convergence comparison
    ax1 = axes[0, 0]
    pi_iters = [h['iteration'] + 1 for h in pi_results['history']]
    pi_v0 = [h['V_initial_state'] for h in pi_results['history']]
    vi_iters = [h['iteration'] + 1 for h in vi_results['history']]
    vi_v0 = [h['V_initial_state'] for h in vi_results['history']]
    
    ax1.plot(pi_iters, pi_v0, 'b-o', label='Policy Iteration', markersize=4)
    ax1.plot(vi_iters, vi_v0, 'r-s', label='Value Iteration', markersize=4)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('V(s=0)')
    ax1.set_title('Convergence: V(initial state)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Value histogram
    ax2 = axes[0, 1]
    ax2.hist(pi_results['V'][:-1], bins=30, alpha=0.7, label='Policy Iteration')  
    ax2.hist(vi_results['V'][:-1], bins=30, alpha=0.7, label='Value Iteration')
    ax2.set_xlabel('State Value V(s)')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of State Values')
    ax2.legend()
    
    # Policy agreement
    ax3 = axes[1, 0]
    policy_match = (pi_results['policy'][:-1] == vi_results['policy'][:-1])
    match_pct = 100 * policy_match.mean()
    ax3.pie([match_pct, 100-match_pct], labels=['Match', 'Differ'], 
            autopct='%1.1f%%', colors=['green', 'red'])
    ax3.set_title(f'Policy Agreement: {match_pct:.1f}%')
    
    # Summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""
    ALGORITHM COMPARISON (243-state MDP)
    {'='*40}
    
    POLICY ITERATION
    • Iterations: {pi_results['iterations']}
    • Time: {pi_results['elapsed_time']:.4f} sec
    • V(s=0): {pi_results['V'][0]:.4f}
    • Converged: {pi_results['converged']}
    
    VALUE ITERATION  
    • Iterations: {vi_results['iterations']}
    • Time: {vi_results['elapsed_time']:.4f} sec
    • V(s=0): {vi_results['V'][0]:.4f}
    • Converged: {vi_results['converged']}
    
    COMPARISON
    • Max V(s) diff: {np.max(np.abs(pi_results['V'] - vi_results['V'])):.6f}
    • Policy match: {policy_match.sum()}/{len(policy_match)} states
    """
    
    ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes, 
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def simulate_episode(algorithm_instance, policy, disease=0, max_steps=20, verbose=True):
    """Simulate episode with the 243-state MDP."""
    DISEASE_NAMES = {0: 'Flu', 1: 'Pneumonia'}
    true_symptoms = algorithm_instance.DISEASE_PATTERNS[disease]
    
    state = 0  # Initial state: all symptoms unknown
    total_reward = 0
    steps = 0
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Episode Simulation (True Disease: {DISEASE_NAMES[disease]})")
        print(f"True Symptoms: {dict(zip(algorithm_instance.SYMPTOM_NAMES, true_symptoms))}")
        print(f"{'='*70}")
    
    for step in range(max_steps):
        action = policy[state]
        action_name = algorithm_instance.get_action_name(action)
        state_desc = algorithm_instance.get_state_description(state)
        
        if verbose:
            print(f"Step {step+1}: {state_desc}")
            print(f"  → Action: {action_name}")
        
        # Get reward and next state
        reward = algorithm_instance.get_reward(state, action, disease)
        new_state = algorithm_instance.get_next_state_for_disease(state, action, disease)
        
        total_reward += reward
        
        if action >= 10:  # Diagnosis
            diagnosed = action - 10
            success = (diagnosed == disease)
            if verbose:
                result = "CORRECT! ✓" if success else "WRONG! ✗"
                print(f"  → Diagnosed: {DISEASE_NAMES[diagnosed]} ({result})")
                print(f"  → Reward: {reward:.2f}")
                print(f"  Total Reward: {total_reward:.2f}")
            return total_reward, step + 1, success
        else:
            if verbose:
                print(f"  → Reward: {reward:.2f}")
        
        state = new_state
        steps = step + 1
    
    # Timeout
    total_reward -= 2.0
    if verbose:
        print(f"TIMEOUT! Total Reward: {total_reward:.2f}")
    return total_reward, steps, False


def main():
    print("\n" + "=" * 70)
    print("  RL MEDICAL DIAGNOSIS - TRAINING")
    print("  Policy Iteration vs Value Iteration (243-state MDP)")
    print("=" * 70 + "\n")
    
    create_results_dir()
    
    # Run Policy Iteration
    print("\n" + "=" * 70)
    print("  RUNNING POLICY ITERATION")
    print("=" * 70)
    pi = PolicyIteration(gamma=0.9, theta=1e-6)
    pi_results = pi.run(verbose=True)
    
    # Run Value Iteration
    print("\n" + "=" * 70)
    print("  RUNNING VALUE ITERATION")
    print("=" * 70)
    vi = ValueIteration(gamma=0.9, theta=1e-6)
    vi_results = vi.run(verbose=True)
    
    # Save results
    np.save('results/policy_iteration/V.npy', pi_results['V'])
    np.save('results/policy_iteration/Q.npy', pi_results['Q'])
    np.save('results/policy_iteration/policy.npy', pi_results['policy'])
    
    np.save('results/value_iteration/V.npy', vi_results['V'])
    np.save('results/value_iteration/Q.npy', vi_results['Q'])
    np.save('results/value_iteration/policy.npy', vi_results['policy'])
    
    print("\nSaved numpy arrays to results/")
    
    # Generate visualizations
    print("\n" + "=" * 70)
    print("  GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    plot_convergence(pi_results['history'], 'Policy Iteration', 
                    'results/policy_iteration/convergence.png')
    plot_convergence(vi_results['history'], 'Value Iteration',
                    'results/value_iteration/convergence.png')
    compare_algorithms(pi_results, vi_results, 'results/comparison/comparison.png')
    
    # Print policies
    print("\n" + "=" * 70)
    print("  LEARNED POLICIES")
    print("=" * 70)
    
    print("\n--- Policy Iteration ---")
    pi.print_policy()
    
    print("\n--- Value Iteration ---")
    vi.print_policy()
    
    # Simulate episodes
    print("\n" + "=" * 70)
    print("  EPISODE SIMULATIONS")
    print("=" * 70)
    
    print("\n--- Using Policy Iteration Policy ---")
    for disease in [0, 1]:
        simulate_episode(pi, pi_results['policy'], disease=disease, verbose=True)
    
    print("\n--- Using Value Iteration Policy ---")
    for disease in [0, 1]:
        simulate_episode(vi, vi_results['policy'], disease=disease, verbose=True)
    
    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"\nPolicy Iteration:")
    print(f"  - Converged in {pi_results['iterations']} iterations")
    print(f"  - V(s=0) = {pi_results['V'][0]:.4f}")
    print(f"\nValue Iteration:")
    print(f"  - Converged in {vi_results['iterations']} iterations")  
    print(f"  - V(s=0) = {vi_results['V'][0]:.4f}")
    
    # Verify correctness
    print("\n" + "=" * 70)
    print("  VERIFICATION")
    print("=" * 70)
    
    # Both algorithms should give same result
    v_diff = np.max(np.abs(pi_results['V'] - vi_results['V']))
    print(f"\nMax V difference between algorithms: {v_diff:.8f}")
    
    policy_match = (pi_results['policy'] == vi_results['policy']).mean()
    print(f"Policy agreement: {100*policy_match:.1f}%")
    
    # Expected behavior: agent asks ONE symptom, then diagnoses correctly
    print(f"\nExpected V(s=0) ≈ 0.19 + 0.9 * 9.99 = 9.18 (one ask, then correct diagnosis)")
    print(f"Actual V(s=0) = {pi_results['V'][0]:.4f}")
    
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE!")
    print("  Results saved to: results/")
    print("  Run 'streamlit run app.py' for interactive visualization")
    print("=" * 70 + "\n")
    
    return pi_results, vi_results


if __name__ == "__main__":
    main()

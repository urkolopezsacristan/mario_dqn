"""
evaluate.py — Evaluate and compare Default vs Optimized agents.

Runs N episodes per agent, computes statistics,
performs Welch's t-test and Mann-Whitney U, generates plots.

Usage:
    python evaluate.py                                       # Compare both agents
    python evaluate.py --episodes 10                         # Quick test
    python evaluate.py --default_model models/mario_model.pth  # Single agent
"""

import argparse
import os
import json
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

from src.wrappers import apply_wrappers
from src.agent import Agent
from src.config import load_config


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def evaluate_agent(model_path, env_name, n_episodes=30, label="Agent"):
    """Run n_episodes. Returns per-episode rewards."""
    env = gym_super_mario_bros.make(env_name, render_mode=None, apply_api_compatibility=True)
    env = JoypadSpace(env, RIGHT_ONLY)
    env = apply_wrappers(env, video_folder=None)

    agent = Agent(input_dims=env.observation_space.shape, num_actions=env.action_space.n)
    agent.load(model_path)
    agent.epsilon = 0.05  ## Not-greedy

    rewards = []
    for ep in range(n_episodes):
        try:
            state, _ = env.reset()
        except Exception:
            try:
                env.close()
            except Exception:
                pass
            env = gym_super_mario_bros.make(env_name, render_mode=None, apply_api_compatibility=True)
            env = JoypadSpace(env, RIGHT_ONLY)
            env = apply_wrappers(env, video_folder=None)
            state, _ = env.reset()

        done, total_reward = False, 0
        while not done:
            action = agent.choose_action(state)
            new_state, reward, done, truncated, info = env.step(action)
            done = done or truncated
            state = new_state
            total_reward += reward

        rewards.append(total_reward)
        print(f"  [{label}] Ep {ep+1}/{n_episodes} — Reward: {total_reward:.1f}")

    try:
        env.close()
    except Exception:
        pass
    return rewards


# ─────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────

def compute_statistics(rewards, label="Agent"):
    """Compute and print descriptive statistics."""
    s = {
        "label": label,
        "n": len(rewards),
        "mean": np.mean(rewards),
        "std": np.std(rewards, ddof=1),
        "median": np.median(rewards),
        "min": np.min(rewards),
        "max": np.max(rewards),
        "q1": np.percentile(rewards, 25),
        "q3": np.percentile(rewards, 75),
    }
    print(f"\n  {label}: mean={s['mean']:.1f} ± {s['std']:.1f} | "
          f"median={s['median']:.1f} | range=[{s['min']:.1f}, {s['max']:.1f}]")
    return s


def statistical_comparison(rewards_a, rewards_b, alpha=0.05):
    """Welch's t-test + Mann-Whitney U + Cohen's d."""
    t_stat, t_p = stats.ttest_ind(rewards_a, rewards_b, equal_var=False)
    u_stat, u_p = stats.mannwhitneyu(rewards_a, rewards_b, alternative="two-sided")

    n1, n2 = len(rewards_a), len(rewards_b)
    pooled = np.sqrt(((n1 - 1) * np.std(rewards_a, ddof=1)**2 +
                      (n2 - 1) * np.std(rewards_b, ddof=1)**2) / (n1 + n2 - 2))
    d = (np.mean(rewards_b) - np.mean(rewards_a)) / pooled if pooled > 0 else 0

    results = {
        "welch_t": t_stat, "welch_p": t_p, "welch_sig": t_p < alpha,
        "mwu_u": u_stat, "mwu_p": u_p, "mwu_sig": u_p < alpha,
        "cohens_d": d, "alpha": alpha,
    }

    print(f"\n{'=' * 60}")
    print(f"  STATISTICAL COMPARISON (α={alpha})")
    print(f"{'=' * 60}")
    print(f"  Welch's t-test:    t={t_stat:.3f}, p={t_p:.6f} {'✓' if t_p < alpha else '✗'}")
    print(f"  Mann-Whitney U:    U={u_stat:.1f}, p={u_p:.6f} {'✓' if u_p < alpha else '✗'}")
    effect = "negligible" if abs(d) < 0.2 else "small" if abs(d) < 0.5 else "medium" if abs(d) < 0.8 else "large"
    print(f"  Cohen's d:         {d:.3f} ({effect})")
    print(f"{'=' * 60}")

    return results


# ─────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────

def plot_boxplot(rewards_dict, output_dir):
    """Box plot comparing reward distributions."""
    fig, ax = plt.subplots(figsize=(8, 6))
    labels, data = list(rewards_dict.keys()), list(rewards_dict.values())
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
    for patch, color in zip(bp['boxes'], ['#3498db', '#2ecc71']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Total Reward")
    ax.set_title("Reward Distribution — Default vs Optimized")
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "boxplot_comparison.png"), dpi=150)
    plt.close()


def plot_episode_rewards(rewards_dict, output_dir):
    """Per-episode reward scatter plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for (label, rewards), color in zip(rewards_dict.items(), ['#3498db', '#2ecc71']):
        eps = range(1, len(rewards) + 1)
        ax.plot(eps, rewards, 'o-', label=label, color=color, alpha=0.7, markersize=4)
        ax.axhline(np.mean(rewards), color=color, linestyle='--', alpha=0.5,
                   label=f"{label} mean ({np.mean(rewards):.1f})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Per-Episode Rewards — Greedy Evaluation")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "episode_rewards.png"), dpi=150)
    plt.close()


def plot_histogram(rewards_dict, output_dir):
    """Overlapping histogram of reward distributions."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for (label, rewards), color in zip(rewards_dict.items(), ['#3498db', '#2ecc71']):
        ax.hist(rewards, bins=15, alpha=0.5, label=label, color=color, edgecolor='black')
    ax.set_xlabel("Total Reward")
    ax.set_ylabel("Frequency")
    ax.set_title("Reward Distribution Histogram")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "histogram_comparison.png"), dpi=150)
    plt.close()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare Mario DQN agents")
    parser.add_argument("--default_model", type=str, default=None)
    parser.add_argument("--optimized_model", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    ec = config["evaluation"]

    n_episodes = args.episodes or ec["n_episodes"]
    alpha = ec["alpha"]
    default_model = args.default_model or os.path.join("models", ec["default_model"])
    optimized_model = args.optimized_model or os.path.join("models", ec["optimized_model"])
    output_dir = ec["output_dir"]
    env_name = config["environment"]["name"]

    os.makedirs(output_dir, exist_ok=True)

    rewards_dict = {}
    stats_all = {}

    # Evaluate agents
    for label, path in [("Default", default_model), ("Optimized", optimized_model)]:
        if os.path.exists(path):
            print(f"\n--- Evaluating {label} Agent ({path}) ---")
            rewards = evaluate_agent(path, env_name, n_episodes, label)
            rewards_dict[label] = rewards
            stats_all[label] = compute_statistics(rewards, f"{label} Agent")
        else:
            print(f"  Skipping {label}: {path} not found")

    # Plots
    if rewards_dict:
        print("\n--- Generating Plots ---")
        plot_boxplot(rewards_dict, output_dir)
        plot_episode_rewards(rewards_dict, output_dir)
        if len(rewards_dict) > 1:
            plot_histogram(rewards_dict, output_dir)
        print(f"  Plots saved to {output_dir}/")

    # Statistical comparison
    comparison = None
    if "Default" in rewards_dict and "Optimized" in rewards_dict:
        comparison = statistical_comparison(
            rewards_dict["Default"], rewards_dict["Optimized"], alpha
        )
        diff = np.mean(rewards_dict["Optimized"]) - np.mean(rewards_dict["Default"])
        if comparison["welch_sig"]:
            word = "better" if diff > 0 else "worse"
            print(f"\n  CONCLUSION: Optimized agent is statistically {word} ({diff:+.1f})")
        else:
            print(f"\n  CONCLUSION: No significant difference ({diff:+.1f})")

    # Save JSON
    results = {"stats": stats_all, "comparison": comparison}
    with open(os.path.join(output_dir, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\nDone!")


if __name__ == "__main__":
    main()

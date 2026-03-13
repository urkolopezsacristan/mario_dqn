"""
optimize.py — Optuna hyperparameter optimization for the Mario Double DQN agent.

Usage:
    python optimize.py                                    # 10 trials, 500 episodes
    python optimize.py --n_trials 5 --n_episodes 100      # Quick test
"""

import argparse
import json
import numpy as np

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

from src.wrappers import apply_wrappers
from src.config import load_config, create_agent_with_params


def sample_dqn_params(trial, search_space):
    """Sample hyperparameters for the Double DQN agent."""
    return {
        "lr": trial.suggest_float("lr", search_space["lr"][0], search_space["lr"][1], log=True),
        "gamma": trial.suggest_float("gamma", search_space["gamma"][0], search_space["gamma"][1]),
        "eps_decay": trial.suggest_float("eps_decay", search_space["eps_decay"][0], search_space["eps_decay"][1], log=True),
        "batch_size": trial.suggest_categorical("batch_size", search_space["batch_size"]),
        "sync_network_rate": trial.suggest_categorical("sync_network_rate", search_space["sync_network_rate"]),
        "replay_buffer_capacity": trial.suggest_categorical("replay_buffer_capacity", search_space["replay_buffer_capacity"]),
    }


def objective(trial, config, n_episodes):
    """Train agent for n_episodes; return avg reward over last 10."""
    print(f"\n[TRIAL {trial.number}] Starting with sampled parameters...")
    env_name = config["environment"]["name"]
    env = gym_super_mario_bros.make(env_name, render_mode=None, apply_api_compatibility=True)
    env = JoypadSpace(env, RIGHT_ONLY)
    env = apply_wrappers(env, video_folder=None)

    params = sample_dqn_params(trial, config["optuna"]["search_space"])
    params["epsilon"] = 0.9
    params["eps_min"] = 0.1
    agent = create_agent_with_params(env, params)

    rewards_history = []
    learn_every = config["training"].get("learn_every_n_steps", 4)

    try:
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

            done = False
            total_reward = 0
            episode_steps = 0

            while not done:
                action = agent.choose_action(state)
                new_state, reward, done, truncated, info = env.step(action)
                done = done or truncated
                agent.store_in_memory(state, action, reward, new_state, done)
                if episode_steps % learn_every == 0:
                    agent.learn()
                state = new_state
                total_reward += reward
                episode_steps += 1

            rewards_history.append(total_reward)

            # Report to Optuna for pruning
            if ep > 0 and ep % 50 == 0:
                recent_avg = np.mean(rewards_history[-50:])
                trial.report(recent_avg, ep)
                if trial.should_prune():
                    print(f"  [TRIAL {trial.number}] Pruned at episode {ep} (Avg: {recent_avg:.1f})")
                    raise optuna.exceptions.TrialPruned()

            if ep % 10 == 0:
                avg = np.mean(rewards_history[-10:]) if len(rewards_history) >= 10 else np.mean(rewards_history)
                print(f"  [TRIAL {trial.number}] Ep {ep}/{n_episodes} | Reward: {total_reward:6.1f} | Avg(10): {avg:6.1f} | Eps: {agent.epsilon:.4f}")

    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        print(f"  Trial {trial.number} failed: {e}")
        return float("nan")
    finally:
        try:
            env.close()
        except Exception:
            pass

    n_eval = min(50, len(rewards_history))
    return np.mean(rewards_history[-n_eval:])


def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter optimization for Mario DQN")
    parser.add_argument("--n_trials", type=int, default=None)
    parser.add_argument("--n_episodes", type=int, default=None)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    n_trials = args.n_trials or config["optuna"]["n_trials"]
    n_episodes = args.n_episodes or config["optuna"]["n_episodes_per_trial"]
    study_name = config["optuna"]["study_name"]
    storage = config["optuna"]["storage"]
    best_params_file = config["optuna"]["best_params_file"]

    print("=" * 60)
    print("  OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)
    print(f"  Trials:         {n_trials}")
    print(f"  Episodes/trial: {n_episodes}")
    print(f"  Study:          {study_name}")
    print("=" * 60)

    sampler = TPESampler(n_startup_trials=5, seed=42)
    pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=2)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=True
    )

    study.optimize(
        lambda trial: objective(trial, config, n_episodes),
        n_trials=n_trials,
        show_progress_bar=True
    )

    # Results
    print(f"\n{'=' * 60}")
    print(f"  BEST TRIAL: {study.best_trial.number}")
    print(f"  BEST VALUE: {study.best_value:.2f}")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")
    print(f"{'=' * 60}")

    # Save best params
    best_params = study.best_params.copy()
    best_params["epsilon"] = 0.9
    best_params["eps_min"] = 0.1
    with open(best_params_file, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)
    print(f"\n  Saved to: {best_params_file}")
    print(f"  Next: python train.py --optimized")


if __name__ == "__main__":
    main()

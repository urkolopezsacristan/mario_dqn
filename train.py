"""
train.py — Unified training script for the Mario Double DQN agent.

Usage:
    python train.py                     # Train with default hyperparams
    python train.py --optimized         # Train with Optuna best params
    python train.py --episodes 1000     # Custom episode count
"""

import argparse
import json
import os
import time
import matplotlib.pyplot as plt

import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

from src.wrappers import apply_wrappers
from src.config import load_config, get_default_hyperparams, create_agent_with_params


def create_env(env_name, episode_idx, video_interval, optimized=False, render_mode=None):
    """Create the Mario environment with wrappers."""
    # Only record video if it's the right episode and we're not manually rendering
    should_record = (episode_idx % video_interval == 0) and (render_mode != "human")
    video_dir = None
    if should_record:
        video_dir = "recordings_optimized" if optimized else "recordings"
        # RecordVideo requires a render mode to capture frames
        render_mode = render_mode or "rgb_array"

    env = gym_super_mario_bros.make(env_name, render_mode=render_mode, apply_api_compatibility=True)
    env = JoypadSpace(env, RIGHT_ONLY)
    env = apply_wrappers(env, video_folder=video_dir)
    return env


def main():
    parser = argparse.ArgumentParser(description="Train Mario Double DQN agent")
    parser.add_argument("--optimized", action="store_true",
                        help="Train with Optuna-optimized hyperparameters")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Number of training episodes (default: from config.yaml)")
    parser.add_argument("--render", action="store_true",
                        help="Render the environment (much slower)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.yaml")
    args = parser.parse_args()

    render_mode = "human" if args.render else None
    config = load_config(args.config) if args.config else load_config()
    env_name = config["environment"]["name"]
    num_episodes = args.episodes or config["training"]["num_episodes"]
    save_interval = config["training"]["save_interval"]
    learn_every = config["training"].get("learn_every_n_steps", 4)
    video_interval = config["training"]["video_interval"]

    # Determine hyperparameters and model path
    if args.optimized:
        params_file = config["optuna"]["best_params_file"]
        model_path = os.path.join("models", "mario_model_optimized.pth")
        progress_img = os.path.join("results", "progreso_mario_optimized.png")

        if not os.path.exists(params_file):
            print(f"ERROR: {params_file} not found. Run optimize.py first.")
            return

        with open(params_file, "r", encoding="utf-8") as f:
            params = json.load(f)
        label = "Optimized"
    else:
        params = get_default_hyperparams(config)
        model_path = os.path.join("models", config["training"]["model_path"])
        progress_img = os.path.join("results", "progreso_mario.png")
        label = "Default"

    # Ensure output directories exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print("=" * 60)
    print(f"  TRAINING {label.upper()} AGENT")
    print("=" * 60)
    print(f"  Episodes:  {num_episodes}")
    print(f"  Model:     {model_path}")
    print(f"  Params:")
    for k, v in params.items():
        print(f"    {k}: {v}")
    print("=" * 60)

    # Create initial agent and env
    env = create_env(env_name, 0, video_interval, args.optimized, render_mode)
    agent = create_agent_with_params(env, params)

    # Resume from checkpoint if available
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"  Resumed from {model_path}")

    rewards_history = []

    def safe_close(target_env):
        """Safely close environment without crashing on nes_py double-close."""
        try:
            if hasattr(target_env, 'unwrapped') and target_env.unwrapped.viewer is not None:
                target_env.close()
            else:
                target_env.close()
        except:
            pass

    try:
        for i in range(num_episodes):
            # Record current epsilon to restore after recreation
            current_eps = agent.epsilon
            
            # Check if we need to recreate the environment for video recording
            if i > 0 and i % video_interval == 0:
                print(f"\nRecreating environment for video recording (Episode {i})...")
                safe_close(env)
                env = create_env(env_name, i, video_interval, args.optimized, render_mode)
                agent.epsilon = current_eps
            elif i > 0 and (i - 1) % video_interval == 0:
                # Close video env and return to fast mode
                print(f"\nReturning to fast mode (Episode {i})...")
                safe_close(env)
                env = create_env(env_name, i, video_interval, args.optimized, render_mode)
                agent.epsilon = current_eps

            try:
                state, _ = env.reset()
            except Exception as e:
                print(f"\nEnvironment error at episode {i}: {e}. Attempting restart...")
                safe_close(env)
                time.sleep(1) 
                env = create_env(env_name, i, video_interval, args.optimized, render_mode)
                agent.epsilon = current_eps
                state, _ = env.reset()

            done = False
            total_reward = 0
            episode_steps = 0
            start_time = time.time()

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

            end_time = time.time()
            sps = episode_steps / (end_time - start_time)
            rewards_history.append(total_reward)

            print(f"Ep: {i} | Reward: {total_reward:6.1f} | SPS: {sps:4.1f} | Eps: {agent.epsilon:.4f}")

            if i % save_interval == 0 and i > 0:
                agent.save(model_path)
                plt.clf()
                plt.plot(rewards_history)
                plt.title(f"{label} Agent — Episode {i}")
                plt.xlabel("Episodes")
                plt.ylabel("Total Reward")
                plt.savefig(progress_img)
                plt.close()

    except KeyboardInterrupt:
        print("\nTraining interrupted.")
        agent.save(model_path)

    finally:
        print("Saving final model and closing environment...")
        agent.save(model_path)
        try:
            env.close()
        except Exception:
            pass
        print("Done.")


if __name__ == "__main__":
    main()

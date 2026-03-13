"""
config.py — Loads hyperparameters from config.yaml.

Provides helpers to create agents with custom parameter overrides.
"""

import yaml
import os
import torch

from src.agent import Agent
from torchrl.data import TensorDictReplayBuffer, ListStorage


CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")


def load_config(path=None):
    """Load the full configuration dictionary from config.yaml."""
    path = path or CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_default_hyperparams(config=None):
    """Return the default hyperparameters dictionary."""
    if config is None:
        config = load_config()
    return config["default_hyperparams"]


def create_agent_with_params(env, params=None):
    """
    Create an Agent and override its hyperparameters.

    Parameters
    ----------
    env : gym.Env
        Environment (for observation/action shapes).
    params : dict, optional
        Hyperparameters to override. If None, uses defaults from config.yaml.

    Returns
    -------
    Agent
    """
    if params is None:
        params = get_default_hyperparams()

    agent = Agent(
        input_dims=env.observation_space.shape,
        num_actions=env.action_space.n
    )

    # Override agent hyperparameters
    for key, value in params.items():
        if hasattr(agent, key):
            setattr(agent, key, value)

    # Rebuild optimizer if learning rate was changed
    if "lr" in params:
        agent.optimizer = torch.optim.Adam(
            agent.online_network.parameters(), lr=params["lr"]
        )

    # Rebuild replay buffer if capacity was changed
    if "replay_buffer_capacity" in params:
        storage = ListStorage(int(params["replay_buffer_capacity"]))
        agent.replay_buffer = TensorDictReplayBuffer(storage=storage)

    return agent

"""
agent.py — Double DQN agent with CNN for Super Mario Bros.

Contains:
    - AgentNN: Convolutional neural network for Q-value estimation.
    - Agent: Double DQN agent with experience replay and epsilon-greedy exploration.
"""

import torch
from torch import nn
import numpy as np
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, ListStorage


# ─────────────────────────────────────────────
# Neural Network
# ─────────────────────────────────────────────

class AgentNN(nn.Module):
    """CNN architecture for processing stacked grayscale frames."""

    def __init__(self, input_shape, n_actions, freeze=False):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.network = nn.Sequential(
            self.conv_layers,
            nn.Flatten(),
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

        if freeze:
            self._freeze()

        self.to(self.device)

    def _get_conv_out(self, shape):
        o = self.conv_layers(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def _freeze(self):
        for p in self.network.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.network(x)


# ─────────────────────────────────────────────
# Double DQN Agent
# ─────────────────────────────────────────────

class Agent:
    """
    Double DQN agent with experience replay and epsilon-greedy exploration.

    Attributes
    ----------
    lr : float
        Learning rate for Adam optimizer.
    gamma : float
        Discount factor.
    epsilon : float
        Current exploration rate.
    eps_decay : float
        Multiplicative epsilon decay per learn step.
    eps_min : float
        Minimum epsilon value.
    batch_size : int
        Mini-batch size for training.
    sync_network_rate : int
        Steps between target network syncs.
    """

    def __init__(self, input_dims, num_actions):
        self.num_actions = num_actions
        self.learn_step_counter = 0

        self.lr = 0.0001
        self.gamma = 0.99
        self.epsilon = 0.9
        self.eps_decay = 0.9999975
        self.eps_min = 0.1
        self.batch_size = 256
        self.sync_network_rate = 10000

        self.online_network = AgentNN(input_dims, num_actions)
        self.target_network = AgentNN(input_dims, num_actions, freeze=True)

        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=self.lr)
        self.loss = torch.nn.MSELoss()

        replay_buffer_capacity = 100000
        storage = ListStorage(replay_buffer_capacity)
        self.replay_buffer = TensorDictReplayBuffer(storage=storage)

    def choose_action(self, observation):
        """Select action using epsilon-greedy policy."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        observation = torch.tensor(np.array(observation), dtype=torch.float32).unsqueeze(0).to(self.online_network.device)
        observation = observation.squeeze(-1) / 255.0
        return self.online_network(observation).argmax().item()

    def decay_epsilon(self):
        """Decay epsilon by multiplicative factor."""
        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)

    def store_in_memory(self, state, action, reward, next_state, done):
        """Store a transition in the replay buffer."""
        self.replay_buffer.add(TensorDict({
            "state": torch.tensor(np.array(state), dtype=torch.uint8),
            "action": torch.tensor(action),
            "reward": torch.tensor(reward),
            "next_state": torch.tensor(np.array(next_state), dtype=torch.uint8),
            "done": torch.tensor(done)
        }, batch_size=[]))

    def sync_networks(self):
        """Copy online network weights to target network."""
        if self.learn_step_counter % self.sync_network_rate == 0 and self.learn_step_counter > 0:
            self.target_network.load_state_dict(self.online_network.state_dict())

    def learn(self):
        """Perform one training step using Double DQN."""
        if len(self.replay_buffer) < self.batch_size:
            return

        self.sync_networks()
        self.optimizer.zero_grad()

        samples = self.replay_buffer.sample(self.batch_size).to(self.online_network.device)
        keys = ("state", "action", "reward", "next_state", "done")
        states, actions, rewards, next_states, dones = [samples[key] for key in keys]

        states = states.squeeze(-1) / 255.0
        next_states = next_states.squeeze(-1) / 255.0

        predicted_q_values = self.online_network(states)
        predicted_q_values = predicted_q_values[np.arange(self.batch_size), actions.squeeze()]

        with torch.no_grad():
            next_q_online = self.online_network(next_states)
            best_actions = next_q_online.argmax(dim=1, keepdim=True)

            next_q_target = self.target_network(next_states)
            target_q_values = next_q_target.gather(1, best_actions).squeeze()
            target_q_values = rewards + self.gamma * target_q_values.detach() * (1 - dones.float())

        loss = self.loss(predicted_q_values, target_q_values)
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        self.decay_epsilon()

    def save(self, filename):
        """Save model checkpoint."""
        torch.save({
            'online_model_state_dict': self.online_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_counter': self.learn_step_counter
        }, filename)

    def load(self, filename):
        """Load model checkpoint."""
        checkpoint = torch.load(filename, map_location=self.online_network.device)
        self.online_network.load_state_dict(checkpoint["online_model_state_dict"])
        self.target_network.load_state_dict(checkpoint["online_model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
        self.learn_step_counter = checkpoint["step_counter"]

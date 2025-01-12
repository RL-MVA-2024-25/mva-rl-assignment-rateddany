import os
import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

# Define the environment
env = TimeLimit(env=HIVPatient(domain_randomization=False), max_episode_steps=200)

# Define the Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.buffer = []
        self.pointer = 0
        self.device = device

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pointer] = (state, action, reward, next_state, done)
        self.pointer = (self.pointer + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32, device=self.device),
            torch.tensor(actions, dtype=torch.long, device=self.device),
            torch.tensor(rewards, dtype=torch.float32, device=self.device),
            torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device),
            torch.tensor(dones, dtype=torch.float32, device=self.device),
        )

    def __len__(self):
        return len(self.buffer)


# Define the Q-Network
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_units=256):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, output_dim)
        )

    def forward(self, x):
        return self.network(x)


# Define the Agent
class ProjectAgent:
    def __init__(self):
        self.model = None
        self.target_model = None
        self.memory = None

    def act(self, observation, use_random=False):
        if self.model is None:
            raise ValueError("Model is not initialized!")

        if use_random or random.random() < 0.1:  # Epsilon-greedy exploration
            return env.action_space.sample()

        device = next(self.model.parameters()).device
        state_tensor = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def save(self, path):
        if self.model is None:
            raise ValueError("Model is not initialized!")

        model_path = os.path.join(path, "dqn_model.pt")
        torch.save({
            "model_state": self.model.state_dict(),
            "target_state": self.target_model.state_dict()
        }, model_path)
        print(f"Model saved at {model_path}")

    def load(self):
        device = torch.device("cpu")
        model_path = os.path.join(os.getcwd(), "dqn_model.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        checkpoint = torch.load(model_path, map_location=device)
        self.model = QNetwork(env.observation_space.shape[0], env.action_space.n).to(device)
        self.target_model = deepcopy(self.model).to(device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.target_model.load_state_dict(checkpoint["target_state"])
        print(f"Model loaded from {model_path}")

    def train(self):
        config = {
            "learning_rate": 0.001,
            "gamma": 0.98,
            "buffer_capacity": 75000,
            "batch_size": 128,
            "epsilon_start": 1.0,
            "epsilon_min": 0.02,
            "epsilon_decay_steps": 20000,
            "target_update_freq": 1000,
            "max_episodes": 500,
        }

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on {device}")

        self.model = QNetwork(env.observation_space.shape[0], env.action_space.n).to(device)
        self.target_model = deepcopy(self.model).to(device)
        self.memory = ReplayBuffer(config["buffer_capacity"], device)

        optimizer = optim.Adam(self.model.parameters(), lr=config["learning_rate"])
        loss_fn = nn.SmoothL1Loss()

        epsilon = config["epsilon_start"]
        epsilon_decay = (epsilon - config["epsilon_min"]) / config["epsilon_decay_steps"]

        for episode in range(config["max_episodes"]):
            state, _ = env.reset()
            done = False
            episode_reward = 0

            while not done:
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = self.act(state)

                next_state, reward, done, truncated, _ = env.step(action)
                terminal = done or truncated

                self.memory.add(state, action, reward, next_state, terminal)
                episode_reward += reward
                state = next_state

                if len(self.memory) >= config["batch_size"]:
                    states, actions, rewards, next_states, dones = self.memory.sample(config["batch_size"])
                    q_current = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                    with torch.no_grad():
                        q_next = self.target_model(next_states).max(1)[0]
                        q_target = rewards + config["gamma"] * q_next * (1 - dones)
                    loss = loss_fn(q_current, q_target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if terminal:
                    break

            epsilon = max(config["epsilon_min"], epsilon - epsilon_decay)

            if episode % config["target_update_freq"] == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            print(f"Episode {episode + 1}/{config['max_episodes']} | "
                  f"Epsilon: {epsilon:.3f} | Reward: {episode_reward:.2f}")

        self.save(os.getcwd())
        print("Training complete.")

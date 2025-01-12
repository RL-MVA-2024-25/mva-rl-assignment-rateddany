from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

import os
import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from tqdm import trange

from evaluate import evaluate_HIV



env = TimeLimit(
    env=HIVPatient(domain_randomization=True), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!


class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.data = []
        self.index = 0
        self.device = device

    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), device=self.device, dtype=torch.float32),
            torch.tensor(actions, device=self.device, dtype=torch.long),
            torch.tensor(rewards, device=self.device, dtype=torch.float32),
            torch.tensor(np.array(next_states), device=self.device, dtype=torch.float32),
            torch.tensor(dones, device=self.device, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.data)

class ProjectAgent:
    def __init__(self):
        self.gamma = None
        self.batch_size = None
        self.model = None
        self.target_model = None
        self.memory = None
        self.best_model = None
        self.path = None
        self.optimizer = None
        self.criterion = None

    def act(self, observation):
        """
        Select an action based on the current policy.
        """
        device = "cuda" if next(self.model.parameters()).is_cuda else "cpu"
        with torch.no_grad():
            Q = self.model(torch.Tensor(observation).unsqueeze(0).to(device))
            return torch.argmax(Q).item()

    def save(self, path):
        """
        Save the trained model to a file.
        """
        self.path = path + "/dqn_best.pt"
        torch.save(self.model.state_dict(), self.path)
        # print(f"Model saved to {self.path}")

    def load(self):
        """
        Load a trained model from a file.
        """
        device = torch.device('cpu')
        self.path = os.getcwd() + "/dqn_best.pt"
        print(f"Loading model from: {self.path}")
        self.model = self.dqn({}, device)
        self.model.load_state_dict(torch.load(self.path, map_location=device))
        self.model.eval()
        return

    def dqn(self, config, device):
        """
        Define the neural network architecture.
        """
        state_dim = env.observation_space.shape[0]
        n_action = env.action_space.n
        nb_neurons = 256

        return nn.Sequential(
            nn.Linear(state_dim, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, n_action),
        ).to(device)

    def gradient_step(self):
        """
        Perform one step of gradient update on the Q-network.
        """
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = R + (1 - D) * self.gamma * QYmax
            QXA = self.model(X).gather(1, A.unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(self):
        """
        Train the agent using Deep Q-Learning.
        """
        config = {
            'nb_actions': env.action_space.n,
            'learning_rate': 0.001,
            'gamma': 0.98,
            'buffer_size': 100000,
            'epsilon_min': 0.02,
            'epsilon_max': 1.0,
            'epsilon_decay_period': 21000,
            'epsilon_delay_decay': 100,
            'batch_size': 64,
            'gradient_steps': 1,
            'update_target_strategy': 'replace',
            'update_target_freq': 1000,
            'criterion': nn.SmoothL1Loss(),
        }

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {device}')

        self.model = self.dqn(config, device)
        self.target_model = deepcopy(self.model).to(device)
        self.gamma = config['gamma']
        self.batch_size = config['batch_size']
        self.nb_actions = config['nb_actions']
        self.memory = ReplayBuffer(config['buffer_size'], device)
        self.criterion = config['criterion']
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])

        epsilon_max = config['epsilon_max']
        epsilon_min = config['epsilon_min']
        epsilon_stop = config['epsilon_decay_period']
        epsilon_delay = config['epsilon_delay_decay']
        epsilon_step = (epsilon_max - epsilon_min) / epsilon_stop

        nb_gradient_steps = config['gradient_steps']
        update_target_freq = config['update_target_freq']

        max_episode = 500
        episode_return = []
        episode = 0
        step = 0
        previous_val = 0

        state, _ = env.reset()
        epsilon = epsilon_max
        episode_cum_reward = 0

        while episode < max_episode:
            if step > epsilon_delay:
                epsilon = max(epsilon_min, epsilon - epsilon_step)

            # Select action using epsilon-greedy policy
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = self.act(state)

            # Step in the environment
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward

            # Gradient steps
            for _ in range(nb_gradient_steps):
                self.gradient_step()

            # Update target network
            if step % update_target_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            step += 1

            if done or trunc:
                episode += 1

                # Evaluate validation score
                validation_score = evaluate_HIV(agent=self, nb_episode=1)

                print(f"Episode {episode:3d} | "
                      f"Epsilon {epsilon:6.2f} | "
                      f"Batch Size {len(self.memory):5d} | "
                      f"Episode Return {episode_cum_reward:.2e} | "
                      f"Evaluation Score {validation_score:.2e}")

                # Save best model
                if validation_score > previous_val:
                    previous_val = validation_score
                    self.best_model = deepcopy(self.model).to(device)
                    self.save(os.getcwd())

                state, _ = env.reset()
                episode_return.append(episode_cum_reward)
                episode_cum_reward = 0
            else:
                state = next_state

        # Load the best model
        self.model.load_state_dict(self.best_model.state_dict())
        self.save(os.getcwd())
        return episode_return

if __name__ == "__main__":
    agent = ProjectAgent()
    agent.train()
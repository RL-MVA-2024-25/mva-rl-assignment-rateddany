import os
import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.wrappers import TimeLimit
from tqdm import trange

from env_hiv import HIVPatient
from evaluate import evaluate_HIV

# Environment
env = TimeLimit(env=HIVPatient(domain_randomization=True), max_episode_steps=200)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.data = []
        self.index = 0
        self.device = device

    def append(self, s, a, r, s_next, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_next, d)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        states, actions, rewards, nxt_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), device=self.device, dtype=torch.float32),
            torch.tensor(actions, device=self.device, dtype=torch.long),
            torch.tensor(rewards, device=self.device, dtype=torch.float32),
            torch.tensor(np.array(nxt_states), device=self.device, dtype=torch.float32),
            torch.tensor(dones, device=self.device, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.data)

class ProjectAgent:
    def act(self, observation, use_random=False):
        dev = "cuda" if next(self.model.parameters()).is_cuda else "cpu"
        with torch.no_grad():
            q_val = self.model(torch.Tensor(observation).unsqueeze(0).to(dev))
            return torch.argmax(q_val).item()

    def save(self, path):
        self.path = path + "/model_best.pt"
        torch.save(self.model.state_dict(), self.path)
        return

    def load(self):
        dev = torch.device('cpu')
        self.path = os.getcwd() + "/model_best.pt"
        self.model = self.myDQN({}, dev)
        self.model.load_state_dict(torch.load(self.path, map_location=dev))
        self.model.eval()
        return

    def act_greedy(self, myDQN, state):
        dev = "cuda" if next(myDQN.parameters()).is_cuda else "cpu"
        with torch.no_grad():
            qv = myDQN(torch.Tensor(state).unsqueeze(0).to(dev))
            return torch.argmax(qv).item()

    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            maxQY = self.target_model(Y).max(1)[0].detach()
            update_val = torch.addcmul(R, 1 - D, maxQY, value=self.gamma)
            pred_q = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(pred_q, update_val.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def myDQN(self, config, device):
        s_dim = env.observation_space.shape[0]
        act_num = env.action_space.n
        hidden = 256
        net = torch.nn.Sequential(
            nn.Linear(s_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_num)
        ).to(device)
        return net

    def train(self):
        config = {
            'nb_actions': env.action_space.n,
            'learning_rate': 0.001,
            'gamma': 0.98,
            'buffer_size': 50000,
            'epsilon_min': 0.02,
            'epsilon_max': 1.,
            'epsilon_decay_period': 21000,
            'epsilon_delay_decay': 100,
            'batch_size': 64,
            'gradient_steps': 1,
            'update_target_strategy': 'replace',
            'update_target_freq': 1000,
            'update_target_tau': 0.005,
            'criterion': torch.nn.SmoothL1Loss()
        }

        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', dev)
        self.model = self.myDQN(config, dev)
        self.target_model = deepcopy(self.model).to(dev)

        self.gamma = config['gamma']
        self.batch_size = config['batch_size']
        self.nb_actions = config['nb_actions']

        epsilon_max = config['epsilon_max']
        epsilon_min = config['epsilon_min']
        eps_decay = config['epsilon_decay_period']
        eps_delay = config['epsilon_delay_decay']
        e_step = (epsilon_max - epsilon_min) / eps_decay

        self.memory = ReplayBuffer(config['buffer_size'], dev)

        self.criterion = config['criterion']
        learning_rate = config['learning_rate']
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        nb_gsteps = config['gradient_steps']
        tgt_freq = config['update_target_freq']

        prev_val = 0
        max_ep = 500
        ep_returns = []
        ep_count = 0
        ep_cum_reward = 0

        state, _ = env.reset()
        epsilon = epsilon_max
        step = 0

        while ep_count < max_ep:
            if step > eps_delay:
                epsilon = max(epsilon_min, epsilon - e_step)
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = self.act_greedy(self.model, state)

            nxt_state, rew, done, trunc, _ = env.step(action)
            self.memory.append(state, action, rew, nxt_state, done)
            ep_cum_reward += rew

            for _ in range(nb_gsteps):
                self.gradient_step()

            if step % tgt_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            step += 1
            if done or trunc:
                ep_count += 1
                val_score = evaluate_HIV(agent=self, nb_episode=1)
                print(f"Episode {ep_count:3d} | "
                      f"Epsilon {epsilon:6.2f} | "
                      f"Batch Size {len(self.memory):5d} | "
                      f"Episode Return {ep_cum_reward:.2e} | "
                      f"Evaluation Score {val_score:.2e}")
                state, _ = env.reset()
                if val_score > prev_val:
                    prev_val = val_score
                    self.best_model = deepcopy(self.model).to(dev)
                    pth = os.getcwd()
                    self.save(pth)
                ep_returns.append(ep_cum_reward)
                ep_cum_reward = 0
            else:
                state = nxt_state

        self.model.load_state_dict(self.best_model.state_dict())
        final_path = os.getcwd()
        self.save(final_path)
        return ep_returns

if __name__ == "__main__":
    agent = ProjectAgent()
    agent.train()

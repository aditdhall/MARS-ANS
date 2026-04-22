import math
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MarsRoverEnv:
    def __init__(self, cost_map, start, goal):
        self.cost_map = np.asarray(cost_map, dtype=float)
        self.H, self.W = self.cost_map.shape
        self.start = (float(start[0]), float(start[1]))
        self.goal = (float(goal[0]), float(goal[1]))
        self.current_pos = [self.start[0], self.start[1]]
        self.max_dist = math.hypot(self.H, self.W)

    def reset(self):
        self.current_pos = [self.start[0], self.start[1]]
        return self.get_state()

    def get_state(self):
        r, c = self.current_pos
        ri, ci = int(np.clip(r, 0, self.H - 1)), int(np.clip(c, 0, self.W - 1))
        cost = float(self.cost_map[ri, ci])
        dist = math.hypot(r - self.goal[0], c - self.goal[1])
        return np.array([
            r / self.H,
            c / self.W,
            self.goal[0] / self.H,
            self.goal[1] / self.W,
            cost / 10.0,
            dist / self.max_dist,
        ], dtype=np.float32)

    def step(self, action):
        angle_deg = action * 45.0
        rad = math.radians(angle_deg)
        dr = -math.cos(rad) * 0.5
        dc = math.sin(rad) * 0.5
        prev_dist = math.hypot(self.current_pos[0] - self.goal[0], self.current_pos[1] - self.goal[1])
        new_r = self.current_pos[0] + dr
        new_c = self.current_pos[1] + dc

        if new_r < 0 or new_r >= self.H or new_c < 0 or new_c >= self.W:
            return self.get_state(), -10.0, True
        ri, ci = int(new_r), int(new_c)
        if self.cost_map[ri, ci] >= 999:
            return self.get_state(), -10.0, True

        self.current_pos = [new_r, new_c]
        new_dist = math.hypot(new_r - self.goal[0], new_c - self.goal[1])
        h_score = min(float(self.cost_map[ri, ci]) / 10.0, 1.0)
        reward = -1.0 - 5.0 * h_score + 10.0 * (prev_dist - new_dist)
        done = False
        if new_dist < 1.0:
            reward += 100.0
            done = True
        return self.get_state(), reward, done


class DoubleDQN:
    def __init__(self, state_dim=6, action_dim=8, hidden=128):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.online_net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )
        self.target_net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.replay_buffer = deque(maxlen=10000)
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.gamma = 0.99
        self.lr = 1e-4
        self.batch_size = 64
        self.tau = 0.005
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.lr)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            q = self.online_net(s)
            return int(q.argmax(dim=1).item())

    def store(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.as_tensor(np.array(states), dtype=torch.float32)
        actions = torch.as_tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.as_tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(1)

        q_vals = self.online_net(states).gather(1, actions)
        with torch.no_grad():
            next_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target = rewards + self.gamma * next_q * (1.0 - dones)
        loss = F.mse_loss(q_vals, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for tp, op in zip(self.target_net.parameters(), self.online_net.parameters()):
            tp.data.mul_(1 - self.tau).add_(self.tau * op.data)
        return float(loss.item())

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

import torch
import torch.optim as optim
import random
from collections import deque
from dqn_model import DQN

class Agent:
    def __init__(self):
        self.model = DQN()
        self.target = DQN()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.memory = deque(maxlen=5000)

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05

        self.actions = [1.5, 2.0, 2.5, 3.0, 5.0]

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, len(self.actions)-1)

        state = torch.FloatTensor(state)
        q_vals = self.model(state)
        return torch.argmax(q_vals).item()

    def remember(self, s, a, r, ns, done):
        self.memory.append((s, a, r, ns, done))

    def train(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)

        for s, a, r, ns, done in batch:
            s = torch.FloatTensor(s)
            ns = torch.FloatTensor(ns)

            target = r
            if not done:
                target += self.gamma * torch.max(self.target(ns)).item()

            current = self.model(s)
            current[a] = target

            loss = (self.model(s) - current).pow(2).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

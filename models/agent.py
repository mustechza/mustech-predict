import torch
import torch.optim as optim
import random
from collections import deque
from models.dqn import DQN
import config

class Agent:
    def __init__(self):
        self.model = DQN()
        self.target = DQN()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LR)

        self.memory = deque(maxlen=5000)

        self.gamma = config.GAMMA
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05

        self.actions = config.CASHOUT_ACTIONS

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, len(self.actions)-1)
            
state = torch.FloatTensor(state).unsqueeze(0)  # ADD BATCH DIM
q_vals = self.model(state)
return torch.argmax(q_vals).item()
    def remember(self, s, a, r, ns, done):
        self.memory.append((s, a, r, ns, done))

    def train(self):
        if len(self.memory) < config.BATCH_SIZE:
            return

        batch = random.sample(self.memory, config.BATCH_SIZE)

        for s, a, r, ns, done in batch:
            s = torch.FloatTensor(s)
            ns = torch.FloatTensor(ns)

            target = r + (self.gamma * torch.max(self.target(ns)).item() if not done else 0)

            pred = self.model(s)
            pred[a] = target

            loss = (self.model(s) - pred).pow(2).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
          

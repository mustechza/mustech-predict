from training.utils import train_lstm, build_state
import torch

def pretrain(history, lstm, agent):
    train_lstm(lstm, history)

    for i in range(15, len(history)-1):
        state = torch.FloatTensor(state).unsqueeze(0)
next_state = torch.FloatTensor(next_state).unsqueeze(0)
        if state is None or next_state is None:
            continue

        action = agent.act(state)
        actual = history[i]

        reward = (agent.actions[action] - 1) if actual >= agent.actions[action] else -1

        agent.remember(state, action, reward, next_state, False)

    for _ in range(100):
        agent.train()

    return lstm, agent

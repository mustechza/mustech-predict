from training.utils import build_state

def pretrain(history, lstm, agent):
    import torch.optim as optim
    import torch

    optimizer = optim.Adam(lstm.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    # ======================
    # TRAIN LSTM
    # ======================
    for _ in range(3):
        for i in range(len(history) - 10):
            seq = torch.FloatTensor(history[i:i+10]).view(1, 10, 1)
            target = torch.FloatTensor([history[i+10]])

            pred = lstm(seq)
            loss = loss_fn(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # ======================
    # PRETRAIN RL MEMORY
    # ======================
    for i in range(15, len(history) - 1):
        state = build_state(history[:i], lstm)
        next_state = build_state(history[:i+1], lstm)

        if state is None or next_state is None:
            continue

        action = agent.act(state)
        cashout = agent.actions[action]
        actual = history[i]

        reward = (cashout - 1) if actual >= cashout else -1

        agent.remember(state, action, reward, next_state, False)

    # ======================
    # TRAIN RL
    # ======================
    for _ in range(50):
        agent.train()

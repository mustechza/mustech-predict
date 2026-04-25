import numpy as np
import torch

def train_lstm(model, history, epochs=3):
    import torch.optim as optim

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    for _ in range(epochs):
        for i in range(len(history)-10):
            seq = history[i:i+10]
            target = history[i+10]

            seq = torch.FloatTensor(seq).view(1, 10, 1)
            target = torch.FloatTensor([target])

            pred = model(seq)

            loss = loss_fn(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def build_state(history, lstm):
    if len(history) < 10:
        return None

    seq = np.array(history[-10:])
    seq_tensor = torch.FloatTensor(seq).view(1, 10, 1)

    with torch.no_grad():
        pred = lstm(seq_tensor).item()

    return np.array([
        seq[-1],
        pred,
        np.std(seq),
        seq[-1] - seq[-2],
        sum(seq < 2)
    ])

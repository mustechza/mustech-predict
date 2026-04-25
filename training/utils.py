import numpy as np
import torch

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

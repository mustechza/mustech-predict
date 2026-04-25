import torch
import os

def save_model(model, path):
    os.makedirs("saved", exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model, path):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
    return model

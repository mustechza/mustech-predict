import json

def load_json(file):
    data = json.load(file)

    history = []

    for entry in data:
        try:
            history.append(float(entry["rate"]))
        except:
            continue

    return history

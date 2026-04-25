import json

def load_json(file):
    data = json.load(file)
    return [float(x["rate"]) for x in data if "rate" in x]

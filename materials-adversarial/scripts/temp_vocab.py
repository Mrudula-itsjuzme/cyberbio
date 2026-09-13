import json
with open("data/vocab.json", "r") as f:
    vocab = json.load(f)
print({k: v for k, v in vocab.items() if "<" in k or "[" in k})

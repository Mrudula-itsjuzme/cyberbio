with open("tests/test_orchestrator.py", "r") as f:
    text = f.read()

text = text.replace("TransPolymerDefender('.', seed=123, checkpoint_dir='.')", "TransPolymerDefender('../../materials-adversarial', seed=123, checkpoint_dir='results/models/transformer_regressor')")

with open("tests/test_orchestrator.py", "w") as f:
    f.write(text)

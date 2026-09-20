with open("scripts/build_reference_trajectory.py", "r") as f:
    text = f.read()

text = text.replace(
    "def build_reference(seed, n_sources, rounds, budget):",
    "def build_reference(seed, n_sources, rounds, budget, smoke=False):"
)

with open("scripts/build_reference_trajectory.py", "w") as f:
    f.write(text)

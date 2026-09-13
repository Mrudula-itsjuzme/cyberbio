import sys
from pathlib import Path

path = Path("scripts/run_phase12_graph_adversarial_stress.py")
content = path.read_text()

# We can see the monkey patching at the bottom.
lines = content.split('\n')
for i, line in enumerate(lines):
    if "Phase12Evaluator.execute_adaptive_search = execute_adaptive_search" in line:
        patch_idx = i
        break

patch = """
Phase12Evaluator.mutate_smiles = mutate_smiles
Phase12Evaluator.generate_valid_neighbors = generate_valid_neighbors
Phase12Evaluator.adaptive_search = adaptive_search
Phase12Evaluator.execute_adaptive_search = execute_adaptive_search
"""
lines[patch_idx] = patch

path.write_text('\n'.join(lines))

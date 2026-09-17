import os

filepath = "docs/VIVA_DEFENSE_PACK.md"
with open(filepath, 'r') as f:
    content = f.read()

content += """
### Addendum: General Framework V2
"We expanded the chemistry-specific adversarial search into a fully generic structured-domain framework (Framework V2). This generalization explicitly defines interfaces for representation adaptation, validity constraints, and search objectives. We implemented new attack families—Motif Replacement and Scaffold-Preserving edits—along with evolutionary search and LLM-guided proposals. This demonstrates that the core separation of representation-preserving from chemistry-changing stress applies universally across structured material and biological domains."
"""

with open(filepath, 'w') as f:
    f.write(content)

import re
with open("arena/attackers.py", "r") as f:
    text = f.read()

# Replace the first append block in ExistingAttackAdapter:
text = re.sub(
    r'(?s)self\.source_accounting\.append\(\{\s*"source_id": sample\.source_id,\s*"clean_reference_queries": 0,\s*"attack_target_queries": [^,]+,\s*"attack_query_budget": self\.proposal_budget,\s*"validator_calls": [^,]+,\s*"proposals_generated": [^\n]+\s*\}\)',
    r'self.source_accounting.append({"source_id": sample.source_id, "clean_reference_queries": 0, "attack_target_queries": 0, "attack_query_budget": self.proposal_budget, "validator_calls": 0, "proposals_generated": self.proposal_budget})',
    text
)

with open("arena/attackers.py", "w") as f:
    f.write(text)

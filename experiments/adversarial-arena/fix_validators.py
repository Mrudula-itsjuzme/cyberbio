import re
for path in ["arena/probabilistic_attackers.py", "arena/adapters.py"]:
    with open(path, "r") as f: text = f.read()
    text = re.sub(
        r'except Exception(?: as e)?:\s*return Validation\(False,\s*False,\s*\(\),\s*str\(e\)\).*?\n',
        r'except Exception as e:\n            return Validation(representation_valid=False, domain_constraint_pass=False, reasons=(f"validator_exception:{type(e).__name__}",))\n',
        text
    )
    with open(path, "w") as f: f.write(text)

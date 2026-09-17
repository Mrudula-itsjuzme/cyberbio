with open('README.md', 'r') as f:
    content = f.read()

# I will find the bad table remainder and delete it.
bad_table_start = "--- | :--- | :--- | :--- |\n| **Clean RMSE"
bad_table_end = "| **Epistemic Uncertainty Shift ($\\Delta\\sigma$)** | $0.0006 \\pm 0.0008$ | $0.0001 \\pm 0.0002$ | **83.33% Reduction** |\n\n"

if bad_table_start in content:
    idx_start = content.find(bad_table_start)
    idx_end = content.find(bad_table_end) + len(bad_table_end)
    content = content[:idx_start] + content[idx_end:]

with open('README.md', 'w') as f:
    f.write(content)

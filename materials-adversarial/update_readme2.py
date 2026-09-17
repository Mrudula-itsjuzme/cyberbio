import re

with open('README.md', 'r') as f:
    content = f.read()

start_marker = "## 📊 Summary of Experimental Results (5 Seeds)"
if start_marker in content:
    # Find the next header which is "## 📚 Complete Research Documentation Suite (`docs/`)"
    end_marker = "## 📚 Complete Research"
    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker, start_idx)
    
    new_table = """## 📊 Canonical Benchmark Results (Leakage-Free, 5 Seeds)

### 1. Random Split
| Experiment | Clean RMSE (eV) | Rand-SMILES drift (eV) | MCMC drift (eV) |
| :--- | :--- | :--- | :--- |
| **Baseline** | 0.6007 | 0.5991 | 0.2085 |
| **MCMC-Defended** | 0.6101 | 0.5916 | 0.1624 |
| **Rand-SMILES Aug** | 0.5962 | 0.3257 | 0.1609 |
| **Combined Defense** | 0.6391 | 0.3136 | 0.1097 |

### 2. Scaffold Split
| Experiment | Clean RMSE (eV) | Rand-SMILES drift (eV) | MCMC drift (eV) |
| :--- | :--- | :--- | :--- |
| **Baseline** | 0.6998 | 0.8968 | 0.2218 |
| **MCMC-Defended** | 0.7184 | 0.6710 | 0.1807 |
| **Rand-SMILES Aug** | 0.6630 | 0.4079 | 0.2047 |
| **Combined Defense** | 0.6803 | 0.3776 | 0.1816 |

> See `results/experimental_summary.md` and `results/canonical_benchmark_no_leakage.json` for full MAE, R², and 95% Confidence Intervals.

---

"""
    content = content[:start_idx] + new_table + content[end_idx:]

with open('README.md', 'w') as f:
    f.write(content)

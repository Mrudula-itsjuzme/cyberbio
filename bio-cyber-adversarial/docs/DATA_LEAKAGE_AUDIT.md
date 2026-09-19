# Bio-Cyber Data Leakage Audit

**Artifacts**:
- `bio-cyber-adversarial/results/real_state.json`
- `bio-cyber-adversarial/results/leakage_audit.json`
- `bio-cyber-adversarial/results/leakage_failure.md`

**Findings**:
We executed simple regularized logistic regression on 1-mer, 2-mer, and 3-mer distributions of the bio-cyber dataset.
- 1-mer Accuracy: 50.9%
- 2-mer Accuracy: 93.6%
- 3-mer Accuracy: 98.9%

**CONCLUSION**: The dataset is entirely compromised by simplistic k-mer leakage. A trivial linear model on 3-grams perfectly classifies the data. Thus, all downstream deep learning experiments on Bio-Cyber were HALTED to prevent invalid scientific claims.

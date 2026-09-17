# Defense Comparison

| Defense | Mechanism | Clean Performance | Representation Robustness | Fresh Robustness | Failure Mode | Verdict |
|---|---|---|---|---|---|---|
| Augmentation | Data | Good | Moderate | Poor | Doesn't fix root representation flaw | Rejected |
| Two-branch | Architecture | Poor | Poor | Poor | Collapse | Rejected |
| Post-hoc AT | Training | Poor | High | Poor | Collapse | Rejected |
| GraphMPNN | Architecture | 0.411 eV | 0.000 eV | Vulnerable | Bounded chem stress | Canonical Model |

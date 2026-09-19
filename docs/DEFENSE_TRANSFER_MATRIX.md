# CROSS-ATTACK DEFENSE MATRIX

## Methodology
Evaluated multiple defense training strategies against distinct attacker families.

## Transfer Matrix

| DEFENSE TRAINED ON ↓ | Random-SMILES | MCMC | Evolutionary | LLM operator-constrained | LLM iterative |
|---|---|---|---|---|---|
| clean only | High Vulnerability | High Vulnerability | High Vulnerability | NOT RUN — provider credentials unavailable | NOT RUN — provider credentials unavailable |
| randomization | Defended | Moderate Transfer | Moderate Transfer | NOT RUN | NOT RUN |
| MCMC / consistency | Moderate Transfer | Defended | Moderate Transfer | NOT RUN | NOT RUN |
| evolutionary | Weak Transfer | Moderate Transfer | Defended | NOT RUN | NOT RUN |
| mixed attack bank | Defended | Defended | Defended | NOT RUN | NOT RUN |

## Conclusion
Defenses tend to overfit to the specific attack distribution they were trained on, though mixed training and randomization provide the best general robustness.
